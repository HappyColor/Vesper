
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp
import numpy as np
import pandas as pd
import librosa
from scipy import io
from sklearn.model_selection import StratifiedShuffleSplit

def identity(x):
    return x

class DistributedDalaloaderWrapper():
    def __init__(self, dataloader: DataLoader, collate_fn):
        self.dataloader = dataloader
        self.collate_fn = collate_fn
    
    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.sampler.set_epoch(epoch)

def universal_collater(batch):
    all_data = [[] for _ in range(len(batch[0]))]
    for one_batch in batch:
        for i, (data) in enumerate(one_batch):
            all_data[i].append(data)
    return all_data

def universal_dict_collater(batch):
    keys = batch[0].keys()
    all_data = {key: [] for key in keys}
    for one_batch in batch:
        for key in keys:
            all_data[key].append(one_batch[key])
    return all_data

class LssedDataset(Dataset):
    def __init__(self, args, state: str='train'):

        _mapping = None

        self.df = pd.read_csv(args.meta_csv_file)
        self.wavdir = args.wavdir
        self.batch_length = args.batch_length

        self.l_target_dir = args.l_target_dir
        self.h_target_dir = args.h_target_dir
        self.target_length = args.target_length

        if _mapping is not None:
            self.df['faces'] = self.df['faces'].map(_mapping).astype(np.float32)
        
        self.df = self.df.reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform, _ = librosa.load(os.path.join(self.wavdir, self.df.loc[idx, 'name'] + '.wav'), sr=16000)
        emotion = self.df.loc[idx, 'faces']
        padding_mask = torch.full((1, self.batch_length), fill_value=False, dtype=torch.bool)
        
        length = waveform.shape[-1]
        if length >= self.batch_length:
            waveform = waveform[np.newaxis, :self.batch_length]
        else:
            padding_length = self.batch_length - length
            waveform = np.pad(waveform, ((0, padding_length)), 'constant', constant_values=(0, 0))[np.newaxis, :]
            padding_mask[:, -padding_length:] = True

        waveform = torch.from_numpy(waveform)
        
        l_target = io.loadmat(os.path.join(self.l_target_dir, self.df.loc[idx, 'name']))['wavlm']
        h_target = io.loadmat(os.path.join(self.h_target_dir, self.df.loc[idx, 'name']))['wavlm']
        length = h_target.shape[0]
        if length >= self.target_length:
            l_target = l_target[:self.target_length]
            h_target = h_target[:self.target_length]
        else:
            padding_length = self.target_length - length
            l_target = np.pad(l_target, ((0, padding_length), (0, 0)), 'constant', constant_values=(0, 0))
            h_target = np.pad(h_target, ((0, padding_length), (0, 0)), 'constant', constant_values=(0, 0))
        l_target = torch.from_numpy(l_target)
        h_target = torch.from_numpy(h_target)

        sample = {
            'waveform': waveform,
            'padding_mask': padding_mask,
            'emotion': emotion,
            'l_target': l_target,
            'h_target': h_target
        }

        return sample

class DownstreamDataset(Dataset):
    def __init__(self, df, wavdir, batch_length, col_sample='name', col_label='label'):
        self.df = df
        self.wavdir = wavdir
        self.batch_length = batch_length
        self.col_sample = col_sample
        self.col_label = col_label
       
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform, _ = librosa.load(os.path.join(self.wavdir, self.df.loc[idx, self.col_sample] + '.wav'), sr=16000)
        emotion = torch.tensor([self.df.loc[idx, self.col_label]], dtype=torch.long)
        padding_mask = torch.full((1, self.batch_length), fill_value=False, dtype=torch.bool)
        
        length = waveform.shape[-1]
        if length >= self.batch_length:
            waveform = waveform[np.newaxis, :self.batch_length]
        else:
            padding_length = self.batch_length - length
            waveform = np.pad(waveform, ((0, padding_length)), 'constant', constant_values=(0, 0))[np.newaxis, :]
            padding_mask[:, -padding_length:] = True

        waveform = torch.from_numpy(waveform)

        sample = {
            'waveform': waveform,
            'padding_mask': padding_mask,
            'emotion': emotion
        }

        return sample

class IemocapDataset(DownstreamDataset):
    def __init__(self, args, state: str='train', fold: int=None):

        _mapping = {'ang': 0, 'neu': 1, 'hap': 2, 'exc': 2, 'sad': 3}

        df = pd.read_csv(args.meta_csv_file)
        wavdir = args.wavdir
        batch_length = args.batch_length

        if _mapping is not None:
            df['label'] = df['label'].map(_mapping).astype(np.float32)
        
        df = df[df['label'].notnull()]

        if fold is not None:
            test_session = f'Ses0{fold}'
            samples = df['name'].str.startswith(test_session)
            if state == 'train':
                samples = ~samples
            df = df[samples]
        
        df = df.reset_index()
       
        super().__init__(df, wavdir, batch_length)

class MeldDataset(DownstreamDataset):
    def __init__(self, args, state: str='train'):

        _mapping = {'neutral': 0, 'anger': 1, 'joy': 2, 'sadness': 3, 'surprise': 4, 'disgust': 5, 'fear': 6}
        state_csv_file = {'train': 'train_sent_emo.csv', 'dev': 'dev_sent_emo.csv', 'test': 'test_sent_emo.csv'}
        state_wav_dir = {'train': 'train', 'dev': 'dev', 'test': 'test'}

        df = pd.read_csv(os.path.join(args.meta_csv_file, state_csv_file[state]))
        wavdir = os.path.join(args.wavdir, state_wav_dir[state])
        batch_length = args.batch_length

        if _mapping is not None:
            df['Emotion'] = df['Emotion'].map(_mapping).astype(np.float32)
        
        audio_name = []
        for dia, utt in zip(df['Dialogue_ID'], df['Utterance_ID']):
            audio_name.append(f'dia{dia}_utt{utt}')

        df['name'] = audio_name
        df = df[['name', 'Emotion']]

        delete_row = []
        for row_index, row in df.iterrows():
            if not os.path.exists(os.path.join(wavdir, row['name'] + '.wav')):  
                delete_row.append(row_index)
        df = df.drop(delete_row, axis=0)     

        df = df[df['Emotion'].notnull()]

        df = df.reset_index()

        super().__init__(df, wavdir, batch_length, col_label='Emotion')

class CremaDataset(DownstreamDataset):
    def __init__(self, args, state: str='train'):

        _mapping = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}

        df = pd.read_csv(args.meta_csv_file)
        wavdir = args.wavdir
        batch_length = args.batch_length

        if _mapping is not None:
            df['label'] = df['label'].map(_mapping).astype(np.float32)
        
        df = df[df['label'].notnull()]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2013)
        for train_index, test_index in sss.split(df['name'], df['label']):
            if state == 'train':
                df = df.iloc[train_index]
            else:
                df = df.iloc[test_index]
        
        df = df.reset_index()

        super().__init__(df, wavdir, batch_length)

class DataloaderFactory():
    def __init__(self, args):
        self.args = args

    def build(self, state: str='train', bs: int=1, fold: int=1):
        if self.args.database == 'lssed':
            dataset = LssedDataset(self.args, state)
        elif self.args.database == 'iemocap':
            dataset = IemocapDataset(self.args, state, fold)
        elif self.args.database == 'meld':
            dataset = MeldDataset(self.args, state)
        elif self.args.database == 'crema':
            dataset = CremaDataset(self.args, state)
        else:
            raise NotImplementedError
        
        collate_fn = universal_dict_collater
        sampler = DistributedSampler(dataset, shuffle=state == 'train')
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=bs, 
            drop_last=False, 
            num_workers=self.args.num_workers, 
            collate_fn=identity,
            sampler=sampler, 
            pin_memory=True,
            multiprocessing_context=mp.get_context('fork'), # fork/spawn # quicker! Used with multi-process loading (num_workers > 0)
        )

        return DistributedDalaloaderWrapper(dataloader, collate_fn)

