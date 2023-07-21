
import torch
import sys
import os
import soundfile as sf
import scipy.signal as signal
from scipy import io
import pandas as pd
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from extract_feature.WavLM.WavLM import WavLM, WavLMConfig

def read_audio(path, sample_rate=16000):
    wav, sr = sf.read(path)
     
    if sr != sample_rate:
        num = int((wav.shape[0]) / sr * sample_rate)
        wav = signal.resample(wav, num)
        print(f'Resample {sr} to {sample_rate}')
     
    if wav.ndim == 2:
        wav = wav.mean(-1)
    assert wav.ndim == 1, wav.ndim

    if wav.shape[0] > sample_rate * 20:
        print(f'Crop raw wav from {wav.shape[0]} to {sample_rate * 20}')
        wav = wav[:sample_rate * 20]

    return wav

def extract_wavlm(model, wavfile, savefile, layer=24):
    '''
    Args:
        layer (int): varies from 1 to 24.
    '''

    if isinstance(savefile, str):
        if os.path.exists(savefile):
            print('File existed:', savefile)
            return
        savefile = [savefile]
        layer = [layer]
    else:
        for file in savefile:
            if os.path.exists(file):
                print('File existed:', file)
                return
    assert len(savefile) == len(layer)

    wav_input_16khz = read_audio(wavfile)
    wav_input_16khz = torch.from_numpy(wav_input_16khz).float().unsqueeze(dim=0).cuda()
    
    ############################################
    # extract the representation of last layer #
    ############################################
    # with torch.no_grad():
    #     rep = model.extract_features(wav_input_16khz)[0]
    
    # rep = rep.squeeze(dim=0).cpu().detach().numpy()   # (t, 768)  / (t, 1024)
    # dict = {'wavlm': rep}
    # io.savemat(savefile, dict)
    # print(savefile, '->', rep.shape)
    
    ############################################
    # extract the representation of each layer #
    ############################################
    with torch.no_grad():
        if model.cfg.normalize:
            wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
        rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]   # layer_results: [(x, z), (x, z), ...] z is attn_weight
        layer_attn = [z for _, z in layer_results]  # z is the average attention weights over heads with shape (B, T, T)

    for save, l in zip(savefile, layer):
        rep_l = layer_reps[l]
        rep_l = rep_l.squeeze(dim=0).cpu().detach().numpy()   # (t, 768)  / (t, 1024)
        dict = {'wavlm': rep_l}
        io.savemat(save, dict)
        print(save, '->', rep_l.shape)

def main(args):
    wavdir = args.wavdir
    savedir = args.savedir
    ckpt = args.wavlm
    layer = args.layer
    csvfile = args.csvfile
    gpu = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    checkpoint = torch.load(ckpt)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval().cuda()
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if csvfile is not None:
        df = pd.read_csv(csvfile)
        file_names = df['name'].tolist()
    else:
        file_names = os.listdir(wavdir)
    
    total = len(file_names)
    for i, name in enumerate(file_names):
        wavfile = os.path.join(wavdir, name+'.wav')
        savefile = os.path.join(savedir, name)

        if os.path.exists(savefile):
            print('Pass', name)
            continue

        print(f'----------- {i+1} / {total} -----------')
        extract_wavlm(model, wavfile, savefile, layer=layer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavdir', type=str, help='wav directory')
    parser.add_argument('--savedir', type=str, help='save directory')
    parser.add_argument('--wavlm', type=str, default=None, help='wavlm model')
    parser.add_argument('--layer', type=int, default=24, help='layer index, varies from 1 to 24')
    parser.add_argument('--csvfile', type=str, default=None, help='csv file with name column')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    args = parser.parse_args()

    main(args)
