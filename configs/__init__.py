from .train_config import _C as train_cfg
from .dataset_config import _C as dataset_cfg
from .model_config import _C as model_cfg

import os
import torch
import argparse
from yacs.config import CfgNode as CN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--model_type", help="modify cfg.train.model.type", default='Vesper', type=str) # required=True
    parser.add_argument("-d", "--dataset_database", help="specify the database used", default='lssed', type=str)
    parser.add_argument("-f", "--dataset_feature", help="specify the feature used", type=str)
    parser.add_argument("-e", "--train_EPOCH", help="total training epoch", type=int)
    parser.add_argument("-b", "--train_batch_size", help="training batch size", type=int)
    parser.add_argument("-l", "--train_lr", help="learning rate", type=float)
    parser.add_argument("-g", "--train_device_id", help="GPU ids", default='0', type=str)
    parser.add_argument("-s", "--train_seed", help="random seed", default=123, type=int)
    parser.add_argument("-S", "--train_save_best", help="save model with the best performance", action='store_true')
    parser.add_argument("-p", "--train_patience", help="the patience used in the early stopping", default=15, type=int)
    parser.add_argument("-r", "--model_output_rep", help="weighted sum or last layer", type=str)
    parser.add_argument("-m", "--mark", help="mark the current run", type=str)
    parser.add_argument("--dataset_num_workers", help="the number of workers", default=12, type=int)
    parser.add_argument("--train_warmup_epoch", help="set the warmup epoch", default=0.05, type=float)
    parser.add_argument("--train_resume", help="resume an experiment", type=str)
    parser.add_argument("--train_load_model", help="load a model", type=str)
    parser.add_argument("--train_device", help="run on cuda or cpu", default='cuda', type=str)
    parser.add_argument("--model_path_to_vesper", help="initialize model with Vesper's checkpoint", type=str)
    parser.add_argument("--model_path_to_wavlm", help="initialize model with pre-trained WavLM", type=str)
    args = parser.parse_args()
    return args

def create_workshop(cfg, local_rank, fold):
    modeltype = cfg.model.type
    database = cfg.dataset.database
    batch = cfg.train.batch_size
    feature = cfg.dataset.feature
    lr = cfg.train.lr
    epoch = cfg.train.EPOCH
    
    world_size = torch.cuda.device_count()
    batch = batch * world_size

    config_name = f'./exp/{modeltype}/{database}_e{epoch}_b{batch}_lr{lr}_{feature}'

    if cfg.mark is not None:
        config_name = config_name + '_mark_{}'.format(cfg.mark)

    cfg.workshop = os.path.join(config_name, f'fold_{fold}')
    cfg.ckpt_save_path = os.path.join(cfg.workshop, 'checkpoint')
    
    if local_rank == 0:
        if os.path.exists(cfg.workshop):
            if cfg.train.resume is None:
                raise ValueError(f'workshop {cfg.workshop} already existed.')
        else:
            os.makedirs(cfg.workshop)
            os.makedirs(cfg.ckpt_save_path)

def get_config(mode=''):
    args = get_args()

    cfg = CN(new_allowed=True)
    cfg.model = CN(new_allowed=True)
    cfg.dataset = CN(new_allowed=True)
    cfg.train = CN(new_allowed=True)
    
    if len(args.model_type.split('-')) > 1:
        args.model_type, version = args.model_type.split('-')[0], args.model_type.split('-')[1]
    if args.model_type == 'WavLM':
        is_wavlm = True
        args.model_type = 'Vesper'
    else:
        is_wavlm = False
    
    cfg.model.update(model_cfg[args.model_type])
    cfg.dataset.update(dataset_cfg[args.dataset_database])
    cfg.train.update(train_cfg[args.model_type+mode])
    
    # Namespace -> Dict
    args = vars(args)
    verbose = []
    for key, value in args.items():
        key_list = key.split('_', maxsplit=1)
        if len(key_list) > 1:
            if value is not None or not hasattr(cfg[key_list[0]], key_list[1]):
                cfg[key_list[0]][key_list[1]] = value
                verbose.append((key, value))
        else:
            if value is not None or not hasattr(cfg, key_list[0]):
                cfg[key_list[0]] = value
                verbose.append((key, value))
    # print('Arguments from command line:', verbose)

    if is_wavlm:
        cfg.model.init_with_wavlm = True
        cfg.model.init_with_ckpt = not cfg.model.init_with_wavlm
        if version == 'Base':
            cfg.model.path_to_wavlm = cfg.model.path_to_wavlm[0]
            cfg.model.encoder_layers = 12
            cfg.model.encoder_embed_dim = 768
            cfg.model.ffn_embed_dim = 3072
            cfg.model.num_heads = 12
            cfg.model.extractor_mode = 'default'
            cfg.model.normalize = False
            cfg.model.normalize_before = False
        elif version == 'Large':
            cfg.model.path_to_wavlm = cfg.model.path_to_wavlm[1]
            cfg.model.encoder_layers = 24
        else:
            raise ValueError(f'Unknown WavLM version: {version}')
    else:
        cfg.model.init_with_wavlm = True if 'pretrain' in mode else False
        cfg.model.init_with_ckpt = not cfg.model.init_with_wavlm
        cfg.model.encoder_layers = eval(version)
    
    cfg.model.num_classes = cfg.dataset.num_classes 
    if cfg.model.type == 'ALLSpeech':
        cfg.dataset.num_queries = cfg.model.num_queries
        cfg.dataset.distractors = cfg.model.distractors
        cfg.dataset.mask_span = cfg.model.mask_span
        cfg.dataset.mask_chunk = cfg.model.mask_chunk

    # modify cfg.train.batch_size in the case of multi-GPUs training
    num_gpus = len(cfg.train.device_id.split(','))
    if num_gpus > 1:
        ddp_batch_size = round(cfg.train.batch_size / num_gpus)
        print(f'Modified batch size: {cfg.train.batch_size} -> {ddp_batch_size}.')
        cfg.train.batch_size = ddp_batch_size
    return cfg

def dict_2_list(dict):
    lst = []
    for key, value in dict.items():
        if value is not None:
            lst.extend([key, value])
    return lst

