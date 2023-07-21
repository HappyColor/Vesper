#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed, Mar 8, 2023
@author: lab-chen.weidong
"""

import os
from tqdm import tqdm
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import math
import json
import shutil

import utils
import models
from configs import create_workshop, get_config, dict_2_list

class Engine():
    def __init__(self, cfg, local_rank: int, world_size: int):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = self.cfg.train.device
        self.EPOCH = self.cfg.train.EPOCH
        self.current_epoch = 0
        self.iteration = 0
        self.lowest_loss = 1e4
        self.loss_weight_l = self.cfg.train.loss_weight_l
        self.loss_weight_h = self.cfg.train.loss_weight_h
        self.loss_weight_x = self.cfg.train.loss_weight_x
        
        self.early_stopping = utils.earlystopping.EarlyStopping(patience=self.cfg.train.patience, verbose=self.local_rank == 0)

        ### prepare meters
        self.loss_meter = utils.avgmeter.AverageMeter(device='cuda')
        self.l_loss_meter = utils.avgmeter.AverageMeter(device='cuda')
        self.h_loss_meter = utils.avgmeter.AverageMeter(device='cuda')
        self.x_loss_meter = utils.avgmeter.AverageMeter(device='cuda')
    
    def prepare_staff(self, fold=1):
        ''' We move this part out of the __init__ function to avoid the weird error:
            DataLoader worker (pid xxx) is killed by signal: Aborted
            This error is probably caused by a conflict between lmdb and ddp.
        '''
        ### prepare dataloader
        dataloader_feactory = utils.dataset.DataloaderFactory(self.cfg.dataset)
        self.dataloader_train = dataloader_feactory.build(
            state='train', 
            bs=self.cfg.train.batch_size, 
            fold=fold
        )
        self.cfg.model.freeze_cnn = self.cfg.train.freeze_cnn
        self.cfg.model.device = self.device
        
        ### prepare model, optimizer and scheduler
        model = models.vesper.Vesper_PretrainWrapper(self.cfg.model).to(self.device)
        
        if self.cfg.train.freeze_cnn:
            for param in model.vesper.feature_extractor.parameters():
                param.requires_grad = False
        
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])

        if self.cfg.train.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                params=filter(lambda x: x.requires_grad, self.model.parameters()), 
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay
            )
        elif self.cfg.train.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                params=filter(lambda x: x.requires_grad, self.model.parameters()), 
                lr=self.cfg.train.lr,
                momentum=0.9,
                weight_decay=self.cfg.train.weight_decay
            )
        else:
            raise ValueError(f'Unknown optimizer: {self.cfg.train.optimizer}')
        
        if self.local_rank == 0:
            print(f'Optimizer: {self.cfg.train.optimizer}')
            
        # CosineAnnealingLR with Warm-up
        warmup_epoch = int(self.cfg.train.warmup_epoch * self.EPOCH)
        lr_max = self.cfg.train.lr
        lr_min = self.cfg.train.lr * 0.01
        T_max = self.EPOCH
        lr_lambda = lambda epoch: (epoch + 1) / warmup_epoch if epoch < warmup_epoch else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((epoch-warmup_epoch)/(T_max-warmup_epoch)*math.pi))) / self.cfg.train.lr
        self.scheduler = lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lr_lambda)

        if self.cfg.train.load_model is not None:
            ckpt = torch.load(self.cfg.train.load_model, map_location=self.device)
            self.model.module.load_state_dict(ckpt['model'])
            if self.local_rank == 0:
                print(f'Loading model from {self.cfg.train.load_model}')
            del ckpt
        
        if self.cfg.train.resume is not None:
            ckpt = torch.load(self.cfg.train.resume, map_location=self.device)
            self.model.module.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.scheduler.step()
            self.current_epoch = ckpt['epoch'] + 1
            self.iteration = ckpt['iteration']
            self.lowest_loss = ckpt['lowest_loss']
            if self.local_rank == 0:
                print(f'Resuming from {self.cfg.train.resume}')
            del ckpt
        
        ### prepare writer and logger
        if self.local_rank == 0:
            self.writer = SummaryWriter(self.cfg.workshop)
            self.logger_train = utils.logger.create_logger(self.cfg.workshop, name='train')
            self.logger_train.info(f'workshop: {self.cfg.workshop}')
            self.logger_train.info(f'seed: {self.cfg.train.seed}')
            self.logger_train.info(f'pid: {os.getpid()}')
            print('Main pid:', os.getpid())
        else:
            self.writer = None
            self.logger_train = None
        
        self.config_2_json()

    def config_2_json(self, jsonfile=None):
        self.jsonfile = os.path.join(self.cfg.workshop, 'config.json') if jsonfile is None else jsonfile
        with open(self.jsonfile, 'w') as f:
            json.dump(dict(self.cfg), f, indent=2)

    def json_2_config(self, jsonfile=None):
        if jsonfile is not None:
            self.jsonfile = jsonfile
        assert hasattr(self, 'jsonfile'), 'Please provide the .json file first.'
        with open(self.jsonfile, 'r') as f:
            data = json.load(f)
            self.cfg.merge_from_list(dict_2_list(data))

    def reset_meters(self):
        self.loss_meter.reset()
        self.l_loss_meter.reset()
        self.h_loss_meter.reset()
        self.x_loss_meter.reset()

    def gather_distributed_data(self, gather_data):
        if isinstance(gather_data, torch.Tensor):
            _output = [torch.zeros_like(gather_data) for _ in range(self.world_size)]
            dist.all_gather(_output, gather_data, async_op=False)
            output = torch.cat(_output)
        else:
            if gather_data[0] is not None:
                _output = [None for _ in range(self.world_size)]
                if hasattr(dist, 'all_gather_object'):
                    dist.all_gather_object(_output, gather_data)
                else:
                    utils.distributed.all_gather_object(_output, gather_data, self.world_size)
                output = []
                for lst in _output:
                    output.extend(lst)
            else:
                output = None
        return output

    def train_epoch(self):
        self.dataloader_train.set_epoch(self.current_epoch)
        if self.local_rank == 0:
            print(f'-------- {self.cfg.workshop} --------')
        discrip_str = f'Epoch-{self.current_epoch}/{self.EPOCH}'
        pbar_train = tqdm(self.dataloader_train, disable=self.local_rank != 0, dynamic_ncols=True)
        pbar_train.set_description('Train' + discrip_str)

        self.reset_meters()

        self.model.train()
        for data in pbar_train:
            self.iteration += 1
            
            waveform = torch.cat(data['waveform'], dim=0).to(self.device)
            padding_mask = torch.cat(data['padding_mask'], dim=0).to(self.device)

            self.optimizer.zero_grad()
            
            l_target = torch.stack(data['l_target'], dim=0).to(self.device) if data['l_target'][0] is not None else None
            h_target = torch.stack(data['h_target'], dim=0).to(self.device) if data['h_target'][0] is not None else None
            l_loss, h_loss, x_loss = self.model(waveform, padding_mask, l_target, h_target)
            loss = self.loss_weight_l * l_loss + self.loss_weight_h * h_loss + self.loss_weight_x * x_loss
            loss.backward()

            self.optimizer.step()

            self.loss_meter.update(loss.item())
            self.l_loss_meter.update(l_loss.item())
            self.h_loss_meter.update(h_loss.item())
            self.x_loss_meter.update(x_loss.item())

            pbar_train_dic = OrderedDict()
            pbar_train_dic['iter'] = self.iteration
            pbar_train_dic['lr'] = self.optimizer.param_groups[0]['lr']
            pbar_train_dic['l_loss'] = f'{self.l_loss_meter.avg:.5f}'
            pbar_train_dic['h_loss'] = f'{self.h_loss_meter.avg:.5f}'
            pbar_train_dic['x_loss'] = f'{self.x_loss_meter.avg:.5f}'
            pbar_train_dic['loss'] = f'{self.loss_meter.avg:.5f}'
            pbar_train.set_postfix(pbar_train_dic)
            
            if self.iteration % (len(self.dataloader_train) // 20) == 0:
                if self.local_rank == 0:
                    self.writer.add_scalar('Step/l_loss', l_loss.item(), self.iteration)
                    self.writer.add_scalar('Step/h_loss', h_loss.item(), self.iteration)
                    self.writer.add_scalar('Step/x_loss', x_loss.item(), self.iteration)
                    self.writer.add_scalar('Step/loss', loss.item(), self.iteration)

        self.loss_meter.sync_distributed()
        self.l_loss_meter.sync_distributed()
        self.h_loss_meter.sync_distributed()
        self.x_loss_meter.sync_distributed()

        l_loss_epoch = self.l_loss_meter.avg
        h_loss_epoch = self.h_loss_meter.avg
        x_loss_epoch = self.x_loss_meter.avg
        loss_epoch = self.loss_meter.avg

        if self.local_rank == 0:
            self.writer.add_scalar('Epoch/l_loss', l_loss_epoch, self.current_epoch)
            self.writer.add_scalar('Epoch/h_loss', h_loss_epoch, self.current_epoch)
            self.writer.add_scalar('Epoch/x_loss', x_loss_epoch, self.current_epoch)
            self.writer.add_scalar('Epoch/loss', loss_epoch, self.current_epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)

            self.logger_train.info(
                f'Training epoch: {self.current_epoch}, l_loss: {l_loss_epoch:.5f}, h_loss: {h_loss_epoch:.5f}, x_loss: {x_loss_epoch:.5f}, loss: {loss_epoch:.5f}'
            )

            is_best = loss_epoch < self.lowest_loss
            self.lowest_loss = min(loss_epoch, self.lowest_loss)
            self.model_save(is_best)

        self.early_stopping(loss_epoch, self.model)

    def model_save(self, is_best=False, filename='checkpoint.pt'): 
        ckpt_save_file = os.path.join(self.cfg.ckpt_save_path, filename)
        save_dict = {
            'cfg': self.cfg,
            'epoch': self.current_epoch,
            'iteration': self.iteration,
            'lowest_loss': self.lowest_loss,
            'model': self.model.module.state_dict(),   # save DDP model
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }
        torch.save(save_dict, ckpt_save_file)
        if is_best:
            shutil.copyfile(ckpt_save_file, os.path.join(self.cfg.ckpt_save_path, 'model_best.pt'))
    
    def run(self, fold=1):
        self.prepare_staff(fold=fold)
        
        while self.current_epoch < self.EPOCH:
            self.train_epoch()
            self.scheduler.step()

            self.current_epoch += 1
            
            if self.early_stopping.early_stop:
                print(f"Early stopping (patience: {self.early_stopping.patience})")
                break
        
        self.cleanup()

    def cleanup(self):
        if self.logger_train is not None:
            utils.logger.close_logger(self.logger_train)
        if self.writer is not None:
            self.writer.close()
        # torch.cuda.empty_cache()
        self.early_stopping.clean()
        self.current_epoch = 0
        self.iteration = 0
        self.lowest_loss = 1e4

def main_worker(local_rank, cfg, world_size, dist_url):
    mp.set_sharing_strategy('file_system')
    utils.environment.set_seed(cfg.train.seed + local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=world_size,
        rank=local_rank,
    )
    # torch.autograd.set_detect_anomaly(True)
    engine = Engine(cfg, local_rank, world_size)
    for fold in cfg.dataset.folds:
        create_workshop(cfg, local_rank, fold)
        engine.run(fold)

    if local_rank == 0:
        criterion = ['l_loss', 'h_loss', 'x_loss', 'loss']
        evaluate = ['loss']
        outfile = f'result/result_{cfg.model.type}.csv'
        wantlow = True
        utils.collect_result.path_to_csv(os.path.dirname(cfg.workshop), criterion, evaluate, csvfile=outfile, logname='train.log', wantlow=wantlow)

def main(cfg):
    utils.environment.visible_gpus(cfg.train.device_id)

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'   
    world_size = torch.cuda.device_count()    # num_gpus
    print(f'world_size={world_size} Using dist_url={dist_url}')

    mp.spawn(fn=main_worker, args=(cfg, world_size, dist_url), nprocs=world_size)

if __name__=='__main__':
    cfg = get_config(mode='_pretrain')
    main(cfg)
