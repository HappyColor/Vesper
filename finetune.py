#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13, 2023
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
from sklearn.metrics import accuracy_score
import json
import shutil
import re

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
        self.best_score = 0

        self.dataloader_feactory = utils.dataset.DataloaderFactory(self.cfg.dataset)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.calculate_score = utils.metric.calculate_score_classification
        self.early_stopping = utils.earlystopping.EarlyStopping(patience=self.cfg.train.patience, verbose=self.local_rank == 0, higher_is_better=True)

        ### prepare meters
        data_type = torch.int64
        self.loss_meter = utils.avgmeter.AverageMeter(device='cuda')
        self.acc_meter = utils.avgmeter.AverageMeter(device='cuda')
        self.predict_recoder = utils.recoder.TensorRecorder(device='cuda', dtype=data_type)
        self.label_recoder = utils.recoder.TensorRecorder(device='cuda', dtype=data_type)
    
    def prepare_staff(self):
        ''' We move this part out of the __init__ function to avoid the weird error:
            DataLoader worker (pid xxx) is killed by signal: Aborted
            This error is probably caused by a conflict between lmdb and ddp.
        '''
        ### prepare dataloader
        self.dataloader_train = self.dataloader_feactory.build(
            state='train', 
            bs=self.cfg.train.batch_size, 
            fold=self.fold
        )
        self.dataloader_test = self.dataloader_feactory.build(
            state='dev', 
            bs=self.cfg.train.batch_size, 
            fold=self.fold
        )

        ### prepare model, optimizer and scheduler
        self.cfg.model.freeze_cnn = self.cfg.train.freeze_cnn
        self.cfg.model.freeze_upstream = self.cfg.train.freeze_upstream
        model = models.vesper.VesperFinetuneWrapper(self.cfg.model).to(self.device)

        if self.cfg.train.freeze_cnn:
            for param in model.vesper.feature_extractor.parameters():
                param.requires_grad = False
        if self.cfg.train.freeze_upstream:
            for param in model.vesper.parameters():
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
        # warmup_epoch = int(self.cfg.train.warmup_epoch * self.EPOCH)
        warmup_epoch = 0
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
            self.best_score = ckpt['best_score']
            if self.local_rank == 0:
                print(f'Resuming from {self.cfg.train.resume}')
            del ckpt
        
        ### prepare writer and logger
        if self.local_rank == 0:
            self.writer = SummaryWriter(self.cfg.workshop)
            self.logger_train = utils.logger.create_logger(self.cfg.workshop, name='train')
            self.logger_test = utils.logger.create_logger(self.cfg.workshop, name='test')
            self.logger_train.info(f'workshop: {self.cfg.workshop}')
            self.logger_train.info(f'seed: {self.cfg.train.seed}')
            self.logger_train.info(f'pid: {os.getpid()}')
            print('Main pid:', os.getpid())
        else:
            self.writer = None
            self.logger_train = None
            self.logger_test = None
        
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
        self.acc_meter.reset()
    
    def reset_recoders(self):
        self.predict_recoder.reset()
        self.label_recoder.reset()

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
        self.reset_recoders()

        self.model.train()
        for data in pbar_train:
            self.iteration += 1
            
            waveform = torch.cat(data['waveform'], dim=0).to(self.device)
            padding_mask = torch.cat(data['padding_mask'], dim=0).to(self.device)
            y = torch.cat(data['emotion'], dim=0).to(self.device)
            batch_size = y.shape[0]

            self.optimizer.zero_grad()
            
            pred = self.model(waveform, padding_mask)
            loss = self.loss_func(pred, y)
            loss.backward()

            self.optimizer.step()

            y_pred = torch.argmax(pred, dim=1)

            self.predict_recoder.record(y_pred)
            self.label_recoder.record(y)

            accuracy = accuracy_score(y.cpu(), y_pred.cpu())
            self.loss_meter.update(loss.item())
            self.acc_meter.update(accuracy, batch_size)

            pbar_train_dic = OrderedDict()
            pbar_train_dic['iter'] = self.iteration
            pbar_train_dic['lr'] = self.optimizer.param_groups[0]['lr']
            pbar_train_dic['acc'] = f'{self.acc_meter.avg:.5f}'
            pbar_train_dic['loss'] = f'{self.loss_meter.avg:.5f}'
            pbar_train.set_postfix(pbar_train_dic)

        epoch_preds = self.gather_distributed_data(self.predict_recoder.data).cpu()
        epoch_labels = self.gather_distributed_data(self.label_recoder.data).cpu()

        self.loss_meter.sync_distributed()
        epoch_loss = self.loss_meter.avg

        if self.local_rank == 0:
            accuracy, recall, f1, precision, Matrix = self.calculate_score(epoch_preds, epoch_labels, self.cfg.dataset.f1)
            self.writer.add_scalar('Train/WA', accuracy, self.current_epoch)
            self.writer.add_scalar('Train/UA', recall, self.current_epoch)
            self.writer.add_scalar('Train/F1', f1, self.current_epoch)
            self.writer.add_scalar('Train/Loss', epoch_loss, self.current_epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)

        if self.logger_train is not None:
            self.logger_train.info(
                f'Training epoch: {self.current_epoch}, accuracy: {accuracy:.5f}, precision: {precision:.5f}, recall: {recall:.5f}, F1: {f1:.5f}, loss: {epoch_loss:.5f}'
            )     

    def test(self):
        discrip_str = f'Epoch-{self.current_epoch}'
        pbar_test = tqdm(self.dataloader_test, disable=self.local_rank != 0, dynamic_ncols=True)
        pbar_test.set_description('Test' + discrip_str)

        self.reset_meters()
        self.reset_recoders()

        self.model.eval()
        with torch.no_grad():
            for data in pbar_test:
                waveform = torch.cat(data['waveform'], dim=0).to(self.device)
                padding_mask = torch.cat(data['padding_mask'], dim=0).to(self.device)
                y = torch.cat(data['emotion'], dim=0).to(self.device)
                batch_size = y.shape[0]

                pred = self.model(waveform, padding_mask)
                loss = self.loss_func(pred, y)

                y_pred = torch.argmax(pred, dim=1)

                self.predict_recoder.record(y_pred)
                self.label_recoder.record(y)

                accuracy = accuracy_score(y.cpu(), y_pred.cpu())
                self.loss_meter.update(loss.item())
                self.acc_meter.update(accuracy, batch_size)

                pbar_test_dic = OrderedDict()
                pbar_test_dic['acc'] = f'{self.acc_meter.avg:.5f}'
                pbar_test_dic['loss'] = f'{self.loss_meter.avg:.5f}'
                pbar_test.set_postfix(pbar_test_dic)

            epoch_preds = self.gather_distributed_data(self.predict_recoder.data).cpu()
            epoch_labels = self.gather_distributed_data(self.label_recoder.data).cpu()

            self.loss_meter.sync_distributed()
            epoch_loss = self.loss_meter.avg

            if self.local_rank == 0:
                # Calculate accuracy, recall, f1, precision, confuse_matrix
                accuracy, recall, f1, precision, Matrix = self.calculate_score(epoch_preds, epoch_labels, self.cfg.dataset.f1)
                self.writer.add_scalar('Test/WA', accuracy, self.current_epoch)
                self.writer.add_scalar('Test/UA', recall, self.current_epoch)
                self.writer.add_scalar('Test/F1', f1, self.current_epoch)
                self.writer.add_scalar('Test/Loss', epoch_loss, self.current_epoch)

                score = 0
                for metric in self.cfg.dataset.evaluate:
                    score += eval(metric)
                
                if self.cfg.train.save_best or self.cfg.dataset.have_test_set:
                    if score > self.best_score:
                        self.best_score = score
                        self.model_save(True)

                self.logger_test.info(
                    f'Testing epoch: {self.current_epoch}, accuracy: {accuracy:.5f}, precision: {precision:.5f}, recall: {recall:.5f}, F1: {f1:.5f}, loss: {epoch_loss:.5f}, confuse_matrix: \n{Matrix}'
                )

                self.early_stopping(score, self.model)

    def evaluate(self):
        self.dataloader_test = self.dataloader_feactory.build(
            state='test', 
            bs=self.cfg.train.batch_size, 
            fold=self.fold
        )

        ckpt = torch.load(self.ckpt_best_file, map_location=self.device)
        self.model.module.load_state_dict(ckpt['model'])
        if self.local_rank == 0:
            print(f'Loading model from {self.ckpt_best_file}')
        del ckpt

        self.current_epoch = -1
        self.test()
        
    def model_save(self, is_best=False, filename='checkpoint.pt'): 
        self.ckpt_save_file = os.path.join(self.cfg.ckpt_save_path, filename)
        save_dict = {
            'cfg': self.cfg,
            'epoch': self.current_epoch,
            'iteration': self.iteration,
            'best_score': self.best_score,
            'model': self.model.module.state_dict(),   # save DDP model
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }
        torch.save(save_dict, self.ckpt_save_file)
        if is_best:
            self.ckpt_best_file = os.path.join(self.cfg.ckpt_save_path, 'model_best.pt')
            shutil.copyfile(self.ckpt_save_file, self.ckpt_best_file)
    
    def run(self, fold=1):
        self.fold = fold
        self.prepare_staff()

        while self.current_epoch < self.EPOCH:
            self.train_epoch()
            self.scheduler.step()
            self.test()

            self.current_epoch += 1
            
            if self.early_stopping.early_stop:
                print(f"Early stopping (patience: {self.early_stopping.patience})")
                break
        
        if self.cfg.dataset.have_test_set:
            self.evaluate()
        
        self.cleanup()

    def cleanup(self):
        if self.logger_train is not None:
            utils.logger.close_logger(self.logger_train)
        if self.logger_test is not None:
            utils.logger.close_logger(self.logger_test)
        if self.writer is not None:
            self.writer.close()
        # torch.cuda.empty_cache()
        self.early_stopping.clean()
        self.current_epoch = 0
        self.iteration = 0
        self.best_score = 0

        if not self.cfg.train.save_best:
            if hasattr(self, 'ckpt_save_file') and os.path.exists(self.ckpt_save_file):
                os.remove(self.ckpt_save_file)
            if hasattr(self, 'ckpt_best_file') and os.path.exists(self.ckpt_best_file):
                os.remove(self.ckpt_best_file)

def main_worker(local_rank, cfg, world_size, dist_url):
    utils.environment.set_seed(cfg.train.seed + local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=world_size,
        rank=local_rank,
    )

    if cfg.model.init_with_ckpt:
        mark = re.search('(?<=_mark_)\w+', cfg.model.path_to_vesper)
        if mark is not None:
            if cfg.mark is None:
                cfg.mark = mark.group()
            else:
                cfg.mark = mark.group() + '_' + cfg.mark

    # torch.autograd.set_detect_anomaly(True)
    engine = Engine(cfg, local_rank, world_size)
    for fold in cfg.dataset.folds:
        create_workshop(cfg, local_rank, fold)
        engine.run(fold)

    if local_rank == 0:
        criterion = ['accuracy', 'precision', 'recall', 'F1']
        evaluate = cfg.dataset.evaluate
        outfile = f'result/result_{cfg.model.type}_Finetune.csv'
        wantlow = False
        return_epoch = -1 if cfg.dataset.have_test_set else None
        utils.collect_result.path_to_csv(os.path.dirname(cfg.workshop), criterion, evaluate, csvfile=outfile, logname='test.log', wantlow=wantlow, epoch=return_epoch)

def main(cfg):
    utils.environment.visible_gpus(cfg.train.device_id)

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'   
    world_size = torch.cuda.device_count()    # num_gpus
    print(f'world_size={world_size} Using dist_url={dist_url}')

    mp.spawn(fn=main_worker, args=(cfg, world_size, dist_url), nprocs=world_size)

if __name__=='__main__':
    cfg = get_config(mode='_finetune')
    main(cfg)
