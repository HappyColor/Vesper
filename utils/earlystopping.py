
import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss (or other metric) doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, ckpt_path=None, higher_is_better=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            ckpt_path (str): Path to save the checkpoint.
                            Default: None
            higher_is_better (bool): If True, the higher scorce denotes the better performance.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.higher_is_better = higher_is_better
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.ckpt_path = ckpt_path
        if self.verbose:
            print(f'Early Stopping: patience {patience}')

    def __call__(self, score, model):

        score = score if self.higher_is_better else -score

        if self.patience <= 0:
            self.early_stop = False
        else:
            if self.best_score is None:
                 self.best_score = score
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                # self.save_checkpoint(score, model)
                self.best_score = score
                self.counter = 0
                self.early_stop = False
    
    def clean(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def save_checkpoint(self, score, model):
        '''
        Saves model when score imporves.
        '''
        if self.verbose:
            print(f'Score imporves ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        if self.ckpt_path is not None:
            torch.save(model.state_dict(), os.path.join(self.ckpt_path, 'checkpoint.pt'))
    
