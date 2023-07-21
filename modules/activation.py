import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_activation_fn(activation: str='relu', module: bool=False):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return nn.ReLU() if module else F.relu
    elif activation == "gelu":
        return nn.GELU() if module else F.gelu
    elif activation == "tanh":
        return nn.Tanh() if module else torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

