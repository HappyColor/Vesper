
import torch.nn as nn
from modules.activation import _get_activation_fn

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout, activation):
        super().__init__()
    
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            _get_activation_fn(activation, module=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim//2, input_dim//4),
            _get_activation_fn(activation, module=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim//4, num_classes),
        )
    
    def forward(self, x):
        pred = self.net(x)
        return pred

