
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Using Mish as activation function instead of ReLu

[1] Diganta Misra
    Mish: A Self Regularized Non-Monotonic Activation Function (2019)
    https://arxiv.org/abs/1908.08681

To use Mish as activation function, 
E.g. In init: 
            self.act1 = Mish()
            
"""

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x