import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Block(nn.Module)
    eps = 1e-5
    momentum = 0.05
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=1)
        self.bn = nn.BatchNorm2d(out_planes, eps, momentum, affine=True),
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out

class SimpleNet(nn.Module):
    cfg1 = [64, 128, 128, 128, (128, 2), 128, 128, 128, (128, 2), 128, (128, 2), 128, 128]
    cfg2 = [128, (128,2), 128, (128,2)]
    eps = 1e-5
    momentum = 0.05
    def __init__(self, num_classes = 100):
        super(SimpleNet, self).__init__()
        
        #Block 1 to 10
        self.layers1 = self._make_layers1(in_planes=3)
        
        #Block 11 to 13
        self.layers2 = self._make_layers2(in_planes=128)
           
        self.linear = nn.Linear(128, num_classes)
        
    def _make_layers1(self, in_planes):
        layers = []
        for x in self.cfg:
            if isinstance(x, int):
                out_planes = x
                stride = 1
                layers.append(Block(in_planes, out_planes, kernel_size = 3, stride))
            else:
                stride = x[1]
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                nn.Dropout2d(p=0.1))
            
            in_planes = out_planes
        return nn.Sequential(*layers)
    
    def _make_layers2(self, in_planes):
        layers = []
        for x in self.cfg2:
            if isinstance(x, int):
                out_planes = x
                stride = 1
                layers.append(Block(in_planes, out_planes, kernel_size = 3, stride))
            else:
                stride = x[1]
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                nn.Dropout2d(p=0.1))
            
            in_planes = out_planes
        return nn.Sequential(*layers)    
    
    def forward(self, x):
        out = self.layers1(x)
        out = self.layers2(out)
        out = self.linear(out)
        return out
        
        
          
        
