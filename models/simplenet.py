import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    eps = 1e-5
    momentum = 0.05
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=1)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.05, affine=True)
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out

class SimpleNet(nn.Module):
    cfg1 = [64, 128, 128, 128, (128, 2), 128, 128, 128, (128, 2), 128, (128, 2), 128, 128]
    cfg2 = [128, (128,2), 128, 128]
    eps = 1e-5
    momentum = 0.05
    def __init__(self, num_classes = 100):
        super(SimpleNet, self).__init__()
        
        #Block 1 to 10
        self.layers1 = self._make_layers1(in_planes=3)
        
        #Block 11 to 13
        self.layers2 = self._make_layers2(in_planes=128)
           
        self.linear = nn.Linear(128, num_classes)
        self.drp = nn.Dropout(0.1)
        
    def _make_layers1(self, in_planes):
        layers = []
        for x in self.cfg1:
            if isinstance(x, int):
                out_planes = x
                stride = 1
                layers.append(Block(in_planes, out_planes, kernel_size = 3, stride=1))
            else:

                layers.append(nn.MaxPool2d(kernel_size=2, stride =2 , dilation=1, ceil_mode=False))
                layers.append(nn.Dropout2d(p=0.1))
            
            in_planes = out_planes
        return nn.Sequential(*layers)
    
    def _make_layers2(self, in_planes):
        layers = []
        for x in self.cfg2:
            if isinstance(x, int):
                out_planes = x
                stride = 1
                layers.append(Block(in_planes, out_planes, kernel_size = 3, stride=1))
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride = 2 , dilation=1, ceil_mode=False))
                layers.append(nn.Dropout2d(p=0.1))
            in_planes = out_planes
        return nn.Sequential(*layers)    
    
    def forward(self, x):
        out = self.layers1(x)
        out = self.layers2(out)
        out = F.max_pool2d(out, kernel_size=out.size()[2:]) 
        out = self.drp(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
