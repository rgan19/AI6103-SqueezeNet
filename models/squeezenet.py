import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Mish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.tanh(F.softplus(x)) if self.inplace else x.clone() * torch.tanh(F.softplus(x))

class Fire(nn.Module):
    def __init__(self, in_planes, squeeze1x1, expand1x1, expand3x3):
        super().__init__()
        self.squeeze = nn.Conv2d(in_planes, squeeze1x1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(squeeze1x1)
        self.act1 = Mish(inplace=True)
        self.expand1 = nn.Conv2d(squeeze1x1, expand1x1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(expand1x1)
        self.expand3 = nn.Conv2d(squeeze1x1, expand3x3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(expand3x3)
        self.act2 = Mish(inplace=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                m.weight.data = torch.sign(m.weight.data)

    def forward(self, x):
        out = self.act1(self.bn1(self.squeeze(x)))
        out1 = self.bn2(self.expand1(out))
        out2 = self.bn3(self.expand3(out))
        out = self.act2(torch.cat([out1, out2], 1))
        return out

class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire5 = Fire(256, 32, 128, 128)
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire9 = Fire(512, 64, 256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Sequential(nn.Conv2d(512, 100, kernel_size=1, stride=1), nn.AvgPool2d(kernel_size=4, stride=4))
        
   
    def match_channels(self, x, target_channels):
        if x.size(1) == target_channels:
            return x
        else:
            return nn.Conv2d(x.size(1), target_channels, kernel_size=1)(x)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        out_fire2 = self.fire2(out)
        out_fire3 = self.fire3(out_fire2)
        out_fire4 = self.fire4(out_fire2 + self.match_channels(out_fire3, out_fire2.size(1)))

        out = self.maxpool2(out_fire4)
        out_sum = out_fire2 + self.match_channels(out_fire3, out_fire2.size(1)) + self.match_channels(out_fire4, out_fire2.size(1))
        out_fire5 = self.fire5(out_sum)
        out_fire6 = self.fire6(out_sum + self.match_channels(out_fire5, out_fire2.size(1)))

        out_sum = out_sum + self.match_channels(out_fire5, out_fire2.size(1)) + self.match_channels(out_fire6, out_fire2.size(1))
        out_fire7 = self.fire7(out_sum)
        out_fire8 = self.fire8(out_sum + self.match_channels(out_fire7, out_fire2.size(1)))

        out = self.maxpool3(out_fire8)
        out_sum = out_sum + self.match_channels(out_fire7, out_fire2.size(1)) + self.match_channels(out_fire8, out_fire2.size(1))
        out_fire9 = self.fire9(out_sum)
        out_fire9 = self.dropout(out_fire9)

        out = self.classifier(out_sum + self.match_channels(out_fire9, out_fire2.size(1)))

        out = torch.flatten(out, 1)

        return out
