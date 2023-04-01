import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Fire(nn.Module):
  def __init__(self, in_planes, squeeze1x1, expand1x1, expand3x3):
    super().__init__()
    self.squeeze = nn.Conv2d(in_planes, squeeze1x1, kernel_size = 1)
    self.bn1 = nn.BatchNorm2d(squeeze1x1)
    self.relu1 = nn.ReLU(inplace=True)                
    self.expand1 = nn.Conv2d(squeeze1x1, expand1x1, kernel_size = 1)
    self.bn2 = nn.BatchNorm2d(expand1x1)
    self.expand3 = nn.Conv2d(squeeze1x1, expand3x3, kernel_size = 3, padding = 1)
    self.bn3 = nn.BatchNorm2d(expand3x3)
    self.relu2 = nn.ReLU(inplace=True)
    #for m in self.modules():
    #  if isinstance(m, nn.Conv2d):
    #    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
    #    m.weight.data.normal_(0, math.sqrt(2./n))
  
  def forward(self, x):
    out = self.squeeze(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out1 = self.expand1(out)
    out1 = self.bn2(out1)
    out2 = self.expand3(out)
    out2 = self.bn3(out2)
    out = torch.cat([out1, out2], 1)
    out = self.relu2(out)

    return out
  
class SqueezeNet(nn.Module):
  def __init__(self):
    super(SqueezeNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(96)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) #16x16
    self.fire2 = Fire(96, 16, 64, 64)
    self.fire3 = Fire(128, 16, 64, 64)
    self.fire4 = Fire(128, 32, 128, 128)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) #8x8
    self.fire5 = Fire(256, 32, 128, 128)
    self.fire6 = Fire(256, 48, 192, 192)
    self.fire7 = Fire(384, 48, 192, 192)
    self.fire8 = Fire(384, 64, 256, 256)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) #4x4
    self.fire9 = Fire(512, 64, 256, 256)
    self.conv2 = nn.Conv2d(512, 100, kernel_size=1, stride=1)
    #self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4) #1x1
    self.classifier = nn.Sequential(self.conv2, nn.AvgPool2d(kernel_size=4, stride=4))
    #for m in self.modules():
    #  if isinstance(m, nn.Conv2d):
    #    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
    #    m.weight.data.normal_(0, math.sqrt(2. / n))
    #  elif isinstance(m, nn.BatchNorm2d):
    #      m.weight.data.fill_(1)
    #      m.bias.data.zero_()
  
  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.maxpool1(out)
    out = self.fire2(out)
    out = self.fire3(out)
    out = self.fire4(out)
    out = self.maxpool2(out)
    out = self.fire5(out)
    out = self.fire6(out)
    out = self.fire7(out)
    out = self.fire8(out)
    out = self.maxpool3(out)
    out = self.fire9(out)
    out = self.classifier(out)
    #out = self.avg_pool(out)
    out = torch.flatten(out, 1)
    #out = self.softmax(out)
    # flatten to [128, 100]
    return out
