class Fire(nn.Module):
  def __init__(self, in_planes, squeeze1x1, expand1x1, expand1x3):
    super(Fire, self).__init__()
    self.in_planes = in_planes
    self.squeeze1x1 = nn.Conv2d(in_planes, squeeze1x1, kernel_size = 1)
    self.expand1x1 = nn.Conv2d(squeeze1x1, expand1x1, kernel_size = 1)
    self.expand3x3 = nn.Conv2d(squeeze1x1, expand3x3,kernel_size = 3, padding = 1)
  def forward(self, x):
    out = F.relu(self.squeeze1x1(x))
    return torch.cat([F.relu(self.expand1x1(out)),F.relu(self.expand3x3(out))], 1)

  
class SqueezeNet(nn.Module):
  super(SqueezeNet, self).__init__()
  self.layers = 
