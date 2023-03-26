class SimpleNet(nn.Module):
    cfg = #add layers here

    def __init__(self, num_classes = 100, in_planes = 3, stride=1):
        #super(Block, self).__init__()
        # WIP

    def forward(self, x):
        # WIP
        # for batch norm use nn.BatchNorm2d(in_planes)
        # for relu use torch.nn.functional.relu
                
        out = F.relu(self.bn1(self.conv1(x)))
        return out
        
        
    def _make_layers(self):
        layers: List[nn.Module] = []
        input_channel = self.in_planes
        for x in self.cfg:
          # WIP
          
          
        
