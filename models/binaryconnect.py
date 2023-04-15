import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy

class BC():
    def __init__(self, model):
        counter = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                counter = counter + 1

        start_range = 0
        end_range = counter-1
        self.bin_range = numpy.linspace(start_range, end_range, end_range-start_range+1).astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.save_params()
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign())

            
    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)


    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):
        clip_scale=[]
        m=nn.Hardtanh(-1, 1)
        for index in range(self.num_of_params):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(clip_scale[index].data)
