# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from ..registry import BACKBONES
from openselfsup.utils import get_root_logger
from mmcv.runner import load_checkpoint

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

    #def init_weights(self):
    #    for layer in self.parametrized_layers:
    #        init_layer(layer)
        

@BACKBONES.register_module
class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        self.depth = depth
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return tuple([out])
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger, revise_keys=[(r'^backbone\.', '')])
        elif pretrained is None:
            for i in range(self.depth):
                for layer in self.trunk[i].parametrized_layers:
                    init_layer(layer)
        else:
            raise TypeError('pretrained must be a str or None')

def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)