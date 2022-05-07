import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from openselfsup.utils import get_root_logger
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
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

# Simple ResNet Block

class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

# Bottleneck block

class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out

@BACKBONES.register_module
class ResNetCloser(nn.Module):
    maml = False #Default
    def __init__(self,block_name,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNetCloser,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        self.list_of_num_layers = list_of_num_layers
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        if block_name == 'SimpleBlock':
            block = SimpleBlock
        elif block_name == 'BottleneckBlock':
            block = BottleneckBlock
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is None:
            init_layer(self.trunk[0])
            init_layer(self.trunk[1])
            idx = 4
            for i in range(4):
                for j in range(self.list_of_num_layers[i]):
                    for layer in self.trunk[idx].parametrized_layers:
                        init_layer(layer)
                    idx = idx + 1
            
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self,x):
        out = self.trunk(x)
        return tuple([out])




def ResNet10( flatten = True):
    return ResNetCloser(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18( flatten = True):
    return ResNetCloser(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34( flatten = True):
    return ResNetCloser(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50( flatten = True):
    return ResNetCloser(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101( flatten = True):
    return ResNetCloser(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)
