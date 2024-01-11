# -*- coding: utf-8 -*-
# Practice on ECG waves in pytorch
# Original Paper || MobileNetV2: Inverted Residuals and Linear BottleNecks

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 



# input data shape: [batch size, 768 points] or [batch size, 1536 points]
# Conv1d --> reshape into: [batch size, channels=1, 768] 

class Block(nn.Module):
    '''
    expansion + Depthwise conv + Pointwise conv
    '''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv1d(in_planes, 
                               planes, 
                               kernel_size=1, 
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, 
                               planes, 
                               kernel_size=3, 
                               stride=stride,
                               padding=1,
                               # set the groups=planes
                               groups=planes,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, 
                               out_planes, 
                               kernel_size=1, 
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(out_planes)

        # inverted residual bottlenet shortcut
        # if stride != 1, no shortcut 
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes,
                          out_planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm1d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out 

    
# (expansion(t), out_planes(c), num_blocks(n), stride(s))
cfg = [(1,  16, 1, 1),
       (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       (6,  96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]
cfg1 = [(1,  8, 1, 1), # --> B, 8, x_points/2
        (6, 12, 2, 2), # --> B, 12, x_points/4
        (6, 16, 3, 2), # --> B, 16, x_points/8
        (6, 32, 3, 2), # --> B, 32, x_points/16
        (6, 48, 2, 2), # --> B, 48, x_points/32
        (6, 64, 2, 2), # --> B, 64, x_points/64
        (6, 64, 1, 2)] # --> B, 64, x_points/128

cfg2 = [(1,  8, 1, 1), # --> B, 8, x_points/2
        (6, 12, 1, 2), # --> B, 12, x_points/4
        (6, 16, 2, 2), # --> B, 16, x_points/8
        (6, 32, 2, 2), # --> B, 32, x_points/16
        (6, 48, 1, 2), # --> B, 48, x_points/32
        (6, 64, 1, 2), # --> B, 64, x_points/64
        (6, 64, 1, 2)] # --> B, 64, x_points/128

cfg3 = [(1,  8, 1, 1), # --> B, 8, x_points/2
        (6, 12, 1, 2), # --> B, 12, x_points/4
        (6, 16, 1, 2), # --> B, 16, x_points/8
        (6, 16, 1, 2), # --> B, 32, x_points/16
        (6, 24, 1, 2), # --> B, 48, x_points/32
        (6, 48, 1, 2), # --> B, 64, x_points/64
        (6, 64, 1, 2)] # --> B, 64, x_points/128


class MobileNetV2(nn.Module):

    def __init__(self, CFG=cfg1):
        super().__init__()

        self.cfg = CFG
        # The first conv layer is traditional
        self.conv1 = nn.Conv1d(1, 
                               4, 
                               kernel_size=3, 
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(4)
        self.layers = self._make_layers(in_planes=4)
        self.conv2 = nn.Conv1d(64, 
                               32, 
                               kernel_size=1, 
                               stride=2,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x [batch size, channels, points]          # --> B, 1, 768/384 
        out = F.relu(self.bn1(self.conv1(x)))       # --> B, 4, 768/394
        out = self.layers(out)                      # --> B, 64, 6/3
        out = F.relu(self.bn2(self.conv2(out)))     # --> B, 32, 6/3
        out = F.avg_pool1d(out, out.size(-1))       # --> B, 32, 1
        out = out.view(out.size(0), -1)             # --> B, 32
        out = self.fc(out)                          # --> B, 1
        out = self.sigmoid(out)                     # --> B, 1
        return out

    def _make_layers(self, in_planes):
        
        layers = []
        
        for expansion, out_planes, num_blocks, stride in self.cfg:
            # [stride] only works in the first layer in one bottleneck structure
            # the other layers' stride = 1 --> strides: [stride, 1, 1, ...]
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers) # *list

from utils.model_utils import *


def test():
    model = MobileNetV2(cfg=cfg2)
    x = torch.randn(3, 1, 768)
    y = model(x)
    print(y.size())
    #print(y.max(1))
    print(y)


#test()


