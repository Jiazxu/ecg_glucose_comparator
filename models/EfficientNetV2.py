# -*- coding: utf-8 -*-
"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
"""
# Reference: https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py

# Practice on ECG waves with pytorch.
# EffNet on small data
# Input: one dimension waves, [batch size, channel, points]



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm1d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm1d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv1d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


# subtract and conv
class EffNetV2(nn.Module):
    def __init__(self, cfgs, out_features=16, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        self.normal_proprecessing = nn.BatchNorm1d(1)

        # building first layer, 3x3 conv
        # channels nums: 8
        input_channel = _make_divisible(8 * width_mult, 8)
        layers = [conv_3x3_bn(1, input_channel, 1)]

        # building inverted residual blocks
        block = MBConv
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t, c, n, s, SE
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(out_features * width_mult, 8) if width_mult > 1.0 else out_features
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool1d((1))

        self.classifier = nn.Linear(output_channel, 1)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()


    def forward(self, x):
        
        x = self.normal_proprecessing(x)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        out = self.sigmoid(x)

        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()



# Encoder, feature extractor
def effnetv2_ecg_s(**kwargs):
    """
    Design for ecg dataset
    Total Params: 535,257
    """
    cfgs = [
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t,  c,  n, s, SE,     # input[batch, H, W, channels]
                                # --> B,  3, 32, 32  -- first layer --> B, 24, 32, 32
        [1,  24,  2, 1, 0],     # --> B, 18, 32, 32            
        [4,  36,  2, 2, 0],     # --> B, 24, 16, 16
        [4,  48,  4, 2, 1],     # --> B, 36,  8,  8
        [6,  48,  4, 1, 1],     # --> B, 48,  8,  8
        [6,  64,  4, 2, 1],     # --> B, 64,  4,  4
    ]
    return EffNetV2(cfgs, **kwargs)

# Encoder, feature extractor
def effnetv2_ecg_wider(**kwargs):
    """
    Design for ecg dataset
    Total Params: 
    """
    cfgs = [
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t,  c,  n, s, SE,     # input[batch, H, W, channels]
                                # --> B,  3, 32, 32  -- first layer --> B, 24, 32, 32
        [1,  48,  2, 1, 0],     # --> 32, 32            
        [4,  64,  2, 2, 0],     # --> 16, 16
        [4,  96,  4, 2, 1],     # --> 8,  8
        [6,  128,  4, 1, 1],    # --> 8,  8
        [6,  128,  4, 2, 1],    # --> 4,  4
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_ecg_xs(**kwargs):
    """
    Design for ecg dataset
    Total Params: 266,889
    """
    cfgs = [
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t,  c,  n, s, SE,     # input[batch, channels, H, W]
                                # --> B,  1, 768  -- first layer --> B, 8, 768
        [1,  24,  1, 1, 0],     # --> B,  1, 768           
        [4,  32,  1, 2, 0],     # --> B,  1, 384
        [4,  64,  1, 2, 1],     # --> B,  1, 192
        [6,  96,  1, 1, 1],     # --> B,  1, 96
        [6,  128, 1, 2, 1],     # --> B,  1, 96
    ]
    return EffNetV2(cfgs, **kwargs)

def effnetv2_ecg_l(**kwargs):
    """
    Design for ecg dataset
    Total Params: 62,265
    """
    cfgs = [
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t,  c,  n, s, SE,     # input[batch, H, W, channels]
                                # --> B,  3, 32, 32  -- first layer --> B, 24, 32, 32
        [1,  8,  1, 1, 0],     # --> B,  8, 384           
        [4,  8,  2, 2, 0],     # --> B,  8, 192
        [6,  16, 2, 2, 1],     # --> B,  16, 96
        [6,  16, 2, 2, 1],     # --> B,  16, 48
        [6,  16, 2, 2, 1],     # --> B,  16, 24
        [1,  16,  1, 1, 0],     # --> B, 16, 24
        [4,  16,  2, 2, 1],     # --> B, 16, 12
        [6,  16,  2, 2, 1],     # --> B, 16, 6
        [6,  16,  2, 2, 1],     # --> B, 16, 3
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_ecg_l_s(**kwargs):
    """
    Design for ecg dataset
    Total Params: 36 425
    """
    cfgs = [
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t,  c,  n, s, SE,     # input[batch, H, W, channels]
                                # --> B,  3, 32, 32  -- first layer --> B, 24, 32, 32
        [1,  8,  1,  1, 0],     # --> B,  8, 384           
        [4,  8,  2,  1, 0],     # --> B,  8, 192
        [6,  8,  4,  2, 1],     # --> B,  8, 96
        [6,  8,  4,  2, 1],     # --> B,  16, 48
        [6,  8,  4,  2, 1],     # --> B,  16, 24
        #[1,  8,  1,  1, 0],     # --> B,  16, 24
        [4,  8,  2,  2, 1],     # --> B,  16, 12
        [6,  8,  2,  2, 1],     # --> B,  16, 6
        [6,  8,  2,  2, 1],     # --> B,  16, 3
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_ecg_l_xs(**kwargs):
    """
    Design for ecg dataset
    Total Params: 14651
    """
    cfgs = [
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t,  c,  n, s, SE,     # input[batch, H, W, channels]
                                # --> B,  3, 32, 32  -- first layer --> B, 24, 32, 32
        [1,  8,  1,  1, 0],     # --> B,  8, 384           
        [4,  8,  1,  1, 0],     # --> B,  8, 192
        [4,  8,  2,  2, 1],     # --> B,  8, 96
        [4,  8,  2,  2, 1],     # --> B,  16, 24
        #[1,  8,  1,  1, 0],     # --> B,  16, 24
        [4,  8,  2,  2, 1],     # --> B,  16, 12
        [4,  8,  2,  2, 1],     # --> B,  16, 6
        [4,  8,  2,  2, 1],     # --> B,  16, 3
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_ecg_l_xxs(**kwargs):
    """
    Design for ecg dataset
    Total Params: 10739
    """
    cfgs = [
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t,  c,  n, s, SE,     # input[batch, H, W, channels]
                                # --> B,  3, 32, 32  -- first layer --> B, 24, 32, 32
        [1,  8,  1,  1, 0],     # --> B,  8, 384           
        [4,  8,  1,  1, 0],     # --> B,  8, 384
        [4,  8,  2,  2, 1],     # --> B,  8, 192
        [4,  8,  2,  2, 1],     # --> B,  8, 96
        #[1,  8,  1,  1, 0],     # --> B,  16, 24
        [4,  4,  1,  2, 1],     # --> B, 4, 48
        [4,  4,  1,  2, 1],     # --> B, 4, 24
        [4,  4,  1,  2, 1],     # --> B, 4, 12
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_ecg_l_xxxs(**kwargs):
    """
    Design for ecg dataset
    Total Params: 8419
    """
    cfgs = [
        # t: expansion ration, c: channels, n: repeating times, s: stride, SE
        # t,  c,  n, s, SE,     # input[batch, H, W, channels]
                                # --> B,  3, 32, 32  -- first layer --> B, 24, 32, 32
        [1,  4,  1,  1, 0],     # --> B,  8, 384           
        [4,  4,  1,  1, 0],     # --> B,  8, 384
        [4,  4,  1,  2, 1],     # --> B,  8, 192
        [4,  4,  1,  2, 1],     # --> B,  8, 96
        [1,  4,  1,  1, 0],     # --> B,  16, 24
        [4,  4,  1,  2, 1],     # --> B, 4, 48
        [4,  4,  1,  2, 1],     # --> B, 4, 24
        [4,  4,  1,  2, 1],     # --> B, 4, 12
    ]
    return EffNetV2(cfgs, **kwargs)





import numpy as np


def cal_params(model):

    Total_params = 0
    Trainable_params = 0 
    NonTrainable_params = 0 

    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue 

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'NonTrainable params: {NonTrainable_params}')


def test():
    net = effnetv2_ecg_l_xxxs()
    cal_params(net)

    x = torch.randn(3, 1, 384)

    out = net(x)

    print(out.size())
    print()
    print(out)

#test()
