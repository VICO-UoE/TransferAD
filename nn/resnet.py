import gdown
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

PARAMS_URL = "https://drive.google.com/uc?id=1uRmyJfs5OW2s0USoue6WbPdr_04MNOFI"


def resnet26(config, num_classes):
    net = ResNet(3*[4], num_classes, pre_relu=True, ra=config.model == "adra")
    
    if not os.path.isfile(config.params_path):
        gdown.download(PARAMS_URL, config.params_path, quiet=False)
    
    state_dict_pretrained = torch.load(config.params_path)
        
    # Throw away linear layer
    del state_dict_pretrained["linears.weight"]
    del state_dict_pretrained["linears.bias"]

    q = net.load_state_dict(state_dict_pretrained, strict=False)

    assert len(q.unexpected_keys) == 0
    m_keys = ["linears", "pre_ra", "ra1", "ra2"] if config.model == "adra" else ["linears"]
    assert all(any(k in key for k in m_keys) for key in q.missing_keys)

    if config.model == "adra":
        for _, m in net.named_modules():
            if isinstance(m, torch.nn.Conv2d) and (m.kernel_size[0] == 3):
                m.weight.requires_grad = False  # Fix 3x3 convolutions

    return net


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=0):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = shortcut
        if self.shortcut:
            self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.shortcut:
            residual = self.avgpool(x)
            residual = torch.cat((residual, residual * 0), 1)

        y += residual
        y = F.relu(y)

        return y


class AdapterBlock(BasicBlock):
    def __init__(self, in_planes, planes, stride=1, shortcut=0):
        super(AdapterBlock, self).__init__(in_planes, planes, stride=stride, shortcut=shortcut)

        self.ra1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.ra2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        residual = x

        y = self.conv1(x) + self.ra1(x)
        y = self.bn1(y)
        y = F.relu(y)

        y = self.conv2(y) + self.ra2(y)
        y = self.bn2(y)

        if self.shortcut:
            residual = self.avgpool(x)
            residual = torch.cat((residual, residual * 0), 1)

        y += residual
        y = F.relu(y)

        return y


class ResNet(nn.Module):
    def __init__(self, nblocks, num_classes, pre_relu=True, ra=False):
        super(ResNet, self).__init__()

        width = [64, 128, 256]
        self.in_planes = width[0]

        self.pre_conv = nn.Conv2d(3, width[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.pre_ra = nn.Conv2d(3, width[0], kernel_size=1, stride=1, padding=0, bias=False) if ra else None
        self.pre_bn = nn.BatchNorm2d(width[0])
        self.pre_relu = nn.ReLU(True) if pre_relu else nn.Identity()

        self.layer1 = self._make_layer(AdapterBlock if ra else BasicBlock, width[0], nblocks[0], stride=1)
        self.layer2 = self._make_layer(AdapterBlock if ra else BasicBlock, width[1], nblocks[1], stride=2)
        self.layer3 = self._make_layer(AdapterBlock if ra else BasicBlock, width[2], nblocks[2], stride=2)

        self.end_bn_relu = nn.Sequential(nn.BatchNorm2d(width[2]), nn.ReLU(True))
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linears = nn.Linear(width[2], num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, nblocks, stride=1):
        shortcut = 0
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1

        layers = nn.ModuleList()
        layers.append(block(self.in_planes, planes, stride, shortcut))

        self.in_planes = planes * block.expansion
        for _ in range(1, nblocks):
            layers.append(block(self.in_planes, planes))
        return layers

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def params(self, backprop=False):
        param = torch.Tensor().cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and (m.kernel_size[0] == 3):
                param = torch.cat((param, m.weight.flatten()), 0)

        if not backprop:
            param = param.data

        return param

    def forward(self, x):
        if self.pre_ra is not None:
            x = self.pre_conv(x) + self.pre_ra(x)
        else:
            x = self.pre_conv(x)

        x = self.pre_bn(x)
        x = self.pre_relu(x)

        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)

        x = self.end_bn_relu(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.linears(x)
        
        return x
