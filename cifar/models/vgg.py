"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=100,embedding=False):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )    

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=False):
    #layers = []
    layers = nn.Sequential()            
    input_channel = 3
    for i,l in enumerate(cfg):
        if l == 'M':
            layers.add_module('maxpool-{}'.format(i),nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        layers.add_module('conv-{}'.format(i),nn.Conv2d(input_channel, l, kernel_size=3, padding=1,bias=False))

        if batch_norm:
            layers.add_module('bn-{}'.format(i),nn.BatchNorm2d(l))

        layers.add_module('relu-{}'.format(i),nn.ReLU(inplace=True))
        input_channel = l
    return layers

def vgg11(num_classes=100):
    return VGG(make_layers(cfg['A'], batch_norm=False),num_classes=num_classes)

def vgg13(num_classes=100):
    return VGG(make_layers(cfg['B'], batch_norm=False),num_classes=num_classes)

def vgg16(num_classes=100):
    return VGG(make_layers(cfg['D'], batch_norm=False),num_classes=num_classes)

def vgg19(num_classes=100):
    return VGG(make_layers(cfg['E'], batch_norm=False),num_classes=num_classes)


def vgg11_bn(num_classes=100):
    return VGG(make_layers(cfg['A'], batch_norm=True),num_classes=num_classes)

def vgg13_bn(num_classes=100):
    return VGG(make_layers(cfg['B'], batch_norm=True),num_classes=num_classes)

def vgg16_bn(num_classes=100):
    return VGG(make_layers(cfg['D'], batch_norm=True),num_classes=num_classes)

def vgg19_bn(num_classes=100):
    return VGG(make_layers(cfg['E'], batch_norm=True),num_classes=num_classes)


