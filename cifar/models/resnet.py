"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, affine=True,bn=True,bias=False,res=True,scale=1.0,momentum=0.1):
        super().__init__()
        self.scale = scale
        self.bn = bn
        self.res = res
        print('non linearity scale: {}'.format(scale))

        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv',nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias))
        
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv',nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=bias))
        
        if self.bn:
            self.conv1.add_module('bn',nn.BatchNorm2d(out_channels, affine=affine,momentum=momentum))
            self.conv2.add_module('bn',nn.BatchNorm2d(out_channels * BasicBlock.expansion, affine=affine,momentum=momentum))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:            
            self.shortcut.add_module('conv',nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=bias))
            if self.bn:
                self.shortcut.add_module('bn',nn.BatchNorm2d(out_channels * BasicBlock.expansion, affine=affine,momentum=momentum))

    def forward(self, x):
        if self.res:
            residual = self.shortcut(x)
            output = self.conv1(x)
            output = nn.ReLU(inplace=True)(output * self.scale)
            output = self.conv2(output)
            output = nn.ReLU(inplace=True)(output + residual)
            return output 
        else:
            output = self.conv1(x)
            output = nn.ReLU(inplace=True)(output * self.scale)
            output = self.conv2(output)
            output = nn.ReLU(inplace=True)(output * self.scale)
            return output 

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, affine=True,bn=True,bias=False,res=True,scale=1.0,momentum=0.1):
        super().__init__()
        self.scale = scale
        self.bn = bn
        self.res = res

        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv',nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv',nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=bias))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv',nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=bias))

        if self.bn:
            self.conv1.add_module('bn',nn.BatchNorm2d(out_channels, affine=affine,momentum=momentum))
            self.conv2.add_module('bn',nn.BatchNorm2d(out_channels, affine=affine,momentum=momentum))
            self.conv3.add_module('bn',nn.BatchNorm2d(out_channels * BottleNeck.expansion, affine=affine,momentum=momentum))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module('conv',nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=bias))
            if self.bn:
                self.shortcut.add_module('bn',nn.BatchNorm2d(out_channels * BottleNeck.expansion, affine=affine,momentum=momentum))
        
    def forward(self, x):
        if self.res:
            residual = self.shortcut(x)
            output = self.conv1(x)
            output = nn.ReLU(inplace=True)(output * self.scale)
            output = self.conv2(output)
            output = nn.ReLU(inplace=True)(output * self.scale)
            output = self.conv3(output)
            output = nn.ReLU(inplace=True)(output + residual)
            return output 
        else:
            output = self.conv1(x)
            output = nn.ReLU(inplace=True)(output * self.scale)
            output = self.conv2(output)
            output = nn.ReLU(inplace=True)(output * self.scale)
            output = self.conv3(output)
            output = nn.ReLU(inplace=True)(output * self.scale)
            return output 

def SRELU(x):
    return F.ReLU(x*1.4142)

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100,affine=True,bn=True,bias=False,nl=nn.ReLU):
        super().__init__()
        zero_init_residual=False
        momentum = 0.1
        scale = 1.0
        res = True 
        affine = True 
        bn = True
        bias = False

        self.in_channels = 64
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=bias))
        if bn:
            self.conv1.add_module('bn',nn.BatchNorm2d(64, affine=affine))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=momentum)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=momentum)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=momentum)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=momentum)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes,bias=True) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    if res and bn and affine:
                        nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    if res and bn and affine:
                        nn.init.constant_(m.bn2.weight, 0)
                        
    def _make_layer(self, block, out_channels, num_blocks, stride, affine=True,bn=True,bias=True,res=True,scale=1.0,momentum=0.1):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=momentum))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output 

def resnet18(num_classes=100):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)

def resnet34(num_classes=100):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes)

def resnet50(num_classes=100):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],num_classes=num_classes)

def resnet101(num_classes=100):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=num_classes)

def resnet152(num_classes=100):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



