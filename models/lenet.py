
'''
lenet for cifar in pytorch
Reference:
'''

import torch
import torch.nn as nn
import math

from models.quant_layer import *
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F


def Conv2D_custom(in_planes, out_planes, kernel_size, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False)


def QuantConv2D_custom(in_planes, out_planes, kernel_size, stride=1):
    " 3x3 quantized convolution with padding "
    return QuantConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False)


def QuantLinear_custom(in_planes, out_planes):
    " 3x3 quantized Linear "
    return QuantLinear2d(in_planes, out_planes, bias=False)



class LeNet(nn.Module):
    def __init__(self, float=float):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1)
        # self.conv1 = QuantConv2D_custom(3, 6, 5, stride=1)

        self.pool = nn.MaxPool2d(2, 2)  #
        # self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        self.conv2 = QuantConv2D_custom(6, 16, 5, stride=1)

        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

        self.fc1 = QuantLinear_custom(16 * 5 * 5, 120)
        self.fc2 = QuantLinear_custom(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x_in):
        # x = self.pool(F.relu(self.conv1(x_in)))
        x = self.pool(F.relu(self.conv1(x_in)))
        # x2 = self.pool(F.relu(self.conv12(x_in)))
        # print(x - x2)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print(self.fc3(x).shape)
        # x = F.softmax(self.fc3(x), dim=1)
        x =self.fc3(x)
        return x
    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()


def lenet(**kwargs):
    model = LeNet(**kwargs)
    return model



if __name__ == '__main__':
    pass
    # net = resnet20_cifar(float=True)
    # y = net(torch.randn(1, 3, 64, 64))
    # print(net)
    # print(y.size())