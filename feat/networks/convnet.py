import torch.nn as nn
import torch

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2)
        )

def conv_block_no_pooling(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        )

class ConvNet4(nn.Module):

    def __init__(self, x_dim = 3, hid_dim = 64, z_dim = 64, pooling = False):
        super(ConvNet4, self).__init__()
        self.layer1 = conv_block(x_dim, hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        if pooling:
            self.layer3 = conv_block(hid_dim, hid_dim)
            self.layer4 = conv_block(hid_dim, z_dim)
        else:
            self.layer3 = conv_block_no_pooling(hid_dim, hid_dim)
            self.layer4 = conv_block_no_pooling(hid_dim, z_dim)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

class ConvNet6(nn.Module):
    
    def __init__(self, x_dim = 3, hid_dim = 64, z_dim = 64):
        super(ConvNet6, self).__init__()
        self.layer1 = conv_block(x_dim, hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        self.layer3 = conv_block_no_pooling(hid_dim, hid_dim)
        self.layer4 = conv_block_no_pooling(hid_dim, hid_dim)
        self.layer5 = conv_block_no_pooling(hid_dim, hid_dim)
        self.layer6 = conv_block_no_pooling(hid_dim, z_dim)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x4)

        return x6
