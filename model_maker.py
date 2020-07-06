"""
The model maker for Eye nerve segmentation project, copied from : https://github.com/usuyama/pytorch-unet
"""
import os
import torch
import torch.nn as nn
from utils.time_recorder import time_keeper
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, flags, n_class=1):
        super().__init__()
        # setting up the time keeper to do the prifling of the time
        #self.tk = time_keeper('forward_model_time.txt')
        if flags.network_backbone == 'resnet_18':
            self.base_model = models.resnet18(pretrained=flags.pretrain)
        elif flags.network_backbone == 'resnet_50':
            self.base_model = models.resnet50(pretrained=flags.pretrain)
        else:
            raise ValueError("Your flags.network_backbone is neither resnet18 nor resnet50! \
            Which are the only supported models currently")
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        # tk = self.tk
        #tk.record(0)
        x_original = self.conv_original_size0(input)
        #tk.record(1)
        x_original = self.conv_original_size1(x_original)
        #tk.record(2)

        layer0 = self.layer0(input)
        #tk.record(3)
        layer1 = self.layer1(layer0)
        #tk.record(4)
        layer2 = self.layer2(layer1)
        #tk.record(5)
        layer3 = self.layer3(layer2)
        #tk.record(6)
        layer4 = self.layer4(layer3)
        #tk.record(7)

        layer4 = self.layer4_1x1(layer4)
        #tk.record(8)
        x = self.upsample(layer4)
        #tk.record(9)
        layer3 = self.layer3_1x1(layer3)
        #tk.record(10)
        x = torch.cat([x, layer3], dim=1)
        #tk.record(11)
        x = self.conv_up3(x)
        #tk.record(12)

        x = self.upsample(x)
        #tk.record(13)
        layer2 = self.layer2_1x1(layer2)
        #tk.record(14)
        x = torch.cat([x, layer2], dim=1)
        #tk.record(15)
        x = self.conv_up2(x)
        #tk.record(16)

        x = self.upsample(x)
        #tk.record(17)
        layer1 = self.layer1_1x1(layer1)
        #tk.record(18)
        x = torch.cat([x, layer1], dim=1)
        #tk.record(19)
        x = self.conv_up1(x)
        #tk.record(20)

        x = self.upsample(x)
        #tk.record(21)
        layer0 = self.layer0_1x1(layer0)
        #tk.record(22)
        x = torch.cat([x, layer0], dim=1)
        #tk.record(23)
        x = self.conv_up0(x)
        #tk.record(24)

        x = self.upsample(x)
        #tk.record(25)
        x = torch.cat([x, x_original], dim=1)
        #tk.record(26)
        x = self.conv_original_size2(x)
        #tk.record(27)

        out = self.conv_last(x)
        #tk.record(28)

        return out
