import torch
from torch.nn import ReLU, Sequential
from torch import nn
from torch.nn import functional
from layer import *


class QuanResidualBlock(nn.Module):
    def __init__(self, n_bits, in_channel, out_channel, down_sample=False):
        super().__init__()

        self.residual_function = nn.Sequential(
            QuanConv(n_bits, in_channel, out_channel, kernel_size=3, stride=2 if down_sample else 1, padding=1),
            nn.ReLU(),
            QuanConv(n_bits, out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Sequential()

        if down_sample:
            self.shortcut = nn.Sequential(
                QuanConv(n_bits, in_channel, out_channel, kernel_size=1, stride=2, padding=0)
            )

    def forward(self, x):
        added = self.residual_function(x) + self.shortcut(x)
        return torch.nn.functional.relu(added)


class IntegerResNet(torch.nn.Module):
    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()

        self.res = Sequential(
            QuanConv(n_bits=n_bits, in_channels=img_size[2], out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLU(),

            QuanResidualBlock(n_bits, 64, 64, down_sample=False),
            QuanResidualBlock(n_bits, 64, 64, down_sample=False),

            QuanResidualBlock(n_bits, 64, 128, down_sample=True),
            QuanResidualBlock(n_bits, 128, 128, down_sample=False),

            QuanResidualBlock(n_bits, 128, 256, down_sample=True),
            QuanResidualBlock(n_bits, 256, 256, down_sample=False),

            QuanResidualBlock(n_bits, 256, 512, down_sample=True),
            QuanResidualBlock(n_bits, 512, 512, down_sample=False)
        )

        n_feature = img_size[0] * img_size[1] // 64 * 512
        self.fc = Sequential(
            QuanFc(n_bits, n_feature, 512),
            QuanFc(n_bits, 512, num_classes)
        )

    def forward(self, x):
        out = self.res(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
