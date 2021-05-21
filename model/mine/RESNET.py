import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, BatchNorm1d, ReLU, Sequential, MaxPool2d
from torch.nn import functional


class BasicBlock(nn.Module):
    """
    The basic block in resnet.
    consist of ---conv-bn-relu-conv-bn +-->relu structure
                |                     |
                +---------------------+
    """

    def __init__(self, in_channels, out_channels, down_sample=False):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if down_sample else 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        # shortcut应该保证输出与residual的输出是相符合的
        if down_sample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return torch.nn.functional.relu(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    Bottleneck block in resnet.
    """

    def __init__(self, in_channels, bottleneck_channels, out_channels, down_sample=False):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            # 在bottleneck处使用stride参数，若stride等于2，在此处发生下采样
            nn.Conv2d(bottleneck_channels, bottleneck_channels, stride=2 if down_sample else 1, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, out_channels, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()

        if down_sample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.functional.relu(self.residual_function(x) + self.shortcut(x))


class RESNET(torch.nn.Module):
    """
    RESNET implementation.
    """

    def __init__(self, img_size, num_classes):
        super().__init__()

        self.res = Sequential(
            # 256 * 256

            Conv2d(in_channels=img_size[2], out_channels=32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(),
            # 256 * 256

            BasicBlock(32, 32, down_sample=True),
            # 128 * 128

            BasicBlock(32, 32, down_sample=False),
            # 128 * 128

            BasicBlock(32, 64, down_sample=True),
            # 64 * 64

            BasicBlock(64, 64, down_sample=False),
            # 64 * 64

            BasicBlock(64, 128, down_sample=True),
            # 32 * 32

            BasicBlock(128, 128, down_sample=False),
            # 32 * 32

            BasicBlock(128, 256, down_sample=True),
            # 16 * 16

            BasicBlock(256, 256, down_sample=True),
            # 8 * 8

            BasicBlock(256, 256, down_sample=True),
            # 4 * 4
        )

        n_feature = 4 * 4 * 256
        self.fc = Sequential(
            Linear(n_feature, 1024),
            BatchNorm1d(1024),
            ReLU(),

            Linear(1024, num_classes),
            BatchNorm1d(num_classes)
        )

    def forward(self, x):
        out = self.res(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
