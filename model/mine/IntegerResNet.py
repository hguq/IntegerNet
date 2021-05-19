import torch
from torch import nn
# from torch.nn import Linear, Conv2d, BatchNorm2d, BatchNorm1d, ReLU, Sequential
from layer.quantize import QuantizeConv, QuantizeFc, QuantizeSignedActSBN, ReLUQuantizeUnsignedActSBN
from torch.nn import MaxPool2d
from torch.nn import functional


class BasicBlockQuantize(nn.Module):
    """
    The basic block in resnet.
    consist of ---conv bn relu conv bn +-->relu structure
                |                     |
                +---------------------+
    consist of  int4 --- conv --- QuanUnsigned --- conv --- QuanUnsigned --- int4
                      |                                  |
                      +-------------(conv)---------------+

    """

    def __init__(self, in_channels, out_channels, n_bits, down_sample=False):
        super().__init__()

        """self.residual_function = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if down_sample else 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )"""
        self.residual_function = nn.Sequential(
            QuantizeConv(n_bits=n_bits, in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                         stride=1),
            MaxPool2d(2, 2) if down_sample else nn.Sequential(),
            ReLUQuantizeUnsignedActSBN(n_bits=n_bits, num_features=out_channels, is_conv=True),
            QuantizeConv(n_bits=n_bits, in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                         stride=1)
        )

        self.shortcut = nn.Sequential()

        # shortcut应该保证输出与residual的输出是相符合的
        """if down_sample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )"""
        if down_sample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                QuantizeConv(n_bits=n_bits, in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                             stride=1, padding=0),
                MaxPool2d(2, 2) if down_sample else nn.Sequential()  # if down_sample else nn.Sequential(),
            )

        self.post = ReLUQuantizeUnsignedActSBN(n_bits=n_bits, num_features=out_channels, is_conv=True)

    def forward(self, x):
        # print("x:" + str(x.shape))
        # print("res:" + str(self.residual_function(x).shape))
        # print("short:" + str(self.shortcut(x).shape))
        s = self.residual_function(x) + self.shortcut(x)
        return self.post(s)


class IntegerResNet(torch.nn.Module):
    """
    ResNet implementation.
    """

    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()

        self.res = torch.nn.Sequential(
            QuantizeConv(n_bits=n_bits, in_channels=img_size[2], out_channels=64, kernel_size=7, stride=2, padding=3),
            # 128 * 128
            ReLUQuantizeUnsignedActSBN(n_bits=n_bits, num_features=64),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            # 64 * 64

            BasicBlockQuantize(64, 64, n_bits=n_bits, down_sample=False),

            BasicBlockQuantize(64, 64, n_bits=n_bits, down_sample=True),
            # 32 * 32

            BasicBlockQuantize(64, 128, n_bits=n_bits, down_sample=True),
            # 16 * 16

            BasicBlockQuantize(64, 128, n_bits=n_bits, down_sample=True),
            # 8 * 8
            # 16 * 16

            BasicBlockQuantize(128, 256, n_bits=n_bits, down_sample=True),
            # 8 * 8

            BasicBlockQuantize(256, 256, n_bits=n_bits, down_sample=True),
            # 4 * 4

            # BasicBlockQuantize(128, 256, n_bits=n_bits, down_sample=True),
            # BasicBlockQuantize(256, 256, n_bits=n_bits, down_sample=False),

            # BasicBlockQuantize(256, 512, n_bits=n_bits, down_sample=True),
            # BasicBlockQuantize(512, 512, n_bits=n_bits, down_sample=False),

            # BasicBlockQuantize(512, 512, n_bits=n_bits, down_sample=True),
            # BasicBlockQuantize(512, 512, n_bits=n_bits, down_sample=False)
        )
        n_feature = 4 * 4 * 256
        self.fc = torch.nn.Sequential(
            QuantizeFc(n_bits, n_feature, 1024, quantize_act=None),
            ReLUQuantizeUnsignedActSBN(n_bits=n_bits, num_features=1024, is_conv=False),

            QuantizeFc(n_bits, 1024, 512, quantize_act=None),
            ReLUQuantizeUnsignedActSBN(n_bits=n_bits, num_features=512, is_conv=False),

            QuantizeFc(n_bits, 512, num_classes, quantize_act=None),
            QuantizeSignedActSBN(n_bits=n_bits, num_features=num_classes, is_conv=False)
        )

    def forward(self, x):
        out = self.res(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
