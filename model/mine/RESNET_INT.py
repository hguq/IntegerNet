import torch
from torch import nn
from layer.quantize import QuantizeConv, QuantizeFc, QuantizeActSBN


class QuantizeBasicBlock(nn.Module):
    """
    Quantized basic block in resnet.
    consist of  int4 --- conv --- QuanUnsigned --- conv --- QuanUnsigned --- int4
                      |                                  |
                      +--------------conv----------------+
    """

    def __init__(self, n_bits, in_channels, out_channels, down_sample=False):
        super().__init__()

        self.residual_function = nn.Sequential(
            QuantizeConv(n_bits=n_bits, in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                         stride=2 if down_sample else 1, padding=1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=out_channels, is_conv=True),
            QuantizeConv(n_bits=n_bits, in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                         stride=1, padding=1)
        )

        self.shortcut = nn.Sequential(
            QuantizeConv(n_bits=n_bits, in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                         stride=2 if down_sample else 1, padding=0),
        )

        self.post = QuantizeActSBN(signed=False, n_bits=n_bits, num_features=out_channels, is_conv=True)

    def forward(self, x):
        s = self.residual_function(x) + self.shortcut(x)
        return self.post(s)


class RESNET_INT(torch.nn.Module):
    """
    RESNET implementation.
    """

    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()

        self.res = torch.nn.Sequential(
            # 256 * 256
            QuantizeConv(n_bits=n_bits, in_channels=img_size[2], out_channels=32, kernel_size=3, stride=1, padding=1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=32),
            # 256 * 256

            QuantizeBasicBlock(n_bits, 32, 32, down_sample=True),
            # 128 * 128

            QuantizeBasicBlock(n_bits, 32, 32, down_sample=False),
            # 128 * 128

            QuantizeBasicBlock(n_bits, 32, 64, down_sample=True),
            # 64 * 64

            QuantizeBasicBlock(n_bits, 64, 64, down_sample=False),
            # 64 * 64

            QuantizeBasicBlock(n_bits, 64, 128, down_sample=True),
            # 32 * 32

            QuantizeBasicBlock(n_bits, 128, 128, down_sample=False),
            # 32 * 32

            QuantizeBasicBlock(n_bits, 128, 256, down_sample=True),
            # 16 * 16

            QuantizeBasicBlock(n_bits, 256, 256, down_sample=True),
            # 8 * 8

            QuantizeBasicBlock(n_bits, 256, 256, down_sample=True),
            # 4 * 4

        )
        n_feature = 4 * 4 * 256
        self.fc = torch.nn.Sequential(
            QuantizeFc(n_bits, n_feature, 1024),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=1024, is_conv=False),

            QuantizeFc(n_bits, 1024, num_classes),
            QuantizeActSBN(signed=True, n_bits=n_bits, num_features=num_classes, is_conv=False)
        )

    def forward(self, x):
        out = self.res(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
