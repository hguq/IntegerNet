from torch.nn import *
from layer import *


class IntegerImageNet(torch.nn.Module):
    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()

        self.conv = torch.nn.Sequential(
            QuantizeConv(n_bits, img_size[2], 16, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 16),

            QuantizeConv(n_bits, 16, 16, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 16),
            MaxPool2d(2, 2),
            # 128 * 128

            QuantizeConv(n_bits, 16, 32, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 32),

            QuantizeConv(n_bits, 32, 32, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 32),
            MaxPool2d(2, 2),
            # 64 * 64

            QuantizeConv(n_bits, 32, 64, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 64),

            QuantizeConv(n_bits, 64, 64, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 64),
            MaxPool2d(2, 2),
            # 32 * 32

            QuantizeConv(n_bits, 64, 128, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 128),

            QuantizeConv(n_bits, 128, 128, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 128),
            MaxPool2d(2, 2),
            # 16 * 16

            QuantizeConv(n_bits, 128, 256, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 256),

            QuantizeConv(n_bits, 256, 256, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 256),
            MaxPool2d(2, 2),
            # 8 * 8

            QuantizeConv(n_bits, 256, 256, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 256),

            QuantizeConv(n_bits, 256, 256, 3, 1, 1),
            ReLUQuantizeUnsignedActSBN(n_bits, 256),
            MaxPool2d(2, 2)
            # 4 * 4
        )

        self.fc = Sequential(
            QuantizeFc(n_bits, 4 * 4 * 256, 1024, ),
            ReLUQuantizeUnsignedActSBN(n_bits=n_bits, num_features=1024, is_conv=False),

            QuantizeFc(n_bits, 1024, 512, ),
            ReLUQuantizeUnsignedActSBN(n_bits=n_bits, num_features=512, is_conv=False),

            QuantizeFc(n_bits, 512, num_classes, ),
            QuantizeSignedActSBN(n_bits=n_bits, num_features=num_classes, is_conv=False),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
