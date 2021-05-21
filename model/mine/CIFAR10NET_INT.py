from torch.nn import *
from layer import *


class CIFAR10NET_INT(torch.nn.Module):
    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 32 * 32
            QuantizeConv(n_bits, img_size[2], 64, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=64),

            QuantizeConv(n_bits, 64, 64, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=64),
            MaxPool2d(2, 2),
            # 16 * 16

            QuantizeConv(n_bits, 64, 128, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=128),

            QuantizeConv(n_bits, 128, 128, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=128),
            MaxPool2d(2, 2),
            # 8 * 8

            QuantizeConv(n_bits, 128, 256, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=256),

            QuantizeConv(n_bits, 256, 256, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=256),
            MaxPool2d(2, 2),
            # 4 * 4
        )

        self.fc = Sequential(
            QuantizeFc(n_bits, 4 * 4 * 256, 512),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=512, is_conv=False),

            QuantizeFc(n_bits, 512, num_classes),
            QuantizeActSBN(signed=True, n_bits=n_bits, num_features=num_classes, is_conv=False),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
