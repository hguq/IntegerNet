from torch.nn import *
from layer import *


class MNISTNET_INT(torch.nn.Module):
    """
    Implementation of a small network for mnist dataset
    """

    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()
        self.conv = Sequential(
            QuantizeConv(n_bits, img_size[2], 10, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=10),

            QuantizeConv(n_bits, 10, 10, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=10),
            MaxPool2d(2, 2),
            # 110 * 110

            QuantizeConv(n_bits, 10, 10, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=10),

            QuantizeConv(n_bits, 10, 10, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=10),
            MaxPool2d(2, 2),
            # 7 * 7

            QuantizeConv(n_bits, 10, 10, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=10),

            QuantizeConv(n_bits, 10, 10, 3, 1, 1),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=10),
            MaxPool2d(2, 2),
            # 3 * 3
        )

        self.fc = Sequential(
            QuantizeFc(n_bits, 3 * 3 * 10, 24),
            QuantizeActSBN(signed=False, n_bits=n_bits, num_features=24, is_conv=False),
            QuantizeFc(n_bits, 24, num_classes),
            QuantizeActSBN(signed=True, n_bits=n_bits, num_features=num_classes, is_conv=False)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
