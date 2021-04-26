from torch.nn import *
from layer import *

qs = QuantizeSignedActSBN


class IntegerCifar100Net(torch.nn.Module):
    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()
        self.conv = torch.nn.Sequential(
            QuantizeConv(n_bits, img_size[2], 64, 3, 1, 1),
            qs(n_bits=n_bits, num_features=64),
            ReLU(),
            QuantizeConv(n_bits, 64, 64, 3, 1, 1),
            qs(n_bits=n_bits, num_features=64),
            ReLU(),
            QuantizeConv(n_bits, 64, 64, 3, 1, 1),
            qs(n_bits=n_bits, num_features=64),
            ReLU(),
            MaxPool2d(2, 2),

            QuantizeConv(n_bits, 64, 128, 3, 1, 1),
            qs(n_bits=n_bits, num_features=128),
            ReLU(),
            QuantizeConv(n_bits, 128, 128, 3, 1, 1),
            qs(n_bits=n_bits, num_features=128),
            ReLU(),
            QuantizeConv(n_bits, 128, 128, 3, 1, 1),
            qs(n_bits=n_bits, num_features=128),
            ReLU(),
            MaxPool2d(2, 2),

            QuantizeConv(n_bits, 128, 256, 3, 1, 1),
            qs(n_bits=n_bits, num_features=256),
            ReLU(),
            QuantizeConv(n_bits, 256, 256, 3, 1, 1),
            qs(n_bits=n_bits, num_features=256),
            ReLU(),
            QuantizeConv(n_bits, 256, 256, 3, 1, 1),
            qs(n_bits=n_bits, num_features=256),
            ReLU(),
            MaxPool2d(2, 2),
        )

        self.fc = Sequential(
            QuantizeFc(n_bits, img_size[0] * img_size[1] // 64 * 256, 512),
            qs(n_bits=n_bits, num_features=512, is_conv=False),
            ReLU(),

            QuantizeFc(n_bits, 512, num_classes),
            qs(n_bits=n_bits, num_features=num_classes, is_conv=False),

        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
