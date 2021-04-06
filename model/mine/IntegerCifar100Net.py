from torch.nn import *
from layer import *


class IntegerCifar100Net(torch.nn.Module):
    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()
        self.conv = torch.nn.Sequential(
            QuanConv(n_bits, img_size[2], 64, 3, 1, 1),
            ReLU(),
            QuanConv(n_bits, 64, 64, 3, 1, 1),
            ReLU(),
            QuanConv(n_bits, 64, 64, 3, 1, 1),
            ReLU(),
            MaxPool2d(2, 2),

            QuanConv(n_bits, 64, 128, 3, 1, 1),
            ReLU(),
            QuanConv(n_bits, 128, 128, 3, 1, 1),
            ReLU(),
            QuanConv(n_bits, 128, 128, 3, 1, 1),
            ReLU(),
            MaxPool2d(2, 2),

            QuanConv(n_bits, 128, 256, 3, 1, 1),
            ReLU(),
            QuanConv(n_bits, 256, 256, 3, 1, 1),
            ReLU(),
            QuanConv(n_bits, 256, 256, 3, 1, 1),
            ReLU(),
            MaxPool2d(2, 2),
        )

        self.fc = Sequential(
            QuanFc(n_bits, img_size[0] * img_size[1] // 64 * 256, 512),
            ReLU(),

            QuanFc(n_bits, 512, num_classes),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
