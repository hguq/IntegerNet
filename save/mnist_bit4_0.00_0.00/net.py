from torch.nn import *
from layer import *


class IntegerMnistNet(torch.nn.Module):
    def __init__(self, img_size, num_classes, n_bits):
        super().__init__()
        self.conv = torch.nn.Sequential(
            QuanConv(n_bits, img_size[2], 16, 3, 1, 1, quantize_act=False),
            QuantizeSignedAct(n_bits=n_bits, num_features=16),
            ReLU(),
            MaxPool2d(2, 2),

            QuanConv(n_bits, 16, 16, 3, 1, 1, quantize_act=False),
            QuantizeSignedAct(n_bits=n_bits, num_features=16),
            ReLU(),
            MaxPool2d(2, 2),
        )

        self.fc = Sequential(
            QuanFc(n_bits, img_size[0] * img_size[1] // 16 * 16, 128, quantize_act=False),
            QuantizeSignedAct(n_bits=n_bits, num_features=128),
            ReLU(),

            QuanFc(n_bits, 128, num_classes),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
