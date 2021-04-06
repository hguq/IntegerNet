from torch.nn import *
from layer import *


class MnistNet(torch.nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.conv = torch.nn.Sequential(
            Conv2d(img_size[2], 16, 3, 1, 1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2, 2),

            Conv2d(16, 16, 3, 1, 1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2, 2),
        )

        self.fc = Sequential(
            Linear(img_size[0] * img_size[1] // 16 * 16, 128),
            BatchNorm1d(128),
            ReLU(),

            Linear(128, num_classes),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
