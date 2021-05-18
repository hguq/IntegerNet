from torch.nn import *
from layer import *


class Cifar10Net(torch.nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.conv = torch.nn.Sequential(
            Conv2d(img_size[2], 64, 3, 1, 1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 64, 3, 1, 1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2, 2),

            Conv2d(64, 128, 3, 1, 1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 128, 3, 1, 1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, 2),

            Conv2d(128, 256, 3, 1, 1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, 3, 1, 1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(2, 2),
        )

        self.fc = Sequential(
            Linear(img_size[0] * img_size[1] // 64 * 256, 1024),
            BatchNorm1d(1024),
            ReLU(),

            Linear(1024, 512),
            BatchNorm1d(512),
            ReLU(),

            Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
