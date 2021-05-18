from torch.nn import *
from layer import *


class ImagenetNet(torch.nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()

        self.conv = torch.nn.Sequential(
            Conv2d(img_size[2], 16, 3, 1, 1),
            BatchNorm2d(num_features=16),
            ReLU(),

            Conv2d(16, 16, 3, 1, 1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2, 2),
            # 128 * 128

            Conv2d(16, 32, 3, 1, 1),
            BatchNorm2d(num_features=32),
            ReLU(),

            Conv2d(32, 32, 3, 1, 1),
            BatchNorm2d(num_features=32),
            ReLU(),
            MaxPool2d(2, 2),
            # 64 * 64

            Conv2d(32, 64, 3, 1, 1),
            BatchNorm2d(num_features=64),
            ReLU(),

            Conv2d(64, 64, 3, 1, 1),
            BatchNorm2d(num_features=64),
            ReLU(),
            MaxPool2d(2, 2),
            # 32 * 32

            Conv2d(64, 128, 3, 1, 1),
            BatchNorm2d(num_features=128),
            ReLU(),

            Conv2d(128, 128, 3, 1, 1),
            BatchNorm2d(num_features=128),
            ReLU(),
            MaxPool2d(2, 2),
            # 16 * 16

            Conv2d(128, 256, 3, 1, 1),
            BatchNorm2d(num_features=256),
            ReLU(),

            Conv2d(256, 256, 3, 1, 1),
            BatchNorm2d(num_features=256),
            ReLU(),
            MaxPool2d(2, 2),
            # 8 * 8

            Conv2d(256, 256, 3, 1, 1),
            BatchNorm2d(num_features=256),
            ReLU(),

            Conv2d(256, 256, 3, 1, 1),
            BatchNorm2d(num_features=256),
            ReLU(),
            MaxPool2d(2, 2)
            # 4 * 4
        )

        self.fc = Sequential(
            Linear(4 * 4 * 256, 1024),
            BatchNorm1d(num_features=1024),

            Linear(1024, 512),
            BatchNorm1d(num_features=512),

            Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
