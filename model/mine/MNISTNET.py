from torch.nn import *
from layer import *


class MNISTNET(torch.nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.conv = torch.nn.Sequential(
            Conv2d(img_size[2], 16, 3, 1, 1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2, 2),
            # 14 * 14

            Conv2d(16, 16, 3, 1, 1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2, 2),
            # 7 * 7

            Conv2d(16, 16, 3, 1, 1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2, 2),
            # 3 * 3
        )

        self.fc = Sequential(
            Linear(3 * 3 * 16, 128),
            BatchNorm1d(128),
            ReLU(),

            Linear(128, num_classes),
            BatchNorm1d(num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
