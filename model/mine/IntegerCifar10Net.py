from torch.nn import *
from layer import *


class IntegerCifar10Net(torch.nn.Module):
    def __init__(self, img_size, num_classes, quan_bits, acc_bits):
        super().__init__()
        self.conv = torch.nn.Sequential(
            QuanConv(quan_bits, img_size[2], 64, 3, 1, 1, quantize_act=False),
            LimitAcc(acc_bits),
            VerifyInteger(acc_bits, signed=True),
            ReLU(),
            QuantizeUnsignedAct(in_bits=acc_bits, out_bits=quan_bits, num_features=64),
            VerifyInteger(quan_bits, signed=False),

            QuanConv(quan_bits, 64, 64, 3, 1, 1, quantize_act=False),
            LimitAcc(acc_bits),
            VerifyInteger(acc_bits, signed=True),
            ReLU(),
            QuantizeUnsignedAct(in_bits=acc_bits, out_bits=quan_bits, num_features=64),
            VerifyInteger(quan_bits, signed=False),
            MaxPool2d(2, 2),

            QuanConv(quan_bits, 64, 128, 3, 1, 1, quantize_act=False),
            LimitAcc(acc_bits),
            VerifyInteger(acc_bits, signed=True),
            ReLU(),
            QuantizeUnsignedAct(in_bits=acc_bits, out_bits=quan_bits, num_features=128),
            VerifyInteger(quan_bits, signed=False),

            QuanConv(quan_bits, 128, 128, 3, 1, 1, quantize_act=False),
            LimitAcc(acc_bits),
            VerifyInteger(acc_bits, signed=True),
            ReLU(),
            QuantizeUnsignedAct(in_bits=acc_bits, out_bits=quan_bits, num_features=128),
            VerifyInteger(quan_bits, signed=False),
            MaxPool2d(2, 2),

            QuanConv(quan_bits, 128, 256, 3, 1, 1, quantize_act=False),
            LimitAcc(acc_bits),
            VerifyInteger(acc_bits, signed=True),
            ReLU(),
            QuantizeUnsignedAct(in_bits=acc_bits, out_bits=quan_bits, num_features=256),
            VerifyInteger(quan_bits, signed=False),

            QuanConv(quan_bits, 256, 256, 3, 1, 1, quantize_act=False),
            LimitAcc(acc_bits),
            VerifyInteger(acc_bits, signed=True),
            ReLU(),
            QuantizeUnsignedAct(in_bits=acc_bits, out_bits=quan_bits, num_features=256),
            VerifyInteger(quan_bits, signed=False),
            MaxPool2d(2, 2),
        )

        self.fc = Sequential(
            QuanFc(quan_bits, img_size[0] * img_size[1] // 64 * 256, 512, quantize_act=False),
            ReLU(),

            QuanFc(quan_bits, 512, num_classes, quantize_act=False),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
