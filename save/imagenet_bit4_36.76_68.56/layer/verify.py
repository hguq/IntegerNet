from abc import ABC

import torch


class VerifyInteger(torch.nn.Module, ABC):
    """
    以一定的频率，验证输入值确实是整数
    """

    def __init__(self, n_bits=None, signed=None, freq=1000):
        super().__init__()
        self.n_bits = n_bits
        if n_bits is not None:
            if signed:
                self.upper_bound = 2 ** (self.n_bits - 1) - 1
                self.lower_bound = -self.upper_bound
            else:
                self.upper_bound = 2 ** self.n_bits - 1
                self.lower_bound = 0
        else:
            self.upper_bound = self.lower_bound = None
        self.cnt = 0
        self.freq = freq

    def forward(self, x):
        self.cnt += 1
        if self.cnt == self.freq:
            self.cnt = 0
            loss = torch.sum(torch.abs(torch.round(x) - x))
            if loss != 0:
                raise ValueError("Not Integer at %s" % x)
            if self.n_bits is not None:
                max_val, min_val = torch.max(x), torch.min(x)
                if max_val > self.upper_bound:
                    raise ValueError("Exceed quantize max value at %s" % x)
                if min_val < self.lower_bound:
                    raise ValueError("Exceed quantize min value at %s" % x)
        return x
