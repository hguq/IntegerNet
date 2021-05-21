from abc import ABC
from layer.misc import get_bounds
import torch


class VerifyInteger(torch.nn.Module, ABC):
    """
    Verify input is integer in specified frequency.
    """

    def __init__(self, n_bits=None, signed=None, freq=1000):
        super().__init__()
        self.n_bits = n_bits
        if n_bits is not None:
            self.lower_boud, self.upper_bound = get_bounds(n_bits, signed)
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
