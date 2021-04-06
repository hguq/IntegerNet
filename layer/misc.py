from abc import ABC
from torch import nn
import torch.nn
import torch
from tensorboardX import SummaryWriter
import time


def get_bounds(n_bits):
    upper_bound = 2 ** (n_bits - 1) - 1
    lower_bound = -upper_bound
    return lower_bound, upper_bound


class MyBatchNorm(nn.Module, ABC):
    def __init__(self, num_features, is_conv=True, decay=0.9):
        super().__init__()
        self.num_features = num_features
        self.is_conv = is_conv
        self.decay = decay

        if is_conv:
            self.reduce_dim = (0, 2, 3)
            self.buffer_size = (1, self.num_features, 1, 1)
        else:
            self.reduce_dim = (0,)
            self.buffer_size = (1, self.num_features)

        self.register_buffer("running_mean", torch.zeros(self.buffer_size), persistent=True)
        self.register_buffer("running_std", torch.ones(self.buffer_size), persistent=True)

        self.hot = False

    def _update_buffer(self, cur_mean, cur_std):
        with torch.no_grad():
            if self.hot:
                self.running_mean = self.running_mean * self.decay + cur_mean * (1 - self.decay)
                self.running_std = self.running_std * self.decay + cur_std * (1 - self.decay)
            else:
                self.hot = True
                self.running_mean.copy_(cur_mean)
                self.running_std.copy_(cur_std)

    def forward(self, x):
        if self.training:
            cur_mean = x.mean(dim=self.reduce_dim, keepdim=True)
            cur_std = x.std(dim=self.reduce_dim, keepdim=True)
            self._update_buffer(cur_mean, cur_std)
            return (x - cur_mean) / cur_std
        else:
            return (x - self.running_mean) / self.running_std

    def __str__(self):
        return super().__str__() + ("conv" if self.is_conv else "fc") + "_" + str(self.num_features)


class DistMonitor(nn.Module, ABC):
    def __init__(self, module_name, log_freq=100):
        super().__init__()
        self.t = time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))
        self.writer = SummaryWriter("log/" + module_name + "_" + self.t)
        self.step = 0
        self.log_freq = log_freq
        self._on = True

    def turn_on(self):
        self._on = True

    def turn_off(self):
        self._on = False

    def update(self):
        self.step = self.step + 1

    def forward(self, x, prev):
        if self._on and self.step % self.log_freq == 0:
            self.writer.add_histogram(str(prev), x, global_step=self.step)
            for name, par in prev.named_parameters():
                self.writer.add_histogram(str(prev) + name, par, global_step=self.step)
            for name, par in prev.named_buffers():
                self.writer.add_histogram(str(prev) + name, par, global_step=self.step)
        return x
