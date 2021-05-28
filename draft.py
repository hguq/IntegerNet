import numpy as np
import torch
import matplotlib.pyplot as plt
from math import log2
from numba import jit


def sbn(act, std, mean):
    shift = round(log2(std))
    bias = round(mean)

    return torch.floor((act - bias) / 2 ** shift)


def bn(act, std, mean):
    return (act - mean) / std


def mbn(act, std, mean):
    # 2^15 < Ms < 2^ 16
    # Ms * 2 ** -ms = 1/std
    # 2^15 < 2 ** ms / std < 2 ^ 16
    # 15 < ms - log2(std) < 16
    # ms = 15 + log2(std)
    # Ms = 2 ** ms / std
    ms = 15 + log2(std)
    Ms = 2 ** ms / std
    return (act - round(mean)) * Ms * 2 ** -ms


xs = []
bns = []
sbns = []
mbns = []
for input_std in torch.arange(1, 10, 0.01):
    random_act = torch.randn(1000000).cuda() * input_std + 1000
    xs.append(input_std.item())
    bns.append(bn(random_act, input_std, 1000).std().item())
    sbns.append(sbn(random_act, input_std, 1000).std().item())
    mbns.append(mbn(random_act, input_std, 1000).std().item())

# plt.plot(xs, bns)
plt.plot(xs, sbns)
plt.plot(xs, mbns)
plt.legend(["SBN", "mul-shift"], loc="upper right")
plt.xlabel("Standard deviation of input tensor", fontsize=14)
plt.ylabel("Deviation after applying BN", fontsize=14)
plt.show()
