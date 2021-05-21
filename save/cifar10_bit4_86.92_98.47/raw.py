import numpy as np
import math
import torch
from typing import Tuple

from layer.quantize import QuantizeFc, QuantizeConv, QuantizeActSBN
from torch.nn.modules import MaxPool2d, ReLU
import json
from dataset import get_loader
import matplotlib.pyplot as plt
from numba import jit


@jit(target="cpu", nopython=True)
def asrt_int(x: np.ndarray):
    eps = 1e-100
    assert np.sum(np.abs(x - np.floor(x))) < eps


@jit(target="cpu", nopython=True)
def raw_relu(x: np.ndarray):
    asrt_int(x)
    return np.maximum(x, 0)


@jit(target="cpu", nopython=True)
def raw_pool(image: np.ndarray, kernel: Tuple[int, int]):
    asrt_int(image)
    N, C, HI, WI = image.shape
    HO = math.floor(HI / kernel[0])
    WO = math.floor(WI / kernel[1])
    res = np.zeros(shape=(N, C, HO, WO))
    for n in range(N):
        for c in range(C):
            for ho in range(HO):
                for wo in range(WO):
                    res[n, c, ho, wo] = -1e20
                    for dh in range(kernel[0]):
                        for dw in range(kernel[1]):
                            h = ho * 2 + dh
                            w = wo * 2 + dw
                            res[n, c, ho, wo] = np.maximum(res[n, c, ho, wo], image[n, c, h, w])
    return res


@jit(target="cpu", nopython=True)
def raw_conv(image: np.ndarray, weight: np.ndarray):
    # image[N, CI, H, W] weight[CO, CI, K, K]
    asrt_int(image)
    asrt_int(weight)

    assert image.shape[1] == weight.shape[1]
    assert weight.shape[2] == weight.shape[3]
    N, CI, H, W = image.shape
    CO, _, K, _ = weight.shape
    # K must be odd
    assert K % 2

    # Kernel span range, for example: K=3, then k=1
    k = K // 2
    res = np.zeros((N, CO, H, W))
    for n in range(N):
        for w in range(W):
            for h in range(H):
                for co in range(CO):
                    for dh in range(- k, k + 1):
                        for dw in range(- k, k + 1):
                            image_h = h + dh
                            image_w = w + dw
                            if 0 <= image_h < H and 0 <= image_w < W:
                                weight_h = k + dh
                                weight_w = k + dw
                                for ci in range(CI):
                                    res[n, co, h, w] += image[n, ci, image_h, image_w] * \
                                                        weight[co, ci, weight_h, weight_w]
    return res


# @jit(target="cpu", nopython=True)
def raw_fc(x, w):
    asrt_int(x)
    asrt_int(w)
    return np.matmul(x, w)


@jit(target="cpu", nopython=True)
def raw_quantize(x, bias, shift, n_bit, signed):
    asrt_int(x)
    asrt_int(bias)
    asrt_int(shift)
    res = np.floor((x - bias) / (2 ** shift))
    if signed:
        upper_bound = 2 ** (n_bit - 1) - 1
        lower_bound = -upper_bound
    else:
        lower_bound = 0
        upper_bound = 2 ** n_bit - 1

    # return np.clip(res, -bound, bound)
    res = np.maximum(res, lower_bound)
    res = np.minimum(res, upper_bound)
    return res


def raw_inference(x, net):
    for layer in net.conv:
        if isinstance(layer, QuantizeConv):
            x = raw_conv(x, layer.w.numpy())
        elif isinstance(layer, QuantizeActSBN):
            x = raw_quantize(x, layer.bias.numpy(), layer.shift.numpy(), layer.n_bits, signed=layer.signed)
        elif isinstance(layer, ReLU):
            x = raw_relu(x)
        elif isinstance(layer, MaxPool2d):
            x = raw_pool(x, (2, 2))
    x = x.reshape((x.shape[0], -1))
    for layer in net.fc:
        if isinstance(layer, QuantizeFc):
            x = raw_fc(x, layer.w.numpy())
        elif isinstance(layer, QuantizeActSBN):
            x = raw_quantize(x, layer.bias.numpy(), layer.shift.numpy(), layer.n_bits, signed=layer.signed)
        elif isinstance(layer, ReLU):
            x = raw_relu(x)
    return x


def draw_images(images, s):
    for i in range(images.shape[0]):
        image = images[0]
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:  # only 1 channel
            image = np.concatenate([image] * 3, axis=2)
        plt.imshow(image / 255)
        plt.title(s)
        plt.show()


def raw_replay(net, n):
    """
    replay a model in raw mode.
    with n image.
    """
    d = json.loads(open("config.json").read())
    train_loader, test_loader = get_loader(d["DATASET"], 1, 0)

    for t, (image, label) in enumerate(test_loader):
        if t == n:
            break
        image = image.cpu().numpy()
        res = raw_inference(image, net)[0].argmax().item()
        label = label[0].item()

        draw_images(image, f"{label}:{res}")


def save_raw_parameter(net, file_path):
    d = json.loads(open("config.json").read())
    train_loader, test_loader = get_loader(d["DATASET"], 1, 0)
    x = next(iter(train_loader))[0].numpy()
    with open(file_path, "w") as f:
        for layer in net.conv:
            if isinstance(layer, QuantizeConv):
                weight = layer.w.numpy()
                _, CI, H, W = x.shape  # N is deserted
                CO, *_ = weight.shape
                f.write("CONV ")
                f.write(f"CO {CO} CI {CI} H {H} W {W}\n")
                for co in range(CO):
                    for ci in range(CI):
                        for h in range(3):
                            for w in range(3):
                                f.write(f"{round(weight[co, ci, h, w].item())} ")
                f.write("\n\n")

                x = raw_conv(x, weight)

            elif isinstance(layer, QuantizeActSBN):
                bias = layer.bias.numpy()
                shift = layer.shift.numpy()  # here shift is always negative, so change it to positive

                f.write("SIGNEDQUAN " if layer.signed else "UNSIGNEDQUAN ")
                _, C, H, W = x.shape
                f.write(f"C {C} H {H} W {W}\n")

                f.write("BIAS ")
                for c in range(C):
                    f.write(f"{round(bias[0, c, 0, 0].item())} ")
                f.write("\n")
                f.write("SHIFT ")
                for c in range(C):
                    f.write(f"{round(shift[0, c, 0, 0].item())} ")
                f.write("\n\n")

                x = raw_quantize(x, layer.bias.numpy(), layer.shift.numpy(), layer.n_bits, signed=layer.signed)

            elif isinstance(layer, ReLU):
                _, C, H, W = x.shape
                f.write("RELU ")
                f.write(f"C {C} H {H} W {W}\n\n")

                x = raw_relu(x)

            elif isinstance(layer, MaxPool2d):
                _, C, H, W = x.shape
                f.write("POOL ")
                f.write(f"C {C} H {H} W {W}\n\n")

                x = raw_pool(x, (2, 2))

        x = x.reshape((x.shape[0], -1))
        for layer in net.fc:
            if isinstance(layer, QuantizeFc):
                w = layer.w.numpy()

                CI, CO = w.shape
                f.write("FC ")
                f.write(f"CI {CI} CO {CO}\n")

                for ci in range(CI):
                    for co in range(CO):
                        f.write(f"{round(w[ci, co].item())} ")
                f.write("\n\n")

                x = raw_fc(x, w)

            elif isinstance(layer, QuantizeActSBN):

                bias = layer.bias.numpy()
                shift = layer.shift.numpy()
                f.write("SIGNEDQUAN " if layer.signed else "UNSIGNEDQUAN ")
                _, C = x.shape
                f.write(f"C {C} H {1} W {1}\n")

                f.write("BIAS ")
                for c in range(C):
                    f.write(f"{round(bias[0, c].item())} ")
                f.write("\n")
                f.write("SHIFT ")
                for c in range(C):
                    f.write(f"{round(shift[0, c].item())} ")
                f.write("\n\n")

                x = raw_quantize(x, bias, shift, layer.n_bits, signed=layer.signed)

            elif isinstance(layer, ReLU):
                f.write("RELU ")
                f.write(f"C {x.shape[1]} H 1 W 1\n\n")

                x = raw_relu(x)
        return x


if __name__ == "__main__":
    net = torch.load("net.pth").cpu()
    net.eval()
    with torch.no_grad():
        save_raw_parameter(net, "model.txt")
        raw_replay(net, 10)
