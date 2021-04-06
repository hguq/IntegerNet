import torch
import numpy as np
from model.mine import IntegerCifar10Net
from dataset import get_loader
from matplotlib import pyplot as plt


def validate_quantized(tensor, bits):
    upper_bound = 2 ** (bits - 1) - 1
    lower_bound = -upper_bound
    rounded_tensor = torch.clamp(torch.round(tensor), lower_bound, upper_bound)
    diff = torch.sum(torch.abs(rounded_tensor - tensor) ** 2)


def validate_quan(tensor, bits):
    upper_bound = 2 ** (bits - 1) - 1
    lower_bound = -upper_bound
    rounded_tensor = torch.clamp(torch.round(tensor), lower_bound, upper_bound)
    diff = torch.sum(torch.abs(rounded_tensor - tensor))
    if diff != 0:
        print("Difference of tensor: %f" % (diff.item()))


def validate_integer(tensor):
    rounded_tensor = torch.round(tensor)
    diff = torch.sum(torch.abs(rounded_tensor - tensor))
    if diff != 0:
        print("Difference of tensor %s: %f" % (tensor, diff.item()))


net = IntegerCifar10Net(img_size=(32, 32, 3), num_classes=10, n_bits=4)
net.requires_grad_(False)
net.load_state_dict(torch.load("old_save/cifar10bit4_86.84%.pth"))
net.eval()

train_loader, test_loader = get_loader("cifar10", 256)

batch_data, batch_label = next(test_loader.__iter__())
one_data = batch_data[0]
one_label = batch_label[0]

qc0, relu0, qc1, relu1, mp0, qc2, relu2, qc3, relu3, mp1, qc4, relu4, qc5, relu5, mp2 = net.conv
fc0, relu6, fc1 = net.fc

plt.hist(batch_data.reshape(-1), bins=np.linspace(0, 256, 257))
plt.text(0.5, 0.1, "input data")
plt.show()

bins = np.linspace(-7, 8, 16)
print(bins)


def validate_conv(layer, data):
    validate_quan(data, 4)
    w = layer.weight_quantize_op(layer.w)
    validate_quan(w, 4)
    plt.hist(w.detach().reshape(-1), bins=bins)
    plt.text(0.5, 0.9, str(layer) + "_w")
    plt.show()
    res = torch.nn.functional.conv2d(data, w, None, 1, 1, 1, 1)
    validate_integer(res)
    plt.hist(res.detach().numpy().reshape(-1), bins=15)
    plt.text(0.5, 0.9, str(layer) + "_act_max:%d" % res.max())
    plt.show()
    res = layer.act_quantize_op(res)
    validate_quan(res, 4)
    plt.hist(res.detach().numpy().reshape(-1), bins=bins)
    plt.text(0.5, 0.9, str(layer) + "_quan")
    plt.show()
    return res


def validate_fc(layer, data):
    validate_quan(data, 4)
    w = layer.weight_quantize_op(layer.w)
    validate_quan(w, 4)
    plt.hist(w.detach().reshape(-1), bins=bins)
    plt.text(0.5, 0.9, str(layer) + "_w")
    plt.show()
    res = torch.matmul(data, w)
    validate_integer(res)
    plt.hist(res.detach().numpy().reshape(-1), bins=15)
    plt.text(0.5, 0.9, str(layer) + "_act_max:%d" % res.max())
    plt.show()
    res = layer.act_quantize_op(res)
    validate_quan(res, 4)
    plt.hist(res.detach().numpy().reshape(-1), bins=bins)
    plt.text(0.5, 0.9, str(layer) + "_quan")
    plt.show()
    return res


out = validate_conv(qc0, batch_data)
out = relu0(out)
out = validate_conv(qc1, out)
out = relu1(out)
out = mp0(out)
out = validate_conv(qc2, out)
out = relu2(out)
out = validate_conv(qc3, out)
out = relu3(out)
out = mp1(out)
out = validate_conv(qc4, out)
out = relu4(out)
out = validate_conv(qc5, out)
out = relu5(out)
out = mp2(out)

out = out.reshape(out.size(0), -1)

out = validate_fc(fc0, out)
out = relu6(out)
out = validate_fc(fc1, out)

res = torch.argmax(out, dim=1)

total_correct = torch.eq(res, batch_label).sum().item()
rate = total_correct / batch_label.numel()
print("Accuracy: %2.2f%%" % (rate * 100))
