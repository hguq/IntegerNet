from layer.quantize import get_normal_weight
import torch

a = torch.ones(10)
b = a
b = b / 2
print(a)
