import torch

for i in range(-10, 11):
    x = torch.tensor(i, dtype=torch.float)
    x.requires_grad = True
    print(x)

    y = torch.clamp(x, -5, 5)

    y.backward()

    print(x.grad)
