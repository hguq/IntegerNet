import torch


def RightShift(x, s):
    return torch.floor(x / (2 ** s))


def STERightShift(x, s):
    return STEFloor.apply(x / (2 ** s))


class STERound(torch.autograd.Function):
    """
    A class to implement STE round function.
    Override backward gradient with 1.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class STEFloor(torch.autograd.Function):
    """
    A class to implement STE round function.
    Override backward gradient with 1.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class STECeil(torch.autograd.Function):
    """
    A class to implement STE ceil function.
    Override backward gradient with 1.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.ceil(x)

    @staticmethod
    def backward(ctx, grad):
        return grad
