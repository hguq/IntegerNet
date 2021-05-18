import torch


def RightShift(x, s):
    """
    A class to implement right shift operation.
    Make sure that input shift bits is integer!
    """
    return torch.floor(x / 2 ** s)


class STEClamp(torch.autograd.Function):
    """
    A class to implement STE clamp function.
    Override backward gradient with 1.
    """

    @staticmethod
    def forward(ctx, x, min_val, max_val):
        return torch.clamp(x, min_val, max_val)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


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
