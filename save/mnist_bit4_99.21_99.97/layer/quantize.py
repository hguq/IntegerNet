from torch import nn
from layer.function import *
from layer.misc import get_bounds
from torch.nn import functional


def get_normal_weight(size: tuple, n_bits: int) -> torch.Tensor:
    """
    Generate normal weight.
    Assume that weight obeys standard normal distribution.
    First initialize the weight to standard normal distribution.
    Then scale it to exploit n-bits-integer's representing ability.
    Assume that normal distribution data spans from -4 to 4.
    So scale factor is 2 ^ n_bits / 8 = 2 ^ (n_bits - 3)
    This function has been verified.
    """
    w = torch.nn.init.normal_(torch.zeros(size))
    return torch.clamp(w * 2 ** (n_bits - 3), *get_bounds(n_bits, signed=True))


class QuantizeWeight(nn.Module):
    """
    Weight quantization operation.
    Do rounding and clamping.
    Use STE trick to make gradient continuous.
    Weight is always signed.
    """

    def __init__(self, n_bits: int):
        super().__init__()
        self.n_bits = n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(STERound.apply(x), *get_bounds(self.n_bits, signed=True))


class LimitAcc(nn.Module):
    """
    Limit activation accumulated values.
    Use saturate limitation(clamp).
    Assume that input is already quantized value.
    """

    def __init__(self, n_bits: int, signed: bool):
        super().__init__()
        self.n_bits = n_bits
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, *get_bounds(self.n_bits, self.signed))


class RecordMeanStd(nn.Module):
    """
    An abstract class that record shift and bias
    Bias and Shift will be float
    """

    def __init__(self, num_features: int, is_conv: bool = True, decay: float = 0.9):
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
        self.register_buffer("running_std", torch.zeros(self.buffer_size), persistent=True)

        self.hot = False

    def _update_buffer(self, cur_mean: torch.Tensor, cur_std: torch.Tensor):
        with torch.no_grad():
            if self.hot:
                self.running_mean = self.running_mean * self.decay + cur_mean * (1 - self.decay)
                self.running_std = self.running_std * self.decay + cur_std * (1 - self.decay)
            else:
                self.running_mean.copy_(cur_mean)
                self.running_std.copy_(cur_std)
                self.hot = True

    def forward(self, x: torch.Tensor) -> tuple:
        if self.training:
            cur_mean = x.mean(dim=self.reduce_dim, keepdim=True)
            cur_std = x.std(dim=self.reduce_dim, keepdim=True)
            self._update_buffer(cur_mean, cur_std)
            return cur_mean, cur_std
        else:
            return None, None


class QuantizeActSBN(RecordMeanStd):
    """
    Use Shift based Batch Normalization to do quantize.
    First scale it to normal distribution, then scale it to
    fully exploit n bit presentation ability.

    First (x-bias)/std to get normal distribution
    Then multiply quantize_range/normal_range to fully use quantized scale
    quantize_range is 2^n_bits
    normal_range is 8 (signed) or 4 (unsigned)
    so (x-bias)/(std*normal_range/quantize_range)
    use shift to replicate dividing
               divide (std*normal_range/quantize_range)
    equals to: right shift log2(std*normal_range/quantize_range)=log2(std)+log2(signed?8:4)-n_bits
    approx:    right shift round(log2(std))+log2(signed?8:4)-n_bits
    """

    def __init__(self, n_bits: int, signed: bool, num_features: int, is_conv=True, decay=0.9):
        super().__init__(num_features=num_features, is_conv=is_conv, decay=decay)
        self.n_bits = n_bits
        self.signed = signed

        # persistent bias and shift
        self.register_buffer("bias", torch.zeros(self.buffer_size), persistent=True)
        self.register_buffer("shift", torch.zeros(self.buffer_size), persistent=True)

        self.hot = False

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._prepare_eval()

    def _prepare_eval(self):
        self.bias = torch.round(self.running_mean)
        if self.signed:
            self.shift = torch.round(torch.log2(self.running_std)) + 3 - self.n_bits
        else:
            self.shift = torch.round(torch.log2(self.running_std)) + 2 - self.n_bits

    def forward(self, x):
        if self.training:
            cur_mean, cur_std = super().forward(x)
            cur_bias = STERound.apply(cur_mean)
            if self.signed:
                cur_shift = STERound.apply(torch.log2(cur_std)) + 3 - self.n_bits
            else:
                cur_shift = STERound.apply(torch.log2(cur_std)) + 2 - self.n_bits

            return torch.clamp(
                STERightShift(x - cur_bias, cur_shift),
                *get_bounds(self.n_bits, signed=self.signed)
            )
        else:
            return torch.clamp(
                RightShift(x - self.bias, self.shift),
                *get_bounds(self.n_bits, signed=self.signed)
            )


class QuantizeConv(nn.Module):
    """
    Use quantized weights and quantized activations, do quantized convolution.
    """

    def __init__(self, n_bits, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.n_bits = n_bits
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        # Operator that quantize weight.
        self.weight_quantize_op = QuantizeWeight(self.n_bits)

        # create float shadow w and real integer w.
        weight_size = (out_channels, in_channels, kernel_size, kernel_size)
        self.shadow_w = torch.nn.Parameter(get_normal_weight(weight_size, n_bits))
        self.register_buffer("w", torch.zeros(weight_size), persistent=True)

    def forward(self, x):
        w = self.weight_quantize_op(self.shadow_w) if self.training else self.w
        return functional.conv2d(x, w, bias=None, stride=self.stride, padding=self.padding)

    def _prepare_eval(self):
        self.w = self.weight_quantize_op(self.shadow_w)

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._prepare_eval()


class QuantizeFc(nn.Module):
    """
    Implementation of quantized fully connected layer
    """

    def __init__(self, n_bits, in_features, out_features):
        super().__init__()
        self.n_bits = n_bits
        self.in_features = in_features
        self.out_features = out_features

        # Operator that quantize weight.
        self.weight_quantize_op = QuantizeWeight(self.n_bits)

        # This weight is not integer. It is quantized before forward propagating.
        weight_size = (in_features, out_features)
        self.shadow_w = torch.nn.Parameter(get_normal_weight(weight_size, n_bits))
        self.register_buffer("w", torch.zeros(weight_size), persistent=True)

    def forward(self, x):
        w = self.weight_quantize_op(self.shadow_w) if self.training else self.w
        return torch.matmul(x, w)

    def _prepare_eval(self):
        self.w = self.weight_quantize_op(self.shadow_w)

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._prepare_eval()
