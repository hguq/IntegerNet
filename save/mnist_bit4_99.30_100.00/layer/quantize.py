from abc import ABC
from torch import nn
from layer.function import *
from layer.misc import get_bounds


def get_normal_weight(size, n_bits):
    """
    Generate normal weight.
    Assume that weight obeys standard normal distribution.
    First initialize the weight to standard normal distribution.
    Then scale it to exploit n-bits-integer's representing ability.
    Assume that normal distribution data spans from -4 to 4.
    So scale factor is 2 ^ n_bits / 8 = 2 ^ (n_bits - 3)
    This function has been verified.
    """
    w = torch.nn.init.normal_(torch.zeros(size)).cuda()
    return torch.clamp(w * 2 ** (n_bits - 3), *get_bounds(n_bits))


class QuantizeWeight(nn.Module):
    """
    Weight quantization operation.
    Do rounding and clamping.
    Use STE trick to make gradient continuous.
    """

    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits

    def forward(self, x):
        # return STEClamp.apply(STERound.apply(x), *get_bounds(self.n_bits, signed=True))
        return torch.clamp(STERound.apply(x), *get_bounds(self.n_bits, signed=True))


class LimitAcc(nn.Module):
    """
    Limit activation accumulated values.
    Use saturate limitation(clamp).
    Assume that input is already quantized value.
    """

    def __init__(self, n_bits, signed=True):
        super().__init__()
        self.n_bits = n_bits
        self.signed = signed

    def forward(self, x):
        # return STEClamp.apply(x, *get_bounds(self.n_bits, self.signed))
        return torch.clamp(x, *get_bounds(self.n_bits, self.signed))


class QuantizeSignedActSBN(nn.Module):
    def __init__(self, n_bits, num_features, is_conv=True, decay=0.9, adaptive=False):
        super().__init__()
        self.n_bits = n_bits
        self.num_features = num_features
        self.is_conv = is_conv
        self.decay = decay
        self.adaptive = adaptive

        if is_conv:
            self.reduce_dim = (0, 2, 3)
            self.buffer_size = (1, self.num_features, 1, 1)
        else:
            self.reduce_dim = (0,)
            self.buffer_size = (1, self.num_features)

        if self.adaptive:
            self.gamma = nn.Parameter(torch.zeros(self.buffer_size))  # mean
            self.beta = nn.Parameter(torch.zeros(self.buffer_size))  # shift

        self.register_buffer("running_mean", torch.zeros(self.buffer_size), persistent=True)
        self.register_buffer("running_std", torch.zeros(self.buffer_size), persistent=True)

        self.register_buffer("bias", torch.zeros(self.buffer_size), persistent=True)
        self.register_buffer("shift", torch.zeros(self.buffer_size), persistent=True)

        self.hot = False

    def _update_buffer(self, cur_mean, cur_std):
        with torch.no_grad():
            if self.hot:
                self.running_mean = self.running_mean * self.decay + cur_mean * (1 - self.decay)
                self.running_std = self.running_std * self.decay + cur_std * (1 - self.decay)
            else:
                self.running_mean.copy_(cur_mean)
                self.running_std.copy_(cur_std)
                self.hot = True

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._prepare_eval()

    def _prepare_eval(self):
        if self.adaptive:
            self.bias = torch.round(self.running_mean + self.gamma)
            self.shift = self.n_bits - 3 + torch.round(-torch.log2(self.running_std) + self.beta)
        else:
            self.bias = torch.round(self.running_mean)
            self.shift = self.n_bits - 3 + torch.round(-torch.log2(self.running_std))

    def forward(self, x):
        if self.training:
            cur_mean, cur_std = x.mean(dim=self.reduce_dim, keepdim=True), x.std(dim=self.reduce_dim, keepdim=True)
            self._update_buffer(cur_mean, cur_std)

            if self.adaptive:
                cur_bias = STERound.apply(cur_mean + self.gamma)
                cur_shift = self.n_bits - 3 + STERound.apply(-torch.log2(cur_std) + self.beta)
            else:
                cur_bias = STERound.apply(cur_mean)
                cur_shift = self.n_bits - 3 + STERound.apply(-torch.log2(cur_std))

            # return STEClamp.apply(
            return torch.clamp(
                STEFloor.apply((x - cur_bias) * 2 ** cur_shift),
                *get_bounds(self.n_bits, signed=True)
            )
        else:
            return torch.clamp(
                torch.floor((x - self.bias) * 2 ** self.shift),
                *get_bounds(self.n_bits, signed=True)
            )


class ReLUQuantizeUnsignedActSBN(nn.Module):
    # This function will do relu and quantize.
    def __init__(self, n_bits, num_features, is_conv=True, decay=0.9, adaptive=False):
        super().__init__()
        self.n_bits = n_bits
        self.num_features = num_features
        self.is_conv = is_conv
        self.decay = decay
        self.adaptive = adaptive

        if is_conv:
            self.reduce_dim = (0, 2, 3)
            self.buffer_size = (1, self.num_features, 1, 1)
        else:
            self.reduce_dim = (0,)
            self.buffer_size = (1, self.num_features)

        if self.adaptive:
            self.gamma = nn.Parameter(torch.zeros(self.buffer_size))  # mean
            self.beta = nn.Parameter(torch.zeros(self.buffer_size))  # shift

        self.register_buffer("running_mean", torch.zeros(self.buffer_size), persistent=True)
        self.register_buffer("running_std", torch.zeros(self.buffer_size), persistent=True)

        self.register_buffer("bias", torch.zeros(self.buffer_size), persistent=True)
        self.register_buffer("shift", torch.zeros(self.buffer_size), persistent=True)

        self.relu = nn.ReLU()
        self.hot = False

    def _update_buffer(self, cur_mean, cur_std):
        with torch.no_grad():
            if self.hot:
                self.running_mean = self.running_mean * self.decay + cur_mean * (1 - self.decay)
                self.running_std = self.running_std * self.decay + cur_std * (1 - self.decay)
            else:
                self.running_mean.copy_(cur_mean)
                self.running_std.copy_(cur_std)
                self.hot = True

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._prepare_eval()

    def _prepare_eval(self):
        if self.adaptive:
            self.bias = torch.round(self.running_mean + self.gamma)
            self.shift = self.n_bits - 2 + torch.round(-torch.log2(self.running_std) + self.beta)
        else:
            self.bias = torch.round(self.running_mean)
            self.shift = self.n_bits - 2 + torch.round(-torch.log2(self.running_std))

    def forward(self, x):
        if self.training:
            cur_mean, cur_std = x.mean(dim=self.reduce_dim, keepdim=True), x.std(dim=self.reduce_dim, keepdim=True)
            self._update_buffer(cur_mean, cur_std)

            if self.adaptive:
                cur_bias = STERound.apply(cur_mean + self.gamma)
                cur_shift = self.n_bits - 2 + STERound.apply(-torch.log2(cur_std) + self.beta)
            else:
                cur_bias = STERound.apply(cur_mean)
                cur_shift = self.n_bits - 2 + STERound.apply(-torch.log2(cur_std))

            # return STEClamp.apply(
            return torch.clamp(
                STEFloor.apply((x - cur_bias) * 2 ** cur_shift),
                *get_bounds(self.n_bits, signed=False)
            )
        else:
            return torch.clamp(
                torch.floor((x - self.bias) * 2 ** self.shift),
                *get_bounds(self.n_bits, signed=False)
            )


class QuantizeSignedActPercentage(nn.Module):
    """
    Use dynamic binary search to determine the quantization granularity.
    """

    def __init__(self, n_bits, num_features, is_conv=True, decay=0.9, percentage=(0.01, 0.02), adaptive=False):
        super().__init__()
        self.n_bits = n_bits
        self.num_features = num_features
        self.is_conv = is_conv
        self.decay = decay
        self.percentage = percentage
        self.adaptive = adaptive

        if is_conv:
            self.reduce_dim = (0, 2, 3)
            self.buffer_size = (1, self.num_features, 1, 1)
        else:
            self.reduce_dim = (0,)
            self.buffer_size = (1, self.num_features)

        # These buffers are learnable parameters
        if self.adaptive:
            self.gamma = nn.Parameter(torch.zeros(self.buffer_size))
            self.beta = nn.Parameter(torch.zeros(self.buffer_size))
        # These buffers are updated during training.
        self.register_buffer("running_bias", torch.zeros(self.buffer_size))
        self.register_buffer("running_shift", torch.ones(self.buffer_size))

        # These buffers are saved into persistent model.
        self.register_buffer("bias", torch.zeros(self.buffer_size))
        self.register_buffer("shift", torch.zeros(self.buffer_size))

        # A flag to determine whether the first time to forward.
        self.hot = False

    def _get_percent_bounds(self, x):
        # This is a function to get the 1~2% boundaries of a tensor (convolutional type, 4D)
        size = 1
        for dim in self.reduce_dim:
            size *= x.shape[dim]

        min_val, max_val = x, x
        for dim in self.reduce_dim:
            min_val = min_val.min(dim=dim, keepdim=True)[0]
            max_val = max_val.max(dim=dim, keepdim=True)[0]

        # min_val, max_val = min_val.detach(), max_val.detach()

        # Solve upper bound
        left, right = min_val, max_val
        while True:
            upper_pivot = (left + right) / 2
            per = (x > upper_pivot).double().sum(dim=self.reduce_dim, keepdim=True) / size
            if torch.logical_or(
                    torch.logical_and(self.percentage[0] <= per, per <= self.percentage[1]),
                    right - left <= 1).all():
                break
            left = torch.where(per > self.percentage[1], upper_pivot, left)
            right = torch.where(per < self.percentage[0], upper_pivot, right)

        # Solve lower bound
        left, right = min_val, max_val
        while True:
            lower_pivot = (left + right) / 2
            per = (x < lower_pivot).double().sum(dim=self.reduce_dim, keepdim=True) / size
            if torch.logical_or(
                    torch.logical_and(self.percentage[0] <= per, per <= self.percentage[1]),
                    right - left <= 1).all():
                break
            left = torch.where(per < self.percentage[0], lower_pivot, left)
            right = torch.where(per > self.percentage[1], lower_pivot, right)

        return lower_pivot, upper_pivot

    def _get_bias_shift(self, x):
        lower, upper = self._get_percent_bounds(x)
        # Target range is 2^n_bits, current range is (upper - lower)
        # So shift = log2(2^n_bits/(upper-lower))=n_bits-log2(upper-lower)
        # Attention! shift is left shift.
        return (lower + upper) / 2, self.n_bits - torch.log2(upper - lower)

    def _update_buffer(self, cur_bias, cur_shift):
        with torch.no_grad():
            if self.hot:
                self.running_bias = self.running_bias * self.decay + cur_bias * (1 - self.decay)
                self.running_shift = self.running_shift * self.decay + cur_shift * (1 - self.decay)
            else:
                self.running_bias.copy_(cur_bias)
                self.running_shift.copy_(cur_shift)
                self.hot = True

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._prepare_eval()

    def _prepare_eval(self):
        if self.adaptive:
            self.bias = torch.round(self.running_bias + self.gamma)
            self.shift = torch.round(self.running_shift + self.beta)
        else:
            self.bias = torch.round(self.running_bias)
            self.shift = torch.round(self.running_shift)

    def forward(self, x):
        if self.training:
            cur_bias, cur_shift = self._get_bias_shift(x)
            self._update_buffer(cur_bias, cur_shift)
            if self.adaptive:
                cur_bias += self.gamma
                cur_shift += self.beta
            # return STEClamp.apply(
            return torch.clamp(
                STEFloor.apply(
                    (x - STERound.apply(cur_bias)) * 2 ** STERound.apply(cur_shift)
                ),
                *get_bounds(self.n_bits, signed=True)
            )
        else:
            return torch.clamp(
                torch.floor(
                    (x - self.bias) * 2 ** self.shift),
                *get_bounds(self.n_bits, signed=True)
            )


class QuantizeUnsignedActPercentage(nn.Module):
    """
    Use dynamic binary search to determine the quantization granularity.
    """

    def __init__(self, n_bits, num_features, is_conv=True, decay=0.9, percentage=(0.01, 0.02), adaptive=False):
        super().__init__()
        self.n_bits = n_bits
        self.num_features = num_features
        self.is_conv = is_conv
        self.decay = decay
        self.percentage = percentage
        self.adaptive = adaptive

        if is_conv:
            self.reduce_dim = (0, 2, 3)
            self.buffer_size = (1, self.num_features, 1, 1)
        else:
            self.reduce_dim = (0,)
            self.buffer_size = (1, self.num_features)

        self.beta = nn.Parameter(torch.zeros(self.buffer_size))

        # These buffers are updated during training.
        self.register_buffer("running_shift", torch.ones(self.buffer_size))

        # These buffers are saved into persistent model.
        self.register_buffer("shift", torch.zeros(self.buffer_size), persistent=True)

        # A flag to determine whether the first time to forward.
        self.hot = False

    def _get_percent_bound(self, x):
        # This is a function to get the 1~2% boundaries of a tensor (convolutional type, 4D)
        size = 1
        for dim in self.reduce_dim:
            size *= x.shape[dim]

        min_val = torch.zeros(self.buffer_size).cuda()
        max_val = x
        for dim in self.reduce_dim:
            max_val = max_val.max(dim=dim, keepdim=True)[0]

        # min_val, max_val = min_val.detach(), max_val.detach()

        # Solve upper bound
        left, right = min_val, max_val
        while True:
            upper_pivot = (left + right) / 2
            per = (x > upper_pivot).double().sum(dim=self.reduce_dim, keepdim=True) / size
            if torch.logical_or(
                    torch.logical_and(self.percentage[0] <= per, per <= self.percentage[1]),
                    right - left <= 1).all():
                break
            left = torch.where(per > self.percentage[1], upper_pivot, left)
            right = torch.where(per < self.percentage[0], upper_pivot, right)

        return upper_pivot

    def _get_shift(self, x):
        upper = self._get_percent_bound(x)
        # Attention! shift is left shift.
        return self.n_bits - torch.log2(upper)

    def _update_buffer(self, cur_shift):
        with torch.no_grad():
            if self.hot:
                self.running_shift = self.running_shift * self.decay + cur_shift * (1 - self.decay)
            else:
                self.running_shift.copy_(cur_shift)
                self.hot = True

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._prepare_eval()

    def _prepare_eval(self):
        self.shift = torch.round(self.running_shift)

    def forward(self, x):
        if self.training:
            cur_shift = self._get_shift(x)
            self._update_buffer(cur_shift)
            if self.adaptive:
                cur_shift += self.beta
            # return STEClamp.apply(
            return torch.clamp(
                STEFloor.apply(
                    x * 2 ** STERound.apply(cur_shift)
                ),
                *get_bounds(self.n_bits, signed=False)
            )
        else:
            return torch.clamp(
                torch.floor(x * 2 ** self.shift),
                *get_bounds(self.n_bits, signed=False)
            )


class QuantizeConv(nn.Module):
    """
    Use quantized weights and quantized activations, do quantized convolution.
    """

    def __init__(self, n_bits, in_channels, out_channels, kernel_size, stride=1, padding=0, quantize_act=None):
        super().__init__()

        self.n_bits = n_bits
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        # Operator that quantize weight.
        self.weight_quantize_op = QuantizeWeight(self.n_bits)

        # Operator that quantize activation.
        if quantize_act == "signed":
            self.act_quantize_op = QuantizeSignedActPercentage(self.n_bits, self.out_channels, is_conv=True)
        else:
            assert quantize_act is None
            self.act_quantize_op = None

        # Get specially generated weight that ready to be quantized.
        # This weight is not integer. It is quantized before forward propagating.
        self.w = torch.nn.Parameter(
            get_normal_weight((out_channels, in_channels, kernel_size, kernel_size), n_bits))

    def forward(self, x):
        conv_res = torch.nn.functional.conv2d(
            x, self.weight_quantize_op(self.w), bias=None, stride=self.stride, padding=self.padding)
        return self.act_quantize_op(conv_res) if self.act_quantize_op else conv_res

    def train(self, mode=True):
        super().train(mode)
        # if not mode:
        # self.w = nn.Parameter(self.weight_quantize_op(self.w))


class QuantizeFc(nn.Module):
    # Do quantization as Linear layer
    def __init__(self, n_bits, in_features, out_features, quantize_act=None):
        super().__init__()
        self.n_bits = n_bits
        self.in_features = in_features
        self.out_features = out_features

        # Operator that quantize weight.
        self.weight_quantize_op = QuantizeWeight(self.n_bits)

        # Operator that quantize activation.
        if quantize_act == "signed":
            self.act_quantize_op = QuantizeSignedActPercentage(self.n_bits, self.out_features, is_conv=False)
        else:
            assert (quantize_act is None)
            self.act_quantize_op = None

        # Get specially generated weight that ready to be quantized.
        # This weight is not integer. It is quantized before forward propagating.
        self.w = torch.nn.Parameter(get_normal_weight((in_features, out_features), n_bits))

    def forward(self, x):
        mat_res = torch.matmul(x, self.weight_quantize_op(self.w))
        return self.act_quantize_op(mat_res) if self.act_quantize_op else mat_res

    def train(self, mode=True):
        super().train(mode)
        # if not mode:
        # self.w = nn.Parameter(self.weight_quantize_op(self.w))
