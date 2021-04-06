from abc import ABC
from torch import nn
from layer.function import *
from layer.misc import get_bounds


def get_quantized_weight(size, n_bits):
    """
    Produce quantized weight.
    Assume that weight obeys standard normal distribution.
    Scale it to exploit n bits integer representing ability.
    Assume that normal distribution data spans from -4 to 4.
    So scale factor is 2 ^ n_bits / 8 = 2 ^ (n_bits - 3)
    """
    w = torch.zeros(size)
    torch.nn.init.normal_(w)
    lower_bound, upper_bound = get_bounds(n_bits)
    return torch.clamp(torch.round(w * 2 ** (n_bits - 3)), lower_bound, upper_bound).cuda()


class QuantizeWeight(nn.Module):
    """
    Fake quantization operation.
    Do rounding and clamping.
    Use STE trick to make gradient continuous.
    """

    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.lower_bound, self.upper_bound = get_bounds(n_bits)

    def forward(self, x):
        return STEClamp.apply(STERound.apply(x), self.lower_bound, self.upper_bound)


class LimitAcc(nn.Module):
    """
    Limit activation accumulated values.
    Use saturate limitation.(clamp)
    """

    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.lower_bound, self.upper_bound = get_bounds(n_bits)

    def forward(self, x):
        return STEClamp.apply(x, self.lower_bound, self.upper_bound)


class QuantizeSignedAct(nn.Module):
    """
    First use batch normalization, then do quantization.
    The two steps are composed into one.
    Learnable parameters are added make this quantization self-adaptive.
    Assume that activation obeys normal distribution with "mean" & "std",
    First minus "mean",
    Then divide "std" ( can be approximated as 2^round(log2(std)) )
    Then scale. (2 ^ (n_bits - 3) )
    So right shift bit is round(log2(std))-(n_bits-3)
    """

    def __init__(self, n_bits, num_features, is_conv=True, decay=0.9):
        super().__init__()

        self.n_bits = n_bits
        self.num_features = num_features
        self.is_conv = is_conv
        self.decay = decay

        if is_conv:
            self.reduce_dim = (0, 2, 3)
            self.buffer_size = (1, self.num_features, 1, 1)
        else:
            self.reduce_dim = (0,)
            self.buffer_size = (1, self.num_features)

        self.register_buffer("running_mean", torch.zeros(self.buffer_size))
        self.register_buffer("running_std", torch.ones(self.buffer_size))

        self.register_buffer("persistent_mean", torch.zeros(self.buffer_size))
        self.register_buffer("persistent_shift", torch.zeros(self.buffer_size))

        self.mean_bias = torch.nn.Parameter(torch.zeros(self.buffer_size))
        self.shift_bias = torch.nn.Parameter(torch.zeros(self.buffer_size))

        self.lower_bound, self.upper_bound = get_bounds(n_bits)

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
        self.persistent_mean = torch.round(self.running_mean) + torch.round(self.mean_bias)
        # self.persistent_shift = self.n_bits - 3 + torch.round(-torch.log2(self.running_std)) +
        # torch.round(self.shift_bias)
        self.persistent_shift = torch.round(torch.log2(self.running_std)) - (self.n_bits - 3)

    def forward(self, x):
        if self.training:
            cur_mean = x.mean(dim=self.reduce_dim, keepdim=True)
            cur_std = x.std(dim=self.reduce_dim, keepdim=True)
            self._update_buffer(cur_mean, cur_std)

            # cur_mean_round = STERound.apply(cur_mean)
            # cur_shift = self.n_bits - 3 + STERound.apply(-torch.log2(cur_std))

            cur_mean_biased = STERound.apply(cur_mean) + STERound.apply(self.mean_bias)
            # cur_shift_biased = self.n_bits - 3 + STERound.apply(-torch.log2(cur_std)) +
            # STERound.apply(self.shift_bias)
            cur_shift_biased = STERound.apply(torch.log2(cur_std)) - (self.n_bits - 3) + STERound.apply(self.shift_bias)

            return STEClamp.apply(
                STEFloor.apply((x - cur_mean_biased) * 2 ** -cur_shift_biased),
                self.lower_bound,
                self.upper_bound
            )
        else:
            return torch.clamp(
                torch.floor((x - self.persistent_mean) * 2 ** -self.persistent_shift),
                self.lower_bound,
                self.upper_bound
            )


class QuantizeUnsignedAct(nn.Module):
    """
    Do quantization to unsigned activation.
    """

    def __init__(self, in_bits, out_bits, num_features, is_conv=True, decay=0.9):
        super().__init__()

        self.in_bits = in_bits
        self.out_bits = out_bits
        self.num_features = num_features
        self.is_conv = is_conv
        self.decay = decay

        if is_conv:
            self.reduce_dim = (0, 2, 3)
            self.buffer_size = (1, self.num_features, 1, 1)
        else:
            self.reduce_dim = (0,)
            self.buffer_size = (1, self.num_features)

        self.running_shift = torch.nn.Parameter(
            torch.full(self.buffer_size, self.in_bits - self.out_bits, dtype=torch.float))
        self.register_buffer("persistent_shift", torch.zeros(self.buffer_size))

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._prepare_eval()

    def _prepare_eval(self):
        self.persistent_shift = torch.round(self.running_shift)

    def forward(self, x):
        if self.training:
            return x * 2 ** -STERound.apply(self.running_shift)
        else:
            return x * 2 ** -self.persistent_shift


class QuanConv(nn.Module, ABC):
    """
    使用量化的权重，并且在卷积后自动将feature map进行量化
    """

    def __init__(self, n_bits, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, quantize_act=False):
        super().__init__()

        self.n_bits = n_bits
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        self.w_quan_op = QuantizeWeight(self.n_bits)
        self.a_quan_op = QuantizeSignedAct(self.n_bits, self.out_channels, is_conv=True) if quantize_act else None

        # self.w = torch.nn.Parameter(
        #     torch.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float))
        self.w = torch.nn.Parameter(get_quantized_weight((out_channels, in_channels, kernel_size, kernel_size), n_bits))

    def forward(self, x):
        conv_res = torch.nn.functional.conv2d(
            x, self.w_quan_op(self.w), bias=None, stride=self.stride, padding=self.padding)
        return self.a_quan_op(conv_res) if self.a_quan_op else conv_res

    def __str__(self):
        return super().__str__() + "_quan:%d_in:%d_out:%d" % (self.n_bits, self.in_channels, self.out_channels)


class QuanFc(nn.Module):
    # Do quantization as Linear layer
    def __init__(self, n_bits, in_features, out_features, quantize_act=False):
        super().__init__()
        self.n_bits = n_bits
        self.in_features = in_features
        self.out_features = out_features

        self.w_quan_op = QuantizeWeight(self.n_bits)
        self.a_quan_op = QuantizeSignedAct(self.n_bits, self.out_features, is_conv=False) if quantize_act else None

        # self.w = torch.nn.Parameter(
        #    torch.zeros((in_features, out_features), dtype=torch.float))
        self.w = torch.nn.Parameter(get_quantized_weight((in_features, out_features), n_bits))

    def forward(self, x):
        mat_res = torch.matmul(x, self.w_quan_op(self.w))
        return self.a_quan_op(mat_res) if self.a_quan_op else mat_res

    def __str__(self):
        return super().__str__() + "_quan:%d_in:%d_out:%d" % (self.n_bits, self.in_features, self.out_features)
