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
    w = torch.nn.init.normal_(torch.zeros(size))
    return torch.clamp(w * 2 ** (n_bits - 3), *get_bounds(n_bits)).cuda()


class QuantizeWeight(nn.Module):
    """
    Fake quantization operation.
    Do rounding and clamping.
    Use STE trick to make gradient continuous.
    """

    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.bounds = get_bounds(n_bits)

    def forward(self, x):
        return STEClamp.apply(STERound.apply(x), *self.bounds)


class LimitAcc(nn.Module):
    """
    Limit activation accumulated values.
    Use saturate limitation(clamp).
    Assume that input is already quantized value.
    """

    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.bounds = get_bounds(n_bits)

    def forward(self, x):
        return STEClamp.apply(x, *self.bounds)


class QuantizeSignedAct(nn.Module):
    """
    First use batch normalization, then do quantization.
    The two steps are composed into one.
    Learnable parameters are added to make this quantization self-adaptive.
    Assume that activation obeys normal distribution with "mean" & "std",
    First minus "mean";
    Then divide "std", can be approximated as 2^round(log2(std));
    Then scale. (2 ^ (n_bits - 3) )
    So right shift bit is round(log2(std))-(n_bits-3)
    Learnable parameters are added to make this layer self-adaptive.
    """

    def __init__(self, n_bits, num_features, is_conv=True, decay=0.9, learnable=False):
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

        # These buffers are updated during training.
        self.register_buffer("running_mean", torch.zeros(self.buffer_size))
        self.register_buffer("running_std", torch.ones(self.buffer_size))

        # These buffers are saved into persistent model.
        # Running parameter and learnable parameter are composed into one.
        self.register_buffer("persistent_mean", torch.zeros(self.buffer_size))
        self.register_buffer("persistent_shift", torch.zeros(self.buffer_size))

        # Learnable parameters.
        self.mean_bias = torch.nn.Parameter(torch.zeros(self.buffer_size), requires_grad=learnable)
        self.shift_bias = torch.nn.Parameter(torch.zeros(self.buffer_size), requires_grad=learnable)

        self.bounds = get_bounds(self.n_bits)

        # A flag to determine whether the first time to forward.
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
        self.persistent_mean = torch.round(self.running_mean + self.mean_bias)
        self.persistent_shift = torch.round(
            torch.log2(self.running_std) - (self.n_bits - 3) + self.shift_bias)

    def forward(self, x):
        if self.training:
            cur_mean = x.mean(dim=self.reduce_dim, keepdim=True)
            cur_std = x.std(dim=self.reduce_dim, keepdim=True)
            self._update_buffer(cur_mean, cur_std)

            # The mean and shift composed by original mean/shift and learnable parameter.
            cur_mean_biased = STERound.apply(cur_mean + self.mean_bias)
            cur_shift_biased = STERound.apply(
                torch.log2(cur_std) - (self.n_bits - 3) + self.shift_bias)

            return STEClamp.apply(
                STEFloor.apply((x - cur_mean_biased) * 2 ** -cur_shift_biased), *self.bounds
            )
        else:
            return torch.clamp(
                torch.floor((x - self.persistent_mean) * 2 ** -self.persistent_shift), *self.bounds
            )


class QuantizeUnsignedAct(nn.Module):
    """
    Do quantization to unsigned activation.
    Use direct right shift to get MSB.
    Learnable parameters are added to make this layer self-adaptive.
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
                 kernel_size, stride=1, padding=0, quantize_act=None):
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
            self.act_quantize_op = QuantizeSignedAct(self.n_bits, self.out_channels, is_conv=True)
        elif quantize_act == "unsigned":
            self.act_quantize_op = QuantizeUnsignedAct(in_bits=16, out_bits=self.n_bits, num_features=self.out_channels,
                                                       is_conv=True)
        else:
            assert (quantize_act is None)
            self.act_quantize_op = None

        # Get specially generated weight that ready to be quantized.
        # This weight is not integer. It is quantized before forward propagating.
        self.w = torch.nn.Parameter(get_normal_weight((out_channels, in_channels, kernel_size, kernel_size), n_bits))

    def forward(self, x):
        conv_res = torch.nn.functional.conv2d(
            x, self.weight_quantize_op(self.w), bias=None, stride=self.stride, padding=self.padding)
        return self.act_quantize_op(conv_res) if self.act_quantize_op else conv_res

    def __str__(self):
        return super().__str__() + "_quan:%d_in:%d_out:%d" % (self.n_bits, self.in_channels, self.out_channels)


class QuanFc(nn.Module):
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
            self.act_quantize_op = QuantizeSignedAct(self.n_bits, self.out_features, is_conv=False)
        elif quantize_act == "unsigned":
            self.act_quantize_op = QuantizeUnsignedAct(in_bits=16, out_bits=self.n_bits, num_features=self.out_features,
                                                       is_conv=False)
        else:
            assert (quantize_act is None)
            self.act_quantize_op = None

        # Get specially generated weight that ready to be quantized.
        # This weight is not integer. It is quantized before forward propagating.
        self.w = torch.nn.Parameter(get_normal_weight((in_features, out_features), n_bits))

    def forward(self, x):
        mat_res = torch.matmul(x, self.weight_quantize_op(self.w))
        return self.act_quantize_op(mat_res) if self.act_quantize_op else mat_res

    def __str__(self):
        return super().__str__() + "_quan:%d_in:%d_out:%d" % (self.n_bits, self.in_features, self.out_features)
