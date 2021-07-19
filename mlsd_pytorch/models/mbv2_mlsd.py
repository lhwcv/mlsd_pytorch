import torch
import timm
import torch.nn as nn
from mlsd_pytorch.models.layers import *
import torch.utils.model_zoo as model_zoo


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        padding = (kernel_size - 1) // 2

        # TFLite uses slightly different padding than PyTorch
        #         if stride == 2:
        #             padding = 0
        #         else:
        #             padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        # TFLite uses slightly different padding
        #         if self.stride == 2:
        #             x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        #             #print(x.shape)

        for module in self:
            if not isinstance(module, nn.MaxPool2d):
                x = module(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        width_mult = 1.0
        round_nearest = 8

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        # features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        # print('fea blocks:', len(features))
        self.features = nn.Sequential(*features)

        self.fpn_selected = [1, 3, 6, 10]
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if pretrained:
            self._load_pretrained_model()

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)

        c1, c2, c3, c4 = fpn_features
        return c1, c2, c3, c4

    def forward(self, x):
        return self._forward_impl(x)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, channels, stride, groups=1):
        """Set up the layer.

        Parameters
        ----------
        channels: int
            The number of input and output channels

        stride: int or tuple
            The amount of upsampling to do

        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)

        assert groups in (1, channels), "Must use no grouping, " + \
            "or one group per channel"

        kernel_size = (2 * stride[0] - 1, 2 * stride[1] - 1)
        padding = (stride[0] - 1, stride[1] - 1)
        super().__init__(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=padding,
            groups=groups)

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant(self.bias, 0)
        nn.init.constant(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.stride)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(stride):
        """Generate a bilinear upsampling kernel."""
        num_dims = len(stride)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        # The bilinear kernel is separable in its spatial dimensions
        # Build up the kernel channel by channel
        for channel in range(num_dims):
            channel_stride = stride[channel]
            kernel_size = 2 * channel_stride - 1
            # e.g. with stride = 4
            # delta = [-3, -2, -1, 0, 1, 2, 3]
            # channel_filter = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            delta = torch.arange(1 - channel_stride, channel_stride)
            channel_filter = (1 - torch.abs(delta / channel_stride))
            # Apply the channel filter to the current channel
            shape = [1] * num_dims
            shape[channel] = kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel


class MobileV2_MLSD(nn.Module):
    def __init__(self, cfg):
        super(MobileV2_MLSD, self).__init__()

        self.backbone = MobileNetV2(pretrained=True)

        self.block12 = BlockTypeA(in_c1=32, in_c2=64,
                                  out_c1=64, out_c2=64)
        self.block13 = BlockTypeB(128, 64)

        self.block14 = BlockTypeA(in_c1=24, in_c2=64,
                                  out_c1=32, out_c2=32)
        self.block15 = BlockTypeB(64, 64)

        self.block16 = BlockTypeC(64, 16)

        # self.block17 = nn.ConvTranspose2d(in_channels=16,
        #                                   out_channels=16,
        #                                   kernel_size=3,
        #                                   stride=2,
        #                                   padding=1,
        #                                   output_padding=1,
        #                                   bias=False)

        self.with_deconv = cfg.model.with_deconv

        if self.with_deconv:
            self.block17 = BilinearConvTranspose2d(16, 2, 1)
            self.block17.reset_parameters()


    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)

        # print(c1.shape)

        x = self.block12(c3, c4)
        x = self.block13(x)
        x = self.block14(c2, x)
        x = self.block15(x)
        x = self.block16(x)

        if self.with_deconv:
            x = self.block17(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)

        #x = x[:, 7:, :, :]
        #print(x.shape)
        return x


if __name__ == '__main__':
    from mlsd_pytorch.cfg.default import  get_cfg_defaults
    cfg = get_cfg_defaults()
    model = MobileV2_MLSD(cfg)
    x = torch.randn((1, 3, 512, 512))
    y = model(x)

    from thop import profile

    flops, params = profile(model, inputs=(x,))
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))
