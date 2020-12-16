import torch
import torch.nn as nn


class ConvBlock2d(nn.Module):
    """Block of 1x1, 3x3, 1x1 convolutions with non linearities. Shape of input
    and output is the same.

    Args:
        in_channels (int): Number of channels in input.
        num_filters (list of ints): List of two ints with the number of filters
            for the first and second conv layers. Third conv layer must have the
            same number of input filters as there are channels.
        add_groupnorm (bool): If True adds GroupNorm.
    """
    def __init__(self, in_channels, num_filters, add_groupnorm=True):
        super(ConvBlock2d, self).__init__()
        if add_groupnorm:
            self.forward_layers = nn.Sequential(
                nn.GroupNorm(num_channels_to_num_groups(in_channels), in_channels),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels, num_filters[0], kernel_size=1, stride=1,
                          bias=False),
                nn.GroupNorm(num_channels_to_num_groups(num_filters[0]), num_filters[0]),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(num_filters[0], num_filters[1], kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.GroupNorm(num_channels_to_num_groups(num_filters[1]), num_filters[1]),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(num_filters[1], in_channels, kernel_size=1, stride=1,
                          bias=False)
            )
        else:
            self.forward_layers = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels, num_filters[0], kernel_size=1, stride=1,
                          bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(num_filters[0], num_filters[1], kernel_size=3,
                          stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(num_filters[1], in_channels, kernel_size=1, stride=1,
                          bias=True)
            )

    def forward(self, inputs):
        return self.forward_layers(inputs)


class ConvBlock3d(nn.Module):
    """Block of 1x1, 3x3, 1x1 convolutions with non linearities. Shape of input
    and output is the same.

    Args:
        in_channels (int): Number of channels in input.
        num_filters (list of ints): List of two ints with the number of filters
            for the first and second conv layers. Third conv layer must have the
            same number of input filters as there are channels.
        add_groupnorm (bool): If True adds BatchNorm.
    """
    def __init__(self, in_channels, num_filters, add_groupnorm=True):
        super(ConvBlock3d, self).__init__()
        if add_groupnorm:
            self.forward_layers = nn.Sequential(
                nn.GroupNorm(num_channels_to_num_groups(in_channels), in_channels),
                nn.LeakyReLU(0.2, True),
                nn.Conv3d(in_channels, num_filters[0], kernel_size=1, stride=1,
                          bias=False),
                nn.GroupNorm(num_channels_to_num_groups(num_filters[0]), num_filters[0]),
                nn.LeakyReLU(0.2, True),
                nn.Conv3d(num_filters[0], num_filters[1], kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.GroupNorm(num_channels_to_num_groups(num_filters[1]), num_filters[1]),
                nn.LeakyReLU(0.2, True),
                nn.Conv3d(num_filters[1], in_channels, kernel_size=1, stride=1,
                          bias=False)
            )
        else:
            self.forward_layers = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv3d(in_channels, num_filters[0], kernel_size=1, stride=1,
                          bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv3d(num_filters[0], num_filters[1], kernel_size=3,
                          stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv3d(num_filters[1], in_channels, kernel_size=1, stride=1,
                          bias=True)
            )

    def forward(self, inputs):
        return self.forward_layers(inputs)


class ResBlock2d(nn.Module):
    """Residual block of 1x1, 3x3, 1x1 convolutions with non linearities. Shape
    of input and output is the same.

    Args:
        in_channels (int): Number of channels in input.
        num_filters (list of ints): List of two ints with the number of filters
            for the first and second conv layers. Third conv layer must have the
            same number of input filters as there are channels.
        add_groupnorm (bool): If True adds GroupNorm.
    """
    def __init__(self, in_channels, num_filters, add_groupnorm=True):
        super(ResBlock2d, self).__init__()
        self.residual_layers = ConvBlock2d(in_channels, num_filters,
                                           add_groupnorm)

    def forward(self, inputs):
        return inputs + self.residual_layers(inputs)


class ResBlock3d(nn.Module):
    """Residual block of 1x1, 3x3, 1x1 convolutions with non linearities. Shape
    of input and output is the same.

    Args:
        in_channels (int): Number of channels in input.
        num_filters (list of ints): List of two ints with the number of filters
            for the first and second conv layers. Third conv layer must have the
            same number of input filters as there are channels.
        add_groupnorm (bool): If True adds GroupNorm.
    """
    def __init__(self, in_channels, num_filters, add_groupnorm=True):
        super(ResBlock3d, self).__init__()
        self.residual_layers = ConvBlock3d(in_channels, num_filters,
                                           add_groupnorm)

    def forward(self, inputs):
        return inputs + self.residual_layers(inputs)


def num_channels_to_num_groups(num_channels):
    """Returns number of groups to use in a GroupNorm layer with a given number
    of channels. Note that these choices are hyperparameters.

    Args:
        num_channels (int): Number of channels.
    """
    if num_channels < 8:
        return 1
    if num_channels < 32:
        return 2
    if num_channels < 64:
        return 4
    if num_channels < 128:
        return 8
    if num_channels < 256:
        return 16
    else:
        return 32
