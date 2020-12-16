import torch
import torch.nn as nn
from transforms3d.rotations import rotate, rotate_source_to_target


class Rotate3d(nn.Module):
    """Layer used to rotate 3D feature maps.

    Args:
        mode (string): One of 'bilinear' and 'nearest' for interpolation mode
            used when resampling rotated values on the grid.
    """
    def __init__(self, mode='bilinear'):
        super(Rotate3d, self).__init__()
        self.mode = mode

    def forward(self, volume, rotation_matrix):
        """Rotates the volume by the rotation matrix.

        Args:
            volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            rotation_matrix (torch.Tensor): Batch of rotation matrices of shape
                (batch_size, 3, 3).
        """
        return rotate(volume, rotation_matrix, mode=self.mode)

    def rotate_source_to_target(self, volume, azimuth_source, elevation_source,
                                azimuth_target, elevation_target):
        """Rotates volume from source coordinate frame to target coordinate
        frame.

        Args:
            volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            azimuth_source (torch.Tensor): Shape (batch_size,). Azimuth of
                source view in degrees.
            elevation_source (torch.Tensor): Shape (batch_size,). Elevation of
                source view in degrees.
            azimuth_target (torch.Tensor): Shape (batch_size,). Azimuth of
                target view in degrees.
            elevation_target (torch.Tensor): Shape (batch_size,). Elevation of
                target view in degrees.
        """
        return rotate_source_to_target(volume, azimuth_source, elevation_source,
                                       azimuth_target, elevation_target,
                                       mode=self.mode)


class SphericalMask(nn.Module):
    """Sets all features outside the largest sphere embedded in a cubic tensor
    to zero.

    Args:
        input_shape (tuple of ints): Shape of 3D feature map. Should have the
            form (channels, depth, height, width).
        radius_fraction (float): Fraction of radius to keep as non zero. E.g.
            if radius_fraction=0.9, only elements within the sphere of radius
            0.9 of half the cube length will not be zeroed. Must be in [0., 1.].
    """
    def __init__(self, input_shape, radius_fraction=1.):
        super(SphericalMask, self).__init__()
        # Check input
        _, depth, height, width = input_shape
        assert depth == height, "Depth, height, width are {}, {}, {} but must be equal.".format(depth, height, width)
        assert height == width, "Depth, height, width are {}, {}, {} but must be equal.".format(depth, height, width)

        self.input_shape = input_shape

        # Build spherical mask
        mask = torch.ones(input_shape)
        mask_center = (depth - 1) / 2  # Center of cube (in terms of index)
        radius = (depth - 1) / 2  # Distance from center to edge of cube is radius of sphere
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    squared_distance = (mask_center - i) ** 2  + (mask_center - j) ** 2 + (mask_center - k) ** 2
                    if squared_distance > (radius_fraction * radius) ** 2:
                        mask[:, i, j, k] = 0.

        # Register buffer adds a key to the state dict of the model. This will
        # track the attribute without registering it as a learnable parameter.
        # This also means mask will be moved to device when calling
        # model.to(device)
        self.register_buffer('mask', mask)

    def forward(self, volume):
        """Applies a spherical mask to input.

        Args:
            volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        """
        return volume * self.mask
