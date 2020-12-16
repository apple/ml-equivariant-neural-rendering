import torch
from transforms3d.conversions import (rotation_matrix_source_to_target,
                                      transpose_matrix)


def rotate(volume, rotation_matrix, mode='bilinear'):
    """Performs 3D rotation of tensor volume by rotation matrix.

    Args:
        volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        rotation_matrix (torch.Tensor): Batch of rotation matrices of shape
            (batch_size, 3, 3).
        mode (string): One of 'bilinear' and 'nearest' for interpolation mode
            used in grid_sample. Note that the 'bilinear' option actually
            performs trilinear interpolation.

    Notes:
        We use align_corners=False in grid_sample. See
        https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        for a nice illustration of why this is.
    """
    # The grid_sample function performs the inverse transformation of the input
    # coordinates, so invert matrix to get forward transformation
    inverse_rotation_matrix = transpose_matrix(rotation_matrix)
    # The grid_sample function swaps x and z (i.e. it assumes the tensor
    # dimensions are ordered as z, y, x), therefore we need to flip the rows and
    # columns of the matrix (which we can verify is equivalent to multiplying by
    # the appropriate permutation matrices)
    inverse_rotation_matrix_swap_xz = torch.flip(inverse_rotation_matrix,
                                                 dims=(1, 2))
    # Apply transformation to grid
    affine_grid = get_affine_grid(inverse_rotation_matrix_swap_xz, volume.shape)
    # Regrid volume according to transformation grid
    return torch.nn.functional.grid_sample(volume, affine_grid, mode=mode,
                                           align_corners=False)


def get_affine_grid(matrix, grid_shape):
    """Given a matrix and a grid shape, returns the grid transformed by the
    matrix (typically a rotation matrix).

    Args:
        matrix (torch.Tensor): Batch of matrices of size (batch_size, 3, 3).
        grid_shape (torch.size): Shape of returned affine grid. Should be of the
            form (batch_size, channels, depth, height, width).

    Notes:
        We use align_corners=False in affine_grid. See
        https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        for a nice illustration of why this is.
    """
    batch_size = matrix.shape[0]
    # Last column of affine matrix corresponds to translation which is 0 in our
    # case. Therefore pad original matrix with zeros, so shape changes from
    # (batch_size, 3, 3) to (batch_size, 3, 4)
    translations = torch.zeros(batch_size, 3, 1, device=matrix.device)
    affine_matrix = torch.cat([matrix, translations], dim=2)
    return torch.nn.functional.affine_grid(affine_matrix, grid_shape,
                                           align_corners=False)


def rotate_source_to_target(volume, azimuth_source, elevation_source,
                            azimuth_target, elevation_target, mode='bilinear'):
    """Performs 3D rotation matching two coordinate frames defined by a source
    view and a target view.

    Args:
        volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        azimuth_source (torch.Tensor): Shape (batch_size,). Azimuth of source
            view in degrees.
        elevation_source (torch.Tensor): Shape (batch_size,). Elevation of
            source view in degrees.
        azimuth_target (torch.Tensor): Shape (batch_size,). Azimuth of target
            view in degrees.
        elevation_target (torch.Tensor): Shape (batch_size,). Elevation of
            target view in degrees.
    """
    rotation_matrix = rotation_matrix_source_to_target(azimuth_source,
                                                       elevation_source,
                                                       azimuth_target,
                                                       elevation_target)
    return rotate(volume, rotation_matrix, mode=mode)
