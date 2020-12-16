import torch
import torch.nn as nn
from misc.utils import pretty_print_layers_info, count_parameters
from models.submodels import ResNet2d, ResNet3d, Projection, InverseProjection
from models.rotation_layers import SphericalMask, Rotate3d


class NeuralRenderer(nn.Module):
    """Implements a Neural Renderer with an implicit scene representation that
    allows both forward and inverse rendering.

    The forward pass from 3d scene to 2d image is (rendering):
    Scene representation (input) -> ResNet3d -> Projection -> ResNet2d ->
    Rendered image (output)

    The inverse pass from 2d image to 3d scene is (inverse rendering):
    Image (input) -> ResNet2d -> Inverse Projection -> ResNet3d -> Scene
    representation (output)

    Args:
        img_shape (tuple of ints): Shape of the image input to the model. Should
            be of the form (channels, height, width).
        channels_2d (tuple of ints): List of channels for 2D layers in inverse
            rendering model (image -> scene).
        strides_2d (tuple of ints): List of strides for 2D layers in inverse
            rendering model (image -> scene).
        channels_3d (tuple of ints): List of channels for 3D layers in inverse
            rendering model (image -> scene).
        strides_3d (tuple of ints): List of channels for 3D layers in inverse
            rendering model (image -> scene).
        num_channels_inv_projection (tuple of ints): Number of channels in each
            layer of inverse projection unit from 2D to 3D.
        num_channels_projection (tuple of ints): Number of channels in each
            layer of projection unit from 2D to 3D.
        mode (string): One of 'bilinear' and 'nearest' for interpolation mode
            used when rotating voxel grid.

    Notes:
        Given the inverse rendering channels and strides, the model will
        automatically build a forward renderer as the transpose of the inverse
        renderer.
    """
    def __init__(self, img_shape, channels_2d, strides_2d, channels_3d,
                 strides_3d, num_channels_inv_projection, num_channels_projection,
                 mode='bilinear'):
        super(NeuralRenderer, self).__init__()
        self.img_shape = img_shape
        self.channels_2d = channels_2d
        self.strides_2d = strides_2d
        self.channels_3d = channels_3d
        self.strides_3d = strides_3d
        self.num_channels_projection = num_channels_projection
        self.num_channels_inv_projection = num_channels_inv_projection
        self.mode = mode

        # Initialize layers

        # Inverse pass (image -> scene)
        # First transform image into a 2D representation
        self.inv_transform_2d = ResNet2d(self.img_shape, channels_2d,
                                         strides_2d)

        # Perform inverse projection from 2D to 3D
        input_shape = self.inv_transform_2d.output_shape
        self.inv_projection = InverseProjection(input_shape, num_channels_inv_projection)

        # Transform 3D inverse projection into a scene representation
        self.inv_transform_3d = ResNet3d(self.inv_projection.output_shape,
                                         channels_3d, strides_3d)
        # Add rotation layer
        self.rotation_layer = Rotate3d(self.mode)

        # Forward pass (scene -> image)
        # Forward renderer is just transpose of inverse renderer, so flip order
        # of channels and strides
        # Transform scene representation to 3D features
        forward_channels_3d = list(reversed(channels_3d))[1:] + [channels_3d[0]]
        forward_strides_3d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_3d[1:]))] + [strides_3d[0]]
        self.transform_3d = ResNet3d(self.inv_transform_3d.output_shape,
                                     forward_channels_3d, forward_strides_3d)

        # Layer for projection of 3D representation to 2D representation
        self.projection = Projection(self.transform_3d.output_shape,
                                     num_channels_projection)

        # Transform 2D features to rendered image
        forward_channels_2d = list(reversed(channels_2d))[1:] + [channels_2d[0]]
        forward_strides_2d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_2d[1:]))] + [strides_2d[0]]
        final_conv_channels_2d = img_shape[0]
        self.transform_2d = ResNet2d(self.projection.output_shape,
                                     forward_channels_2d, forward_strides_2d,
                                     final_conv_channels_2d)

        # Scene representation shape is output of inverse 3D transformation
        self.scene_shape = self.inv_transform_3d.output_shape
        # Add spherical mask before scene rotation
        self.spherical_mask = SphericalMask(self.scene_shape)

    def render(self, scene):
        """Renders a scene to an image.

        Args:
            scene (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        """
        features_3d = self.transform_3d(scene)
        features_2d = self.projection(features_3d)
        return torch.sigmoid(self.transform_2d(features_2d))

    def inverse_render(self, img):
        """Maps an image to a (spherical) scene representation.

        Args:
            img (torch.Tensor): Shape (batch_size, channels, height, width).
        """
        # Transform image to 2D features
        features_2d = self.inv_transform_2d(img)
        # Perform inverse projection
        features_3d = self.inv_projection(features_2d)
        # Map 3D features to scene representation
        scene = self.inv_transform_3d(features_3d)
        # Ensure scene is spherical
        return self.spherical_mask(scene)

    def rotate(self, scene, rotation_matrix):
        """Rotates scene by rotation matrix.

        Args:
            scene (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            rotation_matrix (torch.Tensor): Batch of rotation matrices of shape
                (batch_size, 3, 3).
        """
        return self.rotation_layer(scene, rotation_matrix)

    def rotate_source_to_target(self, scene, azimuth_source, elevation_source,
                                azimuth_target, elevation_target):
        """Assuming the scene is being observed by a camera at
        (azimuth_source, elevation_source), rotates scene so camera is observing
        it at (azimuth_target, elevation_target).

        Args:
            scene (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            azimuth_source (torch.Tensor): Shape (batch_size,). Azimuth of source.
            elevation_source (torch.Tensor): Shape (batch_size,). Elevation of source.
            azimuth_target (torch.Tensor): Shape (batch_size,). Azimuth of target.
            elevation_target (torch.Tensor): Shape (batch_size,). Elevation of target.
        """
        return self.rotation_layer.rotate_source_to_target(scene,
                                                           azimuth_source,
                                                           elevation_source,
                                                           azimuth_target,
                                                           elevation_target)

    def forward(self, batch):
        """Given a batch of images and poses, infers scene representations,
        rotates them into target poses and renders them into images.

        Args:
            batch (dict): A batch of images and poses as returned by
                misc.dataloaders.scene_render_dataloader.

        Notes:
            This *must* be a batch as returned by the scene render dataloader,
            i.e. the batch must be composed of pairs of images of the same
            scene. Specifically, the first time in the batch should be an image
            of scene A and the second item in the batch should be an image of
            scene A observed from a different pose. The third item should be an
            image of scene B and the fourth item should be an image scene B
            observed from a different pose (and so on).
        """
        # Slightly hacky way of extracting model device. Device on which
        # spherical is stored is the one where model is too
        device = self.spherical_mask.mask.device
        imgs = batch["img"].to(device)
        params = batch["render_params"]
        azimuth = params["azimuth"].to(device)
        elevation = params["elevation"].to(device)

        # Infer scenes from images
        scenes = self.inverse_render(imgs)

        # Rotate scenes so that for every pair of rendered images, the 1st
        # one will be reconstructed as the 2nd and then 2nd will be
        # reconstructed as the 1st
        swapped_idx = get_swapped_indices(azimuth.shape[0])

        # Each pair of indices in the azimuth vector corresponds to the same
        # scene at two different angles. Therefore performing a pairwise swap,
        # the first index will correspond to the second index in the original
        # vector. Since we want to rotate camera angle 1 to camera angle 2 and
        # vice versa, we can use these swapped angles to define a target
        # position for the camera
        azimuth_swapped = azimuth[swapped_idx]
        elevation_swapped = elevation[swapped_idx]
        scenes_swapped = \
            self.rotate_source_to_target(scenes, azimuth, elevation,
                                         azimuth_swapped, elevation_swapped)

        # Swap scenes, so rotated scenes match with original inferred scene.
        # Specifically, we have images x1, x2 from which we inferred the scenes
        # z1, z2. We then rotated these scenes into z1' and z2'. Now z1' should
        # be almost equal to z2 and z2' should be almost equal to z1, so we swap
        # the order of z1', z2' to z2', z1' so we can easily render them to
        # x1 and x2.
        scenes_rotated = scenes_swapped[swapped_idx]

        # Render scene using model
        rendered = self.render(scenes_rotated)

        return imgs, rendered, scenes, scenes_rotated

    def print_model_info(self):
        """Prints detailed information about model, such as how input shape is
        transformed to output shape and how many parameters are trained in each
        block.
        """
        print("Forward renderer")
        print("----------------\n")
        pretty_print_layers_info(self.transform_3d, "3D Layers")
        print("\n")
        pretty_print_layers_info(self.projection, "Projection")
        print("\n")
        pretty_print_layers_info(self.transform_2d, "2D Layers")
        print("\n")

        print("Inverse renderer")
        print("----------------\n")
        pretty_print_layers_info(self.inv_transform_2d, "Inverse 2D Layers")
        print("\n")
        pretty_print_layers_info(self.inv_projection, "Inverse Projection")
        print("\n")
        pretty_print_layers_info(self.inv_transform_3d, "Inverse 3D Layers")
        print("\n")

        print("Scene Representation:")
        print("\tShape: {}".format(self.scene_shape))
        # Size of scene representation corresponds to non zero entries of
        # spherical mask
        print("\tSize: {}\n".format(int(self.spherical_mask.mask.sum().item())))

        print("Number of parameters: {}\n".format(count_parameters(self)))

    def get_model_config(self):
        """Returns the complete model configuration as a dict."""
        return {
            "img_shape": self.img_shape,
            "channels_2d": self.channels_2d,
            "strides_2d": self.strides_2d,
            "channels_3d": self.channels_3d,
            "strides_3d": self.strides_3d,
            "num_channels_inv_projection": self.num_channels_inv_projection,
            "num_channels_projection": self.num_channels_projection,
            "mode": self.mode
        }

    def save(self, filename):
        """Saves model and its config.

        Args:
            filename (string): Path where model will be saved. Should end with
                '.pt' or '.pth'.
        """
        torch.save({
            "config": self.get_model_config(),
            "state_dict": self.state_dict()
        }, filename)


def load_model(filename):
    """Loads a NeuralRenderer model from saved model config and weights.

    Args:
        filename (string): Path where model was saved.
    """
    model_dict = torch.load(filename, map_location="cpu")
    config = model_dict["config"]
    # Initialize a model based on config
    model = NeuralRenderer(
        img_shape=config["img_shape"],
        channels_2d=config["channels_2d"],
        strides_2d=config["strides_2d"],
        channels_3d=config["channels_3d"],
        strides_3d=config["strides_3d"],
        num_channels_inv_projection=config["num_channels_inv_projection"],
        num_channels_projection=config["num_channels_projection"],
        mode=config["mode"]
    )
    # Load weights into model
    model.load_state_dict(model_dict["state_dict"])
    return model


def get_swapped_indices(length):
    """Returns a list of swapped index pairs. For example, if length = 6, then
    function returns [1, 0, 3, 2, 5, 4], i.e. every index pair is swapped.

    Args:
        length (int): Length of swapped indices.
    """
    return [i + 1 if i % 2 == 0 else i - 1 for i in range(length)]
