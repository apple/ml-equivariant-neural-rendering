import models.layers
import models.submodels
import torch
from math import pi


def full_rotation_angle_sequence(num_steps):
    """Returns a sequence of angles corresponding to a full 360 degree rotation.
    Useful for generating gifs.

    Args:
        num_steps (int): Number of steps in sequence.
    """
    return torch.linspace(0., 360. - 360. / num_steps, num_steps)


def constant_angle_sequence(num_steps, value=0.):
    """Returns a sequence of constant angles. Useful for generating gifs.

    Args:
        num_steps (int): Number of steps in sequence.
        value (float): Constant angle value.
    """
    return value * torch.ones(num_steps)


def back_and_forth_angle_sequence(num_steps, start, end):
    """Returns a sequence of angles linearly increasing from start to end and
    back.

    Args:
        num_steps (int): Number of steps in sequence.
        start (float): Angle at which to start (in degrees).
        end (float): Angle at which to end (in degrees).
    """
    half_num_steps = int(num_steps / 2)
    # Increase angle from start to end
    first = torch.linspace(start, end - end / half_num_steps, half_num_steps)
    # Decrease angle from end to start
    second = torch.linspace(end, start - start / half_num_steps, half_num_steps)
    # Return combined sequence of increasing and decreasing angles
    return torch.cat([first, second], dim=0)


def sinusoidal_angle_sequence(num_steps, minimum, maximum):
    """Returns a sequence of angles sinusoidally varying between minimum and
    maximum.

    Args:
        num_steps (int): Number of steps in sequence.
        start (float): Angle at which to start (in degrees).
        end (float): Angle at which to end (in degrees).
    """
    period = 2 * pi * torch.linspace(0., 1. - 1. / num_steps, num_steps)
    return .5 * (minimum + maximum + (maximum - minimum) * torch.sin(period))


def sine_squared_angle_sequence(num_steps, start, end):
    """Returns a sequence of angles increasing from start to end and back as the
    sine squared function.

    Args:
        num_steps (int): Number of steps in sequence.
        start (float): Angle at which to start (in degrees).
        end (float): Angle at which to end (in degrees).
    """
    half_period = pi * torch.linspace(0., 1. - 1. / num_steps, num_steps)
    return start + (end - start) * torch.sin(half_period) ** 2


def count_parameters(model):
    """Returns number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layers_info(model):
    """Returns information about input shapes, output shapes and number of
    parameters in every block of model.

    Args:
        model (torch.nn.Module): Model to analyse. This will typically be a
            submodel of models.neural_renderer.NeuralRenderer.
    """
    in_shape = model.input_shape
    layers_info = []

    if isinstance(model, models.submodels.Projection):
        out_shape = (in_shape[0] * in_shape[1], *in_shape[2:])
        layer_info = {"name": "Reshape", "in_shape": in_shape,
                      "out_shape": out_shape, "num_params": 0}
        layers_info.append(layer_info)
        in_shape = out_shape

    for layer in model.forward_layers:
        if isinstance(layer, torch.nn.Conv2d):
            if layer.stride[0] == 1:
                out_shape = (layer.out_channels, *in_shape[1:])
            elif layer.stride[0] == 2:
                out_shape = (layer.out_channels, in_shape[1] // 2, in_shape[2] // 2)
            name = "Conv2D"
        elif isinstance(layer, torch.nn.ConvTranspose2d):
            if layer.stride[0] == 1:
                out_shape = (layer.out_channels, *in_shape[1:])
            elif layer.stride[0] == 2:
                out_shape = (layer.out_channels, in_shape[1] * 2, in_shape[2] * 2)
            name = "ConvTr2D"
        elif isinstance(layer, models.layers.ResBlock2d):
            out_shape = in_shape
            name = "ResBlock2D"
        elif isinstance(layer, torch.nn.Conv3d):
            if layer.stride[0] == 1:
                out_shape = (layer.out_channels, *in_shape[1:])
            elif layer.stride[0] == 2:
                out_shape = (layer.out_channels, in_shape[1] // 2, in_shape[2] // 2, in_shape[3] // 2)
            name = "Conv3D"
        elif isinstance(layer, torch.nn.ConvTranspose3d):
            if layer.stride[0] == 1:
                out_shape = (layer.out_channels, *in_shape[1:])
            elif layer.stride[0] == 2:
                out_shape = (layer.out_channels, in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2)
            name = "ConvTr3D"
        elif isinstance(layer, models.layers.ResBlock3d):
            out_shape = in_shape
            name = "ResBlock3D"
        else:
            # If layer is just an activation layer, skip
            continue

        num_params = count_parameters(layer)
        layer_info = {"name": name, "in_shape": in_shape,
                      "out_shape": out_shape, "num_params": num_params}
        layers_info.append(layer_info)

        in_shape = out_shape

    if isinstance(model, models.submodels.InverseProjection):
        layer_info = {"name": "Reshape", "in_shape": in_shape,
                      "out_shape": model.output_shape, "num_params": 0}
        layers_info.append(layer_info)

    return layers_info


def pretty_print_layers_info(model, title):
    """Prints information about a model.

    Args:
        model (see get_layers_info)
        title (string): Title of model.
    """
    # Extract layers info for model
    layers_info = get_layers_info(model)
    # Print information in a nice format
    print(title)
    print("-" * len(title))
    print("{: <12} \t {: <14} \t {: <14} \t {: <10} \t {: <10}".format("name", "in_shape", "out_shape", "num_params", "feat_size"))
    print("---------------------------------------------------------------------------------------------")

    min_feat_size = 2 ** 20  # Some huge number
    for info in layers_info:
        feat_size = tuple_product(info["out_shape"])
        print("{: <12} \t {: <14} \t {: <14} \t {: <10} \t {: <10}".format(info["name"],
                                                                            str(info["in_shape"]),
                                                                            str(info["out_shape"]),
                                                                            info["num_params"],
                                                                            feat_size))
        if feat_size < min_feat_size:
            min_feat_size = feat_size
    print("---------------------------------------------------------------------------------------------")
    # Only print model info if model is not empty
    if len(layers_info):
        print("{: <12} \t {: <14} \t {: <14} \t {: <10} \t {: <10}".format("Total",
                                                                str(layers_info[0]["in_shape"]),
                                                                str(layers_info[-1]["out_shape"]),
                                                                count_parameters(model),
                                                                min_feat_size))


def tuple_product(input_tuple):
    """Returns product of elements in a tuple."""
    product = 1
    for elem in input_tuple:
        product *= elem
    return product
