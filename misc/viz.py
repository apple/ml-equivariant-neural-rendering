import imageio
import torch
import torchvision
from misc.dataloaders import create_batch_from_data_list


def generate_novel_views(model, img_source, azimuth_source, elevation_source,
                         azimuth_shifts, elevation_shifts):
    """Generates novel views of an image by inferring its scene representation,
    rotating it and rendering novel views. Returns a batch of images
    corresponding to the novel views.

    Args:
        model (models.neural_renderer.NeuralRenderer): Neural rendering model.
        img_source (torch.Tensor): Single image. Shape (channels, height, width).
        azimuth_source (torch.Tensor): Azimuth of source image. Shape (1,).
        elevation_source (torch.Tensor): Elevation of source image. Shape (1,).
        azimuth_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
        elevation_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
    """
    # No need to calculate gradients
    with torch.no_grad():
        num_views = len(azimuth_shifts)
        # Batchify image
        img_batch = img_source.unsqueeze(0)
        # Infer scene
        scenes = model.inverse_render(img_batch)
        # Copy scene for each target view
        scenes_batch = scenes.repeat(num_views, 1, 1, 1, 1)
        # Batchify azimuth and elevation source
        azimuth_source_batch = azimuth_source.repeat(num_views)
        elevation_source_batch = elevation_source.repeat(num_views)
        # Calculate azimuth and elevation targets
        azimuth_target = azimuth_source_batch + azimuth_shifts
        elevation_target = elevation_source_batch + elevation_shifts
        # Rotate scenes
        rotated = model.rotate_source_to_target(scenes_batch, azimuth_source_batch,
                                                elevation_source_batch,
                                                azimuth_target, elevation_target)
    # Render images
    return model.render(rotated).detach()


def batch_generate_novel_views(model, imgs_source, azimuth_source,
                               elevation_source, azimuth_shifts,
                               elevation_shifts):
    """Generates novel views for a batch of images. Returns a list of batches of
    images, where each item in the list corresponds to a novel view for all
    images.

    Args:
        model (models.neural_renderer.NeuralRenderer): Neural rendering model.
        imgs_source (torch.Tensor): Source images. Shape (batch_size, channels, height, width).
        azimuth_source (torch.Tensor): Azimuth of source. Shape (batch_size,).
        elevation_source (torch.Tensor): Elevation of source. Shape (batch_size,).
        azimuth_shifts (torch.Tensor): Batch of angle shifts at which to generate
            novel views. Shape (num_views,).
        elevation_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
    """
    num_imgs = imgs_source.shape[0]
    num_views = azimuth_shifts.shape[0]

    # Initialize novel views, i.e. a list of length num_views with each item
    # containing num_imgs images
    all_novel_views = [torch.zeros_like(imgs_source) for _ in range(num_views)]

    for i in range(num_imgs):
        # Generate novel views for single image
        novel_views = generate_novel_views(model, imgs_source[i],
                                           azimuth_source[i:i+1],
                                           elevation_source[i:i+1],
                                           azimuth_shifts, elevation_shifts).cpu()
        # Add to list of all novel_views
        for j in range(num_views):
            all_novel_views[j][i] = novel_views[j]

    return all_novel_views


def dataset_novel_views(device, model, dataset, img_indices, azimuth_shifts,
                        elevation_shifts):
    """Helper function for generating novel views from specific images in a
    dataset.

    Args:
        device (torch.device):
        model (models.neural_renderer.NeuralRenderer):
        dataset (misc.dataloaders.SceneRenderDataset):
        img_indices (tuple of ints): Indices of images in dataset to use as
            source views for novel view synthesis.
        azimuth_shifts (torch.Tensor): Batch of angle shifts at which to generate
            novel views. Shape (num_views,).
        elevation_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
    """
    # Extract image and pose information for all views
    data_list = []
    for img_idx in img_indices:
        data_list.append(dataset[img_idx])
    imgs_source, azimuth_source, elevation_source = create_batch_from_data_list(data_list)
    imgs_source = imgs_source.to(device)
    azimuth_source = azimuth_source.to(device)
    elevation_source = elevation_source.to(device)
    # Generate novel views
    return batch_generate_novel_views(model, imgs_source, azimuth_source,
                                      elevation_source, azimuth_shifts,
                                      elevation_shifts)


def shapenet_test_novel_views(device, model, dataset, source_scenes_idx=(0, 1, 2, 3),
                              source_img_idx_shift=64, subsample_target=5):
    """Helper function for generating novel views on an archimedean spiral for
    the test images for ShapeNet chairs and cars.

    Args:
        device (torch.device):
        model (models.neural_renderer.NeuralRenderer):
        dataset (misc.dataloaders.SceneRenderDataset): Test dataloader for a
            ShapeNet dataset.
        source_scenes_idx (tuple of ints): Indices of source scenes to use for
            generating novel views.
        source_img_idx_shift (int): Index of source image for each scene. For
            example if 00064.png is the source view, then
            source_img_idx_shift = 64.
        subsample_target (int): Amount by which to subsample target views. If
            set to 1, uses all 250 target views.
    """
    num_imgs = len(source_scenes_idx)
    # Extract source azimuths and elevations
    # Note we can extract this from the first scene since for the shapenet test
    # sets, the test poses are the same for all scenes
    render_params = dataset[source_img_idx_shift]["render_params"]
    azimuth_source = torch.Tensor([render_params["azimuth"]]).to(device)
    elevation_source = torch.Tensor([render_params["elevation"]]).to(device)

    # Extract target azimuths and elevations (do not use final view as it is
    # slightly off in dataset)
    azimuth_target = torch.zeros(dataset.num_imgs_per_scene - 1)
    elevation_target = torch.zeros(dataset.num_imgs_per_scene - 1)
    for i in range(dataset.num_imgs_per_scene - 1):
        render_params = dataset[i]["render_params"]
        azimuth_target[i] = torch.Tensor([render_params["azimuth"]])
        elevation_target[i] = torch.Tensor([render_params["elevation"]])
    # Move to GPU
    azimuth_target = azimuth_target.to(device)
    elevation_target = elevation_target.to(device)
    # Subsample
    azimuth_target = azimuth_target[::subsample_target]
    elevation_target = elevation_target[::subsample_target]

    # Calculate azimuth and elevation shifts
    azimuth_shifts = azimuth_target - azimuth_source
    elevation_shifts = elevation_target - elevation_source

    # Ensure source angles have same batch_size as imgs_source
    azimuth_source = azimuth_source.repeat(num_imgs)
    elevation_source = elevation_source.repeat(num_imgs)

    # Create source image batch
    imgs_source = torch.zeros(num_imgs, 3, 128, 128).to(device)
    for i in range(num_imgs):
        scene_idx = source_scenes_idx[i]
        source_img_idx = scene_idx * dataset.num_imgs_per_scene + source_img_idx_shift
        imgs_source[i] = dataset[source_img_idx]["img"].to(device)

    return batch_generate_novel_views(model, imgs_source, azimuth_source,
                                      elevation_source, azimuth_shifts,
                                      elevation_shifts)


def save_generate_novel_views(filename, model, img_source, azimuth_source,
                              elevation_source, azimuth_shifts,
                              elevation_shifts):
    """Generates novel views of an image by inferring its scene representation,
    rotating it and rendering novel views. Saves the source image and novel
    views as png files.

    Args:
        filename (string): Filename root for saving images.
        model (models.neural_renderer.NeuralRenderer): Neural rendering model.
        img_source (torch.Tensor): Single image. Shape (channels, height, width).
        azimuth_source (torch.Tensor): Azimuth of source image. Shape (1,).
        elevation_source (torch.Tensor): Elevation of source image. Shape (1,).
        azimuth_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
        elevation_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
    """
    # Generate novel views
    novel_views = generate_novel_views(model, img_source, azimuth_source,
                                       elevation_source, azimuth_shifts,
                                       elevation_shifts)
    # Save original image
    torchvision.utils.save_image(img_source, filename + '.png', padding=4,
                                 pad_value=1.)
    # Save novel views (with white padding)
    torchvision.utils.save_image(novel_views, filename + '_novel.png',
                                 padding=4, pad_value=1.)


def save_img_sequence_as_gif(img_sequence, filename, nrow=4):
    """Given a sequence of images as tensors, saves a gif of the images.
    If images are in batches, they are converted to a grid before being
    saved.

    Args:
        img_sequence (list of torch.Tensor): List of images. Tensors should
            have shape either (batch_size, channels, height, width) or shape
            (channels, height, width). If there is a batch dimension, images
            will be converted to a grid.
        filename (string): Path where gif will be saved. Should end in '.gif'.
        nrow (int): Number of rows in image grid, if image has batch dimension.
    """
    img_grid_sequence = []
    for img in img_sequence:
        if len(img.shape) == 4:
            img_grid = torchvision.utils.make_grid(img, nrow=nrow)
        else:
            img_grid = img
        # Convert to numpy array and from float in [0, 1] to int in [0, 255]
        # which is what imageio expects
        img_grid = (img_grid * 255.).byte().cpu().numpy().transpose(1, 2, 0)
        img_grid_sequence.append(img_grid)
    # Save gif
    imageio.mimwrite(filename, img_grid_sequence)
