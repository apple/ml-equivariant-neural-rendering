import glob
import json
import torch
from numpy import float32 as np_float32
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms


def scene_render_dataloader(path_to_data='chairs-train', batch_size=16,
                            img_size=(3, 128, 128), crop_size=128):
    """Dataloader for scene render datasets. Returns scene renders in pairs,
    i.e. 1st and 2nd images are of some scene, 3rd and 4th are of some different
    scene and so on.

    Args:
        path_to_data (string): Path to folder containing dataset.
        batch_size (int): Batch size for data.
        img_size (tuple of ints): Size of output images.
        crop_size (int): Size at which to center crop rendered images.

    Notes:
        Batch size must be even.
    """
    assert batch_size % 2 == 0, "Batch size is {} but must be even".format(batch_size)

    dataset = scene_render_dataset(path_to_data, img_size, crop_size)

    sampler = RandomPairSampler(dataset)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      drop_last=True)


def scene_render_dataset(path_to_data='chairs-train', img_size=(3, 128, 128),
                         crop_size=128, allow_odd_num_imgs=False):
    """Helper function for creating a scene render dataset.

    Args:
        path_to_data (string): Path to folder containing dataset.
        img_size (tuple of ints): Size of output images.
        crop_size (int): Size at which to center crop rendered images.
        allow_odd_num_imgs (int): If True, allows datasets with an odd number
            of views. Such a dataset cannot be used for training, since each
            training iteration requires a *pair* of images. Datasets with an odd
            number of images are used for PSNR calculations.
    """
    img_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(img_size[1:]),
        transforms.ToTensor()
    ])

    dataset = SceneRenderDataset(path_to_data=path_to_data,
                                 img_transform=img_transform,
                                 allow_odd_num_imgs=allow_odd_num_imgs)

    return dataset


class SceneRenderDataset(Dataset):
    """Dataset of rendered scenes and their corresponding camera angles.

    Args:
        path_to_data (string): Path to folder containing dataset.
        img_transform (torchvision.transform): Transforms to be applied to
            images.
        allow_odd_num_imgs (bool): If True, allows datasets with an odd number
            of views. Such a dataset cannot be used for training, since each
            training iteration requires a *pair* of images.

    Notes:
        - Image paths must be of the form "XXXXX.png" where XXXXX are *five*
        integers indexing the image.
        - We assume there are the same number of rendered images for each scene
        and that this number is even.
        - We assume angles are given in degrees.
    """
    def __init__(self, path_to_data='chairs-train', img_transform=None,
                 allow_odd_num_imgs=False):
        self.path_to_data = path_to_data
        self.img_transform = img_transform
        self.allow_odd_num_imgs =  allow_odd_num_imgs
        self.data = []
        # Each folder contains a single scene with different rendering
        # parameters and views
        self.scene_paths = glob.glob(path_to_data + '/*')
        self.scene_paths.sort()  # Ensure consistent ordering of scenes
        self.num_scenes = len(self.scene_paths)
        # Extract number of rendered images per object (which we assume is constant)
        self.num_imgs_per_scene = len(glob.glob(self.scene_paths[0] + '/*.png'))
        # If number of images per scene is not even, drop last image
        if self.num_imgs_per_scene % 2 != 0:
            if not self.allow_odd_num_imgs:
                self.num_imgs_per_scene -= 1
        # For each scene, extract its rendered views and render parameters
        for scene_path in self.scene_paths:
            # Name of folder defines scene name
            scene_name = scene_path.split('/')[-1]

            # Load render parameters
            with open(scene_path + '/render_params.json') as f:
                render_params = json.load(f)

            # Extract path to rendered images of scene
            img_paths = glob.glob(scene_path + '/*.png')
            img_paths.sort()  # Ensure consistent ordering of images
            # Ensure number of image paths is even
            img_paths = img_paths[:self.num_imgs_per_scene]

            for img_path in img_paths:
                # Extract image filename
                img_file = img_path.split('/')[-1]
                # Filenames are of the type "<index>.png", so extract this
                # index to match with render parameters.
                img_idx = img_file.split('.')[0][-5:]  # This should be a string
                # Convert render parameters to float32
                img_params = {key: np_float32(value)
                              for key, value in render_params[img_idx].items()}
                self.data.append({
                    "scene_name": scene_name,
                    "img_path": img_path,
                    "render_params": img_params
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["img_path"]
        render_params = self.data[idx]["render_params"]

        img = Image.open(img_path)

        # Transform images
        if self.img_transform:
            img = self.img_transform(img)

        # Note some images may contain 4 channels (i.e. RGB + alpha), we only
        # keep RGB channels
        data_item = {
            "img": img[:3],
            "scene_name": self.data[idx]["scene_name"],
            "render_params": self.data[idx]["render_params"]
        }

        return data_item


class RandomPairSampler(Sampler):
    """Samples random elements in pairs. Dataset is assumed to be composed of a
    number of scenes, each rendered in a number of views. This sampler returns
    rendered image in pairs. I.e. for a batch of size 6, it would return e.g.:

    [object 4 - img 5,
     object 4 - img 12,
     object 6 - img 3,
     object 6 - img 19,
     object 52 - img 10,
     object 52 - img 3]


    Arguments:
        dataset (Dataset): Dataset to sample from. This will typically be an
            instance of SceneRenderDataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        num_scenes = self.dataset.num_scenes
        num_imgs_per_scene = self.dataset.num_imgs_per_scene

        # Sample num_imgs_per_scene / 2 permutations of the objects
        scene_permutations = [torch.randperm(num_scenes) for _ in range(num_imgs_per_scene // 2)]
        # For each scene, sample a permutation of its images
        img_permutations = [torch.randperm(num_imgs_per_scene) for _ in range(num_scenes)]

        data_permutation = []

        for i, scene_permutation in enumerate(scene_permutations):
            for scene_idx in scene_permutation:
                # Extract image permutation for this object
                img_permutation = img_permutations[scene_idx]
                # Add 2 images of this object to data_permutation
                data_permutation.append(scene_idx.item() * num_imgs_per_scene + img_permutation[2*i].item())
                data_permutation.append(scene_idx.item() * num_imgs_per_scene + img_permutation[2*i + 1].item())

        return iter(data_permutation)

    def __len__(self):
        return len(self.dataset)


def create_batch_from_data_list(data_list):
    """Given a list of datapoints, create a batch.

    Args:
        data_list (list): List of items returned by SceneRenderDataset.
    """
    imgs = []
    azimuths = []
    elevations = []
    for data_item in data_list:
        img, render_params = data_item["img"], data_item["render_params"]
        azimuth, elevation = render_params["azimuth"], render_params["elevation"]
        imgs.append(img.unsqueeze(0))
        azimuths.append(torch.Tensor([azimuth]))
        elevations.append(torch.Tensor([elevation]))
    imgs = torch.cat(imgs, dim=0)
    azimuths = torch.cat(azimuths)
    elevations = torch.cat(elevations)
    return imgs, azimuths, elevations
