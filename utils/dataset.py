import os
import math

from PIL import Image
from torchvision import datasets, transforms


def get_transform(config):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    t = []
    to_resize = int(math.ceil(config.input_size / 0.875))

    t.append(transforms.Resize(to_resize, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(config.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)


def pil_loader_RGB(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def build_dataset(config):
    root = os.path.expanduser(config.valid_set.root)
    loader = pil_loader_RGB
    return datasets.ImageFolder(root, transform=get_transform(config), loader=loader)