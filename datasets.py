# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torchvision
import torch
from PIL import Image

ROOT_ = "/your/data/dir/"


IMAGE_LISTS_ = {
    "ImageNet-va": "my_imagenet_va.txt",
    "ImageNet-te": "my_imagenet_te.txt",
    "ImageNet-v2": "test_imagenet_v2.txt",
    "ImageNet-C": "test_imagenet_c.txt",
    "ImageNet-R": "test_imagenet_r.txt",
    "NINCO": "test_ninco.txt",
    "SSB-Hard": "test_ssb_hard.txt",
    "iNaturalist": "test_inaturalist.txt",
    "Texture": "test_textures.txt",
    "OpenImage-O": "test_openimage_o.txt",
    "ImageNet-O": "test_imagenet_o.txt",
}


class ImageListDataset:
    def __init__(self, root, list_, transform):
        self.transform = transform
        self.images = []
        self.labels = []

        with open(list_, "r") as f:
            for line in f.readlines():
                x, y = line.strip().split(" ")
                y = 1000 if y == "-1" else int(y)
                self.images.append(root + x)
                self.labels.append(y)

    def __getitem__(self, i):
        x = self.transform(Image.open(self.images[i]).convert("RGB"))
        y = self.labels[i]
        return x, y

    def __len__(self):
        return len(self.images)


def get_loader(root, list_, batch_size=128):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    dataset = ImageListDataset(root, list_, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True)

    return loader


def get(name):
    root_lists = "benchmark_imglist/imagenet/"

    if name == "names":
        return list(IMAGE_LISTS_.keys())

    return get_loader(ROOT_, root_lists + IMAGE_LISTS_[name])
