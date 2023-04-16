import numpy as np
import polars as pl
from PIL import Image
import random

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as TF


class ClawDataset(Dataset):

    def __init__(self, annotations: str):
        self.file = pl.read_csv(annotations)
        self.is_train = True if annotations.split(".")[0].split("/")[1] == "train" else False
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((384, 512), antialias=True),
        ])
        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Resize((384, 512), antialias=True),
        ])

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index: int):
        item = self.file[index]
        x = Image.open(item["imgs"].item()).convert("RGB")

        # if test (no y)
        if not self.is_train:
            x = self.test_transform(x)
            return x, item["imgs"].item().split("/")[-1]

        y = Image.open(item["masks"].item()).convert("1")
        x = self.transform(x)
        y = self.transform(y)

        # crop
        i, j, h, w = transforms.RandomCrop.get_params(
            torch.randn(600, 600), output_size=(576, 576))
        x = TF.crop(x, i, j, h, w)
        y = TF.crop(y, i, j, h, w)

        # rotation
        angle = random.randrange(-20, 20)
        x = TF.rotate(x, angle)
        y = TF.rotate(y, angle)

        # Random horizontal flipping
        if random.random() > 0.5:
            x = TF.hflip(x)
            y = TF.hflip(y)

        # image stuff
        colorjitter = transforms.ColorJitter((0.5, 1.5), (0.5, 1.5), (0.5, 1.5), None)
        x = colorjitter(x)

        # blur
        if random.random() > 0.5:
            blur = torchvision.transforms.GaussianBlur((9,9), (0.1, 2))
            x = blur(x)

        return x, y
