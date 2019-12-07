import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import VOCSegmentation


def generate_datasets(parameters={}):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_dataset = VOCSegmentation(
        root=".",
        year="2012",
        image_set="train",
        download=True,
        transform=data_transform,
        target_transform=None,
        transforms=None,
    )
    val_dataset = VOCSegmentation(
        root=".",
        year="2012",
        image_set="val",
        download=False,
        transform=data_transform,
        target_transform=None,
        transforms=None,
    )
    return train_dataset, val_dataset
