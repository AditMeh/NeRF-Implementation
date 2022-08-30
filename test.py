import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from dataloader import TinyDataset, load_data
from model import NerfModel


if __name__ == "__main__":

    images, poses, focal, w, h = load_data('tiny_nerf_data.npz')

    idx = 0
    image =images[idx]
    pose = poses[idx]
    rotation = pose[0:3, 0:3]
    translation = pose[0:3, 3]

    dataset = TinyDataset(images, poses, focal, w, h, 3, 6, 100)

    points, directions, rgbs = dataset[idx]

    print(points.shape, directions.shape, rgbs.shape)

    nerf_model = NerfModel(3, 3)

    points = points.reshape(points.shape[0] * points.shape[1], 3)
    directions = points.reshape(directions.shape[0] * directions.shape[1], 3)
    print(points.shape, directions.shape)

    print(nerf_model(points, directions)[0].shape, 
    nerf_model(points, directions)[1].shape)