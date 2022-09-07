import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from utils import pose_to_rays


def load_data(file):
    data = np.load(file)

    images = data['images']
    poses = data['poses']
    focal = data['focal']

    w, h = images.shape[1:3]

    return images, poses, focal, w, h


class TinyDataset(Dataset):
    def __init__(self, images, poses, focal, w, h, t_n, t_f, num_samples):
        self.images = images
        self.poses = poses
        self.focal = focal
        self.w, self.h = w, h
        self.t_n, self.t_f = t_n, t_f
        self.num_samples = num_samples

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """"Generate samples along the ray through each pixel on the image[idx]."""
        image = self.images[idx]
        pose = self.poses[idx]
        rotation = pose[0:3, 0:3]
        translation = pose[0:3, 3]

        ray_points = pose_to_rays(rotation, translation, self.focal,
                                  self.h, self.w, self.t_n, self.t_f, self.num_samples)
        return ray_points, torch.tensor(image)


if __name__ == '__main__':
    images, poses, focal, w, h = load_data('tiny_nerf_data.npz')

    d = TinyDataset(images, poses, focal, w, h, 3, 6, 30)
    print(d[0][0].shape)
