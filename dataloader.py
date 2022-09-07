import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D


def load_data(file):
    data = np.load('tiny_nerf_data.npz')

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

        # Sample pixels

        xs = torch.arange(self.w)
        ys = torch.arange(self.h)

        h_mesh, w_mesh = torch.meshgrid(xs, ys, indexing='ij')

        pixels_unflatten = torch.stack([(w_mesh - self.w * .5) / self.focal, -(
            h_mesh - self.h * .5) / self.focal, -torch.ones_like(h_mesh)], dim=-1)
        pixels = torch.reshape(pixels_unflatten, (self.h*self.w, 3))

        dirs = torch.matmul(torch.tensor(rotation), pixels.T).T
        dirs_tformed = torch.reshape(dirs, (self.h, self.w, 3))

        origin = torch.broadcast_to(torch.tensor(
            translation), dirs_tformed.shape)
        
        ts = torch.linspace(self.t_n, self.t_f, steps=self.num_samples)

        ray_points = origin[..., None, :] + \
            dirs_tformed[..., None, :] * ts[:, None]

        # ray_points is of shape [num_pixels, num_samples, 3]

        # Come back to this later, once nerf works on plain (x,y,z) values.
        # dirs = torch.unsqueeze(dirs, dim=1).repeat(1, self.num_samples, 1)
        return ray_points, torch.tensor(image)


if __name__ == '__main__':
    images, poses, focal, w, h = load_data('tiny_nerf_data.npz')

    d = TinyDataset(images, poses, focal, w, h, 3, 6, 30)
    print(d[0][0].shape)

    # rays = d[0][0].detach().cpu().numpy()
    # x, z, y = rays[:, :, 0], rays[:, :, 1], rays[:, :, 2]

    # x, z, y = np.reshape(x, -1), np.reshape(z, -1), np.reshape(y, -1)

    # fig = plt.figure(figsize=(4,4))

    # fig = plt.figure()

    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot(x,y,z)

    # plt.show()
