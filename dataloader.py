import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def load_data(file):
    data = np.load('tiny_nerf_data.npz')

    images = data['images']
    poses = data['poses']
    focal = data['focal']

    w, h = images.shape[1:3]

    return images, poses, focal, w, h


def dataloader(images, poses, focal, w, h):
    raise NotImplementedError


class TinyDataset(Dataset):
    def __init__(self, images, poses, focal, w, h, t_n, t_f, num_samples):
        self.images = images
        self.poses = poses
        self.focal = focal
        self.w, self.h = w, h
        self.t_n, self.t_f = t_n, t_f
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        """"Generate samples along the ray through each pixel on the image[idx]."""
        image = self.images[idx]
        pose = self.poses[idx]
        rotation = pose[0:3, 0:3]
        translation = pose[0:3, 3]

        # Sample pixels

        xs = torch.linspace(-self.w//2 + 1, self.w//2, steps=self.w)
        ys = torch.linspace(-self.h//2 + 1, self.h//2, steps=self.h)
        h_mesh, w_mesh = torch.meshgrid(xs, ys, indexing='ij')

        pixels_unflatten = torch.stack(
            [h_mesh / self.focal, -w_mesh / self.focal, -torch.ones_like(h_mesh)], dim=-1)
        pixels = torch.reshape(pixels_unflatten, (self.h*self.w, 3))

        dirs = torch.matmul(torch.tensor(rotation), pixels.T).T

        dirs_unflattened = torch.reshape(pixels_unflatten, (self.h, self.w, 3))

        origin = torch.broadcast_to(torch.tensor(
            translation), dirs_unflattened.shape)

        ts = torch.linspace(self.t_n, self.t_f, steps=self.num_samples)

        #                      (100, 100, 1, 3)       (100, 100, 1, 3)     (100, 1)
        ray_points = origin[..., None, :] + \
            dirs_unflattened[..., None, :] * ts[:, None]

        # ray_points is of shape [num_pixels, num_samples, 3]

        # Come back to this later, once nerf works on plain (x,y,z) values.
        # dirs = torch.unsqueeze(dirs, dim=1).repeat(1, self.num_samples, 1)
        return ray_points, torch.tensor(image)


if __name__ == '__main__':
    images, poses, focal, w, h = load_data('tiny_nerf_data.npz')

    d = TinyDataset(images, poses, focal, w, h, 3, 6, 100)

    print(d[0].shape)
    # idx = 0
    # image =images[idx]
    # pose = poses[idx]
    # rotation = pose[0:3, 0:3]
    # translation = pose[0:3, 3]

    # # Sample pixels
    # xs = torch.linspace(-w//2 + 1, w//2, steps=w)
    # ys = torch.linspace(-h//2 + 1, h//2, steps=h)
    # h_mesh, w_mesh = torch.meshgrid(xs, ys)

    # z_mesh = -torch.ones_like(h_mesh) * focal

    # pixels = torch.stack([h_mesh, w_mesh, z_mesh], dim=-1)

    # """
    # data = [[x_1, y_1, z_1],
    #        [x_2, y_2, z_2],
    #         ...
    #        ]

    # data = [x_1^T,
    #         x_2^T,
    #         ...]
    # rotation = [[r_1, r_2 r_3],
    #             [r_4, r_5, r_6],
    #             [r_7, r_8, r_9]]

    #   [[r_1, r_2 r_3],  [[x_1],
    #   [r_4, r_5, r_6],  [x_2],
    #   [r_7, r_8, r_9]]  [x_3]]
    # """

    # pixels = torch.reshape(pixels, (h*w, 3))
    # rgbs = torch.reshape(torch.tensor(image), (h*w, 3))

    # dirs = torch.matmul(torch.tensor(rotation), pixels.T).T
    # dirs = torch.nn.functional.normalize(dirs, dim=1)

    # origin = torch.tensor(translation)

    # t_n, t_f = 2, 6
    # num_samples = 100
    # ts = torch.linspace(t_n, t_f, steps=num_samples)
    # print(ts)

    # print(torch.stack([origin + dirs*t for t in ts], axis = 0).permute(1, 0, 2).shape)
    # print(rgbs.shape)
    # # (10000, 100, 3) -> (10000 * 100, 3)
    # # (10000*100, 3) -> (10000*100, 4) -> (10000, 100, 4) -> (10000, 3)
    # # run on N images -> wait, don't update gradients -> update after N are done
    # # Dataloader will give us the rays for one image at a time -> run on network -> do above

    # # optimizer.step() updates gradients, we don't want to do this until it is time
    # # more images means better approximation of true gradient

    # # Get image -> compute rendered image -> loss -> repeat and at the end do optimizer.step()
    # # (B ,)
    # # N images, 10000 pixels
    # # (N*10000, 100, 3)
    # # (B, 6)
    # # N*10000/32 -> ((32, 100), 3) generate the estimated pixel value
