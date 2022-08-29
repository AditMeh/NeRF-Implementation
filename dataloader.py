from re import M, X
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
    def __init__(self, images, poses, focal, w, h, t_n, t_f):
        self.images = images
        self.poses = poses
        self.focal = focal
        self.w, self.h = w, h
        self.t_n, self.t_f = t_n, t_f

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        rotation = pose[0:3, 0:3]
        translation = pose[0:3, 3]

        # Sample pixels

        xs = torch.linspace(-self.w/2, self.w/2, steps=self.w + 1)
        ys = torch.linspace(-self.h/2, self.h/2, steps =self.h + 1)
        W_mesh, H_mesh = torch.meshgrid(xs, ys)

        print(W_mesh)
        print(H_mesh)
        image = ToTensor()(image)
        return image


if __name__ == '__main__':
    images, poses, focal, w, h = load_data('tiny_nerf_data.npz')

    idx = 0
    image =images[idx]
    pose = poses[idx]
    rotation = pose[0:3, 0:3]
    translation = pose[0:3, 3]

    # Sample pixels
    print(w, h)
    xs = torch.linspace(-w//2 + 1, w//2, steps=w)
    ys = torch.linspace(-h//2 + 1, h//2, steps=h)
    h_mesh, w_mesh = torch.meshgrid(xs, ys)

    
    z_mesh = -torch.ones_like(h_mesh) * focal
    

    # print(h_mesh)
    # print("-----------------------------------------------------------")
    # print(w_mesh)
    # print("-----------------------------------------------------------")
    # print(z_mesh)

    pixels = torch.stack([h_mesh, w_mesh, z_mesh], dim=-1)
    
    """
    data = [[x_1, y_1, z_1],
           [x_2, y_2, z_2],
            ...
           ]

    data = [x_1^T,
            x_2^T,
            ...]
    rotation = [[r_1, r_2 r_3],
                [r_4, r_5, r_6],
                [r_7, r_8, r_9]]

      [[r_1, r_2 r_3],  [[x_1],
      [r_4, r_5, r_6],  [x_2],
      [r_7, r_8, r_9]]  [x_3]]
    """

    pixels = torch.reshape(pixels, (h*w, 3))
    rgbs = torch.reshape(torch.tensor(image), (h*w, 3))


    dirs = torch.matmul(torch.tensor(rotation), pixels.T).T
    dirs = torch.nn.functional.normalize(dirs, dim=1)

    origin = torch.tensor(translation)

    t_n, t_f = 2, 6
    num_samples = 100
    ts = torch.linspace(t_n, t_f, steps=num_samples)
    print(ts)

    print(torch.stack([origin + dirs*t for t in ts], axis = 0).permute(1, 0, 2).shape)