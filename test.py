from logging import LogRecord
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
from rendering import rendering
import tqdm


if __name__ == "__main__":

    images, poses, focal, w, h = load_data('tiny_nerf_data.npz')

    print(images.shape)
    print(images[:100, ...].shape)
    print(images[:100, :, :, :3].shape)

    idx = 0
    image =images[idx]
    pose = poses[idx]
    rotation = pose[0:3, 0:3]
    translation = pose[0:3, 3]

    near = 3
    far = 6
    samples_num = 30

    dataset = TinyDataset(images, poses, focal, w, h, near, far, samples_num)

    points, directions, rgbs = dataset[idx]

    print(points.shape, directions.shape, rgbs.shape)

    nerf_model = NerfModel(3, 3)

    points = points.reshape(points.shape[0] * points.shape[1], 3)
    directions = points.reshape(directions.shape[0] * directions.shape[1], 3)
    print(points.shape, directions.shape)

    rgbs, density = nerf_model(points, directions)

    print('rgb:', rgbs.shape, '\n density:', density.shape)

    # reshape
    total_points = rgbs.shape[0]
    pixels_num = int(total_points / samples_num)
    rgbs = torch.reshape(rgbs, (pixels_num, samples_num, 3))
    density = torch.reshape(rgbs, (pixels_num, samples_num, 3))

    # rendering
    delta = (far - near) / samples_num
    C = rendering(rgbs, density, delta)
    print('C:', C.shape)

    # f, axarr = plt.subplots(1,1)

    # axarr.imshow(torch.reshape(rgbs, (h, w, 3)).detach().numpy())
    # plt.show()

    epochs = 10
    lr = 1e-3
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(epochs)):
        rgbs, density = nerf_model(points, directions)
        total_points = rgbs.shape[0]
        pixels_num = int(total_points / samples_num)
        rgbs = torch.reshape(rgbs, (pixels_num, samples_num, 3))
        density = torch.reshape(rgbs, (pixels_num, samples_num, 3))

        # rendering
        delta = (far - near) / samples_num
        C = rendering(rgbs, density, delta)

        rendered_img = torch.reshape(C, (h, w, 3))

        mse = nn.MSELoss(reduction='sum')(torch.tensor(image), rendered_img)
        print('\nepoch', epoch + 1, ': ', mse)
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

    # f, axarr = plt.subplots(1,1)

    # axarr.imshow(torch.reshape(C, (h, w, 3)).detach().numpy())
    # plt.show()
