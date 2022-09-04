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

    near = 2
    far = 6
    samples_num = 64

    dataset = TinyDataset(images, poses, focal, w, h, near, far, samples_num)

    points = dataset[idx]

    print(points.shape)

    nerf_model = NerfModel(6)

    # f, axarr = plt.subplots(1,1)

    # axarr.imshow(torch.reshape(rgbs, (h, w, 3)).detach().numpy())
    # plt.show()

    epochs = 30
    lr = 5e-4
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(epochs)):
        rgbs, density = nerf_model(points)
        # rendering
        delta = (far - near) / samples_num
        C = rendering(rgbs, density, delta)

        rendered_img = torch.reshape(C, (h, w, 3))

        mse = nn.MSELoss(reduction='sum')(torch.tensor(image), rendered_img)
        print('\nepoch', epoch + 1, ': ', mse)
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        print('rendered: ', torch.min(rendered_img), torch.max(rendered_img))

    torch.save(nerf_model, "model.pt")
    f, axarr = plt.subplots(1,1)

    axarr.imshow(torch.reshape(C, (h, w, 3)).detach().numpy())
    plt.show()
