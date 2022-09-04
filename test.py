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

    idx = 0
    image =images[idx]
    pose = poses[idx]
    rotation = pose[0:3, 0:3]
    translation = pose[0:3, 3]

    near = 3
    far = 6
    samples_num = 30

    dataset = TinyDataset(images, poses, focal, w, h, near, far, samples_num)

    points = dataset[idx]

    nerf_model = NerfModel(3)

    epochs = 40
    lr = 1e-3
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(epochs)):
        rgbs, density = nerf_model(points)
        total_points = rgbs.shape[0]
        pixels_num = int(total_points / samples_num)

        # rendering
        delta = (far - near) / samples_num
        C = rendering(rgbs, density, delta)

        rendered_img = torch.reshape(C, (h, w, 3))

        mse = nn.MSELoss(reduction='sum')(torch.tensor(image), rendered_img)
        print('\nepoch', epoch + 1, ': ', mse)
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

    f, axarr = plt.subplots(1,1)

    axarr.imshow(torch.reshape(C, (h, w, 3)).detach().numpy())
    plt.show()
