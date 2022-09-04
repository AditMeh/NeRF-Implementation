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
import random


def train(data_file, near, far, freq_num, samples_num, epochs, lr):
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    images, poses, focal, w, h = load_data(data_file)

    dataset = TinyDataset(images, poses, focal, w, h, near, far, samples_num)

    nerf_model = NerfModel(freq_num).to(device=device)

    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(epochs)):
        idx = random.randint(0, len(dataset) - 1)
        points, image = dataset[idx]

        points, image = points.to(device=device), image.to(device=device)

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
    f, axarr = plt.subplots(1, 1)

    axarr.imshow(rendered_img.detach().cpu().numpy())
    plt.savefig('result.png')


if __name__ == "__main__":
    near = 2
    far = 6
    freq_num = 6
    samples_num = 64
    epochs = 30
    lr = 5e-4

    data_file = 'tiny_nerf_data.npz'
