import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from dataloader import TinyDataset, load_data
from blender_datasets import BlenderDataset
from model import NerfModel, ReplicateNeRFModel
from rendering import rendering
from utils import create_parser
import tqdm
import random
import json


def train(data_file, near, far, freq_num, samples_num, epochs, lr):
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    # h, w = 100, 100
    # dataset = BlenderDataset("lego", 100, 100, 2, 6, 128)

    images, poses, focal, w, h = load_data(data_file)
    dataset = TinyDataset(images, poses, focal, w, h, near, far, samples_num)

    nerf_model = ReplicateNeRFModel().to(device=device)

    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(epochs)):
        idx = random.randint(0, len(dataset) - 1)
        points, dirs, image = dataset[idx]

        points, dirs, image = points.to(device=device), dirs.to(
            device=device), image.to(device=device)

        rgbs, density = nerf_model(points, dirs)

        # rendering
        delta = (far - near) / samples_num
        rendered_image = rendering(rgbs, density, delta, device, permute=True)


        mse = nn.MSELoss(reduction='sum')(image, rendered_image)
        print('\nepoch', epoch + 1, ': ', mse.item())
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        torch.save(nerf_model, "model.pt")
        f, axarr = plt.subplots(1, 1)

        axarr.imshow(rendered_image.permute(1,2,0).detach().cpu().numpy())
        plt.savefig('result.png')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path) as json_file:
        hparams = json.load(json_file)

    data_file = 'tiny_nerf_data.npz'
    train(data_file, **hparams)
