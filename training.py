import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import tqdm

from dataloader import TinyDataset, load_data
from blender_datasets import BlenderDataset
from model import *
from rendering import rendering
from utils import create_parser
import tqdm
import random
import json
import os


class DictMap():
    def __init__(self, item):
        for k, v in item.items():
            setattr(self, k, v)

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def train(rank, world_size, data_file, hparams):
    
    device = f'cuda:{rank}'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    # h, w = 100, 100

    # images, poses, focal, w, h = load_data(data_file)
    # dataset = TinyDataset(images, poses, focal, w, h, hparams.near, hparams.far, hparams.samples_num)
    
    dataset = BlenderDataset("lego", 100, 100, 2, 6, 128)
    
    nerf_model = ReplicateNeRFModel(use_viewdirs=True).to(device=device)
    
    if os.path.exists("model.pt"):

        nerf_model.load_state_dict(torch.load("model.pt"))
        
    nerf_model = nn.parallel.DistributedDataParallel(nerf_model, device_ids=[rank])
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=hparams.lr)
    for epoch in range(hparams.epochs):
        idx = random.randint(0, len(dataset) - 1)
        points, dirs, image = dataset[idx]

        points, dirs, image = points.to(device=device), dirs.to(
            device=device), image.to(device=device)

        
        rgbs, density = nerf_model(points, dirs)

        # rgbs, density = nerf_model(points)

        # rendering
        delta = (hparams.far - hparams.near) / hparams.samples_num
        rendered_image = rendering(rgbs, density, delta, device, permute=True)

        mse = nn.MSELoss(reduction='sum')(image, rendered_image)
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        if rank == 0:
            print('\nepoch', epoch + 1, ': ', mse.item())

            torch.save(nerf_model.module.state_dict(), "model.pt")
            
            torchvision.utils.save_image(rendered_image, "result.png")
            # axarr.imshow(rendered_image.permute(1, 2, 0).detach().cpu().numpy())
            # plt.savefig('result.png')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path) as json_file:
        hparams = json.load(json_file)

    world_size = args.world_size
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = find_free_port()

    data_file = 'tiny_nerf_data.npz'
    # train(data_file, **hparams)
    mp.spawn(train, nprocs=world_size, args=(world_size, data_file, DictMap(hparams)))
