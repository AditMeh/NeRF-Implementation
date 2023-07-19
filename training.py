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

def eval(nerf_model, hparams, rank, world_size):
    # Distributed eval
    with torch.no_grad():
        device = f'cuda:{rank}'
        loss_accum = 0
        val_dataset = BlenderDataset(mode="val", **(hparams.__dict__))
        
        # Fetch idx from the sampler for each rank
        val_sampler = list(iter(torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)))

        for _, idx in enumerate(val_sampler):
            points, dirs, image = val_dataset[idx]
            
            points, dirs, image = points.to(device), dirs.to(device), image.to(device)
            
            rgbs, density = nerf_model(points, dirs)
            
            # rendering
            delta = (hparams.t_f - hparams.t_n) / hparams.num_samples
            rendered_image = rendering(rgbs, density, delta, device, permute=True)

            mse = nn.MSELoss(reduction='sum')(image, rendered_image)
            loss_accum += mse
        return loss_accum
    
        

def train(rank, world_size, hparams):
    
    device = f'cuda:{rank}'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    
    dataset = BlenderDataset(mode="train", **(hparams.__dict__))
    
    nerf_model = ReplicateNeRFModel(use_viewdirs=hparams.use_viewdirs).to(device=device)
    
    if os.path.exists(f'model_{str(hparams.use_viewdirs)}.pt'):

        nerf_model.load_state_dict(torch.load(f'model_{w}x{h}_viewdir={str(hparams.use_viewdirs)}.pt'))
        
    nerf_model = nn.parallel.DistributedDataParallel(nerf_model, device_ids=[rank])
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=hparams.lr)
    for epoch in range(1, hparams.epochs + 1):
        idx = random.randint(0, len(dataset) - 1)
        points, dirs, image = dataset[idx]

        points, dirs, image = points.to(device=device), dirs.to(
            device=device), image.to(device=device)

        
        rgbs, density = nerf_model(points, dirs)

        # rgbs, density = nerf_model(points)

        # rendering
        delta = (hparams.t_f - hparams.t_n) / hparams.num_samples
        rendered_image = rendering(rgbs, density, delta, device, permute=True)

        mse = nn.MSELoss(reduction='sum')(image, rendered_image)
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()


        if (epoch % hparams.log_every) == 0:            
            
            val_loss = eval(nerf_model, hparams, rank, world_size)

            dist.reduce(val_loss, 0, op= torch.distributed.ReduceOp.AVG)

            dist.barrier() 
            
            if rank == 0:
                print(f'Epoch {epoch}, Val Loss {val_loss.item()}')
                torch.save(nerf_model.module.state_dict(), f'model_{w}x{h}_viewdir={str(hparams.use_viewdirs)}.pt')    
                torchvision.utils.save_image(rendered_image, "result.png")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path) as json_file:
        hparams = json.load(json_file)

    world_size = args.world_size
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = find_free_port()


    mp.spawn(train, nprocs=world_size, args=(world_size, DictMap(hparams)))
