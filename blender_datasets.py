import torch
import torch.nn as nn
import numpy as np

import tqdm
import os
from PIL import Image
import logging 

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import json

from utils import pose_to_rays, pose_to_rays_sampled


class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, w, h, t_n, t_f, num_samples, num_rays, mode, **_):

        self.w = w
        self.h = h
        self.t_n, self.t_f = t_n, t_f
        self.num_samples = num_samples
        self.num_rays = num_rays
        self.mode = mode

        if dataset_name not in os.listdir("NeRF-Scenes/"):
            logging.error("Invalid dataset name")

        self.dataset_path = f'NeRF-Scenes/{dataset_name}/'

        if w/h != 1:
            logging.warning("Try to keep the aspect ratio square")

        if w != 800 or h != 800:
            logging.info("Resizing Training set images")

        self.transforms = transforms.Compose(
            [transforms.Resize((w, h)), transforms.ToTensor()])

        with open(f'{self.dataset_path}/transforms_{mode}.json', 'r') as f:
            config = json.load(f)
            self.frames = config["frames"]

            # The default h/w is 800x800, so use that to compute focal
            self.focal = .5 * 800 / \
                np.tan(.5 * float(config['camera_angle_x']))
            self.focal /= (800/w)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        im = self.transforms(Image.open(
            self.dataset_path + self.frames[idx]["file_path"][2:] + ".png").convert('RGB'))

        trans_mat = torch.FloatTensor(
            self.frames[idx]["transform_matrix"])

        R, t = trans_mat[:3, :3], torch.squeeze(trans_mat[:3, 3:])

        if self.mode == "train":
            rays_points, rays_dirs, rand_ray_coords = pose_to_rays_sampled(
                R, t, self.focal, self.h, self.w, self.t_n, self.t_f, self.num_samples, self.num_rays)
        
            return rays_points, rays_dirs, im[:, rand_ray_coords[:, 0], rand_ray_coords[:, 1]]
        
        elif self.mode in ["val", "test"]:
            rays_points, rays_dirs = pose_to_rays(
                R, t, self.focal, self.h, self.w, self.t_n, self.t_f, self.num_samples)
            
            return rays_points, rays_dirs, im