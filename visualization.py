import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from model import NerfModel, ReplicateNeRFModel
from rendering import rendering
from utils import pose_to_rays, create_parser, DictMap
from dataloader import load_data
import tqdm

from blender_datasets import BlenderDataset

import math
import imageio
import json
import cv2


def get_rotation_and_translation(c2w):
    return c2w[:3, :3], c2w[:3, -1]


def show_view(c2w, focal, h, w, t_n, t_f, num_samples, chunk, **_):
    with torch.no_grad():
        rot, trans = get_rotation_and_translation(c2w)
        points, direction, ts = pose_to_rays(
            rot, trans, focal, h, w, t_n, t_f, num_samples)

        points, dirs, ts= points.to(device), direction.to(device), ts.to(device)
                    
        flat_points = points.reshape(-1, 3)
        flat_dirs = dirs.reshape(-1, 3)
        
        concat = batchify(chunk, nerf_model)(flat_points, flat_dirs)
        flat_rgbs, flat_density = concat[..., :3], concat[..., 3:]

        rgbs, density = torch.reshape(flat_rgbs, points.shape), torch.reshape(flat_density, points.shape[0:-1])

        # rendering
        # delta = (hparams.t_f - hparams.t_n) / hparams.num_samples
            
        # Change delta to be actual adjacent points
        delta = ts.roll(shifts=-1,dims=0) - ts

        rendered_image = rendering(rgbs, density, delta, device, permute=False)

    return rendered_image.detach().cpu().numpy()


def lookat(origin, loc):
    dir = loc - origin
    dir = dir / np.linalg.norm(dir)

    tmp = np.asarray([0, 0, 1])
    right = np.cross(tmp, dir)
    up = np.cross(dir, right)

    R = np.hstack([right[..., None], up[..., None], dir[..., None]])

    return torch.tensor(np.vstack(
        [np.hstack([R, loc[..., None]]),
         np.asarray([0, 0, 0, 1])[None, ...]]))


def circle_points(z, radius, num_points):
    split = (2 * math.pi) / num_points

    vals = []
    for i in range(num_points):
        angle = split * i
        vals.append(
            np.asarray([radius * math.cos(angle), radius * math.sin(angle),
                        z]))
    return vals

def batchify(chunk, mlp):
    if chunk is None:
        return self.mlp

    def process_chunks(xyz, dirs):
        assert len(xyz.shape) == len(dirs.shape)
        assert [xyz.shape[i] == dirs.shape[i] for i in range(len(xyz.shape))]

        return torch.cat([torch.cat(mlp(xyz[i:i+chunk], dirs[i:i+chunk]), dim=-1) for i in range(0, xyz.shape[0], chunk)], 0)
    return process_chunks

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path) as json_file:
        hparams = json.load(json_file)

    hparams = DictMap(hparams)

    device = (torch.device('cuda')
              if torch.cuda.is_available() else torch.device('cpu'))

    # images, poses, focal, w, h = load_data('tiny_nerf_data.npz')
    dataset = BlenderDataset(mode="train", **hparams.__dict__)
    focal, w, h = dataset.focal, 100, 100

    nerf_model = ReplicateNeRFModel(use_viewdirs=hparams.use_viewdirs
                                    ).to(device=device)

    nerf_model.load_state_dict(torch.load(
        f'model_{hparams.w}x{hparams.h}_viewdir={str(hparams.use_viewdirs)}.pt', map_location=device))

    imgs = []
    camera_positions = circle_points(2, 5, 100)
    for position in tqdm.tqdm(camera_positions):
        c2w = lookat(np.asarray([0, 0, 0]), position).type(torch.float32)
        intermediate = (255 * np.clip(show_view(c2w, focal, **
                        hparams.__dict__), 0, 1)).astype(np.uint8)

        # imgs.append(cv2.resize(intermediate, (512, 512)
        imgs.append(intermediate)
        
    f = f'video_{hparams.use_viewdirs}.mp4'
    imageio.mimwrite(f, imgs, fps=30, quality=7)
