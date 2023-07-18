import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from model import NerfModel, ReplicateNeRFModel
from rendering import rendering
from utils import pose_to_rays, create_parser
from dataloader import load_data
import tqdm

from blender_datasets import BlenderDataset

import math
import imageio
import json
import cv2


def get_rotation_and_translation(c2w):
    return c2w[:3, :3], c2w[:3, -1]


def show_view(c2w, focal, h, w, near, far, samples_num, **_):
    rot, trans = get_rotation_and_translation(c2w)
    points, direction = pose_to_rays(
        rot, trans, focal, h, w, near, far, samples_num)

    points, direction = points.to(device=device), direction.to(device)

    rgbs, density = nerf_model(points, direction)
    delta = (far - near) / samples_num
    C = rendering(rgbs, density, delta, device, permute=False)

    return C.detach().cpu().numpy()


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


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path) as json_file:
        hparams = json.load(json_file)

    device = (torch.device('cuda')
              if torch.cuda.is_available() else torch.device('cpu'))

    # images, poses, focal, w, h = load_data('tiny_nerf_data.npz')
    dataset = BlenderDataset("lego", 100, 100, 2, 6, 128)
    focal, w, h = dataset.focal, 100, 100

    nerf_model = ReplicateNeRFModel(use_viewdirs=True
                                    ).to(device=device)
    
    nerf_model.load_state_dict(torch.load("model.pt", map_location=device))


    imgs = []
    camera_positions = circle_points(2, 5, 200)
    for position in tqdm.tqdm(camera_positions):
        c2w = lookat(np.asarray([0, 0, 0]), position).type(torch.float32)
        intermediate = (255 * np.clip(show_view(c2w, focal, h,
                        w, **hparams), 0, 1)).astype(np.uint8)

        imgs.append(cv2.resize(intermediate, (512, 512)))

    f = 'video.mp4'
    imageio.mimwrite(f, imgs, fps=30, quality=7)
