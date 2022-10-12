import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from model import NerfModel
from rendering import rendering
from utils import pose_to_rays, create_parser
from dataloader import load_data
import tqdm

import math
import imageio
import json

def get_rotation_and_translation(c2w):
    return c2w[:3, :3], c2w[:3, -1]


def show_view(c2w, focal, h, w, near, far, samples_num, **_):
    rot, trans = get_rotation_and_translation(c2w)
    points = pose_to_rays(rot, trans, focal, h, w, near, far, samples_num)

    points = points.to(device=device)

    rgbs, density = nerf_model(points)
    delta = (far - near) / samples_num
    C = rendering(rgbs, density, delta, device)
    rendered_img = torch.reshape(C, (h, w, 3))

    return rendered_img.detach().cpu().numpy()


def lookat(origin, loc):
    dir = loc - origin
    dir = dir / np.linalg.norm(dir)

    tmp = np.asarray([0, 0, 1])
    right = np.cross(tmp, dir)
    up = np.cross(dir, right)

    R = np.hstack([right[..., None], up[..., None], dir[..., None]])

    return np.vstack(
        [np.hstack([R, loc[..., None]]),
         np.asarray([0, 0, 0, 1])[None, ...]])


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

    images, poses, focal, w, h = load_data('tiny_nerf_data.npz')
    nerf_model = torch.load("model.pt").to(device=device)

    imgs = []
    camera_positions = circle_points(2, 5, 110)
    for position in camera_positions:
        c2w = lookat(np.asarray([0, 0, 0]), position).astype(np.float32)
        imgs.append(
            (255 *
             np.clip(show_view(c2w, focal, h, w, **hparams), 0, 1)).astype(
                 np.uint8))

    f = 'video.mp4'
    imageio.mimwrite(f, imgs, fps=30, quality=7)
