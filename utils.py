import torch
import argparse



def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", help="path to NeRF hparam config", type=str)
    parser.add_argument(
        "--world_size", help="number of gpus", default=0, type=int)
    return parser

class DictMap():
    def __init__(self, item):
        for k, v in item.items():
            setattr(self, k, v)

def sample_ts(t_n, t_f, num_samples):
    accum = []
    for i in range(1, num_samples + 1):
        lower = t_n + ((i - 1) / num_samples) * (t_f - t_n)
        upper = t_n + (i / num_samples) * (t_f - t_n)

        t_i = (lower - upper) * torch.rand(1) + upper

        accum.append(t_i)
    return torch.hstack(accum)


def pose_to_rays(rotation, translation, focal, h, w, t_n, t_f, num_samples):
    xs = torch.arange(w)
    ys = torch.arange(h)

    h_mesh, w_mesh = torch.meshgrid(xs, ys, indexing='ij')

    pixels_unflatten = torch.stack([(w_mesh - w * .5) / focal, -(
        h_mesh - h * .5) / focal, -torch.ones_like(h_mesh)], dim=-1)
    pixels = torch.reshape(pixels_unflatten, (h*w, 3))

    dirs = torch.matmul(rotation, pixels.T).T
    dirs_tformed = torch.reshape(dirs, (h, w, 3))

    origin = torch.broadcast_to(translation, dirs_tformed.shape)

    # ts = torch.linspace(t_n, t_f, steps=num_samples)
    ts = sample_ts(t_n, t_f, num_samples)
    ray_points = origin[..., None, :] + \
        dirs_tformed[..., None, :] * ts[:, None]

    return ray_points, dirs_tformed[..., None, :].repeat(1, 1, num_samples, 1)
