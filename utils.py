import torch


def pose_to_rays(rotation, translation, focal, h, w, t_n, t_f, num_samples):
    xs = torch.arange(w)
    ys = torch.arange(h)

    h_mesh, w_mesh = torch.meshgrid(xs, ys, indexing='ij')

    pixels_unflatten = torch.stack([(w_mesh - w * .5) / focal, -(
        h_mesh - h * .5) / focal, -torch.ones_like(h_mesh)], dim=-1)
    pixels = torch.reshape(pixels_unflatten, (h*w, 3))

    dirs = torch.matmul(torch.tensor(rotation), pixels.T).T
    dirs_tformed = torch.reshape(dirs, (h, w, 3))

    origin = torch.broadcast_to(torch.tensor(
        translation), dirs_tformed.shape)

    ts = torch.linspace(t_n, t_f, steps=num_samples)

    ray_points = origin[..., None, :] + \
        dirs_tformed[..., None, :] * ts[:, None]

    # ray_points is of shape [num_pixels, num_samples, 3]

    # Come back to this later, once nerf works on plain (x,y,z) values.
    # dirs = torch.unsqueeze(dirs, dim=1).repeat(1, num_samples, 1)
    return ray_points
