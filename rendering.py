import torch

def rendering(color, density, dist_delta):
    """
    color: (# of rays, # of sample points of each ray, 3)
    density: (# of rays, # of sample points of each ray, 1)
    dist_delta: the distance between the two neighbouring points on a ray, 
                assume it's constant and a float
    """
    density_times_delta = density * dist_delta

    T = torch.exp(-torch.cumsum(density_times_delta, dim=1))

    S = 1 - torch.exp(-density_times_delta)

    T = T.expand(color.shape)

    points_color = T * S * color

    C = torch.sum(points_color, dim=1)

    return C