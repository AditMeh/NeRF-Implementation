import torch
import torch.distributed as dist

def rendering(color, density, dist_delta, device, permute, rank = None):
    """
    color: (h, w, num_samples along each ray, 3)
    density: (h, w, num_samples along each ray)
    dist_delta: the distance between the two neighbouring points on a ray, 
                assume it's constant and a float
    """
    
    if len(density.shape) == 3:
        density = torch.squeeze(density)

    dist_delta[-1] = 1e10    
    density_times_delta = density * dist_delta    
    density_times_delta = density_times_delta.to(device=device)
    


    T = torch.exp(-cumsum_exclusive(density_times_delta))
    
    # roll T to right by one postion and replace the first column with 1
    S = 1 - torch.exp(-density_times_delta)
    points_color = (T * S)[..., None] * color
    C = torch.sum(points_color, dim=-2)

    if len(C.shape) == 2:
        return C.permute(1, 0) if permute else C
    elif len(C.shape) == 3:
        return C.permute(2, 0, 1) if permute else C


def cumsum_exclusive(t):
    dim = -1
    cumsum = torch.cumsum(t, dim)
    cumsum = torch.roll(cumsum, 1, dim)
    cumsum[..., 0] = 0.
    return cumsum

def cumprod_exclusive(t):
    dim = -1
    cumprod = torch.cumprod(t, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.
    return cumprod


if __name__ == "__main__":
    c = torch.clip(torch.randn((100, 100, 64, 3)), min=0.01, max=0.99)
    d = torch.clip(torch.randn((100, 100, 64)), min=0.01)
    dist = 100

    res = rendering(c, d, dist, "cpu")
    print(torch.unique(res))
    print(res.shape)