import torch

def rendering(color, density, dist_delta):
    """
    color: (h, w, num_samples along each ray, 3)
    density: (h, w, num_samples along each ray)
    dist_delta: the distance between the two neighbouring points on a ray, 
                assume it's constant and a float
    """
    dists = torch.ones(density.shape[-1]) * dist_delta
    dists[-1] = 1e10

    density_times_delta = density * dist_delta

    S = 1. - torch.exp(-density_times_delta)

    T = cumprod_exclusive(torch.exp(-density_times_delta))


    points_color = (T * S)[..., None] * color

    C = torch.sum(points_color, dim=-2)


    # alpha = 1. - torch.exp(-density_times_delta) 
    # weights = alpha * cumprod_exclusive(1.-alpha)
    # rgb_map = torch.sum(weights[...,None] * color, dim=-2)

    # print(torch.equal(C, rgb_map))

    # c_list = list(torch.flatten(C).detach().numpy())
    # r_list = list(torch.flatten(rgb_map).detach().numpy())

    # for i, ele in enumerate(c_list):
    #     if round(r_list[i], 5) != round(ele, 5):
    #         print(r_list[i], ele)
    return C


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
    c = torch.clip(torch.randn((100, 100, 30, 3)), min = 0.01, max = 0.99)
    d = torch.clip(torch.randn((100, 100, 30)), min = 0.01)


    print(rendering(c, d, 4.).shape)
