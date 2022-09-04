from torch import nn
import torch


class NerfModel(nn.Module):
    def __init__(self, freqs):
        super(NerfModel, self).__init__()
        """
        position_dim: The dimension of the last axis of the points
        direction_dim: The dimension of the last axis of the direction
        """
        self.freqs = freqs

        self.posenc_dim = 3 + 3*2*freqs

        self.model_1 = nn.Sequential(
            nn.Linear(self.posenc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # concatenate with the position vector
        self.model_2 = nn.Sequential(
            nn.Linear(256 + self.posenc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # output density value
        self.density_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )

        # output RGB value
        self.rgb_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, position):
        position_flat = torch.reshape(position, (-1, 3))
        position_flat = self.posenc(position_flat)

        intermediate_rep = self.model_1(position_flat)

        concat_pos = torch.cat([intermediate_rep, position_flat], dim=1)

        intermediate_rep = self.model_2(concat_pos)

        density = self.density_head(intermediate_rep)

        rgb = self.rgb_head(intermediate_rep)

        return torch.reshape(rgb, position.shape), torch.reshape(density, position.shape[:-1])
    def posenc(self, t):
        terms = [t]
        for i in range(self.freqs):
            for fn in [torch.sin, torch.cos]:
                terms.append(fn(2.**i * t))
        return torch.concat(terms, axis = -1)

if __name__ == "__main__":

    nerf_model = NerfModel(6)

    points = torch.ones(size=(50, 50, 64, 3))

    print(nerf_model(points)[0].shape,
          nerf_model(points)[1].shape)
