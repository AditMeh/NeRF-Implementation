from cmath import sin
import torch
from torch import nn



class NerfModel(nn.Module):
    def __init__(self, freq_num):
        super(NerfModel, self).__init__()
        """
        position_dim: The dimension of the last axis of the points
        """
        self.position_dim = 3 + 3 * 2 * freq_num
        self.freq_num = freq_num

        self.model_1 = nn.Sequential(
            nn.Linear(self.position_dim, 256),
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
            nn.Linear(256 + self.position_dim, 256),
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

        flat_pos = position.reshape(-1, 3)
        flat_pos = self.positional_encoding(flat_pos)

        intermediate_rep = self.model_1(flat_pos)

        concat_pos = torch.cat([intermediate_rep, flat_pos], dim=1)

        intermediate_rep = self.model_2(concat_pos)

        density = self.density_head(intermediate_rep)

        rgb = self.rgb_head(intermediate_rep)

        return torch.reshape(rgb, position.shape), torch.reshape(density, position.shape[0:-1])

    def positional_encoding(self, position):
        terms = [position]
        for i in range(self.freq_num):
            sin_encoding = torch.sin(2 ** i * torch.pi * position)
            cos_encoding = torch.cos(2 ** i * torch.pi * position)
            terms.append(sin_encoding)
            terms.append(cos_encoding)

        return torch.concat(terms, dim=1)

class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_freqs_xyz=8,
        num_freqs_dir=6,
        use_viewdirs=True
    ):
        super(ReplicateNeRFModel, self).__init__()
        
        self.use_viewdirs = use_viewdirs
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        self.num_freqs_xyz = num_freqs_xyz
        self.num_freqs_dir = num_freqs_dir

        self.dim_xyz = 3 + 2 * 3 * num_freqs_xyz
        self.dim_dir = 3 + 2 * 3 * num_freqs_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        if use_viewdirs == True:
            self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        else:
            self.layer4 = torch.nn.Linear(hidden_size, hidden_size // 2)
        
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, position, dir=None):
        
        if len(position.shape) < 3:
            position = position.reshape(-1, 3)

        xyz = self.positional_encoding(position, self.num_freqs_xyz)
        
        if self.use_viewdirs:
            if len(dir.shape) < 3:  
                dir = dir.reshape(-1, 3)
            direction = self.positional_encoding(dir, self.num_freqs_dir)
        
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = torch.nn.ReLU()(self.fc_alpha(x_))
        
        if self.use_viewdirs:
            y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        else:
            y_ = self.relu(self.layer4(feat))

        y_ = self.relu(self.layer5(y_))
        rgb = torch.nn.Sigmoid()(self.fc_rgb(y_))

        return torch.reshape(rgb, position.shape), torch.reshape(alpha, position.shape[0:-1]).unsqueeze(-1)

    @staticmethod
    def positional_encoding(tensor, num_encoding_functions=6):
        encoding = [tensor]
        
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,    
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        return torch.cat(encoding, dim=-1) 

if __name__ == "__main__":

    nerf_model = NerfModel(6)

    points = torch.ones(size=(100, 100, 100, 3))

    print(nerf_model(points)[0].shape,
          nerf_model(points)[1].shape)
