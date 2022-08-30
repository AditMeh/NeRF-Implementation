from torch import nn
import torch

class NerfModel(nn.Module):
    def __init__(self, position_dim, direction_dim):
        super(NerfModel, self).__init__()
        """
        position_dim: The dimension of the last axis of the points
        direction_dim: The dimension of the last axis of the direction
        """
        self.position_dim = position_dim
        self.direction_dim = direction_dim

        self.model_1 = nn.Sequential(
            nn.Linear(position_dim, 256),
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
            nn.Linear(256 + position_dim, 256),
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
        # Concatenate
        # get the feature vector to concatenate with the direction vector
        self.pre_rgb = nn.Linear(256,256)

        # output RGB value
        self.rgb_head = nn.Sequential(
            nn.Linear(256 + direction_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        
    def forward(self, position, direction):
        
        intermediate_rep = self.model_1(position)

        concat_pos = torch.cat([intermediate_rep, position], dim=1)

        intermediate_rep = self.model_2(concat_pos)

        density = self.density_head(intermediate_rep)

        intermediate_rep = torch.cat([self.pre_rgb(intermediate_rep), direction], dim=1)

        rgb = self.rgb_head(intermediate_rep)

        return rgb, density


    
if __name__ == "__main__":

    nerf_model = NerfModel(3, 3)

    points = torch.ones(size= (10000, 3))

    direction_vectors = torch.ones(size=(10000, 3))


    print(nerf_model(points, direction_vectors)[0].shape, 
    nerf_model(points, direction_vectors)[1].shape)
