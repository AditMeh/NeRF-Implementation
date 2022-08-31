import torch
from rendering import rendering

c = torch.ones(4, 3, 3)
density = torch.ones(4, 3, 1)

r = rendering(c, density, 2)
print(r)
# T = torch.exp(-torch.cumsum(density, dim=1))
# print('T:', T)

# S = 1 - torch.exp(-density)

# T = T.expand(4, 3, 3)
# print('T:', T)

# product = T * S * c
# print(product.shape)

# C = torch.sum(product, dim=1)
# print(C.shape)

# C.reshape(2, 2, 3)