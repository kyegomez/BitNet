import torch

from bitnet import BitLinear

# Input
x = torch.randn(10, 10000, 512)

# BitLinear layer
layer = BitLinear(512, 400)

# Output
y = layer(x)

print(y)
print(y.shape)
