import torch

from bitnet.bitlinear import BitLinear

# Random Inputs
x = torch.randn(10, 512)

# BiLinear Layer
layer = BitLinear(512)

# Apply BiLinear Layer
quantizate = layer(x)

# Print
print(quantizate)
