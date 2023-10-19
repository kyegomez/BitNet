import torch

from bitnet.transformer import BitLinear

# Input
x = torch.randn(10, 512)
print(x)

# BitLinear layer
layer = BitLinear(512, 512)

# Output
y = layer(x)

print(y)
