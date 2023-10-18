import torch

from bitnet.bitlinear import BitLinear

# example
x = torch.randn(10, 512)

layer = BitLinear(512)

y, dequant = layer(x)

print(y, dequant)
