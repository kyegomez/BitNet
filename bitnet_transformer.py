import torch

from bitnet.main import BitNetTransformer

# # random inputs
# x = torch.randn(10, 512)

# # transformer layer
# model = BitNetTransformer(dim=512, depth=8, heads=8, dim_head=64)

# # apply transformer
# y = model(x)

# print(y)


x = torch.randn(1, 1, 10, 512)

layer = BitNetTransformer(512, 8, 8, 64)

y = layer(x)

print(y)