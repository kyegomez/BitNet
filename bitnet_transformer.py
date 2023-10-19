import torch

from bitnet.main import BitNetTransformer

# random input ints
x = torch.randint(0, 256, (10, 512))

# transformer layer
model = BitNetTransformer(512, 8, 8, 64)

# apply transformer
y = model(x)

print(y)
