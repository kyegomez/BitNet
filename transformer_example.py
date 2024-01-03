import torch
from bitnet import BitNetTransformer

x = torch.randint(0, 20000, (1, 512))
bitnet = BitNetTransformer(
    num_tokens=20000,
    dim=512,
    depth=6,
    heads=8,
    ff_mult=4,
)
logits = bitnet(x)
print(logits.shape)