import torch 
from bitnet.main import BitNetTransformer

#random inputs
x = torch.randn(10, 512)

#transformer layer
model = BitNetTransformer(512, 8, 8, 64)

#apply transformer
y = model(x)

print(y)