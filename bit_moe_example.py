import torch
from bitnet.bit_moe import BitMoE

# Create input tensor
x = torch.randn(2, 4, 8)

# Create BitMoE model with specified input and output dimensions
model = BitMoE(8, 4, 2)

# Forward pass through the model
output = model(x)

# Print the output
print(output)
