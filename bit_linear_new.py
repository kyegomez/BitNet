import torch
from bitnet import BitLinearNew

# Create a random tensor of shape (16, 10)
x = torch.randn(16, 10)

# Create an instance of the BitLinearNew class with input size 10, output size 20, and 2 groups
layer = BitLinearNew(10, 20, num_groups=2)

# Perform a forward pass through the BitLinearNew layer with input x
output = layer(x)

# Print the output tensor
print(output)
