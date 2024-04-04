import torch
from bitnet import BitMamba

# Create a random tensor of shape (2, 10, 512)
x = torch.randn(2, 10, 512)

# Create an instance of the BitMamba model with input size 512 and output size 6
model = BitMamba(512, 6)

# Pass the input tensor through the model to get the output
output = model(x)

# Print the output tensor
print(output)

# Print the shape of the output tensor
print(output.shape)
