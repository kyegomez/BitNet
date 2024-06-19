import torch
from bitnet import BitMamba

# Create a tensor of size (2, 10) with random values between 0 and 100
x = torch.randint(0, 100, (2, 10))

# Create an instance of the BitMamba model with input size 512, hidden size 100, output size 10, and depth size 6
model = BitMamba(512, 100, 10, 6, return_tokens=True)

# Pass the input tensor through the model and get the output
output = model(x)

# Print the output tensor
print(output)

# Print the shape of the output tensor
print(output.shape)
