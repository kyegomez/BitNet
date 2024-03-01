import torch
from bitnet import BitMGQA

# Create a random tensor of shape (1, 10, 512)
x = torch.randn(1, 10, 512)

# Create an instance of the BitMGQA model with input size 512, 8 attention heads, and 4 layers
gqa = BitMGQA(512, 8, 4)

# Pass the input tensor through the BitMGQA model and get the output and attention weights
out, attn = gqa(x, x, x, need_weights=True)

# Print the shapes of the output tensor and attention tensor
print(out.shape, attn.shape)
