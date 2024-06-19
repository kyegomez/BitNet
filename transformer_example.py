# Import the necessary libraries
import torch
from bitnet import BitNetTransformer

# Create a random tensor of integers
x = torch.randint(0, 20000, (1, 1024))

# Initialize the BitNetTransformer model
bitnet = BitNetTransformer(
    num_tokens=20000,  # Number of unique tokens in the input
    dim=1024,  # Dimension of the input and output embeddings
    depth=6,  # Number of transformer layers
    heads=8,  # Number of attention heads
    ff_mult=4,  # Multiplier for the hidden dimension in the feed-forward network
)

# Pass the tensor through the transformer model
logits = bitnet(x)

# Print the shape of the output
print(logits.shape)
