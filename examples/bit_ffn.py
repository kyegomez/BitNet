import torch
from bitnet import BitFeedForward

# Create a random input tensor of shape (10, 512)
x = torch.randn(10, 512)

# Create an instance of the BitFeedForward class with the following parameters:
# - input_dim: 512
# - hidden_dim: 512
# - num_layers: 4
# - swish: True (use Swish activation function)
# - post_act_ln: True (apply Layer Normalization after each activation)
# - dropout: 0.1 (apply dropout with a probability of 0.1)
ff = BitFeedForward(512, 512, 4, swish=True, post_act_ln=True, dropout=0.1)

# Apply the BitFeedForward network to the input tensor x
y = ff(x)

# Print the shape of the output tensor y
print(y)  # torch.Size([10, 512])
