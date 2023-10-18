# Zeta Library Documentation

## Module Name: TokenLearner

The `TokenLearner` is a PyTorch module designed for learning tokens from input data. It is a part of the Zeta library, a collection of modules and functions designed for efficient and flexible implementation of various deep learning tasks. The `TokenLearner` class is particularly useful for tasks such as image classification, object detection, and other applications where it is beneficial to extract tokens (representative features) from the input data.

## Introduction

In various deep learning tasks, it is common to extract tokens (representative features) from the input data. These tokens are then used for downstream tasks like classification, detection, etc. The `TokenLearner` class is designed to efficiently extract tokens from the input data. It does this by utilizing a convolutional neural network (CNN) with grouped convolutions and a gating mechanism.

## Class Definition

```python
class TokenLearner(nn.Module):
    def __init__(
            self,
            *,
            dim: int = None,
            ff_mult: int = 2,
            num_output_tokens: int = 8,
            num_layers: int = 2
    ):
        ...
```

### Parameters:

- `dim` (int, optional): The dimension of the input data. Default is `None`.
- `ff_mult` (int, optional): The factor by which the inner dimension of the network will be multiplied. Default is `2`.
- `num_output_tokens` (int, optional): The number of tokens to be output by the network. Default is `8`.
- `num_layers` (int, optional): The number of layers in the network. Default is `2`.

## Functionality and Usage

The `TokenLearner` class is a PyTorch `nn.Module` that learns tokens from the input data. The input data is first packed and then processed through a series of grouped convolutions followed by a gating mechanism. The output is a set of tokens that are representative of the input data.

The forward method of the `TokenLearner` class takes an input tensor `x` and performs the following operations:

1. The input tensor `x` is packed using the `pack_one` helper function.
2. The packed tensor is then rearranged and passed through a series of grouped convolutions and activation functions.
3. The output of the convolutions is then rearranged and multiplied with the input tensor.
4. The resulting tensor is then reduced to obtain the final tokens.

### Method:

```python
def forward(self, x):
    ...
```

### Parameters:

- `x` (Tensor): The input tensor of shape `(batch_size, channels, height, width)`.

### Returns:

- `x` (Tensor): The output tokens of shape `(batch_size, channels, num_output_tokens)`.

## Usage Examples

### Example 1: Basic Usage

```python
from zeta import TokenLearner
import torch

# Initialize the TokenLearner
token_learner = TokenLearner(dim=64)

# Generate some random input data
x = torch.randn(1, 64, 32, 32)

# Forward pass
tokens = token_learner.forward(x)

print(tokens.shape)
```

In this example, a `TokenLearner` is initialized with an input dimension of 64. A random tensor of shape `(1, 64, 32, 32)` is then passed through the `TokenLearner` to obtain the tokens. The output will be a tensor of shape `(1, 64, 8)`.

### Example 2: Custom Parameters

```python
from zeta import TokenLearner
import torch

# Initialize the TokenLearner with custom parameters
token_learner = TokenLearner(dim=128, ff_mult=4, num_output_tokens=16)

# Generate some random input data
x = torch.randn(2, 128, 64, 64)

# Forward pass
tokens = token_learner.forward(x)

print(tokens.shape)
# Output: torch.Size([2, 128, 16])
```

In this example, a `TokenLearner` is initialized with custom parameters. A random tensor of shape `(2, 128, 64, 64)` is then passed through the `TokenLearner` to obtain the tokens. The output will be a tensor of shape `(2, 128, 16)`.

### Example 3: Integration with Other PyTorch Modules

```python
from zeta import TokenLearner
import torch
import torch.nn as nn

# Initialize the TokenLearner
token_learner = TokenLearner(dim=64)

# Generate some random input data
x = torch.randn(1, 64, 32, 32)

# Define a simple model
model = nn.Sequential(
    token_learner,
    nn.Flatten(),
    nn.Linear(64*8, 10)
)

# Forward pass
output = model(x)

print(output.shape)
# Output: torch.Size([1, 10])
```

In this example, the `TokenLearner` is integrated into a simple model consisting of the `TokenLearner`, a `Flatten` layer, and a `Linear` layer. A random tensor of shape `(1, 64, 32, 32)` is then passed through the model to obtain the final output. The output will be a tensor of shape `(1, 10)`.

## Mathematical Formulation

The `TokenLearner` can be mathematically formulated as follows:

Let `X` be the input tensor of shape `(B, C, H, W)`, where `B` is the batch size, `C` is the number of channels, `H` is the height, and `W` is the width. The `TokenLearner` first rearranges `X` to a tensor of shape `(B, G*C, H, W)`, where `G` is the number of output tokens. This is done by repeating `X` along the channel dimension `G` times.

The rearranged tensor is then passed through a series of grouped convolutions and activation functions to obtain a tensor `A` of shape `(B, G, H, W)`. This tensor is then rearranged and multiplied with the input tensor `X` to obtain a tensor of shape `(B, C, G, H, W)`.

The final tokens are obtained by reducing this tensor along the `H` and `W` dimensions to obtain a tensor of shape `(B, C, G)`.

## Additional Information and Tips

- The `num_output_tokens` parameter controls the number of tokens that will be output by the `TokenLearner`. A larger number of output tokens will result in a more detailed representation of the input data, but will also increase the computational requirements.

- The `ff_mult` parameter controls the inner dimension of the `TokenLearner`. A larger `ff_mult` will result in a larger capacity model, but will also increase the computational requirements.

- The `TokenLearner` works best with input data that has a relatively small spatial dimension (e.g. 32x32 or 64x64). For larger input sizes, it may be beneficial to use a downsampling layer (e.g. `nn.MaxPool2d`) before passing the data through the `TokenLearner`.

