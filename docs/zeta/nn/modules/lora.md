# Lora

The `Lora` class is a module of the Zeta library that provides a simple linear transformation of the input data. It is a part of the `torch.nn` module and extends the `nn.Module` class from PyTorch.

## Overview and Introduction

The `Lora` class is designed to provide a scalable and efficient linear transformation operation. It is particularly useful in scenarios where the dimensionality of the input data is very high and computational efficiency is of paramount importance. The `Lora` class achieves this by breaking down the weight matrix into two lower rank matrices `A` and `B`, and a scale factor `alpha`, which are learned during the training process. This results in a significant reduction in the number of parameters to be learned, and consequently, a more computationally efficient model.

## Key Concepts and Terminology

- **Linear Transformation**: A linear transformation is a mathematical operation that transforms input data by multiplying it with a weight matrix. It is a fundamental operation in many machine learning models.

- **Low Rank Approximation**: Low rank approximation is a technique used to approximate a matrix by another matrix of lower rank. This is often used to reduce the dimensionality of data and to make computations more efficient.

- **Scale Factor**: A scale factor is a number by which a quantity is multiplied, changing the magnitude of the quantity.

## Class Definition

The `Lora` class is defined as follows:

```python
class Lora(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        r=8,
        alpha=None
    ):
        super().__init__()
        self.scale = alpha / r

        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.randn(r, dim_out))

    @property
    def weight(self):
        return (self.A @ self.B) * self.scale
    
    def forward(self, x):
        return x @ self.weight
```

### Parameters

- `dim` (`int`): The dimensionality of the input data. It is the number of features in the input data.
- `dim_out` (`int`): The desired dimensionality of the output data. It is the number of features in the output data.
- `r` (`int`, optional): The rank of the matrices `A` and `B`. It determines the size of the matrices `A` and `B`. Default is 8.
- `alpha` (`float`, optional): The scale factor. If not provided, it is set to 1 by default.

### Methods

#### `forward`

The `forward` method is used to compute the forward pass of the `Lora` module.

##### Parameters

- `x` (`Tensor`): The input data. It is a tensor of shape `(batch_size, dim)`.

##### Returns

- `Tensor`: The transformed data. It is a tensor of shape `(batch_size, dim_out)`.

## Functionality and Usage

The `Lora` class is used to perform a linear transformation of the input data. The transformation is defined by the weight matrix `W`, which is approximated by the product of two lower rank matrices `A` and `B`, and a scale factor `alpha`. The `Lora` class learns the matrices `A` and `B`, and the scale factor `alpha` during the training process. 

The forward pass of the `Lora` module computes the product of the input data `x` and the weight matrix `W`, which is approximated by `(A @ B) * scale`.

### Mathematical Formula

The mathematical formula for the forward pass of the `Lora` module is:

\[ y = xW \]

Where:
- \( y \) is the transformed data.
- \( x \) is the input data.
- \( W \) is the weight matrix, which is approximated by \( (A @ B) * \text{scale} \).

### Usage Examples

Below are three examples of how to use the `Lora` class.

#### Example 1: Basic Usage

```python
import torch
from zeta import Lora

# Define the input data
x = torch.randn(32, 128) # batch size of 32, and 128 features

# Define the Lora module
lora = Lora(dim=128, dim_out=64)

# Compute the forward pass
y = lora(x)
```

#### Example 2: Specifying the Rank and Scale Factor

```python
import torch
from zeta import Lora

# Define the input data
x = torch.randn(32, 128) # batch size of 32, and 128 features

# Define the Lora module with specified rank and scale factor
lora = Lora(dim=128, dim_out=64, r=16, alpha=0.1)

# Compute the forward pass
y = lora(x)
```

#### Example 3: Using the Lora Module in a Neural Network

```python
import torch
from torch import nn
from zeta import Lora

# Define a simple neural network with a Lora layer
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = Lora(dim=128, dim_out=64)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.lora(x)
        x = self.fc(x)
        return x

# Define the input data
x = torch.randn(32, 128) # batch size of 32, and 128 features

# Define the model
model = Net()

# Compute the forward pass
output = model(x)
```

## Additional Information and Tips

- The `Lora` class is particularly useful in scenarios where the dimensionality of the input data is very high and computational efficiency is of paramount importance. However, it may not be suitable for all applications, as the approximation of the weight matrix may result in a loss of accuracy.

- The rank `r` and the scale factor `alpha` are hyperparameters that need to be tuned for the specific application. A higher value of `r` will

 result in a more accurate approximation of the weight matrix, but will also increase the computational cost. Similarly, the scale factor `alpha` needs to be tuned to achieve the desired trade-off between accuracy and computational efficiency.

## References and Resources

- [PyTorch nn.Module documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Low Rank Matrix Factorization for Deep Neural Network Training with High-dimensional Output Targets](https://arxiv.org/abs/2005.08735)

For further exploration and implementation details, you can refer to the above resources and the official PyTorch documentation.