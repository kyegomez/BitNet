# AlibiPositionalBias Documentation

## Introduction

The `AlibiPositionalBias` module belongs to the zeta library and plays a crucial role in handling positional bias for multi-head attention mechanisms. Specifically, it attempts to alleviate the absolute positional bias based on the number of attention heads.

## Class Definition:

```python
class AlibiPositionalBias(nn.Module):
```

### Parameters:
- **heads** (`int`): Number of attention heads for which the slopes need to be calculated.
- **total_heads** (`int`): Total number of attention heads in the network.

### Attributes:
- **slopes** (`Tensor`): Tensor containing slope values, which are computed based on the number of heads.
- **bias** (`Tensor` or `None`): Tensor for storing positional bias values. If not initialized or needs recomputation, it would be None.

### Methods:
#### `__init__(self, heads, total_heads, **kwargs) -> None`:
Initializes the `AlibiPositionalBias` module.

#### `get_bias(self, i, j, device) -> Tensor`:
Computes the positional bias for given dimensions i and j.

- **Parameters**:
  - **i** (`int`): One dimension of the required positional bias.
  - **j** (`int`): Second dimension of the required positional bias.
  - **device** (`torch.device`): The device on which computations are to be performed.

#### `_get_slopes(heads) -> List[float]`:
A static method that calculates slopes based on the number of attention heads.

- **Parameters**:
  - **heads** (`int`): Number of attention heads.

#### `forward(self, i, j) -> Tensor`:
Computes or retrieves the bias tensor for given dimensions.

- **Parameters**:
  - **i** (`int`): One dimension for the required positional bias.
  - **j** (`int`): Second dimension for the required positional bias.

## Mathematical Formula:

Given `n` attention heads, the alibi positional bias can be represented as:

\[ \text{Bias} = \text{-abs}(j_{\text{range}}) \times \text{slope} \]

Where:
- \( j_{\text{range}} \) is an array of numbers from `0` to `j-1`.
- `slope` is computed based on the number of heads using `_get_slopes` method.

## Usage Examples:

### Example 1: Initialize and compute bias
```python
from zeta import AlibiPositionalBias
import torch

bias_module = AlibiPositionalBias(heads=4, total_heads=8)
bias = bias_module(10, 10)
print(bias)
```

### Example 2: Retrieve stored bias
```python
bias = bias_module(5, 5)
print(bias)
```

### Example 3: Computing bias for different dimensions
```python
bias = bias_module(8, 15)
print(bias)
```

## Note:

- It's crucial to ensure that the `total_heads` parameter is always greater than or equal to the `heads` parameter during initialization.
- The device property is internally used to determine the computation device based on the registered buffers.

## References:

For a deeper understanding and applications of positional bias in attention mechanisms, one may refer to the foundational paper on Transformer architectures:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Also, the `einops` library provides a versatile interface for tensor manipulations. More details can be found at its official [documentation](https://einops.rocks/).