# RotaryEmbedding

`RotaryEmbedding` is a PyTorch module implementing the rotary embedding mechanism. It is designed to handle sequences of any length without the need for fine-tuning, and can also incorporate positional information into the embeddings.

## Class Definition

```python
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.,
        base=10000,
        base_rescale_factor=1.,
        ):
        ...
```

### Parameters

- `dim` (int): The dimensionality of the embeddings.
- `use_xpos` (bool, optional): Whether to use positional information in the embeddings. Default: `False`.
- `scale_base` (int, optional): Base of the scale for positional information. Default: `512`.
- `interpolation_factor` (float, optional): Factor used for interpolating the embeddings. Default: `1.0`.
- `base` (int, optional): Base of the frequencies used in the embeddings. Default: `10000`.
- `base_rescale_factor` (float, optional): Factor used for rescaling the base of the frequencies. Default: `1.0`.

### Method: `forward`

```python
def forward(self, seq_len, device):
    ...
```

#### Parameters

- `seq_len` (int): The length of the sequence.
- `device` (torch.device): The device on which the computation will be performed.

#### Returns

- `freqs` (Tensor): The computed frequencies for the embeddings.
- `scale` (Tensor): The computed scale for the embeddings.

## Functionality and Usage

The `RotaryEmbedding` module computes rotary embeddings for a sequence of a given length. The embeddings are computed based on the frequency and scale of each position in the sequence. The frequency and scale are computed using the `inv_freq` and `scale` buffers registered in the module.

The `forward` method computes the `freqs` and `scale` tensors based on the `seq_len` and `device` provided. The `freqs` tensor is computed by multiplying the `t` tensor, which contains the indices of the sequence, with the `inv_freq` tensor. The `scale` tensor is computed using the `scale` buffer and the `scale_base` parameter.

The `freqs` and `scale` tensors are then concatenated along the last dimension and returned.

### Usage Examples

#### Example 1: Basic Usage

```python
from zeta.nn import RotaryEmbedding
import torch
from torch import nn

# Initialize the RotaryEmbedding module
rotary_embedding = RotaryEmbedding(dim=64, use_xpos=True)

# Compute the embeddings for a sequence of length 10
seq_len = 10
device = torch.device('cuda')
freqs, scale = rotary_embedding(seq_len, device)

print(freqs)
print(scale)
```

#### Example 2: Using a Different Scale Base

```python
from zeta.nn import RotaryEmbedding
import torch
from torch import nn

# Initialize the RotaryEmbedding module with a different scale base
rotary_embedding = RotaryEmbedding(dim=64, use_xpos=True, scale_base=1024)

# Compute the embeddings for a sequence of length 10
seq_len = 10
device = torch.device('cuda')
freqs, scale = rotary_embedding(seq_len, device)

print(freqs)
print(scale)
```

#### Example 3: Without Positional Information

```python
from zeta.nn import RotaryEmbedding
import torch
from torch import nn

# Initialize the RotaryEmbedding module without positional information
rotary_embedding = RotaryEmbedding(dim=64, use_xpos=False)

# Compute the embeddings for a sequence of length 10
seq_len = 10
device = torch.device('cuda')
freqs, scale = rotary_embedding(seq_len, device)

print(freqs)
print(scale)
```

## Mathematical Formula

The mathematical formula for computing the `freqs` tensor is:

\[ \text{freqs} = t \cdot \text{inv\_freq} \]

Where:
- \( t \) is a tensor containing the indices of the sequence.
- \( \text{inv\_freq} \) is a tensor containing the inverse frequencies.

The mathematical formula for computing the `scale` tensor is:

\[ \text{scale} = \text{scale}^{\frac{\text{power}}{\text{scale\_base}}} \]

Where:
- \( \text{power} \) is a tensor containing the power of each position in the sequence.
- \( \text{scale\_base} \) is a scalar containing the base of the scale.
- \( \text{scale} \) is a tensor containing the scale of each position in the sequence.

## Additional Information and Tips

- The `interpolation_factor` parameter can be used to interpolate the embeddings for sequences of different lengths. A larger `interpolation_factor` will result in a smoother interpolation.
- The `base_rescale_factor` parameter can be used to rescale the base of the frequencies. This can be useful for adjusting the embeddings for sequences of different lengths.
- If `use_xpos` is set to `False`, the `scale` tensor will not be used, and the `freqs` tensor will be returned as is.

## References and Resources

- [Paper: Link to the paper](https://arxiv.org/pdf/2308.10882.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/indehtml)
- [Einops Documentation](https://einops.rocks/pytorch-examples.html)

Note: The above template includes the class definition, parameters, description, functionality, usage examples, mathematical formula, additional information and tips, and references and resources. To replicate the documentation for any other module or framework, follow the same structure and provide the specific details for that module or framework.