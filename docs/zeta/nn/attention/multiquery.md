# MultiQueryAttention

## Overview and Introduction:

The `MultiQueryAttention` class is a part of the Zeta library, designed to perform self-attention operations on given input data. Unlike traditional attention mechanisms that use a single query, this class leverages multiple queries to capture a broader range of context information. This class allows for various implementations of attention, including Flash, Triton, and Torch. It also provides the flexibility to choose normalization type, fully connected layer type, and offers debugging verbosity.

## Class Definition:

```python
class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.
    Using torch or triton attention implementation enables the user to also use
    additive bias.
    """
```

### Parameters:
- `d_model` (int): Dimension of the model.
- `heads` (int): Number of parallel attention heads.
- `attn_impl` (str, optional): Attention implementation type, can be either 'triton', 'flash', or 'torch'. Default is 'triton'.
- `clip_qkv` (Optional[float]): Clipping value for query, key, and value. If specified, qkv is clamped within the range [-clip_qkv, clip_qkv].
- `qk_ln` (bool, optional): If True, layer normalization is applied to query and key.
- `softmax_scale` (Optional[float]): Scale for softmax. Default value is computed as 1/sqrt(head_dim).
- `attn_pdrop` (float, optional): Attention dropout probability. Default is 0.0.
- `norm_type` (str, optional): Normalization type, default is 'low_precision_layernorm'.
- `fc_type` (str, optional): Fully connected layer type, default is 'torch'.
- `verbose` (int, optional): Verbosity level, default is 0.
- `device` (Optional[str]): Device to which the tensors should be moved.

## Functionality and Usage:

The `MultiQueryAttention` class operates by using multiple queries to capture broader context information from given data. This is achieved through the forward method which computes the self-attention on the given inputs.

### Method: `forward`
```python
def forward(
    self,
    x,
    past_key_value=None,
    bias=None,
    mask=None,
    causal=True,
    needs_weights=False,
):
```

#### Parameters:

- `x` (Tensor): Input tensor.
- `past_key_value` (Optional): Past key and value for attention computation. Default is None.
- `bias` (Optional): Additive bias for attention scores. Default is None.
- `mask` (Optional): Key padding mask. Default is None.
- `causal` (bool, optional): If True, a causal mask is applied to prevent information flow from future tokens. Default is True.
- `needs_weights` (bool, optional): If True, attention weights are also returned. Default is False.

#### Returns:

- `context` (Tensor): Contextualized tensor after attention computation.
- `attn_weights` (Tensor, Optional): Attention weights. Only returned if `needs_weights` is True.
- `past_key_value` (Tensor, Optional): New past key and value.

## Usage Examples:

1. Basic Usage:
```python
from zeta import MultiQueryAttention
import torch

# Initialize the attention module
attention_layer = MultiQueryAttention(d_model=512, heads=8, attn_impl='torch')

# Random input tensor
x = torch.rand(16, 10, 512)  # Batch of 16, sequence length 10, embedding size 512
output, attn_weights, _ = attention_layer(x)
```

2. Using Past Key and Value:
```python
past_key_value = (torch.rand(16, 8, 10, 64), torch.rand(16, 8, 10, 64))  # Past key and value for 8 heads
output, attn_weights, new_past_key_value = attention_layer(x, past_key_value=past_key_value)
```

3. With Causal Masking and Weights:
```python
output, attn_weights, _ = attention_layer(x, causal=True, needs_weights=True)
```

## Mathematical Formula:

For the self-attention mechanism, the computation involves using multiple queries (\( Q \)), keys (\( K \)), and values (\( V \)):

```latex
\[ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}} + \text{Bias}\right) \times V \]
```
Where:
- \( Q \), \( K \), and \( V \) are the queries, keys, and values respectively.
- \( d_k \) is the dimension of the keys.
- Bias is the optional additive bias.

## Additional Information and Tips:

- It's crucial to select the correct attention implementation (`attn_impl`) based on your needs and the hardware you're running on.
- The `triton` implementation might be faster than `flash` but can use more memory. Ensure that you have adequate GPU memory if using `triton`.
- If using the `torch` implementation, it's advisable to check if CUDA is available for GPU acceleration.
- The clipping of qkv (`clip_qkv`) can be beneficial for stability in training.

## References and Resources:
For a deeper understanding of the self-attention mechanism and its variants, you can refer to the "Attention is All You Need" paper by Vaswani et al., 2017.