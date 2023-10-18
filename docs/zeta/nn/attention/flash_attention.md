# FlashAttention

The FlashAttention module performs efficient attention computations, specifically designed for leveraging hardware capabilities on certain NVIDIA GPUs. It offers the option to perform "flash" attention which can be computationally faster on specific GPU architectures.

---

## Class Definition:

```python
class FlashAttention(nn.Module):
```

### Parameters:

- `causal` (bool, optional): Determines whether to apply causal masking. Default: False.
- `dropout` (float, optional): Dropout probability. Default: 0.
- `flash` (bool, optional): Whether to use flash attention. Requires PyTorch version 2.0 or above. Default: True.

---

## Methods:

### `__init__(self, causal=False, dropout=0., flash=True)`

Initializes the FlashAttention module.

### `get_mask(self, i, j, device)`

Generates a mask for attention computation.

#### Parameters:
- `i` (int): Length of the query sequence.
- `j` (int): Length of the key sequence.
- `device` (torch.device): Device to place the mask tensor.

#### Returns:
- `torch.Tensor`: Mask tensor of shape `(i, j)`.

### `flash_attn(self, q, k, v, mask=None, attn_bias=None)`

Performs flash attention computation.

#### Parameters:
- `q` (torch.Tensor): Query tensor of shape `(batch, heads, q_len, dim)`.
- `k` (torch.Tensor): Key tensor of shape `(batch, heads, k_len, dim)`.
- `v` (torch.Tensor): Value tensor of shape `(batch, heads, v_len, dim)`.
- `mask` (torch.Tensor, optional): Mask tensor of shape `(batch, heads, q_len, k_len)`. Default: None.
- `attn_bias` (torch.Tensor, optional): Attention bias tensor of shape `(batch, heads, q_len, k_len)`. Default: None.

#### Returns:
- `torch.Tensor`: Output tensor of shape `(batch, heads, q_len, dim)`.

### `forward(self, q, k, v, mask=None, attn_bias=None)`

Performs the attention computation using einstein notation.

#### Parameters:
- `q` (torch.Tensor): Query tensor of shape `(batch, heads, q_len, dim)`.
- `k` (torch.Tensor): Key tensor of shape `(batch, heads, k_len, dim)`.
- `v` (torch.Tensor): Value tensor of shape `(batch, heads, v_len, dim)`.
- `mask` (torch.Tensor, optional): Mask tensor of shape `(batch, heads, q_len, k_len)`. Default: None.
- `attn_bias` (torch.Tensor, optional): Attention bias tensor of shape `(batch, heads, q_len, k_len)`. Default: None.

#### Returns:
- `torch.Tensor`: Attention output tensor.

---

## Usage Examples:

1. **Basic Usage**:
```python
from zeta.nn import FlashAttention
attn_module = FlashAttention()
output = attn_module(query_tensor, key_tensor, value_tensor)
```

2. **Using Flash Attention with Masking**:
```python
from zeta.nn import FlashAttention
attn_module = FlashAttention(flash=True)
mask = attn_module.get_mask(query_length, key_length, device)
output = attn_module(query_tensor, key_tensor, value_tensor, mask=mask)
```

3. **Using Causal Flash Attention with Dropout**:
```python
from zeta.nn import FlashAttention
attn_module = FlashAttention(causal=True, dropout=0.1, flash=True)
output = attn_module(query_tensor, key_tensor, value_tensor)
```

---

## Additional Tips:

- The `FlashAttention` module is optimized for NVIDIA A100 GPUs. On these GPUs, using `flash=True` is recommended for faster computation.
- Ensure that PyTorch version is 2.0 or above when enabling flash attention.
- The mask generated using `get_mask` method is useful for attention computations where certain positions need to be masked out.

---

## References:

- Original Attention Mechanism: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)