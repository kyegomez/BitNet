# Multihead Attention Documentation for Zeta Library

## Introduction

`MultiheadAttention` is a module in the Zeta library that provides multi-head attention mechanism. This mechanism enables the model to focus on different parts of the input sequence simultaneously. It's widely used in models such as transformers for capturing various aspects of information in the input.

## Purpose

The purpose of the `MultiheadAttention` module is to allow joint information representation from different subspaces of the input sequence. This results in capturing a richer context when modeling sequences.

## Architecture

The `MultiheadAttention` class extends from the `nn.Module` base class. Internally, it uses linear transformations for keys, values, and queries (`k_proj`, `v_proj`, `q_proj`). These projections are wrapped using the `MultiwayWrapper`. It also utilizes layer normalization (`inner_attn_ln`) and optionally uses relative positional embeddings (`xpos`).

## Class Definition

```python
class zeta.nn.embeddings.MultiheadAttention(nn.Module):
```

### Parameters:
- `args`: General arguments passed for configuring the module.
- `embed_dim` (int): Total dimension of the model.
- `num_heads` (int): Number of parallel attention heads. The embed_dim will be split across num_heads.
- `dropout` (float): Dropout probability. Default: 0.0.
- `self_attention` (bool): Whether to apply self attention. Only one of `self_attention` or `encoder_decoder_attention` can be True. Default: False.
- `encoder_decoder_attention` (bool): Whether to apply encoder-decoder attention. Only one of `self_attention` or `encoder_decoder_attention` can be True. Default: False.
- `subln` (bool): If True, applies layer normalization after self attention. Default: False.

### Methods:

#### `reset_parameters()`
Reinitialize the parameters of the attention module.

#### `forward(query, key, value, ...)`
Computes the forward pass of the attention mechanism.

- Parameters:
  - `query` (Tensor): The query tensor.
  - `key` (Tensor): The key tensor.
  - `value` (Tensor): The value tensor.
  - Other arguments including `incremental_state`, `key_padding_mask`, `attn_mask`, `rel_pos`, and `is_first_step`.

- Returns:
  - `attn` (Tensor): The computed attention tensor.
  - `attn_weights` (Tensor): The attention weights.

### Mathematical Formulation:

Given a query \( Q \), key \( K \), and value \( V \), the multihead attention mechanism is mathematically represented as:

\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \]

Where \( d_k \) is the dimension of the key.

## Usage Examples:

### Example 1: Basic Usage

```python
from zeta.nn.embeddings import MultiheadAttention
import torch

args = ...  # Some configuration
attention = MultiheadAttention(args, embed_dim=512, num_heads=8, dropout=0.1, self_attention=True)
query = torch.rand((32, 10, 512))
key = torch.rand((32, 10, 512))
value = torch.rand((32, 10, 512))

attn, attn_weights = attention(query, key, value)
```

### Example 2: With Masking

```python
from zeta.nn.embeddings import MultiheadAttention
import torch

args = ...  # Some configuration
attention = MultiheadAttention(args, embed_dim=512, num_heads=8, dropout=0.1, self_attention=True)
query = torch.rand((32, 10, 512))
key = torch.rand((32, 10, 512))
value = torch.rand((32, 10, 512))
attn_mask = torch.ones((10, 10)).triu_() * -1e9  # Upper triangular mask

attn, attn_weights = attention(query, key, value, attn_mask=attn_mask)
```

### Example 3: Encoder-Decoder Attention

```python
from zeta.nn.embeddings import MultiheadAttention
import torch

args = ...  # Some configuration
attention = MultiheadAttention(args, embed_dim=512, num_heads=8, dropout=0.1, encoder_decoder_attention=True)
query = torch.rand((32, 10, 512))  # Decoder query
key = torch.rand((32, 20, 512))  # Encoder key
value = torch.rand((32, 20, 512))  # Encoder value

attn, attn_weights = attention(query, key, value)
```

## Additional Tips:
- For encoder-decoder attention, make sure the dimensions of the encoder and decoder tensors match the expected input sizes.
- Using masks can be helpful to prevent the attention mechanism from focusing on certain parts of the sequence, such as padding.
