# Transformer Documentation

## Overview

The `Transformer` class in the Zeta library is a versatile deep learning architecture that combines attention mechanisms with feedforward neural networks for various natural language processing tasks, such as language modeling, machine translation, and text generation. The Transformer architecture was introduced in the paper "Attention is All You Need" by Vaswani et al.

The main purpose of the `Transformer` class is to provide a flexible and configurable interface for creating transformer-based models for sequence-to-sequence tasks. The class allows users to specify the number of tokens, maximum sequence length, attention layers, embeddings, and other parameters necessary for creating and training transformer models.

The Transformer class supports both autoregressive and non-autoregressive training settings and includes features such as relative positional biases, rotary positional embeddings, memory tokens, and more.

## Class Signature

```python
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        embedding_provider: BaseEmbedding,
        emb_dim = None,
        max_mem_len = 0.,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        tie_embedding = False,
        logits_dim = None,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        l2norm_embed = False,
        emb_frac_gradient = 1.
    )
```

## Parameters

- `num_tokens` (int): The total number of tokens in the vocabulary.
- `max_seq_len` (int): The maximum length of the input sequences.
- `attn_layers` (AttentionLayers): An instance of the `AttentionLayers` class representing the core attention layers of the transformer.
- `embedding_provider` (BaseEmbedding): An instance of the `BaseEmbedding` class providing token embeddings.
- `emb_dim` (int, optional): The embedding dimension. Default is `None`, in which case `emb_dim` is set to the same dimension as the `attn_layers`.
- `max_mem_len` (float, optional): Maximum memory length for memory tokens. Default is `0.0`, indicating no memory tokens.
- `shift_mem_down` (int, optional): Number of positions to shift memory tokens down in each layer. Default is `0`.
- `emb_dropout` (float, optional): Dropout rate applied to the embedding layer. Default is `0.0`.
- `post_emb_norm` (bool, optional): Apply layer normalization to the post-embedding inputs. Default is `False`.
- `num_memory_tokens` (int, optional): Number of memory tokens to use. Default is `None`, indicating no memory tokens.
- `tie_embedding` (bool, optional): Tie the output projection weights with the input token embeddings. Default is `False`.
- `logits_dim` (int, optional): Dimensionality of the output logits. Default is `None`, indicating that it's the same as `num_tokens`.
- `use_abs_pos_emb` (bool, optional): Use absolute positional embeddings. Default is `True`.
- `scaled_sinu_pos_emb` (bool, optional): Use scaled sinusoidal positional embeddings. Default is `False`.
- `l2norm_embed` (bool, optional): Apply L2 normalization to the embeddings. Default is `False`.
- `emb_frac_gradient` (float, optional): Fraction of the gradient that should go to the embedding. Default is `1.0`.

## Methods

### `forward`

```python
def forward(
    self,
    x,
    return_embeddings = False,
    return_logits_and_embeddings = False,
    return_intermediates = False,
    mask = None,
    return_mems = False,
    return_attn = False,
    mems = None,
    pos = None,
    prepend_embeds = None,
    sum_embeds = None,
    **kwargs
)
```

This method computes the forward pass of the transformer.

#### Parameters

- `x` (torch.Tensor): Input tensor representing the sequence of token indices.
- `return_embeddings` (bool, optional): If `True`, return only the embeddings without applying the output projection. Default is `False`.
- `return_logits_and_embeddings` (bool, optional): If `True`, return both the logits and embeddings. Default is `False`.
- `return_intermediates` (bool, optional): If `True`, return intermediate attention values. Default is `False`.
- `mask` (torch.Tensor, optional): Attention mask indicating positions to be masked. Default is `None`.
- `return_mems` (bool, optional): If `True`, return updated memory tokens. Default is `False`.
- `return_attn` (bool, optional): If `True`, return attention maps. Default is `False`.
- `mems` (list of torch.Tensor, optional): Memory tokens for each layer. Default is `None`.
- `pos` (torch.Tensor, optional): External positional embeddings. Default is `None`.
- `prepend_embeds` (torch.Tensor, optional): Prepend embeddings to the input sequence. Default is `None`.
- `sum_embeds` (torch.Tensor, optional): Sum external embeddings to the input sequence. Default is `None`.
- `kwargs`: Additional keyword arguments passed to the attention layers.

#### Returns

The method returns the output logits or embeddings based on the specified return options.

## Usage Examples

Here are three usage examples of the `Transformer` class from the Zeta library:

```python
from zeta.nn import Transformer

# Example 1: Basic Usage
transformer = Transformer(
    num_tokens=10000,
    max_seq_len=256,
    attn_layers=attn_layers_instance,
    embedding_provider=embedding_provider_instance
)
logits = transformer(input_tokens)

# Example 2: Return Embeddings
embeddings = transformer(input_tokens, return_embeddings=True)

# Example 3: Return Intermediate Attention Maps
logits, attn_maps = transformer(input_tokens, return_attn=True)
```

In these examples, replace `attn_layers_instance` and `embedding_provider_instance` with actual instances of `AttentionLayers` and `BaseEmbedding`, respectively, and `input_tokens` with your input tensor containing token indices.

## Mathematical Formula

The mathematical formula for the `Transformer` class can be represented as follows:

```
Input -> Embedding -> Post-embedding Norm -> Embedding Dropout -> Project Embedding -> Attention Layers -> Layer Normalization -> To Logits/Embeddings
```

In this formula, "Attention Layers" represents the core attention mechanism of the transformer, which includes self-attention and feedforward neural networks.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is All You Need. Advances in neural information processing systems, 30.
- Zeta Library: Link to the official documentation of the Zeta library.
- Insert any additional references or resources as needed.
```

