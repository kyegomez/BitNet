# Decoder Class Documentation

Module/Class Name: Decoder

```python
class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal=True, **kwargs)
```

## Overview and Introduction

The `Decoder` class is a component of the Zeta library designed for creating a decoder model with multiple attention layers. It extends the functionality of the `AttentionLayers` class to enable the construction of a decoder architecture. The decoder is a key component in various sequence-to-sequence tasks, such as machine translation, text generation, and more.

The decoder employs multi-head self-attention mechanisms and feed-forward networks to transform input sequences into meaningful output sequences while maintaining the causal property. It is particularly suitable for autoregressive tasks, where each step depends only on previous steps in the sequence.

## Class Definition

```python
class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal=True, **kwargs)
```

The `Decoder` class inherits from the `AttentionLayers` class and introduces the causality constraint by setting `causal=True`. It is initialized with various parameters that configure the architecture and behavior of the decoder.

## Parameters

The `Decoder` class constructor accepts various parameters that control the behavior of the decoder. The most important parameters are inherited from the `AttentionLayers` class, and additional parameters specific to the decoder are introduced. Below is a summary of the parameters:

- `dim` (int): Dimensionality of the model.
- `depth` (int): Number of decoder layers.
- `heads` (int): Number of parallel attention heads.
- `cross_attend` (bool): Enable cross-attention between input and output sequences.
- `sandwich_coef` (int): Coefficient for configuring sandwich normalization.
- `residual_attn` (bool): Enable residual connection for self-attention layers.
- `cross_residual_attn` (bool): Enable residual connection for cross-attention layers.
- `layer_dropout` (float): Dropout probability applied to each layer.
- ... (additional parameters inherited from `AttentionLayers`)

## Functionality and Usage

The `Decoder` class extends the functionality of the `AttentionLayers` class to specifically create decoder models. It employs multi-head self-attention mechanisms and feed-forward networks to process input sequences and generate output sequences.

### Initialization

To create a decoder instance, you can use the following code:

```python
from zeta import Decoder

decoder = Decoder(
    dim=512,
    depth=6,
    heads=8,
    causal=True,
    cross_attend=True,
    residual_attn=True,
    layer_dropout=0.1
)
```

### Forward Pass

The forward pass of the decoder can be performed using the following code:

```python
output = decoder(input_sequence, context=context_sequence, mask=mask_sequence, context_mask=context_mask_sequence)
```

Here, `input_sequence` represents the input sequence to the decoder, `context_sequence` represents the context sequence for cross-attention (if enabled), `mask_sequence` is an optional mask to ignore certain elements in the input, and `context_mask_sequence` is an optional mask for the context sequence.

### Return Intermediates

If desired, you can also obtain intermediate outputs at each layer using the `return_hiddens` parameter:

```python
output, intermediates = decoder(input_sequence, context=context_sequence, mask=mask_sequence, context_mask=context_mask_sequence, return_hiddens=True)
```

The `intermediates` object will contain information about intermediate hidden states and attention outputs for each layer.

## Mathematical Formula

The `Decoder` class is built upon the foundation of multi-head self-attention and feed-forward networks. It can be summarized using the following mathematical formula:

1. Input Embedding: \( X \)
2. Multi-Head Self-Attention: \( A = \text{MultiHeadAttention}(X) \)
3. Feed-Forward Network: \( Y = \text{FeedForward}(A) \)
4. Residual Connection: \( Z = X + Y \)

The above formula represents the basic forward pass of each layer in the decoder. The decoder iteratively applies these operations across its layers to generate meaningful output sequences while maintaining causal dependencies.

## References

- [Zeta Library Documentation](https://example.com/zeta/docs)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PAR: Prompted Attention](https://arxiv.org/abs/2207.04503)
```

This documentation provides an in-depth overview of the `Decoder` class in the Zeta library. It covers its purpose, parameters, usage examples, and includes a simplified mathematical formula to illustrate its functionality.