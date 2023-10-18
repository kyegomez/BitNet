# RelativePositionBias

`RelativePositionBias` is a specialized PyTorch module designed to generate relative position biases, which can be vital for certain attention mechanisms in deep learning architectures. This module quantizes the distance between two positions into a certain number of buckets and then uses an embedding to get the relative position bias. This mechanism aids in the attention mechanism by providing biases based on relative positions between the query and key, rather than relying solely on their absolute positions.

## Architecture:
The architecture can be visualized in three major steps:
1. **Bucketing:** Convert relative distances between two positions into bucket indices.
2. **Embedding:** Use the bucket indices to get embeddings for each pair of positions.
3. **Computing Bias:** Computes the bias values based on the embeddings.

## Purpose:
In the context of attention mechanisms, especially the transformer-based architectures, the position of tokens can provide valuable information. The `RelativePositionBias` class helps introduce this information in a compact form by bucketing relative positions and then embedding them to serve as biases for the attention scores.

## Mathematical Formula:
Given a relative position \( r \), the bucket index \( b \) is computed as:
\[ b = 
\begin{cases} 
      n + \text{num_buckets} \div 2 & \text{if } n < 0 \text{ and bidirectional is True} \\
      \min\left( \max_{\text{exact}} + \left(\frac{\log(\frac{n}{\max_{\text{exact}}})}{\log(\frac{\text{max_distance}}{\max_{\text{exact}}})} \times (\text{num_buckets} - \max_{\text{exact}})\right), \text{num_buckets} - 1 \right) & \text{otherwise} 
   \end{cases}
\]
Where \( n \) is the negative of the relative position, and \( \max_{\text{exact}} \) is \( \text{num_buckets} \div 2 \).

## Class Definition:

```python
class RelativePositionBias(nn.Module):
    """
    Compute relative position bias which can be utilized in attention mechanisms.
    
    Parameters:
    - bidirectional (bool): If True, considers both forward and backward relative positions. Default: True.
    - num_buckets (int): Number of buckets to cluster relative position distances. Default: 32.
    - max_distance (int): Maximum distance to be considered for bucketing. Distances beyond this will be mapped to the last bucket. Default: 128.
    - n_heads (int): Number of attention heads. Default: 12.
    """
```

### Key Methods:
- **_relative_position_bucket**: This static method is responsible for converting relative positions into bucket indices.
- **compute_bias**: Computes the relative position bias for given lengths of queries and keys.
- **forward**: Computes and returns the relative position biases for a batch.

## Usage Examples:

```python
from zeta import RelativePositionBias
import torch

# Initialize the RelativePositionBias module
rel_pos_bias = RelativePositionBias()

# Example 1: Compute bias for a single batch
bias_matrix = rel_pos_bias(1, 10, 10)

# Example 2: Utilize in conjunction with an attention mechanism
# NOTE: This is a mock example, and may not represent an actual attention mechanism's complete implementation.
class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel_pos_bias = RelativePositionBias()

    def forward(self, queries, keys):
        bias = self.rel_pos_bias(queries.size(0), queries.size(1), keys.size(1))
        # Further computations with bias in the attention mechanism...
        return None  # Placeholder

# Example 3: Modify default configurations
custom_rel_pos_bias = RelativePositionBias(bidirectional=False, num_buckets=64, max_distance=256, n_heads=8)
```

## Tips:
1. The choice of `num_buckets` and `max_distance` might need tuning based on the dataset and application.
2. If the architecture doesn't need bidirectional biases, set `bidirectional` to `False` to reduce computation.
3. Ensure that the device of tensors being processed and the device of the `RelativePositionBias` module are the same.

## References:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformer Architectures](https://www.aclweb.org/anthology/D18-1422.pdf)

Note: This documentation is based on the provided code and might need adjustments when integrated into the complete `zeta` library.