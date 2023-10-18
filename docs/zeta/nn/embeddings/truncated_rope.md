# Module/Function Name: TruncatedRotaryEmbedding

The `TruncatedRotaryEmbedding` class is part of the Zeta library and is designed to implement the rotary embeddings with a truncation mechanism. The rotary embedding is a positional encoding method that aims to provide the model with information about the relative positions of the tokens in a sequence. The `TruncatedRotaryEmbedding` class extends the rotary embedding concept by incorporating a truncation mechanism, which sets the rotary embedding to zero for positions where the frequency is higher than a specified threshold.

The architecture and workings of this class are inspired by the paper [link to the paper](https://arxiv.org/pdf/2308.10882.pdf).

## Parameters:

- `dim` (int): Dimensionality of the embeddings.
- `a` (float): Lower bound of the truncation region. Rotary embeddings with frequency lower than `a` will be set to zero.
- `b` (float): Upper bound of the truncation region. Rotary embeddings with frequency higher than or equal to `b` will not be truncated.
- `rho` (float): Value to which the rotary embeddings will be truncated in the region [a, b).

The `dim` parameter is required to determine the dimensionality of the embeddings, while `a`, `b`, and `rho` are hyperparameters that control the truncation mechanism.

## Method:

### `forward(seq_len, device)`

Computes the truncated rotary embeddings for a given sequence length.

#### Parameters:

- `seq_len` (int): Length of the sequence for which the rotary embeddings are to be computed.
- `device` (torch.device): Device on which the computations are to be performed.

#### Returns:

- `result` (Tensor): A tensor containing the truncated rotary embeddings for the specified sequence length.

## Functionality and Usage:

The `TruncatedRotaryEmbedding` class is used to compute the truncated rotary embeddings for a given sequence length. The rotary embeddings are computed by multiplying a tensor containing the position indices of the tokens in the sequence by the inverse frequencies. The inverse frequencies are computed based on the specified embedding dimension `dim` and are stored in the `inv_freq` buffer.

The truncation mechanism is implemented by creating a `theta_star` tensor, which is used to multiply the computed `freqs`. The `theta_star` tensor is created based on the specified `a`, `b`, and `rho` parameters, and the computed `freqs` tensor. For positions where the frequency is higher than or equal to `b`, the rotary embeddings are not truncated, and `theta_star` is set to the frequency at that position. For positions where the frequency is lower than `a`, the rotary embeddings are set to zero, and `theta_star` is set to zero. For positions where the frequency is in the range [a, b], the rotary embeddings are truncated to `rho`, and `theta_star` is set to `rho`.

Once the `theta_star` tensor is created, it is multiplied element-wise by the `freqs` tensor to compute the final truncated rotary embeddings.

### Usage Example:

```python
from zeta.nn.embeddings.truncated_rope import TruncatedRotaryEmbedding
import torch

# Define the parameters
dim = 64
a = 0.1
b = 0.9
rho = 0.5
seq_len = 100
device = torch.device('cuda')

# Create the TruncatedRotaryEmbedding module
trunc_rotary_emb = TruncatedRotaryEmbedding(dim, a, b, rho)

# Compute the truncated rotary embeddings for the specified sequence length
rotary_embeddings = trunc_rotary_emb(seq_len, device)

print(rotary_embeddings)
```

In this example, the `TruncatedRotaryEmbedding` module is created with the specified `dim`, `a`, `b`, and `rho` parameters. The `forward` method is then called with the specified `seq_len` and `device` parameters to compute the truncated rotary embeddings for a sequence of length `seq_len`.

## Additional Information and Tips:

- The `a`, `b`, and `rho` parameters control the truncation mechanism and may need to be tuned based on the specific application and data being used. In particular, the `a` parameter should be set to a value that effectively removes the high-frequency noise in the rotary embeddings, while the `b` parameter should be set to a value that retains the useful positional information in the rotary embeddings.

- The `dim` parameter should be set to the same value as the embedding dimension used in the model.

- The `device` parameter in the `forward` method should be set to the same device on which the model is being trained.

## Mathematical Formulation:

The mathematical formulation of the truncated rotary embeddings can be expressed as follows:

\[ \text{freqs} = t \cdot \text{inv\_freq} \]

\[ \theta = \text{base}^{-2 \cdot i / \text{dim}}, \, i = 0, 2, \ldots, \text{dim}-2 \]

\[ \theta^* = 
\begin{cases}
0, & \text{if } \theta < a \\
\rho, & \text{if } a \leq \theta < b \\
\theta, & \text{if } \theta \geq b
\end{cases}
\]

\[ \text{result} = \text{freqs} \cdot \theta^* \]

Where:

- \( t \) is a tensor containing the position indices of the tokens in the sequence.
- \( \text{inv\_freq} \) is a tensor containing the inverse frequencies computed based on the specified `dim` parameter.
- \( \text{freqs} \) is a tensor containing the computed frequencies for each position in the sequence.
- \( \theta \) is a tensor containing the computed theta values for each position in the sequence.
- \( \theta^* \) is a tensor containing the truncated theta values for each position in the sequence.
- \( \text{result} \) is the final tensor containing the truncated rotary embeddings for each position in the sequence.

## References and Resources:

- Paper: [Link to the paper](https://arxiv.org/pdf/2308.10882.pdf)

For further exploration and implementation details, refer to the paper linked above.