# **Documentation for `MultiwayEmbedding` in Zeta Library**

**Table of Contents**

1. Overview
2. Class Definition and Parameters
3. Methods and Functionalities
4. Usage Examples
5. Additional Tips and Information
6. References

---

## 1. Overview

The `MultiwayEmbedding` class in the Zeta library provides a way to apply two separate embeddings to two distinct parts of the input tensor. It splits the input tensor at the specified position and applies one embedding to the first part and another embedding to the second part. This can be particularly useful when dealing with inputs that require diverse representations or embeddings.

---

## 2. Class Definition and Parameters

```python
class MultiwayEmbedding(MultiwayNetwork):
    """
    A specialized version of the MultiwayNetwork to perform multi-way embeddings on an input tensor.

    Parameters:
    - modules (List[nn.Module]): A list containing exactly two PyTorch modules. Typically these would be embedding layers.
    - dim (int): The dimension along which to split and concatenate the input tensor. Default is 1.
    """

    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        ...
```

---

## 3. Methods and Functionalities

**forward(x, **kwargs)**
```python
def forward(self, x, **kwargs):
    """
    Forward method to apply embeddings on the split input tensor.

    Parameters:
    - x (torch.Tensor): The input tensor.
    - **kwargs: Additional arguments that might be needed for the embeddings.

    Returns:
    - torch.Tensor: Concatenated tensor after applying the embeddings.
    """
    ...
```

---

## 4. Usage Examples

**Example 1:** Basic Usage
```python
from zeta import MultiwayEmbedding
import torch.nn as nn

emb1 = nn.Embedding(10, 5)
emb2 = nn.Embedding(10, 5)
multiway_emb = MultiwayEmbedding([emb1, emb2])

x = torch.LongTensor([[1,2,3],[4,5,6]])
output = multiway_emb(x)
print(output)
```

**Example 2:** Setting a Split Position
```python
from zeta import MultiwayEmbedding, set_split_position
import torch.nn as nn

emb1 = nn.Embedding(10, 5)
emb2 = nn.Embedding(10, 5)
multiway_emb = MultiwayEmbedding([emb1, emb2])
multiway_emb.apply(set_split_position(2))

x = torch.LongTensor([[1,2,3],[4,5,6]])
output = multiway_emb(x)
print(output)
```

**Example 3:** Working with Different Embedding Dimensions
```python
from zeta import MultiwayEmbedding
import torch.nn as nn

emb1 = nn.Embedding(10, 5)
emb2 = nn.Embedding(10, 7)
multiway_emb = MultiwayEmbedding([emb1, emb2], dim=2)

x = torch.LongTensor([[1,2,3],[4,5,6]])
output = multiway_emb(x)
print(output)
```

---

## 5. Additional Tips and Information

- Ensure that the input tensor's dimensions align with the expected embeddings. If there's a mismatch in dimensions, a runtime error will occur.
- The split position determines the point at which the tensor is divided. It's crucial to set this appropriately, especially if the embeddings have different dimensions.
- Using the provided `set_split_position` utility function makes it easy to apply the split position for the embeddings.

---

## 6. References

- Torch documentation: [Link to PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- Agora: [Link to Agora's GitHub](#) (assuming there might be a GitHub link or other resource for Agora)

---

**Note:** Ensure that the tensor operations align mathematically, especially if you're concatenating tensors with different dimensions. In such cases, ensure the embeddings produce tensors that can be concatenated along the specified dimension.

**Mathematical Explanation:** Given an input tensor \( X \) split into \( X_1 \) and \( X_2 \), and two embeddings \( A \) and \( B \), the output is given by concatenating \( A(X_1) \) and \( B(X_2) \).