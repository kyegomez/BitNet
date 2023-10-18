# BaseAttention Abstract Class
============================

The `BaseAttention` class is an abstract base class that defines the interface for all attention mechanisms. It includes the basic structure and methods that all attention mechanisms should have.

```python
from abc import  abstractmethod
import torch.nn as nn

class BaseAttention(nn.Module):
    @abstractmethod
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    @abstractmethod
    def forward(self, x, context=None, mask=None):
        pass
```


## Usage
-----------------------

The `FlashAttentionTwo` class extends the `BaseAttention` abstract base class and implements the specific attention mechanism.

```python
class FlashAttentionTwo(BaseAttention):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = False,
        q_bucket_size = 512,
        k_bucket_size = 1024,
        parallel = False,
        mixed_precision = False
    ):
        super().__init__(dim, heads, dim_head)
        self.causal = causal
        self.parallel = parallel
        self.mixed_precision = mixed_precision
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size
        # ... rest of the implementation ...

    def forward(
        self,
        x,
        context = None,
        mask = None,
        q_bucket_size = None,
        k_bucket_size = None,
    ):
        # ... implementation of the forward method ...
```


## Rules for Using the BaseAttention Class
---------------------------------------

1.  Any class that extends the `BaseAttention` class must implement the `forward` method. This method defines how the attention mechanism operates.

2.  The `__init__` method of the `BaseAttention` class takes three parameters: `dim`, `heads`, and `dim_head`. Any class that extends `BaseAttention` should pass these parameters to the `__init__` method of the base class.

3.  The `forward` method of the `BaseAttention` class takes three parameters: `x`, `context`, and `mask`. Any class that extends `BaseAttention` should include these parameters in its `forward` method.

---

## Example of Using the FlashAttentionTwo Class
--------------------------------------------

```python
from zeta import FlashAttentionTwo

# Create an instance of the FlashAttentionTwo class
attention = FlashAttentionTwo(dim=512, heads=8, dim_head=64)

# Create some input data
x = torch.randn(1, 10, 512)

# Apply the attention mechanism
out = attention(x)
```


In this example, we first create an instance of the `FlashAttentionTwo` class. We then create some input data `x` and apply the attention mechanism to this data by calling the `forward` method of the `attention` instance.