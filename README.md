[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BitNet
![bitnet](/bitnet.png)
Implementation of the "BitNet: Scaling 1-bit Transformers for Large Language Models"

[Paper link:](https://arxiv.org/pdf/2310.11453.pdf)

BitLinear = tensor -> layernorm -> Binarize -> abs max quantization 

## Installation
`pip install bitnet`

## Usage:
```python
import torch 
from bitnet import BitLinear
from bitnet.main import Transformer


#example 1
x = torch.randn(10, 512)
layer = BitLinear(512)
y, dequant = layer(x)
print(y, dequant)

#example 2
x = torch.randn(1, 1, 10, 512)
layer = Transformer(512, 8, 8, 64)
y = layer(x)
print(y)
```

----

- Full BitNet Transformer as shown in paper:
```python
x = torch.randn(10, 512)
layer = Transformer(512, 8, 8, 64)
y = layer(x)
print(y)
```

# License
MIT


# Todo
- [ ] Fix transformer pass error [issue](https://github.com/kyegomez/BitNet/issues/5)

