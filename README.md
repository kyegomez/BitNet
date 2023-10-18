[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BitNet
Implementation of the "BitNet: Scaling 1-bit Transformers for Large Language Models"

[Paper link:](https://arxiv.org/pdf/2310.11453.pdf)

BitLinear = tensor -> layernorm -> Binarize -> abs max quantization 

## Installation
`pip install bitnet`

## Usage:
```python
import torch 
from bitnet import BitLinear


#example
x = torch.randn(10, 512)
layer = BitLinear(512)
y, dequant = layer(x)
print(y, dequant)
```

# License
MIT


# Todo
- [ ] Fix transformer pass error [issue](https://github.com/kyegomez/BitNet/issues/5)

