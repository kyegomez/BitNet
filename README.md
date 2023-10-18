[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BitNet
![bitnet](/bitnet.png)
Implementation of the "BitNet: Scaling 1-bit Transformers for Large Language Models"

[Paper link:](https://arxiv.org/pdf/2310.11453.pdf)

BitLinear = tensor -> layernorm -> Binarize -> abs max quantization 

## Installation
`pip install bitnet`

## Usage:
- Example of the BitLinear layer which is the main innovation of the paper!
```python
import torch 
from bitnet import BitLinear

# random inputs
x = torch.randn(10, 512)

#apply linear
layer = BitLinear(512)

#layer
y, dequant = layer(x)

#print
print(y, dequant)

```
----

- Running an example to a full BitNet Transformer as shown in paper:
```python
import torch 
from bitnet.main import BitNetTransformer

#random inputs
x = torch.randn(10, 512)

#transformer layer
model = BitNetTransformer(512, 8, 8, 64)

#apply transformer
y = model(x)

print(y)
```

# License
MIT


# Todo
- [ ] Fix transformer pass error [issue](https://github.com/kyegomez/BitNet/issues/5)

