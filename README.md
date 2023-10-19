[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BitNet
![bitnet](/bitnet.png)
Implementation of the "BitNet: Scaling 1-bit Transformers for Large Language Models"

[Paper link:](https://arxiv.org/pdf/2310.11453.pdf)

BitLinear = tensor -> layernorm -> Binarize -> abs max quantization 

"The implementation of the BitNet architecture is quite simple, requiring only the replacement of linear projections (i.e., nn.Linear in PyTorch) in the Transformer. " -- BitNet is really easy to implement! 

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

- Running random inputs to a full BitNet Transformer as shown in paper:
```python
import torch
from bitnet import BitNetTransformer

bitnet = BitNetTransformer(
    num_tokens=20000,
    dim=512,
    depth=6,
    dim_head=64,
    heads=8,
    ff_mult=4,
)

tokens = torch.randint(0, 20000, (1, 512))
logits = bitnet(tokens)
print(logits.shape)

```

# License
MIT

# Citation
```bibtex
@misc{2310.11453,
Author = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
Title = {BitNet: Scaling 1-bit Transformers for Large Language Models},
Year = {2023},
Eprint = {arXiv:2310.11453},
}

```


# Todo
- [ ] Fix transformer pass error [issue](https://github.com/kyegomez/BitNet/issues/5)
- [ ] Split up q, k, v in one line 