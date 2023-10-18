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

- Running random inputs to a full BitNet Transformer as shown in paper:
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

