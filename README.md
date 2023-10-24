[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BitNet
![bitnet](/bitnet.png)
Implementation of the "BitNet: Scaling 1-bit Transformers for Large Language Models"

[Paper link:](https://arxiv.org/pdf/2310.11453.pdf)

BitLinear = tensor -> layernorm -> Binarize -> abs max quantization -> dequant

"The implementation of the BitNet architecture is quite simple, requiring only the replacement of linear projections (i.e., nn.Linear in PyTorch) in the Transformer. " -- BitNet is really easy to implement! 

## **NEWS**
- BitNet Transformer has been trained using the `train.py` file that trains on enwiki8 a small 1gb dataset of wikipedia: [HERE IS THE LINK](https://drive.google.com/file/d/1gBuZRFBqMV3cVD902LXA_hmZl4e0dLyY/view)

## Installation
`pip install bitnet`

## Usage:
- Example of the BitLinear layer which is the main innovation of the paper!
```python
import torch

from bitnet import BitLinear

# Input
x = torch.randn(10, 512)

# BitLinear layer
layer = BitLinear(512, 400)

# Output
y = layer(x)

print(y)

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

## Inference
```python

import gzip
import random

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from bitnet.transformer import BitNetTransformer
from bitnet.at import AutoregressiveWrapper

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

# helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# instantiate GPT-like decoder model

model = BitNetTransformer(num_tokens=256, dim=512, depth=8)

model = AutoregressiveWrapper(model, max_seq_len=SEQ_LEN)
model.load_state_dict(torch.load('../model_checkpoint.pth'))
model.cuda()
model.eval()

inp = torch.from_numpy(np.fromstring("The dog jumped over the ", dtype=np.uint8)).long().to('cuda:0')

sample = model.generate(inp[None, ...], GENERATE_LENGTH)
output_str = decode_tokens(sample[0])
print(output_str)



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
- [ ] Double check BitLinear implementation and make sure it works exactly as in paper 
- [ ] Implement training script for `BitNetTransformer`
- [ ] Train on Enwiki8, copy and past code and data from Lucidrains repos
- [ ] Benchmark peformance
- [ ] Look into Straight Through Estimator for non-differentiable backprop