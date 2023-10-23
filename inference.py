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
