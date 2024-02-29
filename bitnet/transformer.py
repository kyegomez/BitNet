import torch
from torch import nn
from zeta.nn import MultiheadAttention
import torch.nn.functional as F
from bitnet.bitffn import BitFeedForward


def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)


class RMSNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.0

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale


class Transformer(nn.Module):
    """
    Transformer module that applies multi-head attention and feed-forward layers.

    Args:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        depth (int): The number of transformer layers.
        ff_mult (int, optional): The multiplier for the hidden dimension in the feed-forward layers.
            Defaults to 2.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        layers (nn.ModuleList): List of multi-head attention layers.
        ffn_layers (nn.ModuleList): List of feed-forward layers.

    """

    def __init__(self, dim: int, heads: int, depth: int, ff_mult=2, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(MultiheadAttention(dim, heads))

            self.ffn_layers.append(
                BitFeedForward(dim=dim, ff_mult=ff_mult),
            )

    def forward(self, x):
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x = attn(x, x, x) + x
            x = ffn(x) + x
        return x


# [MAIN MODEL] BitNetTransformer
class BitNetTransformer(nn.Module):
    """
    BitNetTransformer is a transformer-based model for BitNet.

    Args:
        dim (int): The dimension of the token embeddings.
        depth (int): The number of transformer layers.
        num_tokens (int): The number of tokens in the vocabulary.
        heads (int, optional): The number of attention heads in the transformer. Defaults to 8.
        ff_mult (int, optional): The multiplier for the feed-forward layer dimension. Defaults to 4.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_tokens: int,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, ff_mult=ff_mult
        )

        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)
