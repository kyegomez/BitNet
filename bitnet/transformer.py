import torch
from torch import nn
from zeta.nn.attention import MultiQueryAttention
from bitnet.bitffn import BitFeedForward
from zeta.nn import RMSNorm
# helpers


# [TRANSFORMER] Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        ff_mult=2,
        *args,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        
        
        for _ in range(depth):    
            self.layers.append(
                MultiQueryAttention(d_model=dim, heads=heads)
            )
            
            self.ffn_layers.append(
                BitFeedForward(dim=dim, ff_mult=ff_mult),
            )
            

    def forward(self, x):
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x, _, _ = attn(x, x, x) + x
            x = ffn(x) + x
        return x


# [MAIN MODEL] BitNetTransformer
class BitNetTransformer(nn.Module):
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

        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, ff_mult=ff_mult)

        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)
    
    
