import torch
import torch.nn.functional as F
from torch import nn
from zeta.nn.attention.attend import Attend


def absmax_quantize(x):
    #calculate scale
    scale = 127 / torch.max(torch.abs(x))

    #quantize
    quant = (scale * x).round()

    #dequantize
    dequant = quant / scale


    return quant.to(torch.int8), dequant


class BitLinear(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.abs_max_quantization = absmax_quantize

    def forward(self, x):
        x = self.norm(x)

        #Binarize the weights
        weight = self.linear.weight
        weight_binarized = torch.sign(weight)

        #Apply the linear operation with the binarized weights
        x = F.linear(x, weight_binarized, self.linear.bias)

        #quantize the output
        x, dequant = self.abs_max_quantization(x)

        #dequant the output
        dequant = dequant * torch.norm(weight) / (self.dim ** -0.5)

        return x, dequant
    

def FeedForward(dim, dropout=0.):
    """
    Feedforward network for Transformer with BitLinear layers instead.

    Args:
        dim: int, dimension of the input.
        dropout: float, dropout rate.
    
    Returns:
        nn.Sequential, feedforward network.

    Usage:
        >>> x = torch.randn(10, 512)
        >>> ff = FeedForward(512)
        >>> y = ff(x)
        >>> print(y)
    
    """
    return nn.Sequential(
        nn.LayerNorm(dim),
        BitLinear(dim),
        nn.GELU(),
        nn.Dropout(dropout),
        BitLinear(dim),
        nn.Dropout(dropout),
    )

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        dropout=0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attend(dropout=dropout, heads=heads, flash=False),
                FeedForward(dim, dropout)
            ]))
        self.norm = nn.LayerNorm(dim)
        
        self.bitlinear = BitLinear(dim)


    def forward(
        self,
        x,
        mask = None,
        # attn_mask = None
    ):
        for attn, ff in self.layers:
            q = self.bitlinear(x)
            k = self.bitlinear(x)
            v = self.bitlinear(x)

            x = attn(q, k, v, mask=mask) + x

            x = self.bitlinear(x)

            x = ff(x) + x
        
        return self.norm(x)

#example
# x = torch.randn(10, 512)
# layer = Transformer(512, 8, 8, 64)
# y = layer(x)
# print(y)


# #random tensors
x = torch.randn(10, 512)
feedforward = FeedForward(512)
out = feedforward(x)
print(out)