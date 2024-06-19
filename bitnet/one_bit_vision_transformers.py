import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from bitnet.bitlinear import BitLinear
from zeta import MultiQueryAttention

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper

# in latest tweet, seem to claim more stable training at higher learning rates
# unsure if this has taken off within Brain, or it has some hidden drawback


class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim) / self.scale)

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            BitLinear(dim, hidden_dim),
            nn.GELU(),
            BitLinear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [MultiQueryAttention(dim, heads), FeedForward(dim, mlp_dim)]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x, _, _ = attn(x)
            x = self.norm(x) + x
            x = ff(x) + x
        return self.norm(x)


class OneBitViT(nn.Module):
    """
    OneBitViT is a vision transformer model for image classification tasks.

    Args:
        image_size (int or tuple): The size of the input image. If an integer is provided, it is assumed to be a square image.
        patch_size (int or tuple): The size of each patch in the image. If an integer is provided, it is assumed to be a square patch.
        num_classes (int): The number of output classes.
        dim (int): The dimensionality of the token embeddings and the positional embeddings.
        depth (int): The number of transformer layers.
        heads (int): The number of attention heads in the transformer.
        mlp_dim (int): The dimensionality of the feed-forward network in the transformer.
        channels (int): The number of input channels in the image. Default is 3.
        dim_head (int): The dimensionality of each attention head. Default is 64.

    Attributes:
        to_patch_embedding (nn.Sequential): Sequential module for converting image patches to embeddings.
        pos_embedding (torch.Tensor): Positional embeddings for the patches.
        transformer (Transformer): Transformer module for processing the embeddings.
        pool (str): Pooling method used to aggregate the patch embeddings. Default is "mean".
        to_latent (nn.Identity): Identity module for converting the transformer output to the final latent representation.
        linear_head (nn.LayerNorm): Layer normalization module for the final linear projection.

    Methods:
        forward(img): Performs a forward pass through the OneBitViT model.

    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            BitLinear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.LayerNorm(dim)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


# import torch
# from bitnet import SimpleViT

# v = OneBitViT(
#     image_size=256,
#     patch_size=32,
#     num_classes=1000,
#     dim=1024,
#     depth=6,
#     heads=16,
#     mlp_dim=2048,
# )

# img = torch.randn(1, 3, 256, 256)

# preds = v(img)  # (1, 1000)
# print(preds)
