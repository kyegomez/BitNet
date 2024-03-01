from torch import Tensor, nn

from bitnet.bitlinear import BitLinear


class BitFeedForward(nn.Module):
    """
    BitFeedForward module applies feed-forward transformation to the input tensor.

    Args:
        dim (int): The input dimension.
        ff_mult (int, optional): The multiplier for the hidden dimension. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        layer (nn.Sequential): The sequential layer consisting of BitLinear and nn.GELU layers.

    Methods:
        forward(x: Tensor) -> Tensor: Performs the forward pass of the BitFeedForward module.

    Examples:
        >>> import torch
        >>> from bitnet.bitffn import BitFeedForward
        >>> x = torch.randn(10, 512)
        >>> ff = BitFeedForward(512)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([10, 512])

    """

    def __init__(self, dim: int, ff_mult: int = 4, *args, **kwargs):
        super().__init__()
        hidden_dim = dim * ff_mult

        self.layer = nn.Sequential(
            BitLinear(dim, hidden_dim, *args, **kwargs),
            nn.GELU(),
            BitLinear(hidden_dim, dim, *args, **kwargs),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the BitFeedForward module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the feed-forward transformation.

        """
        return self.layer(x)
