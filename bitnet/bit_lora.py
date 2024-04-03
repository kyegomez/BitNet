import torch
from torch import nn, Tensor
from bitnet.bitlinear import BitLinear
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm


def weight_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


class BitLora(BitLinear):
    """
    BitLora class represents a custom linear layer with LoRa (Low Rank) regularization.

    Args:
        rank (int): The rank of the LoRa regularization. Default is 4.
        lora_alpha (int): The scaling factor for LoRa regularization. Default is 1.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        rank (int): The rank of the LoRa regularization.
        lora_alpha (int): The scaling factor for LoRa regularization.
        scaling (float): The scaling factor for LoRa regularization.
        merged (bool): Indicates whether the LoRa regularization has been merged with the weight matrix.
        lora_a (nn.Parameter): The learnable parameter matrix of shape (in_features, rank).
        lora_b (nn.Parameter): The learnable parameter matrix of shape (rank, out_features).

    Examples:

    """

    def __init__(self, rank: int = 4, lora_alpha: int = 1, *args, **kwargs):
        super(BitLora, self).__init__(*args, **kwargs)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.rank
        self.merged = False

        self.lora_a = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, self.out_features))

        # Rmsnorm
        self.rms_norm = SimpleRMSNorm(self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLora layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight

        # Normalize the input tensor
        x_norm = self.rms_norm(x)

        # Activation Quant
        x_quant = activation_quant(x_norm)

        if not self.merged and self.rank > 0:
            lora = self.lora_a @ self.lora_b
            w = w + lora * self.scaling

        # w_quant, scale = weight_quant(w)
        w_quant = weight_quant(w)
        # scale = weight_quant(w)
        scale = 1.0
        output = nn.functional.linear(x_quant, w_quant, self.bias)
        return output * scale

    def merge(self):
        """
        Merge the LoRa regularization with the weight matrix.

        """
        if not self.merged and self.rank > 0:
            self.weight.data += (self.lora_b @ self.lora_a) * self.scaling
            self.merged = True
