import torch
import torch.nn.functional as F
from torch import nn


class BitLinear(nn.Module):
    """
    BitLinear module as described in the BitNet architecture.

    This module performs a linear transformation with 1-bit quantized weights.
    The transformation includes a quantization step, matrix multiplication,
    and a subsequent dequantization step. Both the quantization and
    dequantization steps utilize learnable parameters gamma and beta.

    Attributes:
    - in_features: size of each input sample
    - out_features: size of each output sample
    - gamma: scaling factor for absmax quantization (learnable parameter)
    - beta: scaling factor for dequantization (learnable parameter)
    - weight: the 1-bit quantized weights of the linear transformation
    - bias: the bias term for the linear transformation (optional)
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the BitLinear module.

        Parameters:
        - in_features: An integer, the number of input features.
        - out_features: An integer, the number of output features.
        - bias: A boolean, whether the layer includes a bias.
        """
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable parameters for quantization and dequantization
        self.gamma = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, input):
        """
        Forward pass of the BitLinear module.

        Parameters:
        - input: A tensor of shape (batch_size, in_features).

        Returns:
        - output: A tensor of shape (batch_size, out_features).
        """
        # Apply Layer Normalization
        input_norm = F.layer_norm(input, (self.in_features,))

        # Absmax Quantization
        quant_scale = torch.max(torch.abs(input_norm), dim=1, keepdim=True).values
        input_quant = torch.sign(input_norm) * (quant_scale / self.gamma)

        # 1-bit Weights Quantization
        weight_quant = torch.sign(self.weight)

        # MatMul with 1-bit weights using torch.matmul for explicit operation
        output = torch.matmul(input_quant, weight_quant.t())

        # Adding bias if it exists
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)

        # Dequantization with learnable parameters
        output = output * self.beta.unsqueeze(0).expand_as(output)

        return output
