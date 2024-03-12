import torch
from torch import Tensor, nn


def absmean_quantize_weights(weights):
    """
    Quantizes the weights to -1, 0, or +1 using an absmean quantization function.

    Parameters:
    - weights (Tensor): The weights of a neural network layer.

    Returns:
    - Tensor: The quantized weights.
    """
    # Calculate the average absolute value (γ) of the weights
    gamma = torch.mean(torch.abs(weights))

    # Scale weights by γ and round to the nearest integer among {-1, 0, +1}
    quantized_weights = torch.clamp(torch.round(weights / gamma), min=-1, max=1)
    # quantized_weights = torch.clamp(torch.round(weights / (gamma + 1e-5)), min=-1, max=1)

    return quantized_weights


class BitLinearNew(nn.Linear):
    """
    BitLinear is a custom linear layer that performs binarization of weights and quantization of activations
    in a group-wise manner.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        num_groups (int, optional): Number of groups to divide the weights and activations into. Default is 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_groups: int = 1,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.b = b
        self.num_groups = num_groups
        self.eps = 1e-5
        self.norm = nn.LayerNorm(in_features)

    def ste(self, x):
        """
        Applies the sign function for binarization and uses Straight-Through Estimator (STE) during backward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Binarized tensor.
        """
        binarized_x = torch.sign(x)
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        """
        Binarizes the weights of the layer in a group-wise manner using STE.

        Returns:
            Tensor: Binarized weights tensor.
        """
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste(weight_group - alpha_g)

        return binarized_weights

    def quantize_activations_groupwise(self, x):
        """
        Quantizes the activations of the layer in a group-wise manner.

        Args:
            x (Tensor): Input tensor.
            b (int, optional): Number of bits for quantization. Default is 8.

        Returns:
            Tensor: Quantized activations tensor.
        """
        Q_b = 2 ** (self.b - 1)

        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def dequantize_activations_groupwise(self, x):
        """
        Dequantizes the activations of the layer in a group-wise manner.

        Args:
            x (Tensor): Quantized input tensor.
            b (int, optional): Number of bits used during the quantization. Default is 8.

        Returns:
            Tensor: Dequantized activations tensor.
        """
        Q_b = 2 ** (self.b - 1)
        dequantized_x = torch.zeros_like(x)
        for g in range(self.num_groups):
            start_idx = g * x.shape[0] // self.num_groups
            end_idx = (g + 1) * x.shape[0] // self.num_groups
            quantized_group = x[start_idx:end_idx]
            gamma_g = quantized_group.abs().max()
            dequantized_x[start_idx:end_idx] = quantized_group * gamma_g / Q_b
        return dequantized_x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        
        # Binarize weights and quantize activations
        binarized_weights = self.binarize_weights_groupwise()

        # Perform linear transformation
        output = torch.nn.functional.linear(x, binarized_weights, self.bias)
    
        # Quantize activations
        output = absmean_quantize_weights(output)
        print(output)

        # Dequantize activations
        # output = self.dequantize_activations_groupwise(output)

        # Return output
        return output


# # Forward pass
# x = torch.randn(16, 10)
# layer = BitLinearNew(10, 20, num_groups=2)
# output = layer(x)
# print(output)