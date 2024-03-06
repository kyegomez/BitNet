import torch
from torch import Tensor, nn


class BitLinear(nn.Linear):
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
        self.num_groups = num_groups
        self.eps = 1e-5
        self.norm = nn.LayerNorm(in_features)

        # Quantiziation and dequantization
        self.Q_b = 2 ** (b - 1)
        self.beta = torch.zeros((1, self.weight.shape[0]))
        self.gamma = torch.zeros((1, self.weight.shape[0]))

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
            self.beta[start_idx:end_idx] = weight_group.abs().mean()
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
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            gamma_g = activation_group.abs().max()
            self.gamma[start_idx:end_idx] = gamma_g
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * self.Q_b / (gamma_g + self.eps),
                -self.Q_b + self.eps,
                self.Q_b - self.eps,
            )

        return quantized_x

    def dequantize_activations_groupwise(self, x):
        """
        Dequantizes the activations of the layer in a group-wise manner.

        Args:
            x (Tensor): Quantized input tensor.

        Returns:
            Tensor: Dequantized activations tensor.
        """
        return x * self.gamma * self.beta / self.Q_b

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Normalize input
        x = self.norm(x)

        # Binarize weights and quantize activations
        binarized_weights = self.binarize_weights_groupwise()

        # Quantize input
        x_quant = self.quantize_activations_groupwise(x)

        # Perform linear transformation
        output = torch.nn.functional.linear(x_quant, binarized_weights, self.bias)

        # Dequantize activations
        output = self.dequantize_activations_groupwise(output)

        # Return output
        return output


if __name__ == "__main__":
    # Example usage
    bitlinear = BitLinear(10, 6, num_groups=2, b=8)
    input_tensor = torch.randn(6, 10)  # Example input tensor
    output = bitlinear(input_tensor)
    print(output)  # Example output tensor