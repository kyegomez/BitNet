import torch
from torch import nn
from bitnet.bitlinear import BitLinear
import torch.nn.functional as F


# Expert module
class Expert(nn.Module):
    """An MLP is a simple linear layer followed by a non-linearity i.e. each Expert

    Args:
        dim (int): The input dimension of the linear layer.
        dropout (float, optional): The dropout probability. Defaults to 0.1.

    Attributes:
        net (nn.Sequential): The sequential network consisting of linear layers, ReLU activation, and dropout.

    """

    def __init__(self, dim: int, dropout: int = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(dim, 4 * dim),
            nn.ReLU(),
            BitLinear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Changing the above to accomodate noisy top-k gating
class NoisyTopkRouter(nn.Module):
    """
    A class representing a Noisy Top-k Router module.

    This module takes the output tensor from a multihead self attention block and performs routing
    by selecting the top-k experts based on the logits. It adds scaled unit Gaussian noise to the logits
    and applies softmax to obtain the final router output.

    Args:
        dim (int): The input dimension of the tensor.
        num_experts (int): The number of experts.
        top_k (int): The number of experts to select.

    Attributes:
        top_k (int): The number of experts to select.
        topkroute_linear (BitLinear): The linear layer for router logits.
        noise_linear (BitLinear): The linear layer for noise logits.
    """

    def __init__(self, dim, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = BitLinear(dim, num_experts)
        self.noise_linear = BitLinear(dim, num_experts)

    def forward(self, mh_output):
        """
        Forward pass of the NoisyTopkRouter module.

        Args:
            mh_output (torch.Tensor): The output tensor from the multihead self attention block.

        Returns:
            tuple: A tuple containing the router output tensor and the indices of the selected experts.
        """
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class BitMoE(nn.Module):
    """
    BitMoE (Bitwise Mixture of Experts) module.

    Args:
        dim (int): The input dimension.
        num_experts (int): The number of experts in the mixture.
        top_k (int, optional): The number of experts to select for each input. Defaults to 2.
    """
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super(BitMoE, self).__init__()
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


# x = torch.randn(2, 4, 8)
# model = BitMoE(8, 4, 2)
# output = model(x)
# print(output)