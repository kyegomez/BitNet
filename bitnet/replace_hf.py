from torch import nn
from bitnet.bitbnet_b158 import BitLinear15b
from transformers import AutoModel


def replace_linears_in_hf(model):
    """
    Replaces all instances of nn.Linear in the given model with BitLinear15b.

    Args:
        model (nn.Module): The model to modify.

    Returns:
        None
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace the nn.Linear with BitLinear matching in features and and out_features, and add it to the model
            setattr(
                model,
                name,
                BitLinear15b(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                ),
            )
        else:
            # Recursively apply to child modules
            replace_linears_in_hf(module)


# # Example usage:
# # Load a model from Hugging Face's Transformers
# model_name = "bert-base-uncased"  # Example model
# model = AutoModel.from_pretrained(model_name)

# # Replace its Linear layers with BitLinear
# replace_linears_in_hf(model)

# # Now, `model` has its Linear layers replaced with BitLinear
# print(model)
