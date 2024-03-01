import numpy as np
import torch

from bitnet.at import AutoregressiveWrapper
from bitnet.bit_transformer import BitNetTransformer


class BitNetInference:
    """
    A class used to perform inference with the BitNetTransformer model.

    ...

    Attributes
    ----------
    model : torch.nn.Module
        an instance of the BitNetTransformer model
    device : str
        the device to run the model on ('cpu' or 'cuda')

    Methods
    -------
    load_model(model_path)
        Loads a trained model from a .pth file.
    generate(input_str, length)
        Generates a sequence of tokens based on the input string.
    """

    def __init__(self, device="cuda"):
        """
        Parameters
        ----------
        device : str, optional
            The device to run the model on ('cpu' or 'cuda'). By default, 'cuda' is used.
        """
        self.device = device
        self.model = BitNetTransformer(num_tokens=256, dim=512, depth=8)
        self.model = AutoregressiveWrapper(self.model, max_seq_len=1024)
        self.model.to(self.device)

    def load_model(self, model_path):
        """Loads a trained model from a .pth file."""
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    @staticmethod
    def decode_token(token):
        """Decodes a token into a character."""
        return str(chr(max(32, token)))

    @staticmethod
    def decode_tokens(tokens):
        """Decodes a sequence of tokens into a string."""
        return "".join(list(map(BitNetInference.decode_token, tokens)))

    def generate(self, input_str, length):
        """Generates a sequence of tokens based on the input string."""
        inp = (
            torch.from_numpy(np.fromstring(input_str, dtype=np.uint8))
            .long()
            .to(self.device)
        )
        sample = self.model.generate(inp[None, ...], length)
        output_str = self.decode_tokens(sample[0])
        return output_str
