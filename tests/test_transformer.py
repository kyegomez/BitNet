import torch

from bitnet.transformer import BitFeedForward, BitNetTransformer, MultiheadAttention


def test_bitnet_transformer_initialization():
    bitnet = BitNetTransformer(num_tokens=20000, dim=512, heads=8, depth=6, ff_mult=4)
    assert len(bitnet.layers) == 6
    assert len(bitnet.ffn_layers) == 6
    assert all(isinstance(layer, MultiheadAttention) for layer in bitnet.layers)
    assert all(isinstance(layer, BitFeedForward) for layer in bitnet.ffn_layers)


def test_bitnet_transformer_forward_pass():
    bitnet = BitNetTransformer(num_tokens=20000, dim=512, heads=8, depth=6, ff_mult=4)
    x = torch.randn(1, 100, 512)
    out = bitnet(x)
    assert out.shape == x.shape
