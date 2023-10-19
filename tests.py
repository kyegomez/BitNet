import pytest
import torch
from torch.nn import functional as F

from bitnet.bitlinear import BitLinear, absmax_quantize
from bitnet.transformer import BitNetTransformer, ParallelTransformerBlock, Transformer

# Basic Tests


def test_absmax_quantize():
    """
    Test the absmax_quantize function for a given tensor.
    Check if the quantized tensor is of dtype int8 and if
    the dequantized value is close to the original tensor.
    """
    x = torch.tensor([2.5, -3.0, 4.0, -5.0])
    quant, dequant = absmax_quantize(x)
    assert isinstance(quant, torch.Tensor) and quant.dtype == torch.int8
    assert torch.allclose(dequant, x, atol=1e-2)


def test_bit_linear_forward():
    """
    Test the forward method of the BitLinear class.
    Ensure the output shape matches the input shape.
    """
    x = torch.randn(10, 512)
    layer = BitLinear(512)
    output = layer(x)
    assert x.shape == output.shape


# Parameterized Testing


@pytest.mark.parametrize("input_size", [128, 256, 512, 1024])
def test_bit_linear_different_input_sizes(input_size):
    """
    Test the BitLinear class with different input sizes.
    Ensure the output shape matches the input shape for each size.
    """
    x = torch.randn(10, input_size)
    layer = BitLinear(input_size)
    output = layer(x)
    assert x.shape == output.shape


# Exception Testing


def test_absmax_quantize_wrong_dtype():
    """
    Test the absmax_quantize function with a tensor of incorrect dtype.
    Ensure it raises an appropriate exception.
    """
    x = torch.tensor([2, 3, 4, 5], dtype=torch.int32)
    with pytest.raises(
        ValueError
    ):  # This can be changed to the appropriate expected exception type
        absmax_quantize(x)


# ================================ Transformer with bitlinear ================================


@pytest.fixture
def random_tensor():
    """A fixture to generate a random tensor"""
    return torch.randn(32, 512)


@pytest.fixture
def bitnet_model():
    """A fixture to create an instance of BitNetTransformer model"""
    return BitNetTransformer(
        num_tokens=20000,
        dim=512,
        depth=6,
        dim_head=64,
        heads=8,
        ff_mult=4,
    )


@pytest.mark.parametrize(
    "dim, dim_head, heads, ff_mult",
    [
        (512, 64, 8, 4),
        (256, 32, 4, 2),
        (128, 16, 2, 1),
    ],
)
def test_parallel_transformer_block(dim, dim_head, heads, ff_mult, random_tensor):
    block = ParallelTransformerBlock(dim, dim_head, heads, ff_mult)
    output = block(random_tensor)
    assert output.shape == random_tensor.shape


@pytest.mark.parametrize(
    "dim, depth, heads, dim_head, ff_mult",
    [
        (512, 6, 8, 64, 4),
        (256, 3, 4, 32, 2),
        (128, 2, 2, 16, 1),
    ],
)
def test_transformer(dim, depth, heads, dim_head, ff_mult, random_tensor):
    transformer = Transformer(dim, depth, heads, dim_head, ff_mult)
    output = transformer(random_tensor)
    assert output.shape == random_tensor.shape


def test_bitnet_transformer_forward(bitnet_model):
    tokens = torch.randint(0, 20000, (1, 512))
    logits = bitnet_model(tokens)
    assert logits.shape == (1, 20000)


def test_parallel_transformer_block_masking(random_tensor):
    block = ParallelTransformerBlock(512, 64, 8, 4)
    mask1 = block.get_mask(100, random_tensor.device)
    mask2 = block.get_mask(200, random_tensor.device)
    assert mask1.shape == (100, 100)
    assert mask2.shape == (200, 200)


def test_bitnet_transformer_embed(bitnet_model):
    tokens = torch.randint(0, 20000, (1, 512))
    embedded = bitnet_model.emb(tokens)
    assert embedded.shape == (1, 512, 512)


@pytest.mark.parametrize(
    "dim, dim_head, heads, ff_mult",
    [
        (512, 64, 8, 4),
        (256, 32, 4, 2),
        (128, 16, 2, 1),
    ],
)
def test_parallel_transformer_block_raises_for_incorrect_input(
    dim, dim_head, heads, ff_mult
):
    block = ParallelTransformerBlock(dim, dim_head, heads, ff_mult)
    with pytest.raises(Exception):
        block(torch.randn(32, 100))


@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 512),
        (32, 128),
        (64, 256),
    ],
)
def test_bitnet_transformer_for_various_input_shapes(bitnet_model, batch_size, seq_len):
    tokens = torch.randint(0, 20000, (batch_size, seq_len))
    logits = bitnet_model(tokens)
    assert logits.shape == (batch_size, 20000)


def test_rotary_embedding(bitnet_model, random_tensor):
    block = ParallelTransformerBlock(512, 64, 8, 4)
    rotary_emb1 = block.get_rotary_embedding(100, random_tensor.device)
    rotary_emb2 = block.get_rotary_embedding(200, random_tensor.device)
    assert rotary_emb1.shape == (100, 64)
    assert rotary_emb2.shape == (200, 64)


@pytest.mark.parametrize("mask_value", [100, 200, 300])
def test_mask_persistency(random_tensor, mask_value):
    block = ParallelTransformerBlock(512, 64, 8, 4)
    block.get_mask(mask_value, random_tensor.device)
    assert block.mask.shape[0] == mask_value


@pytest.mark.parametrize(
    "input_value, expected_output_shape",
    [
        (torch.randint(0, 20000, (1, 512)), (1, 20000)),
        (torch.randint(0, 20000, (32, 256)), (32, 20000)),
    ],
)
def test_bitnet_transformer_output_shapes(
    bitnet_model, input_value, expected_output_shape
):
    logits = bitnet_model(input_value)
    assert logits.shape == expected_output_shape


def test_exceptions_on_wrong_dtype():
    block = ParallelTransformerBlock(512, 64, 8, 4)
    with pytest.raises(Exception):
        block(torch.randn(32, 512).int())


def test_bitnet_transformer_logit_values(bitnet_model):
    tokens = torch.randint(0, 20000, (1, 512))
    logits = bitnet_model(tokens)
    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0))


# Mocking and Monkeypatching


def test_mocking_get_mask(monkeypatch, random_tensor):
    mock_mask = torch.zeros(100, 100)
    monkeypatch.setattr(
        ParallelTransformerBlock, "get_mask", lambda self, n, device: mock_mask
    )
    block = ParallelTransformerBlock(512, 64, 8, 4)
    assert torch.equal(block.get_mask(100, random_tensor.device), mock_mask)


# Add more tests based on the scenarios and edge cases you want to cover.
