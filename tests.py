import pytest
import torch
from torch.nn import functional as F

from bitnet.bitlinear import BitLinear, absmax_quantize
from bitnet.transformer import BitNetTransformer, ParallelTransformerBlock, Transformer

# Basic Tests:

def test_absmax_quantize():
    tensor = torch.tensor([1.5, -2.0, 3.0, -4.0])
    quant, dequant = absmax_quantize(tensor)
    assert quant.dtype == torch.int8
    assert torch.allclose(dequant, tensor, atol=1e-2)

def test_bitlinear_initialization():
    layer = BitLinear(10, 20)
    assert layer.in_features == 10
    assert layer.out_features == 20
    assert layer.weight.shape == (20, 10)

def test_bitlinear_forward():
    layer = BitLinear(10, 20)
    input_tensor = torch.randn(5, 10)
    output = layer(input_tensor)
    assert output.shape == (5, 20)

# Fixtures:

@pytest.fixture
def random_tensor():
    return torch.randn(5, 10)

# Parameterized Testing:

@pytest.mark.parametrize("bits", [4, 8, 12, 16])
def test_absmax_quantize_bits(random_tensor, bits):
    quant, dequant = absmax_quantize(random_tensor, bits=bits)
    assert quant.dtype == torch.int8
    assert torch.allclose(dequant, random_tensor, atol=1e-2)

# More Tests for BitLinear:

@pytest.mark.parametrize("in_features,out_features", [(10, 20), (20, 40), (5, 10), (15, 10)])
def test_bitlinear_shapes(in_features, out_features):
    layer = BitLinear(in_features, out_features)
    assert layer.weight.shape == (out_features, in_features)

@pytest.mark.parametrize("groups", [1, 2, 5])
def test_bitlinear_groups(groups):
    layer = BitLinear(10, 20, groups=groups)
    assert layer.groups == groups

def test_bitlinear_reset_parameters():
    layer = BitLinear(10, 20)
    original_weights = layer.weight.clone()
    layer.reset_parameters()
    assert not torch.equal(original_weights, layer.weight)

@pytest.mark.parametrize("groups", [1, 2, 5])
def test_bitlinear_forward_with_groups(random_tensor, groups):
    layer = BitLinear(10, 20, groups=groups)
    output = layer(random_tensor)
    assert output.shape == (5, 20)

def test_bitlinear_zero_input():
    layer = BitLinear(10, 20)
    input_tensor = torch.zeros(5, 10)
    output = layer(input_tensor)
    assert torch.allclose(output, torch.zeros(5, 20), atol=1e-2)

def test_bitlinear_weight_sign():
    layer = BitLinear(10, 20)
    input_tensor = torch.randn(5, 10)
    output_before = layer(input_tensor)
    layer.weight.data = torch.abs(layer.weight.data)
    output_after = layer(input_tensor)
    assert not torch.allclose(output_before, output_after)

@pytest.mark.parametrize("groups", [1, 2, 5])
def test_bitlinear_weight_group_normalization(groups):
    layer = BitLinear(10, 20, groups=groups)
    weight = layer.weight.view(groups, -1)
    mean = weight.mean(dim=1, keepdim=True)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-2)

def test_bitlinear_weight_group_scaling():
    layer = BitLinear(10, 20, groups=5)
    weight = layer.weight.view(layer.groups, -1)
    beta = torch.abs(weight).sum(dim=1, keepdim=True) / (weight.shape[0] * weight.shape[1])
    scaled_weight = weight * beta
    assert torch.allclose(scaled_weight, layer.weight.view(20, 10))

def test_bitlinear_input_quantization(random_tensor):
    layer = BitLinear(10, 20)
    quant_input, _ = absmax_quantize(random_tensor)
    output = layer(quant_input.float())
    assert output.shape == (5, 20)

# ... Continue adding more tests ...
# - Test the forward pass with extreme input values.
# - Test with different types of input tensors (e.g., int8, float16).
# - Test the forward pass with batch sizes other than 5.
# - Verify that using different initializations produces different results.
# - Test the weight and input interactions during the forward pass.
# - And many more...

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
