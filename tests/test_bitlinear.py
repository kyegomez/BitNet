import torch
from bitnet.bitlinear import BitLinear

def test_bitlinear_initialization():
    bitlinear = BitLinear(in_features=512, out_features=256, bias=True)
    assert bitlinear.in_features == 512
    assert bitlinear.out_features == 256
    assert bitlinear.weight.shape == (256, 512)
    assert bitlinear.bias.shape == (256,)
    assert bitlinear.gamma.shape == (512,)
    assert bitlinear.beta.shape == (256,)

def test_bitlinear_forward_pass():
    bitlinear = BitLinear(in_features=512, out_features=256, bias=True)
    x = torch.randn(1, 512)
    out = bitlinear(x)
    assert out.shape == (1, 256)

def test_bitlinear_no_bias():
    bitlinear = BitLinear(in_features=512, out_features=256, bias=False)
    assert bitlinear.bias is None

def test_bitlinear_quantization():
    bitlinear = BitLinear(in_features=512, out_features=256, bias=True)
    x = torch.randn(1, 512)
    out = bitlinear(x)
    assert torch.all(out <= bitlinear.beta.unsqueeze(0).expand_as(out))
    assert torch.all(out >= -bitlinear.beta.unsqueeze(0).expand_as(out))