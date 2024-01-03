import torch
import torch.nn as nn
import unittest
from bitnet import BitLinear, BitFeedForward


class TestBitFeedForwardArgsKwargs(unittest.TestCase):
    def setUp(self):
        self.dim = 512
        self.ff_mult = 4
        self.args = (0.1,)
        self.kwargs = {"bias": False}
        self.bitffn = BitFeedForward(self.dim, self.ff_mult, *self.args, **self.kwargs)

    def test_bitffn_creation_with_args_kwargs(self):
        self.assertIsInstance(self.bitffn, nn.Sequential)
        self.assertEqual(len(self.bitffn), 3)
        self.assertIsInstance(self.bitffn[0], BitLinear)
        self.assertIsInstance(self.bitffn[1], nn.GELU)
        self.assertIsInstance(self.bitffn[2], BitLinear)

    def test_bitffn_dimensions_with_args_kwargs(self):
        self.assertEqual(self.bitffn[0].in_features, self.dim)
        self.assertEqual(self.bitffn[0].out_features, self.dim * self.ff_mult)
        self.assertEqual(self.bitffn[2].in_features, self.dim * self.ff_mult)
        self.assertEqual(self.bitffn[2].out_features, self.dim)

    def test_bitffn_forward_pass_with_args_kwargs(self):
        x = torch.randn(1, 100, self.dim)
        out = self.bitffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_bitffn_args_kwargs(self):
        self.assertEqual(
            self.bitffn[0].weight.data.shape, (self.dim * self.ff_mult, self.dim)
        )
        self.assertEqual(
            self.bitffn[2].weight.data.shape, (self.dim, self.dim * self.ff_mult)
        )
        self.assertIsNone(self.bitffn[0].bias)
        self.assertIsNone(self.bitffn[2].bias)
