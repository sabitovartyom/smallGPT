import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smallGPT.layers.attantion import CausalSelfAttantion, MultiHeadCausalSelfAttantion


class AttentionTests(unittest.TestCase):
    def test_outputs_and_gradients_match_torch_attention(self):
        torch.manual_seed(42)
        for heads in (1, 2, 4):
            for shape in ((5, 8), (2, 5, 8), (2, 1, 8)):
                with self.subTest(heads=heads, shape=shape):
                    own = (CausalSelfAttantion(8, 8) if heads == 1
                           else MultiHeadCausalSelfAttantion(8, 8, heads)).double()
                    reference = nn.MultiheadAttention(8, heads, batch_first=True).double()
                    projections = (own.qwery_proj, own.key_proj, own.value_proj)
                    with torch.no_grad():
                        reference.in_proj_weight.copy_(torch.cat([p.weight for p in projections]))
                        reference.in_proj_bias.copy_(torch.cat([p.bias for p in projections]))
                        reference.out_proj.weight.copy_(own.output_proj.weight)
                        reference.out_proj.bias.copy_(own.output_proj.bias)
                    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                    reference_x = x.detach().clone().requires_grad_()
                    mask = own.causal_mask[:shape[-2], :shape[-2]]
                    output = own(x)
                    expected, _ = reference(reference_x, reference_x, reference_x,
                                            attn_mask=mask, need_weights=False)
                    torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-10)
                    upstream = torch.randn_like(output)
                    (output * upstream).sum().backward()
                    (expected * upstream).sum().backward()
                    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-9, atol=1e-10)
                    for index, projection in enumerate(projections):
                        torch.testing.assert_close(projection.weight.grad,
                                                   reference.in_proj_weight.grad[index * 8:(index + 1) * 8],
                                                   rtol=1e-9, atol=1e-10)
                        torch.testing.assert_close(projection.bias.grad,
                                                   reference.in_proj_bias.grad[index * 8:(index + 1) * 8],
                                                   rtol=1e-9, atol=1e-10)
                    torch.testing.assert_close(own.output_proj.weight.grad, reference.out_proj.weight.grad,
                                               rtol=1e-9, atol=1e-10)
                    torch.testing.assert_close(own.output_proj.bias.grad, reference.out_proj.bias.grad,
                                               rtol=1e-9, atol=1e-10)

    def test_future_tokens_do_not_change_prefix(self):
        layer = MultiHeadCausalSelfAttantion(8, 8, 4).eval()
        # Вход после transpose не contiguous: reshape тоже должен работать.
        x = torch.randn(2, 8, 6).transpose(1, 2)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100
        with torch.no_grad():
            torch.testing.assert_close(layer(x)[:, :3], layer(changed)[:, :3])
            torch.testing.assert_close(layer(x)[:, :3], layer(x[:, :3]))

    def test_invalid_heads_and_context(self):
        for heads in (0, -1, 3):
            with self.assertRaises(ValueError):
                MultiHeadCausalSelfAttantion(8, 8, heads)
        with self.assertRaises(ValueError):
            MultiHeadCausalSelfAttantion(8, 8, 4)(torch.randn(2, 9, 8))

    def test_multi_head_with_one_head_matches_original_class(self):
        original = CausalSelfAttantion(8, 8).double()
        multi = MultiHeadCausalSelfAttantion(8, 8, 1).double()
        multi.load_state_dict(original.state_dict())
        for shape in ((5, 8), (2, 5, 8)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float64)
                # Для входа без batch mm и bmm могут отличаться округлением float64.
                torch.testing.assert_close(original(x), multi(x), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
