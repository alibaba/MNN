import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.transformers import Attention


class HeadGateTest(unittest.TestCase):
    def test_prefill_and_cached_decode(self):
        torch.manual_seed(7)
        source = torch.nn.Module()
        for name, width in [('q_proj', 16), ('k_proj', 8), ('v_proj', 8),
                            ('o_proj', 16), ('gate_proj', 4)]:
            setattr(source, name, torch.nn.Linear(16, width, bias=False))
        config = SimpleNamespace(hidden_size=16, head_dim=4, num_attention_heads=4,
                                 num_key_value_heads=2, model_type='test')
        mapping = {name: name for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']}
        mapping['output_gate'] = 'gate_proj'
        exported = Attention(source, 0, config, None, {'attention': mapping})
        tokens = torch.randn(1, 6, 16)
        for start, end in [(0, 5), (5, 6)]:
            current = tokens[:, start:end]
            q = source.q_proj(current).view(1, -1, 4, 4).transpose(1, 2)
            k = source.k_proj(tokens[:, :end]).view(1, -1, 2, 4).transpose(1, 2)
            v = source.v_proj(tokens[:, :end]).view(1, -1, 2, 4).transpose(1, 2)
            mask = torch.zeros(end - start, end)
            mask.masked_fill_(torch.arange(end)[None, :] > torch.arange(start, end)[:, None],
                              float('-inf'))
            expected = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                                                      v.repeat_interleave(2, 1), attn_mask=mask)
            gate = source.gate_proj(current).sigmoid().unsqueeze(-1)
            expected = source.o_proj((expected.transpose(1, 2) * gate).reshape(1, -1, 16))
            torch.testing.assert_close(exported(current, attention_mask=mask), expected)


if __name__ == '__main__':
    unittest.main()
