import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch
from transformers import LlamaConfig, LlamaForCausalLM, SmolLM3Config, SmolLM3ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import LlmConfig
from utils.torch_utils import onnx_export
from utils.transformers import Attention, Decoder, Rotary


def mapped_config(config):
    with tempfile.TemporaryDirectory() as directory:
        config.save_pretrained(directory)
        return LlmConfig.from_pretrained(directory, local_files_only=True)


def small_config(config_class, **kwargs):
    return config_class(vocab_size=64, hidden_size=64, intermediate_size=96,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=131072, pad_token_id=0, bos_token_id=1,
                        eos_token_id=2, **kwargs)


class ExportAttention(torch.nn.Module):
    def __init__(self, attention, rotary):
        super().__init__()
        self.attention = attention
        self.rotary = rotary

    def forward(self, hidden, mask, positions):
        return self.attention(hidden, mask, self.rotary(positions))


class RopeSemanticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_llama3_legacy_config_migration(self):
        scaling = dict(rope_type='llama3', factor=32.0, low_freq_factor=1.0,
                       high_freq_factor=4.0, original_max_position_embeddings=8192)
        config = small_config(LlamaConfig, rope_theta=500000.0, rope_scaling=scaling)
        mapped = mapped_config(config)
        rotary = Rotary(mapped)
        expected, _ = ROPE_INIT_FUNCTIONS['llama3'](config)
        torch.testing.assert_close(rotary.theta, expected, rtol=1e-6, atol=1e-9)

    def test_llama3_explicit_parameters_and_split_tables(self):
        for factor in (8.0, 32.0):
            with self.subTest(factor=factor):
                parameters = dict(rope_type='llama3', rope_theta=500000.0, factor=factor,
                                  low_freq_factor=1.0, high_freq_factor=4.0,
                                  original_max_position_embeddings=8192)
                config = small_config(LlamaConfig, rope_parameters=parameters)
                mapped = mapped_config(config)
                self.assertEqual(mapped.rope_parameters, config.rope_parameters)
                rotary = Rotary(mapped)
                expected, _ = ROPE_INIT_FUNCTIONS['llama3'](config)
                torch.testing.assert_close(rotary.theta, expected, rtol=1e-6, atol=1e-9)
                positions = torch.tensor([0, 1, 17, 2047, 2048, 4096, 8192, 131071])
                cos, sin = rotary._rope_cos_sin(positions)
                angles = positions.double().unsqueeze(-1) * expected.double()
                torch.testing.assert_close(cos, angles.cos().float(), rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(sin, angles.sin().float(), rtol=1e-6, atol=1e-6)

    def test_legacy_scaling_without_parameters(self):
        mapped = SimpleNamespace(model_type='llama', head_dim=16, rope_ratio=1.0,
                                 rope_theta=500000.0, rope_parameters=None,
                                 max_position_embeddings=131072,
                                 rope_scaling=dict(type='llama3', factor=8.0, low_freq_factor=1.0,
                                                   high_freq_factor=4.0,
                                                   original_max_position_embeddings=8192))
        reference = small_config(LlamaConfig, rope_theta=mapped.rope_theta,
                                 rope_scaling=copy.deepcopy(mapped.rope_scaling))
        self.assertIsNone(mapped.rope_parameters)
        self.assertNotIn('rope_type', mapped.rope_scaling)
        expected, _ = ROPE_INIT_FUNCTIONS['llama3'](reference)
        torch.testing.assert_close(Rotary(mapped).theta, expected, rtol=1e-6, atol=1e-9)

    def test_mobilemoe_rope_and_nope_unchanged(self):
        config = small_config(LlamaConfig, rope_parameters=dict(
            rope_type='llama3', rope_theta=500000.0, factor=8.0,
            low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192))
        source = LlamaForCausalLM(config).eval()
        mapped = mapped_config(config)
        mapped.model_type = 'mobilemoe'
        rotary = Rotary(mapped)
        expected, _ = ROPE_INIT_FUNCTIONS['llama3'](config)
        torch.testing.assert_close(rotary.theta, expected, rtol=1e-6, atol=1e-9)
        for use_rope in (True, False):
            original = copy.deepcopy(source.model.layers[0].self_attn)
            original.use_rope = use_rope
            original.attn_temperature_tuning = False
            attention = Attention(original, 0, mapped, rotary, mapped.model_map)
            self.assertEqual(attention.rotary is not None, use_rope)

    def test_default_rope_unchanged(self):
        config = small_config(LlamaConfig, rope_parameters=dict(rope_type='default', rope_theta=10000.0))
        mapped = mapped_config(config)
        source = LlamaForCausalLM(config).eval()
        expected = source.model.rotary_emb.inv_freq
        torch.testing.assert_close(Rotary(mapped).theta, expected, rtol=1e-6, atol=1e-9)
        attention = Attention(source.model.layers[0].self_attn, 0, mapped, Rotary(mapped), mapped.model_map)
        self.assertIsNotNone(attention.rotary)

    def check_model_alignment(self, source):
        source.eval()
        source.config._attn_implementation = 'eager'
        mapped = mapped_config(source.config)
        rotary = Rotary(mapped)
        blocks = [Decoder(copy.deepcopy(layer), i, mapped, rotary, mapped.model_map).eval()
                  for i, layer in enumerate(source.model.layers)]
        captured = []
        hooks = [layer.register_forward_hook(lambda module, inputs, output: captured.append(
            output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()))
            for layer in source.model.layers]
        cache = DynamicCache(config=source.config)
        try:
            for start, tokens in ((0, 17), (17, 1)):
                captured.clear()
                ids = torch.randint(3, source.config.vocab_size, (1, tokens))
                positions = torch.arange(start, start + tokens).unsqueeze(0)
                mask = torch.zeros(tokens, start + tokens).masked_fill(
                    torch.arange(start + tokens).unsqueeze(0) > positions.transpose(0, 1), -10000.0)
                mask = mask.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    expected = source(ids, attention_mask=mask, position_ids=positions,
                                      past_key_values=cache, use_cache=True).logits
                    hidden = source.model.embed_tokens(ids)
                    embeddings = rotary(positions)
                    for i, block in enumerate(blocks):
                        hidden = block(hidden, embeddings, mask)
                        torch.testing.assert_close(hidden, captured[i], rtol=1e-5, atol=1e-6)
                        actual_key = block.self_attn.past_key_value[0].transpose(1, 2)
                        torch.testing.assert_close(actual_key, cache.layers[i].keys, rtol=1e-5, atol=1e-6)
                    actual = source.lm_head(source.model.norm(hidden))
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        finally:
            for hook in hooks:
                hook.remove()

    def test_llama3_prefill_and_decode(self):
        config = small_config(LlamaConfig, rope_parameters=dict(
            rope_type='llama3', rope_theta=500000.0, factor=32.0,
            low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192))
        self.check_model_alignment(LlamaForCausalLM(config))

    def test_smollm3_nope_prefill_and_decode(self):
        config = small_config(SmolLM3Config, no_rope_layers=[1, 1, 1, 0])
        self.check_model_alignment(SmolLM3ForCausalLM(config))

    def test_smollm3_export_omits_rope_only_on_nope_layers(self):
        config = small_config(SmolLM3Config, no_rope_layers=[1, 1, 1, 0])
        source = SmolLM3ForCausalLM(config).eval()
        mapped = mapped_config(config)
        rotary = Rotary(mapped)
        for index in (0, 3):
            for fused_rope in (False, True):
                with self.subTest(layer=index, fused_rope=fused_rope):
                    attention = Attention(copy.deepcopy(source.model.layers[index].self_attn),
                                          index, mapped, rotary, mapped.model_map).eval()
                    self.assertEqual(attention.rotary is not None, bool(config.no_rope_layers[index]))
                    attention.export_fused_attn = True
                    attention.export_fused_rope = fused_rope
                    buffer = io.BytesIO()
                    inputs = (torch.randn(1, 7, 64), torch.zeros(1, 1, 7, 7), torch.arange(7))
                    onnx_export(ExportAttention(attention, rotary), inputs, buffer,
                                ['hidden', 'mask', 'positions'], ['output'])
                    graph = onnx.load_model_from_string(buffer.getvalue())
                    nodes = [node.op_type for node in graph.graph.node]
                    self.assertEqual(nodes.count('FusedAttention'), 1)
                    self.assertEqual(nodes.count('FusedRoPE'), int(index == 0 and fused_rope))
                    self.assertEqual('positions' in {value.name for value in graph.graph.input}, index == 0)


if __name__ == '__main__':
    unittest.main()
