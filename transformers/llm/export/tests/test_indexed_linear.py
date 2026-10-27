#
#  test_indexed_linear.py
#  MNN
#
#  Created by MNN on 2026/09/29.
#  Copyright © 2018, Alibaba Group Holding Limited
#

"""CPU-only export regression tests; no pretrained model or MNN binary is required.

Run from the repository root with:
    python transformers/llm/export/tests/test_indexed_linear.py -v
"""

import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx.reference import ReferenceEvaluator

EXPORT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPORT_DIR.parents[2]
TEST_ARTIFACTS = REPOSITORY_ROOT / 'build/moe_refactor/export-tests'
sys.path.insert(0, str(EXPORT_DIR))
from utils.custom_op import IndexedLinear
from utils.mnn_converter import MNNConverter
from utils.onnx_rebuilder import OnnxRebuilder
from utils.torch_utils import _quant_on_device, onnx_export
from utils.transformers import Decoder, GptOssExpert, IndexedExperts, Mlp, Qwen3Expert


class SeparateExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(8, 5)
        self.up_proj = torch.nn.Linear(8, 5)
        self.down_proj = torch.nn.Linear(5, 8)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def indexed_reference(bank):
    def forward(module, x, ids):
        projections = bank[module.name]
        values = torch.stack([
            torch.stack([projections[int(ids[t, k])][1](x[t] if module.input_mode == 0 else x[t, k])
                         for k in range(ids.shape[1])])
            for t in range(ids.shape[0])])
        if module.fusedActivation != -1:
            if module.up_name:
                up = torch.stack([torch.stack([bank[module.up_name][int(ids[t, k])][1](
                    x[t] if module.input_mode == 0 else x[t, k]) for k in range(ids.shape[1])])
                    for t in range(ids.shape[0])])
                return F.silu(values) * up
            gate, up = values.chunk(2, dim=-1)
            return F.silu(gate) * up
        return values
    return forward


class IndexedLinearTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        torch.manual_seed(17)

    def test_expert_activations_and_projection_boundaries(self):
        for make_expert in (lambda: Qwen3Expert(8, 5, F.silu), lambda: Qwen3Expert(8, 5, F.gelu),
                            lambda: GptOssExpert(8, 5), SeparateExpert):
            experts = torch.nn.ModuleList(make_expert() for _ in range(3))
            bank = {}
            indexed = IndexedExperts(experts, 4, bank)
            self.assertEqual(indexed.fusedActivation,
                             IndexedLinear.SILU if not indexed.gpt_oss and indexed.act_fn is F.silu else -1)
            self.assertTrue(all(name.startswith('/expert/4_') for entries in bank.values() for name, _ in entries))
            self.assertEqual(len(bank), 3 if make_expert is SeparateExpert else 2)
            for tokens in (1, 7):
                x = torch.randn(tokens, 8) * 4
                ids = torch.randint(3, (tokens, 2), dtype=torch.int32)
                expected = torch.stack([torch.stack([experts[int(e)](x[t]) for e in ids[t]])
                                        for t in range(tokens)])
                with patch.object(IndexedLinear, 'forward', indexed_reference(bank)):
                    actual = indexed(x, ids)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_router_modes(self):
        for mode in ('mobilemoe', 'lfm2_moe', 'gpt_oss', 'qwen3_moe'):
            for tokens in (1, 5):
                mlp = Mlp.__new__(Mlp)
                torch.nn.Module.__init__(mlp)
                mlp.is_moe, mlp.export_moe, mlp.moe_type = True, False, mode
                mlp.top_k, mlp.num_experts, mlp.norm_topk_prob = 2, 3, True
                mlp.routed_scaling_factor = 1.7
                mlp.expert_bias = torch.tensor([0.1, -0.3, 0.5])
                mlp.gate = torch.nn.Linear(8, 3)
                mlp.shared_expert = torch.nn.Linear(8, 8)
                mlp.shared_expert_gate = torch.nn.Linear(8, 1)
                mlp.experts = torch.nn.ModuleList(Qwen3Expert(8, 5, F.silu) for _ in range(3))
                x = torch.randn(1, tokens, 8)
                expected = mlp(x)
                bank = {}
                mlp.experts = IndexedExperts(mlp.experts, 2, bank)
                mlp.export_moe = True
                with patch.object(IndexedLinear, 'forward', indexed_reference(bank)):
                    actual = mlp(x)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_fused_swiglu_export_and_onnx(self):
        for separate in (False, True):
            experts = torch.nn.ModuleList(SeparateExpert() if separate else Qwen3Expert(8, 5, F.silu)
                                         for _ in range(3))
            bank = {}
            fused = IndexedExperts(experts, 3, bank)
            unfused = IndexedExperts(experts, 3, bank, fuse_gate_up=False)
            self.assertEqual(fused.fusedActivation, IndexedLinear.SILU)
            self.assertEqual(unfused.fusedActivation, -1)
            x = torch.randn(7, 8)
            ids = torch.tensor([[0, 2], [1, 1], [2, 0], [0, 0], [2, 2], [1, 0], [0, 2]], dtype=torch.int32)
            with patch.object(IndexedLinear, 'forward', indexed_reference(bank)):
                expected = unfused(x, ids).detach().numpy()
                torch.testing.assert_close(fused(x, ids), torch.from_numpy(expected))
            with tempfile.TemporaryDirectory(prefix='fused-onnx-', dir=TEST_ARTIFACTS) as directory:
                path = str(Path(directory) / 'indexed.onnx')
                onnx_export(fused, (x, ids), path, ['x', 'ids'], ['output'],
                            dynamic_axes={'x': {0: 'tokens'}, 'ids': {0: 'tokens'}, 'output': {0: 'tokens'}})
                graph = onnx.load(path)
                self.assertEqual(sum(node.op_type == 'IndexedLinear' for node in graph.graph.node), 2)
                self.assertFalse(any(node.op_type == 'Mul' for node in graph.graph.node))
                OnnxRebuilder(path, bank).rebuild()
                onnx.checker.check_model(path)
                actual = ReferenceEvaluator(onnx.load(path)).run(None, {'x': x.numpy(), 'ids': ids.numpy()})[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_fused_quantization_boundaries_and_fallback(self):
        args = SimpleNamespace(mnnconvert='', quant_bit=4, quant_block=4, sym=False, quant_config=None,
                               lm_quant_bit=4, lm_quant_block=4, scale_bit=16, hqq=True, awq=False,
                               skip_weight=False, tie_word_embeddings=False)
        bank = {name: [(f'/expert/0_{i}/{name}', torch.nn.Linear(8, 5)) for i in range(3)]
                for name in ('gate', 'up')}
        exporter = SimpleNamespace(args=args)
        original, fused = MNNConverter(exporter), MNNConverter(exporter, bank)
        original.mnn_weight, fused.mnn_weight = io.BytesIO(), io.BytesIO()
        op = {'main': {'attr': [{'key': 'name', 's': 'gate'}, {'key': 'input_mode', 'i': 0},
                               {'key': 'out_features', 'i': 5},
                               {'key': 'fusedActivation', 'i': IndexedLinear.SILU},
                               {'key': 'up_name', 's': 'up'}]}, 'inputIndexes': [0, 1], 'outputIndexes': [2]}
        activation_attr = next(attr for attr in op['main']['attr'] if attr['key'] == 'fusedActivation')
        for activation in (0, 1, 32, -2, 999):
            activation_attr['i'] = activation
            with self.assertRaisesRegex(ValueError, 'Unsupported IndexedLinear UnaryOpOperation'):
                fused.rebuild_indexed_linear(op, {})
        activation_attr['i'] = IndexedLinear.SILU
        # Compare with separately serialized source projections, including their
        # exact FP16/FP32 alpha bytes. The merged bank must never run HQQ on [gate; up].
        for bits in (4, 8, 16):
            for scale_bits in (16, 32):
                for symmetric in (False, True):
                    args.quant_bit, args.scale_bit, args.sym = bits, scale_bits, symmetric
                    original, fused = MNNConverter(exporter), MNNConverter(exporter, bank)
                    original.mnn_weight, fused.mnn_weight = io.BytesIO(), io.BytesIO()
                    # Include a missing bias on either side, and two bias-free projections.
                    bank['gate'][0][1].bias = None
                    bank['up'][1][1].bias = None
                    bank['gate'][2][1].bias = bank['up'][2][1].bias = None
                    with patch('utils.mnn_converter.torch_quant', side_effect=_quant_on_device) as quant:
                        expected = [[original.build_linear_parameter(name, linear) for name, linear in bank[key]]
                                    for key in ('gate', 'up')]
                        actual = fused.rebuild_indexed_linear(op, {})[0]['main']
                    self.assertEqual(actual['fusedActivation'], IndexedLinear.SILU)
                    self.assertEqual(set(actual), {'linears', 'inputMode', 'fusedActivation'})
                    self.assertEqual(len(actual['linears']), 3)
                    self.assertTrue(all(call.args[0].shape == (5, 8) for call in quant.call_args_list))
                    source, merged = original.mnn_weight.getvalue(), fused.mnn_weight.getvalue()
                    header = 0 if bits == 16 else 1 + 2 * 2 + 1 + (1 << bits)
                    for index, linear in enumerate(actual['linears']):
                        self.assertEqual(linear['common']['outputCount'], 10)
                        offset, weight_len, alpha_len, bias_len, _ = linear['external']
                        parts = [[], [], []]
                        for projection in (0, 1):
                            src_offset, w, alpha, bias, _ = expected[projection][index]['external']
                            parts[0].append(source[src_offset + header:src_offset + w])
                            parts[1].append(source[src_offset + w:src_offset + w + alpha])
                            parts[2].append(source[src_offset + w + alpha:src_offset + w + alpha + bias]
                                            if bias else bytes(5 * 4))
                        self.assertEqual(merged[offset + header:offset + weight_len], b''.join(parts[0]))
                        self.assertEqual(merged[offset + weight_len:offset + weight_len + alpha_len],
                                         b''.join(parts[1]))
                        self.assertEqual(merged[offset + weight_len + alpha_len:
                                                offset + weight_len + alpha_len + bias_len],
                                         b''.join(parts[2]) if bias_len else b'')
                        if bits != 16:
                            self.assertEqual(np.frombuffer(merged[offset + 1:offset + 5], dtype=np.uint16).tolist(),
                                             [10, 8])
                            self.assertEqual(linear['quanParameter']['readType'], 0 if symmetric else 20)
        args.quant_bit, args.scale_bit, args.sym = 4, 16, False
        quantization = fused.linear_quantization
        for up_layout in ((8, 4, False), (4, 8, False), (4, 4, True), (16, 4, False)):
            with patch.object(fused, 'linear_quantization', side_effect=lambda name, linear:
                              up_layout if name.endswith('/up') else quantization(name, linear)), \
                    patch('utils.mnn_converter.torch_quant', side_effect=_quant_on_device):
                graph = {'tensorName': ['x', 'ids', 'y'], 'tensorNumber': 3}
                fallback = fused.rebuild_indexed_linear(op, graph)
            self.assertEqual([node['type'] for node in fallback],
                             ['IndexedLinear', 'IndexedLinear', 'UnaryOp', 'BinaryOp'])
            self.assertTrue(all(node['main'].get('fusedActivation', -1) == -1 for node in fallback))

    def test_gemma4_decoder_moe(self):
        class ZeroAttention(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return torch.zeros_like(hidden_states)

        decoder = Decoder.__new__(Decoder)
        torch.nn.Module.__init__(decoder)
        decoder.hidden_size, decoder.layer_type, decoder.alpha = 8, 'full_attention', 1
        decoder.has_gemma4_moe, decoder.export_moe = True, False
        for name in ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm',
                     'post_feedforward_layernorm', 'post_feedforward_layernorm_1',
                     'pre_feedforward_layernorm_2', 'post_feedforward_layernorm_2', 'moe_router_norm'):
            setattr(decoder, name, torch.nn.Identity())
        decoder.self_attn = ZeroAttention()
        decoder.mlp = torch.nn.Linear(8, 8)
        decoder.moe_gate = torch.nn.Linear(8, 3)
        decoder.moe_router_scale, decoder.moe_router_scalar_root = torch.randn(8), 0.4
        decoder.moe_per_expert_scale = torch.tensor([0.8, 1.1, 1.4])
        decoder.moe_top_k, decoder.moe_num_experts = 2, 3
        decoder.experts = torch.nn.ModuleList(Qwen3Expert(8, 5, F.silu) for _ in range(3))
        x = torch.randn(1, 5, 8)
        expected = decoder(x)
        bank = {}
        decoder.experts = IndexedExperts(decoder.experts, 7, bank)
        decoder.export_moe = True
        with patch.object(IndexedLinear, 'forward', indexed_reference(bank)):
            actual = decoder(x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_onnx_reference_lowering(self):
        experts = torch.nn.ModuleList(GptOssExpert(8, 5) for _ in range(3))
        bank = {}
        model = IndexedExperts(experts, 1, bank)
        x, ids = torch.randn(3, 8), torch.tensor([[0, 2], [1, 1], [2, 0]], dtype=torch.int32)
        with patch.object(IndexedLinear, 'forward', indexed_reference(bank)):
            expected = model(x, ids).detach().numpy()
        with tempfile.TemporaryDirectory(prefix='onnx-', dir=TEST_ARTIFACTS) as directory:
            path = str(Path(directory) / 'indexed.onnx')
            onnx_export(model, (x, ids), path, ['x', 'ids'], ['output'],
                        dynamic_axes={'x': {0: 'tokens'}, 'ids': {0: 'tokens'}, 'output': {0: 'tokens'}})
            graph = onnx.load(path)
            self.assertEqual(sum(node.op_type == 'IndexedLinear' for node in graph.graph.node), 2)
            self.assertFalse(any(node.op_type == 'MoE' for node in graph.graph.node))
            self.assertEqual(graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value, 2)
            OnnxRebuilder(path, bank).rebuild()
            onnx.checker.check_model(path)
            actual = ReferenceEvaluator(onnx.load(path)).run(None, {'x': x.numpy(), 'ids': ids.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_quantization_keeps_each_expert_call(self):
        linears = [(f'/expert/0_{i}/gate_proj', torch.nn.Linear(8, 5)) for i in range(3)]
        args = SimpleNamespace(mnnconvert='', quant_bit=4, quant_block=4, sym=False, quant_config=None,
                               lm_quant_bit=4, lm_quant_block=4, scale_bit=16, hqq=True, awq=False,
                               skip_weight=False, tie_word_embeddings=False)
        exporter = SimpleNamespace(args=args)
        original = MNNConverter(exporter)
        original.mnn_weight = io.BytesIO()
        converted = MNNConverter(exporter, {'bank': linears})
        converted.mnn_weight = io.BytesIO()
        op = {'main': {'attr': [{'key': 'name', 's': 'bank'}, {'key': 'input_mode', 'i': 0},
                                {'key': 'out_features', 'i': 5}]},
              'inputIndexes': [0, 1], 'outputIndexes': [2]}
        with patch('utils.mnn_converter.torch_quant', side_effect=_quant_on_device) as quant:
            expected = [original.build_linear_parameter(name, linear) for name, linear in linears]
            actual = converted.rebuild_indexed_linear(op, {})[0]
            self.assertEqual(quant.call_count, 6)
            self.assertTrue(all(tuple(call.args[0].shape) == (5, 8) for call in quant.call_args_list))
        self.assertEqual(actual['main']['linears'], expected)
        self.assertEqual(original.mnn_weight.getvalue(), converted.mnn_weight.getvalue())
        for bits in (1, 2, 3):
            with self.subTest(unsupported_bits=bits):
                args.quant_bit = bits
                with self.assertRaisesRegex(ValueError, 'supports only 4/8-bit or FP16'):
                    converted.rebuild_indexed_linear(op, {})


if __name__ == '__main__':
    unittest.main()
