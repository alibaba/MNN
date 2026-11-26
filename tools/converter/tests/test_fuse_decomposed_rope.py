#
#  test_fuse_decomposed_rope.py
#  MNN
#
#  Created by MNN on 2026-11-19.
#  Copyright © 2018, Alibaba Group Holding Limited
#

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
BUILD = Path(os.environ.get('MNN_CONVERTER_TEST_BUILD', ROOT / 'build')).resolve()

SEQ = 4
HEAD_DIM = 8
NUM_HEAD_Q = 2
NUM_HEAD_K = 1
IN_C = 8
EPS = 1e-5
SCALE = 1.5


class GraphBuilder:
    def __init__(self):
        self.ops = []
        self.names = []

    def tensor(self, name):
        self.names.append(name)
        return len(self.names) - 1

    def op(self, name, typ, main_type, main, inputs, out_name, fmt='NHWC'):
        out = self.tensor(out_name)
        self.ops.append({'name': name, 'type': typ, 'main_type': main_type, 'main': main,
                         'inputIndexes': list(inputs), 'outputIndexes': [out],
                         'defaultDimentionFormat': fmt})
        return out

    def const_int(self, name, value):
        return self.op(name, 'Const', 'Blob',
                       {'dims': [1], 'dataType': 'DT_INT32', 'dataFormat': 'NCHW', 'int32s': [value]},
                       [], name + '_t')

    def const_float(self, name, dims, values):
        return self.op(name, 'Const', 'Blob',
                       {'dims': list(dims), 'dataType': 'DT_FLOAT', 'dataFormat': 'NCHW',
                        'float32s': list(values)}, [], name + '_t')

    def net(self, outputs):
        return {'bizCode': 'fuse-rope-test', 'sourceType': 'ONNX', 'tensorName': self.names,
                'outputName': outputs, 'tensorNumber': len(self.names), 'oplists': self.ops}


def rope_tables(scale, asymmetric=False):
    half = HEAD_DIM // 2
    theta = [1.0 / (10000.0 ** (2 * i / HEAD_DIM)) for i in range(half)]
    cos, sin = [], []
    for p in range(SEQ):
        for i in range(half):
            angle = p * theta[i]
            c, s = math.cos(angle) * scale, math.sin(angle) * scale
            if asymmetric:
                other = angle + 0.37
                c2, s2 = math.cos(other) * scale, math.sin(other) * scale
            else:
                c2, s2 = c, s
            cos += [c, c2]
            sin += [s, s2]
    return cos, sin


def build_graph(include_attention, q_norm_before_k_proj=False, table_scale=SCALE, asymmetric=False,
                gamma_len=None, extra_outputs=(), pre_norm_side=None):
    g = GraphBuilder()
    x_in = g.op('x_in', 'Input', 'Input',
                {'dims': [SEQ, IN_C, 1, 1], 'dtype': 'DT_FLOAT', 'dformat': 'NC4HW4'},
                [], 'x_in_t', fmt='NC4HW4')
    cos_vals, sin_vals = rope_tables(table_scale, asymmetric)
    cos = g.const_float('cos', [1, SEQ, 1, HEAD_DIM], cos_vals)
    sin = g.const_float('sin', [1, SEQ, 1, HEAD_DIM], sin_vals)

    batch_c = g.const_int('batch_c', 1)
    seq_c = g.const_int('seq_c', SEQ)
    head_c = g.const_int('head_c', HEAD_DIM)
    begin0 = g.const_int('begin0', 0)
    begin1 = g.const_int('begin1', 1)
    end_c = g.const_int('end_c', 1 << 30)
    axes_c = g.const_int('axes_c', -1)
    step2 = g.const_int('step2', 2)
    unsq_c = g.const_int('unsq_c', -1)

    def slice_onnx(name, x, begin):
        return g.op(name, 'StridedSlice', 'StridedSliceParam',
                    {'Index': 'DT_INT32', 'T': 'DT_FLOAT', 'beginMask': 0, 'endMask': 0,
                     'ellipsisMask': 0, 'newAxisMask': 0, 'shrinkAxisMask': 0, 'fromType': 1},
                    [x, begin, end_c, axes_c, step2], name + '_t')

    def norm(name, inp, out):
        gamma = [] if gamma_len is None else [0.9 + 0.01 * i for i in range(gamma_len)]
        return g.op(name, 'LayerNorm', 'LayerNorm',
                    {'axis': [-1], 'epsilon': EPS, 'group': 1, 'gamma': gamma, 'useRMSNorm': True},
                    [inp], out)

    def side(prefix, num_head, weight):
        channels = num_head * HEAD_DIM
        nh_c = g.const_int(prefix + 'nh_c', num_head)
        conv = g.op(prefix + 'conv', 'Convolution', 'Convolution2D',
                    {'common': {'kernelX': 1, 'kernelY': 1, 'strideX': 1, 'strideY': 1,
                                'group': 1, 'outputCount': channels, 'inputCount': IN_C,
                                'padMode': 'CAFFE', 'relu': False, 'relu6': False},
                     'weight': weight, 'bias': [0.0] * channels},
                    [x_in], prefix + 'conv_t', fmt='NC4HW4')
        nchw = g.op(prefix + 'cvt', 'ConvertTensor', 'TensorConvertInfo',
                    {'source': 'NC4HW4', 'dest': 'NCHW'}, [conv], prefix + 'nchw_t', fmt='NCHW')
        shape = g.op(prefix + 'shape', 'Concat', 'Axis', {'axis': 0},
                     [batch_c, seq_c, nh_c, head_c], prefix + 'shape_t')
        x = g.op(prefix + 'reshape', 'Reshape', 'Reshape', {'dimType': 'NCHW'},
                 [nchw, shape], prefix + 'x_t')
        if pre_norm_side == prefix:
            x = norm(prefix + 'norm', x, prefix + 'pre_norm_t')
        even = slice_onnx(prefix + 'even', x, begin0)
        odd = slice_onnx(prefix + 'odd', x, begin1)
        neg_odd = g.op(prefix + 'neg', 'UnaryOp', 'UnaryOp', {'opType': 'NEG', 'T': 'DT_FLOAT'},
                       [odd], prefix + 'neg_t')
        unsq_neg = g.op(prefix + 'unsq_neg', 'Unsqueeze', 'SqueezeParam', {},
                        [neg_odd, unsq_c], prefix + 'unsq_neg_t')
        unsq_even = g.op(prefix + 'unsq_even', 'Unsqueeze', 'SqueezeParam', {},
                         [even, unsq_c], prefix + 'unsq_even_t')
        rot_cat = g.op(prefix + 'rot_cat', 'Concat', 'Axis', {'axis': -1},
                       [unsq_neg, unsq_even], prefix + 'rot_cat_t')
        rot = g.op(prefix + 'rot_reshape', 'Reshape', 'Reshape', {'dimType': 'NCHW'},
                   [rot_cat, shape], prefix + 'rot_t')
        mul_cos = g.op(prefix + 'mul_cos', 'BinaryOp', 'BinaryOp',
                       {'opType': 'MUL', 'T': 'DT_FLOAT', 'activationType': 0},
                       [x, cos], prefix + 'mul_cos_t')
        mul_sin = g.op(prefix + 'mul_sin', 'BinaryOp', 'BinaryOp',
                       {'opType': 'MUL', 'T': 'DT_FLOAT', 'activationType': 0},
                       [rot, sin], prefix + 'mul_sin_t')
        add = g.op(prefix + 'add', 'BinaryOp', 'BinaryOp',
                   {'opType': 'ADD', 'T': 'DT_FLOAT', 'activationType': 0},
                   [mul_cos, mul_sin], prefix + ('norm_out' if pre_norm_side == prefix else 'add_t'))
        return add

    wq = [((i % 13) - 6) * 0.05 for i in range(NUM_HEAD_Q * HEAD_DIM * IN_C)]
    wk = [((i % 11) - 5) * 0.07 for i in range(NUM_HEAD_K * HEAD_DIM * IN_C)]

    q_add = side('q_', NUM_HEAD_Q, wq)
    if q_norm_before_k_proj:
        q_norm = q_add if pre_norm_side == 'q_' else norm('q_norm', q_add, 'q_norm_out')
    k_add = side('k_', NUM_HEAD_K, wk)
    if not q_norm_before_k_proj:
        q_norm = q_add if pre_norm_side == 'q_' else norm('q_norm', q_add, 'q_norm_out')
    k_norm = k_add if pre_norm_side == 'k_' else norm('k_norm', k_add, 'k_norm_out')

    if include_attention:
        g.op('attn', 'Attention', 'AttentionParam', {'kv_cache': False, 'output_c4': False},
             [q_norm, k_norm, k_norm], 'attn_out')
    return g.net(['q_norm_out', 'k_norm_out'] + list(extra_outputs)), ['q_norm_out', 'k_norm_out']


class FuseDecomposedRoPETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {n: BUILD / n for n in
                     ('MNNConvert', 'MNNRevert2Buffer', 'MNNDump2Json', 'MNNV2Basic.out')}
        if not all(p.is_file() for p in cls.tools.values()):
            raise unittest.SkipTest(f'Build converter + MNNV2Basic or set MNN_CONVERTER_TEST_BUILD: {BUILD}')
        artifacts = ROOT / 'build' / 'fuse_rope_converter_tests'
        artifacts.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=artifacts)
        cls.path = Path(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_tool(self, name, *args, cwd=None):
        result = subprocess.run([str(self.tools[name]), *map(str, args)],
                                cwd=str(cwd or self.path), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def revert(self, name, graph):
        src = self.path / f'{name}.json'
        src.write_text(json.dumps(graph))
        model = self.path / f'{name}.mnn'
        self.run_tool('MNNRevert2Buffer', src, model)
        return model

    def convert(self, source, target):
        return self.run_tool('MNNConvert', '-f', 'MNN', '--modelFile', source,
                             '--MNNModel', target, '--optimizeLevel', '1')

    def dump(self, model):
        out = Path(str(model) + '.json')
        self.run_tool('MNNDump2Json', model, out)
        return json.loads(out.read_text())

    def run_model(self, model, tag):
        run_dir = self.path / tag
        (run_dir / 'output').mkdir(parents=True, exist_ok=True)
        values = [((i % 17) - 8) * 0.125 for i in range(IN_C * SEQ)]
        (run_dir / 'input_0.txt').write_text('\n'.join(f'{v:.6f}' for v in values) + '\n')
        link = run_dir / 'MNNV2Basic.out'
        if not link.exists():
            link.symlink_to(self.tools['MNNV2Basic.out'])
        argv = ['./MNNV2Basic.out', str(model.resolve()), '1', '0', '0', '1', '1']
        # A signal kills the child before its block-buffered stdout is flushed, which would
        # otherwise report an empty message; line buffering keeps the stage markers.
        cmd = (['stdbuf', '-oL', '-eL'] if shutil.which('stdbuf') else []) + argv
        result = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stdout + result.stderr
            if result.returncode < 0 and shutil.which('gdb'):
                bt = subprocess.run(['gdb', '-batch', '-ex', 'run', '-ex', 'bt', '--args'] + argv,
                                    cwd=str(run_dir), capture_output=True, text=True)
                detail += '\n--- gdb backtrace ---\n' + bt.stdout + bt.stderr
            self.fail(f'{tag}: MNNV2Basic.out exit {result.returncode}\n{detail}')
        result = {}
        for name in ('q_norm_out', 'k_norm_out'):
            text = (run_dir / 'output' / f'{name}.txt').read_text()
            result[name] = [float(x) for x in text.replace('\t', ' ').split()]
        return result

    def _verify_equivalence(self, q_norm_before_k_proj, tag):
        ref_graph, _ = build_graph(include_attention=False, q_norm_before_k_proj=q_norm_before_k_proj)
        full_graph, _ = build_graph(include_attention=True, q_norm_before_k_proj=q_norm_before_k_proj)
        ref_src = self.revert(tag + '_ref', ref_graph)
        full_src = self.revert(tag + '_full', full_graph)
        ref_conv = self.path / (tag + '_ref_conv.mnn')
        fused = self.path / (tag + '_fused.mnn')
        ref_log = self.convert(ref_src, ref_conv)
        full_log = self.convert(full_src, fused)
        self.assertNotIn('FuseDecomposedRoPE: fused', ref_log.stdout)
        self.assertIn('FuseDecomposedRoPE: fused 1 attention', full_log.stdout)

        ops = self.dump(fused)['oplists']
        ropes = [op for op in ops if op['type'] == 'RoPE']
        self.assertEqual(len(ropes), 1, 'expected exactly one fused RoPE')
        rope = ropes[0]
        main = rope['main']
        self.assertTrue(main.get('interleaved'))
        self.assertTrue(main.get('q_norm'), 'q_norm not carried by the fused RoPE')
        self.assertTrue(main.get('k_norm'), 'k_norm not carried by the fused RoPE')
        self.assertTrue(main.get('norm_after'), 'post-RoPE norms commuted to pre (order regression)')
        norms = [op for op in ops if op['type'] == 'LayerNorm']
        self.assertEqual(len(norms), 0, 'post-RoPE norms should be folded into the RoPE op')
        self.assertFalse([op for op in ops if op['type'] == 'BinaryOp'
                          and op['main'].get('opType') == 'ADD'])
        producer = {}
        for idx, op in enumerate(ops):
            for t in op.get('outputIndexes', []):
                producer[t] = idx
        for idx, op in enumerate(ops):
            for t in op.get('inputIndexes', []):
                if t in producer:
                    self.assertLess(producer[t], idx,
                                    f"op {idx} ({op['name']}) reads tensor produced later at {producer[t]}")

        ref_out = self.run_model(ref_conv, tag + '_run_ref')
        fused_out = self.run_model(fused, tag + '_run_fused')
        for name in ('q_norm_out', 'k_norm_out'):
            self.assertEqual(len(ref_out[name]), len(fused_out[name]), f'{name} size differs')
            self.assertTrue(ref_out[name], f'{name} empty')
            diff = max(abs(a - b) for a, b in zip(ref_out[name], fused_out[name]))
            self.assertLess(diff, 1e-4,
                            f'{name}: fused != decomposed under scaled tables (max diff {diff})')

    def test_fused_graph_matches_decomposed_original(self):
        self._verify_equivalence(q_norm_before_k_proj=False, tag='rope')

    def test_mobilemoe_order_fuses_and_matches(self):
        self._verify_equivalence(q_norm_before_k_proj=True, tag='rope_moe')

    def test_asymmetric_tables_keep_post_norm(self):
        graph, _ = build_graph(include_attention=True, table_scale=1.0, asymmetric=True)
        src = self.revert('rope_asym', graph)
        fused = self.path / 'rope_asym_fused.mnn'
        log = self.convert(src, fused)
        self.assertIn('FuseDecomposedRoPE: fused 1 attention', log.stdout)
        ops = self.dump(fused)['oplists']
        ropes = [op for op in ops if op['type'] == 'RoPE']
        self.assertEqual(len(ropes), 1)
        main = ropes[0]['main']
        self.assertTrue(main.get('norm_after'), 'asymmetric pair coefficients commuted to pre-norm')

    def test_mixed_norm_order_keeps_original_ops(self):
        for prefix in ('q_', 'k_'):
            with self.subTest(pre_norm_side=prefix):
                graph, _ = build_graph(include_attention=True, pre_norm_side=prefix, gamma_len=HEAD_DIM)
                src = self.revert(prefix + 'rope_mixed', graph)
                target = self.path / (prefix + 'rope_mixed_converted.mnn')
                log = self.convert(src, target)
                self.assertNotIn('FuseDecomposedRoPE: fused', log.stdout)
                ops = self.dump(target)['oplists']
                self.assertFalse([op for op in ops if op['type'] == 'RoPE'])
                self.assertEqual(len([op for op in ops if op['type'] == 'LayerNorm']), 2)

    def test_commutable_mixed_order_can_share_pre_norm(self):
        for prefix in ('q_', 'k_'):
            with self.subTest(pre_norm_side=prefix):
                tag = prefix + 'rope_commutable'
                ref_graph, _ = build_graph(include_attention=False, pre_norm_side=prefix, table_scale=1.0)
                graph, _ = build_graph(include_attention=True, pre_norm_side=prefix, table_scale=1.0)
                ref = self.path / (tag + '_ref_conv.mnn')
                target = self.path / (tag + '_fused.mnn')
                self.convert(self.revert(tag + '_ref', ref_graph), ref)
                log = self.convert(self.revert(tag, graph), target)
                self.assertIn('FuseDecomposedRoPE: fused 1 attention', log.stdout)
                ops = self.dump(target)['oplists']
                ropes = [op for op in ops if op['type'] == 'RoPE']
                self.assertEqual(len(ropes), 1)
                main = ropes[0]['main']
                self.assertTrue(main.get('q_norm'))
                self.assertTrue(main.get('k_norm'))
                self.assertFalse(main.get('norm_after'))
                self.assertFalse([op for op in ops if op['type'] == 'LayerNorm'])

                ref_out = self.run_model(ref, tag + '_run_ref')
                fused_out = self.run_model(target, tag + '_run_fused')
                for name in ('q_norm_out', 'k_norm_out'):
                    self.assertEqual(len(ref_out[name]), len(fused_out[name]))
                    diff = max(abs(a - b) for a, b in zip(ref_out[name], fused_out[name]))
                    self.assertLess(diff, 1e-4, f'{name}: mixed-order fusion max diff {diff}')

    def test_gamma_length_mismatch_not_folded(self):
        graph, _ = build_graph(include_attention=True, gamma_len=HEAD_DIM - 1)
        src = self.revert('rope_gamma', graph)
        fused = self.path / 'rope_gamma_fused.mnn'
        self.convert(src, fused)
        ops = self.dump(fused)['oplists']
        ropes = [op for op in ops if op['type'] == 'RoPE']
        self.assertEqual(len(ropes), 1)
        main = ropes[0]['main']
        self.assertFalse(main.get('q_norm'), 'mismatched gamma folded into RoPE')
        self.assertFalse(main.get('k_norm'), 'mismatched gamma folded into RoPE')
        norms = [op for op in ops if op['type'] == 'LayerNorm']
        self.assertEqual(len(norms), 2, 'norms with mismatched gamma must stay in the graph')

    def test_rotation_output_is_graph_output_not_fused(self):
        graph, _ = build_graph(include_attention=True, extra_outputs=('q_add_t',))
        src = self.revert('rope_gout', graph)
        fused = self.path / 'rope_gout_fused.mnn'
        log = self.convert(src, fused)
        self.assertNotIn('FuseDecomposedRoPE: fused', log.stdout)
        dumped = self.dump(fused)
        ops = dumped['oplists']
        self.assertFalse([op for op in ops if op['type'] == 'RoPE'])
        self.assertIn('q_add_t', dumped['outputName'])
        produced = {t for op in ops for t in op.get('outputIndexes', [])}
        self.assertIn(dumped['tensorName'].index('q_add_t'), produced,
                      'declared graph output lost its producer')


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2 if '-v' in sys.argv else 1).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    failed = len(result.failures) + len(result.errors)
    print('TEST_NAME_UNIT: RoPE 融合转换回归测试')
    print('TEST_CASE_AMOUNT_UNIT: {"blocked":0,"failed":%d,"passed":%d,"skipped":%d}'
          % (failed, passed, len(result.skipped)))
    print('TEST_CASE={"name":"RoPE 融合转换回归测试","failed":%d,"passed":%d,"skipped":%d}'
          % (failed, passed, len(result.skipped)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())