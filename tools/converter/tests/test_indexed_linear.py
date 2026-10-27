#
#  test_indexed_linear.py
#  MNN
#
#  Created by MNN on 2026/09/29.
#  Copyright © 2018, Alibaba Group Holding Limited
#

"""Native IndexedLinear conversion regressions; all artifacts stay in the workspace."""
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
BUILD = Path(os.environ.get('MNN_CONVERTER_TEST_BUILD', ROOT / 'build')).resolve()


class IndexedLinearConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {name: BUILD / name for name in ('MNNConvert', 'MNNRevert2Buffer', 'MNNDump2Json')}
        if not all(path.is_file() for path in cls.tools.values()):
            raise unittest.SkipTest(f'Build converter tools or set MNN_CONVERTER_TEST_BUILD: {BUILD}')
        artifacts = ROOT / 'build' / 'indexed_linear_converter_tests'
        artifacts.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=artifacts)
        cls.path = Path(cls.directory.name)
        cls.weight = b''.join(struct.pack('<e', (expert + 1) * 0.125) * 8 for expert in range(3))
        cls.graph = {
            'bizCode': 'indexed-linear-test', 'sourceType': 'ONNX', 'tensorName': ['x', 'ids', 'y'],
            'outputName': ['y'], 'tensorNumber': 3,
            'oplists': [
                {'name': 'x', 'type': 'Input', 'main_type': 'Input',
                 'main': {'dims': [1, 4], 'dtype': 'DT_FLOAT', 'dformat': 'NCHW'}, 'outputIndexes': [0]},
                {'name': 'ids', 'type': 'Input', 'main_type': 'Input',
                 'main': {'dims': [1, 2], 'dtype': 'DT_INT32', 'dformat': 'NCHW'}, 'outputIndexes': [1]},
                {'name': 'bank', 'type': 'IndexedLinear', 'main_type': 'IndexedLinearParam',
                 'main': {'inputMode': 0, 'linears': [
                     {'common': {'inputCount': 4, 'outputCount': 2, 'kernelX': 1, 'kernelY': 1, 'group': 1},
                      'quanParameter': {'type': 3}, 'external': [expert * 16, 16, 0, 0, 0]}
                     for expert in range(3)]},
                 'inputIndexes': [0, 1], 'outputIndexes': [2], 'defaultDimentionFormat': 'NCHW'},
            ],
        }

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_tool(self, name, *args, success=True):
        result = subprocess.run([str(self.tools[name]), *map(str, args)], cwd=self.path,
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('Converted Success!', result.stdout)
        return result

    def source(self, name, graph=None, weight=None):
        model = self.path / f'{name}.mnn'
        source = model.with_suffix('.json')
        source.write_text(json.dumps(self.graph if graph is None else graph))
        self.run_tool('MNNRevert2Buffer', source, model)
        Path(str(model) + '.weight').write_bytes(self.weight if weight is None else weight)
        return model

    def convert(self, source, target, *args, success=True):
        return self.run_tool('MNNConvert', '-f', 'MNN', '--modelFile', source, '--MNNModel', target,
                             '--optimizeLevel', '1', *args, success=success)

    def bank(self, model):
        output = Path(str(model) + '.json')
        self.run_tool('MNNDump2Json', model, output)
        return next(op for op in json.loads(output.read_text())['oplists'] if op['type'] == 'IndexedLinear')

    def test_inplace_external_preserves_weights(self):
        source = self.source('inplace')
        self.convert(source, source, '--saveExternalData')
        self.assertEqual(Path(str(source) + '.weight').read_bytes(), self.weight)
        self.assertFalse(Path(str(source) + '.weight.tmp').exists())
        self.assertFalse(self.bank(source).get('externalPath'))

    def fused_graph(self):
        graph = copy.deepcopy(self.graph)
        parameter = graph['oplists'][-1]['main']
        parameter['fusedActivation'] = 34  # UnaryOpOperation.SILU.
        weights = bytearray()
        for index, linear in enumerate(parameter['linears']):
            linear['common']['outputCount'] *= 2
            data = self.weight[index * 16:(index + 1) * 16]
            linear['external'] = [len(weights), 32, 0, 0, 0]
            weights.extend(data + data[::-1])
        return graph, bytes(weights)

    def test_fused_merged_bank_external_roundtrip(self):
        graph, weights = self.fused_graph()
        source = self.source('fused', graph, weights)
        inline, external = self.path / 'fused-inline.mnn', self.path / 'fused-external.mnn'
        self.convert(source, inline)
        self.convert(source, external, '--saveExternalData')
        for output in (inline, external):
            converted = self.bank(output)['main']
            self.assertEqual(converted['fusedActivation'], 34)
            self.assertEqual(len(converted['linears']), 3)
        self.assertEqual(Path(str(external) + '.weight').read_bytes(), weights)
        Path(str(source) + '.weight').unlink()
        self.convert(inline, self.path / 'fused-inline-again.mnn')
        self.convert(external, self.path / 'fused-external-again.mnn', '--saveExternalData')

    def test_fused_mixed_storage_and_invalid_linear(self):
        graph, weights = self.fused_graph()
        for index, linear in enumerate(graph['oplists'][-1]['main']['linears']):
            if index % 2 == 0:
                linear.pop('external')
                linear['quanParameter']['buffer'] = list(weights[index * 32:(index + 1) * 32])
        source = self.source('mixed-storage', graph, weights)
        inline = self.path / 'mixed-inline.mnn'
        self.convert(source, inline)
        self.assertFalse(any(linear.get('external') for linear in self.bank(inline)['main']['linears']))
        graph['oplists'][-1]['main']['linears'][1]['external'][1] = 1000
        source = self.source('bad-linear', graph, weights)
        original_model, original_weights = source.read_bytes(), Path(str(source) + '.weight').read_bytes()
        self.convert(source, source, '--saveExternalData', success=False)
        self.assertEqual(source.read_bytes(), original_model)
        self.assertEqual(Path(str(source) + '.weight').read_bytes(), original_weights)

    def test_separate_output_is_independent(self):
        source = self.source('separate-source')
        inline, external = self.path / 'inline.mnn', self.path / 'external.mnn'
        self.convert(source, inline)
        self.convert(source, external, '--saveExternalData')
        inline_bank = self.bank(inline)
        self.assertFalse(inline_bank.get('externalPath'))
        self.assertFalse(self.bank(external).get('externalPath'))
        for index, expert in enumerate(inline_bank['main']['linears']):
            self.assertFalse(expert.get('external'))
            actual = bytes(value & 255 for value in expert['quanParameter']['buffer'])
            self.assertEqual(actual, self.weight[index * 16:(index + 1) * 16])
        self.assertEqual(Path(str(external) + '.weight').read_bytes(), self.weight)
        Path(str(source) + '.weight').unlink()
        self.convert(inline, self.path / 'inline-again.mnn')
        self.convert(external, self.path / 'external-again.mnn', '--saveExternalData')

    def test_invalid_external_preserves_existing_files(self):
        for index, layout in enumerate(([0, 16], [-1, 16, 0], [0, 16, 3], [0, 1000, 0],
                                        [2**63 - 1, 16, 0])):
            with self.subTest(layout=layout):
                graph = copy.deepcopy(self.graph)
                graph['oplists'][2]['main']['linears'][0]['external'] = layout
                source = self.source(f'invalid-{index}', graph)
                original_model = source.read_bytes()
                self.convert(source, source, '--saveExternalData', success=False)
                self.assertEqual(source.read_bytes(), original_model)
                self.assertEqual(Path(str(source) + '.weight').read_bytes(), self.weight)
                self.assertFalse(Path(str(source) + '.weight.tmp').exists())

    def test_missing_or_truncated_weights_fail(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                source = self.source(f'missing-{missing}', weight=self.weight[:-1])
                if missing:
                    Path(str(source) + '.weight').unlink()
                # Exercise both the graph-only shortcut and the rewriting path.
                self.convert(source, source, success=False)
                self.convert(source, self.path / f'failed-{missing}.mnn', '--saveExternalData', success=False)

    def test_late_output_failure_preserves_model_weight_pair(self):
        # Force a failure after post-treatment has produced the new weights.
        # Requantizing FP16 to W4 ensures the staged weights differ from the old.
        for suffix in ('.tmp', '.bak'):
            source = self.source('late-failure-' + suffix[1:])
            weights = Path(str(source) + '.weight')
            original = (source.read_bytes(), weights.read_bytes())
            blocker = Path(str(source) + suffix)
            blocker.mkdir()
            self.convert(source, source, '--saveExternalData', '--weightQuantBits', '4', success=False)
            self.assertEqual((source.read_bytes(), weights.read_bytes()), original)
            self.assertTrue(blocker.is_dir())
            self.assertFalse(Path(str(weights) + '.tmp').exists())
            self.assertFalse(Path(str(weights) + '.bak').exists())

    def test_static_external_output(self):
        source = self.source('static-source')
        target = self.path / 'static-output.mnn'
        self.convert(source, target, '--saveExternalData', '--saveStaticModel')
        self.bank(target)
        data = json.loads(Path(str(target) + '.json').read_text())
        self.assertEqual(data['usage'], 'INFERENCE_STATIC')
        self.assertEqual(Path(str(target) + '.weight').read_bytes(), self.weight)

    def test_indexed_linear_compression_names(self):
        graph = copy.deepcopy(self.graph)
        parameter = graph['oplists'][-1]['main']
        parameter['fusedActivation'] = 34  # UnaryOpOperation.SILU.
        for linear in parameter['linears']:
            linear.pop('external')
            linear.pop('quanParameter')
            linear['weight'] = [0.125] * 8
        source = self.source('compression-names', graph)
        guide = self.path / 'compression-guide.json'
        self.convert(source, self.path / 'compression-names-output.mnn', '--compressionParamsFile', guide)
        layers = json.loads(guide.read_text())['algo'][0]['quantParams']['layer']
        names = [layer['opName'] for layer in layers]
        self.assertEqual(set(names), {f'bank/linear.{i}' for i in range(3)})
        self.assertEqual(len(names), 3)

    def test_biasfree_bank_skips_sparse_detection(self):
        graph = copy.deepcopy(self.graph)
        for expert in graph['oplists'][2]['main']['linears']:
            expert.pop('external')
            expert.pop('quanParameter')
            expert['weight'] = [0.125] * 8
            expert['bias'] = []
        source = self.source('biasfree', graph)
        normal, sparse = self.path / 'normal.mnn', self.path / 'sparse.mnn'
        self.convert(source, normal)
        self.convert(source, sparse, '--detectSparseSpeedUp', '1')
        expected, actual = self.bank(normal)['main']['linears'], self.bank(sparse)['main']['linears']
        self.assertTrue(all(not expert.get('sparseParameter') for expert in actual))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
