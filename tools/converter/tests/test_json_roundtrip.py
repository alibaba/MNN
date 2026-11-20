import copy
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
BUILD = Path(os.environ.get('MNN_CONVERTER_TEST_BUILD', ROOT / 'build')).resolve()


class JsonRoundtripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {name: BUILD / name for name in ('MNNRevert2Buffer', 'MNNDump2Json')}
        if not all(path.is_file() for path in cls.tools.values()):
            raise unittest.SkipTest(f'Build converter tools or set MNN_CONVERTER_TEST_BUILD: {BUILD}')
        artifacts = ROOT / 'build' / 'json_roundtrip_converter_tests'
        artifacts.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=artifacts)
        cls.path = Path(cls.directory.name)
        bits = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
                0x3dcccccd, 0xbeaaaaab, 0x33800000, 0x4d8ac150]
        cls.boundaries = [struct.unpack('<f', struct.pack('<I', value))[0] for value in bits]
        generator = random.Random(20261120)
        cls.values = list(cls.boundaries)
        while len(cls.values) < 20000:
            value = generator.getrandbits(32)
            if value & 0x7f800000 != 0x7f800000:
                cls.values.append(struct.unpack('<f', struct.pack('<I', value))[0])
        ops = []
        for name, values in (('small', cls.boundaries), ('large', cls.values)):
            ops.append({'name': name, 'type': 'Const', 'main_type': 'Blob',
                        'main': {'dims': [len(values)], 'dataType': 'DT_FLOAT',
                                 'dataFormat': 'NCHW', 'float32s': values},
                        'outputIndexes': [len(ops)]})
        ops.append({'name': 'scalars', 'type': 'Extra', 'main_type': 'Extra',
                    'main': {'engine': 'MNN', 'type': 'FloatRoundtrip',
                             'attr': [{'key': str(i), 'f': value}
                                      for i, value in enumerate(cls.boundaries)]},
                    'outputIndexes': [2]})
        ops.append({'name': 'conv', 'type': 'Convolution', 'main_type': 'Convolution2D',
                    'main': {'common': {'inputCount': 1, 'outputCount': 1, 'kernelX': 1, 'kernelY': 1},
                             'weight': [cls.boundaries[2]], 'bias': [cls.boundaries[1]]},
                    'inputIndexes': [0], 'outputIndexes': [3]})
        cls.graph = {'tensorName': [op['name'] for op in ops], 'outputName': ['small'],
                     'oplists': ops, 'subgraphs': [{'name': 'float_subgraph', 'inputs': [],
                                                  'outputs': [0], 'tensors': [op['name'] for op in ops],
                                                  'nodes': copy.deepcopy(ops)}]}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_tool(self, name, *args):
        result = subprocess.run([str(self.tools[name]), *map(str, args)], cwd=self.path,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def roundtrip(self, name, graph, *flags):
        source = self.path / f'{name}-source.json'
        original = self.path / f'{name}-original.mnn'
        dumped = self.path / f'{name}-dump.json'
        rebuilt = self.path / f'{name}-rebuilt.mnn'
        repeated = self.path / f'{name}-repeat.json'
        source.write_text(json.dumps(graph))
        self.run_tool('MNNRevert2Buffer', source, original)
        self.assertGreater(original.stat().st_size, 0)
        self.run_tool('MNNDump2Json', original, dumped, *flags)
        first = json.loads(dumped.read_text())
        self.run_tool('MNNRevert2Buffer', dumped, rebuilt)
        self.assertGreater(rebuilt.stat().st_size, 0)
        self.run_tool('MNNDump2Json', rebuilt, repeated)
        second = json.loads(repeated.read_text())
        return dumped, first, second

    def assert_float_bits(self, expected, actual):
        self.assertEqual([struct.pack('<f', value) for value in actual],
                         [struct.pack('<f', value) for value in expected])

    def assert_ops(self, ops, stripped=False):
        by_name = {op['name']: op for op in ops}
        self.assert_float_bits(self.boundaries, by_name['small']['main']['float32s'])
        self.assert_float_bits([] if stripped else self.values,
                               by_name['large']['main'].get('float32s', []))
        attrs = {attr['key']: attr['f'] for attr in by_name['scalars']['main']['attr']}
        self.assert_float_bits(self.boundaries, [attrs[str(i)] for i in range(len(self.boundaries))])
        conv = by_name['conv']['main']
        self.assert_float_bits([] if stripped else [self.boundaries[2]], conv.get('weight', []))
        self.assert_float_bits([] if stripped else [self.boundaries[1]], conv.get('bias', []))

    def test_full_dump_preserves_float_bits(self):
        _, first, second = self.roundtrip('full', self.graph)
        for graph in (first, second):
            self.assert_ops(graph['oplists'])
            self.assert_ops(graph['subgraphs'][0]['nodes'])

    def test_stripped_dump_preserves_retained_float_bits(self):
        _, first, second = self.roundtrip('stripped', self.graph, 'strip')
        for graph in (first, second):
            self.assert_ops(graph['oplists'], stripped=True)
            self.assert_ops(graph['subgraphs'][0]['nodes'], stripped=True)

    def test_separate_subgraph_preserves_retained_float_bits(self):
        dumped, first, second = self.roundtrip('separate', self.graph, 'strip', 'separate')
        for graph in (first, second):
            self.assert_ops(graph['oplists'], stripped=True)
            self.assertFalse(graph.get('subgraphs'))
        subgraph = json.loads(Path(str(dumped) + '_0').read_text())
        self.assert_ops(subgraph['nodes'], stripped=True)
        _, wrapped, repeated = self.roundtrip('subgraph', {'subgraphs': [subgraph]})
        for graph in (wrapped, repeated):
            self.assert_ops(graph['subgraphs'][0]['nodes'], stripped=True)


if __name__ == '__main__':
    unittest.main()
