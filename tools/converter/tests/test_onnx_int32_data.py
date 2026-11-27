#
#  test_onnx_int32_data.py
#  MNN
#
#  Created by MNN on 2026/11/24.
#  Copyright © 2018, Alibaba Group Holding Limited
#

"""ONNX initializers of narrow types stored in int32_data (issue #5172)."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    onnx = None


ROOT = Path(__file__).resolve().parents[3]
BUILD = Path(os.environ.get('MNN_CONVERTER_TEST_BUILD', ROOT / 'build')).resolve()


@unittest.skipIf(onnx is None, 'onnx is not installed')
class OnnxInt32DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {name: BUILD / name for name in ('MNNConvert', 'MNNDump2Json')}
        if not all(path.is_file() for path in cls.tools.values()):
            raise unittest.SkipTest(f'Build converter tools or set MNN_CONVERTER_TEST_BUILD: {BUILD}')
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def convert(self, graph, name):
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 8
        source, target, dump = (self.path / f'{name}{suffix}' for suffix in ('.onnx', '.mnn', '.json'))
        onnx.save(model, str(source))
        result = subprocess.run([str(self.tools['MNNConvert']), '-f', 'ONNX', '--modelFile', str(source),
                                 '--MNNModel', str(target), '--bizCode', 'MNN'], capture_output=True, text=True)
        self.assertIn('Converted Success!', result.stdout, result.stdout + result.stderr)
        subprocess.run([str(self.tools['MNNDump2Json']), str(target), str(dump)], check=True, capture_output=True)
        return {op.get('name'): op for op in json.loads(dump.read_text())['oplists']}

    def test_cast_constants(self):
        # helper.make_tensor writes these types to int32_data, one element per int32.
        cases = {
            'bool': (TensorProto.BOOL, [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1]),
            'int8': (TensorProto.INT8, [-2, -1, 0, 1, 2, 127], [-2, -1, 0, 1, 2, 127]),
            'int16': (TensorProto.INT16, [-300, -1, 0, 1, 2, 32767], [-300, -1, 0, 1, 2, 32767]),
            'uint16': (TensorProto.UINT16, [0, 1, 2, 300, 40000, 65535], [0, 1, 2, 300, 40000, 65535]),
            'float16': (TensorProto.FLOAT16, [1.5, -2.25, 3.0, 0.0, -0.5, 1024.0],
                        [1.5, -2.25, 3.0, 0.0, -0.5, 1024.0]),
        }
        for name, (dtype, values, expected) in cases.items():
            with self.subTest(dtype=name):
                constant = helper.make_tensor('c', dtype, [6], values)
                self.assertGreater(len(constant.int32_data), 0)
                graph = helper.make_graph(
                    [helper.make_node('Cast', ['c'], ['f'], to=TensorProto.FLOAT),
                     helper.make_node('Add', ['x', 'f'], ['y'])], name,
                    [helper.make_tensor_value_info('x', TensorProto.FLOAT, [6])],
                    [helper.make_tensor_value_info('y', TensorProto.FLOAT, [6])], [constant])
                constants = [op['main'] for op in self.convert(graph, name).values() if op['type'] == 'Const']
                self.assertEqual([constant.get('float32s') for constant in constants], [expected])

    def test_nonzero_bool_mask(self):
        mask = helper.make_tensor('m', TensorProto.BOOL, [2, 3], [1, 0, 1, 1, 0, 1])
        graph = helper.make_graph([helper.make_node('NonZero', ['m'], ['y'])], 'nonzero', [],
                                  [helper.make_tensor_value_info('y', TensorProto.INT64, [2, 4])], [mask])
        # NonZero folds to a [count, rank] coordinate constant followed by a transpose.
        constants = {tuple(op['main']['dims']): op['main'].get('int32s')
                     for op in self.convert(graph, 'nonzero').values() if op['type'] == 'Const'}
        self.assertIn((4, 2), constants)
        self.assertEqual(constants[(4, 2)], [0, 0, 0, 2, 1, 0, 1, 2])


if __name__ == '__main__':
    unittest.main()
