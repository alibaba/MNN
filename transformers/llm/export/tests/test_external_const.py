import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mnn_converter import MNNConverter


class ExternalConstTest(unittest.TestCase):
    def test_integer_indices_survive_weight_rebuild(self):
        values = np.arange(16, dtype=np.int32).repeat(128)
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'model.weight'
            graph = Path(directory) / 'model.json'
            weights.write_bytes(values.tobytes())
            graph.write_text(json.dumps({'oplists': [{'type': 'Const', 'main': {
                'dataType': 'DT_INT32', 'dims': [2048], 'external': [0, values.nbytes]}}]}))
            converter = MNNConverter.__new__(MNNConverter)
            converter.mnn_weight_path = str(weights)
            MNNConverter.rebuild.__wrapped__(converter, str(graph))
            restored = json.loads(graph.read_text())['oplists'][0]['main']
            self.assertEqual(restored['int32s'], values.tolist())
            self.assertNotIn('float32s', restored)
            self.assertNotIn('external', restored)


if __name__ == '__main__':
    unittest.main()
