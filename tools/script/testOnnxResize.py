#!/usr/bin/env python3
"""Check ONNX Resize's default mode and explicit mode overrides.

Requires onnx and a converter built with MNN_BUILD_CONVERTER=ON. Run:
    python tools/script/testOnnxResize.py /path/to/MNNConvert
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper


class OnnxResizeTest(unittest.TestCase):
    def test_resize_mode(self):
        with tempfile.TemporaryDirectory(prefix="mnn-resize-") as directory:
            workdir = Path(directory)
            (workdir / "input.json").write_text(json.dumps({
                "inputs": [{"name": "x", "shape": [1, 1, 1, 2]}],
                "outputs": ["y"],
            }))
            (workdir / "x.txt").write_text("0 10\n")
            backend = workdir / "cpu.json"
            backend.write_text(json.dumps({"backend": 0, "precision": 1, "mode": 1}))
            for mode, expected in [(None, "0 0 10 10\n"),
                                   ("nearest", "0 0 10 10\n"),
                                   ("linear", "0 2.5 7.5 10\n")]:
                with self.subTest(mode=mode):
                    attributes = {} if mode is None else {"mode": mode}
                    graph = helper.make_graph(
                        [helper.make_node("Resize", ["x", "", "", "sizes"], ["y"],
                                          **attributes)],
                        "resize_mode",
                        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 2])],
                        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 4])],
                        [helper.make_tensor("sizes", TensorProto.INT64, [4], [1, 1, 1, 4])],
                    )
                    model = helper.make_model(
                        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
                    )
                    onnx.checker.check_model(model, full_check=True)
                    onnx.save(model, str(workdir / "resize.onnx"))
                    (workdir / "y.txt").write_text(expected)
                    result = subprocess.run(
                        [self.converter, "-f", "ONNX", "--modelFile", "resize.onnx",
                         "--MNNModel", "resize.mnn", "--bizCode", "resize-test",
                         "--optimizeLevel", "1", "--testdir", str(workdir),
                         "--testconfig", str(backend), "--thredhold", "1e-6"],
                        cwd=str(workdir), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, timeout=60,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn("Converted Success!", result.stdout)
                    self.assertNotIn("TESTERROR", result.stdout)
                    self.assertNotIn("Skip check", result.stdout)
                    self.assertIn("TEST_SUCCESS", result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converter", nargs="?", default="./MNNConvert")
    args = parser.parse_args()
    converter = Path(args.converter).resolve()
    if not converter.is_file():
        parser.error("MNNConvert not found: " + str(converter))
    OnnxResizeTest.converter = str(converter)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
