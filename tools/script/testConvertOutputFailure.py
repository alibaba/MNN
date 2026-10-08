#!/usr/bin/env python3
"""Check that failed model writes reach MNNConvert's exit status.

Requires the onnx Python package and a converter built with
MNN_BUILD_CONVERTER=ON. Run:
    python tools/script/testConvertOutputFailure.py /path/to/MNNConvert

The writable-output controls reload the regular and static Relu models and
check their CPU inference results. Linux /dev/full covers errors deferred until the
small serialized model is flushed/closed, rather than just open failures.
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


class ConvertOutputFailureTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mnn-output-failure-")
        self.addCleanup(temporary.cleanup)
        self.workdir = Path(temporary.name)
        self.model = self.workdir / "relu.onnx"
        graph = helper.make_graph(
            [helper.make_node("Relu", ["input"], ["output"])],
            "output_write_failure",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
        )
        onnx.checker.check_model(model)
        onnx.save(model, str(self.model))

    def convert(self, output, *extra_args):
        return subprocess.run(
            [self.converter, "-f", "ONNX", "--modelFile", str(self.model),
             "--MNNModel", str(output), "--bizCode", "output-write-test",
             *extra_args],
            cwd=str(self.workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )

    def assert_write_failed(self, output, *extra_args):
        result = self.convert(output, *extra_args)
        # A crash is not a correctly propagated conversion failure.
        self.assertGreater(result.returncode, 0, result.stdout)
        self.assertNotIn("Converted Success!", result.stdout)
        self.assertIn("Converted Failed!", result.stdout)
        self.assertIn(str(output), result.stdout)

    def assert_writable_output_and_cpu_inference(self, *extra_args):
        output = self.workdir / "relu.mnn"
        (self.workdir / "input.json").write_text(json.dumps({
            "inputs": [{"name": "input", "shape": [1, 4]}],
            "outputs": ["output"],
        }))
        (self.workdir / "input.txt").write_text("-2 -0.5 0 3\n")
        (self.workdir / "output.txt").write_text("0 0 0 3\n")
        backend = self.workdir / "cpu.json"
        backend.write_text(json.dumps({"backend": 0, "precision": 1}))
        result = self.convert(output, "--testdir", str(self.workdir),
                              "--testconfig", str(backend), *extra_args)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Converted Success!", result.stdout)
        self.assertIn("TEST_SUCCESS", result.stdout)
        self.assertNotIn("TESTERROR", result.stdout)
        self.assertNotIn("Skip check", result.stdout)
        self.assertGreater(output.stat().st_size, 0)
        return result

    def test_writable_output_and_cpu_inference(self):
        self.assert_writable_output_and_cpu_inference()

    def test_static_writable_output_and_cpu_inference(self):
        result = self.assert_writable_output_and_cpu_inference("--saveStaticModel")
        self.assertIn("gen Static Model", result.stdout)

    def test_missing_parent(self):
        output = self.workdir / "missing" / "relu.mnn"
        self.assert_write_failed(output)
        self.assertFalse(output.exists())

    def test_directory_as_output(self):
        self.assert_write_failed(self.workdir)
        self.assertTrue(self.workdir.is_dir())

    def test_static_missing_parent(self):
        output = self.workdir / "missing" / "relu.mnn"
        self.assert_write_failed(output, "--saveStaticModel")
        self.assertFalse(output.exists())

    def test_static_directory_as_output(self):
        self.assert_write_failed(self.workdir, "--saveStaticModel")
        self.assertTrue(self.workdir.is_dir())

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/dev/full").exists(),
                         "requires Linux /dev/full")
    def test_buffered_write_failure(self):
        self.assert_write_failed(Path("/dev/full"))

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/dev/full").exists(),
                         "requires Linux /dev/full")
    def test_static_buffered_write_failure(self):
        self.assert_write_failed(Path("/dev/full"), "--saveStaticModel")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converter", nargs="?", default="./MNNConvert")
    args = parser.parse_args()
    converter = Path(args.converter).resolve()
    if not converter.is_file():
        parser.error("MNNConvert not found: " + str(converter))
    ConvertOutputFailureTest.converter = str(converter)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
