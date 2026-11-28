"""Run with onnx installed and MNN_CONVERTER_TEST_BUILD pointing at the converter build."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

BUILD = Path(os.environ.get("MNN_CONVERTER_TEST_BUILD", Path(__file__).resolve().parents[3] / "build"))


def converted_ops(nodes, constants, inputs):
    tensor = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 4])
    graph = helper.make_graph(nodes, "fusion", [tensor(name) for name in inputs], [tensor("y")], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    with tempfile.TemporaryDirectory() as directory:
        onnx.save(model, Path(directory) / "net.onnx")
        for args in [
            ["-f", "ONNX", "--modelFile", "net.onnx", "--MNNModel", "net.mnn", "--bizCode", "test"],
            ["-f", "MNN", "--modelFile", "net.mnn", "--JsonFile", "net.json"],
        ]:
            subprocess.run([BUILD / "MNNConvert", *args], cwd=directory, check=True, stdout=subprocess.PIPE)
        return json.loads((Path(directory) / "net.json").read_text())["oplists"]


class RmsEpsilonTest(unittest.TestCase):
    def test_epsilon(self):
        for dims, values in [([], [0.125]), ([4], [0.125, 0.25, 1.0, 2.0])]:
            with self.subTest(dims=dims):
                constants = [
                    helper.make_tensor("epsilon", TensorProto.FLOAT, dims, values),
                    helper.make_tensor("exponent", TensorProto.FLOAT, [], [-0.5]),
                ]
                nodes = [
                    helper.make_node("Mul", ["x", "x"], ["square"]),
                    helper.make_node("ReduceMean", ["square"], ["mean"], axes=[1], keepdims=1),
                    helper.make_node("Add", ["mean", "epsilon"], ["adjusted"]),
                    helper.make_node("Pow", ["adjusted", "exponent"], ["inv"]),
                    helper.make_node("Mul", ["x", "inv"], ["y"]),
                ]
                ops = converted_ops(nodes, constants, ["x"])
                self.assertEqual(any(op["type"] == "LayerNorm" for op in ops), len(values) == 1)


if __name__ == "__main__":
    unittest.main()
