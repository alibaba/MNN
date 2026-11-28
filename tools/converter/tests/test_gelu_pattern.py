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


class GeluPatternTest(unittest.TestCase):
    def test_formula_and_inputs(self):
        for factor, outer, expected in [(0.5, "x", True), (0.25, "x", False), (0.5, "other", False)]:
            with self.subTest(factor=factor, outer=outer):
                coefficients = {"half": factor, "one": 1, "scale": 0.7978845608, "cubic": 0.044715, "three": 3}
                constants = [helper.make_tensor(k, TensorProto.FLOAT, [], [v]) for k, v in coefficients.items()]
                nodes = [helper.make_node(op, args, [result]) for op, args, result in [
                    ("Pow", ["x", "three"], "cube"),
                    ("Mul", ["cube", "cubic"], "term"),
                    ("Add", ["x", "term"], "poly"),
                    ("Mul", ["poly", "scale"], "scaled"),
                    ("Tanh", ["scaled"], "activation"),
                    ("Add", ["activation", "one"], "sum"),
                    ("Mul", ["sum", "half"], "gate"),
                    ("Mul", ["gate", outer], "y"),
                ]]
                ops = converted_ops(nodes, constants, ["x"] if outer == "x" else ["x", "other"])
                self.assertEqual(any((op.get("main") or {}).get("opType") == "GELU" for op in ops), expected)


if __name__ == "__main__":
    unittest.main()
