#!/usr/bin/env python3
"""Regression test for the FuseTransformerC4 projection-fusion switches.

The pass rewrites the main graph and every subgraph. --transformerFuseGateUpProj=0
must be honoured in both: llmexport splices one subgraph per MoE expert into the
model, and an expert body is exactly the SwiGLU gate/up pattern the pass matches,
so a leak there silently fuses projections that --lora_split needs kept apart.

Builds a minimal graph carrying the SwiGLU pattern in the main graph AND in a
subgraph, then converts it twice. The flag-on run is the control: it must produce
FusedLinear in both places, otherwise the fixture no longer matches the pattern
and the flag-off assertion would pass vacuously.

Usage: testTransformerC4Switches.py [path/to/MNNConvert]
"""

import json
import os
import subprocess
import sys
import tempfile

# Reshape dims[1] must be positive and a multiple of 4 (matchPreConvertFromConv).
HIDDEN = 8
INTER = 8


def _conv(name, src, dst, in_ch, out_ch, seed):
    # Distinct weights per member: FuseDupOp runs in the same pass list and would
    # merge two byte-identical convolutions, dissolving the group under test.
    weight = [((i * 7 + seed) % 13) * 0.03125 for i in range(in_ch * out_ch)]
    bias = [(seed + i) * 0.0625 for i in range(out_ch)]
    return {
        "type": "Convolution",
        "name": name,
        "inputIndexes": [src],
        "outputIndexes": [dst],
        "main_type": "Convolution2D",
        "main": {
            "common": {
                "kernelX": 1, "kernelY": 1, "strideX": 1, "strideY": 1,
                "padX": 0, "padY": 0, "dilateX": 1, "dilateY": 1,
                "group": 1, "outputCount": out_ch, "inputCount": in_ch,
                "relu": False, "relu6": False, "padMode": "CAFFE",
            },
            "weight": weight,
            "bias": bias,
        },
        "defaultDimentionFormat": "NCHW",
    }


def _reshape(name, src, dst, dims):
    return {
        "type": "Reshape", "name": name,
        "inputIndexes": [src], "outputIndexes": [dst],
        "main_type": "Reshape", "main": {"dims": dims, "dimType": "NCHW"},
        "defaultDimentionFormat": "NCHW",
    }


def _convert(name, src, dst, source, dest):
    return {
        "type": "ConvertTensor", "name": name,
        "inputIndexes": [src], "outputIndexes": [dst],
        "main_type": "TensorConvertInfo", "main": {"source": source, "dest": dest},
        "defaultDimentionFormat": "NCHW",
    }


def _unary(name, src, dst, op):
    return {
        "type": "UnaryOp", "name": name,
        "inputIndexes": [src], "outputIndexes": [dst],
        "main_type": "UnaryOp", "main": {"opType": op, "T": "DT_FLOAT"},
        "defaultDimentionFormat": "NCHW",
    }


def _mul(name, a, b, dst):
    return {
        "type": "BinaryOp", "name": name,
        "inputIndexes": [a, b], "outputIndexes": [dst],
        "main_type": "BinaryOp",
        "main": {"opType": "MUL", "T": "DT_FLOAT", "activationType": 0},
        "defaultDimentionFormat": "NCHW",
    }


def build_swiglu(prefix, first_tensor, input_index):
    """SwiGLU gate/up cluster: the exact shape fuseGateUpProjGroups matches.

        x -> Reshape[-1,H,1,1] -> Convert(NCHW->NC4HW4) -+-> conv_gate -> Convert -> Reshape3d -> SILU -+-> MUL -> ABS
                                                         +-> conv_up   -> Convert -> Reshape3d --------+

    The MUL output must not be a graph output (the matcher rejects that), hence
    the trailing ABS. Returns (ops, tensor_names, output_tensor_index).
    """
    t = first_tensor
    def new(label):
        nonlocal t
        t += 1
        return t

    packed = new("packed")        # Reshape [-1,H,1,1]
    c4 = new("c4")                # NCHW -> NC4HW4
    gate_c = new("gate_conv")
    up_c = new("up_conv")
    gate_n = new("gate_nchw")
    up_n = new("up_nchw")
    gate_r = new("gate_3d")
    up_r = new("up_3d")
    silu = new("silu")
    mul = new("mul")
    out = new("out")

    ops = [
        _reshape(prefix + "reshape_in", input_index, packed, [-1, HIDDEN, 1, 1]),
        _convert(prefix + "to_c4", packed, c4, "NCHW", "NC4HW4"),
        _conv(prefix + "gate_proj", c4, gate_c, HIDDEN, INTER, seed=1),
        _conv(prefix + "up_proj", c4, up_c, HIDDEN, INTER, seed=5),
        _convert(prefix + "gate_to_nchw", gate_c, gate_n, "NC4HW4", "NCHW"),
        _reshape(prefix + "gate_view", gate_n, gate_r, [1, -1, INTER]),
        _convert(prefix + "up_to_nchw", up_c, up_n, "NC4HW4", "NCHW"),
        _reshape(prefix + "up_view", up_n, up_r, [1, -1, INTER]),
        _unary(prefix + "silu", gate_r, silu, "SILU"),
        _mul(prefix + "mul", silu, up_r, mul),
        _unary(prefix + "sink", mul, out, "ABS"),
    ]
    names = [prefix + n for n in ["packed", "c4", "gate_conv", "up_conv", "gate_nchw",
                                  "up_nchw", "gate_3d", "up_3d", "silu", "mul", "out"]]
    return ops, names, out


def build_model():
    # Main graph: Input at tensor 0, cluster on tensors 1..11.
    tensors = ["input"]
    ops = [{
        "type": "Input", "name": "input", "outputIndexes": [0],
        "main_type": "Input",
        "main": {"dims": [1, 1, HIDDEN], "dtype": "DT_FLOAT", "dformat": "NCHW"},
        "defaultDimentionFormat": "NCHW",
    }]
    main_ops, main_names, main_out = build_swiglu("main_", 0, 0)
    ops += main_ops
    tensors += main_names

    # Subgraph: its own tensor space, Input at 0.
    sub_tensors = ["sub_input"]
    sub_ops = [{
        "type": "Input", "name": "sub_input", "outputIndexes": [0],
        "main_type": "Input",
        "main": {"dims": [1, 1, HIDDEN], "dtype": "DT_FLOAT", "dformat": "NCHW"},
        "defaultDimentionFormat": "NCHW",
    }]
    s_ops, s_names, s_out = build_swiglu("sub_", 0, 0)
    sub_ops += s_ops
    sub_tensors += s_names

    return {
        "oplists": ops,
        "tensorName": tensors,
        "tensorNumber": len(tensors),
        "sourceType": "TENSORFLOW",
        "bizCode": "test",
        "outputName": [tensors[main_out]],
        "subgraphs": [{
            "name": "/expert/0_0",
            "inputs": [0],
            "outputs": [s_out],
            "tensors": sub_tensors,
            "nodes": sub_ops,
        }],
    }


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(r.stdout.decode("utf-8", "replace"))
        raise SystemExit("command failed: " + " ".join(cmd))
    return r.stdout.decode("utf-8", "replace")


def count_fused(graph):
    main = sum(1 for o in graph["oplists"] if o["type"] == "FusedLinear")
    sub = 0
    for sg in graph.get("subgraphs", []):
        sub += sum(1 for o in sg.get("nodes", []) if o["type"] == "FusedLinear")
    return main, sub, len(graph.get("subgraphs", []))


def convert(mnnconvert, workdir, gate_up_flag):
    src = os.path.join(workdir, "src.json")
    with open(src, "w") as f:
        json.dump(build_model(), f)

    staged = os.path.join(workdir, "staged.mnn")
    run([mnnconvert, "-f", "JSON", "--modelFile", src, "--MNNModel", staged])

    # FuseTransformerC4 only runs for MNN -> MNN at optimizeLevel 1 (cli.cpp).
    opt = os.path.join(workdir, "opt_%d.mnn" % gate_up_flag)
    run([mnnconvert, "-f", "MNN", "--modelFile", staged, "--MNNModel", opt,
         "--optimizeLevel=1", "--transformerFuseC4=1",
         "--transformerFuseGateUpProj=%d" % gate_up_flag])

    dumped = opt + ".json"
    run([mnnconvert, "-f", "MNN", "--modelFile", opt, "--JsonFile", dumped, "--mnn2json"])
    with open(dumped) as f:
        return count_fused(json.load(f))


def main():
    mnnconvert = sys.argv[1] if len(sys.argv) > 1 else "./MNNConvert"
    if not os.path.exists(mnnconvert):
        raise SystemExit("MNNConvert not found: " + mnnconvert)
    failures = []
    with tempfile.TemporaryDirectory() as workdir:
        on_main, on_sub, on_n = convert(mnnconvert, workdir, 1)
        off_main, off_sub, off_n = convert(mnnconvert, workdir, 0)

    print("flag=1: main FusedLinear=%d, subgraph FusedLinear=%d (%d subgraphs)"
          % (on_main, on_sub, on_n))
    print("flag=0: main FusedLinear=%d, subgraph FusedLinear=%d (%d subgraphs)"
          % (off_main, off_sub, off_n))

    # Controls: without these the flag-off assertions below are vacuous.
    if on_n != 1 or off_n != 1:
        failures.append("subgraph did not survive conversion")
    if on_main != 1:
        failures.append("control failed: flag=1 did not fuse the main graph, "
                        "so the fixture no longer matches fuseGateUpProjGroups")
    if on_sub != 1:
        failures.append("control failed: flag=1 did not fuse the subgraph")

    # The regression: 899c1ea41 built subgraph graphs without the switches, so
    # they took the constructor's `= true` defaults and fused regardless.
    if off_sub != 0:
        failures.append("--transformerFuseGateUpProj=0 ignored inside subgraphs")
    if off_main != 0:
        failures.append("--transformerFuseGateUpProj=0 ignored in the main graph")

    # 4 assertions: 2 controls (subgraph survives, flag=1 fuses both) and the
    # 2 flag=0 checks. Reported the way the CI summary collects results.
    total = 4
    for f in failures:
        print("FAIL: " + f)
    print('TEST_NAME_MODULE: FuseTransformerC4 开关测试\n'
          'TEST_CASE_AMOUNT_MODULE: {"blocked":0,"failed":%d,"passed":%d,"skipped":0}\n'
          % (len(failures), total - len(failures)))
    print('TEST_CASE={"name":"FuseTransformerC4 融合开关测试","failed":%d,"passed":%d}\n'
          % (len(failures), total - len(failures)))
    if failures:
        return 1
    print("PASS: gate/up fusion switch honoured in both the main graph and subgraphs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
