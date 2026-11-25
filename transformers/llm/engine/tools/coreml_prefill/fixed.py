"""Build one fixed-weight Core ML model per layer from the benchmark's MIL templates."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import coremltools as ct
from coremltools import libmilstoragepython as storage
from coremltools.proto import MIL_pb2 as mil, Model_pb2
import numpy as np


PROJECTIONS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


def requantize(source, rows, cols, offset):
    """Read MNN's high-nibble-first asymmetric Q4/G64 and produce symmetric W8 per row."""
    if min(rows, cols) <= 0 or cols % 64 or offset < 0:
        raise ValueError("Invalid Q4/G64 projection layout")
    raw = np.frombuffer(source, "u1", rows * cols // 2, offset).reshape(rows, cols // 2)
    coeff = np.frombuffer(source, "<f2", rows * cols // 32, offset + raw.size).reshape(rows, cols // 64, 2)
    out = np.empty((rows, cols), "i1")
    scale = np.empty((rows, 1), "<f2")
    for start in range(0, rows, 96):
        batch = slice(start, start + 96)
        codes = np.empty((len(raw[batch]), cols), "i1")
        codes[:, ::2] = (raw[batch] >> 4).astype("i1") - 8
        codes[:, 1::2] = (raw[batch] & 15).astype("i1") - 8
        bias, step = coeff[batch].astype("f4").transpose(2, 0, 1)
        weights = (codes.reshape(len(codes), -1, 64) * step[:, :, None] + bias[:, :, None]).reshape(codes.shape)
        peak = np.abs(weights).max(axis=1, keepdims=True)
        scale[batch] = np.where(peak == 0, 1, np.maximum(peak / 127, 2**-24))
        out[batch] = np.rint(weights / scale[batch].astype("f4")).clip(-127, 127)
    return out, scale


def tensor(value, dtype, shape):
    target = value.type.tensorType
    target.dataType = dtype
    target.rank = len(shape)
    for size in shape:
        target.dimensions.add().constant.size = size


def constant_weight(name, codes, scales, writer):
    op = mil.Operation(type="constexpr_blockwise_shift_scale")
    output = op.outputs.add(name=name)
    tensor(output, mil.FLOAT16, codes.shape)
    attr = op.attributes["name"]
    tensor(attr, mil.STRING, ())
    attr.immediateValue.tensor.strings.values.append(name)
    offsets = (writer.write_int8_data(codes.ravel()), writer.write_fp16_data(scales.view("u2").ravel()))
    for key, dtype, shape, offset in zip(("data", "scale"), (mil.INT8, mil.FLOAT16),
                                         (codes.shape, scales.shape), offsets):
        value = op.inputs[key].arguments.add().value
        tensor(value, dtype, shape)
        value.blobFileValue.fileName = "@model_path/weights/weight.bin"
        value.blobFileValue.offset = offset
    return op


def rescale_swiglu(block):
    """Keep the up branch and down accumulation in FP16 range in Qwen3-4B layer 6."""
    def scalar(name, value):
        op = mil.Operation(type="const")
        tensor(op.outputs.add(name=name), mil.FLOAT16, ())
        attr = op.attributes["val"]
        attr.type.CopyFrom(op.outputs[0].type)
        attr.immediateValue.tensor.bytes.values = struct.pack("<e", value)
        return op

    def multiply(name, x, factor, output_type):
        op = mil.Operation(type="mul")
        op.inputs["x"].arguments.add(name=x)
        op.inputs["y"].arguments.add(name=factor)
        op.outputs.add(name=name).type.CopyFrom(output_type)
        return op

    ops, up_values = [], {}
    scaled = restored = 0
    for original in block.operations:
        op = copy.deepcopy(original)
        output = op.outputs[0]
        if op.type == "linear":
            weight = op.inputs["weight"].arguments[0].name
            if weight == "wu":
                up_values[output.name] = output.type
            elif weight == "wd":
                name = output.name
                output.name = "small_down"
                ops.extend((op, multiply(name, output.name, "down_gain", output.type)))
                restored += 1
                continue
        elif op.type in ("reshape", "transpose") and op.inputs["x"].arguments[0].name in up_values:
            up_values[output.name] = output.type
        elif op.type == "mul":
            for key in ("x", "y"):
                arg = op.inputs[key].arguments[0]
                if arg.name in up_values:
                    ops.extend((scalar("up_factor", 1 / 32), scalar("down_gain", 32),
                                multiply("small_up", arg.name, "up_factor", up_values[arg.name])))
                    arg.name = "small_up"
                    scaled += 1
        ops.append(op)
    if (scaled, restored) != (1, 1):
        raise ValueError("Could not identify the expected up and down paths for Qwen3-4B")
    del block.operations[:]
    block.operations.extend(ops)


def bind_weights(spec, matrices, source, blobs):
    fn = spec.mlProgram.functions["main"]
    block = fn.block_specializations[fn.opset]
    writer = storage._BlobStorageWriter(str(blobs / "weight.bin"))
    for name in PROJECTIONS:
        codes, scales = requantize(source, *matrices[name])
        block.operations.insert(0, constant_weight(name, codes, scales, writer))
    del writer
    for inputs in (fn.inputs, spec.description.input):
        if set(item.name for item in inputs).intersection(PROJECTIONS) != set(PROJECTIONS):
            raise ValueError("MIL template has unexpected weight inputs")
        for i in reversed(range(len(inputs))):
            if inputs[i].name in PROJECTIONS:
                del inputs[i]
    bound = []
    for op in block.operations:
        if op.type == "matmul" and op.inputs["y"].arguments[0].name in PROJECTIONS:
            bound.append(op.inputs["y"].arguments[0].name)
            op.type = "linear"
            op.inputs["weight"].CopyFrom(op.inputs["y"])
            for key in ("y", "transpose_x", "transpose_y"):
                del op.inputs[key]
    if sorted(bound) != sorted(PROJECTIONS):
        raise ValueError("Expected seven projections")
    return block


def build_fixed(export, destination, qwen_range_fix):
    source = np.memmap((export / "source.txt").read_text(), dtype="u1", mode="r")
    layers = {}
    for line in (export / "projections.tsv").read_text().splitlines():
        index, name, rows, cols, offset = line.split("\t")
        layer = layers.setdefault(int(index), {})
        if name in layer:
            raise ValueError("Duplicate projection in exported layout")
        layer[name] = (int(rows), int(cols), int(offset))
    if not layers or sorted(layers) != list(range(len(layers))) or any(set(x) != set(PROJECTIONS) for x in layers.values()):
        raise ValueError("Incomplete layer export")
    if qwen_range_fix and (len(layers) != 36 or layers[6]["wq"][:2] != (4096, 2560)
                           or layers[6]["wg"][:2] != (9728, 2560)):
        raise ValueError("--qwen-range-fix requires the Qwen3-4B layout")
    metrics = []
    for idx, projections in sorted(layers.items()):
        begin = time.perf_counter()
        spec = Model_pb2.Model.FromString((export / f"{idx}.mlmodel").read_bytes())
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            blobs = Path(tmp) / "weights"
            blobs.mkdir()
            block = bind_weights(spec, projections, source, blobs)
            if idx == 6 and qwen_range_fix:
                rescale_swiglu(block)
            pkg = Path(tmp) / f"128-{idx}.mlpackage"
            ct.utils.save_spec(spec, str(pkg), weights_dir=str(blobs))
            built = time.perf_counter()
            subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(destination)],
                           check=True, stdout=subprocess.DEVNULL)
        metrics.append(dict(layer=idx, convert_seconds=built - begin, compile_seconds=time.perf_counter() - built))
    (destination.parent / "build.json").write_text(json.dumps(metrics, indent=2) + "\n")
