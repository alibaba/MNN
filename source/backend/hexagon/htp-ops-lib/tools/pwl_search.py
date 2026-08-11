#!/usr/bin/env python3
"""Generate and validate FP16 PWL candidates for Hexagon HVX kernels.

The simulator mirrors the current DSP implementation:
  * slopes and biases are stored as IEEE FP16;
  * the FP16 input is multiplied by the stored slope;
  * the bias is accumulated before one final FP16 rounding;
  * symmetry identities and saturation tails are applied by the kernel.

This is intentionally a host-only tool. Candidate tables must still pass the
Hexagon operator tests before being used for performance measurements.
"""

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


def fp16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def fp16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def fp32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits_fp16(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def stable_sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def silu(x: float) -> float:
    return x * stable_sigmoid(x)


def gelu_tanh(x: float) -> float:
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + math.tanh(inner))


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    function: Callable[[float], float]
    positive_limit: Callable[[float], float]
    negative_limit: Callable[[float], float]
    reflect: Callable[[float, float], float]
    range_limit: float
    max_abs_threshold: float


FUNCTIONS: Dict[str, FunctionSpec] = {
    "silu": FunctionSpec(
        "silu", silu, lambda x: x, lambda x: 0.0, lambda x, y: y - x, 8.0, 0.008
    ),
    "sigmoid": FunctionSpec(
        "sigmoid",
        stable_sigmoid,
        lambda x: 1.0,
        lambda x: 0.0,
        lambda x, y: 1.0 - y,
        8.0,
        0.005,
    ),
    "tanh": FunctionSpec(
        "tanh", math.tanh, lambda x: 1.0, lambda x: -1.0, lambda x, y: -y, 4.0, 0.009
    ),
    "gelu": FunctionSpec(
        "gelu", gelu_tanh, lambda x: x, lambda x: 0.0, lambda x, y: y - x, 4.0, 0.009
    ),
}


def quarter_edges(limit: float) -> Tuple[float, ...]:
    return tuple(0.25 * i for i in range(int(limit * 4.0) + 1))


VARIANTS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "uniform": {
        "silu": quarter_edges(8.0),
        "sigmoid": quarter_edges(8.0),
        "tanh": quarter_edges(4.0),
        "gelu": quarter_edges(4.0),
    },
    # One 16-entry VLUT bank for SiLU/Sigmoid:
    # [0, 2): 0.25, [2, 4): 0.5, [4, 8): 1.0.
    # Tanh/GELU need only the first 12 entries.
    "companded16": {
        "silu": (
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ),
        "sigmoid": (
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ),
        "tanh": (
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
        ),
        "gelu": (
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
        ),
    },
}

# A hardware-constrained search selects eight SiLU magnitude intervals. A
# 16-entry halfword LUT maps compressed exponent/mantissa states to them.
LEARNED8_SILU_EDGES: Tuple[float, ...] = (
    0.0,
    0.25,
    0.5,
    1.0,
    1.5,
    3.5,
    5.0,
    6.0,
    8.0,
)

LEARNED8_INDEX_LUT: Tuple[int, ...] = (0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 5, 6, 7, 7)


@dataclass(frozen=True)
class Coefficients:
    variant: str
    edges: Tuple[float, ...]
    slopes: Tuple[float, ...]
    biases: Tuple[float, ...]


@dataclass(frozen=True)
class Metrics:
    count: int
    max_abs: float
    max_abs_x: float
    mean_abs: float
    rmse: float


def generate_coefficients(spec: FunctionSpec, variant: str, edges: Sequence[float]) -> Coefficients:
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != spec.range_limit:
        raise ValueError(f"{spec.name}: edges must cover [0, {spec.range_limit}]")
    slopes: List[float] = []
    biases: List[float] = []
    for x0, x1 in zip(edges[:-1], edges[1:]):
        if x1 <= x0:
            raise ValueError(f"{spec.name}: edges must be strictly increasing")
        slope = fp16((spec.function(x1) - spec.function(x0)) / (x1 - x0))
        # Match the tables in unary_ops.cc: recompute b after quantizing a.
        bias = fp16(spec.function(x0) - slope * x0)
        slopes.append(slope)
        biases.append(bias)
    return Coefficients(variant, tuple(edges), tuple(slopes), tuple(biases))


def centered_chord_coefficients(
    function: Callable[[float], float],
    x0: float,
    x1: float,
    preserve_zero: bool,
    extra_inputs: Sequence[float] = (),
) -> Tuple[float, float]:
    slope = fp16((function(x1) - function(x0)) / (x1 - x0))
    interval_min = min(x0, x1)
    interval_max = max(x0, x1)
    samples: List[Tuple[float, float]] = []
    for x in finite_fp16_values():
        if interval_min <= x < interval_max:
            samples.append((x, function(x)))
    for original_x in extra_inputs:
        quantized_x = fp16(original_x)
        if interval_min <= quantized_x < interval_max:
            samples.append((quantized_x, function(original_x)))
    # Centering the chord's residual range is the minimax bias for a fixed
    # slope. Keep f(0) exact in the first interval; its error remains below
    # the global limit and avoids perturbing exact zero inputs.
    if preserve_zero:
        return slope, 0.0
    residuals = [expected - slope * x for x, expected in samples]
    center = fp16(0.5 * (min(residuals) + max(residuals)))
    center_bits = fp16_bits(center)
    # Final FP16 rounding makes the error piecewise constant. Search a small,
    # deterministic neighborhood around the analytic center.
    best = (math.inf, math.inf, center)
    for bits in range(max(0, center_bits - 8), min(0x10000, center_bits + 9)):
        bias = bits_fp16(bits)
        errors = [abs(fp16(x * slope + bias) - expected) for x, expected in samples]
        candidate = (max(errors), sum(errors), bias)
        if candidate < best:
            best = candidate
    return slope, best[2]


def hexagon_unary_test_inputs() -> List[float]:
    count = 8193
    inputs = [
        fp32(fp32(-12.0) + fp32(fp32(24.0) * fp32(i) / fp32(count - 1)))
        for i in range(count)
    ]
    cursor = 0
    for edge in range(-32, 33):
        x = fp32(fp32(0.25) * fp32(edge))
        for delta in (-0.01, 0.0, 0.01):
            inputs[cursor] = fp32(x + fp32(delta))
            cursor += 1
    for value in (-100.0, -12.0, -8.0, -4.0, -0.0, 0.0, 4.0, 8.0, 12.0, 100.0):
        inputs[cursor] = fp32(value)
        cursor += 1
    return inputs


def generate_learned8_silu_coefficients() -> Coefficients:
    slopes: List[float] = []
    biases: List[float] = []
    unit_inputs = hexagon_unary_test_inputs()
    for x0, x1 in zip(LEARNED8_SILU_EDGES[:-1], LEARNED8_SILU_EDGES[1:]):
        slope, bias = centered_chord_coefficients(
            silu,
            x0,
            x1,
            preserve_zero=(x0 == 0.0),
            extra_inputs=unit_inputs,
        )
        slopes.append(slope)
        biases.append(bias)
    return Coefficients(
        "learned8", LEARNED8_SILU_EDGES, tuple(slopes), tuple(biases)
    )


def segment_index(x: float, edges: Sequence[float]) -> int:
    lo = 0
    hi = len(edges) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if x < edges[mid]:
            hi = mid
        else:
            lo = mid
    return min(lo, len(edges) - 2)


def uniform_index16(x: float) -> int:
    scaled = fp16(x * 4.0)
    scaled = min(scaled, 15.0)
    shifted = fp16(scaled + 16.0)
    return (fp16_bits(shifted) >> 6) & 0xF


def kernel_segment_index(x: float, coeffs: Coefficients) -> int:
    if coeffs.variant == "learned8":
        raw_state = max((fp16_bits(abs(x)) >> 8) - 48, 0)
        state = raw_state - 8 if raw_state >= 16 else raw_state >> 1
        return LEARNED8_INDEX_LUT[min(state, len(LEARNED8_INDEX_LUT) - 1)]
    if coeffs.variant == "companded16":
        if x >= 2.0:
            return 8 + ((fp16_bits(x) >> 8) & 0x7)
        return uniform_index16(x)
    if x >= 4.0:
        return 16 + uniform_index16(fp16(x - 4.0))
    return uniform_index16(x)


def evaluate_positive(x: float, coeffs: Coefficients) -> float:
    index = kernel_segment_index(x, coeffs)
    # QF16 multiply/add keeps extra accumulator precision and rounds to FP16
    # once when converted back to Vhf.
    return fp16(x * coeffs.slopes[index] + coeffs.biases[index])


def evaluate_kernel(x: float, spec: FunctionSpec, coeffs: Coefficients) -> float:
    abs_x = abs(x)
    if abs_x >= spec.range_limit:
        return fp16(spec.negative_limit(abs_x) if x < 0.0 else spec.positive_limit(abs_x))
    positive_y = evaluate_positive(abs_x, coeffs)
    return fp16(spec.reflect(abs_x, positive_y) if x < 0.0 else positive_y)


def finite_fp16_values() -> Iterable[float]:
    for bits in range(0x10000):
        value = bits_fp16(bits)
        if math.isfinite(value):
            yield value


def calculate_metrics(spec: FunctionSpec, coeffs: Coefficients) -> Metrics:
    count = 0
    max_abs = -1.0
    max_abs_x = 0.0
    sum_abs = 0.0
    sum_squared = 0.0
    for x in finite_fp16_values():
        expected = spec.function(x)
        actual = evaluate_kernel(x, spec, coeffs)
        error = abs(actual - expected)
        count += 1
        sum_abs += error
        sum_squared += error * error
        if error > max_abs:
            max_abs = error
            max_abs_x = x
    return Metrics(count, max_abs, max_abs_x, sum_abs / count, math.sqrt(sum_squared / count))


def calculate_quantized_input_metrics(
    spec: FunctionSpec, coeffs: Coefficients, inputs: Sequence[float]
) -> Metrics:
    count = 0
    max_abs = -1.0
    max_abs_x = 0.0
    sum_abs = 0.0
    sum_squared = 0.0
    for original_x in inputs:
        expected = spec.function(original_x)
        actual = evaluate_kernel(fp16(original_x), spec, coeffs)
        error = abs(actual - expected)
        count += 1
        sum_abs += error
        sum_squared += error * error
        if error > max_abs:
            max_abs = error
            max_abs_x = original_x
    return Metrics(count, max_abs, max_abs_x, sum_abs / count, math.sqrt(sum_squared / count))


def count_index_mismatches(spec: FunctionSpec, coeffs: Coefficients) -> int:
    mismatches = 0
    for x in finite_fp16_values():
        if abs(x) >= spec.range_limit:
            continue
        if x < 0.0:
            continue
        expected = segment_index(x, coeffs.edges)
        actual = kernel_segment_index(x, coeffs)
        if actual != expected:
            mismatches += 1
    return mismatches


def format_table(name: str, values: Sequence[float]) -> str:
    bits = [f"0x{fp16_bits(value):04x}" for value in values]
    rows = [", ".join(bits[i : i + 8]) for i in range(0, len(bits), 8)]
    body = ",\n    ".join(rows)
    return f"static const uint16_t {name}[{len(bits)}] = {{\n    {body}\n}};"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=sorted(list(VARIANTS) + ["learned8"]), default="companded16"
    )
    parser.add_argument(
        "--function", choices=["all"] + sorted(FUNCTIONS), default="all", dest="function_name"
    )
    parser.add_argument("--emit-c", action="store_true", help="print FP16 slope/bias tables")
    parser.add_argument(
        "--check", action="store_true", help="fail when max error exceeds the Hexagon unit-test limit"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.variant == "learned8":
        if args.function_name not in ("all", "silu"):
            print("learned8 currently applies only to SiLU", file=sys.stderr)
            return 2
        names = ["silu"]
    else:
        names = sorted(FUNCTIONS) if args.function_name == "all" else [args.function_name]
    failed = False
    for name in names:
        spec = FUNCTIONS[name]
        if args.variant == "learned8":
            coeffs = generate_learned8_silu_coefficients()
        else:
            edges = VARIANTS[args.variant][name]
            coeffs = generate_coefficients(spec, args.variant, edges)
        index_mismatches = count_index_mismatches(spec, coeffs)
        metrics = calculate_metrics(spec, coeffs)
        quantized_metrics = (
            calculate_quantized_input_metrics(spec, coeffs, hexagon_unary_test_inputs())
            if args.variant == "learned8"
            else None
        )
        effective_max = max(
            metrics.max_abs,
            quantized_metrics.max_abs if quantized_metrics is not None else 0.0,
        )
        status = "PASS" if effective_max <= spec.max_abs_threshold else "FAIL"
        quantized_summary = (
            f" quantized_grid_max={quantized_metrics.max_abs:.8f}"
            f" at x={quantized_metrics.max_abs_x:.8f}"
            if quantized_metrics is not None
            else ""
        )
        print(
            f"{name:7s} {args.variant:11s} segments={len(coeffs.slopes):2d} "
            f"max_abs={metrics.max_abs:.8f} at x={metrics.max_abs_x:.8f} "
            f"mean_abs={metrics.mean_abs:.8f} rmse={metrics.rmse:.8f} "
            f"index_mismatches={index_mismatches:4d}{quantized_summary} "
            f"limit={spec.max_abs_threshold:.8f} {status}"
        )
        if args.emit_c:
            print(format_table(f"{name}_slope", coeffs.slopes))
            print(format_table(f"{name}_bias", coeffs.biases))
        failed = failed or status == "FAIL"
    return 1 if args.check and failed else 0


if __name__ == "__main__":
    sys.exit(main())
