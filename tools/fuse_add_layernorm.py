#!/usr/bin/env python3
"""
Fuse Add+LayerNorm patterns in MNN JSON model.
Detects: BinaryOp(Add) → LayerNorm
Rewrites to: LayerNorm as single op with 2 inputs (data + residual)
"""
import json, sys, re
from collections import defaultdict

def fuse(model_path, output_path=None):
    with open(model_path) as f:
        model = json.load(f)

    ops = model['oplists']
    n = len(ops)

    # Build tensor producer map
    producer = {}  # tensor_idx -> op_idx
    for i, op in enumerate(ops):
        for out in (op.get('outputIndex') or []):
            producer[out] = i

    # Find Add+LayerNorm patterns
    add_ln_pairs = []
    for i in range(n - 1):
        op1, op2 = ops[i], ops[i+1]

        # Check op1 is BinaryOp(Add)
        if op1.get('type') != 'BinaryOp':
            continue
        attrs = op1.get('attributes') or {}
        if attrs.get('opType', -1) != 0:  # 0 = ADD
            continue

        # Check op2 is LayerNorm
        if op2.get('type') != 'LayerNorm':
            continue

        # Check tensor connection: op1's output connects to op2's input
        op1_out = set(op1.get('outputIndex') or [])
        op2_in = set(op2.get('inputIndex') or [])
        if not (op1_out & op2_in):
            continue

        add_ln_pairs.append(i)

    print(f"Found {len(add_ln_pairs)} Add+LayerNorm pairs to fuse")

    # Process in reverse order (to keep indices stable)
    for idx in reversed(add_ln_pairs):
        add_op = ops[idx]
        ln_op  = ops[idx + 1]

        # Get Add's inputs (data tensors)
        add_inputs = add_op.get('inputIndex', [])
        if len(add_inputs) < 2:
            continue

        # Get LayerNorm's original input
        ln_inputs = ln_op.get('inputIndex', [])

        # The LayerNorm should have 1 input (the Add's output)
        # We want to change it to 2 inputs: [original_data, residual]
        # original_data ≈ add_inputs[0], residual ≈ add_inputs[1]
        # This is an approximation - may need manual verification

        # Set LayerNorm inputs: data + residual
        ln_op['inputIndex'] = [add_inputs[0], add_inputs[1]]

        # Remove the Add op from the list
        ops.pop(idx)

    # Write output
    if output_path is None:
        output_path = model_path.replace('.json', '_fused.json')

    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)

    print(f"Fused model saved to {output_path}")

    # Stats
    layer_count = len(add_ln_pairs)
    print(f"Total ops: {len(ops)} (removed {layer_count} Add ops)")
    return output_path

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: fuse_add_layernorm.py model.mnn.json [output.json]")
        sys.exit(1)
    fuse(*sys.argv[1:3])
