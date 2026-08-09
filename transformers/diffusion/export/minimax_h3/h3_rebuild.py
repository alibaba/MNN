# SPDX-License-Identifier: Apache-2.0
"""Rebuild the MiniMax-H3 custom export ops into real MNN ops.

`h3_modules` emits the LLM exporter's `FusedAttention` while tracing, so the ONNX carries an
`LlmExporter::FusedAttention` node instead of a matmul chain. That is what keeps export memory independent of
sequence length -- the score tensor is never allocated -- and it emits the attention op directly rather than
relying on the converter's `FuseAttention` pattern, which does not match every form.

MNNConvert imports the unknown node as an `Extra` op. This pass replaces it with an `Attention` op carrying
`kv_cache = false`, mirroring `MNNConverter.rebuild_attnention` in the LLM exporter.

    python h3_rebuild.py --mnn h3_blocks_0.mnn --mnnconvert /path/to/MNNConvert

With `--drop_attention_mask` the mask input is removed from every rebuilt attention. H3 attends over the whole
packed sequence, so the exported mask is all zeros, and an attention op without a mask is MNN's bidirectional
path -- the same computation without a seqLen^2 tensor. At 21763 rows that mask alone is 1.9 GB.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def extra_attr(op, key, field, default=None):
    """Read one Extra-op attribute. The dump carries every field on every attr, so the type must be named."""
    for attr in op.get("main", {}).get("attr", []) or []:
        if attr.get("key") == key:
            return attr.get(field, default)
    return default


def rebuild_attention(op, drop_mask):
    """`LlmExporter::FusedAttention` -> `Attention`, keeping the op's own inputs and outputs."""
    inputs = op["inputIndexes"]
    return {
        "type": "Attention",
        "name": extra_attr(op, "name", "s", op.get("name", "attention")),
        "inputIndexes": inputs[:3] if drop_mask else inputs,
        "outputIndexes": op["outputIndexes"],
        "main_type": "AttentionParam",
        "main": {
            "kv_cache": bool(extra_attr(op, "kv_cache", "i", 0)),
            "layer_index": extra_attr(op, "layer_index", "i", -1),
            "kv_shared_layer_index": extra_attr(op, "kv_shared_layer_index", "i", -1),
            "bidirectional": drop_mask,
        },
        "defaultDimentionFormat": "NHWC",
    }


def shrink_dead_inputs(graph, key):
    """Collapse Input ops nobody reads to one element, so a dropped mask stops costing memory."""
    consumed = set()
    for op in graph[key]:
        for index in op.get("inputIndexes") or []:
            consumed.add(index)
    shrunk = 0
    for op in graph[key]:
        if op.get("type") != "Input" or any(index in consumed for index in op.get("outputIndexes") or []):
            continue
        dims = op.get("main", {}).get("dims") or []
        if len(dims) > 0 and any(dimension > 1 for dimension in dims):
            op["main"]["dims"] = [1] * len(dims)
            shrunk += 1
    return shrunk


def rebuild_graph(graph, drop_mask=False):
    key = "oplists" if "oplists" in graph else "nodes"
    rebuilt = []
    counts = {}
    for op in graph[key]:
        extra_type = op.get("main", {}).get("type") if isinstance(op.get("main"), dict) else None
        if op.get("type") == "Extra" and extra_type == "FusedAttention":
            op = rebuild_attention(op, drop_mask)
            counts["FusedAttention"] = counts.get("FusedAttention", 0) + 1
        if drop_mask and op.get("type") == "Attention":
            # An earlier pass may already have turned the custom op into a real one.
            op["inputIndexes"] = op["inputIndexes"][:3]
            op.setdefault("main", {})["bidirectional"] = True
            counts["droppedMask"] = counts.get("droppedMask", 0) + 1
        rebuilt.append(op)
    graph[key] = rebuilt
    if drop_mask and counts:
        shrunk = shrink_dead_inputs(graph, key)
        if shrunk:
            counts["shrunkInputs"] = counts.get("shrunkInputs", 0) + shrunk
    return counts


def main():
    parser = argparse.ArgumentParser(description="Rebuild MiniMax-H3 custom export ops into MNN ops.")
    parser.add_argument("--mnn", required=True, help="Model to rewrite in place.")
    parser.add_argument("--mnnconvert", help="Defaults to build_h3/MNNConvert or build/MNNConvert.")
    parser.add_argument("--keep_json", action="store_true")
    parser.add_argument("--drop_attention_mask", action="store_true",
                        help="Remove the all-zero additive mask, leaving MNN's bidirectional attention.")
    args = parser.parse_args()

    model = Path(args.mnn)
    mnnconvert = args.mnnconvert
    if not mnnconvert:
        root = Path(__file__).resolve().parents[4]
        for candidate in ("build_h3/MNNConvert", "build/MNNConvert"):
            if (root / candidate).exists():
                mnnconvert = (root / candidate).as_posix()
                break
        mnnconvert = mnnconvert or shutil.which("MNNConvert") or "MNNConvert"

    json_path = model.with_suffix(".rebuild.json")
    subprocess.run(
        [mnnconvert, "-f", "MNN", "--modelFile", model.as_posix(), "--JsonFile", json_path.as_posix()],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    document = json.loads(json_path.read_text())

    counts = rebuild_graph(document, args.drop_attention_mask)
    for subgraph in document.get("subgraphs", []) or []:
        for name, value in rebuild_graph(subgraph, args.drop_attention_mask).items():
            counts[name] = counts.get(name, 0) + value
    if not counts:
        print(f"{model.name}: no custom ops to rebuild")
        json_path.unlink(missing_ok=True)
        return 0

    json_path.write_text(json.dumps(document))
    # The JSON round trip leaves the external .weight sidecar untouched; only the graph is rewritten.
    subprocess.run(
        [mnnconvert, "-f", "JSON", "--modelFile", json_path.as_posix(), "--MNNModel", model.as_posix()],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    if not args.keep_json:
        json_path.unlink(missing_ok=True)
    print(f"{model.name}: rebuilt {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
