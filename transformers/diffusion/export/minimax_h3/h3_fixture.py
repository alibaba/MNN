# SPDX-License-Identifier: Apache-2.0
"""Fixture container shared by the MiniMax-H3 reference dumper, the alignment harness and the MNN tests.

One directory holds `manifest.json` plus one raw little-endian `.bin` per tensor, so a C++ test can mmap a
tensor without a numpy or JSON-array parser in the hot path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "uint16": np.uint16,
}


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # bfloat16 has no numpy dtype, and widening it to float32 is lossless, so that is what lands on disk.
        if value.dtype is _torch_bfloat16():
            return value.float().numpy(), "bfloat16"
        return value.numpy(), None
    return np.asarray(value), None


def _torch_bfloat16():
    import torch

    return torch.bfloat16


class FixtureWriter:
    def __init__(self, root, metadata=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.metadata = dict(metadata or {})

    def add(self, name, value):
        array, logical = _to_numpy(value)
        array = np.ascontiguousarray(array)
        path = self.root / f"{name}.bin"
        path.write_bytes(array.tobytes())
        self.entries[name] = {
            "dtype": logical or str(array.dtype),
            "storage_dtype": str(array.dtype),
            "shape": list(array.shape),
            "file": path.name,
        }
        return array

    def close(self):
        manifest = {"metadata": self.metadata, "tensors": self.entries}
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


class FixtureReader:
    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.metadata = manifest["metadata"]
        self.entries = manifest["tensors"]

    def __contains__(self, name):
        return name in self.entries

    def names(self):
        return sorted(self.entries)

    def get(self, name, dtype=np.float32):
        entry = self.entries[name]
        raw = np.fromfile(self.root / entry["file"], dtype=_DTYPES[entry["storage_dtype"]])
        return raw.reshape(entry["shape"]).astype(dtype, copy=False)


def compare(name, actual, expected, cos_threshold=0.9999, max_rel=None):
    """Report cosine similarity and error norms of one tensor pair against a reference."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual.shape != expected.shape:
        return {"name": name, "ok": False, "reason": f"shape {actual.shape} vs {expected.shape}"}
    denominator = np.linalg.norm(actual) * np.linalg.norm(expected)
    cosine = float(actual @ expected / denominator) if denominator > 0 else 1.0
    absolute = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), 1e-6)
    result = {
        "name": name,
        "cosine": cosine,
        "max_abs": float(absolute.max()) if absolute.size else 0.0,
        "max_rel": float((absolute / scale).max()) if absolute.size else 0.0,
        "rms": float(np.sqrt(np.mean(absolute**2))) if absolute.size else 0.0,
        "ref_rms": float(np.sqrt(np.mean(expected**2))) if expected.size else 0.0,
    }
    result["ok"] = cosine >= cos_threshold and (max_rel is None or result["max_rel"] <= max_rel)
    return result


def format_report(rows):
    header = f"{'tensor':<44}{'cosine':>12}{'max_abs':>12}{'rel_rms':>12}  status"
    lines = [header, "-" * len(header)]
    for row in rows:
        if "reason" in row:
            lines.append(f"{row['name']:<44}{row['reason']:>36}  FAIL")
            continue
        relative_rms = row["rms"] / row["ref_rms"] if row["ref_rms"] > 0 else 0.0
        lines.append(
            f"{row['name']:<44}{row['cosine']:>12.6f}{row['max_abs']:>12.3e}{relative_rms:>12.3e}"
            f"  {'ok' if row['ok'] else 'FAIL'}"
        )
    return "\n".join(lines)
