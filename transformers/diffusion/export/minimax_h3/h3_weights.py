# SPDX-License-Identifier: Apache-2.0
"""Stream MiniMax-H3 checkpoint tensors into the export-friendly module names.

The checkpoint is 66 GB of bfloat16 across 14 shards, so nothing here loads more than the tensors one
partition needs. The mapping also flattens the reference's wrappers: `nn.RMSNorm` weights become plain
parameters, `attn.to_out` loses its `ModuleList` index and the SwiGLU `ff.net.{0,2}` become `ff.{proj,out}`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

# The reference keeps these in float32 in an otherwise bfloat16 checkpoint.
FP32_PATTERNS = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")


@lru_cache(maxsize=4)
def _weight_map(model_path):
    index = json.loads(
        (Path(model_path) / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    return index["weight_map"]


def read_tensors(model_path, keys):
    """Read `keys` from the sharded checkpoint, opening each shard once."""
    from safetensors import safe_open

    directory = Path(model_path)
    weight_map = _weight_map(str(model_path))
    per_shard = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"{key} is not in the checkpoint index of {model_path}")
        per_shard.setdefault(weight_map[key], []).append(key)

    tensors = {}
    for shard, shard_keys in sorted(per_shard.items()):
        with safe_open((directory / shard).as_posix(), framework="pt") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def _attention_mapping(source, target):
    return {
        f"{source}.to_q.weight": f"{target}.to_q.weight",
        f"{source}.to_k.weight": f"{target}.to_k.weight",
        f"{source}.to_v.weight": f"{target}.to_v.weight",
        f"{source}.to_out.0.weight": f"{target}.to_out.weight",
        f"{source}.norm_q.weight": f"{target}.norm_q",
        f"{source}.norm_k.weight": f"{target}.norm_k",
    }


def _block_mapping(source, target):
    mapping = _attention_mapping(f"{source}.attn", f"{target}.attn")
    mapping.update(
        {
            f"{source}.norm1.weight": f"{target}.norm1",
            f"{source}.norm2.weight": f"{target}.norm2",
            f"{source}.ff.net.0.proj.weight": f"{target}.ff.proj.weight",
            f"{source}.ff.net.2.weight": f"{target}.ff.out.weight",
        }
    )
    return mapping


def group_mapping(part, config, layer=None, num_layers=None):
    """Checkpoint key -> module parameter name for one exported partition.

    `part` is `"embed"`, `"block"` (one block, or `num_layers` blocks starting at `layer`) or `"head"`.
    """
    if part == "embed":
        mapping = {
            "proj_in.weight": "proj_in.weight",
            "proj_in.bias": "proj_in.bias",
            "audio_proj_in.weight": "audio_proj_in.weight",
            "audio_proj_in.bias": "audio_proj_in.bias",
            "context_embedder.weight": "context_embedder.weight",
            "context_embedder.bias": "context_embedder.bias",
            "token_refiner.final_norm.weight": "final_norm",
        }
        for index in range(config["num_refiner_layers"]):
            mapping.update(
                _block_mapping(f"token_refiner.refiner_blocks.{index}", f"refiner_blocks.{index}")
            )
        return mapping
    if part == "block":
        if num_layers is None:
            return _block_mapping(f"transformer_blocks.{layer}", "")
        mapping = {}
        for offset in range(num_layers):
            mapping.update(_block_mapping(f"transformer_blocks.{layer + offset}", f"blocks.{offset}"))
        return mapping
    if part == "head":
        return {
            "norm_out.norm.weight": "norm_out",
            "proj_out.weight": "proj_out.weight",
            "proj_out.bias": "proj_out.bias",
            "audio_proj_out.weight": "audio_proj_out.weight",
            "audio_proj_out.bias": "audio_proj_out.bias",
        }
    raise ValueError(f"unknown partition {part!r}")


def load_group_state_dict(model_path, part, config, layer=None, num_layers=None, dtype=None):
    """The state dict of one exported partition, read straight from the shards."""
    mapping = group_mapping(part, config, layer=layer, num_layers=num_layers)
    tensors = read_tensors(model_path, list(mapping))
    state_dict = {}
    for key, name in mapping.items():
        # A single block maps onto the bare module, so its keys have no prefix to strip.
        target = name.lstrip(".") if name.startswith(".") else name
        tensor = tensors[key]
        state_dict[target] = tensor if dtype is None else tensor.to(dtype)
    return state_dict



def time_embedder_state_dict(model_path):
    """The timestep MLP and the final norm's modulation projection, i.e. everything the AdaLN table needs."""
    keys = [
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
        "norm_out.linear.weight",
        "norm_out.linear.bias",
    ]
    return read_tensors(model_path, keys)


def checkpoint_config(model_path):
    config = json.loads((Path(model_path) / "config.json").read_text())
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    return config

