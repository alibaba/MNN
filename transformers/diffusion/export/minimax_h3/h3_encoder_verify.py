# SPDX-License-Identifier: Apache-2.0
"""Verify the MiniMax-H3 text conditioner: the export-friendly stack, and the MNN modules built from it.

Two checks, because they fail for different reasons:

1. **Plumbing** -- a truncated reference `Qwen3VLTextModel` is run against the export-friendly stack on the real
   prompt. This exercises the token embedding, Qwen3-VL's interleaved mrope, the causal mask and the layer math
   through the model's own code path, so a wrong rotary layout or a mis-set mask shows up here.
2. **The exported modules** -- the full 50-layer export-friendly stack against whatever the MNN conditioner
   wrote, which is what the C++ demo actually runs.

    python h3_encoder_verify.py --text_encoder /path/to/text_encoder --resources /path/to/h3_mnn \\
        --prompt "..." --layers 10
    python h3_encoder_verify.py ... --mnn_output /path/to/hidden.bin      # check 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from h3_encoder_export import QwenLayerGroup, capture_rope, layer_mapping, read_tensors, text_config  # noqa: E402
from h3_fixture import compare, format_report  # noqa: E402


def load_embeddings(resources, token_ids, hidden_size):
    """Gather the rows the prompt uses out of the exported bfloat16 table, the way the runtime does."""
    path = Path(resources) / "h3_text_embed.bin"
    rows = np.empty((len(token_ids), hidden_size), dtype=np.float32)
    with path.open("rb") as handle:
        for index, token in enumerate(token_ids):
            handle.seek(token * hidden_size * 2)
            raw = np.frombuffer(handle.read(hidden_size * 2), dtype=np.uint16)
            widened = np.zeros(hidden_size, dtype=np.uint32)
            widened[:] = raw.astype(np.uint32) << 16
            rows[index] = widened.view(np.float32)
    return torch.from_numpy(rows)[None]


def run_export_stack(text_encoder, config, hidden, cos, sin, mask, num_layers, layers_per_group):
    """The export-friendly stack, one partition at a time so peak host memory is one partition."""
    for start in range(0, num_layers, layers_per_group):
        count = min(layers_per_group, num_layers - start)
        group = QwenLayerGroup(config, count)
        mapping = layer_mapping(start, count)
        tensors = read_tensors(text_encoder, list(mapping), torch.float32)
        group.load_state_dict({name: tensors[key] for key, name in mapping.items()}, strict=True)
        with torch.no_grad():
            hidden = group.eval()(hidden, cos, sin, mask)
        del group, tensors
    return hidden


def main():
    parser = argparse.ArgumentParser(description="Verify the MiniMax-H3 text conditioner.")
    parser.add_argument("--text_encoder", required=True)
    parser.add_argument("--resources", required=True, help="Directory holding h3_text_* built resources.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--layers",
        type=int,
        default=10,
        help="Layers for the plumbing check against the reference model. The full 50 would need ~100 GB.",
    )
    parser.add_argument("--mnn_output", help="Hidden states the MNN conditioner wrote, to check against.")
    parser.add_argument("--skip_reference", action="store_true")
    parser.add_argument(
        "--cos_threshold",
        type=float,
        default=0.999,
        help="A 50-layer W4 stack lands near 0.9999; the reference-vs-export check is exact.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    manifest = json.loads((Path(args.resources) / "h3_text_manifest.json").read_text())
    config = text_config(args.text_encoder)
    hidden_size = manifest["hidden_size"]
    max_tokens = manifest["max_tokens"]

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    token_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
    num_tokens = len(token_ids)
    print(f"prompt: {num_tokens} token(s), conditioner exported for {max_tokens}")
    if num_tokens > max_tokens:
        raise SystemExit(f"the prompt is {num_tokens} tokens but the conditioner takes {max_tokens}")

    embeds = load_embeddings(args.resources, token_ids, hidden_size)
    cos_full, sin_full = capture_rope(args.text_encoder, config, max_tokens)
    cos = cos_full[:, :num_tokens].float()
    sin = sin_full[:, :num_tokens].float()
    mask = torch.triu(torch.full((1, 1, num_tokens, num_tokens), float("-inf")), diagonal=1)

    rows = []

    if not args.skip_reference:
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        # A truncated reference model: enough to exercise its own embedding, rotary and mask code, without
        # materializing 25B parameters.
        reference_config = AutoConfig.from_pretrained(args.text_encoder).text_config
        # One layer more than is read: HF appends the *post-norm* state as the last `hidden_states` entry, so a
        # stack truncated to exactly `args.layers` would hand back norm(after layer N) rather than the pre-norm
        # state MiniMax-H3 conditions on. This is the trap the reference implementation documents.
        reference_config.num_hidden_layers = args.layers + 1
        reference_config._attn_implementation = "eager"
        with torch.device("meta"):
            model = Qwen3VLTextModel(reference_config)
        keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight"]
        for start in range(args.layers + 1):
            keys.extend(layer_mapping(start, 1))
        tensors = read_tensors(args.text_encoder, keys, torch.float32)
        prefix = "model.language_model."
        model.load_state_dict(
            {key[len(prefix) :]: tensors[key] for key in keys}, assign=True, strict=True
        )
        # `rotary_emb.inv_freq` is computed rather than loaded, so it is still a meta tensor here; rebuild it or
        # every hidden state downstream is garbage.
        model.rotary_emb = type(model.rotary_emb)(reference_config)
        model = model.eval()

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_hidden_states=True,
            )
        # (embeddings, after layer 1, ..., after layer N-1, norm(after layer N)). Reading index `args.layers`
        # off a stack of `args.layers + 1` therefore lands on a pre-norm state.
        if len(outputs.hidden_states) != args.layers + 2:
            raise SystemExit(
                f"expected {args.layers + 2} hidden states from a {args.layers + 1}-layer reference, got "
                f"{len(outputs.hidden_states)}"
            )
        reference_hidden = outputs.hidden_states[args.layers]
        rows.append(compare("reference embeddings", embeds.numpy(), outputs.hidden_states[0].numpy()))
        del model, tensors

        mine = run_export_stack(
            args.text_encoder, config, embeds, cos[:, :, None, :], sin[:, :, None, :], mask, args.layers,
            manifest["layers_per_group"],
        )
        rows.append(compare(f"export stack[{args.layers}] vs reference", mine.numpy(), reference_hidden.numpy()))

    if args.mnn_output:
        full = run_export_stack(
            args.text_encoder, config, embeds, cos[:, :, None, :], sin[:, :, None, :], mask,
            manifest["num_layers"], manifest["layers_per_group"],
        )
        got = np.fromfile(args.mnn_output, dtype=np.float32).reshape(1, num_tokens, hidden_size)
        rows.append(
            compare(
                f"MNN conditioner vs export stack[{manifest['num_layers']}]", got, full.numpy(),
                cos_threshold=args.cos_threshold,
            )
        )
        print(
            f"conditioning rms: MNN {np.sqrt((got.astype(np.float64) ** 2).mean()):.4f}, "
            f"reference {float((full.double() ** 2).mean().sqrt()):.4f}"
        )

    print()
    print(format_report(rows))
    failed = [row["name"] for row in rows if not row["ok"]]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nthe conditioner matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
