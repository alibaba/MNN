#!/usr/bin/env python3
# Copyright @ 2026 Alibaba. All rights reserved.

import argparse
import contextlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "export"))
if os.path.isdir(EXPORT_DIR) and EXPORT_DIR not in sys.path:
    sys.path.insert(0, EXPORT_DIR)

try:
    from utils.hqq_quantizer import HQQQuantizer
except ImportError:
    HQQQuantizer = None


COMMON_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "W_pack",
    "c_attn",
    "c_proj",
    "wq",
    "wk",
    "wv",
    "wo",
    "w1",
    "w2",
    "w3",
)


@dataclass
class MNNQuantConfig:
    base_model: str
    quant_bit: int
    quant_block: int
    lm_quant_bit: int
    lm_quant_block: int
    symmetric: bool
    scale_bit: int
    quant_method: str
    target_modules: List[str]
    skipped_quant_modules: List[str]


def parse_csv(value: Optional[str]) -> List[str]:
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def get_mnn_block_size(in_features: int, quant_block: int) -> int:
    block_size = in_features if quant_block == 0 else quant_block
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("quant_block must be >= 0")
    while in_features % block_size != 0:
        block_size //= 2
        if block_size <= 0:
            return 1
    return block_size


def mnn_quantize_weight(
    weight: torch.Tensor,
    quant_bit: int,
    quant_block: int,
    symmetric: bool,
    scale_bit: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    if weight.dim() != 2:
        raise ValueError("Only 2D Linear weights are supported")
    if quant_bit not in (1, 2, 3, 4, 8):
        raise ValueError("quant_bit must be one of 1, 2, 3, 4, 8")
    if scale_bit not in (16, 32):
        raise ValueError("scale_bit must be 16 or 32")

    oc, ic = weight.shape
    block_size = get_mnn_block_size(ic, quant_block)
    block_num = ic // block_size
    work = weight.detach().float().reshape(oc, block_num, block_size)
    offset = 1 << (quant_bit - 1)
    clip_max = offset - 1
    eps = torch.finfo(torch.float32).eps

    if symmetric:
        clip_min = -clip_max
        abs_max = torch.amax(torch.abs(work), dim=-1, keepdim=True)
        scale = torch.clamp(abs_max / max(clip_max, 1), min=eps)
        qint = torch.round(work / scale).clamp(clip_min, clip_max).to(torch.int8)
        zero = None
    else:
        clip_min = -offset
        max_val = torch.amax(work, dim=-1, keepdim=True)
        min_val = torch.amin(work, dim=-1, keepdim=True)
        scale = torch.clamp((max_val - min_val) / (clip_max - clip_min), min=eps)
        qint = torch.round((work - min_val) / scale) + clip_min
        qint = qint.clamp(clip_min, clip_max).to(torch.int8)
        zero = min_val - scale * clip_min

    scale_dtype = torch.float16 if scale_bit == 16 else torch.float32
    return qint.reshape(oc, ic), scale.to(scale_dtype), None if zero is None else zero.to(scale_dtype), block_size


class MNNQuantLinear(nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero: Optional[torch.Tensor],
        block_size: int,
        bias: Optional[torch.Tensor],
        out_features: int,
        in_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.register_buffer("qweight", qweight.contiguous(), persistent=False)
        self.register_buffer("scale", scale.contiguous(), persistent=False)
        if zero is None:
            self.zero = None
        else:
            self.register_buffer("zero", zero.contiguous(), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quant_bit: int,
        quant_block: int,
        symmetric: bool,
        scale_bit: int,
    ) -> "MNNQuantLinear":
        qweight, scale, zero, block_size = mnn_quantize_weight(
            linear.weight.data,
            quant_bit=quant_bit,
            quant_block=quant_block,
            symmetric=symmetric,
            scale_bit=scale_bit,
        )
        bias = None if linear.bias is None else linear.bias.data
        return cls(qweight, scale, zero, block_size, bias, linear.out_features, linear.in_features)

    @property
    def weight(self) -> torch.Tensor:
        dtype = torch.float16 if self.scale.dtype == torch.float16 else torch.float32
        return self.dequantize_weight(dtype=dtype, device=self.qweight.device)

    def dequantize_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        block_num = self.in_features // self.block_size
        qweight = self.qweight.to(device=device, dtype=torch.float32).reshape(
            self.out_features, block_num, self.block_size
        )
        scale = self.scale.to(device=device, dtype=torch.float32)
        if self.zero is None:
            weight = qweight * scale
        else:
            zero = self.zero.to(device=device, dtype=torch.float32)
            weight = qweight * scale + zero
        return weight.reshape(self.out_features, self.in_features).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(dtype=x.dtype, device=x.device)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, block_size={self.block_size}"


def hqq_quantize_weight(
    weight: torch.Tensor,
    quant_bit: int,
    quant_block: int,
    symmetric: bool,
    scale_bit: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    if HQQQuantizer is None:
        raise ImportError("HQQQuantizer is unavailable. Run from the MNN repository or add export utils to PYTHONPATH.")
    if weight.dim() != 2:
        raise ValueError("Only 2D Linear weights are supported")
    if quant_bit not in (1, 2, 3, 4, 8):
        raise ValueError("quant_bit must be one of 1, 2, 3, 4, 8")
    if scale_bit not in (16, 32):
        raise ValueError("scale_bit must be 16 or 32")

    oc, ic = weight.shape
    block_size = get_mnn_block_size(ic, quant_block)
    block_num = ic // block_size
    quantizer = HQQQuantizer(
        weight.detach(),
        quant_bit,
        block_size,
        symmetric,
        weight.dtype,
        weight.device,
    )
    quantizer.quant()

    qweight = quantizer.W_q.to(torch.int8).reshape(oc, block_num, block_size)
    scale = quantizer.meta["scale"].reshape(oc, block_num, 1)
    zero = quantizer.meta.get("zero")
    scale_dtype = torch.float16 if scale_bit == 16 else torch.float32

    if symmetric:
        return qweight.reshape(oc, ic), scale.to(scale_dtype), None, block_size

    offset = 1 << (quant_bit - 1)
    zero = zero.reshape(oc, block_num, 1)
    mnn_zero = scale * offset - scale * zero
    return qweight.reshape(oc, ic), scale.to(scale_dtype), mnn_zero.to(scale_dtype), block_size


class HQQQuantLinear(nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero: Optional[torch.Tensor],
        block_size: int,
        quant_bit: int,
        symmetric: bool,
        bias: Optional[torch.Tensor],
        out_features: int,
        in_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.quant_bit = quant_bit
        self.symmetric = symmetric
        self.register_buffer("qweight", qweight.contiguous(), persistent=False)
        self.register_buffer("scale", scale.contiguous(), persistent=False)
        if zero is None:
            self.zero = None
        else:
            self.register_buffer("zero", zero.contiguous(), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quant_bit: int,
        quant_block: int,
        symmetric: bool,
        scale_bit: int,
    ) -> "HQQQuantLinear":
        qweight, scale, zero, block_size = hqq_quantize_weight(
            linear.weight.data,
            quant_bit=quant_bit,
            quant_block=quant_block,
            symmetric=symmetric,
            scale_bit=scale_bit,
        )
        bias = None if linear.bias is None else linear.bias.data
        return cls(
            qweight,
            scale,
            zero,
            block_size,
            quant_bit,
            symmetric,
            bias,
            linear.out_features,
            linear.in_features,
        )

    @property
    def weight(self) -> torch.Tensor:
        dtype = torch.float16 if self.scale.dtype == torch.float16 else torch.float32
        return self.dequantize_weight(dtype=dtype, device=self.qweight.device)

    def dequantize_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        block_num = self.in_features // self.block_size
        qweight = self.qweight.to(device=device, dtype=torch.float32).reshape(
            self.out_features, block_num, self.block_size
        )
        scale = self.scale.to(device=device, dtype=torch.float32)
        if self.symmetric:
            weight = qweight * scale
        else:
            offset = 1 << (self.quant_bit - 1)
            zero = self.zero.to(device=device, dtype=torch.float32)
            weight = (qweight - offset) * scale + zero
        return weight.reshape(self.out_features, self.in_features).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(dtype=x.dtype, device=x.device)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, quant_bit={self.quant_bit}, symmetric={self.symmetric}"
        )


def should_skip(name: str, patterns: Sequence[str]) -> bool:
    return any(pattern and pattern in name for pattern in patterns)


def is_lm_head(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf in ("lm_head", "embed_out", "output")


def find_lora_targets(model: nn.Module, target_modules: str, exclude: Sequence[str]) -> List[str]:
    if target_modules != "auto":
        return parse_csv(target_modules)

    names = set()
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if is_lm_head(full_name) or should_skip(full_name, exclude):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in COMMON_LORA_TARGETS:
            names.add(leaf)

    if not names:
        for full_name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not is_lm_head(full_name) and not should_skip(full_name, exclude):
                names.add(full_name.rsplit(".", 1)[-1])
    return sorted(names)


def quant_params_for_module(name: str, args: argparse.Namespace) -> Tuple[int, int]:
    if is_lm_head(name):
        return args.lm_quant_bit, args.lm_quant_block
    return args.quant_bit, args.quant_block


def quantize_model_linears(module: nn.Module, args: argparse.Namespace, prefix: str = "") -> int:
    count = 0
    skip = parse_csv(args.skip_quant_modules)
    quant_method = getattr(args, "quant_method", "mnn")
    quant_linear_cls = HQQQuantLinear if quant_method == "hqq" else MNNQuantLinear
    for child_name, child in list(module.named_children()):
        full_name = child_name if not prefix else f"{prefix}.{child_name}"

        base_layer = getattr(child, "base_layer", None)
        if isinstance(base_layer, nn.Linear):
            if should_skip(full_name, skip):
                continue
            quant_bit, quant_block = quant_params_for_module(full_name, args)
            child.base_layer = quant_linear_cls.from_linear(
                base_layer,
                quant_bit=quant_bit,
                quant_block=quant_block,
                symmetric=args.sym,
                scale_bit=args.scale_bit,
            )
            count += 1
            continue

        if isinstance(child, nn.Linear):
            if should_skip(full_name, skip):
                continue
            quant_bit, quant_block = quant_params_for_module(full_name, args)
            module._modules[child_name] = quant_linear_cls.from_linear(
                child,
                quant_bit=quant_bit,
                quant_block=quant_block,
                symmetric=args.sym,
                scale_bit=args.scale_bit,
            )
            count += 1
            continue

        count += quantize_model_linears(child, args, full_name)
    return count


class ListDataset:
    def __init__(self, rows: List[Dict]) -> None:
        self.rows = rows

    @property
    def column_names(self) -> List[str]:
        names = set()
        for row in self.rows:
            names.update(row.keys())
        return sorted(names)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        return self.rows[index]

    def select(self, indices) -> "ListDataset":
        return ListDataset([self.rows[i] for i in indices])

    def map(self, function, remove_columns=None, desc: Optional[str] = None) -> "ListDataset":
        return ListDataset([function(row) for row in self.rows])

    def filter(self, function, desc: Optional[str] = None) -> "ListDataset":
        return ListDataset([row for row in self.rows if function(row)])


def load_local_text_dataset(path: str) -> ListDataset:
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            for key in ("data", "train", "examples"):
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break
            if not rows:
                rows = [data]
        else:
            raise ValueError(f"Unsupported JSON dataset root type: {type(data).__name__}")
    elif ext in (".txt", ".text"):
        with open(path, "r", encoding="utf-8") as f:
            rows = [{"text": line.strip()} for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported dataset file extension: {ext}")
    return ListDataset(rows)


def load_text_dataset(path_or_name: str, split: str):
    if os.path.isfile(path_or_name):
        ext = os.path.splitext(path_or_name)[1].lower()
        if load_dataset is None:
            return load_local_text_dataset(path_or_name)
        if ext in (".json", ".jsonl"):
            return load_dataset("json", data_files=path_or_name, split="train")
        if ext in (".txt", ".text"):
            return load_dataset("text", data_files=path_or_name, split="train")
        raise ValueError(f"Unsupported dataset file extension: {ext}")
    if load_dataset is None:
        raise ImportError("Install datasets to load HuggingFace datasets by name.")
    return load_dataset(path_or_name, split=split)


def format_example(example: Dict, tokenizer, args: argparse.Namespace) -> str:
    messages = example.get(args.messages_field)
    if messages is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    text = example.get(args.text_field)
    if text is not None:
        return str(text)

    prompt = example.get(args.prompt_field)
    response = example.get(args.response_field)
    if prompt is not None and response is not None:
        return f"{prompt}{args.response_separator}{response}"

    fields = [args.text_field, args.messages_field, args.prompt_field, args.response_field]
    raise KeyError(f"Could not format sample. Expected one of these fields: {fields}")


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def format_full_and_prompt(example: Dict, tokenizer, args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    messages = example.get(args.messages_field)
    if messages is not None:
        full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
        if not args.train_on_inputs and messages and messages[-1].get("role") == "assistant":
            prompt_text = apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
            return full_text, prompt_text
        return full_text, None

    prompt = example.get(args.prompt_field)
    response = example.get(args.response_field)
    if prompt is not None and response is not None:
        prompt_text = f"{prompt}{args.response_separator}"
        return f"{prompt_text}{response}", None if args.train_on_inputs else prompt_text

    return format_example(example, tokenizer, args), None


def tokenize_dataset(dataset, tokenizer, args: argparse.Namespace):
    eos_id = tokenizer.eos_token_id

    def tokenize_one(example):
        text, prompt_text = format_full_and_prompt(example, tokenizer, args)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=True,
        )
        labels = list(tokenized["input_ids"])
        if prompt_text is not None:
            prompt_tokenized = tokenizer(
                prompt_text,
                truncation=True,
                max_length=args.max_seq_len,
                add_special_tokens=True,
            )
            prompt_len = min(len(prompt_tokenized["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
        if args.add_eos_token and eos_id is not None:
            if not tokenized["input_ids"] or tokenized["input_ids"][-1] != eos_id:
                if len(tokenized["input_ids"]) < args.max_seq_len:
                    tokenized["input_ids"].append(eos_id)
                    tokenized["attention_mask"].append(1)
                    labels.append(eos_id if prompt_text is None else eos_id)
        tokenized["labels"] = labels
        return tokenized

    columns = list(dataset.column_names)
    tokenized = dataset.map(tokenize_one, remove_columns=columns, desc="Tokenizing")
    tokenized = tokenized.filter(lambda item: len(item["input_ids"]) > 1, desc="Filtering empty samples")
    return tokenized


class CausalCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        labels = [feature.get("labels") for feature in features]
        model_features = [{key: value for key, value in feature.items() if key != "labels"} for feature in features]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if any(label is not None for label in labels):
            label_tensor = torch.full_like(batch["input_ids"], -100)
            for i, label in enumerate(labels):
                if label is None:
                    length = int(batch["attention_mask"][i].sum().item())
                    label_tensor[i, :length] = batch["input_ids"][i, :length]
                    continue
                length = min(len(label), label_tensor.shape[1])
                label_tensor[i, :length] = torch.tensor(label[:length], dtype=label_tensor.dtype)
            label_tensor[batch["attention_mask"] == 0] = -100
            batch["labels"] = label_tensor
        else:
            label_tensor = batch["input_ids"].clone()
            label_tensor[batch["attention_mask"] == 0] = -100
            batch["labels"] = label_tensor
        return batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def trainable_parameter_count(model: nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            loss = model(**batch).loss
            losses.append(loss.detach().float().cpu())
    model.train()
    if not losses:
        return float("nan")
    return torch.stack(losses).mean().item()


def write_quant_config(
    output_dir: str,
    args: argparse.Namespace,
    target_modules: List[str],
) -> None:
    config = MNNQuantConfig(
        base_model=args.base_model,
        quant_bit=args.quant_bit,
        quant_block=args.quant_block,
        lm_quant_bit=args.lm_quant_bit,
        lm_quant_block=args.lm_quant_block,
        symmetric=args.sym,
        scale_bit=args.scale_bit,
        quant_method=f"{args.quant_method}_weight_only_fake_quant" if args.fake_quant else "none",
        target_modules=target_modules,
        skipped_quant_modules=parse_csv(args.skip_quant_modules),
    )
    with open(os.path.join(output_dir, "mnn_quant_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def save_adapter(model, tokenizer, output_dir: str, args: argparse.Namespace, target_modules: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_quant_config(output_dir, args, target_modules)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter against MNN weight-only quantized base weights."
    )
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace model id or local model path.")
    parser.add_argument("--train_data", type=str, required=True, help="JSON/JSONL/TXT file or HF dataset name.")
    parser.add_argument(
        "--validation_data", type=str, default=None, help="Optional JSON/JSONL/TXT file or HF dataset name."
    )
    parser.add_argument("--dataset_split", type=str, default="train", help="Split used for HF train datasets.")
    parser.add_argument(
        "--validation_split", type=str, default="validation", help="Split used for HF validation datasets."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the PEFT adapter.")

    parser.add_argument("--text_field", type=str, default="text", help="Dataset text field.")
    parser.add_argument("--messages_field", type=str, default="messages", help="Chat messages field.")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="Prompt field for prompt/response data.")
    parser.add_argument(
        "--response_field", type=str, default="response", help="Response field for prompt/response data."
    )
    parser.add_argument(
        "--response_separator", type=str, default="", help="Separator inserted between prompt and response."
    )
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--add_eos_token", action="store_true", help="Append eos_token when room is available.")
    parser.add_argument("--train_on_inputs", action="store_true", help="Do not mask prompt tokens in the labels.")

    parser.add_argument("--quant_bit", type=int, default=4, help="MNN base quant bits, matching llmexport --quant_bit.")
    parser.add_argument(
        "--quant_block", type=int, default=64, help="MNN base quant block, matching llmexport --quant_block."
    )
    parser.add_argument("--lm_quant_bit", type=int, default=None, help="lm_head quant bits, default follows quant_bit.")
    parser.add_argument(
        "--lm_quant_block", type=int, default=None, help="lm_head quant block, default follows quant_block."
    )
    parser.add_argument("--sym", action="store_true", help="Use MNN symmetric weight quantization.")
    parser.add_argument("--scale_bit", type=int, default=16, choices=[16, 32], help="MNN scale storage bit width.")
    parser.add_argument(
        "--quant_method",
        type=str,
        default="mnn",
        choices=["mnn", "hqq"],
        help="Fake-quant algorithm for the frozen base weights. Use hqq to match llmexport --hqq.",
    )
    parser.add_argument("--hqq", action="store_true", help="Alias for --quant_method hqq.")
    parser.add_argument("--skip_quant_modules", type=str, default="", help="Module-name substrings left unquantized.")
    parser.add_argument(
        "--no_fake_quant",
        action="store_true",
        help="Train a standard LoRA adapter on the floating-point base model instead of MNN fake-quant weights.",
    )

    parser.add_argument("--target_modules", type=str, default="auto", help="Comma-separated LoRA target leaf names.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"], help="PEFT bias mode."
    )

    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override total training steps when > 0.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Train batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Eval batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Transformers scheduler name.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log interval.")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint interval. <=0 disables checkpoints.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional eval cap.")

    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument(
        "--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Model compute dtype."
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Forwarded to from_pretrained.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


def main() -> None:
    args = build_args()
    if args.hqq:
        args.quant_method = "hqq"
    args.fake_quant = not args.no_fake_quant
    if args.lm_quant_bit is None:
        args.lm_quant_bit = args.quant_bit
    if args.lm_quant_block is None:
        args.lm_quant_block = args.quant_block

    set_random_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    skip_quant = parse_csv(args.skip_quant_modules)
    target_modules = find_lora_targets(model, args.target_modules, skip_quant)
    if not target_modules:
        raise RuntimeError("No LoRA target modules found. Pass --target_modules explicitly.")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if args.fake_quant and args.quant_method == "hqq" and device.type != "cpu":
        model.to(device)
    quantized = quantize_model_linears(model, args) if args.fake_quant else 0
    model.to(device)
    trainable, total = trainable_parameter_count(model)
    print(f"LoRA target modules: {target_modules}")
    print(f"Fake quant training: {args.fake_quant}")
    print(f"Quantized Linear modules: {quantized}")
    print(f"Trainable parameters: {trainable} / {total} ({100.0 * trainable / max(total, 1):.4f}%)")

    train_dataset = load_text_dataset(args.train_data, args.dataset_split)
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args)

    eval_loader = None
    if args.validation_data is not None:
        eval_dataset = load_text_dataset(args.validation_data, args.validation_split)
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer, args)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=CausalCollator(tokenizer),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CausalCollator(tokenizer),
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training dataset is empty after tokenization/filtering.")

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.num_train_epochs * steps_per_epoch)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    progress = tqdm(total=total_steps, desc="Training")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    accumulated_loss = 0.0
    def autocast_context():
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type=device.type, dtype=dtype)
        return contextlib.nullcontext()

    epoch = 0
    micro_steps_since_update = 0
    while global_step < total_steps:
        epoch += 1
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context():
                raw_loss = model(**batch).loss
                loss = raw_loss / args.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += raw_loss.detach().float().item() / args.gradient_accumulation_steps
            micro_steps_since_update += 1

            should_update = step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            if not should_update:
                continue

            if micro_steps_since_update != args.gradient_accumulation_steps:
                grad_scale = args.gradient_accumulation_steps / micro_steps_since_update
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(grad_scale)

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    args.max_grad_norm,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro_steps_since_update = 0

            global_step += 1
            progress.update(1)
            if global_step % args.logging_steps == 0:
                avg_loss = accumulated_loss / args.logging_steps
                accumulated_loss = 0.0
                message = f"step={global_step} loss={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.6e}"
                if eval_loader is not None:
                    eval_loss = evaluate(model, eval_loader, device)
                    message += f" eval_loss={eval_loss:.6f}"
                print(message)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_adapter(model, tokenizer, save_dir, args, target_modules)

            if global_step >= total_steps:
                break

        if args.max_steps <= 0 and epoch >= math.ceil(args.num_train_epochs):
            break

    progress.close()
    save_adapter(model, tokenizer, args.output_dir, args, target_modules)
    if not args.fake_quant:
        adapter_type = "standard LoRA"
    elif args.quant_method == "hqq":
        adapter_type = "HQQ-aware LoRA"
    else:
        adapter_type = "MNN-aware LoRA"
    print(f"Saved {adapter_type} adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
