#!/usr/bin/env python3
# Copyright @ 2026 Alibaba. All rights reserved.

import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mnn_qlora import pick_device, pick_dtype, quantize_model_linears


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def messages_to_prompt(tokenizer, messages: List[Dict]) -> Tuple[str, str]:
    prompt_messages = list(messages)
    expected = ""
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        expected = str(prompt_messages[-1].get("content", ""))
        prompt_messages = prompt_messages[:-1]
    try:
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    return prompt, expected


def extract_prompt_expected(row: Dict, tokenizer, args: argparse.Namespace) -> Tuple[str, str]:
    messages = row.get(args.messages_field)
    if messages is not None:
        return messages_to_prompt(tokenizer, messages)
    prompt = str(row.get(args.prompt_field, row.get(args.text_field, "")))
    expected = str(row.get(args.response_field, ""))
    return prompt, expected


def load_quant_config(adapter_path: Optional[str], args: argparse.Namespace) -> SimpleNamespace:
    data = {}
    if adapter_path:
        config_path = os.path.join(adapter_path, "mnn_quant_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    quant_method = args.quant_method
    if args.hqq:
        quant_method = "hqq"
    if quant_method == "auto":
        saved_method = data.get("quant_method", "")
        quant_method = "hqq" if "hqq" in saved_method else "mnn"
    return SimpleNamespace(
        quant_bit=args.quant_bit if args.quant_bit is not None else data.get("quant_bit", 4),
        quant_block=args.quant_block if args.quant_block is not None else data.get("quant_block", 64),
        lm_quant_bit=args.lm_quant_bit if args.lm_quant_bit is not None else data.get("lm_quant_bit", 4),
        lm_quant_block=args.lm_quant_block if args.lm_quant_block is not None else data.get("lm_quant_block", 64),
        sym=args.sym if args.sym else data.get("symmetric", False),
        scale_bit=args.scale_bit if args.scale_bit is not None else data.get("scale_bit", 16),
        quant_method=quant_method,
        skip_quant_modules=args.skip_quant_modules or ",".join(data.get("skipped_quant_modules", [])),
    )


def build_model(args: argparse.Namespace, tokenizer):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    if args.fake_quant:
        quant_args = load_quant_config(args.adapter_path, args)
        if quant_args.quant_method == "hqq" and device.type != "cpu":
            model.to(device)
        quantized = quantize_model_linears(model, quant_args)
        print(f"Quantized Linear modules: {quantized}")
    model.to(device)
    model.eval()
    return model, device


def generate_one(model, tokenizer, device, prompt: str, args: argparse.Namespace) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a LoRA adapter changes expected generations.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--messages_field", type=str, default="messages")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--prompt_field", type=str, default="prompt")
    parser.add_argument("--response_field", type=str, default="response")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--fake_quant", action="store_true")
    parser.add_argument("--quant_bit", type=int, default=None)
    parser.add_argument("--quant_block", type=int, default=None)
    parser.add_argument("--lm_quant_bit", type=int, default=None)
    parser.add_argument("--lm_quant_block", type=int, default=None)
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--scale_bit", type=int, default=None)
    parser.add_argument("--quant_method", type=str, default="auto", choices=["auto", "mnn", "hqq"])
    parser.add_argument("--hqq", action="store_true", help="Alias for --quant_method hqq.")
    parser.add_argument("--skip_quant_modules", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, device = build_model(args, tokenizer)
    rows = read_jsonl(args.eval_data)
    if args.limit is not None:
        rows = rows[:args.limit]

    hits = 0
    for index, row in enumerate(rows):
        prompt, expected = extract_prompt_expected(row, tokenizer, args)
        generated = generate_one(model, tokenizer, device, prompt, args)
        ok = expected != "" and expected in generated
        hits += int(ok)
        print(json.dumps({
            "index": index,
            "ok": ok,
            "expected": expected,
            "generated": generated,
        }, ensure_ascii=False))
    total = max(len(rows), 1)
    print(f"accuracy={hits}/{len(rows)} ({100.0 * hits / total:.2f}%)")


if __name__ == "__main__":
    main()
