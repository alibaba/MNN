#!/usr/bin/env python3
"""Verify concurrent and alternating inference with two split MNN LoRA models."""

import argparse
import concurrent.futures
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import MNN
import MNN.llm as mnnllm


DEFAULT_PROMPT = "适配器切换测试：请只输出当前适配器口令。"


@dataclass
class RunResult:
    name: str
    expected: str
    generated: str
    status: str
    success: bool


def run_adapter(name, model, prompt, expected, other_expected, start_barrier=None):
    model.reset()
    if start_barrier is not None:
        start_barrier.wait()
    generated = model.response(prompt, False)
    status = model.context.status
    success = (
        status != mnnllm.LlmStatus.INTERNAL_ERROR
        and expected in generated
        and other_expected not in generated
    )
    return RunResult(name, expected, generated, str(status), success)


def print_result(phase, result):
    print(f"[{phase}] {result.name}: {'PASS' if result.success else 'FAIL'}")
    print(f"  status: {result.status}")
    print(f"  expected: {result.expected}")
    print(f"  generated: {result.generated}")


def build_args():
    parser = argparse.ArgumentParser(
        description="Test two split LoRA models loaded from one MNN LLM base."
    )
    parser.add_argument("config", type=Path, help="Path to the base config.json.")
    parser.add_argument("--lora-a", default="lora_alpha.mnn")
    parser.add_argument("--expected-a", default="<<ALPHA>>")
    parser.add_argument("--lora-b", default="lora_beta.mnn")
    parser.add_argument("--expected-b", default="[[BETA]]")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--rounds", type=int, default=10)
    return parser.parse_args()


def main():
    args = build_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be greater than zero")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    print(f"MNN module: {MNN.__file__}")
    print(f"Config: {args.config.resolve()}")

    base = mnnllm.create(str(args.config.resolve()))
    base.load()
    base.set_config(
        {
            "async": False,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "max_new_tokens": 16,
        }
    )
    adapter_a = base.create_lora(args.lora_a)
    adapter_b = base.create_lora(args.lora_b)
    adapter_a.set_config({"async": False, "max_new_tokens": 16})
    adapter_b.set_config({"async": False, "max_new_tokens": 16})

    all_passed = True
    start_barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            run_adapter,
            "adapter A",
            adapter_a,
            args.prompt,
            args.expected_a,
            args.expected_b,
            start_barrier,
        )
        future_b = executor.submit(
            run_adapter,
            "adapter B",
            adapter_b,
            args.prompt,
            args.expected_b,
            args.expected_a,
            start_barrier,
        )
        parallel_a = future_a.result()
        parallel_b = future_b.result()

    print_result("parallel", parallel_a)
    print_result("parallel", parallel_b)
    all_passed = parallel_a.success and parallel_b.success

    switch_passed = 0
    for round_index in range(args.rounds):
        switched_a = run_adapter(
            "adapter A", adapter_a, args.prompt, args.expected_a, args.expected_b
        )
        switched_b = run_adapter(
            "adapter B", adapter_b, args.prompt, args.expected_b, args.expected_a
        )
        phase = f"switch {round_index + 1}"
        print_result(phase, switched_a)
        print_result(phase, switched_b)
        switch_passed += int(switched_a.success) + int(switched_b.success)
        all_passed = all_passed and switched_a.success and switched_b.success

    summary = {
        "parallel_passed": int(parallel_a.success) + int(parallel_b.success),
        "parallel_total": 2,
        "switch_passed": switch_passed,
        "switch_total": args.rounds * 2,
        "rounds": args.rounds,
    }
    print(json.dumps(summary, ensure_ascii=False))
    print("PYTHON_MULTI_LORA_TEST_PASS" if all_passed else "PYTHON_MULTI_LORA_TEST_FAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
