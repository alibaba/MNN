#!/usr/bin/env python3
"""TTFA benchmark: compare serial vs interleaved Omni generation over N runs.

Usage:
    python bench_ttfa.py --demo ./build/llm_demo \
                         --config ~/models/qwen2.5/config-inter.json \
                         --prompt prompt.txt \
                         --runs 10
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time


def parse_dump(output: str) -> dict:
    """Parse DUMP_TALKER_PERFORMANCE blocks from llm_demo stdout."""
    metrics = {}

    # --- Interleaved block ---
    m = re.search(r'\[interleaved mode\].*?'
                  r'thinker prefill = ([\d.]+) s.*?'
                  r'thinker decode = ([\d.]+) s.*?'
                  r'talker prefill = ([\d.]+) s.*?'
                  r'talker decode = ([\d.]+) s.*?'
                  r'ttfa \(total\) = ([\d.]+) s.*?'
                  r'token2wav\s+ = ([\d.]+) s.*?'
                  r'tts rtf\s+ = ([\d.]+)',
                  output, re.DOTALL)
    if m:
        metrics['mode'] = 'interleaved'
        metrics['thinker_prefill'] = float(m.group(1))
        metrics['thinker_decode'] = float(m.group(2))
        metrics['talker_prefill'] = float(m.group(3))
        metrics['talker_decode'] = float(m.group(4))
        metrics['ttfa'] = float(m.group(5))
        metrics['token2wav'] = float(m.group(6))
        metrics['tts_rtf'] = float(m.group(7))
        return metrics

    # --- Serial block ---
    m = re.search(r'prompt tokens num.*?'
                  r'prefill time = ([\d.]+) s.*?'
                  r'decode time = ([\d.]+) s.*?'
                  r'ttfa time = ([\d.]+) s.*?'
                  r'token2wav time = ([\d.]+) s.*?'
                  r'tts rtf\s+ = ([\d.]+)',
                  output, re.DOTALL)
    if m:
        metrics['mode'] = 'serial'
        metrics['talker_prefill'] = float(m.group(1))
        metrics['talker_decode'] = float(m.group(2))
        metrics['ttfa'] = float(m.group(3))
        metrics['token2wav'] = float(m.group(4))
        metrics['tts_rtf'] = float(m.group(5))
        # Thinker stats come from the second (thinker) DUMP block
        m2 = re.search(r'prefill time = ([\d.]+) s\n decode time = ([\d.]+) s\n sample time',
                       output)
        if m2:
            metrics['thinker_prefill'] = float(m2.group(1))
            metrics['thinker_decode'] = float(m2.group(2))
        return metrics

    return None


def make_config(base_path: str, interleaved: bool) -> str:
    with open(base_path) as f:
        config = json.load(f)
    config['interleaved'] = interleaved
    dirname = os.path.dirname(os.path.abspath(base_path))
    suffix = '_interleaved.json' if interleaved else '_serial.json'
    path = os.path.join(dirname, '.bench_ttfa' + suffix)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    return path


def run_once(demo: str, config_path: str, prompt: str, timeout: int = 300) -> tuple[dict, float]:
    """Run llm_demo once, return (parsed_metrics, wall_seconds)."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [demo, config_path, prompt],
            capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0
    wall = time.perf_counter() - t0

    combined = result.stdout + result.stderr
    metrics = parse_dump(combined)
    if metrics is None:
        # Print raw output for debugging on first failure
        print(f"  [WARN] Failed to parse output. Raw dump:\n{combined[:2000]}", file=sys.stderr)
    return metrics, wall


def summarize(name: str, results: list[dict], walls: list[float]):
    """Print summary statistics for one mode."""
    if not results:
        print(f"\n{name}: NO VALID RUNS")
        return

    def stats(vals, fmt='.2f'):
        vals = [v for v in vals if v is not None]
        if not vals:
            return '-'
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        return f'{m:{fmt}} ± {s:{fmt}}'

    ttfa = [r.get('ttfa') for r in results]
    t2w = [r.get('token2wav') for r in results]
    t_prefill = [r.get('thinker_prefill') for r in results]
    t_decode = [r.get('thinker_decode') for r in results]
    tk_prefill = [r.get('talker_prefill') for r in results]
    tk_decode = [r.get('talker_decode') for r in results]

    print(f"\n{'='*60}")
    print(f"  {name}  (n={len(results)})")
    print(f"{'='*60}")
    print(f"  TTFA (total)        {stats(ttfa)} s")
    print(f"  Thinker prefill     {stats(t_prefill)} s")
    print(f"  Thinker decode      {stats(t_decode)} s")
    print(f"  Talker prefill      {stats(tk_prefill)} s")
    print(f"  Talker decode       {stats(tk_decode)} s")
    print(f"  Token2wav           {stats(t2w)} s")
    if walls:
        w = statistics.mean(walls)
        ws = statistics.stdev(walls) if len(walls) > 1 else 0
        print(f"  Wall time           {w:.2f} ± {ws:.2f} s")


def main():
    parser = argparse.ArgumentParser(description='Benchmark TTFA: serial vs interleaved')
    parser.add_argument('--demo', required=True, help='Path to llm_demo binary')
    parser.add_argument('--config', required=True, help='Base config.json (interleaved field will be overridden)')
    parser.add_argument('--prompt', required=True, help='Path to prompt file')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per mode (default: 10)')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup runs per mode (default: 1)')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per run in seconds (default: 300)')
    args = parser.parse_args()

    for path in [args.demo, args.config, args.prompt]:
        if not os.path.exists(path):
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    config_serial = make_config(args.config, interleaved=False)
    config_interleaved = make_config(args.config, interleaved=True)

    def run_mode(label, config_path):
        print(f"\n--- {label} ---")
        # Warmup
        for i in range(args.warmup):
            print(f"  warmup {i+1}/{args.warmup}...", end=' ', flush=True)
            m, w = run_once(args.demo, config_path, args.prompt, args.timeout)
            print(f"wall={w:.1f}s" if m else "FAILED")
        # Measurement
        metrics_list = []
        walls = []
        for i in range(args.runs):
            print(f"  run {i+1}/{args.runs}...", end=' ', flush=True)
            m, w = run_once(args.demo, config_path, args.prompt, args.timeout)
            if m:
                metrics_list.append(m)
                walls.append(w)
                print(f"TTFA={m['ttfa']:.2f}s wall={w:.1f}s")
            else:
                print("FAILED (skipped)")
        return metrics_list, walls

    try:
        serial_m, serial_w = run_mode("SERIAL  (interleaved=false)", config_serial)
        inter_m, inter_w = run_mode("INTERLEAVED (interleaved=true)", config_interleaved)

        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        summarize("SERIAL", serial_m, serial_w)
        summarize("INTERLEAVED", inter_m, inter_w)

        if serial_m and inter_m:
            serial_ttfa = statistics.mean([r['ttfa'] for r in serial_m])
            inter_ttfa = statistics.mean([r['ttfa'] for r in inter_m])
            speedup = serial_ttfa / inter_ttfa if inter_ttfa > 0 else float('inf')
            print(f"\n  TTFA speedup: {speedup:.1f}x  ({serial_ttfa:.2f}s → {inter_ttfa:.2f}s)")
    finally:
        os.unlink(config_serial)
        os.unlink(config_interleaved)


if __name__ == '__main__':
    main()
