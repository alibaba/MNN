#!/usr/bin/env python3
"""
Analyse Metal counter-profiler timeline dumps (from MNN_METAL_OP_PROFILE_TIMELINE)
and print gantt-style diagnostics: total GPU busy/idle, gap distribution,
top gap-transition types, top-N single gaps.

Usage:
    export MNN_METAL_OP_PROFILE_TIMELINE=/tmp/decode.csv
    ./llm_demo config.json prompt.txt 100         # requires build with -DMNN_METAL_OP_PROFILE=ON
    tools/script/metal_profile_gantt.py /tmp/decode.csv

CAUTION (learnt on 2026-07-23, see skills/metal-optimize/env-registry.md):
    Numbers from MNN_METAL_OP_PROFILE=ON include a per-op sample-buffer
    attachment overhead that inflates CPU encode from ~0.92us/op to ~4-20us/op.
    That overhead **manufactures GPU idle between ops** — the idle you see
    under profile ON is largely a measurement artifact, not a production
    optimization target. Any optimization based on this data MUST be
    cross-validated with a production build (profile OFF) 3-rep alternating
    A/B before drawing conclusions.
"""
import argparse
import csv
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["start_ns"]), int(r["end_ns"]), r["name"]))
    rows.sort(key=lambda r: r[0])
    return rows


def summarise(rows):
    span_ns = rows[-1][1] - rows[0][0]
    busy_ns = sum(r[1] - r[0] for r in rows)
    idle_ns = span_ns - busy_ns
    return span_ns, busy_ns, idle_ns


def gaps(rows):
    return [
        (rows[i][0] - rows[i - 1][1], rows[i - 1][2], rows[i][2])
        for i in range(1, len(rows))
    ]


def bucket(gs):
    buckets = [1000, 5000, 10000, 50000, 100000, 500000, 10_000_000]
    labels = [
        "<1us",
        "1-5us",
        "5-10us",
        "10-50us",
        "50-100us",
        "100-500us",
        "500us-10ms",
        ">10ms",
    ]
    counts = [0] * (len(buckets) + 1)
    totals = [0] * (len(buckets) + 1)
    for g, _, _ in gs:
        placed = False
        for i, b in enumerate(buckets):
            if g < b:
                counts[i] += 1
                totals[i] += max(0, g)
                placed = True
                break
        if not placed:
            counts[-1] += 1
            totals[-1] += g
    return labels, counts, totals


def top_transitions(gs, n, threshold_ns):
    agg = defaultdict(lambda: [0, 0.0])
    for g, prev, nxt in gs:
        if g <= threshold_ns:
            continue
        key = f"{prev} -> {nxt}"
        agg[key][0] += 1
        agg[key][1] += g
    return sorted(agg.items(), key=lambda x: -x[1][1])[:n]


def top_single_gaps(gs, n):
    return sorted(gs, key=lambda x: -x[0])[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Path to timeline CSV produced by MNN_METAL_OP_PROFILE_TIMELINE")
    ap.add_argument("--top-transitions", type=int, default=15,
                    help="Show top-N gap-transition types (aggregated by prev->next op name)")
    ap.add_argument("--top-single", type=int, default=10,
                    help="Show top-N largest single gaps (raw)")
    ap.add_argument("--transition-threshold-us", type=float, default=1.0,
                    help="Ignore gaps smaller than this when aggregating transitions")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("No samples in CSV.", file=sys.stderr)
        return 1

    span_ns, busy_ns, idle_ns = summarise(rows)
    idle_pct = 100.0 * idle_ns / span_ns if span_ns > 0 else 0.0
    print(f"Samples: {len(rows)}")
    print(f"Timeline span:  {span_ns/1e6:8.2f} ms")
    print(f"GPU busy:       {busy_ns/1e6:8.2f} ms")
    print(f"GPU idle:       {idle_ns/1e6:8.2f} ms  ({idle_pct:.1f}%)")

    gs = gaps(rows)
    labels, counts, totals = bucket(gs)
    print("\nGap size distribution:")
    for lbl, c, t in zip(labels, counts, totals):
        print(f"  {lbl:12s} {c:6d} gaps   {t/1e6:7.2f} ms cumulative")

    threshold = int(args.transition_threshold_us * 1000)
    print(f"\nTop {args.top_transitions} gap transitions (aggregate, >= {args.transition_threshold_us}us):")
    for k, (c, s) in top_transitions(gs, args.top_transitions, threshold):
        print(f"  {c:6d}x  {s/1e6:7.2f} ms   {k}")

    print(f"\nTop {args.top_single} single gaps:")
    for g, prev, nxt in top_single_gaps(gs, args.top_single):
        if g <= 0:
            continue
        print(f"  {g/1000:9.2f} us   {prev[:35]:35s} -> {nxt[:35]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
