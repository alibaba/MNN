"""Compare complete-layer Core ML prefill paths with native Metal in separate processes."""
import argparse
import array
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

MODES = ("metal", "cpu-stream", "gpu-stream", "fixed")
SWITCHES = ("MNN_PREFILL_BENCH_EXPORT", "MNN_PREFILL_BENCH_FIXED", "MNN_PREFILL_BENCH_GPU")


def invoke(binary, config, args, log, env):
    with log.open("w") as output:
        result = subprocess.run([binary, config, *args], stdout=output, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"{binary.name} exited with {result.returncode}; see {log}")
    return log.read_text().splitlines()


def read_logits(path):
    values = array.array("f")
    values.frombytes(path.read_bytes())
    if not values or not all(map(math.isfinite, values)):
        raise ValueError(f"Invalid logits in {path}")
    return values


def compare_logits(actual, reference):
    if len(actual) != len(reference):
        raise ValueError("Vocabulary size differs")
    top = lambda v: max(range(len(v)), key=v.__getitem__)
    energy = sum(x * x for x in reference)
    return {"top1_match": top(actual) == top(reference),
            "relative_l2": math.sqrt(sum((x - y)**2 for x, y in zip(actual, reference)) / max(energy, 1e-30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="MNN LLM config for the original Q4/G64 weights")
    parser.add_argument("--output", type=Path, required=True, help="A new results directory")
    parser.add_argument("--rounds", type=int, default=7, help="Requests per mode, including one cold request")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--qwen-range-fix", action="store_true",
                        help="Apply the measured Qwen3-4B SwiGLU range correction to fixed layer index 6")
    args = parser.parse_args()
    if args.rounds < 2 or len(set(args.modes)) != len(args.modes):
        parser.error("Use at least two rounds and list each mode only once")
    binary = args.binary.resolve(strict=True)
    config_file = args.config.resolve(strict=True)
    config = json.loads(config_file.read_text())
    resources = (config_file.parent / config.get("base_dir", ".")).resolve(strict=True)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config.update(base_dir=str(resources) + os.sep, backend_type="metal", memory="low", precision="normal",
                  thread_num=4, sampler_type="greedy", max_new_tokens=16)
    config["async"] = False
    config.setdefault("jinja", {}).setdefault("context", {})["enable_thinking"] = False
    base_env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    stats = {}
    logits = {}
    tokens = {}
    for mode in args.modes:
        folder = root / mode
        folder.mkdir()
        (folder / "runtime").mkdir()
        settings = dict(config, tmp_path=str(folder / "runtime"))
        settings.pop("prefill_backend", None)
        settings.pop("coreml_prefill_weights", None)
        if mode != "metal":
            settings.update(prefill_backend="coreml", coreml_prefill_weights="int8_per_channel")
        cfg = folder / "config.json"
        cfg.write_text(json.dumps(settings, indent=2))
        env = dict(base_env, MNN_COREML_CACHE_DIR=str(folder / "runtime"))
        if mode == "gpu-stream":
            env["MNN_PREFILL_BENCH_GPU"] = "1"
        if mode == "fixed":
            from fixed import build_fixed
            export = folder / "export"
            export.mkdir()
            invoke(binary, cfg, ["export"], folder / "export.log", dict(env, MNN_PREFILL_BENCH_EXPORT=str(export)))
            models = folder / "models"
            models.mkdir()
            build_fixed(export, models, args.qwen_range_fix)
            env["MNN_PREFILL_BENCH_FIXED"] = str(models)
        prefix = folder / "first"
        lines = invoke(binary, cfg, ["@128", str(args.rounds), str(prefix)], folder / "bench.log", env)
        rows = [json.loads(line[6:]) for line in lines if line.startswith("BENCH ")]
        if (len(rows) != args.rounds or any(x["prompt_tokens"] != 128 or x["generated_tokens"] != 16
                                           or x["tokens"] != rows[0]["tokens"] for x in rows)):
            raise ValueError(f"Invalid or non-repeatable requests in {folder}")
        completions = [int(match[1]) for line in lines
                       if (match := re.fullmatch(r"\[CoreML bench\] completed layers=(\d+) tokens=128", line))]
        expected = [int(match[1]) for line in lines
                    if (match := re.search(r"\[CoreML\] recognized (\d+) complete transformer layers", line))]
        if mode == "metal":
            if completions or expected:
                raise ValueError("Native Metal baseline entered the Core ML layer path")
        elif len(expected) != 1 or expected[0] < 1 or completions != expected * (args.rounds + 1):
            raise ValueError(f"Missing completed Core ML requests in {folder}")
        record = {k: statistics.median(x[k] for x in rows[1:]) for k in ("prefill_ms", "ttft_ms", "e2e_ms")}
        record["coreml_layers_per_request"] = completions[0] if completions else 0
        record["load_ms"] = float(next(s.split()[1] for s in lines if s.startswith("LOAD_MS ")))
        record["first_request_ttft_ms"] = rows[0]["ttft_ms"]
        record["peak_mib"] = int(next(s.split()[1] for s in lines if s.startswith("PEAK_FOOTPRINT "))) / 2**20
        if mode == "fixed":
            build = json.loads((folder / "build.json").read_text())
            record["conversion_seconds"] = sum(row["convert_seconds"] for row in build)
            record["compiler_seconds"] = sum(row["compile_seconds"] for row in build)
            record["compiled_mib"] = sum(f.stat().st_size for f in models.rglob("*") if f.is_file()) / 2**20
        stats[mode], tokens[mode], logits[mode] = record, rows[0]["tokens"], read_logits(prefix.with_suffix(".f32"))
        print(mode, json.dumps(record), flush=True)
    if "metal" in stats:
        for mode in stats:
            stats[mode]["first_logits_vs_metal"] = compare_logits(logits[mode], logits["metal"])
            stats[mode]["tokens_match_metal"] = tokens[mode] == tokens["metal"]
    if "cpu-stream" in stats and "gpu-stream" in stats:
        if logits["cpu-stream"].tobytes() != logits["gpu-stream"].tobytes() or tokens["cpu-stream"] != tokens["gpu-stream"]:
            raise ValueError("CPU and GPU preparation produced different results")
    (root / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
