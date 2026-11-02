#!/usr/bin/env python3
"""Exercise CoreML's temporary program lifecycle across processes and runtime threads.

Usage: python3 test/backend/coreml/run_cache_stress.py /path/to/run_test.out
Runs only public CoreML APIs on macOS. Each native case checks its CPU reference
and rejects MNN CPU fallback. A fresh directory is used unless --cache is supplied.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import shutil
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.processes < 2 or args.timeout <= 0:
        parser.error("at least two processes and a positive timeout are required")
    binary = args.binary.resolve(strict=True)
    output = (args.output or Path(tempfile.mkdtemp(prefix="mnn-coreml-stress-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = (args.cache or output / "program-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".programs.lock").open("a") as root_lock:
        fcntl.flock(root_lock, fcntl.LOCK_EX)
        existing = set(cache.glob("program-*"))
        abandoned = cache / ("program-" + str(uuid.uuid4()).upper())
        active = cache / ("program-" + str(uuid.uuid4()).upper())
        abandoned.mkdir()
        (abandoned / "owner.lock").touch()
        (abandoned / "sentinel").write_text("abandoned")
        active.mkdir()
        active_owner = (active / "owner.lock").open("w")
        fcntl.flock(active_owner, fcntl.LOCK_EX)
        (active / "sentinel").write_text("active")
    command = [str(binary), "backend/coreml/cache_concurrency", "5", "0", "1"]

    def run(worker):
        # Exercise loading while the earlier wave is already predicting.
        if worker >= args.processes // 2:
            time.sleep(2)
        started = time.monotonic()
        log = output / ("worker-%d.log" % worker)
        with log.open("w") as stream:
            try:
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, MNN_COREML_CACHE_DIR=str(cache),
                             MNN_COREML_TRACE="1", MNN_COREML_VERBOSE="1"),
                    timeout=args.timeout,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        text = log.read_text(errors="replace")
        # One warm-up case plus 4 threads x 16 cases. Native runCase validates
        # two inputs per case, finite FP16 errors, and the actual MNN backend.
        cases = text.count("CoreML CV PASS ")
        passed = (code == 0 and cases == 65 and '"failed":0,"passed":1' in text
                  and "Prediction failed:" not in text and "[CoreML] unsupported" not in text)
        return dict(worker=worker, passed=passed, returncode=code, cases=cases,
                    seconds=time.monotonic() - started, log=str(log))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as pool:
            rows = list(pool.map(run, range(args.processes)))
        abandoned_removed = not abandoned.exists()
        active_preserved = (active / "sentinel").is_file()
        new_directories = set(cache.glob("program-*")) - existing - {active}
        # A user-supplied cache may also be used by unrelated live processes.
        released_removed = None if args.cache is not None else not new_directories
    finally:
        active_owner.close()
        shutil.rmtree(active, ignore_errors=True)
    result = dict(
        passed=(all(row["passed"] for row in rows) and abandoned_removed and active_preserved
                and released_removed is not False),
        abandoned_removed=abandoned_removed, active_preserved=active_preserved,
        released_removed=released_removed, remaining_programs=sorted(str(p) for p in new_directories),
        command=command, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        cache=str(cache), workers=rows,
    )
    report = output / "result.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(str(report))
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
