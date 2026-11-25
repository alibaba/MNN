"""Measure adjacent-layer overlap from Xcode Instruments signpost and Neural Engine exports."""
import argparse
import json
import re
from pathlib import Path
import statistics
from xml.etree import ElementTree


def table(path):
    root = ElementTree.parse(path)
    ids = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}

    def read(row, tag, formatted=False):
        item = row.find(tag)
        if item is None:
            return ""
        if "ref" in item.attrib:
            item = ids[item.attrib["ref"]]
        return item.get("fmt", "") if formatted else item.text or ""

    return root.findall(".//row"), read


def analyze(signposts, hardware, layers, skip, count, pid):
    rows, get = table(signposts)
    processes = {get(r, "process", True) for r in rows if get(r, "signpost-name", True) == "CoreMLLayer"}
    if pid:
        processes = {p for p in processes if f"({pid})" in p}
    if len(processes) != 1:
        raise ValueError(f"Choose one target process with --pid, found {sorted(processes)}")
    process = processes.pop()
    spans = {"CoreMLLayer": [], "CoreMLWeight": [], "WeightGPU": []}
    evals, active = [], {}
    for row in rows:
        if get(row, "process", True) != process:
            continue
        name = get(row, "signpost-name", True)
        time = int(get(row, "event-time") or 0)
        meta = get(row, "os-log-metadata", True)
        if name == "_ANEF_MODEL_EVAL" and re.search(r"\bok:1\b", meta):
            evals.append(time)
        elif name == "WeightGPU":
            fields = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", meta.replace(",", "")))
            anchor = time - fields["anchor"]
            spans[name].append((fields["layer"], anchor + fields["start"], anchor + fields["end"]))
        elif name in spans:
            key = name, get(row, "os-signpost-identifier")
            kind = get(row, "event-type", True)
            if kind == "Begin":
                active[key] = (int(re.search(r"layer=(\d+)", meta)[1]), time)
            elif kind == "End" and key in active:
                index, start = active.pop(key)
                spans[name].append((index, start, time))

    rows, get = table(hardware)
    predictions = []
    for row in rows:
        if "Prediction" in get(row, "formatted-label", True):
            start = int(get(row, "start-time"))
            predictions.append((start, start + int(get(row, "duration"))))

    # Requests execute in order. Pair preparation by index inside the same request even
    # if it starts before its predecessor enters the host Core ML call.
    calls = sorted(spans["CoreMLLayer"], key=lambda event: event[1])
    starts = [i for i, call in enumerate(calls) if call[0] == 0]
    device, host, prep, gpu, overlap, gpu_overlap, late = [], [], [], [], [], [], []
    fits = 0
    if len(starts) < skip + count:
        raise ValueError("Trace has too few requests")
    for request in range(skip, skip + count):
        begin = starts[request]
        block = calls[begin:begin + layers]
        if [idx for idx, _, _ in block] != list(range(layers)):
            raise ValueError(f"Incomplete layer sequence in request {request}")
        hw = []
        for idx, start, end in block:
            windows = [(a, b) for a, b in predictions if start <= a <= b <= end]
            if not windows or not any(start <= t <= end for t in evals):
                raise ValueError(f"Missing ANE evidence for request {request} layer {idx}")
            hw.append(windows)
            device.append(sum(b - a for a, b in windows))
            host.append(end - start)
        lower = calls[begin - 1][2] if begin else -1
        upper = block[-1][2]
        for key in ("CoreMLWeight", "WeightGPU"):
            intervals = {i: (a, b) for i, a, b in spans[key] if lower <= a <= b <= upper}
            if not spans[key]:
                continue
            if set(intervals) != set(range(layers)):
                raise ValueError(f"Incomplete {key} evidence in request {request}")
            for i in range(1, layers):
                a, b = intervals[i]
                common = sum(max(0, min(b, y) - max(a, x)) for x, y in hw[i - 1])
                if key == "CoreMLWeight":
                    prep.append(b - a)
                    overlap.append(common)
                    late.append(max(0, b - block[i - 1][2]))
                    fits += b - a <= sum(y - x for x, y in hw[i - 1])
                else:
                    gpu.append(b - a)
                    gpu_overlap.append(common)
    result = {"process": process, "requests": count, "hardware_matched_layers": len(device),
              "host_layer_p50_ms": statistics.median(host) / 1e6, "ane_layer_p50_ms": statistics.median(device) / 1e6}
    if prep:
        result.update(prepare_p50_ms=statistics.median(prep) / 1e6, adjacent_pairs=len(prep),
                      fits_in_previous_ane=fits, prepare_ane_overlap=sum(overlap) / sum(prep),
                      exposed_wait_ms_per_request=sum(late) / count / 1e6)
    if gpu:
        result["gpu_device_ane_overlap"] = sum(gpu_overlap) / sum(gpu)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signposts", type=Path)
    parser.add_argument("hardware", type=Path)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--skip", type=int, default=1, help="Complete requests preceding the measurement")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.layers <= 0 or args.skip < 0 or args.requests <= 0:
        parser.error("Expected positive layers/requests and nonnegative skip")
    print(json.dumps(analyze(args.signposts, args.hardware, args.layers, args.skip, args.requests, args.pid), indent=2))
