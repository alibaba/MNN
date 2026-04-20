#!/usr/bin/python

import argparse
import json
import os
import shutil
import subprocess
import time


def tool_path(args, name):
    return os.path.normpath(os.path.join(os.getcwd(), args.mnn_path, name))


def qnn_convert_script(args):
    return os.path.normpath(os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py"))


def run_and_stream(cmd, cwd=None, env=None):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode


def make_llm_io(args, cache_dir):
    output = os.path.join(cache_dir, "testdir")
    if os.path.isdir(output):
        shutil.rmtree(output)
    cmd = [tool_path(args, "generateLlmIO"), args.model, output, str(args.chunk_size)]
    code = run_and_stream(cmd)
    return code, list_llm_testdirs(output)


def separate_llm(args, cache_dir, testdirs):
    model = os.path.join(os.getcwd(), args.model, "llm.mnn")
    print("model:", model)
    if not testdirs:
        testdirs = [str(args.chunk_size), "1"]
    config = {
        "type": "QNN",
        "skips": [],
        "testdir": [os.path.join("testdir", name) for name in testdirs],
        "KVCACHE_SIZE_LIMIT": args.max_history_token,
        "cache": "qnn",
    }
    with open(os.path.join(cache_dir, "qnn.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=4))
    cmd = [tool_path(args, "compilefornpu"), model, "qnn/llm.mnn", "qnn.json"]
    return run_and_stream(cmd, cwd=cache_dir)


def list_visual_testdirs(testdir_root):
    subdirs = []
    if not os.path.isdir(testdir_root):
        return subdirs
    for name in sorted(os.listdir(testdir_root)):
        path = os.path.join(testdir_root, name)
        if os.path.isdir(path):
            subdirs.append(name)
    return subdirs


def list_llm_testdirs(testdir_root):
    subdirs = []
    if not os.path.isdir(testdir_root):
        return subdirs
    for name in sorted(os.listdir(testdir_root), key=lambda x: int(x) if x.isdigit() else x):
        path = os.path.join(testdir_root, name)
        if os.path.isdir(path) and name.isdigit():
            subdirs.append(name)
    return subdirs


def reorder_llm_testdirs(testdirs, chunk_size):
    if not testdirs:
        return testdirs
    ordered = []
    chunk_name = str(chunk_size)
    if chunk_name in testdirs:
        ordered.append(chunk_name)
    if "1" in testdirs and "1" not in ordered:
        ordered.append("1")
    for name in testdirs:
        if name not in ordered:
            ordered.append(name)
    return ordered


def make_visual_io(args, cache_dir):
    output = os.path.join(cache_dir, "testdir")
    if os.path.isdir(output):
        shutil.rmtree(output)
    cmd = [tool_path(args, "generateVisualIO"), args.model, output]
    if args.visual_sizes:
        cmd.append(args.visual_sizes)
    code = run_and_stream(cmd)
    return code, list_visual_testdirs(output)


def separate_visual(args, cache_dir, testdirs):
    model = os.path.join(os.getcwd(), args.model, "visual.mnn")
    print("visual model:", model)
    config = {
        "type": "QNN",
        "skips": [],
        "testdir": [os.path.join("testdir", name) for name in testdirs],
        "cache": "qnn",
        "graph_name": "visual_graph",
    }
    with open(os.path.join(cache_dir, "qnn.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=4))
    cmd = [tool_path(args, "compilefornpu"), model, "qnn/visual.mnn", "qnn.json"]
    return run_and_stream(cmd, cwd=cache_dir)


def has_npu_subgraphs(cache_dir):
    postreat = os.path.join(cache_dir, "npu_postreat.json")
    if not os.path.exists(postreat):
        return False
    with open(postreat, "r", encoding="utf-8") as f:
        data = json.load(f)
    merges = data.get("merge", {})
    for key, srcs in merges.items():
        for src in srcs:
            src_dir = os.path.join(cache_dir, src)
            if os.path.isdir(src_dir) and any(f.endswith(".cpp") for f in os.listdir(src_dir)):
                return True
    return False


def compile_qnn(args, cache_dir):
    qnn_root = os.environ.get("QNN_SDK_ROOT", "")
    child_env = os.environ.copy()
    if qnn_root:
        qnn_lib = os.path.join(qnn_root, "lib", "x86_64-linux-clang")
        child_env["LD_LIBRARY_PATH"] = qnn_lib
    cmd = ["python3", qnn_convert_script(args), "npu_postreat.json", str(args.soc_id), args.dsp_arch]
    return run_and_stream(cmd, cwd=cache_dir, env=child_env)


def merge_dir(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            merge_dir(src, dst)
            continue
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)


def visual_qnn_complete(visual_cache_dir):
    qnn_dir = os.path.join(visual_cache_dir, "qnn")
    if not os.path.isdir(qnn_dir):
        return False
    visual_mnn = os.path.join(qnn_dir, "visual.mnn")
    if not os.path.isfile(visual_mnn):
        return False

    postreat = os.path.join(visual_cache_dir, "npu_postreat.json")
    if not os.path.isfile(postreat):
        return False

    try:
        with open(postreat, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    merges = data.get("merge", {})
    if not merges:
        return False
    for dst in merges.keys():
        dst_name = os.path.basename(dst)
        if not os.path.isfile(os.path.join(qnn_dir, dst_name)):
            return False
    return True


def output_qnn(args, has_visual):
    cache_root = os.path.join(os.getcwd(), args.cache_path)
    final_qnn_dir = os.path.join(args.model, "qnn")
    if os.path.exists(final_qnn_dir):
        shutil.rmtree(final_qnn_dir)
    os.makedirs(final_qnn_dir, exist_ok=True)
    merge_dir(os.path.join(cache_root, "llm", "qnn"), final_qnn_dir)

    visual_qnn_converted = False
    if has_visual:
        visual_cache_dir = os.path.join(cache_root, "visual")
        visual_qnn_dir = os.path.join(cache_root, "visual", "qnn")
        if visual_qnn_complete(visual_cache_dir):
            merge_dir(visual_qnn_dir, final_qnn_dir)
            visual_qnn_converted = True
            visual_weight = os.path.join(args.model, "visual.mnn.weight")
            if os.path.isfile(visual_weight):
                shutil.copy2(visual_weight, os.path.join(final_qnn_dir, "visual.mnn.weight"))
        else:
            print("Warning: visual qnn artifacts are incomplete, fallback to CPU visual model")

    config_path = os.path.join(args.model, "config.json")
    config_npu = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_npu = json.load(f)
    config_npu.update(
        {
            "llm_model": "qnn/llm.mnn",
            "llm_weight": "llm.mnn.weight",
            "backend_type": "cpu",
            "thread_num": 1,
            "precision": "low",
            "chunk_limits": [args.chunk_size, 1],
            "memory": "low",
            "sampler_type": "penalty",
            "penalty": 1.1,
            "npu_model_dir": ".",
        }
    )
    if has_visual:
        if visual_qnn_converted:
            config_npu["visual_model"] = "qnn/visual.mnn"
        else:
            config_npu["visual_model"] = "visual.mnn"
    with open(os.path.join(args.model, "config_qnn.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(config_npu, indent=4, ensure_ascii=False))
    shutil.rmtree(cache_root)


def run_step(title, fn):
    print(title)
    start = time.time()
    result = fn()
    end = time.time()
    print("Cost: ", end - start, " s")
    return result


def convert(args):
    cache_root = os.path.join(os.getcwd(), args.cache_path)
    llm_cache = os.path.join(cache_root, "llm")
    visual_cache = os.path.join(cache_root, "visual")
    os.makedirs(llm_cache, exist_ok=True)
    has_visual = os.path.exists(os.path.join(args.model, "visual.mnn"))

    code_llm, llm_testdirs = run_step("Step1: Make LLM IO", lambda: make_llm_io(args, llm_cache))
    if code_llm != 0:
        raise RuntimeError(f"Step1 failed with exit code {code_llm}")
    if not llm_testdirs:
        raise RuntimeError("Step1 failed: no llm testdirs were generated")
    llm_testdirs = reorder_llm_testdirs(llm_testdirs, args.chunk_size)

    code = run_step("Step2: Seperate LLM Model", lambda: separate_llm(args, llm_cache, llm_testdirs))
    if code != 0:
        raise RuntimeError(f"Step2 failed with exit code {code}")

    code = run_step("Step3: Compile LLM to QNN", lambda: compile_qnn(args, llm_cache))
    if code != 0:
        raise RuntimeError(f"Step3 failed with exit code {code}")

    if has_visual:
        os.makedirs(visual_cache, exist_ok=True)

        code_vis, testdirs = run_step(
            "Step4: Make Visual IO",
            lambda: make_visual_io(args, visual_cache),
        )
        if code_vis != 0:
            raise RuntimeError(f"Step4 failed with exit code {code_vis}")

        code = run_step(
            "Step5: Separate Visual Model",
            lambda: separate_visual(args, visual_cache, testdirs),
        )
        if code != 0:
            raise RuntimeError(f"Step5 failed with exit code {code}")

        if has_npu_subgraphs(visual_cache):
            code = run_step(
                "Step6: Compile Visual to QNN",
                lambda: compile_qnn(args, visual_cache),
            )
            if code != 0:
                print("Warning: Visual QNN compilation failed, will fallback to CPU visual")
        else:
            print("No NPU subgraphs for visual model, will use CPU visual")

    step_idx = 7 if has_visual else 4
    print(f"Step{step_idx}: Move result file to ", args.model)
    output_qnn(args, has_visual)
    print("End")


def main():
    parser = argparse.ArgumentParser(description="generate_llm_qnn", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="model(`str` or `os.PathLike`):\nCan be either:")
    parser.add_argument(
        "--soc_id",
        type=int,
        required=True,
        help="type(`int`, *optional*):" "\n\tThe soc_id., for 8gen3 is 57",
    )
    parser.add_argument(
        "--dsp_arch",
        type=str,
        required=True,
        help="type(`str`, *optional*):" "\n\tThe dsp_arch, for 8gen3 is v75.",
    )
    parser.add_argument(
        "--mnn_path",
        type=str,
        default="../../../build/",
        help="mnn build path(`str` or `os.PathLike`):\nCan be either:",
    )
    parser.add_argument("--cache_path", type=str, default="tmp", help="cache path for work")
    parser.add_argument("--chunk_size", type=int, default=128, help="chunk_size for npu")
    parser.add_argument(
        "--max_history_token",
        type=int,
        default=0,
        help="max history token, default is 0, which mean no limit for history token number",
    )
    parser.add_argument(
        "--visual_sizes",
        type=str,
        default="",
        help="optional visual sizes, comma separated, eg: 416x416,384x416",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
