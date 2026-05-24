#!/usr/bin/python
import sys
import json
import os
import subprocess
import json
import concurrent.futures
import multiprocessing
post_treat = {}
qnn_sdk = os.environ["QNN_SDK_ROOT"]
print(qnn_sdk)
with open(sys.argv[1]) as f:
    post_treat = json.load(f)
soc_id = int(sys.argv[2])
dsp_arch = sys.argv[3]
print('soc_id:', soc_id, "; dsp_arch:", dsp_arch)
qnn_bin_path = os.path.join(qnn_sdk, 'bin', 'x86_64-linux-clang')
qnnModelLibGenerator = os.path.join(qnn_bin_path, 'qnn-model-lib-generator')
qnnContextBinaryGenerator = os.path.join(qnn_bin_path, 'qnn-context-binary-generator')
merges = post_treat["merge"]
cache_dir = 'res'
if 'cache' in post_treat:
    cache_dir = post_treat['cache']
clean_tmp = True


def run_subprocess(cmd, cwd=None, retries=3):
    result = None
    for attempt in range(1, retries + 1):
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
        if result.returncode == 0:
            return result
        print(f"[Retry {attempt}/{retries}] command failed: {' '.join(cmd)}", flush=True)
    return result


def process_src(task):
    i, src = task
    graphname = src.split('/')[-1]
    workdir = os.path.join(os.getcwd(), src)
    raw_files = sorted(
        file_name for file_name in os.listdir(workdir)
        if file_name.endswith(".raw")
    )
    if not raw_files:
        raise RuntimeError(f"No .raw files found for src={src}")

    tar_cmd = ["tar", "-cf", graphname + ".bin", *raw_files]
    tar_result = run_subprocess(tar_cmd, cwd=workdir, retries=1)
    if tar_result.returncode != 0:
        raise RuntimeError(f"Tar failed for src={src}: {' '.join(tar_cmd)}")

    if clean_tmp:
        rm_raw_result = run_subprocess(["rm", *raw_files], cwd=workdir, retries=1)
        if rm_raw_result.returncode != 0:
            raise RuntimeError(f"Remove raw failed for src={src}")

    compile_cmd = [
        "python3",
        qnnModelLibGenerator,
        "-c", os.path.join(workdir, graphname + '.cpp'),
        "-b", os.path.join(workdir, graphname + '.bin'),
        "-t", "x86_64-linux-clang",
        "-o", workdir,
    ]
    compile_result = run_subprocess(compile_cmd, retries=3)
    if compile_result.returncode != 0:
        raise RuntimeError(f"Compile failed for src={src}: {' '.join(compile_cmd)}")

    if clean_tmp:
        rm_bin_result = run_subprocess(
            ["rm", os.path.join(workdir, graphname + '.bin')],
            retries=1,
        )
        if rm_bin_result.returncode != 0:
            raise RuntimeError(f"Remove bin failed for src={src}")

    lib_path = os.path.join(workdir, 'x86_64-linux-clang', 'lib' + graphname + '.so')
    return i, graphname, workdir, lib_path


context_config = {
    "backend_extensions": {
        "shared_library_path": os.path.join(qnn_sdk, "lib","x86_64-linux-clang","libQnnHtpNetRunExtensions.so"),
        "config_file_path": "./htp_backend_extensions.json"
    }
}
htp_so = os.path.join(qnn_sdk, 'lib','x86_64-linux-clang','libQnnHtp.so')

htp_backend_extensions = {
    "graphs": [
        {
            "vtcm_mb": 8,
            "O": 3.0,
            "fp16_relaxed_precision": 1,
            "hvx_threads": 4
        }
    ],
    "devices": [
        {
            "soc_id": soc_id,
            "dsp_arch": dsp_arch,
            "cores": [
                {
                    "core_id": 0,
                    "perf_profile": "burst",
                    "rpc_control_latency": 100
                }
            ]
        }
    ],
    "context": {
        "weight_sharing_enabled": True
    }
}

def process_merge(key, merge_index):
    srcs = merges[key]
    dstname = key.split('/')[-1].replace('.bin', '')
    graphs = [None] * len(srcs)
    libs = [None] * len(srcs)
    workdirs = [None] * len(srcs)

    src_workers = max(1, min(os.cpu_count()//2 or 1, len(srcs)))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=src_workers,
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        futures = {
            executor.submit(process_src, (i, src)): src
            for i, src in enumerate(srcs)
        }
        for future in concurrent.futures.as_completed(futures):
            src = futures[future]
            try:
                i, graphname, workdir, lib_path = future.result()
            except Exception as e:
                raise RuntimeError(f"Process src failed for {src}: {e}") from e
            graphs[i] = graphname
            libs[i] = lib_path
            workdirs[i] = workdir

    local_htp_backend_extensions = json.loads(json.dumps(htp_backend_extensions))
    local_context_config = json.loads(json.dumps(context_config))
    htp_config_path = os.path.abspath(f'htp_backend_extensions_{merge_index}.json')
    context_config_path = os.path.abspath(f'context_config_{merge_index}.json')
    local_htp_backend_extensions['graphs'][0]['graph_names'] = graphs
    local_context_config['backend_extensions']['config_file_path'] = htp_config_path

    with open(htp_config_path, 'w') as f:
        f.write(json.dumps(local_htp_backend_extensions, indent=4))
    with open(context_config_path, 'w') as f:
        f.write(json.dumps(local_context_config, indent=4))

    libs_str = ",".join(libs)
    context_cmd = [
        qnnContextBinaryGenerator,
        "--model", libs_str,
        "--backend", htp_so,
        "--binary_file", dstname,
        "--config_file", context_config_path,
        "--output_dir", cache_dir,
    ]
    context_result = run_subprocess(context_cmd, retries=3)
    if context_result.returncode != 0:
        raise RuntimeError(f"Context binary generation failed for key={key}: {' '.join(context_cmd)}")

    if clean_tmp:
        for workdir in workdirs:
            rm_workdir_result = run_subprocess(["rm", "-rf", workdir], retries=1)
            if rm_workdir_result.returncode != 0:
                raise RuntimeError(f"Remove workdir failed: {workdir}")

merge_keys = list(post_treat["merge"])
merge_workers = max(1, min(os.cpu_count()//2 or 1, len(merge_keys)))
print(f"[Merge Parallel] running {len(merge_keys)} merge task(s), max_workers={merge_workers}", flush=True)

with concurrent.futures.ProcessPoolExecutor(
    max_workers=merge_workers,
    mp_context=multiprocessing.get_context("fork"),
) as executor:
    futures = {
        executor.submit(process_merge, key, merge_index): key
        for merge_index, key in enumerate(merge_keys)
    }
    for future in concurrent.futures.as_completed(futures):
        key = futures[future]
        try:
            future.result()
        except Exception as e:
            raise RuntimeError(f"Merge failed for key {key}: {e}") from e
