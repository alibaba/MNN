#!/usr/bin/python

import sys
import os
import argparse
import subprocess
import json
import shutil

def makeIO(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "generateLlmIO")
    output = os.path.join(args.cache_path, 'testdir')
    print(os.popen(exe + " " + args.model + " " + output + ' %d' %args.chunk_size).read())

def seperate(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
    model = os.path.join(os.getcwd(), args.model, 'llm.mnn')
    print("model:", model)
    config = {
        "type":"MLDA",
        "skips":[
            "/Reshape_output_0",
            "/Gather_3_output_0",
            "/Gather_4_output_0"
        ],
        "KVCACHE_SIZE_LIMIT":args.max_history_token,
        "testdir":[
        ],
        "cache":"mlda"
    }
    config['testdir'].append(os.path.join("testdir", '1'))
    config['testdir'].append(os.path.join("testdir", '%d' %args.chunk_size))
    cache = os.path.join(os.getcwd(), args.cache_path)
    with open(os.path.join(cache, 'mlda.json'), 'w') as f:
        f.write(json.dumps(config, indent=4))

    process = subprocess.Popen(exe + ' ' + model + ' mlda/llm.mnn mlda.json', bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd = cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')

    process.wait()

def compile_mlda(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "neuropilot", "npu_convert.py")
    cache = os.path.join(os.getcwd(), args.cache_path)
    process = subprocess.Popen("python3 " + exe + ' npu_postreat.json ', bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd = cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()

def output_mlda(args):
    if os.path.exists(os.path.join(args.model, 'mlda')):
        shutil.rmtree(os.path.join(args.model, 'mlda'))
    shutil.move(os.path.join(args.cache_path, 'mlda'), os.path.join(args.model, 'mlda'))
    config_npu = {
        "llm_model": "mlda/llm.mnn",
        "backend_type": "cpu",
        "thread_num": 1,
        "precision": "low",
        "chunk_limits":[args.chunk_size, 1],
        "memory": "low",
        "sampler_type": "penalty",
        "penalty": 1.1
    }
    with open(os.path.join(args.model, "config_mlda.json"), 'w') as f:
        f.write(json.dumps(config_npu, indent = 4))
    shutil.rmtree(args.cache_path)

import time

def convert(args):
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)
    sta = time.time()
    print("Step1: Make IO")
    makeIO(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end
    print("Step2: Seperate Model")
    seperate(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end
    print("Step3: Compile to mlda")
    compile_mlda(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    print("Step4: Move result file to ", args.model)
    output_mlda(args)

    print("End")

def main():
    parser = argparse.ArgumentParser(description='generate_llm_mlda', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help='model(`str` or `os.PathLike`):\nCan be either:')
    parser.add_argument('--mnn_path', type=str, default="../../../build/",
                        help='mnn build path(`str` or `os.PathLike`):\nCan be either:'
    )
    parser.add_argument('--cache_path', type=str, default="tmp",
                        help='cache path for work'
                        )
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='chunk_size for npu'
                        )
    parser.add_argument('--max_history_token', type=int, default=0,
                        help='max history token, default is 0, which mean no limit for history token number'
                        )
    args = parser.parse_args()
    convert(args)


if __name__ == '__main__':
    main()
