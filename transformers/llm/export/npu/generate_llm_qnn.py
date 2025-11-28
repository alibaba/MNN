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
        "type":"QNN",
        "skips":[
        ],
        "testdir":[
        ],
        "cache":"qnn"
    }
    config['testdir'].append(os.path.join("testdir", '1'))
    config['testdir'].append(os.path.join("testdir", '%d' %args.chunk_size))
    cache = os.path.join(os.getcwd(), args.cache_path)
    with open(os.path.join(cache, 'qnn.json'), 'w') as f:
        f.write(json.dumps(config, indent=4))

    process = subprocess.Popen(exe + ' ' + model + ' qnn/llm.mnn qnn.json', bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd = cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')

    process.wait()

def compile_qnn(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
    cache = os.path.join(os.getcwd(), args.cache_path)
    process = subprocess.Popen("python3 " + exe + ' npu_postreat.json %d ' %args.soc_id + ' ' + args.dsp_arch, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd = cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()

def output_qnn(args):
    if os.path.exists(os.path.join(args.model, 'qnn')):
        shutil.rmtree(os.path.join(args.model, 'qnn'))
    shutil.move(os.path.join(args.cache_path, 'qnn'), os.path.join(args.model, 'qnn'))
    config_npu = {
        "llm_model": "qnn/llm.mnn",
        "backend_type": "cpu",
        "thread_num": 1,
        "precision": "low",
        "chunk_limits":[args.chunk_size, 1],
        "memory": "low",
        "sampler_type": "penalty",
        "penalty": 1.1
    }
    with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
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
    print("Step3: Compile to QNN")
    compile_qnn(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    print("Step4: Move result file to ", args.model)
    output_qnn(args)

    print("End")

def main():
    parser = argparse.ArgumentParser(description='generate_llm_qnn', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help='model(`str` or `os.PathLike`):\nCan be either:')
    parser.add_argument('--soc_id', type=int, required=True,
                        help='type(`int`, *optional*):'
                        '\n\tThe soc_id., for 8gen3 is 57'
                        )
    parser.add_argument('--dsp_arch', type=str, required=True,
                        help='type(`str`, *optional*):'
                        '\n\tThe dsp_arch, for 8gen3 is v75.'
                        )
    parser.add_argument('--mnn_path', type=str, default="../../../build/",
                        help='mnn build path(`str` or `os.PathLike`):\nCan be either:'
    )
    parser.add_argument('--cache_path', type=str, default="tmp",
                        help='cache path for work'
                        )
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='chunk_size for npu'
                        )
    args = parser.parse_args()
    convert(args)


if __name__ == '__main__':
    main()
