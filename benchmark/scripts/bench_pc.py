import argparse
parser = argparse.ArgumentParser(description='bench mnn/tensorflow/torch on pc')
parser.add_argument('-f', '--framework', choices=['mnn', 'tf', 'torch'], help='test framework', required=True)
parser.add_argument('--modeldir', help='test model directory', required=True)
parser.add_argument('--thread-num', choices=range(1, 5), default=1, help='model dir')
parser.add_argument('--loop-num', default=10, help='run loop number')
parser.add_argument('--backend', choices=['cpu', 'cuda'])
args = parser.parse_args()

import os
from os.path import join, exists, abspath
import json
import time
import numpy as np

def bench_mnn(config):
    def add_suffix(exe):
        import platform
        if platform.system() == 'Linux':
            return exe
        if platform.system() == 'Windows':
            return exe + '.exe'
        else:
            return exe + '.out'
    mnn_json = {
        'outputs': config['output_layers'],
        'inputs': [{'name': name, 'shape': shape, 'value': 0} for name, shape in zip(config['input_layers'], config['input_shapes'])]
    }
    import tempfile
    from subprocess import Popen, PIPE, STDOUT
    with tempfile.TemporaryDirectory() as dirname:
        with open(join(dirname, 'input.json'), 'w') as f:
            json.dump(mnn_json, f, indent=4)
        run_exe = abspath(join('MNN', 'build', add_suffix('ModuleBasic')))
        mnn_path = abspath(join(args.modeldir, 'mnn', f"{config['model']}.mnn"))
        backend = 0 if args.backend == 'cpu' else 2
        with open(join('result', f"mnn_pc_{args.backend}.txt"), 'a+') as logfile:
            p = Popen([run_exe, mnn_path, dirname, '0', str(backend), str(args.loop_num), str(args.thread_num)], stdout=logfile, stderr=STDOUT, text=True)
            p.wait()
            logfile.flush()

def bench_tf(config):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(args.thread_num)
    model_path = join(args.modeldir, 'pb', f"{config['model']}.pb")
    model = tf.saved_model.load(model_path)
    dtype_map = {'float': tf.float32, 'int': tf.int32}
    input_dict = {name: tf.zeros(shape, dtype=dtype_map[dtype]) for name, shape, dtype in zip(config['input_layers'], config['input_shapes'], config['input_dtypes'])}
    infer_func = model.signatures["serving_default"]
    outputs = infer_func(**input_dict)
    times = []
    for i in range(args.loop_num):
        start_t = time.time()
        outputs = infer_func(**input_dict)
        times.append((time.time() - start_t) * 1000)
    with open(join('result', f"tf_pc_{args.backend}.txt"), 'a+') as logfile:
        f.writelines([
            f"model: {model_path}, backend: {args.backend}, loop_num: {args.loop_num}",
            f"max: {max(times)}, min: {min(times)}, avg: {sum(times) / args.loop_num}"
        ])
    
def bench_torch(config):
    import torch
    model = torch.jit.load(filename, torch.device(args.backend))
    model.eval()
    torch.set_num_threads(args.thread_num)
    dtype_map = {'float': 'torch.FloatTensor', 'int': 'torch.IntTensor'}
    input_list = [torch.rand(shape).type(dtype_map[dtype]) for shape, dtype in zip(config['input_shapes'], config['input_dtypes'])]
    model = ipex.optimize(model, dtype=torch.float32)
    for i in range(10):
        outputs = model.forward(*input_list)
    times = []
    for i in range(args.loop_num):
        start_t = time.time()
        outputs = model.forward(*input_list)
        times.append((time.time() - start_t) * 1000)
    with open(join('result', f"torch_pc_{args.backend}.txt"), 'a+') as logfile:
        f.writelines([
            f"model: {model_path}, backend: {args.backend}, loop_num: {args.loop_num}",
            f"max: {max(times)}, min: {min(times)}, avg: {sum(times) / args.loop_num}"
        ])

def main():
    with open(join(args.modeldir, 'config.json')) as f:
        configs = json.load(f)
    for config in configs:
        if args.framework == 'mnn':
            bench_mnn(config)
        elif args.framework == 'tf':
            bench_tf(config)
        else:
            bench_torch(config)
    

if __name__ == "__main__":
    main()
