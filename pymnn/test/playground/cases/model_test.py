import numpy as np
import MNN
F = MNN.expr
nn = MNN.nn

import time
import os
from os.path import join

def _load_data(path, shape, dtype):
    dtype_map = {}
    dtype_map[F.dtype.float] = np.float32
    dtype_map[F.dtype.double] = np.float64
    dtype_map[F.dtype.int] = np.int32
    dtype_map[F.dtype.int64] = np.int64
    dtype_map[F.dtype.uint8] = np.uint8
    dtype = dtype_map[dtype]

    data = []
    with open(path) as f:
        for line in f.readlines():
            new_line = [float(s) for s in line.strip().split()]
            if len(new_line) == 0:
                continue
            data.append(new_line)
    data = np.reshape(np.array(data, dtype=dtype), shape)
    return data

def _compare(mnn_data, out_data):
    return np.abs(mnn_data - out_data).sum()

def _test(model_paths, from_file):
    for model_res_path in model_paths:
        model_path = join(model_res_path, 'temp.bin')
        var_map = F.load_as_dict(model_path)
        input_dicts, output_dicts = F.get_inputs_and_outputs(var_map)
        input_names = [n for n in input_dicts.keys()]
        output_names = [n for n in output_dicts.keys()]
        input_vars = [input_dicts[n] for n in input_names]
        output_vars = [output_dicts[n] for n in output_names]
        if from_file:
            module = nn.load_module_from_file(model_path, input_names, output_names, dynamic=False, shape_mutable=False)
        else:
            module = nn.load_module(input_vars, output_vars, False)
        inp_data_path = join(model_res_path, 'input_0.txt')
        inp_data = _load_data(inp_data_path, input_vars[0].shape, input_vars[0].dtype)
        inp_var = F.const(inp_data, inp_data.shape, F.data_format.NCHW, input_vars[0].dtype)
        if input_vars[0].data_format == F.data_format.NC4HW4:
            inp_var.reorder(F.data_format.NC4HW4)
        out_data_path = join(model_res_path, 'output.txt')
        out_data = _load_data(out_data_path, output_vars[0].shape, output_vars[0].dtype)
        start = time.time()
        mnn_data = module.forward(inp_var).read()
        print('cost time: %f ms' % ((time.time() - start) * 1000))
        error = _compare(mnn_data, out_data)
        print('error: %f ' % error)

def dynamic_module_test(model_paths):
    _test(model_paths, False)

def static_module_test(model_paths):
    _test(model_paths, True)
