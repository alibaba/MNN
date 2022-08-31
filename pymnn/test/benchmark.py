# -*- coding: UTF-8 -*-
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import time
import MNN.numpy as mp
import numpy as np
from prettytable import PrettyTable

res = PrettyTable()
res.field_names = ["function", "numpy", "MNN.numpy"]
x = {'shape':[64000],'dtype':'float32'}

def gen_data(args):
    np_args = []
    mp_args = []
    for arg in args:
        if type(arg) == type({'a':1}):
            shape = arg['shape']
            dtype = arg['dtype']
            np_x = np.random.rand(*shape).astype(dtype)
            mp_x = mp.random.random(shape).astype(getattr(mp, dtype))
            mp_x.fix_as_const()
        else:
            np_x = arg
            mp_x = arg
        np_args.append(np_x)
        mp_args.append(mp_x)
    return np_args, mp_args

def np_eval(func, args, loop, mode):
    if mode == 3:
        t1 = time.time()
        for i in range(loop):
            np_res = func(args)
            np_res.__str__()
        t2 = time.time()
    else:
        t1 = time.time()
        for i in range(loop):
            np_res = func(*args)
            np_res.__str__()
        t2 = time.time()
    return round((t2 - t1) * 1000 / loop, 3)
    
def mnn_eval(func, args, loop, mode):
    if mode == 0:
        t1 = time.time()
        for i in range(loop):
            mp_res = func(*args)
            mp_res.__str__()
        t2 = time.time()
    elif mode == 1:
        t1 = time.time()
        for i in range(loop):
            mp_res = func(*args)
        t2 = time.time()
    elif mode == 2:
        t1 = time.time()
        for i in range(loop):
            mp_res = func(*args)
            for r in mp_res: r.__str__()
        t2 = time.time()
    elif mode == 3:
        t1 = time.time()
        for i in range(loop):
            mp_res = func(args)
            for r in mp_res: r.__str__()
        t2 = time.time()
    return round((t2 - t1) * 1000 / loop, 3)

def bench_funcs(funcs, args, mode=0):
    loop = 10
    np_args, mp_args = gen_data(args)
    for func in funcs:
        np_func = getattr(np, func)
        mp_func = getattr(mp, func)
        np_time = np_eval(np_func, np_args, loop, mode)
        mp_time = mnn_eval(mp_func, mp_args, loop, mode)
        # np_sum += np_time
        # mp_sum += mp_time
        # count += 1
        res.add_row([func, np_time, mp_time])

def unary():
    inputs = [x]
    maths = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh', 'around',
             'floor', 'ceil', 'trunc', 'exp', 'expm1', 'exp2', 'log', 'log2', 'log10', 'log1p', 'sinc', 'signbit', 'positive', 'cbrt',
             'negative', 'reciprocal', 'sqrt', 'cbrt', 'square', 'sign', 'argwhere', 'flatnonzero', 'sort', 'argsort', 'copy']
    bench_funcs(maths, inputs)
    bench_funcs(['modf'], inputs, 2)

def binary():
    inputs = [x] * 2
    funcs = ['greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal', 'multiply', 'add', 'divide', 'power',
             'subtract', 'true_divide', 'floor_divide', 'mod', 'maximum', 'minimum', 'hypot', 'logaddexp', 'logaddexp2', 
             'copysign' ]
    bench_funcs(funcs, inputs)
    bench_funcs(['divmod'], inputs, 2)
    bench_funcs(['ldexp'], [x, 2], 1)
    bench_funcs(['dot', 'vdot', 'inner', 'matmul'], [{'shape':[1024, 1024], 'dtype':'float32'}]*2)
    bench_funcs(['array_equal', 'array_equiv'], [x, x], 1)
    bench_funcs(['bitwise_and', 'bitwise_or', 'bitwise_xor'], [{'shape':[64000], 'dtype':'int32'}]*2)
    bench_funcs(['where'], [{'shape':[64000], 'dtype':'int32'}, x, x])

def reduce():
    inputs = [x]
    reduce = ['prod', 'sum', 'argmax', 'argmin', 'cumsum', 'cumprod', 'nonzero', 'count_nonzero', 'max', 'min', 'ptp', 'mean', 'var', 'std']
    bench_funcs(reduce, inputs, 1)
    bench_funcs(['all', 'any'], [{'shape':[64000], 'dtype':'int32'}], 1)

def memory():
    y = {'shape':[4, 16, 10, 100],'dtype':'float32'}
    bench_funcs(['reshape'], [y, [10, 64, 100]])
    bench_funcs(['ravel', 'transpose', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'squeeze'], [y])
    bench_funcs(['moveaxis', 'rollaxis', 'swapaxes'], [y, 0, 3])
    bench_funcs(['broadcast_to'], [y, [3, 4, 16, 10, 100]])
    bench_funcs(['expand_dims'], [y, 0])
    bench_funcs(['concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack', 'row_stack'], [y, y], 3)
    bench_funcs(['split', 'dsplit', 'hsplit', 'vsplit'], [y, 2], 2)
    bench_funcs(['pad'], [x, 2])
    bench_funcs(['tile', 'repeat'], [x, 2])

def linalg():
    loop=10
    mode = 2
    np_args, mp_args = gen_data([{'shape':[9, 9],'dtype':'float32'}])
    np_time = np_eval(np.linalg.svd, np_args, loop, mode)
    mp_time = mnn_eval(mp.linalg.svd, mp_args, loop, mode)
    res.add_row(['svd', np_time, mp_time])

def all():
    unary()
    binary()
    reduce()
    memory()
    linalg()

def log():
    np_sum = 0
    mp_sum = 0
    count  = len(res.rows)
    for row in res.rows:
        np_sum += row[1]
        mp_sum += row[2]
    res.add_row(['avg', round(np_sum/count, 3), round(mp_sum/count, 3)])
    print(res)
    
if __name__ == '__main__':
    all()
    log()
