#!/usr/bin/python
import numpy as np

mnn_module = '.tmp.mnn'
ts_module = '.tmp.pt'
onnx_module = '.tmp.onnx'
input_file = '.input.txt'
output_file = '.output.txt'

def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    stdout, _ = Popen(args, stdout=PIPE, stderr=STDOUT).communicate()
    return str(stdout)

def run_torchscript():
    import torchvision.models as models
    import onnxruntime as ort
    import torch
    resnet18 = models.resnet18(pretrained=True)
    x = torch.rand(1, 3, 224, 224)
    resnet18_ts = torch.jit.trace(resnet18, x)
    resnet18_ts.save(ts_module)
    torch.onnx.export(resnet18, x, onnx_module)
    on_module = ort.InferenceSession(onnx_module)
    inputs = {}
    for inp in on_module.get_inputs():
        inputs[inp.name] = x.numpy()
    y = on_module.run(None, inputs)[0]
    nx = x.numpy().reshape(-1)
    ny = y.reshape(-1)
    np.savetxt(input_file, nx, fmt='%f')
    np.savetxt(output_file, ny, fmt='%f')

def run_mnn():
    # convert to mnn module
    conv_res = run_cmd(['./MNNConvert', '-f', 'TS', '--modelFile', ts_module, '--MNNModel', mnn_module, '--bizCode', 'mnn'])
    if (str(conv_res).find('Done') == -1):
        print('Convert Error!')
        return
    # mnn run
    message = run_cmd(['./testModel.out', mnn_module, input_file, output_file, '0', '0.001'])
    # message = run_cmd(['./testModel.out', mnn_module, '/Users/wangzhaode/x.txt', '/Users/wangzhaode/y.txt', '0', '0.001'])
    if (str(message).find('Correct') == -1):
        print('Run Error!')
        # return
    print(message)

if __name__ == '__main__':
    import os
    import sys
    run_torchscript()
    run_mnn()
