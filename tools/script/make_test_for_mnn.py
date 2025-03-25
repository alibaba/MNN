import MNN
import MNN.expr as F
import MNN.numpy as np
import sys
import os
def makeDirForPath(filename):
    if filename.find('/') < 0:
        return
    names = filename.split('/')
    dirname = ""
    for l in range(0, len(names)-1):
        dirname = dirname + names[l] + '/'
    print(dirname)
    if os.path.exists(dirname):
        return
    os.makedirs(dirname)

def run():
    if len(sys.argv) < 3:
        print('Usage: python3 make_test_for_mnn.py XXX.mnn output')
    print('MNN File: ', sys.argv[1])
    print('Target Dir: ', sys.argv[2])
    
    net = MNN.nn.load_module_from_file(sys.argv[1], [], [])
    info = net.get_info()
    print('inputs: ', info['inputNames'])
    print('outputs:', info['outputNames'])
    
    config = {}
    config['outputs'] = info['outputNames']
    config['inputs'] = []
    inputLen = len(info['inputNames'])
    inputVars = info['inputs']
    inputs = []
    outputDir = sys.argv[2]
    for i in range(0, inputLen):
        input = {}
        input['name'] = info['inputNames'][i]
        var = inputVars[i]
        dtype = var.dtype
        dims = var.shape
        for j in range(0, len(dims)):
            if dims[j] == -1:
                dims[j] = 1
        input['shape'] = dims
        dformat = var.data_format
        var = np.random.random(dims)
        if dtype == np.int32:
            var = var * 10.0
        var = var.astype(dtype)
        data = var.read().flatten()
        fname = os.path.join(outputDir, input['name'] + '.txt')
        makeDirForPath(fname)
        with open(fname, 'w') as f:
            for floatValue in data:
                f.write('%f\n' %floatValue)
        var = F.convert(var, dformat)
        inputs.append(var)
        config['inputs'].append(input)
    
    import json
    jsonString = json.dumps(config, indent=4)
    with open(os.path.join(outputDir, 'input.json'), 'w') as f:
        f.write(jsonString)
    
    outputs = net.forward(inputs)
    for i in range(0, len(outputs)):
        out = outputs[i]
        if out.data_format == F.NC4HW4:
            out = F.convert(out, F.NCHW)
        data = out.read().flatten()
        fname = os.path.join(outputDir, info['outputNames'][i] + '.txt')
        makeDirForPath(fname)
        with open(fname, 'w') as f:
            for floatValue in data:
                f.write('%f\n' %floatValue)


if __name__=='__main__':
    run()
