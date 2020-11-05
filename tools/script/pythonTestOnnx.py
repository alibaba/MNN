#!/usr/bin/python
import sys
import onnx
import onnxruntime as ort
import numpy as np
modelName = sys.argv[1]

jsonDict = {}
jsonDict['inputs'] = []
jsonDict['outputs'] = []
import os
print(os.popen("mkdir onnx").read())

inputs = {}
ort_session = ort.InferenceSession(modelName)
model = onnx.load(modelName)
for inputVar in ort_session.get_inputs():
    inp = {}
    shape = inputVar.shape
    inp['shape'] = shape
    inputs[inputVar.name] = np.random.uniform(-0.5, 0.5, shape).astype(np.float32)
    inp['name'] = inputVar.name
    jsonDict['inputs'].append(inp)

print([output.name for output in model.graph.output])
for output in model.graph.output:
    jsonDict['outputs'].append(output.name)

import json
jsonString = json.dumps(jsonDict, indent=4)
with open('onnx/input.json', 'w') as f:
    f.write(jsonString)

print('inputs:')
for key in inputs:
    print(key)
    f = open("onnx/" + key + '.txt', 'w')
    np.savetxt(f, inputs[key].flatten())
    f.close()

#outputs = ort_session.run(None, {inputName: np.random.randn(1, 3, 224, 224).astype(np.float32)})
#outputs = ort_session.run(None, {inputName: np.ones([1, 3, 224, 224]).astype(np.float32)})
outputs = ort_session.run(None, inputs)
print('outputs:')
for i in range(0, len(outputs)):
    outputName = model.graph.output[i].name
    name = 'onnx/' + outputName + '.txt'
    print(name, outputs[i].shape)
    f = open(name, 'w')
    np.savetxt(f, outputs[i].flatten())
    f.close()

print(os.popen("cp " + modelName + " onnx/test.onnx").read())

print(os.popen("mnnconvert -f ONNX --modelFile onnx/test.onnx --MNNModel temp.bin --bizCode test").read());
import MNN.expr as F

varsMap = F.load_as_dict('temp.bin')

for inp in jsonDict['inputs']:
    name = inp['name']
    v = varsMap[name]
    if v.data_format == F.NC4HW4:
        v.reorder(F.NCHW)
    v.resize(inp['shape'])
    v.write(inputs[name])

for i in range(0, len(outputs)):
    out = jsonDict['outputs'][i]
    name = out
    v = varsMap[name]
    if v.data_format == F.NC4HW4:
        v = F.convert(v, F.NCHW)
    if v.dtype != F.float:
        v = F.Cast(v, F.float)
    predict = v.read().flatten()
    target = outputs[i].flatten()
    diff = ((target - predict) * (target - predict)).max()
    print(name, diff)


