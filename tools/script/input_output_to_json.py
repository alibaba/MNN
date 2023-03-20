#!/usr/bin/python
import MNN
import sys
import os
input_mnn = MNN.expr.load_as_dict(sys.argv[1])
output_mnn = MNN.expr.load_as_dict(sys.argv[2])
output_dir = sys.argv[3]
print("Write to ", output_dir)

jsonDict = {}
jsonDict['inputs'] = []

for name in input_mnn:
    inp = {}
    inp['name'] = name
    v = input_mnn[name]
    inp['shape'] = v.shape
    content = v.read().reshape(-1)
    with open(os.path.join(output_dir, name) + ".txt", 'w') as f:
        for i in range(0, content.shape[0]):
            s = '%f' %content[i]
            f.write(s + '\n')
    jsonDict['inputs'].append(inp)

jsonDict['outputs'] = []
for name in output_mnn:
    jsonDict['outputs'].append(name)
    v = output_mnn[name]
    content = v.read().reshape(-1)
    with open(os.path.join(output_dir, name) + ".txt", 'w') as f:
        for i in range(0, content.shape[0]):
            s = '%f' %content[i]
            f.write(s + '\n')
import json
jsonString = json.dumps(jsonDict, indent=4)
with open(os.path.join(output_dir, "input.json"), 'w') as f:
    f.write(jsonString)

