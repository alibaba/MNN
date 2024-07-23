#!/usr/bin/python
import os
import sys
import numpy as np
import torch

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

class TestModel():
    def __copy_to_here(self, modelName):
        newModel = 'torch/test.pt'
        print(os.popen("mkdir torch").read())
        print(os.popen("cp " + modelName + ' ' + newModel).read())
        self.modelName = newModel
        self.model = self.__load_graph(self.modelName)
        self.inputs, self.outputs = self.__analyze_inputs_outputs(self.model)
    def __init__(self, modelName):
        self.__copy_to_here(modelName)
    def __run_mnn(self):
        mnnconvert_name = 'MNNConvert.exe' if os.name == 'nt' else './MNNConvert'
        if not os.path.exists(mnnconvert_name):
            print("./MNNConvert not exist in this path. Use pymnn instead of C++ to test")
            mnnconvert_name = 'mnnconvert'
        convert = mnnconvert_name + ' -f TORCH --bizCode MNN --modelFile torch/test.pt --MNNModel convert_cache.mnn --keepInputFormat=1 --testdir torch'
        result = os.popen(convert).read()
        print(result)
        return result
    def __load_graph(self, filename):
        model = torch.jit.load(filename, torch.device('cpu'))
        return model
    def __analyze_inputs_outputs(self, graph):
        return (['x.1'], ['ret'])
    def __get_shape(self, op):
        return [1, 3, 224, 224] 
    def __run_torch(self):
        jsonDict = {}
        jsonDict['controlflow'] = True
        jsonDict['inputs'] = []
        jsonDict['outputs'] = []
        inputs = {}
        print(self.modelName)
        for inputName in self.inputs:
            inp = {}
            inp['name'] = inputName
            inp['shape'] = self.__get_shape(inputName)
            inputs[inputName] = torch.rand(inp['shape'])
            # inputs[inputName] = torch.ones(inp['shape'])
            jsonDict['inputs'].append(inp)
        for output in self.outputs:
            jsonDict['outputs'].append(output)

        import json
        jsonString = json.dumps(jsonDict, indent=4)
        with open('torch/input.json', 'w') as f:
            f.write(jsonString)

        print('inputs:')
        for key in inputs:
            print(key)
            f = open("torch/" + key + '.txt', 'w')
            np.savetxt(f, inputs[key].flatten())
            f.close()
        self.model.eval()
        outputs = self.model.forward(inputs[self.inputs[0]]).detach().numpy()
        print('outputs:')
        for i in range(len(outputs)):
            outputName = self.outputs[i]
            name = 'torch/' + outputName + '.txt'
            makeDirForPath(name)
            # print(name, outputs[i].shape)
            f = open(name, 'w')
            np.savetxt(f, outputs[i].flatten())
            f.close()
    def Test(self):
        self.__run_torch()
        res = self.__run_mnn()
        return res

if __name__ == '__main__':
    modelName = sys.argv[1]
    specifyOpName = None
    if len(sys.argv) > 2:
        specifyOpName = sys.argv[2]
    t = TestModel(modelName)
    t.Test()
