#!/usr/bin/python
import os
import sys
import numpy as np
import tensorflow as tf

def makeDirForPath(filename):
    if filename.find('/') < 0:
        return
    names = filename.split('/')
    dirname = ""
    for l in range(0, len(names)-1):
        dirname = dirname + names[l] + '/'
    if os.path.exists(dirname):
        return
    os.makedirs(dirname)

class TestModel():
    def __copy_to_here(self, modelName):
        newModel = 'tflite/test.tflite'
        print(os.popen("mkdir tflite").read())
        print(os.popen("cp " + modelName + ' ' + newModel).read())
        self.modelName = newModel
        self.model = self.__load_graph(self.modelName)
        self.inputOps, self.outputOps = self.__analyze_inputs_outputs(self.model)
        self.outputs = [output['name'] for output in self.outputOps]
    def __init__(self, modelName):
        self.__copy_to_here(modelName)
    def __run_mnn(self):
        result = os.popen("./TestConvertResult Tflite tflite").read()
        print(result)
        return result
    def __load_graph(self, filename):
        interpreter = tf.lite.Interpreter(model_path=filename)
        interpreter.allocate_tensors()
        return interpreter
    def __analyze_inputs_outputs(self, graph):
        inputs = graph.get_input_details()
        outputs = graph.get_output_details()
        return (inputs, outputs)
    def __get_shape(self, op):
        shape = list(op['shape'])
        for i in range(len(shape)):
            if shape[i] == None or shape[i] < 0:
                shape[i] = 1
            else:
                shape[i] = int(shape[i])
        return shape
    def __run_tflite(self):
        jsonDict = {}
        jsonDict['inputs'] = []
        jsonDict['outputs'] = []
        inputs = {}
        print(self.modelName)
        for inputVar in self.inputOps:
            inp = {}
            inp['name'] = inputVar['name']
            inp['shape'] = self.__get_shape(inputVar)
            inputs[inp['name']] = np.random.uniform(0.1, 1.2, inputVar['shape']).astype(inputVar['dtype'])
            jsonDict['inputs'].append(inp)
        print([output['name'] for output in self.outputOps])
        for output in self.outputOps:
            jsonDict['outputs'].append(output['name'])

        import json
        jsonString = json.dumps(jsonDict, indent=4)
        with open('tflite/input.json', 'w') as f:
            f.write(jsonString)

        print('inputs:')
        for key in inputs:
            print(key)
            name = "tflite/" + key + '.txt'
            makeDirForPath(name)
            f = open(name, 'w')
            np.savetxt(f, inputs[key].flatten())
            f.close()
        for inp in self.inputOps:
            self.model.set_tensor(inp['index'], inputs[inp['name']])
        self.model.invoke()
        outputs = []
        for outp in self.outputOps:
            outputs.append(self.model.get_tensor(outp['index']))
        print('outputs:')
        for i in range(len(outputs)):
            outputName = self.outputs[i]
            name = 'tflite/' + outputName + '.txt'
            print(name)
            makeDirForPath(name)
            # print(name, outputs[i].shape)
            f = open(name, 'w')
            np.savetxt(f, outputs[i].flatten())
            f.close()
    def Test(self):
        self.__run_tflite()
        res = self.__run_mnn()
        return res

if __name__ == '__main__':
    modelName = sys.argv[1]
    t = TestModel(modelName)
    t.Test()
