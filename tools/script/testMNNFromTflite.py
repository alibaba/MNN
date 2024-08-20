#!/usr/bin/python
import os
import sys
import numpy as np
import tensorflow as tf
import flatbuffers

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

def OutputsOffset(subgraph, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
    if o != 0:
        a = subgraph._tab.Vector(o)
        return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
    return 0

#ref: https://github.com/raymond-li/tflite_tensor_outputter/blob/master/tflite_tensor_outputter.py
def buffer_change_output_tensor_to(model_buffer, new_tensor_i):
    from tensorflow.lite.python import schema_py_generated as schema_fb
    root = schema_fb.Model.GetRootAsModel(model_buffer, 0)
    output_tensor_index_offset = OutputsOffset(root.Subgraphs(0), 0)
    # Flatbuffer scalars are stored in little-endian.
    new_tensor_i_bytes = bytes([
    new_tensor_i & 0x000000FF, \
    (new_tensor_i & 0x0000FF00) >> 8, \
    (new_tensor_i & 0x00FF0000) >> 16, \
    (new_tensor_i & 0xFF000000) >> 24 \
    ])
    # Replace the 4 bytes corresponding to the first output tensor index
    return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]

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
        mnnconvert_name = 'MNNConvert.exe' if os.name == 'nt' else './MNNConvert'
        if not os.path.exists(mnnconvert_name):
            print("./MNNConvert not exist in this path. Use pymnn instead of C++ to test")
            mnnconvert_name = 'mnnconvert'
        convert = mnnconvert_name + ' -f TFLITE --bizCode MNN --modelFile tflite/test.tflite --MNNModel convert_cache.mnn --keepInputFormat=1 --testdir tflite'
        result = os.popen(convert).read()
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
        for output in self.outputs:
            jsonDict['outputs'].append(output)

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
        for i in range(len(self.outputs)):
            outputName = self.outputs[i]
            name = 'tflite/' + outputName + '.txt'
            print(name)
            makeDirForPath(name)
            # print(name, outputs[i].shape)
            f = open(name, 'w')
            np.savetxt(f, outputs[i].flatten())
            f.close()
    def __test_specify_output(self, specify_output_name):
        idx = -1
        for tensor in self.model.get_tensor_details():
            if tensor['name'] == specify_output_name:
                idx = tensor['index']
        if idx == -1:
            print('No tensor name is %s.' % specify_output_name)
            self.Test()
            return
        modelBuffer = open(self.modelName, 'rb').read()
        modelBuffer = buffer_change_output_tensor_to(modelBuffer, idx)
        interpreter = tf.lite.Interpreter(model_content=modelBuffer)
        interpreter.allocate_tensors()
        self.model = interpreter
        self.inputOps, self.outputOps = self.__analyze_inputs_outputs(self.model)
        self.outputs = [specify_output_name]
        self.Test()
    def TestName(self, name):
        self.__test_specify_output(name)
    def Test(self):
        self.__run_tflite()
        res = self.__run_mnn()
        return res

if __name__ == '__main__':
    modelName = sys.argv[1]
    t = TestModel(modelName)
    if len(sys.argv) > 2:
        t.TestName(sys.argv[2])
    else:
        t.Test()
