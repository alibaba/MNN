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
    print(dirname)
    if os.path.exists(dirname):
        return
    os.makedirs(dirname)

class TestModel():
    def __copy_to_here(self, modelName):
        newModel = 'tf/test.pb'
        print(os.popen("mkdir tf").read())
        print(os.popen("cp " + modelName + ' ' + newModel).read())
        self.modelName = newModel
        self.model = self.__load_graph(self.modelName)
        self.inputOps, self.outputOps = self.__analyze_inputs_outputs(self.model)
        self.outputs = [output.name for output in self.outputOps]
    def __init__(self, modelName):
        self.__copy_to_here(modelName)
    def __run_mnn(self):
        mnnconvert_name = 'MNNConvert.exe' if os.name == 'nt' else './MNNConvert'
        if not os.path.exists(mnnconvert_name):
            print("./MNNConvert not exist in this path. Use pymnn instead of C++ to test")
            mnnconvert_name = 'mnnconvert'
        convert = mnnconvert_name + ' -f TF --bizCode MNN --modelFile tf/test.pb --MNNModel convert_cache.mnn --keepInputFormat=1 --testdir tf'
        result = os.popen(convert).read()
        print(result)
        return result
    def __load_graph(self, filename):
        f = tf.io.gfile.GFile(filename, "rb")
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph = tf.compat.v1.get_default_graph() 
        return graph
    def __analyze_inputs_outputs(self, graph):
        ops = graph.get_operations()
        outputs_set = set(ops)
        inputs = []
        testop = None
        for op in ops:
            if op.name == specifyOpName:
                testop = op
            if len(op.inputs) == 0 and op.type != 'Const':
                inputs.append(op)
            else:
                for input_tensor in op.inputs:
                    if input_tensor.op in outputs_set:
                        outputs_set.remove(input_tensor.op)
        outputs = [op for op in outputs_set if op.type != 'Assert' and op.type != 'Const']
        if testop != None:
            outputs = [ testop ]
        return (inputs, outputs)
    def __get_shape(self, op):
        shape = [s.value if tf.__version__[0] == '1' else s for s in op.outputs[0].shape]
        for i in range(len(shape)):
            if shape[i] == None:
                shape[i] = 1
        return shape
    def __run_tf(self):
        jsonDict = {}
        jsonDict['inputs'] = []
        jsonDict['outputs'] = []
        inputs = {}
        print(self.modelName)
        for inputVar in self.inputOps:
            inp = {}
            inp['name'] = inputVar.name
            inp['shape'] = self.__get_shape(inputVar)
            inputs[inputVar.name + ':0'] = np.random.uniform(0.1, 1.2, inp['shape']).astype(np.sctypeDict[inputVar.outputs[0].dtype.name])
            jsonDict['inputs'].append(inp)
        print([output.name for output in self.outputOps])
        for output in self.outputOps:
            jsonDict['outputs'].append(output.name)

        import json
        jsonString = json.dumps(jsonDict, indent=4)
        with open('tf/input.json', 'w') as f:
            f.write(jsonString)

        print('inputs:')
        for key in inputs:
            print(key)
            name = 'tf/' + key[:-2] + '.txt'
            makeDirForPath(name)
            f = open(name, 'w')
            np.savetxt(f, inputs[key].flatten())
            f.close()
        sess = tf.compat.v1.Session()
        outputs_tensor = [(output + ':0') for output in self.outputs]
        outputs = sess.run(outputs_tensor, inputs)
        print('outputs:')
        for i in range(len(outputs)):
            outputName = self.outputs[i]
            name = 'tf/' + outputName + '.txt'
            makeDirForPath(name)
            # print(name, outputs[i].shape)
            f = open(name, 'w')
            np.savetxt(f, outputs[i].flatten())
            f.close()
    def Test(self):
        self.__run_tf()
        res = self.__run_mnn()
        return res

if __name__ == '__main__':
    modelName = sys.argv[1]
    specifyOpName = None
    if len(sys.argv) > 2:
        specifyOpName = sys.argv[2]
    t = TestModel(modelName)
    t.Test()
