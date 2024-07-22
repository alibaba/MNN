#!/usr/bin/python
import os
import shutil
import sys
import onnx
import onnxruntime as ort
import numpy as np

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

# a class to get idom for graph
class IDominate:
    def __init__(self, n):
        self.N = n+1
        self.E = 0
        self.dfs_id = 0
        self.__data__ = [[0 for i in range(self.N)] for i in range(7)]
        for i in range(2):
            self.__data__.append([0 for i in range(self.N*10)])
        for i in range(3):
            self.__data__.append([i for i in range(self.N)])
        self.forward = self.__data__[0]
        self.backward = self.__data__[1]
        self.dfn = self.__data__[2]
        self.id = self.__data__[3]
        self.father = self.__data__[4]
        self.dom = self.__data__[5]
        self.idom = self.__data__[6]
        self.to = self.__data__[7]
        self.next = self.__data__[8]
        self.union_find = self.__data__[9]
        self.sdom = self.__data__[10]
        self.best = self.__data__[11]

    def __addedge(self, graph, _from, _to):
        self.E += 1
        self.next[self.E] = graph[_from]
        self.to[self.E] = _to
        graph[_from] = self.E
    # using union_find to compress path
    def __push_union_find(self, v):
        if v == self.union_find[v]:
            return v
        y = self.__push_union_find(self.union_find[v])
        if self.dfn[self.sdom[self.best[self.union_find[v]]]] < self.dfn[self.sdom[self.best[v]]]:
            self.best[v] = self.best[self.union_find[v]]
        self.union_find[v] = y
        return y
    # dfs get dfs_id
    def __dfs(self, s):
        self.dfs_id += 1
        self.dfn[s] = self.dfs_id
        self.id[self.dfs_id] = s
        t = self.forward[s]
        while t:
            if not self.dfn[self.to[t]]:
                self.__dfs(self.to[t])
                self.father[self.to[t]] = s
            t = self.next[t]
    # Lengauer-Tarjan Algorithm: A fast algorithm for finding dominators in a flowgraph
    # ref : https://dl.acm.org/doi/10.1145/357062.357071 
    def __tarjan(self):
        for i in range(self.dfs_id, 1, -1):
            u = self.id[i]
            j = self.backward[u]
            while j:
                if not self.dfn[self.to[j]]:
                    j = self.next[j]
                    continue
                self.__push_union_find(self.to[j])
                if self.dfn[self.sdom[self.best[self.to[j]]]] < self.dfn[self.sdom[u]]:
                    self.sdom[u] = self.sdom[self.best[self.to[j]]]
                j = self.next[j]
            self.__addedge(self.dom, self.sdom[u], u)
            self.union_find[u] = self.father[u]
            u = self.id[i-1]
            j = self.dom[u]
            while j:
                self.__push_union_find(self.to[j])
                if self.sdom[self.best[self.to[j]]] == u:
                    self.idom[self.to[j]] = u
                else:
                    self.idom[self.to[j]] = self.best[self.to[j]]
                j = self.next[j]
        for i in range(2, self.dfs_id+1):
            u = self.id[i]
            if self.idom[u] != self.sdom[u]:
                self.idom[u] = self.idom[self.idom[u]]
    def addEdge(self, _from, _to):
        self.__addedge(self.forward, _from, _to)
        self.__addedge(self.backward, _to, _from)
    def getIDoms(self):
        self.__dfs(1)
        self.__tarjan()
        return self.idom

class TestModel():
    def __copy_to_here(self, modelName):
        newModel = os.path.join('onnx', 'test.onnx')
        try:
            os.mkdir("onnx")
        except:
            print('Dir exist')
        shutil.copyfile(modelName, newModel)
        self.modelName = newModel
        self.model = onnx.load(self.modelName)
        self.outputs = [output.name for output in self.model.graph.output]
    def __init__(self, modelName):
        self.__copy_to_here(modelName)
    def __run_mnn(self):
        mnnconvert_name = 'MNNConvert.exe' if os.name == 'nt' else './MNNConvert'
        if not os.path.exists(mnnconvert_name):
            print("./MNNConvert not exist in this path. Use pymnn instead of C++ to test")
            mnnconvert_name = 'mnnconvert'
        convert = mnnconvert_name + ' -f ONNX --bizCode MNN --modelFile onnx/test.onnx --MNNModel convert_cache.mnn --keepInputFormat=1 --testdir onnx'
        result = os.popen(convert).read()
        print(result)
        return result
    def __run_onnx(self):
        jsonDict = {}
        jsonDict['inputs'] = []
        jsonDict['outputs'] = []
        inputs = {}
        print(self.modelName)
        ort_session = ort.InferenceSession(self.modelName)
        for inputVar in ort_session.get_inputs():
            inp = {}
            inp['name'] = inputVar.name
            shapes = inputVar.shape
            for i in range(0, len(shapes)):
                if type(shapes[i]) == str:
                    shapes[i] = 1
            inp['shape'] = shapes
            print(inputVar.type)
            if inputVar.type.find("int64") >= 0:
                inputs[inputVar.name] = np.random.uniform(0, 12, shapes).astype(np.int64)
            elif inputVar.type.find("int32") >=0:
                inputs[inputVar.name] = np.random.uniform(0, 12, shapes).astype(np.int32)
            elif inputVar.type.find('bool') >=0:
                inputs[inputVar.name] = np.random.uniform(0, 1, shapes).astype(np.bool_)
            else:
                # Float
                inputs[inputVar.name] = np.random.uniform(0.1, 1.2, shapes).astype(np.float32)
            jsonDict['inputs'].append(inp)
        print([output.name for output in self.model.graph.output])
        for output in self.model.graph.output:
            jsonDict['outputs'].append(output.name)

        import json
        jsonString = json.dumps(jsonDict, indent=4)
        with open('onnx/input.json', 'w') as f:
            f.write(jsonString)

        print('inputs:')
        for key in inputs:
            print(key)
            path = "onnx/" + key + '.txt'
            makeDirForPath(path)
            f = open(path, 'w')
            np.savetxt(f, inputs[key].flatten())
            f.close()
        outputs = ort_session.run(None, inputs)
        print('outputs:')
        for i in range(0, len(outputs)):
            outputName = self.model.graph.output[i].name
            name = 'onnx/' + outputName + '.txt'
            print(name, outputs[i].shape)
            makeDirForPath(name)
            f = open(name, 'w')
            np.savetxt(f, outputs[i].flatten())
            f.close()
    def __test_specify_output(self, specify_output_name):
        while len(self.model.graph.output) > 0:
            self.model.graph.output.pop()
        new_output = onnx.helper.ValueInfoProto()
        new_output.name = specify_output_name
        self.model.graph.output.append(new_output)
        onnx.save(self.model, self.modelName)
        res = self.Test()
        is_right = ('TEST_SUCCESS' in res or 'Can\'t find var' in res)
        if hasattr(self, 'output_map'):
            print('Test Node :', self.output_map[specify_output_name], is_right)
        return is_right
    def __build_graph(self):
        n = len(self.model.graph.node)
        self.nodes = [node.name for node in self.model.graph.node]
        self.nodes.insert(0, '__null__')
        self.output_map = {}
        self.node_map = {}
        self.node_output = {}
        self.node_input = {}
        idom = IDominate(n); idx = 0
        for node in self.model.graph.node:
            idx += 1
            self.node_map[node.name] = idx
            self.node_output[node.name] = []
            self.node_input[node.name] = []
            for output in node.output:
                self.output_map[output] = node.name
                self.node_output[node.name].append(output)
            for input in node.input:
                if self.output_map.__contains__(input):
                    self.node_input[node.name].append(input)
                    idom.addEdge(self.node_map[self.output_map[input]], self.node_map[node.name])
        self.idoms = idom.getIDoms()
        #print(self.idoms)
    def __get_dom_path(self, left_id, right_id):
        path = []
        while left_id != right_id:
            path.insert(0, right_id)
            right_id = self.idoms[right_id]
        path.insert(0, left_id)
        return path
    def __sub_graph(self, last_right_id, first_error_id):
        while last_right_id == self.idoms[first_error_id]:
            right_input = 0
            inputs = self.node_input[self.nodes[first_error_id]]
            for input in inputs:
                if not self.__test_specify_output(input):
                    first_error_id = self.node_map[self.output_map[input]]
                    break
                else:
                    right_input += 1
            if right_input == len(inputs):
                break
        return first_error_id
    def __binary_search(self, left_id, right_id):
        dom_path = self.__get_dom_path(left_id, right_id)
        left = 0; right = len(dom_path) - 1
        while left < right - 1:
            middle = left + (right - left) // 2
            test_node = self.nodes[dom_path[middle]]
            if self.__test_specify_output(self.node_output[test_node][0]):
                left = middle
            else:
                right = middle
        last_right_node = self.nodes[dom_path[left]]
        first_error_node = self.nodes[dom_path[right]]
        last_right_id = self.node_map[last_right_node]
        first_error_id = self.node_map[first_error_node]
        print('Error is between ', last_right_node, ' and ', first_error_node)
        new_first_error_id = -1
        if last_right_id < first_error_id - 1:
            new_first_error_id = self.__sub_graph(last_right_id, first_error_id)
            if self.idoms[new_first_error_id] != last_right_id:
                self.__binary_search(last_right_id, new_first_error_id)
                return
        if new_first_error_id > 0:
            print('### First Error Node is : ', self.nodes[new_first_error_id])
    def TestName(self, name):
        self.__test_specify_output(name)
    def Test(self):
        self.__run_onnx()
        res = self.__run_mnn()
        return res
    def Debug(self):
        self.__build_graph()
        left_id = self.node_map[self.nodes[1]]
        right_id = self.node_map[self.output_map[self.outputs[0]]]
        self.__binary_search(left_id, right_id)

if __name__ == '__main__':
    modelName = sys.argv[1]
    t = TestModel(modelName)
    if len(sys.argv) > 2:
        if sys.argv[2] == 'DEBUG':
            message = t.Test()
            print(message)
            if message.find("TEST_SUCCESS") < 0:
                debugMode = len(sys.argv) > 2
                print('Debug Mode: ', debugMode)
                t.Debug()
        else:
            specifyOpName = sys.argv[2]
            t.TestName(specifyOpName)
    else:
        t.Test()
