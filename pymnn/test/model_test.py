# -*- coding: UTF-8 -*-
import os
import sys
import MNN
import numpy as np

total_num = 0
wrongs = []

def parseConfig(root_dir):
    configName = os.path.join(root_dir, 'config.txt')
    if not os.path.exists(configName):
        return False
    try:
        config = open(configName, 'rt', encoding='utf-8')
    except:
        import io
        config = io.open(configName, 'rt', encoding='utf-8')
    res = {}
    res['model_name'] = os.path.join(root_dir, 'temp.bin')
    for line in config.readlines():
        if line[0] == '#':
            continue
        value = line[line.find(' = ') + 3:].strip()
        if 'input_size' in line:
            res['input_size'] = int(value)
        elif 'input_names' in line:
            input_names = value.split(',')
            res['input_names'] = input_names
            res['given_names'] = [ os.path.join(root_dir, x + '.txt') for x in input_names]
        elif 'input_dims' in line:
            res['input_dims'] = []
            for val in value.split(','):
                res['input_dims'].append([int(x) for x in val.split('x')])
        elif 'output_size' in line:
            res['output_size'] = int(value)
        elif 'output_names' in line:
            output_names = value.split(',')
            res['output_names'] = output_names
            res['expect_names'] = []
            for i in range(len(output_names)):
                expect_name = os.path.join(root_dir, output_names[i] + '.txt')
                if os.path.exists(expect_name):
                    res['expect_names'].append(expect_name)
                else:
                    res['expect_names'].append(os.path.join(root_dir, str(i) + '.txt'))

    return res

def loadtxt(file, shape, dtype=np.float32):
    size  = np.prod(shape)
    try:
        data = np.loadtxt(fname=file, dtype=dtype).flatten()
    except:
        data = []
        data_file = open(file, 'rt')
        for line in data_file.readlines():
            for x in line.strip().split(' '):
                try:
                    a = float(x)
                    data.append(a)
                except:
                    pass
        data = np.asarray(data)
    if data.size >= size:
        data = data[:size].reshape(shape)
    else:
        data = np.pad(data, (0, size - data.size), 'constant').reshape(shape)
    return data

def MNNDataType2NumpyDataType(data_type):
    if data_type == MNN.Halide_Type_Uint8:
        return np.uint8
    elif data_type == MNN.Halide_Type_Double:
        return np.float64
    elif data_type == MNN.Halide_Type_Int:
        return np.int32
    elif data_type == MNN.Halide_Type_Int64:
        return np.int64
    else:
        return np.float32

def createTensor(tensor, file=''):
    shape = tensor.getShape()
    data_type = tensor.getDataType()
    dtype = MNNDataType2NumpyDataType(data_type)
    if file == '':
        data = np.ones(shape, dtype=dtype)
    else:
        data = loadtxt(file, shape, dtype)
    return MNN.Tensor(shape, tensor.getDataType(), data, tensor.getDimensionType())

def compareTensor(tensor, file, atol=5e-2):
    outputNumpyData = tensor.getNumpyData()
    expectNumpyData = loadtxt(file, tensor.getShape())
    return np.allclose(outputNumpyData, expectNumpyData, atol=atol)

def log_result(success, model):
    global total_num
    global wrongs
    total_num += 1
    if success:
        print('Test %s Correct!\n'%model)
    else:
        wrongs.append(model)
        print('Test Failed %s!\n'%model)

def modelTest(modelPath, givenName, expectName):
    print("Testing model %s, input: %s, output: %s" % (modelPath, givenName, expectName))

    net = MNN.Interpreter(modelPath)
    session = net.createSession()
    allInput = net.getSessionInputAll(session)
    # input
    inputTensor = net.getSessionInput(session)
    inputHost = createTensor(inputTensor, givenName)
    inputTensor.copyFrom(inputHost)
    # infer
    net.runSession(session)
    outputTensor = net.getSessionOutput(session)
    # output
    outputShape = outputTensor.getShape()
    outputHost = createTensor(outputTensor)
    outputTensor.copyToHostTensor(outputHost)
    # compare
    success = compareTensor(outputHost, expectName)
    log_result(success, modelPath)

def modelTestWithConfig(config):
    model  = config['model_name']
    inputs = config['input_names']
    shapes = config['input_dims']
    givens = config['given_names']
    outputs = config['output_names']
    expects = config['expect_names']
    print("Testing model %s, input: %s, output: %s" % (model, givens, expects))
    net = MNN.Interpreter(config['model_name'])
    session = net.createSession()
    all_input = net.getSessionInputAll(session)
    # resize
    for i in range(len(inputs)):
        input = inputs[i]
        shape = shapes[i]
        net.resizeTensor(all_input[input], tuple(shape))
    net.resizeSession(session)
    # input
    all_input = net.getSessionInputAll(session)
    for i in range(len(inputs)):
        input = inputs[i]
        given = givens[i]
        input_tensor = all_input[input]
        input_host = createTensor(input_tensor, given)
        input_tensor.copyFrom(input_host)
    # infer
    net.runSession(session)
    all_output = net.getSessionOutputAll(session)
    # output & compare
    success = True
    for i in range(len(outputs)):
        output = outputs[i]
        expect = expects[i]
        output_tensor = all_output[output]
        output_host = createTensor(output_tensor)
        output_tensor.copyToHostTensor(output_host)
        success &= compareTensor(output_host, expect)
    # res
    log_result(success, model)

def testSessionConfig(modelPath, givenName, expectName, session_config, outputTensorName):
    print("Testing model %s, input: %s, output: %s" % (modelPath, givenName, expectName))
    print("with session config:", session_config)

    net = MNN.Interpreter(modelPath)
    session = net.createSession(session_config)
    allInput = net.getSessionInputAll(session)
    # input
    inputTensor = net.getSessionInput(session)
    inputHost = createTensor(inputTensor, givenName)
    inputTensor.copyFrom(inputHost)
    # infer
    net.runSession(session)

    allOutput = net.getSessionOutputAll(session)
    print("output shapes:")
    for key in allOutput.keys():
        print(key, "shape:", allOutput[key].getShape())

    outputTensor = net.getSessionOutput(session, outputTensorName)
    outputHost = createTensor(outputTensor)
    outputTensor.copyToHostTensor(outputHost)
    # compare
    success = compareTensor(outputHost, expectName)
    log_result(success, modelPath)

def testResource(model_root_dir, name):
    root_dir = os.path.join(model_root_dir, 'TestResource')
    print('root: ' + root_dir + '\n')
    for name in os.listdir(root_dir):
        if name == '.DS_Store':
            continue
        modelName = os.path.join(root_dir, name, 'temp.bin')
        inputName = os.path.join(root_dir, name, 'input_0.txt')
        outputName = os.path.join(root_dir, name, 'output.txt')
        modelTest(modelName, inputName, outputName)

def testTestWithDescribe(model_root_dir):
    root_dir = os.path.join(model_root_dir, 'TestWithDescribe')
    print('root: ' + root_dir + '\n')
    for name in os.listdir(root_dir):
        if name == '.DS_Store':
            continue
        config = parseConfig(os.path.join(root_dir, name))
        if config:
            modelTestWithConfig(config)

def testPymnnConfig(model_root_dir):
    root_dir = os.path.join(model_root_dir, "TestResource")
    print("\ntest pymnn session config")
    print('root: ' + root_dir + '\n')

    name = "ocr-single"
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    expectName = os.path.join(root_dir, name, 'output.txt')

    outputTensorName = "topk"
    session_config = {"saveTensors":("conv1", "pool1", outputTensorName)}

    testSessionConfig(modelName, inputName, expectName, session_config, outputTensorName)


if __name__ == '__main__':
    model_root_dir = sys.argv[1]
    testResource(model_root_dir, 'TestResource')
    testResource(model_root_dir, 'OpTestResource')
    testTestWithDescribe(model_root_dir)
    testPymnnConfig(model_root_dir)
    if len(wrongs) > 0:
        print('Wrong: ', len(wrongs))
        for wrong in wrongs:
            print(wrong)
    print('TEST_NAME_PYMNN_MODEL: Pymnn模型测试\nTEST_CASE_AMOUNT_PYMNN_MODEL: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n'%(len(wrongs), total_num - len(wrongs)))
