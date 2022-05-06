import os
import sys
import MNN
import numpy as np
        
def createTensor(tensor):
    shape = tensor.getShape()
    data = np.ones(shape, dtype=np.float32)
    return MNN.Tensor(shape, tensor.getDataType(), data, tensor.getDimensionType())
    
def modelTest(modelPath):
    print("Testing gpu model calling method\n")

    net = MNN.Interpreter(modelPath)
    net.setCacheFile(".cachefile")
    
    # set 7 for Session_Resize_Defer, Do input resize only when resizeSession
    #net.setSessionMode(7)
    
    # set 9 for Session_Backend_Auto, Let BackGround Tuning
    net.setSessionMode(9)
    # set 0 for tune_num
    net.setSessionHint(0, 20)

    config = {}
    config['backend'] = "OPENCL"
    config['precision'] = "high"
    session = net.createSession(config)
    
    print("Run on backendtype: %d \n" % net.getSessionInfo(session, 2))
    
    allInput = net.getSessionInputAll(session)
    # input
    inputTensor = net.getSessionInput(session)
    inputHost = createTensor(inputTensor)
    inputTensor.copyFrom(inputHost)
    # infer
    net.runSession(session)
    outputTensor = net.getSessionOutput(session)
    # output
    outputShape = outputTensor.getShape()
    outputHost = createTensor(outputTensor)
    outputTensor.copyToHostTensor(outputHost)
    
    net.updateCacheFile(session, 0)

if __name__ == '__main__':
    modelName = sys.argv[1] # model path
    modelTest(modelName)
