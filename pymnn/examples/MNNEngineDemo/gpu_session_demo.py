import os
import sys
import MNN
import numpy as np
import cv2
        
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

    image = cv2.imread(sys.argv[2])
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224))
    #resize to mobile_net tensor size
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    #preprocess it
    image = image.transpose((2, 0, 1))
    #change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    
    # input
    inputTensor = net.getSessionInput(session)
    net.resizeTensor(inputTensor, (1, 3, 224, 224))
    net.resizeSession(session)
    inputTensor.copyFrom(tmp_input)
    # infer
    net.runSession(session)
    outputTensor = net.getSessionOutput(session)
    # output
    outputShape = outputTensor.getShape()
    outputHost = createTensor(outputTensor)
    outputTensor.copyToHostTensor(outputHost)
    
    net.updateCacheFile(session, 0)
    print("expect 983")
    print("output belong to class: {}".format(np.argmax(outputHost.getData())))

if __name__ == '__main__':
    modelName = sys.argv[1] # model path
    modelTest(modelName)
