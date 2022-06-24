# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import numpy as np
import MNN
import cv2
import sys

def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter(sys.argv[1])
    interpreter.setCacheFile('.tempcache')
    config = {}
    config['precision'] = 'low'
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    interpreter.resizeTensor(input_tensor, (1, 3, 224, 224))
    interpreter.resizeSession(session)

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
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((1, 1001), MNN.Halide_Type_Float, np.ones([1, 1001]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    print("expect 983")
    print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
if __name__ == "__main__":
    inference()
