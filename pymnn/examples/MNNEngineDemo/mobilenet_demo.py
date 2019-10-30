# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import numpy as np
import MNN
import cv2
def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("mobilenet_v1.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    image = cv2.imread('ILSVRC2012_val_00049999.JPEG')
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224))
    #resize to mobile_net tensor size
    image = image.astype(float)
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    #preprocess it
    image = image.transpose((2, 0, 1))
    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    #construct tensor from np.ndarray
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    print("expect 983")
    print("output belong to class: {}".format(np.argmax(output_tensor.getData())))
if __name__ == "__main__":
    inference()
