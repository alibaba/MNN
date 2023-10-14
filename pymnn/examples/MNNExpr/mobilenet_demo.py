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
    net = MNN.nn.load_module_from_file(sys.argv[1], ["input"], ["MobilenetV1/Predictions/Reshape_1"])
    image = cv2.imread(sys.argv[2])
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224))
    #resize to mobile_net tensor size
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    #change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    #Make var to save numpy
    input_var = MNN.expr.placeholder([1, 224, 224, 3], MNN.expr.NHWC)
    input_var.write(image)
    #cv2 read shape is NHWC, Module's need is NC4HW4, convert it
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    #inference
    output_var = net.forward([input_var])
    output_var = output_var[0]
    #the output from net may be NC4HW4, turn to linear layout
    output_var = MNN.expr.convert(output_var, MNN.expr.NHWC)
    print("expect 983")
    print("output belong to class: {}".format(np.argmax(output_var.read())))

if __name__ == "__main__":
    inference()
