# Copyright @ 2019 Alibaba. All rights reserved.
# Created by MNN on 2021.11.24
""" python demo usage about MNN API """
from __future__ import print_function
import MNN.numpy as np
import MNN
import MNN.cv as cv2
import sys

def inference():
    """ inference mobilenet_v1 using a specific picture """
    net = MNN.nn.load_module_from_file(sys.argv[1], [], [])
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
    #Make var to save numpy; [h, w, c] -> [n, h, w, c]
    input_var = np.expand_dims(image, [0])
    #cv2 read shape is NHWC, Module's need is NC4HW4, convert it
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    #inference
    output_var = net.forward(input_var)
    #the output from net may be NC4HW4, turn to linear layout
    output_var = MNN.expr.convert(output_var, MNN.expr.NHWC)
    print("expect 983")
    print("output belong to class: {}".format(np.argmax(output_var)))

if __name__ == "__main__":
    inference()
