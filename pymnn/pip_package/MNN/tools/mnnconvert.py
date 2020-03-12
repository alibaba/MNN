# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" python wrapper file for mnn converter tool """
from __future__ import print_function
import os
import argparse
import _tools as Tools

def usage():
    """ print usage info """
    print("usage: mnnconvert [-h]")
    print("    [--framework {TF,CAFFE,ONNX,TFLITE,MNN}")
    print("    [--modelFile MODELFILE]")
    print("    [--prototxt PROTOTXT]")
    print("    [--MNNModel MNNMODEL]")
    print("    [--fp16 {True,False}]") 

def main():
    """ main funcion """
    accepted_framework = ['TF', 'CAFFE', 'ONNX', 'TFLITE', 'MNN']
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str,\
        choices=['TF', 'CAFFE', 'ONNX', 'TFLITE', 'MNN'], default='TF',\
        required=True, help="model type, for example:TF/CAFFE/ONNX/TFLITE/MNN")
    parser.add_argument("--modelFile", type=str, required=True,\
        help="tensorflow Pb or caffeModel, for example:xxx.pb/xxx.caffemodel")
    parser.add_argument("--prototxt", type=str,\
        help="only used for caffe, for example: xxx.prototxt")
    parser.add_argument("--MNNModel", type=str, required=True,\
        help="MNN model, ex: xxx.mnn")
    parser.add_argument("--fp16", type=bool, default=False,\
        help="{True,False}\
               Boolean to change the mnn usage. If True, the output\
               model save data in half_float type")
    TF = 0
    CAFFE = 1
    ONNX = 2
    MNN = 3
    TFLITE = 4
    args = parser.parse_args()
    if args.framework.upper() in accepted_framework:
        if args.framework == 'TF':
            framework_type = TF
        elif args.framework.upper() == 'CAFFE':
            framework_type = CAFFE
        elif args.framework.upper() == 'ONNX':
            framework_type = ONNX
        elif args.framework.upper() == 'MNN':
            framework_type = MNN
        elif args.framework.upper() == 'TFLITE':
            framework_type = TFLITE
    else:
        usage()
        return -1
    if args.modelFile is None or not os.path.exists(args.modelFile):
        print("modelfile not exist")
        return -1
    if args.MNNModel is None:
        usage()
        return -1
    if args.framework.upper() == 'CAFFE':
        if args.prototxt is None or not os.path.exists(args.prototxt):
            print("prototxt file not exist")
            return -1
    else:
        ### just cheat with a not exist name ###
        args.prototxt = "NA.mnn"
    Tools.mnnconvert(args.MNNModel, args. modelFile, framework_type,\
        args.fp16, args.prototxt)
    return 0
if __name__ == "__main__":
    main()
