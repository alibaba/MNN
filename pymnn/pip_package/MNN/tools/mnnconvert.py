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
    print("    [--weightQuantBits {num of bits for weight-only-quant, default:0, which means no quant}]")
    print("    [--weightQuantAsymmetric {True,False use asymmetric quant method for weight-only-quant, \
                the default method is symmetric quant, which is compatible with old MNN versions. \
                you can set this flag to True use asymmetric quant method to improve accuracy of the weight-quant model in some cases, \
                but asymmetric quant model cannot run on old MNN versions. You will need to upgrade MNN to new version to solve this problem. \
                default: False, which means using SYMMETRIC quant method}]")
    print("    [--compressionParamsFile COMPRESSION_PARAMS_PATH]")

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
    parser.add_argument("--weightQuantBits", type=int, default=0)
    parser.add_argument("--weightQuantAsymmetric", type=bool, default=False)
    parser.add_argument("--compressionParamsFile", type=str, default=None,
        help="The path of model compression file that stores the int8 calibration \
              table for quantization or auxiliary parameters for sparsity.")

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
    if args.compressionParamsFile is not None and \
           not os.path.exists(args.compressionParamsFile):
        print("Compression params file not exist.")
        return -1
    if args.compressionParamsFile is None:
        args.compressionParamsFile = ""

    Tools.mnnconvert(args.MNNModel, args. modelFile, framework_type,\
        args.fp16, args.prototxt, args.weightQuantBits, args.weightQuantAsymmetric, args.compressionParamsFile)
    return 0
if __name__ == "__main__":
    main()
