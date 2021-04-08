# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" python wrapper file for mnn converter tool """
from __future__ import print_function
import os
import sys
import argparse
import _tools as Tools

def main():
    """ main funcion """
    TF, CAFFE, ONNX, MNN, TFLITE = 0, 1, 2, 3, 4
    framework_map = {'TF': TF, 'CAFFE': CAFFE, 'ONNX': ONNX, 'TFLITE': TFLITE, 'MNN': MNN}
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str,\
        choices=list(framework_map.keys()), default='TF', required=True, help="model type")
    parser.add_argument("--modelFile", type=str, required=True,\
        help="tensorflow Pb or caffeModel, for example:xxx.pb/xxx.caffemodel")
    parser.add_argument("--prototxt", type=str, help="only used for caffe, for example: xxx.prototxt")
    parser.add_argument("--MNNModel", type=str, required=True, help="MNN model, ex: xxx.mnn")
    parser.add_argument("--bizCode", type=str, required=True, help="bizcode, ex: MNN")
    parser.add_argument("--fp16", type=bool, default=False,\
        help="{True,False}\
               Boolean to change the mnn usage. If True, the output\
               model save data in half_float type")
    parser.add_argument("--weightQuantBits", type=int, default=0)
    parser.add_argument("--weightQuantAsymmetric", type=bool, default=False)
    parser.add_argument("--compressionParamsFile", type=str, default=None,
        help="The path of model compression file that stores the int8 calibration \
              table for quantization or auxiliary parameters for sparsity.")

    args = parser.parse_args()
    framework_type = framework_map[args.framework]
    if args.modelFile is None or not os.path.exists(args.modelFile):
        print("modelfile not exist")
        return -1
    if args.MNNModel is None:
        parser.print_help(sys.stderr)()
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
        args.fp16, args.prototxt, args.weightQuantBits, args.weightQuantAsymmetric, args.compressionParamsFile, args.bizCode)
    return 0
if __name__ == "__main__":
    main()
