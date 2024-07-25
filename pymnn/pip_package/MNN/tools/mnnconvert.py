# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" python wrapper file for mnn converter tool """
from __future__ import print_function
import os
import sys
import argparse
import _tools as Tools
import json
import MNN
try:
    from MNN.tools.utils.log import mnn_logger
except:
    mnn_logger = None


def parse_args():
    arg_dict = {}
    args = sys.argv[1:]
    for i in range(len(args)):
        arg = args[i]
        if arg.startswith("--") or arg.startswith("-"):
            arg_name = arg.lstrip("-")
            if i == len(args) - 1:
                arg_value = True
            else:
                arg_value = args[i + 1]
                if arg_value.startswith("--") or arg_value.startswith("-"):
                    arg_value = True
            arg_dict[arg_name] = arg_value
    
    return arg_dict


def main():
    """ main funcion """
    Tools.mnnconvert(sys.argv)

    arg_dict = parse_args()

    if mnn_logger is not None:
        if "modelFile" not in arg_dict.keys() or "MNNModel" not in arg_dict.keys():
            return 0

        log_dict = {}
        log_dict["tool"] = "mnnconvert_python"
        log_dict["model_guid"] = MNN.get_model_uuid(arg_dict["MNNModel"])
        src_model_size = os.path.getsize(arg_dict["modelFile"]) / 1024.0 / 1024.0
        dst_model_size = os.path.getsize(arg_dict["MNNModel"]) / 1024.0 / 1024.0
        compress_rate = src_model_size / dst_model_size
        arg_dict.pop("modelFile")
        arg_dict.pop("MNNModel")
        log_dict["detail"] = {"args": arg_dict, "src_model_size": src_model_size, "dst_model_size": dst_model_size, "compress_rate": compress_rate}
        mnn_logger.put_log(log_dict, "convert")
    
    return 0


if __name__ == "__main__":
    main()
