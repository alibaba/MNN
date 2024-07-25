# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" python wrapper file for mnn quant tool """
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

def main():
    """ main function """
    parser = argparse.ArgumentParser()
    parser.add_argument("src_mnn", type=str,\
        help="src mnn file, for example:src.mnn")
    parser.add_argument("dst_mnn", type=str,\
        help="dst mnn file, for example:dst.mnn") 
    parser.add_argument("config", type=str,\
        help="config json file, for example:config.json") 
    args = parser.parse_args()
          
    src_mnn = args.src_mnn
    dst_mnn = args.dst_mnn
    config_json = args.config
    if not os.path.exists(src_mnn):
        return -1
    if not os.path.exists(config_json):
        return -1
    Tools.mnnquant(src_mnn, dst_mnn, config_json)

    if mnn_logger is not None:
        log_dict = {}
        log_dict["tool"] = "mnnquant_python"
        log_dict["model_guid"] = MNN.get_model_uuid(src_mnn)
        src_model_size = os.path.getsize(src_mnn) / 1024.0 / 1024.0
        dst_model_size = os.path.getsize(dst_mnn) / 1024.0 / 1024.0
        compress_rate = src_model_size / dst_model_size
        config_dict = json.load(open(config_json))
        feature_quantize_method = config_dict['feature_quantize_method']
        weight_quantize_method = config_dict['weight_quantize_method']
        log_dict["detail"] = {"feature_quantize_method": feature_quantize_method, "weight_quantize_method": weight_quantize_method, \
                                "src_model_size": src_model_size, "dst_model_size": dst_model_size, "compress_rate": compress_rate}
        mnn_logger.put_log(log_dict, "quant")

    return 0


if __name__ == "__main__":
    main()
