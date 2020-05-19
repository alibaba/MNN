# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" python wrapper file for mnn quant tool """
from __future__ import print_function
import os
import sys
import argparse
import _tools as Tools
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
    return 0
if __name__ == "__main__":
    main()
