# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" get mnn supported ops """
from __future__ import print_function
import os
import sys
from .mnn_fb import OpType
def main():
    """ main function """
    print("mnn supported ops:")
    op_name  = []
    for k, v in OpType.OpType.__dict__.items():
        if k.find('__') < 0:
            op_name.append(k)
    op_name.sort()
    for name in op_name:
        print(name)
    return 0
if __name__ == "__main__":
    main()
