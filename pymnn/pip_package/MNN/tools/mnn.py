# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" mnn tool summary entry """
from __future__ import print_function
import os
import sys
def usage():
    print("mnn toolsets has following command line tools")
    print("    $mnn")
    print("        list out mnn commands")
    print("    $mnnops")
    print("        get supported ops in mnn engine")
    print("    $mnnconvert")
    print("        convert other model to mnn model")
    print("    $mnnquant")
    print("        quantize  mnn model")
def main():
    """ main function """
    usage()    
    return 0
if __name__ == "__main__":
    main()
