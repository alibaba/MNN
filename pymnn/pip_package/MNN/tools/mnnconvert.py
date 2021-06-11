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
    Tools.mnnconvert(sys.argv)
    return 0
if __name__ == "__main__":
    main()
