# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" build wheel tool """
from __future__ import print_function
import argparse
parser = argparse.ArgumentParser(description='build pymnn wheel')
parser.add_argument('--x86', dest='x86', action='store_true', default=False,
                    help='build wheel for 32bit arch, only usable on windows')
parser.add_argument('--version', dest='version', type=str, required=True,
                    help='MNN dist version')
parser.add_argument('--serving', dest='serving', action='store_true', default=False,
                    help='build for internal serving, default False')
parser.add_argument('--env', dest='env', type=str, required=False,
                    help='build environment, e.g. :daily/pre/production')
args = parser.parse_args()

import os
import shutil
import platform
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
if __name__ == '__main__':
    os.system("pip install -U numpy")
    if os.path.exists('build'):
        shutil.rmtree('build')
    comm_args = '--version ' + args.version
    if IS_LINUX:
        comm_args += ' --plat-name=manylinux1_x86_64'
        comm_args += ' --env ' + args.env  if args.env else ''
        comm_args += ' --serving' if args.serving else ''
    if IS_WINDOWS:
        os.putenv('DISTUTILS_USE_SDK', '1')
        os.putenv('MSSdk', '1')
        comm_args += ' --x86' if args.x86 else ''
    os.system('python setup.py bdist_wheel %s' % comm_args)
