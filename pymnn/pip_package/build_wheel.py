# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" build wheel tool """
from __future__ import print_function
import os
import shutil
import platform
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
if __name__ == '__main__':
    if os.path.exists('build'):
        shutil.rmtree('build')
    if IS_DARWIN:
        os.system('python setup.py bdist_wheel')
    if IS_LINUX:
        os.system('python setup.py bdist_wheel --plat-name=manylinux1_x86_64')
    if IS_WINDOWS:
        os.system('python setup.py bdist_wheel')
