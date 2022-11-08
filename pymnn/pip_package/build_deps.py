# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" python wrapper file for build python depency """
import os
import shutil
import platform
import sys
import argparse
parser = argparse.ArgumentParser(description='build pymnn deps lib')
parser.add_argument('--internal', dest='internal', action='store_true', default=False,
                    help='build deps with internal log')
parser.add_argument('--torch', dest='torch', action='store_true', default=False,
                    help='build convert with torchscript')
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
BUILD_DIR = 'pymnn_build' # avoid overwrite temporary product when build pymnn

USE_TRT=False
if len(sys.argv) > 1 and sys.argv[1] == '-trt':
    USE_TRT=True

def build_deps():
    if os.path.isdir('../../schema/private'):
        IS_INTERNAL_BUILD = args.internal
    else:
        IS_INTERNAL_BUILD = False
    IS_BUILD_TORCH = args.torch

    """ build depency """
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    #build_main_project
    cmake_build_dir = os.path.join(root_dir, BUILD_DIR)
    if os.path.exists(cmake_build_dir):
        shutil.rmtree(cmake_build_dir)
    os.makedirs(cmake_build_dir)
    os.chdir(cmake_build_dir)
    if IS_WINDOWS:
        os.system('cmake -G "Ninja" -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TORCH=OFF\
            -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON\
            -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF .. && ninja MNN MNNTrain MNNConvert')
    elif IS_LINUX:
        extra_opts = '-DMNN_TENSORRT=ON \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ ' if USE_TRT else ' '
        extra_opts += ' -DMNN_INTERNAL=ON ' if IS_INTERNAL_BUILD else ' '
        extra_opts += ' -DMNN_BUILD_TORCH=ON ' if IS_BUILD_TORCH else ' '
        os.system('cmake ' + extra_opts +
            '-DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release \
            -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON \
            -DMNN_USE_THREAD_POOL=ON -DMNN_OPENMP=OFF .. && make MNN MNNTrain MNNConvert -j4')
    else:
        extra_opts = ' -DMNN_INTERNAL=ON ' if IS_INTERNAL_BUILD else ' '
        extra_opts += ' -DMNN_BUILD_TORCH=ON ' if IS_BUILD_TORCH else ' '
        print(extra_opts)
        os.system('cmake ' + extra_opts + '-DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release \
            -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF -DMNN_EXPR_SHAPE_EAGER=ON -DMNN_TRAIN_DEBUG=ON\
            -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON \
            .. && make MNN MNNTrain MNNConvert -j4')
################################################################################
# Building dependent libraries
################################################################################
if __name__ == '__main__':
    build_deps()
