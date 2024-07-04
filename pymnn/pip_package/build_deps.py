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

USE_TRT      = False
USE_CUDA     = False
USE_CUDA_TUNE= False
USE_OPENCL   = False
USE_VULKAN   = False
USE_TORCH    = False
USE_INTERNAL = False
USE_RENDER   = False
USE_SSE      = True
USE_OPENMP   = False
USE_LLM      = False
USE_ARM82    = False

if len(sys.argv) > 1 and sys.argv[1] != None:
    if "trt" in sys.argv[1]:
        USE_TRT = True
    if "cuda" in sys.argv[1]:
        USE_CUDA = True
    if "cuda_tune" in sys.argv[1]:
        USE_CUDA_TUNE = True
    if "opencl" in sys.argv[1]:
        USE_OPENCL = True
    if "vulkan" in sys.argv[1]:
        USE_VULKAN = True
    if "torch" in sys.argv[1]:
        USE_TORCH = True
    if "internal" in sys.argv[1]:
        USE_INTERNAL = True
    if "render" in sys.argv[1]:
        USE_RENDER = True
    if "no_sse" in sys.argv[1]:
        USE_SSE = False
    if "openmp" in sys.argv[1]:
        USE_OPENMP = True
    if "llm" in sys.argv[1]:
        USE_LLM = True
    if "arm82" in sys.argv[1]:
        USE_ARM82 = True

print ("USE_INTERNAL:", USE_INTERNAL)
print ("USE_TRT:", USE_TRT)
print ("USE_CUDA:", USE_CUDA)
if USE_CUDA_TUNE:
    print ("USE_CUDA_TUNE, please note: this function only support Ampere Arch now!")
print ("USE_OPENCL:", USE_OPENCL)
print ("USE_VULKAN:", USE_VULKAN)
print ("USE_RENDER:", USE_RENDER)
print ("USE_SSE:", USE_SSE)
print ("USE_OPENMP:", USE_OPENMP)
print ("USE_LLM:", USE_LLM)
print ("USE_ARM82:", USE_ARM82)

def build_deps():
    """ build depency """
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    #build_main_project
    cmake_build_dir = os.path.join(root_dir, BUILD_DIR)
    if os.path.exists(cmake_build_dir):
        shutil.rmtree(cmake_build_dir)
    os.makedirs(cmake_build_dir)
    os.chdir(cmake_build_dir)
    extra_opts = '-DMNN_LOW_MEMORY=ON'
    if USE_RENDER:
        extra_opts += ' -DMNN_SUPPORT_RENDER=ON'
    if USE_VULKAN:
        extra_opts += ' -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF'
    if USE_OPENCL:
        extra_opts += ' -DMNN_OPENCL=ON'
    if USE_LLM:
        extra_opts += ' -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON'
    if USE_ARM82:
        extra_opts += ' -DMNN_ARM82=ON'
    extra_opts += ' -DMNN_USE_THREAD_POOL=OFF -DMNN_OPENMP=ON' if USE_OPENMP else ' -DMNN_USE_THREAD_POOL=ON -DMNN_OPENMP=OFF'

    if IS_WINDOWS:
        os.system('cmake -G "Ninja" ' + extra_opts +' -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TORCH=OFF\
            -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON\
            -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF .. && ninja MNN MNNConvertDeps')
    elif IS_LINUX:
        extra_opts += '-DMNN_TENSORRT=ON \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ ' if USE_TRT else ' '
        extra_opts += ' -DMNN_INTERNAL=ON ' if USE_INTERNAL else ' '
        extra_opts += ' -DMNN_BUILD_TORCH=ON ' if USE_TORCH else ' '
        if USE_CUDA:
            extra_opts += ' -DMNN_CUDA=ON '
            if USE_CUDA_TUNE:
                extra_opts += ' -DMNN_CUDA_TUNE_PARAM=ON '
        extra_opts += ' ' if USE_SSE else ' -DMNN_USE_SSE=OFF '
        os.system('cmake ' + extra_opts +
            '-DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release \
            -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON \
             .. && make MNN MNNTrain MNNConvertDeps -j32')
    else:
        extra_opts += ' -DMNN_INTERNAL=ON ' if USE_INTERNAL else ' '
        extra_opts += ' -DMNN_BUILD_TORCH=ON ' if USE_TORCH else ' '
        print(extra_opts)
        os.system('cmake ' + extra_opts + '-DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release \
            -DMNN_BUILD_SHARED_LIBS=ON -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF\
            -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON \
            .. && make MNN MNNConvertDeps -j4')
################################################################################
# Building dependent libraries
################################################################################
if __name__ == '__main__':
    build_deps()
