# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" python wrapper file for build python depency """
import os
import shutil
import platform
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
BUILD_DIR = 'pymnn_build' # avoid overwrite temporary product when build pymnn
def build_deps():
    """ build depency """
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    #build_main_project
    cmake_build_dir = os.path.join(root_dir, BUILD_DIR)
    shutil.rmtree(cmake_build_dir)
    os.makedirs(cmake_build_dir)
    os.chdir(cmake_build_dir)
    if IS_WINDOWS:
        os.system('cmake -G "Ninja" -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_CONVERTER=on\
            -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release\
            -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF .. && ninja MNN MNNTrain MNNConvert')
    elif IS_LINUX:
        os.system('cmake -DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release\
            -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF\
            -DMNN_USE_THREAD_POOL=OFF .. && make MNN MNNTrain MNNConvert  -j4')
    else:
        os.system('cmake -DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release\
            -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF\
            .. && make MNN MNNTrain MNNConvert  -j4')
################################################################################
# Building dependent libraries
################################################################################
if __name__ == '__main__':
    build_deps()
