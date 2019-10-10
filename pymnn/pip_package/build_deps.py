# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" python wrapper file for build python depency """
import os
import shutil
import platform
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
BUILD_DIR = 'build'
def build_deps():
    """ build depency """
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    #build_main_project
    cmake_build_dir = os.path.join(root_dir, BUILD_DIR)
    if os.path.exists(cmake_build_dir):
        shutil.rmtree(cmake_build_dir)
    os.makedirs(cmake_build_dir)
    os.chdir(cmake_build_dir)
    if IS_WINDOWS:
        os.system('cmake -G "Ninja" -DMNN_BUILD_QUANTOOLS=ON\
            -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release .. && ninja')
    else:
        os.system('cmake -DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j4')
    #build_converter_project
    converter_dir = os.path.join(root_dir, "tools", "converter")
    converter_build_dir = os.path.join(converter_dir, BUILD_DIR)
    if os.path.exists(converter_build_dir):
        shutil.rmtree(converter_build_dir)
    os.makedirs(converter_build_dir)
    os.chdir(converter_build_dir)
    if IS_WINDOWS:
        os.system('cmake -G "Ninja" -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release .. && ninja')
    else:
        os.system('cmake -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j4')
    os.chdir(root_dir)
################################################################################
# Building dependent libraries
################################################################################
if __name__ == '__main__':
    build_deps()
