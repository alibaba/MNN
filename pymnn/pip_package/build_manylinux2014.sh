#!/bin/bash

# Usage:
# To build python wheels with tensorrt backend:
# sh build_manylinux2014.sh -trt
#
# To build python wheels without tensorrt backend:
# sh build_manylinux2014.sh

set -e

PROJECT_ROOT=$(cd `dirname $0`;cd ../../;pwd)
echo $PROJECT_ROOT
export PROJECT_ROOT
#Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -U numpy
    if [ "$1" == "-trt" ]; then
        USE_TRT=true "${PYBIN}/python" setup.py bdist_wheel
    else
        "${PYBIN}/python" setup.py bdist_wheel
    fi
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    if [ "$1" == "-trt" ]; then
        LD_LIBRARY_PATH=${PROJECT_ROOT}/pymnn_build/source/backend/tensorrt:$LD_LIBRARY_PATH auditwheel repair "$whl" --plat manylinux2014_x86_64 -w wheelhouse/
    else
        auditwheel repair "$whl" --plat manylinux2014_x86_64 -w wheelhouse/
    fi
done
