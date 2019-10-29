#!/bin/bash
set -e
# Install a system package required by our library

PROJECT_ROOT=$(cd `dirname $0`;cd ../../;pwd)
echo $PROJECT_ROOT
export PROJECT_ROOT
#Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" wheel . -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat manylinux2010_x86_64 -w wheelhouse/
done

