#!/bin/bash

if [ -d "build" ]; then
  rm -rf build
fi
./generate_schema.sh

mkdir build
cd build

cmake ..
make clean
make -j16
