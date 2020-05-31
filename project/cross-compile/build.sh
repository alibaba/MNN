#!/usr/bin/env bash

namelist=("hisi3519v101" "hisi3516d100" "arm-gnueabihf" "aarch64-gnueabihf" "hisi3518e200" "hisi3519a100" "hisi3559a100")

if [ ! -n "$1" ]; then
	echo "Usage: build.sh [arm_name]."
	echo "arm_name: ${namelist[@]}"
	exit
fi

if [ $1 == ${namelist[0]} ]; then 
##### cross compile hisi3519v101(a17+a7)
mkdir -p build-$1
pushd build-$1
cmake -DCMAKE_TOOLCHAIN_FILE=../project/cross-compile/arm.toolchain.cmake -DARM_NAME=$1 ..
make
make install
popd

elif [ $1 == ${namelist[1]} ]; then 
###### cross compile hisi3516d100(a7)
mkdir -p build-$1
pushd build-$1
cmake -DCMAKE_TOOLCHAIN_FILE=../project/cross-compile/arm.toolchain.cmake -DARM_NAME=$1 ..
make
make install
popd


elif [ $1 == ${namelist[2]} ]; then
##### cross compile universal armv7-a
mkdir -p build-$1
pushd build-$1
cmake -DCMAKE_TOOLCHAIN_FILE=../project/cross-compile/arm.toolchain.cmake -DARM_NAME=$1 ..
make
make install
popd

elif [ $1 == ${namelist[3]} ]; then
##### cross compile universal armv8
mkdir -p build-$1
pushd build-$1
cmake -DCMAKE_TOOLCHAIN_FILE=../project/cross-compile/arm.toolchain.cmake -DARM_NAME=$1 ..
make
make install
popd

elif [ $1 == ${namelist[4]} ]; then
###### cross compile hisi3518e200(arm926)
mkdir -p build-$1
pushd build-$1
cmake -DCMAKE_TOOLCHAIN_FILE=../project/cross-compile/arm.toolchain.cmake -DARM_NAME=$1 ..
make
make install
popd

elif [ $1 == ${namelist[5]} ]; then
###### cross compile hisi3519a100(a53)
mkdir -p build-$1
pushd build-$1
cmake -DCMAKE_TOOLCHAIN_FILE=../project/cross-compile/arm.toolchain.cmake -DARM_NAME=$1 ..
make
make install
popd

elif [ $1 == ${namelist[6]} ]; then
###### cross compile hisi3559a100(a73)
mkdir -p build-$1
pushd build-$1
cmake -DCMAKE_TOOLCHAIN_FILE=../project/cross-compile/arm.toolchain.cmake -DARM_NAME=$1 ..
make
make install
popd


else
	echo "Usage: build.sh [arm_name]."
	echo "arm_name: ${namelist[@]}"
fi
