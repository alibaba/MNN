[中文版本](Install_CN.md)

# Build and Install

- [Build-Option](#Build-Option)
- [Linux|arm|aarch64|Darwin](#Linux|arm|aarch64|Darwin)
- [Android](#Android)
- [iOS](#iOS)

## Build-Option

### MNN_DEBUG
Defaults `ON`, When `OFF`, remove symbols and build with optimizations.
### MNN_OPENMP
Defaults `ON`, When `OFF`, do not use openmp multi-thread, only effective on Android/Linux.
### MNN_OPENCL
Defaults `OFF`, When `ON`, build the OpenCL backend, apply GPU according to setting the forward type to be MNN_FORWARD_OPENCL at inference time.
### MNN_OPENGL
Defaults `OFF`, When `ON`, build the OpenGL backend, apply GPU according to setting the forward type to be MNN_FORWARD_OPENGL at inference time.
### MNN_VULKAN
Defaults `OFF`, When `ON`, build the Vulkan backend, apply GPU according to setting the forward type to be MNN_FORWARD_VULKAN at inference time.
### MNN_METAL
Defaults `OFF`, When `ON`, build the Metal backend, apply GPU according to setting the forward type to be MNN_FORWARD_Metal at inference time, only effective on iOS/macOS.

## Linux|arm|aarch64|Darwin

### Build on Host
1. Install cmake(cmake version >=3.10 is recommended)
2. `cd /path/to/MNN`
3. `./schema/generate.sh`
4. `mkdir build && cd build && cmake .. && make -j4`

Then you will get the MNN library(libMNN.so)

### Cross Compile
Download cross compile toolchain from [Linaro](https://www.linaro.org/)

Example:

1. Download AArch64 toolchain
```bash
mkdir -p linaro/aarch64
cd linaro/aarch64

wget http://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu.tar.xz

tar xvf gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu.tar.xz

export cross_compile_toolchain=linaro/aarch64/aarch64-linux-gnu
```

2. build with cmake
```bash
cmake .. \
-DCMAKE_SYSTEM_NAME=Linux \
-DCMAKE_SYSTEM_VERSION=1 \
-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
-DCMAKE_C_COMPILER=$cross_compile_toolchain/aarch64-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=$cross_compile_toolchain/aarch64-linux-gnu-g++
```

3. compile
```bash
mkdir build
cd build
cmake ..
make -j4
```

## Android

1. Install cmake(cmake version >=3.10 is recommended)
2. [Download and Install NDK](https://developer.android.com/ndk/downloads/), download the the version before r17 is strongly recommended (otherwise cannot use gcc to build, and building armv7 with clang possibly will get error)
3. Set ANDROID_NDK path, eg: `export ANDROID_NDK=/Users/username/path/to/android-ndk-r14b`
4. `cd /path/to/MNN`
5. `cd schema && ./generate.sh && cd ..`
6. `cd project/android`
7. Build armv7 library: `mkdir build_32 && cd build_32 && ../build_32.sh`
8. Build armv8 library: `mkdir build_64 && cd build_64 && ../build_64.sh`

## iOS

open [MNN.xcodeproj](../project/ios/) with Xcode on macOS, then build.
