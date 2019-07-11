[中文版本](Install_CN.md)

# Build and Install

- [Build-Option](#Build-Option)
- [Linux|arm|aarch64|Darwin](#Linux|arm|aarch64|Darwin)
- [Windows 10 (x64)](#Windows)
- [Android](#Android)
- [iOS](#iOS)

## Build-Option

### MNN_DEBUG
Defaults `OFF`, When `OFF`, remove symbols and build with optimizations.
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
1. Install cmake (version >= 3.10 is recommended), protobuf (version >= 3.0 is required) and gcc (version >= 4.9 is required)
2. `cd /path/to/MNN`
3. `./schema/generate.sh`
4. `./tools/script/get_model.sh`(optional, models are needed only in demo project)
5. `mkdir build && cd build && cmake .. && make -j4`

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

## Windows 10 (x64)
1. Install "Microsoft Visual Studio 2019", cmake (version >= 3.10 is recommended)，powershell
2. Find and click "x64 Native Tools Command Prompt for VS 2019" in Setting
3. Change "Setting -> region and language -> administration -> non-Unicide program language" to English (American), otherwise compile may failed because of encoding format.
4. compile and build MNN
```powershell
cd /path/to/MNN
mkdir build && cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
nmake
```

## Android

1. Install cmake (version >=3.10 is recommended), protobuf (version >= 3.0 is required) and gcc (version >= 4.9 is required)
2. [Download and Install NDK](https://developer.android.com/ndk/downloads/), download the the version before r17 is strongly recommended (otherwise cannot use gcc to build, and building armv7 with clang possibly will get error)
3. Set ANDROID_NDK path, eg: `export ANDROID_NDK=/Users/username/path/to/android-ndk-r14b`
4. `cd /path/to/MNN`
5. `./schema/generate.sh`
6. `./tools/script/get_model.sh`(optional, models are needed only in demo project)
7. `cd project/android`
8. Build armv7 library: `mkdir build_32 && cd build_32 && ../build_32.sh`
9. Build armv8 library: `mkdir build_64 && cd build_64 && ../build_64.sh`

## iOS

1. Install protobuf (version >= 3.0 is required)
2. `cd /path/to/MNN`
3. `./schema/generate.sh`
4. `./tools/script/get_model.sh`(optional, models are needed only in demo project)
5. open [MNN.xcodeproj](../project/ios/) with Xcode on macOS, then build.

Copy `mnn.metallib` to the application's main bundle directory if the Metal backend is in your need. You can also refer to `Run Script` in `Build Phases` of the Playground target.
