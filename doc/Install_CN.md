[English Version](Install_EN.md)

# MNN编译

- [编译选项](#编译选项)
- [Linux|arm|aarch64|Darwin](#Linux|arm|aarch64|Darwin)
- [Android](#Android)
- [iOS](#iOS)

## 编译选项

使用`cmake`编译时，可以修改`CMakeLists.txt`中的选项：

### MNN_DEBUG
默认开启，关闭后，不保留符号，开启优化。
### MNN_OPENMP
默认开启，关闭后，禁用openmp多线程优化，仅限Android/Linux上开启。
### MNN_OPENCL
默认关闭，开启后，编译OpenCL部分，可以通过指定MNN_FORWARD_OPENCL利用GPU进行推理。
### MNN_OPENGL
默认关闭，开启后，编译OpenGL部分，可以通过指定MNN_FORWARD_OPENGL利用GPU进行推理。
### MNN_VULKAN
默认关闭，开启后，编译Vulkan部分，可以通过指定MNN_FORWARD_VULKAN利用GPU进行推理。
### MNN_METAL
默认关闭，开启后，编译Metal部分，可以通过指定MNN_FORWARD_METAL利用GPU进行推理，仅限iOS或macOS上开启。

## Linux|arm|aarch64|Darwin
### 本地编译
步骤如下：
1. 安装cmake（建议使用3.10或以上版本）
2. `cd /path/to/MNN`
3. `./schema/generate.sh`
4. `mkdir build && cd build && cmake .. && make -j4`

编译完成后本地出现MNN的动态库。

### 交叉编译
交叉编译工具链可使用[Linaro](https://www.linaro.org/)

1. 下载AArch64交叉编译工具链
```bash
mkdir -p linaro/aarch64
cd linaro/aarch64

wget http://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu.tar.xz

tar xvf gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu.tar.xz

export cross_compile_toolchain=linaro/aarch64/aarch64-linux-gnu
```

2. 使用cmake命令行构建
```bash
cmake .. \
-DCMAKE_SYSTEM_NAME=Linux \
-DCMAKE_SYSTEM_VERSION=1 \
-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
-DCMAKE_C_COMPILER=$cross_compile_toolchain/aarch64-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=$cross_compile_toolchain/aarch64-linux-gnu-g++
```

3. 编译MNN
```bash
mkdir build
cd build
cmake ..
make -j4
```


## Android

步骤如下:
1. 安装cmake（建议使用3.10或以上版本）
2. 在`https://developer.android.com/ndk/downloads/`下载安装NDK，最好不要超过r17、r18及之后的ndk版本（否则，无法使用gcc编译，且clang在编译32位的so时有bug）
3. 在 .bashrc 或者 .bash_profile 中设置 NDK 环境变量，eg: export ANDROID_NDK=/Users/username/path/to/android-ndk-r14b
4. `cd /path/to/MNN`
5. `cd schema && ./generate.sh && cd ..`
6. `cd project/android`
7. 编译armv7动态库：`mkdir build_32 && cd build_32 && ../build_32.sh`
8. 编译armv8动态库：`mkdir build_64 && cd build_64 && ../build_64.sh`

## iOS

在macOS下，用Xcode打开project/ios/MNN.xcodeproj，点击编译即可
