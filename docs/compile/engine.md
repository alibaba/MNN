# 主库编译
默认编译产物为：`libMNN.so`，`express/libMNN_Express.so`
## Linux/MacOS
- 环境要求
  - cmake >= 3.10
  - gcc >= 4.9 或者使用 clang
- 相关编译选项
  - `MNN_AVX512` 是否使用AVX512指令，需要gcc9以上版本编译
  - `MNN_OPENCL` 是否使用OpenCL后端，针对GPU设备
  - `MNN_METAL` 是否使用Metal后端，针对MacOS/iOSGPU设备
  - `MNN_VULKAN` 是否使用Vulkan后端，针对GPU设备
  - `MNN_CUDA`  是否使用CUDA后端，针对Nivida GPU设备
  - 其他编译选项可自行查看 CMakeLists.txt
- 具体步骤
  1. 准备工作 (可选，修改 MNN Schema 后需要）
        ```bash
        cd /path/to/MNN
        ./schema/generate.sh
        ./tools/script/get_model.sh # 可选，模型仅demo工程需要
        ```
  2. 本地编译
        ```bash
        mkdir build && cd build && cmake .. && make -j8
        ```
## Windows
- 环境要求
  - Microsoft Visual Studio >= 2017
  - cmake >= 3.13
  - powershell
  - Ninja
- 相关编译选项
  - 同`Linux/MacOS`
- 具体步骤
  1. opencl/vulkan
     - *(可选)*下载GPU Caps Viewer，你可以通过这个工具来查看本机设备的详细信息（opencl、opengl、vulkan等）
     - sdk和驱动准备
        - [opencl sdk](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases)，将opencl sdk目录的路径加到AMDAPPSDKROOT环境变量
        - [vulkan sdk](https://vulkan.lunarg.com/)，将vulkan skd路径加入VULKAN_SDK环境变量，以备cmake查找
        - [AMD opencl驱动](https://www.amd.com/zh-hans/support)
        - [NVIDIA opencl驱动](https://developer.nvidia.com/opencl)
        - [AMD vulkan驱动](https://community.amd.com/community/gaming/blog/2016/02/16/radeon-gpus-are-ready-for-the-vulkan-graphics-api)
  2. 编译
     - 64位编译：在设置中找到vcvars64.bat（适用于 VS 2017 的 x64 本机工具命令提示）并单击，打开VS编译x64架构程序的虚拟环境
     - 32位编译：在设置中找到vcvarsamd64_x86.bat（VS 2017的 x64_x86 交叉工具命令提示符）并单击，打开VS交叉编译x86架构程序的虚拟环境 
     - 在虚拟环境中执行如下编译命令：
        ```bash
        cd /path/to/MNN
        ./schema/generate.ps1 # 非必须
        mkdir build && cd build
        cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_WIN_RUNTIME_MT=OFF
        ninja
        ```
     - 若需要编译模型转换工具，cmake 命令加上 -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_WIN_RUNTIME_MT=ON
     - 若需要编译 MNN CUDA，MNN_WIN_RUNTIME_MT 和 MNN_BUILD_SHARED_LIBS 需要设成 ON ，另外加上 -DMNN_CUDA=ON: cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=ON -DMNN_WIN_RUNTIME_MT=ON -DMNN_CUDA=ON
     - Windows 上建议使用 Interpreter::destroy , Tensor::destroy , Module::destroy 等方法进行 MNN 相关内存对象的析构，不要直接使用 delete （直接使用 delete 在 -DMNN_WIN_RUNTIME_MT=ON 时会出问题）
## Android
- 环境要求
  - cmake >= 3.10
  - ndk
- 相关编译选项
  - `MNN_OPENCL` 是否使用OpenCL后端，OpenCL后端可以利用GPU加速
  - `MNN_NNAPI` 是否使用NNAPI后端，NNAPI后端会尝试使用设备上的NPU进行加速
  - `MNN_ARM82`  是否支持fp16推理，开启该编译选项后，在precision设成Precision_Low时，会在支持的设备（ARMv8.2 及以上架构）上启用低精度(fp16)推理，减少内存占用，提升性能
  - `MNN_SUPPORT_BF16`  是否支持bf16推理，开启该编译选项后，在precision设成Precision_Low_BF16 时，会启用bf16推理，减少内存占用，提升性能
- 具体步骤
  1. 在[NDK download](https://developer.android.com/ndk/downloads/)下载安装NDK，建议使用最新稳定版本；
  2. 在 .bashrc 或者 .bash_profile 中设置NDK环境变量，例如：export ANDROID_NDK=/Users/username/path/to/android-ndk-r14b
  3. 执行编译
     -  Android Studio 方式，全平台适用
        - 用 Android Studio 打开`project/android/demo` ，编译`*.apk`
        - 用`unzip`解压编译好的`apk`文件 ，lib目录下包含mnn的`*so`文件
     -  命令行方式，适用 linux / mac 系统
        ```bash
        cd /path/to/MNN
        cd project/android
        # 编译armv7动态库
        mkdir build_32 && cd build_32 && ../build_32.sh
        # 编译armv8动态库
        mkdir build_64 && cd build_64 && ../build_64.sh
        ```
## iOS
- 环境要求
  - xcode
- 相关编译选项
  - `MNN_METAL` 是否使用Metal后端，Metal后端可以利用GPU加速
  - `MNN_COREML`  是否使用CoreML后端，CoreML后端可以利用ANE硬件加速
  - `MNN_ARM82`  是否支持fp16推理，开启该编译选项后，在precision设成Precision_Low时，会在支持的设备（ARMv8.2 及以上架构）上启用低精度(fp16)推理，减少内存占用，提升性能
- 具体步骤
  - 在macOS下，用Xcode打开project/ios/MNN.xcodeproj，点击编译即可
## 其他平台交叉编译
由于交叉编译的目标设备及厂商提供的编译环境类型众多，本文恕无法提供手把手教学。 以下是大致流程，请按照具体场景做相应修改。  
交叉编译大致上分为以下两个步骤，即获取交叉编译器以及配置CMake进行交叉编译。
1. 获取交叉编译工具链
   - 以Linaro工具链为例。首先从[Linaro](https://releases.linaro.org/components/toolchain/binaries/latest-7/)网页中按照宿主机以及交叉编译目标设备来选择合适的工具链。这里我们以`arm-linux-gnueabi`为例，点击网页上的链接，进入[arm-linux-gnueabi](https://releases.linaro.org/components/toolchain/binaries/latest-7/arm-linux-gnueabi/)页面。 按照宿主机类型(这里以X64 Linux为例)选择下载链接, 文件名形如 gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabi.tar.xz 下载后解压到任意目录。
2. 配置交叉编译CMake
   - Toolchain法：对于常用的交叉编译配置，工具链提供方或网络上可能已经有现成的CMake Toolchain。 这种情况下使用如下命令即可:
        ```bash
        mkdir build
        cd build 
        cmake 其他CMake参数 /MNN/源码/路径 -DCMAKE_TOOLCHAIN_FILE=CMake/Toolchain/文件/路径
        ```
   - 手动配置法
        ```bash
        mkdir build && cd build
        cmake .. \
        -DCMAKE_SYSTEM_NAME=宿主系统，例如Linux \
        -DCMAKE_SYSTEM_VERSION=1 \
        -DCMAKE_SYSTEM_PROCESSOR=交叉编译目标处理器的信息。例如armv7或aarch64 \
        -DCMAKE_C_COMPILER=交叉编译器中C编译器的路径 \
        -DCMAKE_CXX_COMPILER=交叉编译器中C++编译器的路径
        ```
3. 以Linaro ARM64为例
   - 下载aarch64交叉编译工具链
        ```bash
        mkdir -p linaro/aarch64
        cd linaro/aarch64
        wget https://releases.linaro.org/components/toolchain/binaries/latest-7/arm-linux-gnueabi/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabi.tar.xz
        tar xvf gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabi.tar.xz
        ```
   - 构建编译
        ```bash
        export cross_compile_toolchain=linaro/aarch64
        mkdir build && cd build
        cmake .. \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_VERSION=1 \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-g++
        make -j4
        ```
