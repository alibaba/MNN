# 主库编译
默认编译产物为：`libMNN.so`，`express/libMNN_Express.so`
## Linux/MacOS
### 环境要求
  - cmake >= 3.10
  - gcc >= 4.9 或者使用 clang
### 相关编译选项
  - `MNN_AVX512` 是否使用AVX512指令，需要gcc9以上版本编译
  - `MNN_OPENCL` 是否使用OpenCL后端，针对GPU设备
  - `MNN_METAL` 是否使用Metal后端，针对MacOS/iOSGPU设备
  - `MNN_VULKAN` 是否使用Vulkan后端，针对GPU设备
  - `MNN_CUDA`  是否使用CUDA后端，针对Nivida GPU设备
  - 其他编译选项可自行查看 CMakeLists.txt
### 具体步骤
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
### Mac M1 上编译
- Mac M1 较为特殊的一点是作为过渡期间的芯片支持Arm/x64双架构，一般需要额外指定来获取需要的架构
- 在 cmake 步骤增加 `-DCMAKE_OSX_ARCHITECTURES=arm64` 可以编译出 Arm 架构的库，对应地编译 x64 架构时加 `-DCMAKE_OSX_ARCHITECTURES=x86_64`:

```
cd /path/to/MNN
mkdir build && cd build && cmake .. -DCMAKE_OSX_ARCHITECTURES=arm64 && make -j8
```

## Windows(非ARM架构)
- 环境要求
  - Microsoft Visual Studio >= 2017
  - cmake >= 3.13
  - Ninja
- 相关编译选项
  - 同`Linux/MacOS`
- 具体步骤
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

## Windows(ARM架构)
- 环境要求
  - Microsoft Visual Studio >= 2017
  - cmake >= 3.13
  - Ninja
  - Clang
    - Clang 安装参考: https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170#install-1
- 相关编译选项
  - 同`Linux/MacOS`
- 具体步骤
  - 打开vs的ARM64命令行工具
  - 进入 MNN 根目录
  - mkdir build && cd build
  - cmake .. -G Ninja -DCMAKE_C_COMPILER="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\ARM64\bin\clang.exe" -DCMAKE_CXX_COMPILER="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\ARM64\bin\clang++.exe"  -DCMAKE_LINKER="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\ARM64\bin\lld.exe" -DCMAKE_BUILD_TYPE=Release
    - Visual Studio 安装路径不一致的，可自行修改脚本
  - ninja -j16

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
可基于脚本编译或者基于xcode工程编译

- 环境要求
  - xcode
  - cmake
- 相关编译选项
  - `MNN_METAL` 是否使用Metal后端，Metal后端可以利用GPU加速
  - `MNN_COREML`  是否使用CoreML后端，CoreML后端可以利用ANE硬件加速
  - `MNN_ARM82`  是否支持fp16推理，开启该编译选项后，在precision设成Precision_Low时，会在支持的设备（ARMv8.2 及以上架构）上启用低精度(fp16)推理，减少内存占用，提升性能

- 基于 xcode 编译：用Xcode打开project/ios/MNN.xcodeproj，点击编译即可，工程中默认打开上述所有编译选项

- 基于脚本编译：运行脚本并开启`MNN_ARM82`选项
```
sh package_scripts/ios/buildiOS.sh "-DMNN_ARM82=true"
```

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

## Web

- 可以把 MNN 源代码编译为 WebAssembly 以便在浏览器中使用

### 安装 emcc
参考 https://emscripten.org/docs/getting_started/downloads.html ，安装完成后并激活，此时可使用 emcmake

### 编译（通用）
- 使用 emcmake cmake 替代 cmake ，然后 make 即可: 
```
mkdir build
cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DMNN_FORBID_MULTI_THREAD=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_USE_SSE=OFF
emmake make MNN -j16
```

编译完成后产出 libMNN.a ，可在后续的 webassembly 程序中链接，链接时一般要添加 -s ALLOW_MEMORY_GROWTH=1 ，避免内存不足后 crash

### SIMD 支持

- 如果确认目标设备支持Web Simd ，在cmake时加上 -msimd128 -msse4.1 ，可以较大提升性能，eg: 
```
mkdir build
cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_TEST=true -DCMAKE_CXX_FLAGS="-msimd128 -msse4.1" -DMNN_FORBID_MULTI_THREAD=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_USE_SSE=ON
emmake make MNN -j16
```

### 测试
由于Web上文件系统不一致，建议只编译run_test.out运行，其他测试工具需要加上--preload-file {dir} 

- 编译示例

```
mkdir build
cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_TEST=true -DCMAKE_CXX_FLAGS="-msimd128 -msse4.1 -s ALLOW_MEMORY_GROWTH=1" -DMNN_FORBID_MULTI_THREAD=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_USE_SSE=ON
emmake make -j16
```

- 运行
```
node run_test.out.js speed/MatMulBConst   //测试性能
node run_test.out.js  //测试功能
```
