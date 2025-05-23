# NPU 及相应后端使用说明

目前 MNN 支持通过如下后端调用部分手机上的NPU能力：
- QNN
- CoreML
- NNAPI
- HIAI

目前NPU相关后端均不支持可变形状、控制流等动态模型，算子数相比CPU/GPU支持要少，建议根据NPU是否能跑通，反复调整模型结构。

同时，由于QNN、CoreML与NNAPI在MNN中共用同一个Backend Type，这三个后端对应的编译宏MNN_QNN、MNN_COREML、MNN_NNAPI在编译时，至多只能打开一个。

## QNN
适用于使用高通芯片且配备高通Hexagon张量处理器（Hexagon Tensor Processor，HTP）的机型，可参考[高通官网的设备支持列表](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices)。

### 获得QNN依赖
QNN后端依赖QNN SDK中的`/include/QNN`与`lib`，首先，我们需要获得相关依赖。
- [注册高通账号](https://myaccount.qualcomm.com/signup)
- 访问Qualcomm AI Engine Direct SDK（即QNN SDK）[官网](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)，下载SDK。
- 参考以下指令，将下载的sdk中的`/include/QNN`与`lib`拷贝到MNN源码中的对应位置。
```
QNN_SDK_ROOT="/YOUR/QNN/SDK/PATH" # modify this variable according to your environment
MNN_ROOT="/YOUR/MNN/PATH" # modify this variable according to your environment
INCLUDE_SRC="${QNN_SDK_ROOT}/include/QNN"
LIB_SRC="${QNN_SDK_ROOT}/lib"
INCLUDE_DEST="${MNN_ROOT}/source/backend/qnn/3rdParty/include"
LIB_DEST="${MNN_ROOT}/source/backend/qnn/3rdParty/lib"
mkdir "${MNN_ROOT}/source/backend/qnn/3rdParty"
cp -r ${INCLUDE_SRC} ${INCLUDE_DEST}
cp -r ${LIB_SRC} ${LIB_DEST}
```

### QNN后端编译
编译 MNN 时打开编译宏`MNN_QNN`，即`-DMNN_QNN=ON`。

### QNN后端运行
- Backend Type设置为`MNN_FORWARD_NN`，即 5 。
- 除MNN相关的库之外，QNN后端在运行时还依赖四个QNN库，可参考以下指令，将QNN中的库拷贝到设备中。其中变量`HEXAGON_ARCH`需要与你的目标机型匹配，可参考[高通官网的设备支持列表](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices)，如8gen3的设备，需要设定`HEXAGON_ARCH="75"`。
```
HEXAGON_ARCH="75" # modify this variable according to your environment
MNN_ROOT="/YOUR/MNN/PATH" # modify this variable according to your environment
ANDROID_PATH="/data/local/tmp"
adb push ${MNN_ROOT}/source/backend/qnn/3rdParty/lib/aarch64-android/libQnnHtp.so ${ANDROID_PATH}/libQnnHtp.so
adb push ${MNN_ROOT}/source/backend/qnn/3rdParty/lib/aarch64-android/libQnnHtpPrepare.so ${ANDROID_PATH}/libQnnHtpPrepare.so
adb push ${MNN_ROOT}/source/backend/qnn/3rdParty/lib/aarch64-android/libQnnHtpV${HEXAGON_ARCH}Stub.so ${ANDROID_PATH}/libQnnHtpV${HEXAGON_ARCH}Stub.so
adb push ${MNN_ROOT}/source/backend/qnn/3rdParty/lib/hexagon-v${HEXAGON_ARCH}/unsigned/libQnnHtpV${HEXAGON_ARCH}Skel.so ${ANDROID_PATH}/libQnnHtpV${HEXAGON_ARCH}Skel.so
```
- 为了动态链接到QNN HTP相关的库，需要在环境变量`ADSP_LIBRARY_PATH`中添加QNN HTP库所在的目录（部分机型上有效）。如果这样也没法成功链接，可将可执行文件push到QNN HTP库所在目录（如`/data/local/tmp`），cd到对应目录后，再运行可执行文件，参考如下指令。
```
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp ./MyExe.out"
```

## CoreML
适用于 Mac / iOS / iPad

### CoreML 后端编译
1. 编译 MNN 时打开编译宏 MNN_COREML ：-DMNN_COREML=ON
2. 编译App / 可执行程序时，增加链接 CoreML.framework

### CoreML 后端使用
backend type设置成：MNN_FORWARD_NN

## NNAPI
适用于 Android 系统，高通/联发科芯片

### NNAPI 后端编译
打开编译宏 MNN_NNAPI 即可
```
cd ${MNN}
cd project/android
mkdir build && cd build
../build_64.sh -DMNN_USE_LOGCAT=ON -DMNN_NNAPI=ON
``` 

### NNAPI 后端使用
backend type设置成：MNN_FORWARD_NN


## 华为 HIAI 
适用于 Android 系统， Kirlin芯片

### HIAI 环境准备
1. 从如下链接下载 DDK 
https://developer.huawei.com/consumer/cn/doc/hiai-Library/ddk-download-0000001053590180



2. 拷贝相对应的so和include文件到 hiai/3rdParty 目录下，如果没有3rdParty目录，新建一个：

```
mkdir ${MNN}/source/backend/hiai/3rdParty
cp -r ${DDK}/lib ${MNN}/source/backend/hiai/3rdParty/armeabi-v7a
cp -r ${DDK}/lib64 ${MNN}/source/backend/hiai/3rdParty/arm64-v8a
cp -r ${DDK}/include ${MNN}/source/backend/hiai/3rdParty/include
```

### HIAI 编译执行
1. cmake 参数打开npu开关： -DMNN_NPU=true 
2. backend type设置成：MNN_FORWARD_USER_0
3. 执行可执行程序（需动态加载：libMNN_NPU.so, libhiai_ir_build.so, libhiai_ir.so, libhiai.so）