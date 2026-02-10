# NPU 及相应后端使用说明

目前 MNN 支持通过如下后端调用部分手机上的NPU能力：
- QNN
- CoreML
- NNAPI
- HIAI

## QNN

### QNN后端整体介绍

- MNN通过调用QNN SDK的CPP API构建了MNN-QNN后端，以期在能够使用高通NPU的设备上取得推理加速。
- 我们支持了两种运行模式：
  - 在线构图模式，在线编译和序列化QNN计算图。
    - 支持静态形状的常规模型的推理。
  - 离线构图模式则先借助MNN的离线工具缓存QNN计算图的序列化产物，接着在运行时直接读取产物，可以节省初始化时间。
    - 支持静态形状/有限形状组合的常规模型的推理。
    - 可支持部分llm模型的推理加速。

### 准备工作

#### 开发环境
- Host
  - 在线构图模式：无要求。
  - 离线构图模式：一台x86_64，Linux的机器（链路中的部分QNN工具必须在此环境中运行）。
- Device
  - 一台可以使用高通NPU的设备；为便于陈述，下文假设这是一台Android系统的设备。

#### 明确硬件架构

QNN后端的部分使用步骤（如生成离线产物，确定QNN的NPU库依赖等）需要指定device的硬件架构对应的SOC ID以及HEXAGON ARCH。对于一些常见的硬件架构，我们列举如下供你参考：

| 硬件    | SOC ID | HEXAGON ARCH |
| :------ | :----- | :----------- |
| 8 Gen 1 | 36     | 69           |
| 8 Gen 2 | 43     | 73           |
| 8 Gen 3 | 57     | 75           |
| 8 Elite | 69     | 79           |

对于其他的硬件架构，你可以参考高通官网的设备支持列表。

#### 获得QNN依赖

MNN-QNN后端依赖QNN SDK中的`include/QNN`与`lib`，可通过以下步骤获取依赖：
- [注册高通账号](https://myaccount.qualcomm.com/signup)
- 访问Qualcomm AI Engine Direct SDK（即QNN SDK），下载SDK，并解压。比如`/home/xiaying/third/qnn/qairt/2.38.0.250901`
- 修改`~/.bashrc` ，增加SDK路径到环境变量, 然后运行 `source ~/.bashrc` 或者重启终端。eg：

```
export QNN_SDK_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
export QNN_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
export HEXAGON_SDK_ROOT=/home/xiaying/third/qnn/qairt/2.38.0.250901
```

### 在线构图模式，推理常规模型
在线构图模式的使用步骤与其他后端基本一致，主要包含以下三部分。

#### Host，交叉编译Device侧的MNN库及AI应用程序
- 参考[“主库编译”](../compile/engine.md#主库编译)，配置Android系统的编译环境及CMake变量。
- 添加额外的CMake变量并编译：`-DMNN_QNN=ON`、`-DMNN_QNN_CONVERT_MODE=OFF`、`-DMNN_WITH_PLUGIN=OFF`。

#### 推送资源至Device

参考下面的指令，将以下资源推送到Device侧
- AI应用程序。
- 交叉编译得到的Device侧的MNN库。
- QNN库（`libQnnHtp.so`、`libQnnHtpV${HEXAGON_ARCH}Stub.so`、`libQnnHtpV${HEXAGON_ARCH}Skel.so`、`libQnnHtpPrepare.so`）。
- MNN模型。
```
HEXAGON_ARCH="75" # modify this variable according to your environment
MNN_ROOT_PATH="/YOUR/MNN/ROOT/PATH" # modify this variable according to your environment
BUILD_ANDROID_PATH="/your/build/andorid/path" # modify this variable according to your environment
ANDROID_WORKING_DIR="/data/local/tmp" # modify this variable according to your environment

# push mnn libs
cd ${BUILD_ANDROID_PATH}
find . -name "*.so" | while read solib; do
    adb push $solib ${ANDROID_WORKING_DIR}
done
cd -

# push your AI exe
adb push /your/AI/exe ${ANDROID_WORKING_DIR}

# push QNN libs
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV${HEXAGON_ARCH}Stub.so ${ANDROID_WORKING_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v${HEXAGON_ARCH}/unsigned/libQnnHtpV${HEXAGON_ARCH}Skel.so ${ANDROID_WORKING_DIR}
# The following lib is only needed in the online case.
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so ${ANDROID_WORKING_DIR}

# push MNN models
adb push model.mnn ${ANDROID_WORKING_DIR}
```

#### Device，链接并运行
- 链接QNN库
  - 为了动态链接到QNN HTP相关的库，需要在环境变量`ADSP_LIBRARY_PATH`中添加QNN HTP库所在的目录（部分机型上有效）。如果这样也没法成功链接，可将可执行文件，QNN HTP库推送至同一目录，cd到对应目录后，再运行可执行文件，参考如下指令。
```
adb shell "cd ${ANDROID_WORKING_DIR} && export LD_LIBRARY_PATH=.:${ANDROID_LD_LIBRARY_PATH} && export ADSP_LIBRARY_PATH=.:${ANDROID_ADSP_LIBRARY_PATH} && ./your/mnn/qnn/ai/exe"
```
- 配置MNN
  - Backend Type设置为`MNN_FORWARD_NN`，即5。
  - 在使用Module API推理时，需要设定`Module::Config`中的`shapeMutable`字段为`false`。

### 离线构图模式，推理常规模型
相较于在线构图模式，离线构图模式额外包含一次编译（构建生成离线产物需要的MNN库）以及一个模型转换步骤（将原始的MNN模型转化成QNN产物），具体如下。

#### Host，编译生成离线模式产物需要的的MNN库及相应MNN离线工具
- 添加额外的CMake变量并编译：`-DMNN_QNN=ON`、`-DMNN_QNN_CONVERT_MODE=ON`、`-DMNN_WITH_PLUGIN=OFF`、`-DMNN_BUILD_TOOLS=ON`。

#### Host，生成QNN离线构图产物
调用`MNN2QNNModel`工具，针对Device的硬件架构，生成QNN离线产物（`model_${SOC_ID}_${HEXAGON_ARCH}.bin`）以及替代模型（`model_${SOC_ID}_${HEXAGON_ARCH}.mnn`），具体可参考[该工具的用法](../tools/convert.md#mnn2qnnmodel)。

#### Host，交叉编译Device侧的MNN库及AI应用程序
- 参考[“主库编译”](../compile/engine.md#主库编译)，配置Android系统的编译环境及CMake变量。
- 添加额外的CMake变量并编译：`-DMNN_QNN=ON`、`-DMNN_QNN_CONVERT_MODE=OFF`、`-DMNN_WITH_PLUGIN=ON`。

#### 推送资源至Device
与[在线构图模式的情况](#推送资源至device)类似，但有以下两点不同：
- 依赖的QNN库变为`libQnnHtp.so`、`libQnnHtpV${HEXAGON_ARCH}Stub.so`、`libQnnHtpV${HEXAGON_ARCH}Skel.so`、`libQnnSystem.so`（不再依赖`libQnnHtpPrepare.so`，而是依赖`libQnnSystem.so`）。
- 不再使用原始的MNN模型，而是需要QNN离线产物（`model_${SOC_ID}_${HEXAGON_ARCH}.bin`）以及替代模型（`model_${SOC_ID}_${HEXAGON_ARCH}.mnn`）。

#### Device，链接并运行
- 配置MNN
    - 指定backend type为0（CPU）。读取并推理QNN离线产物的功能被封装在Plugin算子内，该算子被注册在CPU后端，因此，此时需要指定backend type为CPU。
    - 在Device侧，如果你的离线产物和你的应用的工作目录不一致，那么你需要在程序中通过`Executor::RuntimeManager::setExternalPath`接口设定离线产物所在的目录。
- 链接QNN库
    - 离线构图模式对于链接的要求和在线构图模式一致。


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
