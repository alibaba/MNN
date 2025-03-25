# NPU 及相应后端使用说明

目前 MNN 支持通过如下后端调用部分手机上的NPU能力：
- CoreML
- NNAPI
- HIAI

目前NPU相关后端均不支持可变形状、控制流等动态模型，算子数相比CPU/GPU支持要少，建议根据NPU是否能跑通，反复调整模型结构。

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