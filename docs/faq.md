# 常见问题与解答
## 热点问题
- [模型转换后结果与其他框架不一致](faq.html#id8)
- [compute shape error](faq.html#compute-shape-error-for-xxx)
- [模型转换时有Error信息](faq.html#reshape-error)
- [模型量化后为什么比浮点慢](faq.html#id14)
- [输入输出的elementSize与实际有区别](faq.html#tensor-elementsize)
- [MNN模型如何加密](faq.html#id18)

## 编译相关
### 环境需求
cmake 3.10+
gcc 4.9+
protobuf 3.0+

**更新 gcc 之后请重新 cmake 一下**

### schema/generate.sh 相关问题

```shell
*** building flatc ***
CMake Error: Could not find CMAKE_ROOT !!!
```

这说明你的 CMake 没有正确安装，请尝试
`sudo apt install extra-cmake-modules`
或
`export CMAKE_ROOT=/path/to/where_cmake_installed`

更改 **schema 之后，需要重新运行 schema/generate.sh**

### 找不到 Protobuf
触发问题操作

- 编译MNN模型转换器 (-DMNN_BUILD_CONVERTER=ON)
- tools/script/get_model.sh

报错信息类似：
```shell
Could NOT find Protobuf (missing: Protobuf_INCLUDE_DIR)
```

```shell
Unrecognized syntax identifier "proto3".  This parser only recognizes "proto2".
```

有两种解决方案
#### （建议）使用 MNN 自带的 Protobuf
cmake 加上选项 -DMNN_BUILD_PROTOBUFFER=ON ，使用 MNN 自带的 protobuf 编译
采用这种方案时，如果之前有编译残留，有可能出现原先生成的 protobuf 相关头文件与 MNN 自带的 protobuf 库不兼容的问题（编译出错），清空当前编译目录，重新编译即可。

#### （可选）安装 / 配置 Protobuf
参考 [Protobuf's Installation Instructions](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) 安装.
如果您电脑上安装了多份 protobuffer ，他们之前可能产生冲突（protoc 与链接的 libprotobuf 不一致），可尝试如下方式解决:
```shell
which protoc
# comment the output path in .bashrc if it do NOT direct to the correct protoc.
source .bashrc
sudo ldconfig
```
或
```shell
# uninstall
sudo apt-get remove libprotobuf-dev
sudo apt-get remove protobuf-compiler
sudo apt-get remove python-protobuf
sudo rm -rf /usr/local/bin/protoc
sudo rm -rf /usr/bin/protoc
sudo rm -rf /usr/local/include/google
sudo rm -rf /usr/local/include/protobuf*
sudo rm -rf /usr/include/google
sudo rm -rf /usr/include/protobuf*
# install
sudo apt-get update
sudo ldconfig
sudo apt-get install libprotobuf* protobuf-compiler python-protobuf
```

### MNN静态库的使用
MNN 一般以动态库形式使用，里面有大量自注册函数，如果需要以静态库形式使用 MNN ，需要根据您所用的编译器，增加完全链接的编译选项：

- GCC: -Wl,--whole-archive MNN -Wl,--no-whole-archive
- OSX(Xcode): -Wl,-force_load MNN
- Window(Visio-Studio): /WHOLEARCHIVE:MNN


## 模型转换
### Error for binary op: input0's type != input1's type
此处报错是 Binary 算子的形状计算出错。

- MNN 在模型转换时会尝试进行图优化。
- 图优化过程中部分Pass需要输入Tensor 的形状信息，触发形状计算。
- 对于存在控制流的模型，子图的输入是未知类型，一般设成 float ，此时有可能出现 Binary 两侧 Tensor type 不一致的问题
- 此报错不影响模型的实际使用

### Reshape error

- 在模型未指定输入大小时，MNN 模型转换时的图优化中计算 shape 可能打印该错误
- 一般不影响模型转换结果，看最后是否 convert success 即可

### 不支持算子
```shell
opConverter ==> MNN Converter NOT_SUPPORTED_OP: [ ANY_OP_NAME ]
```

说明存在 MNN 不支持转换的算子，可以考虑如下解决方案：

- 若原始模型为 tflite / caffe（含自定义算子） ， 改成 MNN 支持较好的 Tensorflow pb 格式导出或转成 Onnx ，再转 MNN
- 提 Issue 等待我们支持，并关注 MNN 的更新
- 参考[自定义算子](./contribute/op.md) 

### 模型转换后与原框架结果不一致
先使用MNN中的模型一致性验证脚本进行测试，确定不是调用方法或其他错误，[使用方法](./tools/convert.html#id3)

## Pymnn
### import MNN 出现 import numpy error
临时解决方案：升级 numpy 版本到 1.20.0 或以上

## 运行问题
### 运行结果出错
- 先使用 testMNNFromOnnx.py 等测试工具进行测试，具体参见模型转换工具的正确性校验部分
- 测试工具验证正确，但运行代码结果出错，可能是如下原因：
    1. 使用 Session API 运行不满足运行条件的模型，此时应换用 Module API
    2. 输入的内存布局不对
    3. 输入数据格式不对，int64 需要换成 int32_t ，double 需要换成 float


### 布局转换问题(Tensor 的 elementSize 不为各维度乘积)
MNN 内部对 CV 相关算子采用 NC4HW4 布局，计算 elementSize 时，channel 会上对齐到 4 返回，此内存布局允许实现的硬件自行确定内存排列方式，具体方式不对用户可见，但用户可以通过如下代码，输入或获取自己指定的NCHW / NHWC 内存布局的 Tensor / VARP。

#### Interpreter-Session API
```cpp
auto srcTensor = net->getSessionInput(session, "data");
auto srcTensorHost = new Tensor(srcTensor, Tensor::TENSORFLOW);
// ... set srcTensorHost data
srcTensor->copyFromHostTensor(srcTensorHost);
delete srcTensorHost;
// ... set other inputs, if exist
net->runSession(session);

auto dstTensor = net->getSessionOutput(session, "prob"); 
auto dstTensorHost = new Tensor(dstTensor, Tensor::TENSORFLOW);
dstTensor->copyToHostTensor(dstTensorHost);
// ... use dstTensorHost data
delete dstTensorHost;
```

#### Module API
```
Module* net = Module::load("net.mnn", {"data"}, {"prob"});

VARP input = _Input({1, 224, 224, 3}, NHWC);
float* inputPtr = input->writeMap<float>();
// ... set srcTensor data
auto info = net->getInfo();
input = _Convert(input, info->inputs[0].order);
output = net->onForward({input});
output = _Convert(output, NHWC);

const float* outputPtr = output->readMap<float>();
// ... use outputPtr
```
### compute shape error for XXX

- 输入形状不正确
- MNN 推理过程分形状计算-几何计算-内容计算三步，前两步在 resizeSession 中完成，在 createSession 时，会用初始设定的输入大小进行一次 resizeSession ，若初始 shape 设定不对，则会在某个算子报 shape 计算的 error ，重新设置输入 tensor 的大小并 resizeSession 即可
- 在导出 Onnx 时，shape 没设成 dynamic ，导致部分参数写死，变动大小后无法 resize 网络
- 如果确定输入形状正确，并且执行了`resizeTensor`和`resizeSession`；可以打开`source/shape/SizeComputer.cpp`中的宏`// #define MNN_DEBUG_TENSOR_SIZE`定义，然后执行模型推理；打开宏之后可以看到每一层的形状信息，可以逐层进行Debug

### Android 设备无法查看日志
Android 系统有两类打印日志的方式: printf 和 logcat. 默认 MNN 的编译脚本使用 printf，这样方便在命令行中调试。集成到 App 上时，用 cmake  -DMNN_USE_LOGCAT=ON 将打印日志的方式改成 logcat 即可用 adb logcat 查看
### 
### 如何增加 opencl so 地址?
MNN opencl 后端默认采用 dlopen 的方式动态打开设备的 opencl 驱动，相应位置若找不到您设备上的驱动，请修改 **OpenCLWrapper.cpp**
### 
### TensorArray Op 与 Switch / Merge 控制流支持
TensorArray 和控制流支持需要借助 MNN-Express ，
请参考 demo/exec/transformerDemo.cpp 的接口使用

### 如何获取网络中间结果
默认情况下， MNN 只支持用户访问网络输入输出节点的数据，如果需要取中间结果，参考如下方式：

- Interpreter - Session API
   1. 将需要的中间结果的 tensor 名字加到 config.saveTensors ，然后用这个 config 创建 session.
   1. 在 MNN 的运行过程中插入回调函数，即用 runSessionWithCallBack, 参考 tools/cpp/MNNV2Basic.cpp
- Express - Module API
   - 加载网络时，把需要获取的中间结果加到 output name 中


### OpenCL 或 Vulkan 后端无法使用
Linux系统上的简单解决方案:
cmake .. -DMNN_USE_SYSTEM_LIB=true -DMNN_SEP_BUILD=false

Windows 系统上参考 MNN 静态库的使用，需要加静态库全链接选项

#### 无法找到系统库
为了设备兼容性，MNN Vulkan / OpenCL 默认采用自己搜索路径 dlopen 的方式，不直接依赖系统驱动。您也可以设置 MNN_USE_SYSTEM_LIB = ON , 让 MNN 直接依赖系统驱动
#### 找不到后端 (Can't Find type=3 backend)
OpenCL / Vulkan 采用静态变量自注册的方式往 MNN 主库注册后端. 在 Linux 系统上默认采用懒加载，由于没有直接依赖 MNN_CL.so / MNN_Vulkan.so ，不会初始化静态变量，导致 opencl / vulkan 后端使用时找不到. 可以按如下方式之一解决:

1. 设置 MNN_SEP_BUILD = OFF  （cmake -DMNN_SEP_BUILD=OFF）.  把 opencl / vulkan 后端统一编入 MNN 的 so.
1. 自己在使用的代码中加上 dlopen("libMNN_CL.so") . 参考 [https://github.com/alibaba/MNN/issues/105](https://github.com/alibaba/MNN/issues/105) .

#### Android App 上因权限问题打不开 OpenCL 库
由于Android新版本增强了权限控制，有可能遇到加载OpenCL库失败的问题，可以修改 AndroidManifest.xml 对应栏，加入OpenCL相关 so 的权限需求

```
<application>
        ...

        <uses-native-library android:name="libOpenCL.so"
            android:required="true"/>

        ...

</>
```

### 部分模型用 MNNV2Basic 运行出现段错误，或报 Interpreter don't support case for shape compute need input content, please use module api instead

- 模型不满足运行条件
   - MNNV2Basic 使用  Interpreter + Session 方式运行，此类运行方式要求模型满足一定条件，否则无法运行模型或产生特别的 crash ，条件如下：
      - 模型中所有Tensor的形状可以在输入Tensor形状确定后，预先计算而得
      - 模型中没有子图或其他控制流相关算子
   - 不满足运行条件的模型可以借助 MNN_Express 运行，参考示例代码：
      - demo/exec/transformerDemo.cpp
      - tools/cpp/ModuleBasic.cpp
- MNN 内部算子实现逻辑错误，此概率较小，遇到可提 issue 反馈

### 使用 GPU 时的内存访问问题

- 输入输出指针为空/段错误
   - 一般是直接访问了 tensor 的 host
   - 按 [输入数据](./inference/session.html#id8) 和[获取输出](./inference/session.html#id21) 里面的方式建host tensor 并 copy ，参考相关文档修改使用代码
- 是否可基于 deviceId 直接传 GPU 地址？
   - 可以，可以通过setDevicePtr设置输入VARP的GPU地址,通过copyToDevicePtr设置输出VARP拷贝到的GPU地址
      - 相关使用参考tools/cpp/GpuInterTest.cpp
      - 目前OPENCL推理支持OPENCL/OPENGL内存做输入输出。CUDA推理支持CUDA内存做输入输出
   - 采用 MNN_Express 系列接口，可以支持模型之间的内存直接传递不做拷贝

### 多卡GPU上，用户指定特定GPU做推理问题

- 通过设置MNNDeviceContext结构体参数来指定特定GPU
   - 通过设置platformSize、platformId、deviceId参数来进行指定
   - 目前支持OpenCL和CUDA后端进行设置
   - 具体可以参考：tools/cpp/testModel.cpp

### Register 相关内存泄露说明
用 valgrind 工具检查时会报 MNN Register 相关的内存泄露，这个属于一次性的初始化内存，后续也不会增加，可视为误报


## 性能相关
### 使用 GPU 时，调用 copyToHostTensor / copyFromHostTensor 非常慢
GPU 后端调用 copy 的时间包含两个部分

- 异构数据拷贝
- 等待数据相关的GPU计算完成

对 GPU 后端而言，在数据被要求对用户可见（比如复制 output tensor 数据出来）之前，是允许异步执行的。
在数据被用户要求可见之时，会等待相应的异步操作完成。
因此有可能 复制 output tensor 的过程包括了等待 GPU 算子异步执行完成，导致缓慢。
### GPU 为什么比 CPU 跑得慢？
有如下原因： 

1. 相当一部分移动端设备 (如 pre-iPhone 8), GPU 算力不足，加以内存带宽的限制，本身不如 CPU.

     比如 Apple 由 Imagination 切换到自己的 GPU in iPhone 8, 导致 GPU 性能下降（不如 iphone 7） ，相对地， CPU 是提升的.

2. 存在 GPU 不支持的算子，这些算子会切换到 CPU 执行，相应的输入输出需要 CPU - GPU 之间的内存拷贝，产生额外耗时
2. 模型本身计算量小或者不易并行，发挥不了 GPU 并行计算的优势.
2. GPU 被其他程序占用，或者系统限制了 GPU 频率 
### 模型量化后为什么比浮点慢
这个需要从浮点和量化两个方面说明

- 浮点性能是 MNN 优化的第一优先级
   - MNN 在浮点各架构（x86/x64/ARM）上均做了深入的优化，基本都达到了设备的理想性能。
   - 在浮点计算上，MNN 采用了 Winograd 卷积/ Strassen 矩阵乘等降低计算量的算法，对于对称卷积 (2x2 , 3x3, 4x4, 5x5, 6x6, 7x7)，Winograd 算法有成倍数的性能优化效果。
- 量化性能受多个因素影响
   - 不同架构上，量化计算性能与浮点计算性能相比有快有慢。
   - 模型量化后，由于部分算子不支持量化，出现回退到浮点计算的情况，交接处产生额外转换耗时。
   - 浮点计算的 Winorad 算法/Strassen 算法未应用于量化计算，相应的性能优化效果量化后不支持。
- 架构说明：
   - x86 / x64 架构下，无 vnni 指令，量化计算需要先从 int8 转到 int16 ，乘加到 int32 ，本身计算效率不如浮点直接乘加到 fp32 上快。
   - x64 + vnni 指令，量化计算有 sdot 指令，明显快于 FP32 ，编译 MNN 时需要开启 MNN_AVX512 以支持这个指令，一般相比 AVX512 的浮点运算快 30%
   - ARM v7a / ARMv8 ：量化计算采用 int8 乘加到 int16，再双加到 int32 的方式，计算效率略快于浮点（一般 30% 左右提升）。
   - ARMv8.2 架构有 sdot 指令，但同时 FP32 相对之前架构发射数也提升了一倍，也支持了比 FP32 快一倍的 FP16 向量计算指令，MNN 会检查设备架构以开启 sdot / smmla ，理想情况下量化计算性能比 FP32 快1倍以上，比 FP16 快 20%。

## 其他问题
### MNN模型如何加密
加密与破解是攻防的较量，端侧加密很难做到绝对安全。
可以通过构造独有的模型格式来增加反向的难度，按照以下步骤操作可以得到独特的模型格式：
1. 针对`schema/default/*.fbs`下的文件，对参数顺序，枚举类顺序进行重新排序；比如：可以重新调整`MNN.fbs`中`OpType`的顺序；重新调整`CaffeOp.fbs`中`Convolution2DCommon`成员变量的顺序；
2. 执行`schema/generate.sh`重新生成`flatbuffers`头文件；
3. 重新编译`MNN`库文件， `Convert`等所有工具；
4. 使用新的工具重新转换模型；
5. 在端侧使用新模型和新的`MNN`库文件进行部署；
