# Pymnn - 快速开始

## 将训练好的模型转换为MNN支持的格式
### 模型导出
+ 使用原始训练框架自带的导出包，把模型导出为 onnx / pb / tflite 模型格式
+ Pytorch 导出 onnx参考官网： [https://pytorch.org/docs/stable/onnx.html](https://pytorch.org/docs/stable/onnx.html)

```python
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

model = MyModel()

torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "my_model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=False             # True or False to select the exporter to use
)
```



### 转换到MNN
+ 使用 MNN 的模型转换工具 MNNConvert （Pymnn 中为 mnnconvert）
    - 安装 pymnn
        * pip install MNN
    - 转换

```python
mnnconvert -f ONNX --modelFile user.onnx --MNNModel user.mnn
mnnconvert -f TF --modelFile user.pb --MNNModel user.mnn
mnnconvert -f TFLITE --modelFile user.tflite --MNNModel user.mnn
```

- 若 `pip install MNN` 失败（可能是当前系统和python版本不支持），可以按如下方式步骤编译安装pymnn

```
cd pymnn/pip_package
python3 build_deps.py llm
python3 setup.py install
```

## 使用MNN引擎，加载模型推理
### 预处理
+ 对用户输入数据进行预处理，转换为深度学习模型所需的张量
    - 可使用 MNN.cv / MNN.numpy

```python
import MNN
import MNN.numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
print(x)

y = x * 2
print(y)

import MNN.cv as cv2
image = cv2.imread("cat.jpg")
image = cv2.resize(image, (224, 224))

image = image[..., ::-1]

print(image.mean([0, 1]))
```



打印输出：

```python
The device supports: i8sdot:1, fp16:1, i8mm: 0, sve2: 0
array([[1., 2.],
       [3., 4.]], dtype=float32)
array([[2., 4.],
       [6., 8.]], dtype=float32)
array([125.76422 , 135.97044 ,  85.656685], dtype=float32)
```



MNN.cv 和 MNN.numpy 产出的数据结构即为 MNN 推理所需要的张量 MNN.expr.VARP

### 加载模型/推理
+ 使用 MNN.nn 加载模型
+ 将输入张量传入MNN进行推理，得到输出张量
    - 输入输出均为数组

```python
# x0, x1 是输入，y0, y1 是输出

net = MNN.nn.load_module_from_file("user.mnn", ["x0", "x1"], ["y0", "y1"])
y = net.forward([x0, x1])
y0 = y[0]
y1 = y[1]
```

### 后处理
+ 对模型输出的张量进行后处理，转换为用户的需要的输出数据
    - 可使用 MNN.numpy 等进行计算
+ MNN.expr.VARP 信息读取
    - 可用 shape 获取形状
    - 可用 read_as_tuple 转成 python 的 tuple 类型

示例代码：

```python
import MNN
import MNN.numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
print(x)
print(x.shape)

y = x * 2
print(y.shape)
print(y.read_as_tuple())

```



输出结果

```python
The device supports: i8sdot:1, fp16:1, i8mm: 0, sve2: 0
array([[1., 2.],
       [3., 4.]], dtype=float32)
[2, 2]
[[1. 2.]
 [3. 4.]]
array([[2., 4.],
       [6., 8.]], dtype=float32)
(2.0, 4.0, 6.0, 8.0)
```



### 完整代码示例
```python
from __future__ import print_function
import MNN.numpy as np
import MNN
import MNN.cv as cv2
import sys

def inference(net, imgpath):
    """ inference mobilenet_v1 using a specific picture """
    # 预处理
    image = cv2.imread(imgpath)
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224))
    #resize to mobile_net tensor size
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    #change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    #Make var to save numpy; [h, w, c] -> [n, h, w, c]
    input_var = np.expand_dims(image, [0])
    #cv2 read shape is NHWC, Module's need is NC4HW4, convert it
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    #inference
    output_var = net.forward([input_var])
    # 后处理及使用
    predict = np.argmax(output_var[0])
    print("expect 983")
    print("output belong to class: {}".format(predict))

# 模型加载
net = MNN.nn.load_module_from_file(sys.argv[1], ["image"], ["prob"])
inference(net, sys.argv[2])
 
```

+ MNN 官方文档：[https://mnn-docs.readthedocs.io/en/latest/start/overall.html](https://mnn-docs.readthedocs.io/en/latest/start/overall.html)
+ 开源地址：[https://github.com/alibaba/MNN/](https://github.com/alibaba/MNN/)
    - 内外版本使用方法一致，差别在于模型格式不同，外部模型不能为内部版本使用，反之亦然
    - 具体见 [https://aliyuque.antfin.com/tqwzle/qg2ac8/isq7hl](https://aliyuque.antfin.com/tqwzle/qg2ac8/isq7hl)

# 模型转换进阶
## Onnx 导出细节
### 控制流
+ 如果需要导出控制流（if / for 之类），先 trace 再 导出

```python
import torch
class IfModel(torch.nn.Module):
    def forward(self, x, m):
        if torch.greater(m, torch.zeros((), dtype=torch.int32)):
            return x * x
        return x + x;

model = torch.jit.script(IfModel())
inputs = torch.randn(16)
mask = torch.ones((), dtype=torch.int32)

out = model(inputs, mask)
torch.onnx.export(model, (inputs, mask), 'if.onnx')

```

### 输入维度可变
+ 输入的维度在使用时会发生改变的，导出时增加 dynamic_axes ，指定可变的维度

```python
import torch
class IfModel(torch.nn.Module):
    def forward(self, x, m):
        if torch.greater(m, torch.zeros((), dtype=torch.int32)):
            return x * x
        return x + x;

model = torch.jit.script(IfModel())
inputs = torch.randn((1, 3, 16, 16))
mask = torch.ones((), dtype=torch.int32)

out = model(inputs, mask)
dynamicaxes = {}
dynamicaxes['inputs'] = {
    0:"batch",
    2:"height",
    3:"width"
}
torch.onnx.export(model, (inputs, mask), 
                  'if.onnx',
                  input_names=['inputs','mask'],
                  output_names=['outputs'],
                  dynamic_axes = dynamicaxes)
```

+ Onnx 的输入维度是否可变决定转换到 MNN 之后，输入维度是否可变

## MNN模型转换
可以用如下命令查看模型转换中的参数

```python
mnnconvert -h
```

### 查看信息/校验
+ 查看MNN支持的算子

```python
mnnconvert -f TF --OP
mnnconvert -f ONNX --OP
mnnconvert -f TFLITE --OP
```

+ 校验，参考文档：[https://mnn-docs.readthedocs.io/en/latest/tools/convert.html#id3](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html#id3)
    - 相关脚本可以下载开源版本的 MNN 



+ 如下命令可以查看 mnn 基础信息

```python
mnnconvert -f MNN --modelFile user.mnn --info
```

打印结果示例：

```python
The device supports: i8sdot:1, fp16:1, i8mm: 0, sve2: 0
Model default dimensionFormat is NCHW
Model Inputs:
[ patches ]: dimensionFormat: NCHW, size: [ 1,1176 ], type is float
[ position_ids ]: dimensionFormat: NCHW, size: [ 2,-1 ], type is int32
[ attention_mask ]: dimensionFormat: NCHW, size: [ 1,-1,-1 ], type is float
Model Outputs:
[ image_embeds ]
Model Version: 3.0.4
```



+ <font style="color:#000000;">使用 JsonFile 选项可以将 MNN 文件转换成可查看和编辑的 Json 文件</font>

```python
mnnconvert -f MNN --modelFile user.mnn --JsonFile user.json
```

+ <font style="color:#000000;">如下命令可将Json文件转换成mnn</font>

```python
mnnconvert -f JSON --modelFile user.json --MNNModel user.mnn
```

### 权重量化
模型压缩详细可参考 [https://mnn-docs.readthedocs.io/en/latest/tools/compress.html](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html) ，最常用的是权重量化功能

+ weightQuantBits ：权重量化的Bits数，范围2-8 ，压缩率为 Bits数 / 32 ，选8即为原先的1/4大小。一般选用8，较大的模型，结合 weightQuantBlock 也可以考虑选用 4

```python
mnnconvert -f ONNX --modelFile user.onnx --MNNModel user.mnn --weightQuantBits=8
```

+ weightQuantBlock ：如果模型量化后精度下降较多，可以增加这个参数，以增加线性映射表的密度，一般设 128 或 32 。每weightQuantBlock个weight产出一个线性映射关系（scale/bias），weightQuantBlock越小，线性映射表越大，对应的模型越大，不能低于32 。

```python
mnnconvert -f ONNX --modelFile user.onnx --MNNModel user.mnn --weightQuantBits=8 --weightQuantBlock=128
```


### 权重分离
+ 增加 saveExternalData 参数，将模型转换为模型+权重文件

```python
mnnconvert -f ONNX --modelFile user.onnx --MNNModel user.mnn --weightQuantBits=8 --weightQuantBlock=128 --saveExternalData
```

此时产出两个文件：user.mnn 和 user.mnn.weight

+ 使用时，user.mnn.weight 和 user.mnn 放在同一路径下，MNN 会依据路径自动加载权重文件
+ 优点：降低模型加载后的峰值内存和运行时内存

# 运行时后端配置
## 后端配置
### 计算单元
手机上有 CPU / GPU / NPU 三类计算单元，根据自身需求决定使用硬件的方案。

| | CPU | GPU | NPU |
| --- | --- | --- | --- |
| 可用模型 | 全部模型 | 几乎全部模型 | CV模型 |
| 可变形状/控制流 | 支持 | 支持但有性能损失 | 不支持 |
| 加载时间 | 短 | 中 / 长 | 中 |
| 算力 | 中  | 中 | 高 |
| 功耗 | 高 | 中 | 低 |


+ CPU：主流方案，适用于加载后仅运行少数次的模型以及小模型
+ GPU：适用于加载后多次运行的中大计算量的模型
+ NPU：适用于模型计算量大，功耗高，需要结合模型结构设计进行重点优化的场景



### Forwardtype
通过设置 forwardtype ，可通过 MNN 启用相应的计算单元，具体如下

| 枚举值 | 枚举名 | 计算单元 | 适用设备 |
| --- | --- | --- | --- |
| 0 | <font style="color:rgba(0, 0, 0, 0.85);background-color:#ffffff;">MNN_FORWARD_CPU</font> | CPU | 通用 |
| 1 | <font style="color:rgba(0, 0, 0, 0.85);background-color:#ffffff;">MNN_FORWARD_METAL</font> | GPU | PC-Mac / iPhone |
| 2 | <font style="color:rgba(0, 0, 0, 0.85);background-color:#ffffff;">MNN_FORWARD_CUDA</font> | GPU | PC-NV / 服务器-NV |
| 3 | <font style="color:rgba(0, 0, 0, 0.85);background-color:#ffffff;">MNN_FORWARD_OPENCL</font> | GPU | PC / Android |
| 5 | <font style="color:rgba(0, 0, 0, 0.85);background-color:#ffffff;">MNN_FORWARD_NN</font> | NPU / GPU | PC-Mac / iPhone |
| 7 | <font style="color:rgba(0, 0, 0, 0.85);background-color:#ffffff;">MNN_FORWARD_VULKAN</font> | GPU | PC / Android |
| 9 | <font style="color:rgba(0, 0, 0, 0.85);background-color:#ffffff;">MNN_FORWARD_USER_1</font> | NPU | Android-华为手机 |






### RuntimeManager
在加载 MNN 时，创建 RuntimeManager 并配置，可以让 MNN 使用对应的计算单元，比如以下代码基于 MNN 的 OpenCL 后端启用 GPU 计算单元

```python
config = {}
config['precision'] = 2
config['backend'] = 3
config['numThread'] = 4

rt = MNN.nn.create_runtime_manager((config,))
rt.set_cache(".cachefile")

net = MNN.nn.load_module_from_file(sys.argv[1], ["input"], ["MobilenetV1/Predictions/Reshape_1"], runtime_manager=rt)
```

+ config 中需要配置如下参数，均传整数，具体用法参考后面章节
    - backend
        * 0 : CPU
        * 1 : Metal
        * 2 : CUDA
        * 3 : OpenCL
        * 5: NPU
        * 7: Vulkan
    - precision
        * 0 : normal
        * 1 : high
        * 2 : low
    - memory
        * 0 : normal
        * 1 : high
        * 2 : low
    - power
        * 0 : normal
        * 1 : high
        * 2 : low
    - numThread ：CPU / GPU 配置方式不同

## 使用CPU
+ backend 为 0 

```python
config['backend'] = 0
```



### 多线程
通过 numThread 可以设置线程数

```python
config['backend'] = 0
config['numThread'] = 4
```

+ numThread 需要在 1 和 32 之间
+ numThread 加速率与手机大核数成正比，如果超过大核数则无法加速，手机上一般设成 2 或 4 能有一定加速效果


### 动态量化
memory 设成 low ，且模型为权重量化出来的模型，bits 数为4或8时，可以启用动态量化：

```python
config['backend'] = 0
config['memory'] = 2
rt = MNN.nn.create_runtime_manager((config,))
net = MNN.nn.load_module_from_file(sys.argv[1], ["input"], ["MobilenetV1/Predictions/Reshape_1"], runtime_manager=rt)
```

+ 作用：
    - 降低运行内存大小（模型部分降低到浮点模型的 1/4，但特征部分不变）
    - 模型计算量较大时，且设备支持 sdot / smmla 时可以有1-2倍加速
+ 缺陷：有可能出现较小的精度损失

### FP16
通过 precision 可以设置模型是否使用 fp16 推理

```python
config['precision'] = 2
```

+ 只在支持的设备上生效，一般2020年后出的手机都支持
    - iphone8 及以后的苹果手机
    - Android 手机从 Arm Coretex A75 开启均支持
+ 对输入模型没有要求，一般都有50%-100%加速效果，内存占用减半
+ 与动态量化叠加使用时，只能加速非卷积部分：加速率较小，但可以减少特征部分内存
+ 有可能出现数值越界，造成精度丢失，建议在模型中多加 normalize 层，保证数值范围控制在 -65504 到 65504 之间



参考数据：

+ Mac M1pro - Mobilenet v2

| | | 耗时 / ms | 内存 / mb |
| --- | --- | --- | --- |
| FP32 | precision = 1, memory = 1 | 8.106100 | 19.242306 |
| 基于FP32的动态量化 | precision = 1, memory = 2 | 4.739200 | 9.624172 |
| FP16 | precision = 2, memory = 1 | 4.225200 | 9.762356 |
| 基于FP16的动态量化 | precision = 2, memory = 2 | 3.663600 | 6.616970 |




## 使用GPU
### Forward type
由于不同平台上使用 GPU 的驱动不一样，对应地MNN实现了多个后端，需要按各自平台，选择使用 GPU 的方案

| type | 驱动 | 适用系统 | 适用设备 |
| --- | --- | --- | --- |
| 1 | Metal | iOS / MacOS | 苹果设备 |
| 2 | Cuda | Linux / Windows | Nvdia 系列GPU |
| 3 | OpenCL | Android / MacOs / Windows / Linux | 安装了OpenCL驱动的设备 |
| 7 | Vulkan | Android / MacOs / Windows / Linux | 安装了Vulkan驱动的设备 |


### precision / memory / power
+ precision
    - 0: fp16 存储，转换到 fp32 计算
    - 1: fp32 存储和计算
    - 2: fp16 存储和计算
+ power
    - 目前仅高通的GPU支持调节
        * 1: 高优先级
        * 2: 低优先级（可以避免阻塞渲染）
+ memory
    - 0 / 1: 权重量化的模型，加载时将权重反量化为浮点
    - 2: 权重量化的模型，运行时将权重反量化为浮点，增加计算量，减少占用内存和内存传输量

### numThread
GPU 的 numThread 是一个 mask ，表示多种功能叠加：

+ 1 ：禁止 AutoTuning 
+ 4 ：进行 AutoTuning （会较大地增加初始化耗时，但相应地，运行耗时会降低）
+ 64 ：偏向于使用 buffer 数据类型
+ 128 ：偏向于使用 image 数据类型，无法与 64 叠加
+ 512 ：开启 batch record ，对于支持的设备，如高通芯片上有一定性能提升，8Gen3 上甚至能提升100%
    - 如果模型中存在 GPU 后端不支持的算子，开启时会无法回退，导致推理失败
    - 对于算力较弱的GPU，有可能性能劣化



比如设成 580 (512 + 64 + 4)表示：开启 batch record ，使用 buffer 数据类型，进行 Autotuning

### 初始化时间优化
#### shape_mutable
+ 初始化默认合并在第一次推理的过程中，也就是 module 第一次 forward 时进行初始化。对于输入形状变化频率少的的模型，建议加上 shape_mutable = False ，将初始化转移到 load module 的过程中。此外，设置 shape_mutable = False 后，可以避免每次 forward 时，由于输入内存地址发生变化而导致的重新进行内存分配（使用 CPU / Metal 后端时不会），提升 forward 性能

```python
net = MNN.nn.load_module_from_file(sys.argv[1], ["input"], ["MobilenetV1/Predictions/Reshape_1"], runtime_manager=rt, shape_mutable=False)
```

#### cache file
+ 由于 GPU 初始化时间较长，MNN 提供了 cache file 机制，支持在初始化后保存部分信息到 cache file 里面，下一次初始化时可以复用这些信息
+ 若 cache file 不存在，将在初始化后创建 cache file ，存储初始化信息
+ 若 cache file 存在，会在初始化时读取相应的信息，以减少初始化时间

```python
config = {}
config['precision'] = 2
config['backend'] = 3
config['numThread'] = 4

rt = MNN.nn.create_runtime_manager((config,))
rt.set_cache("user.cache")
```

#### auto_backend
+ 部分任务有响应时间限制，完整进行模型的初始化会超时。auto_backend提供一个机制对模型进行部分初始化，并把结果保存到 cache file 里面。在初始化信息记录完之前回退到 CPU 推理
+ 设置 mode = 9 启用 auto_backend
+ 设置 hint(0, 20) ，表示最多初始化 20 个算子

示例：

```python
config = {}
config['precision'] = 'low'
config['backend'] = 3
config['numThread'] = 4

rt = MNN.nn.create_runtime_manager((config,))
rt.set_cache("user.cache")
# set_mode(type) //type 9 for "auto_backend"
rt.set_mode(9)
# set_hint(type, value) //type 0 for "tune_num"
rt.set_hint(0, 20)

net = MNN.nn.load_module_from_file("user.mnn",
    ["image"], ["prop"],
    runtime_manager=rt,
    shape_mutable=False
    )
```



## 使用NPU
### 配置
+ NPU 对应 Forward type 为 5
+ `shape_mutable` 必须设为 False

```python
config = {}
config['precision'] = 0
config['backend'] = 5
config['numThread'] = 4

rt = MNN.nn.create_runtime_manager((config,))
net = MNN.nn.load_module_from_file("user.mnn",
           ["image"], ["prob"],
           runtime_manager=rt,
           shape_mutable=False)
```



### Precision
+ Precision = 2 时表示特殊的 uint8 输入

### 模型限制
+ 只支持 CV 类模型（CNN等），并有可能出现一些算子不支持的情况
+ 即便算子完全支持，也有可能出现回退到GPU上运行以致性能下降的问题，和系统版本相关
+ 使用 NPU 时需要先调整模型结构，确定支持后再训练



# 输入输出数据
## MNN.CV / MNN.Numpy
+ MNN 实现了轻量级的对标 CV / Numpy 的库，由于核心计算能力基于 MNN 算子实现，库本身只有100 k 左右。
+ 在计算时需要构图->运算->销毁，成本较高，在手机上容易成为性能瓶颈，建议尽量简化前后处理代码逻辑，减少函数调用次数。

## 内存布局说明
- MNN.CV 生产的VARP默认为`NHWC`布局
- MNN.Numpy 生产的VARP默认为`NCHW`布局
- 若布局和模型输入(用`--info`可以查看模型所需的输入内存布局)不匹配，可以用`set_order`函数强制修改布局:

```
import MNN.numpy as np
dims = [2, 21, 3]
var = np.random.random(dims)
var.set_order(MNN.expr.NHWC)
```


## Pytorch 转 MNN
+ 如果希望前后处理不成为性能瓶颈，可以先用 pytorch 实现相关代码，导出成 onnx 再转 mnn 模型
+ 运行时加载 mnn 模型替代前后处理





