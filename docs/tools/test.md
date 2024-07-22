# 测试工具
[从源码编译](../compile/tools.html#id4)使用cmake编译时，build目录下的产物也包含测试使用的工具集，下面逐项说明。

## GetMNNInfo
### 功能
获取MNN模型文件的输入输出和版本信息：
- 输入信息包括输入Tensor的`名称`，`数据排布`，`形状`和`数据类型`
- 输出信息包括输出Tensor的`名称`
- 版本信息为转换该模型使用的`MNNConvert`的版本，当版本小于`2.0.0`时统一显示为`Model Version: < 2.0.0`
### 参数
`Usage: ./GetMNNInfo model`
- `model:str`：模型文件路径
### 示例
```bash
$ ./GetMNNInfo mobilenet_v1.mnn 
Model default dimensionFormat is NCHW
Model Inputs:
[ data ]: dimensionFormat: NC4HW4, size: [ 1,3,224,224 ], type is float
Model Outputs:
[ prob ]
Model Version: < 2.0.0
```

## MNNV2Basic.out
### 功能
测试性能、输出结果，可检查与Caffe/Tensorflow的预期结果是否匹配。
**注意：对非CPU后端来说，只有总耗时是准确的，单个op耗时和op耗时占比都是不准确的**
### 参数
`./MNNV2Basic.out model [runLoops runMask forwardType numberThread precision_memory inputSize]`
- `model:str` 模型文件路径
- `runLoops:int` 性能测试的循环次数，可选，默认为`1`
- `runMask:int` 是否输出推理中间结果，0为不输出，1为只输出每个算子的输出结果（{op_name}.txt）;2为输出每个算子的输入（Input_{op_name}.txt）和输出（{op_name}.txt）结果； 默认输出当前目录的output目录下（使用工具之前要自己建好output目录）; 16为开启自动选择后端；32为针对Winograd算法开启内存优化模式，开启后会降低模型（如果含有Winograd Convolution算子）运行时的内存但可能会导致算子的性能损失。可选，默认为`0`
- `forwardType:int` 执行推理的计算设备，有效值为：0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)，可选，默认为`0`
- `numberThread:int` 线程数仅对CPU有效，可选，默认为`4`
- `precision_memory:int` 测试精度与内存模式，precision_memory % 16 为精度，有效输入为：0(Normal), 1(High), 2(Low), 3(Low_BF16)，可选，默认为`2` ; precision_memory / 16 为内存设置，默认为 0 (memory_normal) 。例如测试 memory 为 low (2) ，precision 为 1 (high) 时，设置 precision_memory = 9 (2 * 4 + 1)
- `inputSize:str` 输入tensor的大小，输入格式为：`1x3x224x224`，可选，默认使用模型默认输入


### 默认输入与输出
只支持单一输入、单一输出。输入为运行目录下的input_0.txt；输出为推理完成后的第一个输出tensor，转换为文本后，输出到output.txt中。
### 示例
```bash
$ ./MNNV2Basic.out mobilenetv2-7.mnn 10 0 0 4 1x3x224x224
Use extra forward type: 0
1 3 224 224 
Open Model mobilenetv2-7.mnn
Load Cache file error.
===========> Resize Again...
test_main, 225, cost time: 8.656000 ms
Session Info: memory use 21.125130 MB, flops is 300.982788 M, backendType is 13
===========> Session Resize Done.
===========> Session Start running...
Input size:200704
    **Tensor shape**: 1, 3, 224, 224, 
output: output
precision:2, Run 10 time:
                            Convolution96_raster_0 	[Raster] run 10 average cost 0.003400 ms, 0.061 %, FlopsRate: 0.000 %
                                            452 	[BinaryOp] run 10 average cost 0.004700 ms, 0.084 %, FlopsRate: 0.002 %
                                            ...
                                            483 	[Convolution] run 10 average cost 0.452600 ms, 8.125 %, FlopsRate: 6.402 %
                                            486 	[ConvolutionDepthwise] run 10 average cost 0.461000 ms, 8.276 %, FlopsRate: 0.900 %
Avg= 5.570600 ms, OpSum = 7.059200 ms min= 3.863000 ms, max= 11.596001 ms
```

## ModuleBasic.out
### 功能
类似`MNNV2Basic.out`，对于带控制流模型，或者多输入多输出的模型，必须采用这个工具
### 参数
`./ModuleBasic.out model dir [runMask forwardType runLoops numberThread precision_memory cacheFile]`
- `model:str` 模型文件路径
- `dir:str` 输入输出信息文件夹，可使用 testMNNFromTf.py / testMNNFromOnnx.py / testMNNFromTflite.py 等脚本生成，参考模型转换的正确性校验部分。
- `runMask:int` 默认为 0 ，为一系列功能的开关，如需开启多个功能，可把对齐的 mask 值相加（不能叠加的情况另行说明），具体见下面的 runMask 参数解析
- `forwardType:int` 执行推理的计算设备，有效值为：0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)，可选，默认为`0`
- `runLoops:int` 性能测试的循环次数，可选，默认为`0`即不做性能测试
- `numberThread:int` GPU的线程数，可选，默认为`1`
- `precision_memory_power:int` 测试精度与内存模式，precision_memory_power % 4 为精度，有效输入为：0(Normal), 1(High), 2(Low), 3(Low_BF16)，可选，默认为`0` ; (precision_memory_power / 4 % 4) 为内存设置，默认为 0 (memory_normal) ; (precision_memory_power / 16 % 4) 为功耗设置，默认为 0 (power_normal)。例如测试 memory 为 2(low) ，precision 为 1 (high) ，power 为 0(normal) 时，设置 precision_memory = 9 (2 * 4 + 1 + 0 * 16)


### 默认输出
在当前目录 output 文件夹下，依次打印输出为 0.txt , 1.txt , 2.txt , etc

### 测试文件夹生成
- 若有原始的tf模型/Onnx模型，可以使用testMNNFromTf.py / testMNNFromOnnx.py / testMNNFromTflite.py 等脚本生成
- 若只有mnn模型，可以用 tools/script/make_test_for_mnn.py 脚本生成测试文件夹，使用方式：mkdir testdir && pythhon3 make_test_for_mnn.py XXX.mnn testdir
- 为了方便模拟应用中的运行性能，可以通过修改测试文件夹下的 input.json ，增加 freq 项，以指定该模型运行的频率（每秒多少次）

### runMask 参数说明
- 1 : 输出推理中间结果，每个算子的输入存到（Input_{op_name}.txt），输出存为（{op_name}.txt）， 默认输出当前目录的output目录下（使用工具之前要自己建好output目录），不支持与 2 / 4 叠加
- 2 : 打印推理中间结果的统计值（最大值/最小值/平均值），只支持浮点类型的统计，不支持与 1 / 4 叠加
- 4 : 统计推理过程中各算子耗时，不支持与 1 / 2 叠加，仅在 runLoops 大于 0 时生效
- 8 : shapeMutable 设为 false （默认为 true）
- 16 : 适用于使用 GPU 的情况，由 MNN 优先选择 CPU 运行，并将 GPU 的 tuning 信息存到 cache 文件，所有算子 tuning 完成则启用 GPU
- 32 : rearrange 设为 true ，降低模型加载后的内存大小，但会增加模型加载的初始化时间
- 64 : 创建模型后，clone 出一个新的模型运行，用于测试 clone 功能（主要用于多并发推理）的正确性
- 128 : 使用文件夹下面的 input.mnn 和 output.mnn 做为输入和对比输出，对于数据量较大的情况宜用此方案
- 512 : 开启使用Winograd算法计算卷积时的内存优化，开启后模型的运行时内存会降低，但可能导致性能损失。
- 1024: 使用动态量化推理时，对输入数据分batch量化以提高模型的推理精度


### 示例
```bash
$ python ../tools/script/testMNNFromOnnx.py mobilenetv2-7.onnx
$ ./ModuleBasic.out mobilenetv2-7.mnn onnx 0 0 10   
Test mobilenetv2-7.mnn from input info: onnx
input
output: output
Use extra forward type: 0
precision=0 in main, 291 
cacheFileName=s .tempcache in main, 292 
Load Cache file error.
before compare output: (1, 1000, )
Write output output to output/0.txt
memoryInMB=f 22.605423 in main, 428 
Avg= 9.946699 ms, min= 9.472000 ms, max= 10.227000 ms
```

## SequenceModuleTest.out
### 功能
类似`ModuleBasic.out`，适用于多份输入输出数据的校验
### 参数
`./SequenceModuleTest.out model [forwardType] [shapeMutable] dir1 dir2 ......`
- `model:str` 模型文件路径
- `forwardType:int` 执行推理的计算设备，有效值为：0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)
- `numberThread:int` 线程数或GPU模式
- `dir_n:str` 输入输出信息文件夹，可使用 testMNNFromOnnx.py 等脚本生成，参考模型转换的正确性校验部分
```bash
./SequenceModuleTest.out transformer.mnn 0 1 tr tr1 tr2 tr3 tr4 > error.txt
```

## checkFile.out
### 功能
对比两个文本文件数据是否一致
### 参数
`./checkFile.out file1 file2 [tolerance]`
- `file_1:str` 比较的第一个文件路径
- `file_2:str` 比较的第二个文件路径
- `tolerance:float` 误差的绝对阈值，误差大于阈值会输出到控制台，可选，默认为`0.001`
### 示例
```bash
$ echo '0.111\n0.222\n0.333\n0.444' > x.txt
$ echo '0.111\n0.2225\n0.335\n0.448' > y.txt
$ ./checkFile.out x.txt y.txt 0.001
Error for 2, v1=0.333000, v2=0.335000
Error for 3, v1=0.444000, v2=0.448000
```

## checkDir.out
### 功能
对比两个文件夹下同名文件数据是否一致
### 参数
`./checkDir.out dir1 dir2 [tolerance order]`
- `dir1:str` 比较的第一个文件夹路径
- `dir2:str` 比较的第二个文件夹路径
- `tolerance:float` 误差的绝对阈值，误差大于阈值会输出到控制台，可选，默认为`0.001`
- `order:str` 比较文件顺序描述文件路径，该参数为一个文本文件，其中每行指定一个比较文件名，将会按照该顺序进行比较，可选，默认直接比较所有文件
### 示例
```bash
$ mkdir dir1 dir2
$ echo '0.111\n0.222\n0.333\n0.444' > dir1/a.txt
$ echo '0.111\n0.2225\n0.335\n0.448' > dir2/a.txt
$ ./checkDir.out dir1 dir2
Compare:
dir1
dir2
tolerance=0.001000
Error for 2, a.txt, 2, v1=0.333000, v2=0.335000
```

## checkInvalidValue.out
### 功能
根据指定`limit`检测指定目录下的文件，是否包含非法数据，非法数据的定义为：
- 数据值为`nan`
- 数据的值小于`10^limit`
### 参数
`./checkInvalidValue.out dir limit`
- `dir:str` 检测的目录路径
- `limit:int` 检测值的最大阈值为`10^limit`，可选，默认为`10`
### 示例
```bash
$ mkdir dir1
$ echo '0.111\n0.222\n0.333\n0.444' > dir1/a.txt
$ ./checkInvalidValue.out dir1 1  
Compare:
dir1
limit=1
Correct : a.txt
$ ./checkInvalidValue.out dir1 -1
Compare:
dir1
limit=-1
Error for a.txt, 0, v1=0.111000
```

## timeProfile.out
### 功能
模型总耗时，逐层耗时统计和模型运算量估计。**注意：不要用这个工具测非CPU后端的性能，需要的话请用MNNV2Basic工具**
### 参数
`./timeProfile.out model [runLoops forwardType inputSize numberThread precision]`
- `model:str` 模型文件路径
- `runLoops:int` 测试的循环次数，可选，默认为`100`
- `forwardType:int` 执行推理的计算设备，有效值为：0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)，可选，默认为`0`；（当执行推理的计算设备不为 CPU 时，Op平均耗时和耗时占比可能不准）
- `inputSize:str` 输入tensor的大小，输入格式为：`1x3x224x224`，可选，默认使用模型默认输入
- `numberThread:int` 线程数仅对CPU有效，可选，默认为`4`
- `precision:int` 精度仅对CPU有效，可选，默认为`0`
### 输出
- 第一列为 Op类型
- 第二列为 平均耗时
- 第三列为 耗时占比
### 示例
```bash
$ ./timeProfile.out mobilenetv2-7.mnn 10 0 1x3x224x224 1
1 3 224 224 
Set ThreadNumber = 1
Open Model mobilenetv2-7.mnn
Sort by node name !
Node Name              	Op Type              	Avg(ms)  	%        	Flops Rate 	
339                    	BinaryOp             	0.039300 	0.364004 	0.023848   	
356                    	BinaryOp             	0.008500 	0.078729 	0.007949 
...  
Convolution96          	Convolution          	0.273800 	2.535987 	0.425273   	
Convolution96_raster_0 	Raster               	0.002100 	0.019451 	0.000406   	
output_raster_0        	Raster               	0.005200 	0.048163 	0.000317   	
Print <=20 slowest Op for Convolution, larger than 3.00
474 -  16.396393 GFlops, 6.12 rate
483 -  22.691771 GFlops, 7.86 rate
498 -  29.693188 GFlops, 3.38 rate
627 -  37.167416 GFlops, 5.00 rate
624 -  37.296322 GFlops, 3.74 rate

Sort by time cost !
Node Type            	Avg(ms)  	%         	Called times 	Flops Rate 	
Raster               	0.007300 	0.067614  	2.000000     	0.000722   	
Pooling              	0.020700 	0.191727  	1.000000     	0.000000   	
BinaryOp             	0.083600 	0.774319  	10.000000    	0.068562   	
ConvolutionDepthwise 	2.162301 	20.027637 	17.000000    	6.882916   	
Convolution          	8.522696 	78.938828 	36.000000    	93.047722  	
total time : 10.796584 ms, total mflops : 300.983246 
main, 138, cost time: 111.161003 ms
```

## backendTest.out
### 功能
对比指定计算设备和CPU执行推理的结果，该工具默认读取当前目录下的`input_0.txt`作为输入
### 参数
`./backendTest.out model [forwardType tolerance precision modeNum stopOp]`
- `model:str` 模型文件路径
- `forwardType:int` 执行推理的计算设备，有效值为：0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)，可选，默认为`0`
- `tolerance:float` 误差的绝对阈值，误差大于阈值会认为不一致，可选，默认为`0.05`
- `precision:int` 测试精度，有效输入为：0(Normal), 1(High), 2(Low), 3(Low_BF16)，可选，默认为`0`
- `modeNum:int` 设置GPU的执行模式，可选，默认为`1`
- `stopOp:str` 指定某一层的名称，当执行到该层时停止对比，可选，默认为空
### 示例
```bash
$ ./backendTest.out mobilenetv2-7.mnn 3 0.15 1
Test forward type: 3
Tolerance Rate: 0.150000
Open Model mobilenetv2-7.mnn
Input: 224,224,3,0
precision=1 in main, 268 
modeNum=1 in main, 273 
stopOp.c_str()=s  in main, 278 
Correct ! Run second pass
Correct !
```
### 在Android中使用
先编译相关的库和可执行文件，然后push到Android手机上，用adb执行命令，参考`project/android/testCommon.sh`
```bash
cd project/android
mkdir build_64
cd build_64 && ../build_64.sh
../updateTest.sh
../testCommon.sh ./backendTest.out temp.mnn 3 0.15 1
```

## getPerformance
### 功能
获取当前设备的CPU性能，打印出每个CPU核心的频率；在Android设备上还会打印该设备CPU的浮点计算能力(GFLOPS)

*各核心频率仅在Linux/Android环境中有效，计算能力仅在Android中有效*
### 参数
`./getPerformance.out` 无参数
### 示例(Linux)
```bash
$ ./getPerformance.out 
Start PERFORMANCE !!! 
CPU PERFORMANCE -> loopCounts : 100000000 
core 0 : max : 3800000, min : 2200000 
core 1 : max : 3800000, min : 2200000 
...
core 23 : max : 3800000, min : 2200000 
 ======================== float ===============================
CPU float gflops : 0.000000
```

## modelCompare.out
### 功能
原始模型与量化模型推理结果对比
### 参数
`./modelCompare.out origin_model quant_model [tolerance]`
- `origin_model:str` 原始浮点模型路径
- `quant_model:str` int8量化模型路径
- `tolerance:float` 误差的绝对阈值，误差大于阈值会认为不一致，可选，默认为`0.05`
- `modeNum:int` 设置GPU的执行模式，可选，默认为`1`
- `stopOp:str` 指定某一层的名称，当执行到该层时停止对比，可选，默认为空
### 示例
```bash
$ ./modelCompare.out mobilnet.mnn mobilnet_quant.mnn 1  
Tolerance Rate: 1.000000
Open Model mobilnet.mnn, mobilnet_quant.mnn
Input: 224,224,3,1
precision=0 in main, 252 
modeNum=1 in main, 257 
stopOp.c_str()=s  in main, 262 
Correct ! Run second pass
Correct !
```

## mobilenetTest.out
### 功能
执行`mobilenet`的推理测试，输入模型与图片，输出结果的`top-10`与执行时间
### 参数
`./mobilenetTest.out model image [forwardType precision label]`
- `model:str` 模型文件路径
- `image:str` 输入图片文件路径
- `forwardType:int` 执行推理的计算设备，有效值为：0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)，可选，默认为`0`
- `precision:int` 测试精度，有效输入为：0(Normal), 1(High), 2(Low), 3(Low_BF16)，可选，默认为`1`
- `label:str` 种类标签文件，imagenet的1000中分类的标签，可选，默认不使用标签，直接输出种类的index
### 示例
```bash
$ ./mobilenetTest.out mobilnet.mnn cat.jpg 
model:mobilnet.mnn, input image: cat.jpg, forwardType:0, precision:1
main, 90, cost time: 7.789001 ms
output size:1000
287, 0.218567
282, 0.141113
285, 0.125122
281, 0.117249
283, 0.039116
278, 0.038887
700, 0.034599
279, 0.014238
277, 0.012278
17, 0.011496
```

## testModel.out
### 功能
输入模型，输入文件和期望输出文件；执行推理判断使用制定输入是否能够得到相同的输出
### 参数
`./testModel.out model input output`
- `model:str` 模型文件路径
- `input:str` 输入数据，文本文件格式，其中为浮点数据
- `output:str` 期望输出数据，文本文件格式，其中为浮点数据
### 示例
```bash
$ ./testModel.out mobilenet.mnn input_0.txt output.txt 
Testing model mobilenet.mnn, input: input_0.txt, output: output.txt
First run pass
Test mobilenet.mnn Correct!
```

## testModel_expr.out
### 功能
功能与`testModel.out`相同，使用`Module`接口执行，能够测试带控制流的模型，输出输出使用`.mnn`文件代替文本
### 参数
`./testModel_expr.out model input output [forwardType tolerance precision]`
- `model:str` 模型文件路径
- `input:str` 输入数据，`.mnn`格式文件，使用表达式接口存储的数据
- `output:str` 期望输出数据，`.mnn`格式文件，使用表达式接口存储的数据
- `forwardType:int` 执行推理的计算设备，有效值为：0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)，可选，默认为`0`
- `tolerance:float` 误差的绝对阈值，误差大于阈值会认为不一致，可选，默认为`0.1`
- `precision:int` 测试精度，有效输入为：0(Normal), 1(High), 2(Low), 3(Low_BF16)，可选，默认为`1`
### 示例
```bash
$ ./testModel_expr.out mobilenet.mnn input_0.mnn output.mnn 
Testing model mobilenet.mnn, input: input.mnn, output: output.mnn
Correct!
```

## testModelWithDescribe.out
### 功能
功能与`testModel.out`相同，输入输出通过配置文件来描述，支持多输入与多输出的对比，同时支持指定输入形状
### 参数
`./testModelWithDescribe.out model confg`
- `model:str` 模型文件路径
- `confg:str` 配置文件路径，配置文件如下：
    ```
    # 多个输入用,分开，不要用空格
    # 目前默认输入都是float
    # 文件名为输入tensor的名字后缀为txt，如0表示./0.txt
    input_size = 1
    input_names = 0
    input_dims = 1x3x416x416
    output_size = 4
    output_names = 1257,1859,2374,655
    ```
### 示例
```bash
./testModelWithDescribe.out mobilenet.mnn config.txt
model dir: mobilenet.mnn
Testing Model ====> mobilenet.mnn
First Time Pass
Correct!
```

## testTrain.out
### 说明
根据指定配置文件对模型执行2次训练的反向传播过程并判断loss是否下降
### 参数
`./testTrain.out config dir`
- `config` 训练测试的配置文件，其指定了训练测试的各种信息，其格式参考`示例`
- `dir` 训练所需文件夹，文件夹内包含续联的模型与输入数据
### 示例
```bash
$ ./testTrain.out config.json ./mnist
From 42.261131 -> 11.569703, Test ./mnist/mnist.train.mnn Correct!
```
配置文件内容为：
```json
{
    "Model": "mnist.train.mnn",
    "Loss": "Loss",
    "LR": "LearningRate",
    "LearningRate":0.005,
    "Input":"Input3",
    "Decay":0.3,
    "Target":"Reshape38_Compare",
    "Data":[
        "data.mnn"
    ]
}
```

## winogradExample.out
### 说明
生成winograd变换矩阵程序
### 参数
`./winogradExample.out unit kernelSize`
- `unit:int` 分块大小
- `kernelSize:int` 卷积核大小
### 示例
```bash
$ ./winogradExample.out 3 3                        
A=
Matrix:
1.0000000	0.0000000	0.0000000	
1.0000000	0.5000000	0.2500000	
1.0000000	-0.5000000	0.2500000	
1.0000000	1.0000000	1.0000000	
0.0000000	0.0000000	1.0000000	
B=
Matrix:
1.0000000	0.0000000	0.0000000	0.0000000	0.0000000	
-1.0000000	2.0000000	-0.6666667	-0.3333333	0.2500000	
-4.0000000	2.0000000	2.0000000	0.0000000	-0.2500000	
4.0000000	-4.0000000	-1.3333334	1.3333334	-1.0000000	
0.0000000	0.0000000	0.0000000	0.0000000	1.0000000	
G=
Matrix:
1.0000000	0.0000000	0.0000000	
1.0000000	0.5000000	0.2500000	
1.0000000	-0.5000000	0.2500000	
1.0000000	1.0000000	1.0000000	
0.0000000	0.0000000	1.0000000
```

## fuseTest
### 功能
测试 GPU 自定义算子的功能，目前仅支持 Vulkan Buffer 模式

### 参数
`Usage: ./fuseTest user.spirv config.json`
- `user.spirv:str`：SPIRV文件路径，可以用 glslangValidator -V user.comp -o user.spirv 编译获得
- `config.json:str`: 配置文件路径
### 示例
```bash
$ ./fuseTest user.spirv user.json

## GpuInterTest.out
### 功能
GPU 内存输入测试用例
### 参数
类似`ModuleBasic.out` 
`./GpuInterTest.out model dir [testmode forwardType numberThread precision_memory]`
- `model:str` 模型文件路径
- `dir:str` 输入输出信息文件夹，可使用 testMNNFromTf.py / testMNNFromOnnx.py / testMNNFromTflite.py 等脚本生成，参考模型转换的正确性校验部分。
- `testmode:int` 默认为 0 ，测试输入GPU内存的类型，0 (OpenCL Buffer) 、 1（OpenGL Texture）
- `forwardType:int` 执行推理的计算设备，有效值为：0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)，可选，默认为`0`
- `numberThread:int` GPU的线程数，可选，默认为`1`
- `precision_memory:int` 测试精度与内存模式，precision_memory % 16 为精度，有效输入为：0(Normal), 1(High), 2(Low), 3(Low_BF16)，可选，默认为`2` ; precision_memory / 16 为内存设置，默认为 0 (memory_normal) 。例如测试 memory 为 2(low) ，precision 为 1 (high) 时，设置 precision_memory = 9 (2 * 4 + 1)
