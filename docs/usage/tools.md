# 其他工具使用

## 测试工具
[从源码编译](../compile/tools.html#id4)使用cmake编译时，build目录下的产物也包含测试使用的工具集，下面逐项说明。
### MNNV2Basic.out
- 功能
  - 测试性能、输出结果，可检查与Caffe/Tensorflow的预期结果是否匹配。
    **注意：对非CPU后端来说，只有总耗时是准确的，单个op耗时和op耗时占比都是不准确的**
- 参数
  - 第一个参数指定 待测试模型的二进制文件。
  - 第二个参数指定 性能测试的循环次数，10就表示循环推理10次。
  - 第三个参数指定 是否输出推理中间结果，0为不输出，1为只输出每个算子的输出结果（{op_name}.txt），2为输出每个算子的输入（Input_{op_name}.txt）和输出（{op_name}.txt）结果； 默认输出当前目录的output目录下（使用工具之前要自己建好output目录）。
  - 第四个参数指定 执行推理的计算设备，有效值为 0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)
  - 第五个参数为线程数，默认为4，仅对CPU有效
  - 第六个参数指定 输入tensor的大小，一般不需要指定。
  - 其他
- 默认输入与输出
  - 只支持单一输入、单一输出。输入为运行目录下的input_0.txt；输出为推理完成后的第一个输出tensor，转换为文本后，输出到output.txt中。
- 示例
    ```bash
    ./MNNV2Basic.out mobilenetv2-7.mnn 10 0 0 4 1x3x224x224

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
### ModuleBasic.out
`./ModuleBasic.out ${test.mnn} ${Dir} [dump] [forwardType] [runLoops] [numberThread] [precision] [cacheFile]`
- 功能
  - 类似 MNNV2Basic.out ，对于带控制流模型，或者多输入多输出的模型，建议采用这个工具
- 参数
  - 第一个参数指定 待测试模型的二进制文件。
  - 第二个参数指定 输入输出信息文件夹，可使用 fastTestOnnx.py / fastTestTf.py / fastTestTflite.py 等脚本生成，参考模型转换的正确性校验部分。
  - 第三个参数指定 是否输出推理中间结果，0为不输出，1为输出每个算子的输出结果（{op_name}.txt）， 默认输出当前目录的output目录下（使用工具之前要自己建好output目录）。
  - 第四个参数指定 执行推理的计算设备，有效值为 0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)
  - 第五个参数指定性能测试的循环次数默认为 0 ，即不做性能测试
  - 第六个参数为线程数或GPU的mode，默认为1
  - 第七个参数指定精度类似，0 为 normal, 1 为high ，2 为 low。
- 默认输出
  - 在当前目录 output 文件夹下，依次打印输出为 0.txt , 1.txt , 2.txt , etc
- 示例
```bash
python ../tools/script/fastTestOnnx.py mobilenetv2-7.onnx
./ModuleBasic.out mobilenetv2-7.mnn onnx 0 0 10
    
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
### SequenceModuleTest.out
`./SequenceModuleTest.out ${test.mnn} [forwardType] [shapeMutable] ${Dir} ${Dir1} ......`
- 功能
  - 类似 ModuleBasic.out ，适用于多份输入输出数据的校验
- 参数
  - 第一个参数指定 待测试模型的二进制文件。
  - 第三个参数指定 执行推理的计算设备，有效值为 0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)
  - 第四个参数指定 输入输出信息文件夹，可使用 fastTestOnnx.py / fastTestTf.py / fastTestTflite.py 等脚本生成，参考模型转换的正确性校验部分。
  - 后续参数类型同第四个参数，指定需要测试的其他文件夹
- 示例
```bash
./SequenceModuleTest.out transformer.mnn 0 1 tr tr1 tr2 tr3 tr4 > error.txt
```
### checkFile.out
`./checkFile.out XXX.txt YYY.txt 0.1`
- 功能
  - 检查两个tensor文本文件是否一致。
- 参数
  - 0.1 表示绝对阈值，不输入则为 0.0001
  - 比对值超过绝对阈值时，会直接输出到控制台
### checkDir.out
`./checkDir.out output android_output 1`
- 功能
  - 比对两个文件夹下同名文件是否一致。
- 参数
  - 1 表示绝对阈值，不输入则为 0.0001
  - 比对值超过绝对阈值时，会直接输出到控制台
### timeProfile.out
`./timeProfile.out ${test.mnn} [runLoops] [forwardType] [shape] [thread]`
- 功能
  - Op 总耗时统计工具和模型运算量估计。
    **注意：不要用这个工具测非CPU后端的性能，需要的话请用MNNV2Basic工具**
- 参数
  - 第一个参数 指定模型文件名
  - 第二个参数 指定运行次数，默认 100
  - 第三个参数 指定 执行推理的计算设备，有效值为 0（浮点 CPU）、1（Metal）、3（浮点OpenCL）、6（OpenGL），7(Vulkan)。（当执行推理的计算设备不为 CPU 时，Op 平均耗时和耗时占比可能不准）
  - 第四个参数 指定输入大小，可不设
  - 第五个参数 指定线程数，可不设，默认为 4
- 输出
  - 第一列为 Op类型
  - 第二列为 平均耗时
  - 第三列为 耗时占比
- 示例
```bash
./timeProfile.out mobilenetv2-7.mnn 10 0 1x3x224x224 1

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
### backendTest.out
`./backendTest.out ${test.mnn} [forwardType] [tolerance] [precision]`
- 功能
  - 这个工具可以对比指定计算设备和CPU执行推理的结果。
- 参数
  - 该工具默认读取当前目录下的 input_0.txt 作为输入
  - 第一个参数：模型文件
  - 第二个参数：执行推理的计算设备
  - 第三个参数：误差容忍率
  - 第四个参数：精度，0 表示 normal ，1 为high，2 为low
- 示例
    ```bash
    ./backendTest.out mobilenetv2-7.mnn 3 0.15 1

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
- Android 中使用
  - 先编译相关的库和可执行文件，然后 push 到 Android 手机上，用 adb 执行命令，参考 project/android/testCommon.sh 
    ```bash
    cd project/android
    mkdir build_64
    cd build_64 && ../build_64.sh
    ../updateTest.sh
    ../testCommon.sh ./backendTest.out temp.mnn 3 0.15 1
    ```

## Benchmark工具
### Linux / macOS / Ubuntu
[从源码编译](../compile/tools.html#benchmark)，然后执行如下命令:
```bash
./benchmark.out models_folder loop_count warm_up_count forwardtype
```
参数如下:
- models_folder: benchmark models文件夹，[benchmark models](https://github.com/alibaba/MNN/tree/master/benchmark/models)。
- loop_count: 可选，默认是10
- warm_up_count: 预热次数
- forwardtype: 可选，默认是0，即CPU，forwardtype有0->CPU，1->Metal，3->OpenCL，6->OpenGL，7->Vulkan
### Android
在[benchmark目录](https://github.com/alibaba/MNN/tree/master/benchmark)下直接执行脚本`bench_android.sh`，默认编译armv7，加参数-64编译armv8，参数-p将[benchmarkModels](https://github.com/alibaba/MNN/tree/master/benchmark/models) push到机器上。
脚本执行完成在[benchmark目录](https://github.com/alibaba/MNN/tree/master/benchmark)下得到测试结果`benchmark.txt`
### iOS
1. 先准备模型文件，进入tools/script目录下执行脚本`get_model.sh`；
2. 打开demo/iOS目录下的demo工程，点击benchmark；可通过底部工具栏切换模型、推理类型、线程数。
### 基于表达式构建模型的Benchmark
[从源码编译](../compile/tools.html#benchmark)，运行以下命令查看帮助：
```bash
 ./benchmarkExprModels.out help
```
示例：
```bash
 ./benchmarkExprModels.out MobileNetV1_100_1.0_224 10 0 4 
 ./benchmarkExprModels.out MobileNetV2_100 10 0 4 
 ./benchmarkExprModels.out ResNet_100_18 10 0 4 
 ./benchmarkExprModels.out GoogLeNet_100 10 0 4 
 ./benchmarkExprModels.out SqueezeNet_100 10 0 4 
 ./benchmarkExprModels.out ShuffleNet_100_4 10 0 4
```
相应模型的paper链接附在头文件里，如`benchmark/exprModels/MobileNetExpr.hpp`

## 模型量化工具
`./quantized.out origin.mnn quan.mnn imageInputConfig.json`

**注意：本工具为“离线量化”工具，即训练之后的量化。 训练量化请看[这里](https://www.yuque.com/mnn/cn/bhz5eu)**

**由于MNN训练框架不太成熟，如果你的模型用MNN训练框架训练不起来，可以试试**[MNNPythonOfflineQuant工具](https://github.com/alibaba/MNN/tree/master/tools/MNNPythonOfflineQuant)

- 参数
  - 第一个参数为原始模型文件路径，即待量化的浮点模
  - 第二个参数为目标模型文件路径，即量化后的模型
  - 第三个参数为预处理的配置项，参考[imageInputConfig.json](https://github.com/alibaba/MNN/blob/master/tools/quantization/imageInputConfig.json)，该Json的配置信息如下表所示：
- 量化模型的使用
  - 和浮点模型同样使用方法，输入输出仍然为浮点类型
- 参考资料
  - [Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16767/16728)

|  key   |  value  |  说明  |
|--------|---------|-------|
| format |  "RGB", "BGR", "RGBA", "GRAY" | 图片统一按RGBA读取，然后转换到`format`指定格式 |
| mean/normal | `[float]` | `dst = (src - mean) * normal` |
| width/height | `int` | 模型输入的宽高 |
| path | `str` | 存放校正特征量化系数的图片目录 |
| used_image_num | `int` | 用于指定使用上述目录下多少张图片进行校正，默认使用`path`下全部图片 |
| feature_quantize_method | "KL", "ADMM", "EMA" | 指定计算特征量化系数的方法，默认："KL" |
| weight_quantize_method | "MAX_ABS", "ADMM" | 指定权值量化方法，默认："MAX_ABS" |
| feature_clamp_value | `int` | 特征的量化范围，默认为127，即[-127, 127]对称量化，有时，量化之后溢出会很多，造成误差较大，可适当减小此范围，如减小至120，但范围减小太多会导致分辨率下降，使用时需测试 |
| weight_clamp_value | `int` | 权值的量化范围，默认127，作用同feature_clamp_value，由于权值精度模型效果影响较大，建议调整feature_clamp_value即可 |
| batch_size | `int` | EMA方法中指定batch size，和模型训练时差不多 |
| quant_bits | `int` | 量化后的bit数，默认为8 |
| skip_quant_op_names | `[str]` | 跳过不量化的op的卷积op名字，因为有些层，如第一层卷积层，对模型精度影响较大，可以选择跳过不量化，可用netron可视化模型，找到相关op名字 |
| input_type | `str` | 输入数据的类型，默认为"image" |
| debug | `bool` | 是否输出debug信息，true或者false，输出的debug信息包含原始模型和量化模型各层输入输出的余弦距离和溢出率 |

| feature_quantize_method | 说明 |
|--------------------|------|
| KL | 使用KL散度进行特征量化系数的校正，一般需要100 ~ 1000张图片(若发现精度损失严重，可以适当增减样本数量，特别是检测/对齐等回归任务模型，样本建议适当减少) |
| ADMM | 使用ADMM（Alternating Direction Method of Multipliers）方法进行特征量化系数的校正，一般需要一个batch的数据 |
| EMA | 使用指数滑动平均来计算特征量化参数，这个方法会对特征进行非对称量化，精度可能比上面两种更好。这个方法也是[MNNPythonOfflineQuant](https://github.com/alibaba/MNN/tree/master/tools/MNNPythonOfflineQuant)的底层方法，建议使用这个方法量化时，保留你pb或onnx模型中的BatchNorm，并使用 --forTraining 将你的模型转到MNN，然后基于此带BatchNorm的模型使用EMA方法量化。另外，使用这个方法时batch size应设置为和训练时差不多最好。 |

| weight_quantize_method | 说明 |
|--------------------|------|
| MAX_ABS | 使用权值的绝对值的最大值进行对称量化 |
| ADMM | 使用ADMM方法进行权值量化 |

## 可视化工具
可视化的效果：
![屏幕快照 2019-10-08 上午12.55.46.png](https://cdn.nlark.com/yuque/0/2019/png/520564/1570514400933-dafad62c-435e-4a8e-9619-bb054cb01cc5.png#align=left&display=inline&height=1760&margin=%5Bobject%20Object%5D&name=%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-10-08%20%E4%B8%8A%E5%8D%8812.55.46.png&originHeight=1760&originWidth=2272&size=724074&status=done&style=none&width=2272)

在详细调研了市面上比较主流的可视化工具后，`Netron`是一款受众面较多、兼容多款模型的可视化模型，同时它还具有跨平台支持、`Python`模块支持的能力。因此，在研究了一番它的设计和架构并考虑后续`MNN`自身的演进，我们决定**官方维护`MNN`模型的可视化能力并将其作为`Pull Request`合并，大家可以放心使用啦。**

- 功能列表
    - 支持加载`.mnn`模型 。
    - 支持将可视化的图导出成图片保存。
    - 支持拓扑结构的展示、`Operator`/`Input`/`Output`的内容展示。
    - 支持结构化的`weight`，`scale`，`bias`等数据的展示，**并支持将此类数据持久化保存**。
- 使用方式（Release版本）
  - [下载地址](https://github.com/lutzroeder/netron/releases)
  - `macOS`: 下载 `.dmg`文件 或者 `brew cask install netron`
  - `Linux`: 下载 `.AppImage`或者`.deb`文件.
  - `Windows`: 下载`.exe`文件.
  - `Python`：`pip install netron`
- 使用开发版本
  - 对仓库地址：`https://github.com/lutzroeder/netron`，进行`clone`。始终使用`master`分支。
  - `cd [your_clone_path]/netron`
  - 安装`npm`，确保`npm`版本大于`6.0.0`
  - `npm install`
- 使用JavaScript调试
  - `npx electron ./`（如果这步失败，单独`npm install -g npx`）
- 使用`Python`调试
  - `python3 setup.py build`
  - `export PYTHONPATH=build/lib:${PYTHONPATH}`
  - `python3 -c "import netron; netron.start(None)"`
- 遗留问题
  - 加载超大模型可能渲染失败（几千个节点）