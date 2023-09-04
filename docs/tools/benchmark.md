# Benchmark工具
## Linux / macOS / Ubuntu
[从源码编译](../compile/tools.html#benchmark)，然后执行如下命令:
```bash
./benchmark.out models_folder loop_count warm_up_count forwardtype numberThread precision weightSparsity weightSparseBlockNumber testQuantizdModel
```
参数如下:
- models_folder: benchmark models文件夹，[benchmark models](https://github.com/alibaba/MNN/tree/master/benchmark/models)。
- loop_count: 可选，默认是10
- warm_up_count: 预热次数
- forwardtype: 可选，默认是0，即CPU，forwardtype有0->CPU，1->Metal，3->OpenCL，6->OpenGL，7->Vulkan
- numberThread: 可选，默认是4，为 CPU 线程数或者 GPU 的运行模式
- precision: 可选，默认是2，有效输入为：0(Normal), 1(High), 2(Low_FP16), 3(Low_BF16)
- weightSparsity: 可选，默认是 0.0 ，在 weightSparsity > 0.5 时且后端支持时，开启稀疏计算
- weightSparseBlockNumber: 可选，默认是 1 ，仅当 weightSparsity > 0.5 时生效，为稀疏计算 block 大小，越大越有利于稀疏计算的加速，一般选择 1, 4, 8, 16
- testQuantizedModel 可选，默认是0，即只测试浮点模型；取1时，会在测试浮点模型后进行量化模型的测试
## Android
在[benchmark目录](https://github.com/alibaba/MNN/tree/master/benchmark/android)下直接执行脚本`bench_android.sh`，默认编译armv7，加参数-64编译armv8，参数-p将[benchmarkModels](https://github.com/alibaba/MNN/tree/master/benchmark/models) push到机器上。
脚本执行完成在[benchmark目录](https://github.com/alibaba/MNN/tree/master/benchmark/android)下得到测试结果`benchmark.txt`
## iOS
1. 先准备模型文件，进入tools/script目录下执行脚本`get_model.sh`；
2. 打开demo/iOS目录下的demo工程，点击benchmark；可通过底部工具栏切换模型、推理类型、线程数。
## 基于表达式构建模型的Benchmark
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
