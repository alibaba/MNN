[English Version](Benchmark_EN.md)

# Benchmark 测试方法
## Mac or Linux
```bash
# 在MNN根目录下
mkdir build
cd build
cmake .. && make -j4
```

然后执行如下命令:
```bash
./benchmark.out models_folder [loop_count] [forwardtype] [number_thread] [precision]
```
选项如下:
- models_folder: benchmark models文件夹，benchmark models[在此](../benchmark/models)。
- loop_count: 可选，默认是10
- forwardtype: 可选，默认是0，即CPU，forwardtype有0->CPU，1->Metal，3->OpenCL，6->OpenGL，7->Vulkan
- number_thread: 可选，默认是4
- precision: 可选，默认是2，即Low，precision有0->Normal，1->High，2->Low

## Android
在[benchmark目录](../benchmark)下直接执行脚本`bench_android.sh`，默认编译armv7，加参数-64编译armv8，参数-p将[benchmarkModels](../benchmark/models) push到机器上。
脚本执行完成在[benchmark目录](../benchmark)下得到测试结果`benchmark.txt`

## iOS
1. 先准备模型文件，进入tools/script目录下执行脚本`get_model.sh`；
2. 打开demo/iOS目录下的demo工程，点击benchmark；可通过底部工具栏切换模型、推理类型、线程数。

# Benchmark测试结果记录
[benchmark结果](../benchmark/result)
