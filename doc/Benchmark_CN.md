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
./benchmark.out models_folder loop_count forwardtype
```
选项如下:
- models_folder: benchmark models文件夹，benchmark models[在此](../benchmark/models)。
- loop_count: 可选，默认是10
- forwardtype: 可选，默认是0，即CPU，forwardtype有0->CPU，1->Metal，3->OpenCL，6->OpenGL，7->Vulkan

## Android
在[benchmark目录](../benchmark)下直接执行脚本`bench_android.sh`，默认编译armv7，加参数-64编译armv8，参数-p将[benchmarkModels](../benchmark/models) push到机器上。
脚本执行完成在[benchmark目录](../benchmark)下得到测试结果`benchmark.txt`

## iOS
1. 先准备模型文件，进入tools/script目录下执行脚本`get_model.sh`；
2. 打开demo/iOS目录下的demo工程，点击benchmark；可通过底部工具栏切换模型、推理类型、线程数。

# Benchmark测试结果记录
## 2019-2-18
### 华为 P10

```bash
Build Flags: ABI=arm64-v8a OpenMP=ON Vulkan=ON OpenCL=ON
MNN benchmark
Forward type: **CPU**
Warming up...
--------> Benchmarking... loop = 10
[ - ] vgg16.mnn                 max =  499.545ms  min =  410.570ms  avg =  445.868ms
[ - ] SqueezeNetV1.0.mnn        max =   49.437ms  min =   38.759ms  avg =   43.901ms
[ - ] MobileNetV2_224.mnn       max =   26.139ms  min =   20.400ms  avg =   24.489ms
[ - ] inception-v3.mnn          max =  413.265ms  min =  262.142ms  avg =  306.542ms
[ - ] resnet-v2-50.mnn          max =  240.009ms  min =  152.649ms  avg =  176.075ms
[ - ] mobilenet-v1-1.0.mnn      max =   89.461ms  min =   29.903ms  avg =   41.547ms
MNN benchmark
Forward type: **Vulkan**
Warming up...
--------> Benchmarking... loop = 10
[ - ] vgg16.mnn                 max =  293.156ms  min =  227.952ms  avg =  240.050ms
[ - ] SqueezeNetV1.0.mnn        max =   47.752ms  min =   31.191ms  avg =   37.727ms
[ - ] MobileNetV2_224.mnn       max =   61.352ms  min =   35.874ms  avg =   46.321ms
[ - ] inception-v3.mnn          max =  396.939ms  min =  180.353ms  avg =  349.952ms
[ - ] resnet-v2-50.mnn          max =  214.694ms  min =  100.377ms  avg =  169.003ms
[ - ] mobilenet-v1-1.0.mnn      max =   45.946ms  min =   23.257ms  avg =   33.217ms
```

### 小米 Max3

```bash
Hardware	: Qualcomm Technologies, Inc SDM636

Build Flags: ABI=arm64-v8a OpenMP=ON Vulkan=ON OpenCL=ON
MNN benchmark
Forward type: **CPU**
Warming up...
--------> Benchmarking... loop = 10
[ - ] vgg16.mnn                 max = 1311.661ms  min = 1248.531ms  avg = 1255.455ms
[ - ] SqueezeNetV1.0.mnn        max =  151.955ms  min =   95.348ms  avg =  101.986ms
[ - ] MobileNetV2_224.mnn       max =   94.336ms  min =   50.987ms  avg =   58.299ms
[ - ] inception-v3.mnn          max =  763.095ms  min =  690.005ms  avg =  698.674ms
[ - ] resnet-v2-50.mnn          max =  453.710ms  min =  389.649ms  avg =  396.409ms
[ - ] mobilenet-v1-1.0.mnn      max =  128.781ms  min =   77.023ms  avg =   83.134ms
MNN benchmark
Forward type: **Vulkan**
Warming up...
--------> Benchmarking... loop = 10
[ - ] vgg16.mnn                 max =  783.093ms  min =  730.928ms  avg =  736.894ms
[ - ] SqueezeNetV1.0.mnn        max =   96.435ms  min =   61.809ms  avg =   65.574ms
[ - ] MobileNetV2_224.mnn       max =   71.107ms  min =   43.912ms  avg =   46.925ms
[ - ] inception-v3.mnn          max =  436.363ms  min =  386.338ms  avg =  391.818ms
[ - ] resnet-v2-50.mnn          max =  303.728ms  min =  262.706ms  avg =  267.613ms
[ - ] mobilenet-v1-1.0.mnn      max =   89.119ms  min =   56.216ms  avg =   59.725ms
```
