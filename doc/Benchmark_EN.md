[中文版本](Benchmark_CN.md)

# How to benchmark
## Mac or Linux
```bash
# in root directory of MNN
mkdir build
cd build
cmake .. && make -j4
```

then execute the commmand:
```bash
./benchmark.out models_folder [loop_count] [forwardtype] [number_thread] [precision]
```
forwardtype is in these options: 0->CPU, 1->Metal, 3->OpenCL, 6->OpenGL, 7->Vulkan.
precision is in these options: 0->Normal, 1->High, 2->Low.
Here are benchmark models: [models](../benchmark/models).

## Android
You can directly execute the script `bench_android.sh` in the [benchmark directory](../benchmark). It builds in armeabi-v7a  architecture by default, and in arm64-v8a architecture if builds with parameter of arm64-v8a. [BenchmarkModels](../benchmark/models) will be pushed to your device if executed with parameter of -p.

`benchmark.txt` will be generated in [benchmark directory](../benchmark) after the execution.

## iOS
1. Prepare models with running the script `get_model.sh` in the tools/scropt;
2. Open demo project in demo/iOS and run with `Benchmark` button at right-top edge, you can switch model, forward type and thread number for banchmark with bottom toolbar.

# Test Result Of Benchmark
[benchmark result](../benchmark/result)
