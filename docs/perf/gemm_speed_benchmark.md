# GEMM 性能基准测试 (GemmSpeed)

`test/speed/GemmSpeed.cpp` 提供了一组通用的 GEMM 性能基准测试，用于评估不同后端（CPU / OpenCL / Vulkan / CUDA / Metal）在 LLM 推理典型矩阵尺寸下的计算吞吐量。

## 功能特性

- **多后端支持**：CPU、OpenCL（Image / Buffer）、Vulkan、CUDA、Metal
- **多精度模式**：Float（FP16/FP32）、Int8-Block0 量化（全通道）、Int4-Block64 量化
- **GPU 时间戳**：在 GPU 后端开启 `MNN_GPU_TIME_PROFILE` + `MNN_GPU_PROFILE_SILENT` 后，同时输出总耗时和 GPU Kernel 耗时
- **Warmup 机制**：3 次 warmup 后再进行 10 次 benchmark 取平均，确保 GPU 时钟稳定
- **灵活的 M 值**：M=8/32/128/512 对应不同 prefill 长度（GEMM）

## 编译

需要开启以下 CMake 选项：

```bash
# 基础编译（CPU 后端）
cmake .. -DMNN_BUILD_TEST=ON -DMNN_LOW_MEMORY=ON

# 开启 GPU 后端 + 性能 Profile
cmake .. -DMNN_BUILD_TEST=ON -DMNN_LOW_MEMORY=ON \
         -DMNN_OPENCL=ON -DMNN_VULKAN=ON \
         -DMNN_GPU_TIME_PROFILE=ON -DMNN_GPU_PROFILE_SILENT=ON

make -j$(nproc) run_test.out
```

> **关键编译宏说明**：
> - `MNN_LOW_MEMORY=ON`：启用低内存模式，支持 Int4/Int8 权值量化推理
> - `MNN_GPU_TIME_PROFILE=ON`：开启 GPU Kernel 时间统计
> - `MNN_GPU_PROFILE_SILENT=ON`：仅累计总耗时，不打印每个 Kernel 的详细信息，通过 `Executor::getLastGpuTimeMs()` 获取 GPU 总耗时

## 测试用例

| 测试名 | 说明 |
|--------|------|
| `speed/GemmSpeedFloat` | 浮点 Conv1x1 GEMM |
| `speed/GemmSpeedInt8`  | Int8-Block0 量化 Conv1x1 GEMM（blockSize=K，全通道量化） |
| `speed/GemmSpeedInt4`  | Int4-Block64 量化 Conv1x1 GEMM |
| `speed/GemmSpeedAll`   | 以上三种模式综合测试 |

## 使用方法

```bash
./run_test.out <测试名> [后端类型] [精度] [线程数/GPU选项]
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 后端类型 | 0=CPU, 3=OpenCL, 7=Vulkan, 1=Metal, 2=CUDA | 0 |
| 精度 | 0=Normal, 1=High, 2=Low(FP16) | 0 |
| 线程数/GPU选项 | CPU: 线程数; OpenCL: 可用位掩码组合 GPU 内存模式和 tuning 策略 | 1 |

### OpenCL 线程数参数位掩码

OpenCL 后端的第三个参数是位掩码组合：

| 位 | 值 | 说明 |
|----|-----|------|
| GPU 内存模式 | 64 (`MNN_GPU_MEMORY_BUFFER`) | 使用 Buffer 模式（默认 Image） |
| Tuning 策略 | 4 (`MNN_GPU_TUNING_FAST`) | 快速 tuning |

例如 `68 = 64 + 4` 表示 OpenCL Buffer 模式 + 快速 tuning。

### 示例

```bash
# CPU 后端，默认精度
./run_test.out speed/GemmSpeedAll

# CPU 后端，FP16 精度，4 线程
./run_test.out speed/GemmSpeedAll 0 2 4

# OpenCL Buffer 模式，FP16 精度
./run_test.out speed/GemmSpeedAll 3 2 68

# Vulkan 后端，FP16 精度
./run_test.out speed/GemmSpeedAll 7 2

# 仅测试 Int4 量化
./run_test.out speed/GemmSpeedInt4 3 2 68

# 仅测试浮点
./run_test.out speed/GemmSpeedFloat 7 2
```

### Android 设备上运行

```bash
# 交叉编译 (aarch64)
mkdir build_android && cd build_android
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 \
         -DMNN_BUILD_TEST=ON -DMNN_LOW_MEMORY=ON \
         -DMNN_OPENCL=ON -DMNN_VULKAN=ON \
         -DMNN_GPU_TIME_PROFILE=ON -DMNN_GPU_PROFILE_SILENT=ON
make -j$(nproc) run_test.out

# 推送到设备
adb push run_test.out /data/local/tmp/
adb push libMNN.so /data/local/tmp/

# 运行测试
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./run_test.out speed/GemmSpeedAll 3 2 68"
```

## 输出格式

### CPU 后端输出
```
  float-gemm                    M=32    K=2560  N=4096   avg=3.768 ms  178.11 GFLOPS
```

### GPU 后端输出（同时显示总耗时和 GPU Kernel 耗时）
```
  int4b64-gemm                  M=32    K=2560  N=4096   total=1.633 ms (410.85 GFLOPS)  gpu=0.745 ms (900.92 GFLOPS)
```

- **total**: 总运行耗时（包含 CPU↔GPU 数据传输、命令提交、同步等全部开销）
- **gpu**: GPU Kernel 纯计算耗时（通过 GPU 硬件时间戳测量，不含传输和同步开销）
- **GFLOPS**: 基于 `2×M×K×N` FLOPs 计算的等效吞吐量

## 默认测试尺寸

当前配置的矩阵尺寸（K×N）：

| K | N | M 范围 |
|---|---|--------|
| 2560 | 4096 | 8, 32, 128, 512 |
| 2560 | 1024 | 8, 32, 128, 512 |
| 4096 | 2560 | 8, 32, 128, 512 |
| 2560 | 9728 | 8, 32, 128, 512 |
| 9728 | 2560 | 8, 32, 128, 512 |

M 值对应不同 prefill 长度（GEMM）。

## 示例数据（Snapdragon 8 Gen5）

以下数据在搭载 Snapdragon 8 Gen5 的设备上测试，4 线程，precision=2（FP16）。

### CPU 后端

```
===== All GEMM Speed Benchmark =====
Backend: CPU (type=0), precision=2, numthread=4

--- K=2560 N=4096 ---
  float-gemm                    M=8     K=2560  N=4096   avg=1.029 ms  163.04 GFLOPS
  float-gemm                    M=32    K=2560  N=4096   avg=4.026 ms  166.71 GFLOPS
  float-gemm                    M=128   K=2560  N=4096   avg=12.399 ms  216.51 GFLOPS
  float-gemm                    M=512   K=2560  N=4096   avg=21.120 ms  508.41 GFLOPS
  int8b0-gemm                   M=8     K=2560  N=4096   avg=0.456 ms  368.24 GFLOPS
  int8b0-gemm                   M=32    K=2560  N=4096   avg=1.515 ms  442.99 GFLOPS
  int8b0-gemm                   M=128   K=2560  N=4096   avg=2.442 ms  1099.38 GFLOPS
  int8b0-gemm                   M=512   K=2560  N=4096   avg=13.529 ms  793.65 GFLOPS
  int4b64-gemm                  M=8     K=2560  N=4096   avg=1.038 ms  161.65 GFLOPS
  int4b64-gemm                  M=32    K=2560  N=4096   avg=5.321 ms  126.11 GFLOPS
  int4b64-gemm                  M=128   K=2560  N=4096   avg=10.988 ms  244.30 GFLOPS
  int4b64-gemm                  M=512   K=2560  N=4096   avg=17.067 ms  629.13 GFLOPS

--- K=2560 N=1024 ---
  float-gemm                    M=8     K=2560  N=1024   avg=0.125 ms  335.28 GFLOPS
  float-gemm                    M=32    K=2560  N=1024   avg=0.547 ms  306.83 GFLOPS
  float-gemm                    M=128   K=2560  N=1024   avg=2.169 ms  309.40 GFLOPS
  float-gemm                    M=512   K=2560  N=1024   avg=7.634 ms  351.65 GFLOPS
  int8b0-gemm                   M=8     K=2560  N=1024   avg=0.053 ms  788.40 GFLOPS
  int8b0-gemm                   M=32    K=2560  N=1024   avg=0.426 ms  393.74 GFLOPS
  int8b0-gemm                   M=128   K=2560  N=1024   avg=0.992 ms  676.23 GFLOPS
  int8b0-gemm                   M=512   K=2560  N=1024   avg=2.418 ms  1110.15 GFLOPS
  int4b64-gemm                  M=8     K=2560  N=1024   avg=0.085 ms  494.61 GFLOPS
  int4b64-gemm                  M=32    K=2560  N=1024   avg=0.317 ms  530.09 GFLOPS
  int4b64-gemm                  M=128   K=2560  N=1024   avg=1.753 ms  382.87 GFLOPS
  int4b64-gemm                  M=512   K=2560  N=1024   avg=4.122 ms  651.23 GFLOPS

--- K=4096 N=2560 ---
  float-gemm                    M=8     K=4096  N=2560   avg=0.522 ms  321.65 GFLOPS
  float-gemm                    M=32    K=4096  N=2560   avg=2.181 ms  307.67 GFLOPS
  float-gemm                    M=128   K=4096  N=2560   avg=7.070 ms  379.70 GFLOPS
  float-gemm                    M=512   K=4096  N=2560   avg=31.386 ms  342.10 GFLOPS
  int8b0-gemm                   M=8     K=4096  N=2560   avg=0.111 ms  1508.74 GFLOPS
  int8b0-gemm                   M=32    K=4096  N=2560   avg=0.567 ms  1184.00 GFLOPS
  int8b0-gemm                   M=128   K=4096  N=2560   avg=2.709 ms  990.98 GFLOPS
  int8b0-gemm                   M=512   K=4096  N=2560   avg=8.531 ms  1258.71 GFLOPS
  int4b64-gemm                  M=8     K=4096  N=2560   avg=0.285 ms  588.06 GFLOPS
  int4b64-gemm                  M=32    K=4096  N=2560   avg=1.118 ms  600.42 GFLOPS
  int4b64-gemm                  M=128   K=4096  N=2560   avg=4.441 ms  604.41 GFLOPS
  int4b64-gemm                  M=512   K=4096  N=2560   avg=15.934 ms  673.86 GFLOPS

--- K=2560 N=9728 ---
  float-gemm                    M=8     K=2560  N=9728   avg=1.299 ms  306.65 GFLOPS
  float-gemm                    M=32    K=2560  N=9728   avg=5.162 ms  308.74 GFLOPS
  float-gemm                    M=128   K=2560  N=9728   avg=18.113 ms  351.97 GFLOPS
  float-gemm                    M=512   K=2560  N=9728   avg=62.028 ms  411.13 GFLOPS
  int8b0-gemm                   M=8     K=2560  N=9728   avg=0.280 ms  1425.10 GFLOPS
  int8b0-gemm                   M=32    K=2560  N=9728   avg=1.344 ms  1186.24 GFLOPS
  int8b0-gemm                   M=128   K=2560  N=9728   avg=5.185 ms  1229.53 GFLOPS
  int8b0-gemm                   M=512   K=2560  N=9728   avg=19.792 ms  1288.48 GFLOPS
  int4b64-gemm                  M=8     K=2560  N=9728   avg=1.180 ms  337.79 GFLOPS
  int4b64-gemm                  M=32    K=2560  N=9728   avg=2.854 ms  558.44 GFLOPS
  int4b64-gemm                  M=128   K=2560  N=9728   avg=10.670 ms  597.51 GFLOPS
  int4b64-gemm                  M=512   K=2560  N=9728   avg=28.721 ms  887.89 GFLOPS

--- K=9728 N=2560 ---
  float-gemm                    M=8     K=9728  N=2560   avg=0.780 ms  510.91 GFLOPS
  float-gemm                    M=32    K=9728  N=2560   avg=3.247 ms  490.92 GFLOPS
  float-gemm                    M=128   K=9728  N=2560   avg=16.341 ms  390.14 GFLOPS
  float-gemm                    M=512   K=9728  N=2560   avg=82.454 ms  309.28 GFLOPS
  int8b0-gemm                   M=8     K=9728  N=2560   avg=0.416 ms  957.14 GFLOPS
  int8b0-gemm                   M=32    K=9728  N=2560   avg=1.881 ms  847.20 GFLOPS
  int8b0-gemm                   M=128   K=9728  N=2560   avg=5.266 ms  1210.57 GFLOPS
  int8b0-gemm                   M=512   K=9728  N=2560   avg=17.917 ms  1423.30 GFLOPS
  int4b64-gemm                  M=8     K=9728  N=2560   avg=0.794 ms  502.09 GFLOPS
  int4b64-gemm                  M=32    K=9728  N=2560   avg=2.400 ms  663.99 GFLOPS
  int4b64-gemm                  M=128   K=9728  N=2560   avg=10.050 ms  634.34 GFLOPS
  int4b64-gemm                  M=512   K=9728  N=2560   avg=34.161 ms  746.51 GFLOPS
```

### Vulkan 后端

```
===== All GEMM Speed Benchmark =====
Backend: Vulkan (type=7), precision=2, numthread=4

--- K=2560 N=4096 ---
  float-gemm                    M=8     K=2560  N=4096   total=0.893 ms (187.83 GFLOPS)  gpu=0.318 ms (528.16 GFLOPS)
  float-gemm                    M=32    K=2560  N=4096   total=0.986 ms (680.55 GFLOPS)  gpu=0.324 ms (2070.53 GFLOPS)
  float-gemm                    M=128   K=2560  N=4096   total=1.870 ms (1435.33 GFLOPS)  gpu=0.499 ms (5374.87 GFLOPS)
  float-gemm                    M=512   K=2560  N=4096   total=7.234 ms (1484.38 GFLOPS)  gpu=1.999 ms (5372.49 GFLOPS)
  int8b0-gemm                   M=8     K=2560  N=4096   total=0.881 ms (190.50 GFLOPS)  gpu=0.307 ms (545.88 GFLOPS)
  int8b0-gemm                   M=32    K=2560  N=4096   total=0.983 ms (682.90 GFLOPS)  gpu=0.320 ms (2098.86 GFLOPS)
  int8b0-gemm                   M=128   K=2560  N=4096   total=2.180 ms (1231.41 GFLOPS)  gpu=0.862 ms (3115.49 GFLOPS)
  int8b0-gemm                   M=512   K=2560  N=4096   total=5.382 ms (1995.21 GFLOPS)  gpu=1.426 ms (7530.90 GFLOPS)
  int4b64-gemm                  M=8     K=2560  N=4096   total=1.315 ms (127.60 GFLOPS)  gpu=0.730 ms (229.87 GFLOPS)
  int4b64-gemm                  M=32    K=2560  N=4096   total=1.469 ms (456.80 GFLOPS)  gpu=0.731 ms (917.86 GFLOPS)
  int4b64-gemm                  M=128   K=2560  N=4096   total=2.293 ms (1170.57 GFLOPS)  gpu=0.909 ms (2954.41 GFLOPS)
  int4b64-gemm                  M=512   K=2560  N=4096   total=7.697 ms (1394.96 GFLOPS)  gpu=2.478 ms (4332.24 GFLOPS)

--- K=2560 N=1024 ---
  float-gemm                    M=8     K=2560  N=1024   total=0.743 ms (56.46 GFLOPS)  gpu=0.192 ms (218.54 GFLOPS)
  float-gemm                    M=32    K=2560  N=1024   total=0.803 ms (208.93 GFLOPS)  gpu=0.191 ms (879.64 GFLOPS)
  float-gemm                    M=128   K=2560  N=1024   total=0.951 ms (705.52 GFLOPS)  gpu=0.181 ms (3714.30 GFLOPS)
  float-gemm                    M=512   K=2560  N=1024   total=2.425 ms (1106.81 GFLOPS)  gpu=0.539 ms (4984.01 GFLOPS)
  int8b0-gemm                   M=8     K=2560  N=1024   total=0.695 ms (60.38 GFLOPS)  gpu=0.137 ms (307.02 GFLOPS)
  int8b0-gemm                   M=32    K=2560  N=1024   total=0.742 ms (226.02 GFLOPS)  gpu=0.138 ms (1219.24 GFLOPS)
  int8b0-gemm                   M=128   K=2560  N=1024   total=0.988 ms (678.90 GFLOPS)  gpu=0.147 ms (4575.60 GFLOPS)
  int8b0-gemm                   M=512   K=2560  N=1024   total=2.419 ms (1109.79 GFLOPS)  gpu=0.443 ms (6062.77 GFLOPS)
  int4b64-gemm                  M=8     K=2560  N=1024   total=0.858 ms (48.86 GFLOPS)  gpu=0.292 ms (143.57 GFLOPS)
  int4b64-gemm                  M=32    K=2560  N=1024   total=0.907 ms (184.89 GFLOPS)  gpu=0.292 ms (573.99 GFLOPS)
  int4b64-gemm                  M=128   K=2560  N=1024   total=1.070 ms (627.24 GFLOPS)  gpu=0.290 ms (2314.93 GFLOPS)
  int4b64-gemm                  M=512   K=2560  N=1024   total=2.760 ms (972.77 GFLOPS)  gpu=0.649 ms (4133.42 GFLOPS)

--- K=4096 N=2560 ---
  float-gemm                    M=8     K=4096  N=2560   total=0.919 ms (182.66 GFLOPS)  gpu=0.342 ms (490.97 GFLOPS)
  float-gemm                    M=32    K=4096  N=2560   total=0.952 ms (704.70 GFLOPS)  gpu=0.326 ms (2057.96 GFLOPS)
  float-gemm                    M=128   K=4096  N=2560   total=1.845 ms (1455.01 GFLOPS)  gpu=0.522 ms (5147.27 GFLOPS)
  float-gemm                    M=512   K=4096  N=2560   total=6.653 ms (1613.92 GFLOPS)  gpu=1.876 ms (5723.28 GFLOPS)
  int8b0-gemm                   M=8     K=4096  N=2560   total=1.102 ms (152.19 GFLOPS)  gpu=0.506 ms (331.44 GFLOPS)
  int8b0-gemm                   M=32    K=4096  N=2560   total=1.244 ms (539.29 GFLOPS)  gpu=0.523 ms (1283.48 GFLOPS)
  int8b0-gemm                   M=128   K=4096  N=2560   total=2.343 ms (1145.84 GFLOPS)  gpu=1.047 ms (2563.52 GFLOPS)
  int8b0-gemm                   M=512   K=4096  N=2560   total=7.630 ms (1407.23 GFLOPS)  gpu=2.670 ms (4022.05 GFLOPS)
  int4b64-gemm                  M=8     K=4096  N=2560   total=2.601 ms (64.51 GFLOPS)  gpu=1.806 ms (92.91 GFLOPS)
  int4b64-gemm                  M=32    K=4096  N=2560   total=2.822 ms (237.81 GFLOPS)  gpu=1.817 ms (369.42 GFLOPS)
  int4b64-gemm                  M=128   K=4096  N=2560   total=3.342 ms (803.15 GFLOPS)  gpu=2.000 ms (1342.07 GFLOPS)
  int4b64-gemm                  M=512   K=4096  N=2560   total=10.797 ms (994.52 GFLOPS)  gpu=3.345 ms (3210.24 GFLOPS)

--- K=2560 N=9728 ---
  float-gemm                    M=8     K=2560  N=9728   total=1.391 ms (286.52 GFLOPS)  gpu=0.786 ms (507.15 GFLOPS)
  float-gemm                    M=32    K=2560  N=9728   total=1.689 ms (943.82 GFLOPS)  gpu=0.785 ms (2029.29 GFLOPS)
  float-gemm                    M=128   K=2560  N=9728   total=3.329 ms (1915.32 GFLOPS)  gpu=1.087 ms (5865.19 GFLOPS)
  float-gemm                    M=512   K=2560  N=9728   total=18.069 ms (1411.35 GFLOPS)  gpu=4.328 ms (5891.94 GFLOPS)
  int8b0-gemm                   M=8     K=2560  N=9728   total=1.572 ms (253.52 GFLOPS)  gpu=0.932 ms (427.59 GFLOPS)
  int8b0-gemm                   M=32    K=2560  N=9728   total=1.904 ms (837.14 GFLOPS)  gpu=0.954 ms (1671.40 GFLOPS)
  int8b0-gemm                   M=128   K=2560  N=9728   total=3.311 ms (1925.50 GFLOPS)  gpu=1.409 ms (4526.03 GFLOPS)
  int8b0-gemm                   M=512   K=2560  N=9728   total=12.127 ms (2102.79 GFLOPS)  gpu=2.902 ms (8787.89 GFLOPS)
  int4b64-gemm                  M=8     K=2560  N=9728   total=2.773 ms (143.71 GFLOPS)  gpu=1.787 ms (223.02 GFLOPS)
  int4b64-gemm                  M=32    K=2560  N=9728   total=3.081 ms (517.28 GFLOPS)  gpu=1.811 ms (879.99 GFLOPS)
  int4b64-gemm                  M=128   K=2560  N=9728   total=6.163 ms (1034.39 GFLOPS)  gpu=2.094 ms (3044.86 GFLOPS)
  int4b64-gemm                  M=512   K=2560  N=9728   total=21.150 ms (1205.73 GFLOPS)  gpu=5.609 ms (4546.54 GFLOPS)

--- K=9728 N=2560 ---
  float-gemm                    M=8     K=9728  N=2560   total=1.371 ms (290.63 GFLOPS)  gpu=0.783 ms (508.57 GFLOPS)
  float-gemm                    M=32    K=9728  N=2560   total=1.561 ms (1020.77 GFLOPS)  gpu=0.735 ms (2167.87 GFLOPS)
  float-gemm                    M=128   K=9728  N=2560   total=2.792 ms (2283.10 GFLOPS)  gpu=1.191 ms (5354.38 GFLOPS)
  float-gemm                    M=512   K=9728  N=2560   total=19.305 ms (1320.95 GFLOPS)  gpu=4.280 ms (5958.92 GFLOPS)
  int8b0-gemm                   M=8     K=9728  N=2560   total=1.540 ms (258.69 GFLOPS)  gpu=0.686 ms (580.46 GFLOPS)
  int8b0-gemm                   M=32    K=9728  N=2560   total=1.486 ms (1072.78 GFLOPS)  gpu=0.739 ms (2157.48 GFLOPS)
  int8b0-gemm                   M=128   K=9728  N=2560   total=3.242 ms (1966.18 GFLOPS)  gpu=1.426 ms (4470.17 GFLOPS)
  int8b0-gemm                   M=512   K=9728  N=2560   total=13.072 ms (1950.78 GFLOPS)  gpu=2.789 ms (9144.54 GFLOPS)
  int4b64-gemm                  M=8     K=9728  N=2560   total=2.513 ms (158.59 GFLOPS)  gpu=1.834 ms (217.27 GFLOPS)
  int4b64-gemm                  M=32    K=9728  N=2560   total=3.157 ms (504.89 GFLOPS)  gpu=1.788 ms (891.29 GFLOPS)
  int4b64-gemm                  M=128   K=9728  N=2560   total=4.254 ms (1498.74 GFLOPS)  gpu=2.220 ms (2871.71 GFLOPS)
  int4b64-gemm                  M=512   K=9728  N=2560   total=27.827 ms (916.43 GFLOPS)  gpu=5.556 ms (4589.72 GFLOPS)
```

### OpenCL Buffer 后端

```
===== All GEMM Speed Benchmark =====
Backend: OpenCL (type=3), precision=2, numthread=68
  OpenCL memory mode: BUFFER

--- K=2560 N=4096 ---
  float-gemm                    M=8     K=2560  N=4096   total=0.985 ms (170.28 GFLOPS)  gpu=0.593 ms (282.92 GFLOPS)
  float-gemm                    M=32    K=2560  N=4096   total=1.711 ms (392.17 GFLOPS)  gpu=1.236 ms (542.95 GFLOPS)
  float-gemm                    M=128   K=2560  N=4096   total=3.138 ms (855.41 GFLOPS)  gpu=2.184 ms (1229.10 GFLOPS)
  float-gemm                    M=512   K=2560  N=4096   total=9.844 ms (1090.79 GFLOPS)  gpu=7.226 ms (1485.94 GFLOPS)
  int8b0-gemm                   M=8     K=2560  N=4096   total=0.697 ms (240.57 GFLOPS)  gpu=0.309 ms (542.95 GFLOPS)
  int8b0-gemm                   M=32    K=2560  N=4096   total=1.054 ms (636.71 GFLOPS)  gpu=0.674 ms (995.68 GFLOPS)
  int8b0-gemm                   M=128   K=2560  N=4096   total=2.343 ms (1145.89 GFLOPS)  gpu=1.722 ms (1558.86 GFLOPS)
  int8b0-gemm                   M=512   K=2560  N=4096   total=8.883 ms (1208.79 GFLOPS)  gpu=5.864 ms (1831.07 GFLOPS)
  int4b64-gemm                  M=8     K=2560  N=4096   total=0.551 ms (304.71 GFLOPS)  gpu=0.180 ms (932.07 GFLOPS)
  int4b64-gemm                  M=32    K=2560  N=4096   total=0.948 ms (707.68 GFLOPS)  gpu=0.546 ms (1229.10 GFLOPS)
  int4b64-gemm                  M=128   K=2560  N=4096   total=1.848 ms (1452.57 GFLOPS)  gpu=1.272 ms (2110.34 GFLOPS)
  int4b64-gemm                  M=512   K=2560  N=4096   total=7.339 ms (1463.02 GFLOPS)  gpu=4.737 ms (2266.71 GFLOPS)

--- K=2560 N=1024 ---
  float-gemm                    M=8     K=2560  N=1024   total=0.572 ms (73.28 GFLOPS)  gpu=0.221 ms (189.79 GFLOPS)
  float-gemm                    M=32    K=2560  N=1024   total=0.765 ms (219.31 GFLOPS)  gpu=0.388 ms (432.40 GFLOPS)
  float-gemm                    M=128   K=2560  N=1024   total=1.173 ms (571.97 GFLOPS)  gpu=0.717 ms (935.97 GFLOPS)
  float-gemm                    M=512   K=2560  N=1024   total=3.435 ms (781.40 GFLOPS)  gpu=2.325 ms (1154.56 GFLOPS)
  int8b0-gemm                   M=8     K=2560  N=1024   total=0.422 ms (99.41 GFLOPS)  gpu=0.078 ms (537.73 GFLOPS)
  int8b0-gemm                   M=32    K=2560  N=1024   total=0.739 ms (227.09 GFLOPS)  gpu=0.361 ms (464.74 GFLOPS)
  int8b0-gemm                   M=128   K=2560  N=1024   total=1.094 ms (613.26 GFLOPS)  gpu=0.622 ms (1078.92 GFLOPS)
  int8b0-gemm                   M=512   K=2560  N=1024   total=2.652 ms (1012.01 GFLOPS)  gpu=1.702 ms (1577.18 GFLOPS)
  int4b64-gemm                  M=8     K=2560  N=1024   total=0.403 ms (104.10 GFLOPS)  gpu=0.055 ms (762.60 GFLOPS)
  int4b64-gemm                  M=32    K=2560  N=1024   total=0.717 ms (233.99 GFLOPS)  gpu=0.331 ms (506.86 GFLOPS)
  int4b64-gemm                  M=128   K=2560  N=1024   total=0.954 ms (703.23 GFLOPS)  gpu=0.513 ms (1308.16 GFLOPS)
  int4b64-gemm                  M=512   K=2560  N=1024   total=2.460 ms (1091.25 GFLOPS)  gpu=1.265 ms (2122.02 GFLOPS)

--- K=4096 N=2560 ---
  float-gemm                    M=8     K=4096  N=2560   total=1.082 ms (154.99 GFLOPS)  gpu=0.696 ms (241.05 GFLOPS)
  float-gemm                    M=32    K=4096  N=2560   total=1.868 ms (359.33 GFLOPS)  gpu=1.312 ms (511.50 GFLOPS)
  float-gemm                    M=128   K=4096  N=2560   total=3.061 ms (876.92 GFLOPS)  gpu=2.285 ms (1174.77 GFLOPS)
  float-gemm                    M=512   K=4096  N=2560   total=10.772 ms (996.82 GFLOPS)  gpu=7.677 ms (1398.65 GFLOPS)
  int8b0-gemm                   M=8     K=4096  N=2560   total=0.682 ms (246.14 GFLOPS)  gpu=0.302 ms (555.54 GFLOPS)
  int8b0-gemm                   M=32    K=4096  N=2560   total=1.265 ms (530.46 GFLOPS)  gpu=0.823 ms (815.42 GFLOPS)
  int8b0-gemm                   M=128   K=4096  N=2560   total=2.560 ms (1048.45 GFLOPS)  gpu=1.915 ms (1401.75 GFLOPS)
  int8b0-gemm                   M=512   K=4096  N=2560   total=8.734 ms (1229.40 GFLOPS)  gpu=6.061 ms (1771.56 GFLOPS)
  int4b64-gemm                  M=8     K=4096  N=2560   total=0.538 ms (311.79 GFLOPS)  gpu=0.173 ms (969.78 GFLOPS)
  int4b64-gemm                  M=32    K=4096  N=2560   total=1.069 ms (627.89 GFLOPS)  gpu=0.667 ms (1006.13 GFLOPS)
  int4b64-gemm                  M=128   K=4096  N=2560   total=1.932 ms (1389.56 GFLOPS)  gpu=1.332 ms (2015.28 GFLOPS)
  int4b64-gemm                  M=512   K=4096  N=2560   total=7.090 ms (1514.47 GFLOPS)  gpu=4.701 ms (2284.07 GFLOPS)

--- K=2560 N=9728 ---
  float-gemm                    M=8     K=2560  N=9728   total=1.816 ms (219.38 GFLOPS)  gpu=1.406 ms (283.40 GFLOPS)
  float-gemm                    M=32    K=2560  N=9728   total=3.528 ms (451.72 GFLOPS)  gpu=2.837 ms (561.80 GFLOPS)
  float-gemm                    M=128   K=2560  N=9728   total=6.585 ms (968.18 GFLOPS)  gpu=4.811 ms (1325.16 GFLOPS)
  float-gemm                    M=512   K=2560  N=9728   total=21.317 ms (1196.28 GFLOPS)  gpu=16.770 ms (1520.65 GFLOPS)
  int8b0-gemm                   M=8     K=2560  N=9728   total=1.133 ms (351.62 GFLOPS)  gpu=0.728 ms (547.33 GFLOPS)
  int8b0-gemm                   M=32    K=2560  N=9728   total=1.725 ms (923.86 GFLOPS)  gpu=1.285 ms (1240.34 GFLOPS)
  int8b0-gemm                   M=128   K=2560  N=9728   total=5.108 ms (1248.04 GFLOPS)  gpu=3.575 ms (1783.31 GFLOPS)
  int8b0-gemm                   M=512   K=2560  N=9728   total=20.004 ms (1274.81 GFLOPS)  gpu=13.406 ms (1902.24 GFLOPS)
  int4b64-gemm                  M=8     K=2560  N=9728   total=0.827 ms (481.93 GFLOPS)  gpu=0.432 ms (922.36 GFLOPS)
  int4b64-gemm                  M=32    K=2560  N=9728   total=1.283 ms (1242.37 GFLOPS)  gpu=0.860 ms (1853.30 GFLOPS)
  int4b64-gemm                  M=128   K=2560  N=9728   total=4.089 ms (1559.34 GFLOPS)  gpu=2.894 ms (2202.95 GFLOPS)
  int4b64-gemm                  M=512   K=2560  N=9728   total=16.696 ms (1527.40 GFLOPS)  gpu=10.786 ms (2364.30 GFLOPS)

--- K=9728 N=2560 ---
  float-gemm                    M=8     K=9728  N=2560   total=1.986 ms (200.66 GFLOPS)  gpu=1.619 ms (246.11 GFLOPS)
  float-gemm                    M=32    K=9728  N=2560   total=3.999 ms (398.53 GFLOPS)  gpu=3.167 ms (503.26 GFLOPS)
  float-gemm                    M=128   K=9728  N=2560   total=7.164 ms (889.93 GFLOPS)  gpu=5.509 ms (1157.26 GFLOPS)
  float-gemm                    M=512   K=9728  N=2560   total=27.432 ms (929.63 GFLOPS)  gpu=19.016 ms (1341.05 GFLOPS)
  int8b0-gemm                   M=8     K=9728  N=2560   total=1.133 ms (351.56 GFLOPS)  gpu=0.730 ms (545.83 GFLOPS)
  int8b0-gemm                   M=32    K=9728  N=2560   total=2.821 ms (564.91 GFLOPS)  gpu=2.007 ms (794.14 GFLOPS)
  int8b0-gemm                   M=128   K=9728  N=2560   total=6.379 ms (999.46 GFLOPS)  gpu=4.698 ms (1357.03 GFLOPS)
  int8b0-gemm                   M=512   K=9728  N=2560   total=20.003 ms (1274.90 GFLOPS)  gpu=14.573 ms (1749.91 GFLOPS)
  int4b64-gemm                  M=8     K=9728  N=2560   total=0.826 ms (482.16 GFLOPS)  gpu=0.435 ms (916.00 GFLOPS)
  int4b64-gemm                  M=32    K=9728  N=2560   total=2.261 ms (704.77 GFLOPS)  gpu=1.729 ms (921.83 GFLOPS)
  int4b64-gemm                  M=128   K=9728  N=2560   total=4.564 ms (1396.91 GFLOPS)  gpu=3.175 ms (2007.98 GFLOPS)
  int4b64-gemm                  M=512   K=9728  N=2560   total=17.008 ms (1499.40 GFLOPS)  gpu=11.313 ms (2254.17 GFLOPS)
```

## 自定义尺寸

如需修改测试尺寸，编辑 `test/speed/GemmSpeed.cpp` 中的 `defaultConfigs()` 函数：

```cpp
static const std::vector<ShapeConfig>& defaultConfigs() {
    static std::vector<ShapeConfig> configs = {
        // {K, N, maxM, "label"}
        // maxM=0 表示测试所有 M 值，maxM>0 表示仅测试 M<=maxM
        {2560,   4096,   0, "K=2560 N=4096"},
        {2560,   9728,   0, "K=2560 N=9728"},
        // 添加更多尺寸...
    };
    return configs;
}