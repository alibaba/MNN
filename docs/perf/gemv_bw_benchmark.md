# GEMV 带宽基准测试 (GemvBW)

`test/speed/GemvBWTest.cpp` 是一个面向 LLM **decode 阶段** (batch=1) 的 GEMV 带宽 microbenchmark，对标 llama.cpp 的 `gemv_roofline.cpp`。固定一个 (M, K) 形状，扫不同 bit 宽 (w8 / w4 / w3 / w2)，输出每种量化下的有效带宽 (effective GB/s)、相对峰值 memcpy 带宽的饱和度 (%peak) 及算术强度 (AI)。

## 功能特性

- **CPU / GPU (Metal) 双后端**：通过 `MNNTestSuite` 的 `forwardType` 参数切换；CPU 跑 w8 / w4 / w3 / w2，Metal 仅跑 w8 / w4（受 `MetalConvolution1x1.mm` 中 `mDequantBits == 4 || == 8` 限制）。
- **峰值带宽 roofline**：通过多线程 memcpy 一块 256 MiB 缓冲区，得到当前线程数下的峰值流式带宽，作为 %peak 的基准。
- **冷缓存测量**：每次迭代前刷一块 64 MiB buffer 强制把权重从 DRAM 重新拉回，避免 L2/L3 命中导致的虚高读数。
- **best-of-3 × N iters**：3 次外层重复，每次内层平均 200 次 cold-cache 迭代，取最优值。
- **W bytes 口径与 llama.cpp 对齐**：仅计算权重 + per-block (alpha + zp, fp16) 元数据；不算输入向量与输出，便于跨实现对比 GEMV 带宽饱和度。

## 编译

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_TEST=ON -DMNN_LOW_MEMORY=ON \
         -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
# Metal 测试请加 -DMNN_METAL=ON
make -j$(nproc) run_test.out
```

> `MNN_LOW_MEMORY=ON` 必开，否则 hybrid 量化路径不可用。

## 使用方法

```bash
./run_test.out speed/GemvBW [precision] [forwardType] [threads]
```

### 参数说明

| 位置 | 含义 | 取值 |
|------|------|------|
| 1 | precision | 0=Normal, 1=High, 2=Low (推荐 2，对应 fp16 累加) |
| 2 | forwardType | 0=CPU, 3=Metal (其余按 `MNNForwardType` 编号) |
| 3 | threads | 默认 4；GPU 后端忽略 |

### 默认形状

```text
M = oc = 4096
K = ic = 14336
blocksize = 64
iters = 200 (best of 3)
```

> 对应 Llama-3-8B FFN 单层一个投影。如需自定义，直接改源码顶部常量重编。

## 输出格式

```text
## GemvBW (backend=CPU, precision=2, blocksize=64)

## Peak streaming bandwidth (memcpy, 256 MiB buffer)
threads | GB/s
-------:|-----:
      4 | 109.7
-> peak 109.7 GB/s @ 4 threads (used as roofline)

## GEMV: y = W(4096x14336) * x(14336), block=64
type | thr |   us/iter |  W MiB | bytes/elem | eff GB/s | %peak |  GFLOPS |  AI (op/B)
-----|----:|----------:|-------:|-----------:|---------:|------:|--------:|----------:
w8   |   4 |     ...   |   ...  |    ...     |   109.4  |  99.7 |   ...   |    ...
w4   |   4 |     ...   |   ...  |    ...     |   100.7  |  91.8 |   ...   |    ...
w3   |   4 |     ...   |   ...  |    ...     |    50.2  |  45.8 |   ...   |    ...
w2   |   4 |     ...   |   ...  |    ...     |    64.5  |  58.8 |   ...   |    ...
```

## 实测数据

### OnePlus PJZ110 (Snapdragon 8 Elite, Oryon, 6 mid + 2 big), Android arm64-v8a, precision=Low (fp16)

i8mm + asimddp + fp16 全开。M=4096, K=14336, blocksize=64。

> **必须用 `taskset c0` 绑双大核 + threads=2 跑**，否则 MNN 默认线程池会随机调度到 mid 核，把 GEMV 拖到大核数据的 1/2~1/3。下表是 `taskset c0 ./run_test.out speed/GemvBW 0 2 2` 的结果。

| type | threads | us/iter | eff GB/s | GFLOPS | bytes/elem |
|------|--------:|--------:|---------:|-------:|-----------:|
| w8   |       2 |  1015.5 |     61.4 |  115.7 |     1.0625 |
| w4   |       2 |   782.7 |     42.2 |  150.0 |     0.5625 |
| w3   |       2 |  1367.9 |     18.8 |   85.9 |     0.4375 |
| w2   |       2 |   697.2 |     26.3 |  168.4 |     0.3125 |

memcpy roofline (`std::thread`, 2 big cores)：38.8 GB/s。SD8 Elite LPDDR5X-8533 理论峰值 ~68 GB/s/channel，工程估计 single-direction 真实可用 ~55-65 GB/s。

> **饱和度估计**（按真实 DRAM read peak ≈ 60 GB/s 估）：w8 ≈ **102%**（实测已撞 LPDDR5X 单方向上限），w4 ≈ 70%，w3 ≈ 31%，w2 ≈ 44%。
>
> 注：测试内置的 memcpy 38.8 GB/s 不是 DRAM peak。memcpy 本身计 2× 字节 (read+write)，且 `std::thread` 在 Android 上不能像 macOS scheduler 一样自动均衡线程，1→8 线程几乎不 scale。GEMV 是 read-only weight，所以 eff GB/s 可以高于 memcpy 数字。后续会把 memcpy roofline 改走 MNN ThreadPool。

观察：
- **w8 已撞墙**：61.4 GB/s 已经达到 LPDDR5X 单方向理论上限，说明 i8mm + 寄存器化 accum 链把 dequant 完全隐藏在 mem latency 下。继续优化只能往压缩(w4)走。
- **w3 仍是最弱项**：18.8 GB/s，饱和度 ~31%，远低于 w2/w4。和 M-Mac/SD8G3 上 P3 后的趋势一致——8 Elite 的 sdot/i8mm 调度还有空间，可参考 [w2w3 优化经验](../../memory/w2w3_optimization_lessons.md)。
- **w2 latency 反而比 w4 短**：697 vs 782 us，不再是 SD8G3 上 "w2≈w4" 的现象——8 Elite Oryon 大核的 ALU 吞吐让 w2 dequant 不再卡瓶颈。
- **运行姿势**：Android 上跑这个测试**必须 `taskset c0` 绑大核**，否则数据会被 mid 核噪声污染。后续考虑在 test 里加 affinity 自动设置。

### Apple M3 Pro (5P + 6E, 36GB LPDDR5-6400), macOS arm64, precision=Low (fp16)

i8mm + asimddp + fp16 全开，36GB 统一内存。M=4096, K=14336, blocksize=64, threads=4。

macOS 调度器自动把 std::thread 分配到 P 核，无需 taskset。

**CPU**：

| type | threads | us/iter | eff GB/s | %peak | GFLOPS |
|------|--------:|--------:|---------:|------:|-------:|
| w8   |       4 |   605.2 |    103.1 |  87.4 |  194.0 |
| w4   |       4 |   334.0 |     98.9 |  83.9 |  351.6 |
| w3   |       4 |   555.8 |     46.2 |  39.2 |  211.3 |
| w2   |       4 |   294.0 |     62.4 |  52.9 |  399.5 |

memcpy roofline：117.9 GB/s @ 4 threads。

**Metal**（仅支持 w8 / w4）：

| type | threads | us/iter | eff GB/s | %peak | GFLOPS |
|------|--------:|--------:|---------:|------:|-------:|
| w8   |       4 |   700.4 |     89.1 |  71.3 |  167.7 |
| w4   |       4 |   464.3 |     71.1 |  57.0 |  252.9 |

memcpy roofline：124.9 GB/s。GPU 下 flushCache 失效，eff GB/s 更接近 warm-cache 估计。

观察：
- **CPU w8/w4 接近 roofline**（87% / 84%），dequant 流水已隐藏在 mem latency 下。
- **CPU w3 仍是短板**（39%），与 Android SD8 Elite 上的相对位置一致——i8mm 调度还有空间。
- **CPU w2 比 w3 快 2.4×**（62.4 vs 46.2 GB/s），因为 w2 用 4-IDX unpack 路径无 ext 链。
- **Metal w8 vs CPU w8**：CPU 反超 Metal（103 vs 89 GB/s）。M3 Pro 上 Metal `MetalConvolution1x1` GEMV 还有 ~15% 的优化空间，主要是 dequant kernel 没充分用 simd-group 寄存器。
- **跨设备对比**（w8 eff GB/s）：M3 Pro CPU 103 / SD8 Elite (taskset c0, 2t) 61 / iPad M5 Metal ~114（见 [ipad_m5_bench.md](../../memory/ipad_m5_bench.md)）。

## 关注指标

- **eff GB/s**：`weight_bytes / latency`，即每次 decode 实际拉取的权重字节速率。LLM decode 是带宽 bound，这个值越接近 peak 越好。
- **%peak**：相对 memcpy 峰值的饱和度。kernel 写得好的目标是 80%+；w3/w2 因 dequant ALU 开销大，目前还有较多优化空间。
- **AI (op/B)**：算术强度 = 2 / bytes_per_elem。w8≈0.25, w4≈0.5, w2≈1.0，bit 越低越远离 mem-bound 区。
- **GFLOPS**：仅作参考，decode 阶段不是 FLOP bound。

## 注意事项

- `flushCache()` 仅刷 CPU L2/L3。在 Metal 等 GPU/统一内存后端上，权重可能仍驻留在 GPU 缓存里，因此 GPU 下 eff GB/s 更接近 warm-cache 估计。
- 测量 `WriteMap → readMap` 总耗时，包含一次 fp16/fp32 输入打包，对小 K 略有误差；K=14336 时影响 < 1%。
- iPad / iOS 真机上需通过 `tools/ios_llm_benchmark_server.py` 间接触发 (设备无 shell)；详见 `memory/ipad_m5_bench.md`。

## 相关文档

- [`arm_low_bit_gemm.md`](./arm_low_bit_gemm.md) — ARM CPU 低 bit GEMM kernel 数据排布与汇编实现
- [`gemm_speed_benchmark.md`](./gemm_speed_benchmark.md) — Prefill 阶段 (M ≥ 8) 的 GEMM 通用基准
