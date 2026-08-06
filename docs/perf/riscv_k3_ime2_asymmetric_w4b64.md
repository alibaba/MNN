# MNN K3 RISC-V IME2 非对称 W4B64 LLM 性能优化

本文记录 MNN 在 SpacemiT K3 RISC-V 平台上的 LLM 推理优化方案与性能结果，重点覆盖
非对称 W4B64 Linear/MatMul、prefill、decode、Attention 和 KV Cache。

本文只展示公开可复现的测试口径、性能数据和实现设计，不依赖运行时调优环境变量。

## 1. 性能结论

测试统一使用 8 线程。prefill 使用 pp512，decode 使用 tg128。每个进程内部重复 5 次，
最终结果取 3 个独立进程的算术平均值。

| 模型 | pp512 | tg128 |
|---|---:|---:|
| Qwen3-0.6B | **381.24 tok/s** | **54.49 tok/s** |
| Qwen3-1.7B | **169.29 tok/s** | **24.90 tok/s** |
| Qwen3.5-0.8B | **127.57 tok/s** | **33.85 tok/s** |
| Qwen3.5-2B | **85.60 tok/s** | **17.68 tok/s** |

表中吞吐量保留两位小数。

## 2. 测试口径

### 2.1 量化格式

MNN 模型使用：

- 权重位宽：4 bit
- 量化 block：64
- 非对称量化：`sym: false`
- scale：FP16
- activation：运行时动态量化

### 2.2 Benchmark 定义

- pp512：一次处理 512 个输入 token
- tg128：实际连续生成 128 个 token
- 线程数：8
- 模型加载时间不计入吞吐量
- 每轮 benchmark 内部重复 5 次
- 每个场景运行 3 个新进程
- MNN 使用 `llm_bench`

Qwen3.5 是 LinearAttention 与 full Attention 混合架构。本文只测试文本主干，不包含视觉输入。
由于执行图不同，Qwen3 与 Qwen3.5 的结果不能简单归因于同一个 Attention kernel。

### 2.3 通用测试命令

Qwen3 MNN pp512：

```bash
./llm_bench \
  -m <MNN_CONFIG> \
  -p 512 -n 0 -rep 5 -t 8 \
  -load false -fa 1 -kv false
```

Qwen3.5 MNN pp512：

```bash
./llm_bench \
  -m <MNN_CONFIG> \
  -p 512 -n 0 -rep 5 -t 8 \
  -load false -fa 0 -kv false
```

MNN tg128：

```bash
./llm_bench \
  -m <MNN_CONFIG> \
  -p 1 -n 128 -rep 5 -t 8 \
  -load false -fa 0 -kv true
```

IME2 路径由构建配置决定，运行命令不需要设置 IME2 调优环境变量。

## 3. K3 A100/IME2 硬件基础与优化方向

本节中的硬件数据来自 SpacemiT 当前公开的 K3 产品简介、架构论文和 IME 指令规范。
TOPS 与内存带宽属于硬件峰值或公开 microbenchmark 数据，不等同于 MNN 的端到端模型吞吐量。

### 3.1 A100 与 IME2

K3 集成 8 个 X100 通用 CPU 核和 8 个 A100 AI CPU 核。A100 仍运行标准 RISC-V 指令和 Linux
线程，同时提供 RVV 1.0 与 SpacemiT IME 矩阵扩展。统一的 ISA 与向量寄存器编程模型可以
减少传统独立加速器常见的显式任务提交和设备切换开销。

A100 由两个 cluster 组成。每个 cluster 包含 4 个标量核、2 个向量核、2 个 local-memory bank
和共享 L2；每组两个向量/计算核共享一个 IME2 Tensor Core。其主要计算资源包括：

- 32 个 1024-bit 向量寄存器，可容纳 4 KiB 的寄存器工作集
- 两条向量执行流水；常用整数与浮点运算资源主要按 `2 × 256 bit` 配置
- 每个参与共享 Tensor Core 的计算核具有独立 512-bit load channel，可双发一次 512-bit load 和一次 MMA
- 每核私有 IME1，用于较小的 INT8 矩阵计算
- 每组两个计算核共享 IME2，用于高吞吐 INT4、INT8、FP16 和 BF16 矩阵计算

1024 bit 是向量寄存器和 IME tile 的体系结构宽度，不表示所有普通 RVV 运算都能在单周期完成
1024-bit 算术。对于共享 IME2，单核即使采用 register blocking 也可能无法持续填满矩阵单元；
两个核并行加载并连续发射 MMA 才能更接近满利用率。

### 3.2 IME2 指令与矩阵形状

IME 扩展复用 RVV 的 `v0`～`v31` 表示二维矩阵 tile，不引入独立的 matrix register file。
当前公开规范包含 46 条 AI 自定义指令，主要分为：

| 指令类别 | 典型指令 | 数据路径与矩阵形状 |
|---|---|---|
| 整数矩阵乘加 | `smt.vmadot*` | INT4：`8 × 32 × 8`；INT8：`8 × 16 × 8`；累加到 INT32 |
| 卷积滑窗矩阵乘加 | `smt.vmadot1/2/3*` | INT8 输入，固定滑窗偏移，累加到 INT32 |
| 4:2 结构化稀疏 | `smt.vmadot.sp*` | INT4：`8 × 64 × 8`；INT8：`8 × 32 × 8` |
| block-scale 矩阵乘加 | `smt.vmadot.hp*` | INT4/INT8 点积乘 FP16/BF16 scale，输出 FP16/BF16 小计 |
| 浮点矩阵乘加 | `smt.vfwmadot*` | FP16/BF16 `8 × 8 × 8`，累加到 FP32 |
| 数据布局转换 | `smt.vpack/vupack/vnpack/vnpack4*` | tile 交织、拆分和 INT4 nibble 重排 |

整数指令通过后缀区分两个输入的 signedness：无后缀为 signed × signed，`u` 为
unsigned × unsigned，`us` 和 `su` 分别表示 unsigned × signed 与 signed × unsigned。

MNN 当前的非对称 W4B64 路径使用 INT8 activation 和 INT4 weight。由于 IME2 的基础矩阵
指令是同位宽 INT4 × INT4 或 INT8 × INT8，kernel 将 INT8 activation 拆成有符号高 4 bit
和无符号低 4 bit，分别执行 `vmadotsu` 与 `vmadotu`，再合并结果。高性能路径还使用
`*.hp` block-scale 变体减少缩放相关指令，并用 `vpack`/`vnpack4` 完成 IME2 所需布局。

IME2 的 block-scale 指令不等价于非对称 zero-point 修正。W4B64 的 offset、row sum 和残差项
仍由 MNN 在 packing 与 kernel 中融合处理，这也是通用 `UNIT/SRC_UNIT` 数据布局无法直接复用的
主要原因。

### 3.3 理论算力

K3 公开的整芯片峰值如下：

| 数据类型 | Dense 峰值 | 4:2 Sparse 峰值 |
|---|---:|---:|
| INT4 | 30 TOPS | 60 TOPS |
| INT8 | 15 TOPS | 30 TOPS |
| FP16 / BF16 | 7.5 TOPS | — |
| FP8 | 7.5 TOPS | — |

K3 对外标称的“最高 60 TOPS”对应 4:2 结构化稀疏 INT4。本文测试的 HQQ 非对称 W4B64 是
dense 权重，不能按 60 TOPS 计算；其硬件上限更接近 dense INT4 路径。当前 W4A8 kernel
还需要两次 INT4 点积表示一次 INT8 × INT4 逻辑点积，并包含动态量化、scale、offset 修正和
FP32 后处理，因此模型有效算力一定低于单纯的 dense INT4 指令峰值。公开资料中的 FP8 路径
会先转换为 BF16 计算，也不应理解为独立的原生 FP8 MMA。

### 3.4 存储层次与带宽

| 层次 | 公开规格 | 软件使用方式 |
|---|---|---|
| 向量寄存器 | 32 × 128 B，共 4 KiB | 保存 A/B tile、累加器、scale 和局部中间值 |
| A100 L1 | 每核 32 KiB I-Cache + 32 KiB D-Cache | 普通 load/store 的自动缓存 |
| A100 cluster | 每 cluster 1 MiB L2 + 1.5 MiB TCM | L2 为共享缓存；TCM 为软件管理的片上 local memory |
| K3 A100 总量 | 两个 cluster，共 2 MiB L2 + 3 MiB TCM | TCM 按运行时提供的 block 分配给 worker |
| 主存 | 64-bit LPDDR5-6400，最高约 51 GB/s | 存放模型权重、KV Cache 和大尺寸中间数据 |

TCM 不是会自动替换 cache line 的硬件缓存，而是需要显式分配和搬运的 scratchpad。MNN 不硬编码
TCM block 数或单块容量，而是从运行时查询 `blkNum` 与 `blkSize`；TCM 不可用、容量不足或返回
模拟内存时均回退到 DRAM 路径。

51 GB/s 是 LPDDR5 接口理论峰值，持续有效带宽会受到刷新、控制器效率、访问模式和多核竞争影响。
官方架构论文的顺序读取测试显示，向量化、多线程分片和硬件带宽优化可达到标量基线的约 3.28 倍，
但论文也明确该结果不是最终峰值。对于 LLM decode，更有意义的上限通常是：

```text
tokens/s ≈ 持续有效内存带宽 / 每个 token 必须读取的权重与元数据字节数
```

因此 decode 即使拥有充足的 IME2 峰值算力，也可能先受到 LPDDR5 带宽限制。

### 3.5 从硬件特征到 MNN 优化

| 硬件特征或瓶颈 | MNN 优化方向 |
|---|---|
| IME2 有固定 tile 和 A row-major、B column-major 布局 | 加载期预排 packed-B，运行期只量化并 pack A |
| 两个向量核共享 IME2 Tensor Core | 使用 worker-pair 持续供数；避免无条件增加线程争抢共享单元 |
| Prefill 具有多行 activation 和较高权重复用 | 使用 M4、strided row、register blocking，并融合动态量化与 packing |
| Decode 每个 token 都流式读取大部分权重 | 按输出 panel 连续分片、多 worker 顺序读取，并减少 barrier 与 dispatch |
| 3 MiB TCM 通常无法容纳大尺寸 Linear 的完整权重 | 只缓存当前 A/B tile，使用 worker-pair copy/compute 流水隐藏搬运延迟 |
| 公开硬件提供异步 AI-DMA | 后续可将当前 RVV copy 升级为 DMA ping-pong，实现更完整的异步双缓冲 |
| 非对称 W4B64 需要额外 offset 修正 | 将 scale、row sum、offset/residual 融入 packed layout 与 kernel |
| 输出和中间 buffer 也消耗带宽 | 使用 direct-C4/direct-output，直接写最终布局并融合 epilogue |
| 1024-bit RVV 适合宽向量归约和 FP 运算 | 用 RVV 完成 absmax、量化、softmax、QK/PV 和通用 fallback |

当前 MNN 的 TCM 流水使用 RVV load/store 完成 DRAM→TCM copy，并由两个 worker 交替执行
copy 与 compute；它尚未使用公开硬件提供的异步 AI-DMA。后续若接入 DMA，应继续保留容量门禁
和小矩阵回退，因为小工作集上的搬运启动与同步成本可能高于收益。

### 3.6 MNN 实现边界

K3 专用实现位于独立的 SpacemiT IME2 target 中，通过以下构建选项启用：

```bash
-DMNN_RVV_SPACEMIT_IME2=ON
```

关闭该选项时使用标准 RVV 实现，不编译 IME2 指令。通用 CPU、标准 RVV 和 K3 IME2
保持三层隔离，K3 的 shape 门禁、TCM 管理和 vendor kernel 不进入 ARM、x86 或普通
RISC-V 热路径。

## 4. Linear/MatMul 优化

### 4.1 非对称 W4B64 IME2 kernel

Linear 权重采用 W4B64 非对称量化。计算过程中保留每个 block 的 scale 与 offset 修正，
通过高、低半字节拆分和两次 IME2 INT4 点积，等效完成 INT8 activation 与 INT4 weight
的点积。

针对 prefill 和 decode 分别提供专用 kernel：

- prefill：M4 kernel，一次计算 4 行 activation
- decode：M1 asym-pair kernel，针对单 token GEMV

不满足量化格式、shape、VLEN、输出 tail 或 packed-weight 条件时，自动回退原有实现。

### 4.2 动态量化与 A packing 融合

原路径先单独遍历 activation 计算动态量化 scale，再执行 A packing。优化后在 A pack
阶段一次完成：

```text
absmax
  -> quant scale / input scale
  -> packed A
  -> srcKernelSum
```

这样可以减少一次 activation 全量扫描、临时 buffer 访问和 worker dispatch。

### 4.3 Prefill strided M4 调度

prefill 使用 strided-row worker 调度：

- 每个 worker 处理固定间隔的 M4 row block
- 避免大量细粒度任务的重复分发
- A pack 和 GEMM 使用一致的行布局
- 保持 direct-C4 输出条件

权重中的非对称 residual 修正直接融合进 packed-B 和 IME2 kernel，避免单独的 residual
后处理 pass。

### 4.4 Direct-C4 epilogue

满足非对称 W4B64、M4、FP32 输出和完整输出通道等条件时，IME2 kernel 直接完成：

```text
GEMM
  -> input scale
  -> bias / clamp
  -> C4 output
```

该路径跳过中间 C buffer 和额外的 layout conversion，是 pp512 的主要优化之一。

## 5. Decode 优化

### 5.1 M1 hierarchical A pack

decode 的 M 等于 1。A packing 按 K256 super-block 组织，每个 block 内完成动态量化、
局部 scale 和 kernel sum 计算，使数据布局与 M1 IME2 kernel 直接匹配。

### 5.2 Direct-output

当以下条件同时满足时，M1 kernel 直接写最终 FP32 输出：

- 使用精确的非对称 M1 pair kernel
- bias 为空
- 不需要额外 input scale/input bias
- 非对称 residual 已在 kernel 中完成
- 不需要 clamp

这会跳过临时 C buffer、worker post 和一次额外的内存读写。

Qwen3-1.7B 的一轮独立配对测试中，direct-output 单独将 tg128 从约
22.42 tok/s 提升到约 22.61 tok/s。

### 5.3 Worker-pair TCM copy/compute 双缓冲流水

profile 显示 Qwen3-1.7B decode 中，量化 Linear 约占总算子时间的 93.9%，Attention
约占 2.9%。主要瓶颈是大 packed-B 的访存，而不是 Attention。

TCM 流水将两个 worker 组成一对：

```text
Worker A: copy B tile 0 -> compute tile 0 -> copy tile 2 -> ...
Worker B:                copy tile 1 -> compute tile 1 -> ...
```

每个 worker 在软件上持有自己的 TCM buffer，pair barrier 只同步 copy/compute 阶段，
不交换 buffer 的逻辑所有权；底层 TCM bank 仍由成对计算核共享。计算使用的
IME2 kernel 和 DRAM 路径完全相同。

TCM 流水只在以下条件下启用：

- 非对称 W4B64
- M1 asym-pair layout
- K 不小于 2048
- packed-B 不小于 2 MiB
- persistent spin worker pool 可用
- 至少可组成一对 worker，并有不少于两个输出 group
- 至少两个可用 TCM block
- 真实 TCM 可用且容量足够

小矩阵继续使用原 DRAM 路径，因为 copy 和 barrier 成本无法被计算量摊薄。任何门禁未命中
都会自动回退，不影响标准 RVV 或其他 CPU 平台。

在随后一轮同机配对中，TCM 流水将 direct-output 版本的约 22.51 tok/s 提升到约
24.91 tok/s，增幅约 10.7%。22.51 与上一节的 22.61 来自不同配对轮次。

### 5.4 Worker 数选择

针对大矩阵将 decode worker 从 6 增加到 8 的实验结果为 21.95～21.99 tok/s，
低于 6-worker control 的 22.30～22.37 tok/s。

这说明该场景已受到内存带宽和同步开销约束，简单增加 worker 会加剧带宽竞争。最终实现保留
6 个计算 worker，并通过 TCM 提高每个 worker 的有效数据供给。

## 6. Attention 与 KV Cache

### 6.1 RVV decode Attention

标准 RVV 路径为 decode QK/PV 提供 direct-matvec，减少临时 QK buffer 和 layout
转换。该优化只依赖 RVV，不依赖 IME2 指令。

### 6.2 K3 fused Attention

满足 head 数、GQA、head dimension、数据类型和 layout 门禁时，K3 使用 fused
Attention 路径，将 QK、online softmax 和 PV 放入同一个分块执行流程，并复用
每 worker TCM scratch。

该路径位于 K3 专用 target 中，但核心计算使用 1024-bit RVV FP16/FP32 FMA、归约和转换指令，
不调用 IME2 `vmadot` 矩阵指令。IME2 主要加速量化 Linear，RVV 则更适合 Attention 中的
softmax、动态边界和较小 head dimension。

门禁未命中时回退标准 RVV Attention。K3 逻辑通过独立 Execution 子类实现，
通用 `CPUAttention` 只保留架构无关的扩展入口。

### 6.3 KV Cache

KV Cache 更新针对连续 FP32 key/value 数据减少循环控制和重复地址计算，并在已验证的
RISC-V 条件下并行更新。量化 KV Cache 和其他数据布局继续使用通用路径。

## 7. 代码组织

主要实现位于：

| 文件 | 作用 |
|---|---|
| `source/backend/cpu/riscv/CMakeLists.txt` | 标准 RVV 与 K3 IME2 target 隔离 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2ConvInt8Executor.cpp` | K3 Linear Execution、prefill/decode 路由 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2GemmInt8.cpp` | A/B packing、worker、cache、TCM 流水 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2GemmI8I4Local.cpp` | IME2 W4B64 kernel |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2AttentionFunctions.cpp` | K3 fused Attention |
| `source/backend/cpu/riscv/rvv/MNNRvvAttentionFunctions.cpp` | 标准 RVV decode Attention |

`CPUAttention` 与通用 `ConvInt8TiledExecutor` 不包含 K3 kernel 或 TCM 实现，只提供稳定的
Execution 扩展接口。这样可以避免为占比更高的 ARM/x86 CPU 路径引入 RISC-V 专用分支。

## 8. 正确性与回归验证

最终实现完成以下验证：

- K3 IME2 完整编译与链接
- DenseConv 低比特测试：1/1 通过
- Attention C4 与 tail 测试：2/2 通过
- ThreadPool 测试：1/1 通过
- Qwen3-1.7B 连续生成 128 token
- 标准 RVV 构建与生成冒烟
- Qwen3-0.6B、Qwen3-1.7B、Qwen3.5-0.8B、Qwen3.5-2B 跨模型回归

TCM 流水还进行过真实模型逐位自检：前 128 个 Linear tile 同时执行 TCM 与原 DRAM
kernel，结果为：

```text
checked tiles: 128
mismatches: 0
```

自检完成后，重复计算和诊断输出均已从生产路径移除。

## 9. 结论与限制

本轮优化的核心不是单纯增加线程，而是根据 prefill/decode 的数据复用特征设计不同路径：

- prefill：融合动态量化与 packing，使用 strided M4 和 direct-C4
- decode：使用 M1 asym-pair、direct-output 和大矩阵 worker-pair TCM 双缓冲流水
- Attention：标准 RVV direct-matvec 与严格门禁的 K3 fused path
- 工程结构：通用 CPU、标准 RVV、K3 IME2 分层隔离

仍需注意：

- 结果只代表本文给定的模型、量化方式、8 线程和输入长度。
- 实际吞吐量会受到系统负载、温度和频率波动影响。
- TCM 门槛根据当前模型工作集验证，扩展到更多 shape 前需要重新测试。
- 更长 context 下，Attention 与 KV Cache 带宽占比会继续上升。

## 10. 公开资料

- [SpacemiT K3 产品简介](https://cdn-resource.spacemit.com/file/chip/K3/K3_brief_zh.pdf)
- [SpacemiT K3 架构论文](https://forum.spacemit.com/uploads/short-url/60aJ8cYNmrFWqHn4ddwwSzMLjlY.pdf)
- [SpacemiT AI Matrix Extension Instruction Set](https://github.com/spacemit-com/docs-ai/blob/main/en/architecture/ime_extension.md)
- [SpacemiT Homogeneous Fusion Architecture](https://github.com/spacemit-com/docs-ai/blob/main/en/architecture/concept.md)

K3 架构论文当前标记为 Preview，IME 指令规范当前为公开版本 0.6；硬件规格与指令描述应以厂商
后续正式版本为准。
