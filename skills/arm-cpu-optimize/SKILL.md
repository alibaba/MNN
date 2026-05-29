---
name: arm-cpu-optimize
description: MNN ARM CPU 算子和低 bit kernel 性能优化。重点覆盖正确性基线、C++ 标量 oracle、C++ SIMD 模拟、寄存器规划、ARM 汇编实现、dispatch/pack 集成、模型级回归和 roofline 性能分析。
---

# MNN ARM CPU 性能优化 Skill

> **触发条件**：用户请求优化 ARM CPU 上的 MNN 算子、低 bit GEMM/GEMV kernel、NEON/SDOT/I8MM/SME2 实现，或要求 review ARM 汇编/dispatch/pack 性能问题。

> **边界**：不要读取、修改或依赖 `schema/private/` 和 `source/internal/`。

## 核心原则

1. **先正确，再加速**：任何性能改动都必须有可复现的正确性门禁。LLM 低 bit kernel 还要做模型级 sanity check，不能只依赖 op 单测。
2. **先写可读参考实现，再写汇编**：复杂 kernel 不要直接上汇编。先用 C++ 标量实现跑通，再用 C++ 模拟目标 SIMD/pack/寄存器行为，最后才分配寄存器并写 `.S`。
3. **优先复用已有 CoreFunctions**：普通算子先拆解为 `MNNPackedMatMul`、`MNNComputeMatMulForE_1`、`MNNScaleAndAddBias`、`MNNSoftmax`、`MNNNorm`、pack/unpack 等已有高性能函数。只有现有函数无法覆盖热点时才新增 Vec4/intrinsic/asm。
4. **替换前验证精确语义**：不能只看函数名。确认参数含义、layout、转置、归一化方式、in-place 安全性、尾部行为和量化后处理完全一致。
5. **数据驱动优化**：每次改动前后记录正确性结果、耗时、有效带宽、GFLOPS/AI。低 bit kernel 要区分 DRAM bound、unpack/issue bound、postprocess bound。
6. **每条 ISA 路径独立验证**：目标设备上的 I8MM、SDOT、FP16/FP32 后处理、SME2 dispatch 都可能走不同 pack 和 kernel，不能用一条路径代表全部。
7. **pack mode 和 kernel 指针必须配套**：新增或调整低 bit ISA 支持时，packer、cell stride、`MNNGetGemmUnit`、kernel 注册、mixed/online reorder 选择必须同步更新。
8. **寄存器生命周期先于 unroll**：加 unroll、hoist 常量、复用临时寄存器前，先写 live range 表。min/max、scale、bias、zero point、accumulator、unpack 常量不能被 postprocess 前的临时逻辑误覆盖。

## 推荐执行流程

### 0. 明确目标路径

开始前先写清楚这次优化覆盖哪些组合：

| 维度 | 需要确认 |
|------|----------|
| 算子/入口 | 哪个 executor、CoreFunctions 指针、asm symbol |
| 数据类型 | FP32、FP16、INT8、w2/w3/w4、per-channel/per-block |
| shape 热点 | E=1 decode、E>1 prefill、block size 32/64、OC split、tail |
| ISA | NEON、SDOT、I8MM、SME2，以及 runtime disable flag |
| 后处理 | bias、scale、zero point、fp32 min/max、add-dst、ReLU/ReLU6 |
| 验收 | op test、模型 prompt、roofline 或 speed test |

### 1. 建立 correctness oracle

- 先保留或补一个最直接的 C++ 标量路径，作为 bit-exact 或误差可解释的参考实现。
- 对低 bit 权重，标量 oracle 必须按真实 pack layout、cell stride、block metadata 读取，覆盖 `tId>0` 的 OC chunk。
- 比较点尽量靠近 kernel 输出：accumulator、dequant 后 FP32、postprocess 后 dst 分开看，避免把 pack、kernel、采样混在一起。
- 临时 debug oracle 可以不作为最终提交的一部分，但在写 asm 前必须先跑通。

### 2. 用 C++ 模拟 SIMD/pack 行为

在正式写汇编前，用 C++ 写一个“寄存器级”的模拟实现：

- 用小的 `std::array`/局部数组模拟 `v0..v31` 的 lane，而不是依赖编译器自动向量化。
- 显式模拟 `sdot`/`smmla` 的输入排列、每 4 byte dot 的分组、w2/w3 bit-plane unpack、sign/zero point 处理。
- 每次改变 pack layout 或 unroll 前，先让模拟版和标量 oracle 对齐，再迁移到 asm。
- 调试 w2/w3 时优先比较单个 block64、单个 output channel group、单个 K tile，避免模型输出才暴露问题。

### 3. 做寄存器和 ABI 计划

写 asm 前先列一张寄存器表：

| 寄存器 | 用途 | live 范围 | 可否复用 | 风险 |
|--------|------|-----------|----------|------|
| `x*` | 指针、loop counter、stride、metadata | 哪个 loop/分支 | 是否跨 call/branch | 指针恢复、tail |
| `v*` | accumulator、input、unpack tmp、scale、min/max | unpack、compute、postprocess | 是否可被 clobber | ReLU/minmax、add-dst |

特别注意：

- AArch64 `x19-x28`、`d8-d15` 的 callee-saved 规则。
- 不要把 fp32 min/max、scale、bias 等跨 postprocess 的值当作 unpack scratch。
- hoist 常量前，确认所有 tile 分支、tail 分支、`fp32minmax != nullptr` 分支都不会覆盖它。

### 4. 实现最小 asm kernel

- 一次只引入一个 tile/ISA 的 asm 改动，先让最小路径正确，再扩大 unroll 或加 block64 分支。
- 对 w2/w3，先保持 packed bytes 不变，再优化 unpack 指令数；扩大字节只有在用户明确接受时才考虑。
- block64 如果是默认量化 case，可以写专门分支，但必须保留 block32/per-channel 的安全路径。
- 常见低风险优化：常量 hoist、减少重复 unpack、`ld1r {.2d}` 替代 `ld1 {.8b}` + `mov d[1]`、消除不必要的 pointer restore。

### 5. 集成 dispatch/pack

新增或替换 kernel 时同时检查：

- `CoreFunctions`/arm init 中的函数指针注册。
- `MNNGetGemmUnit`、`UNIT/SRC_UNIT/DST_XUNIT`、weight reorder、online reorder 是否和 kernel 期望一致。
- 低 bit cell stride 是否使用真实 packed cell 字节数，而不是 useful payload 比例。
- i8mm、sdot、sme2、fallback 是否各自有匹配 packer；不要让 SME2 packer 喂给 i8mm/sdot kernel。

### 6. 验证矩阵

最少验证：

```bash
cmake --build build -j 8
cd build
./run_test.out op/lowMemory/blockConv 0 1 4
./run_test.out op/lowMemory/HybridConv 0 1 4
./run_test.out op/lowMemory/blockConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑
./run_test.out op/lowMemory/HybridConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑
```

LLM 低 bit kernel 额外验证：

```bash
./llm_demo /path/to/w3/config.json prompt.txt 64 1
./llm_demo /path/to/w3/config.json prompt.txt 64 1  # 在 SDOT 目标设备/构建配置上复跑
./llm_demo /path/to/w2/config.json prompt.txt 64 1
./llm_demo /path/to/w2/config.json prompt.txt 64 1  # 在 SDOT 目标设备/构建配置上复跑
```

说明：

- `64 1` 代表短 decode 且关闭 thinking 的 sanity 方式，具体参数以当前 `llm_demo` 为准。
- 判断 kernel 错误前先固定采样变量：greedy 或 no-thinking 对照。w2/w3 低 bit logits 在 mixed sampling 下的复读，不一定是 kernel 算错。
- 如果模型输出乱码、重复、质量变差，要同时比较 FP16/FP32，并分别在 I8MM、SDOT、SME2 等目标路径定位，先判断是实现问题还是量化/采样问题。

性能验证：

```bash
./run_test.out speed/GemvBW 0 2
```

记录 `us/iter`、`W MiB`、`bytes/elem`、`eff GB/s`、`%peak`、`GFLOPS`。w2/w3 的 eff BW 低时，优先检查 unpack 指令/字节比和 issue 压力，而不是直接假设 memory bandwidth 不足。

## CoreFunctions 复用清单

| 函数 | 优先用途 | 注意点 |
|------|----------|--------|
| `gcore->MNNPackedMatMul` | 大规模 GEMM | Pack 开销要能摊薄 |
| `gcore->MNNPackedMatMulRemain` | GEMM tail | 和主 kernel layout 一致 |
| `gcore->MNNComputeMatMulForE_1` | E=1 GEMV/decode | LLM decode 优先看这里 |
| `gcore->MNNComputeMatMulForH_1` | H=1 VecMat | 确认矩阵方向 |
| `gcore->MNNScaleAndAddBias` / `MNNScaleAndAddBiasScalar` | scale+bias | 检查 in-place |
| `MNNSoftmax` | softmax | 确认 axis/layout |
| `MNNNorm` | LayerNorm/RMSNorm | 确认 mean/rms 语义 |
| `MNNExp` / `MNNSiLu` | 激活 | 部分函数不支持 in-place |
| `gcore->MNNPackCUnit` / `MNNUnpackCUnit` | NC4/NC8 重排 | pack size 由 runtime 决定 |
| `gcore->MNNPackC4ForMatMul_A` / `MNNPackForMatMul_B` | MatMul pack | 和 kernel pack mode 配套 |
| `MNN_CONCURRENCY_BEGIN/END` | 多线程 | 注意 per-thread pointer 偏移 |

## 常见陷阱

| 陷阱 | 规避方式 |
|------|----------|
| op 单测通过但 LLM 输出乱码 | 增加 E=1、block64、fp32 min/max、multi-block、模型 prompt 验证 |
| w3 i8mm 错但 sdot 对 | 分开在 I8MM 和 SDOT 路径定位，不要混合定位 |
| min/max 被 unpack scratch 覆盖 | postprocess 前重新检查 live range，必要时延迟加载 min/max |
| 低 bit OC 分线程错位 | pointer 偏移用真实 packed cell stride，测试 `mSplitByOc=true` 和 `tId>0` |
| SME2 pack/kernel 不匹配 | packer、kernel 注册、unit 参数同时改，同时测 fallback |
| w2 偶发复读误判为 kernel bug | 用 greedy/no-thinking、FP16 对照、短 prompt 做区分 |
| 只堆 unroll 性能反降 | 先看寄存器压力、unpack issue、load/store、branch 和 postprocess |
| 小 shape 调用重型函数变慢 | 小规模保留朴素/Vec4 路径，大规模才用 pack+matmul |

## 参考文件

| 文件 | 用途 |
|------|------|
| `source/backend/cpu/compute/CommonOptFunction.h` | CoreFunctions 定义和函数签名 |
| `source/backend/cpu/CPUAttention.cpp` | MatMul/Softmax/Norm/多线程复用参考 |
| `source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp` | pack、tiling、线程拆分参考 |
| `source/backend/cpu/arm/arm64/MNNPackedMatMul.S` | AArch64 asm 风格参考 |
| `source/backend/cpu/arm/arm64/MNNPackedMatMul_int8.S` | SDOT/int8 matmul 参考 |
| `test/speed/MatMulSpeed.cpp` | speed test 组织方式参考 |

## 子步骤文档

- `skills/arm-cpu-optimize/step0-decompose.md`：计算拆解和 CoreFunctions 映射。
- `skills/arm-cpu-optimize/step1-benchmark.md`：建立 baseline 和 speed test。
- `skills/arm-cpu-optimize/step2-analyze.md`：瓶颈分析和方案选择。
- `skills/arm-cpu-optimize/step3-cpp-opt.md`：C++ 层函数复用、多线程、内存池。
- `skills/arm-cpu-optimize/step4-asm.md`：C++ oracle/SIMD 模拟/寄存器规划/asm 实现。
- `skills/arm-cpu-optimize/step5-integrate.md`：集成、回归和性能报告。
