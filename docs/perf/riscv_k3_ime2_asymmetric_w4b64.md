# K3 RISC-V IME2 非对称 W4B64 LLM 优化记录与性能报告

> 日期：2026-07-24；结构解耦更新：2026-07-30；分支：`feature/riscv-ime2-linear-opt`；
> 基线：`e9398e831c`；性能基线源码：`f4ffc5e487`。最终版本已完成单编译宏、
> 零 IME2 环境变量和通用 CPU / 标准 RVV / Spacemit IME2 三层隔离，
> 性能优化主体已压缩为一个提交，本次进一步按 Execution 子类完成结构解耦。
>
> 目标：在 K3 上优化 MNN LLM Linear/MatMul、Attention 和 KV Cache，以 Qwen3-0.6B 为主优化目标，
> 并使用 Qwen3-1.7B、Qwen3.5-0.8B 和 Qwen3.5-2B 验证跨模型泛化表现。

## 1. 结论

最终比较严格采用以下口径：

- MNN：HQQ W4、block64、`sym:false`
- llama.cpp：Q4_1
- 每组比较来自同一份 BF16 源模型的独立转换产物
- 同一块 K3 开发板
- 8 线程
- MNN 只使用 `llm_bench`
- prefill：pp512
- decode：实际生成 128 token，即 tg128
- 每个进程内部重复 5 次，再运行 3 个交替的新进程轮次

下表是 `f4ffc5e487` 上完成的正式三进程、双引擎交替结果。第 4.6～4.9 节另外记录了
后续工程清理和完整分层重构的 MNN 回归数据；这些轮次没有重新交替运行 llama.cpp，
不能替代本表的正式比较。

| 模型 | 场景 | MNN 三轮均值 | llama.cpp 三轮均值 | MNN 相对提升 |
|---|---|---:|---:|---:|
| Qwen3-0.6B | pp512 | 381.24 tok/s | 356.84 tok/s | **+6.84%** |
| Qwen3-0.6B | tg128 | 54.49 tok/s | 53.70 tok/s | **+1.47%** |
| Qwen3-1.7B | pp512 | 167.39 tok/s | 167.85 tok/s | **-0.28%** |
| Qwen3-1.7B | tg128 | 22.06 tok/s | 23.14 tok/s | **-4.64%** |
| Qwen3.5-0.8B | pp512 | 127.57 tok/s | 104.94 tok/s | **+21.57%** |
| Qwen3.5-0.8B | tg128 | 33.85 tok/s | 28.47 tok/s | **+18.91%** |
| Qwen3.5-2B | pp512 | 85.60 tok/s | 78.96 tok/s | **+8.41%** |
| Qwen3.5-2B | tg128 | 17.68 tok/s | 16.14 tok/s | **+9.54%** |

即使使用最保守的“最差 MNN 对最好 llama.cpp”比较：

- Qwen3-0.6B：pp512 `+5.07%`，tg128 `+0.37%`
- Qwen3-1.7B：pp512 `-0.57%`，tg128 `-4.84%`
- Qwen3.5-0.8B：pp512 `+19.10%`，tg128 `+17.46%`
- Qwen3.5-2B：pp512 `+6.98%`，tg128 `+9.09%`

因此，本次锁定的非对称量化与输入长度条件下：

- 主优化目标 Qwen3-0.6B 的 pp512 和 tg128 均超过 llama.cpp Q4_1。
- Qwen3-1.7B 的 pp512 基本持平，tg128 仍落后 4.64%，是后续 decode 优化的明确目标。
- Qwen3.5-0.8B/2B 的四项端到端结果均稳定超过 llama.cpp Q4_1。

Qwen3.5 是 LinearAttention 与全注意力混合架构，其结果不能全部归因于 Linear/MatMul 或
Qwen3-0.6B 专属 fused Attention；第 4 节会单独说明模型结构和执行路径差异。

## 2. 测试环境

### 2.1 机器与仓库

| 项目 | 配置 |
|---|---|
| 开发板 | SpacemiT K3 RISC-V |
| 远端连接 | `ssh k3` |
| MNN 仓库 | `/home/yanxing/AliNNPrivate` |
| MNN build | `/home/yanxing/AliNNPrivate/build-riscv-llm` |
| llama.cpp 仓库 | `/home/yanxing/llama.cpp-spacemit` |
| 线程数 | 8 |

模型与正式日志：

| 模型 | MNN 模型 | llama.cpp Q4_1 模型 | 正式日志 |
|---|---|---|---|
| Qwen3-0.6B | `/home/yanxing/Qwen3-0.6B-MNN-master-c4-ime2` | `/home/yanxing/Qwen3-0.6B-GGUF/qwen3-0.6b-q4_1.gguf` | 原始三轮数据见第 4.1 节 |
| Qwen3-1.7B | `/home/yanxing/Qwen3-1.7B-MNN-master-c4-ime2` | `/home/yanxing/Qwen3-1.7B-GGUF-local/qwen3-1.7b-q4_1.gguf` | `/home/yanxing/bench-qwen17-20260724` |
| Qwen3.5-0.8B | `/home/yanxing/Qwen3.5-0.8B-MNN-master-c4-ime2` | `/home/yanxing/Qwen3.5-0.8B-GGUF-local/qwen3.5-0.8b-q4_1.gguf` | `/home/yanxing/bench-qwen35-20260724/qwen35-0p8b` |
| Qwen3.5-2B | `/home/yanxing/Qwen3.5-2B-MNN-master-c4-ime2` | `/home/yanxing/Qwen3.5-2B-GGUF-local/qwen3.5-2b-q4_1.gguf` | `/home/yanxing/bench-qwen35-20260724/qwen35-2b` |

K3 运行时报告：

- 16 个 CPU 核心，分为两个 8 核组
- 支持 FP16
- 不支持标准 `i8sdot`、`i8mm`、SVE2、SME2
- llama.cpp K3 后端和 MNN 优化路径均使用 IME2

Qwen3-0.6B 早期正式测试没有完整保存逐轮温频快照。后续 Qwen3-1.7B 和 Qwen3.5 测试为每个正式进程保存
pre/post 频率与 thermal zone 3～6；所有这些采样点均为 1.8 GHz，最高温度 63°C，未在采样点观察到降频。
这些快照不是运行过程中的连续频率监控，因此不能扩大解释为“全程绝无降频”。

文档保留三轮原始数据、均值和最保守比较，避免只引用受温度或 DVFS 影响的最好结果。

### 2.2 量化口径

MNN 模型的远端 `export_args.json` 已确认：

```json
{
    "quant_bit": 4,
    "quant_block": 64,
    "lm_quant_bit": 4,
    "lm_quant_block": 64,
    "hqq": true,
    "sym": false,
    "transformer_c4": true
}
```

Qwen3-1.7B 和两个 Qwen3.5 MNN 产物均再次核对了以上字段；scale 与 embedding 使用 16 bit。

llama.cpp benchmark 输出分别确认模型类型为 Qwen3/Qwen3.5 Q4_1。新增三个模型采用：

```text
同一 BF16 源模型 -> F16 GGUF -> K3 llama.cpp Q4_1
```

Q4_1 由 F16 GGUF 直接量化，没有经过 Q8 中间模型再次量化。Qwen3.5 使用官方上游转换器
`0cea36222fe9bac5ebfc45716c9eef11f37046c4` 生成 F16 GGUF，再由 vendor K3 工具量化和运行。
因此本文的 “llama.cpp K3” 指官方模型转换定义加 K3 vendor runtime/kernel，而不是未经修改的纯上游 runtime。

Qwen3-0.6B benchmark 输出确认模型类型为：

```text
qwen3 0.6B Q4_1
```

两者都是非对称/仿射 W4，但不是相同的字节格式：

- MNN 使用 HQQ W4 block64，权重导出参数为 `sym:false`
- llama.cpp Q4_1 使用自身的 scale/min block 编码，通常是 block32

因此这里比较的是两个引擎各自正式低比特模型格式的端到端性能，而不是相同权重字节布局上的单 kernel 比较。

还需要区分权重与激活量化：

- 权重 W：非对称 W4B64
- 激活 A：运行时动态对称 int8 量化

代码中的 `BatchSymDynamicQuantScaleOnly` 或 A 的 `absmax` 动态量化描述的是激活量化，不能据此把模型误判为对称权重量化。非对称权重计算仍需保留每个 block 的 zero-point/source-sum 修正，概念上等价于：

```text
output_block =
    activation_scale * weight_scale
    * (dot(quant_A, quant_W) - weight_zero_point * sum(quant_A))
```

最终 benchmark 使用 `sym:false` 模型；生产构建固定走非对称 W4B64 路径，
不存在可在运行时启用的对称 W4 实验开关。

### 2.3 Qwen3.5 文本主干口径

Qwen3.5-0.8B/2B 的源目录是 `Qwen3_5ForConditionalGeneration` 多模态包，但本报告只比较文本 LLM 主干：

- MNN 导出得到独立 `llm.mnn` 和 `visual.mnn`，`llm_bench` 只加载 `llm.mnn`。
- llama.cpp 转换显式使用 `--no-mtp`，且没有生成或加载 `mmproj`。
- 两侧均排除视觉塔和 MTP/speculative decode。
- 两个文本模型都是 24 层，其中 18 层为 `LinearAttention`，6 层为 full `Attention`。
- Qwen3.5 的 full Attention 部分使用 8 个 query head、2 个 KV head、head dimension 256，不满足第 5.8 节
  Qwen3-0.6B fused Attention 的 16Q/8KV/head dimension 128 门禁。

因此 Qwen3.5 数据是混合架构的端到端结果，包含 LinearAttention recurrence、full Attention、Linear/MLP
和 lm head，不能作为单独 IME2 Linear kernel 的 microbenchmark。

## 3. Benchmark 协议

### 3.1 Qwen3-0.6B MNN pp512

```bash
cd /home/yanxing/AliNNPrivate

./build-riscv-llm/llm_bench \
  -m /home/yanxing/Qwen3-0.6B-MNN-master-c4-ime2/config.json \
  -p 512 -n 0 -rep 5 -t 8 -load false -fa 1 -kv false
```

### 3.2 Qwen3-0.6B MNN tg128

```bash
cd /home/yanxing/AliNNPrivate

./build-riscv-llm/llm_bench \
  -m /home/yanxing/Qwen3-0.6B-MNN-master-c4-ime2/config.json \
  -p 1 -n 128 -rep 5 -t 8 -load false -fa 0 -kv true
```

MNN 输出中包含两项：

```text
prompt=1
decode=128
```

性能报告只取 `decode=128` 对应的第二个速度。`-p 1` 用于启动真实生成，没有减少或替换 `-n 128`，也没有把单 token prompt 速度当作 tg128。

### 3.3 Qwen3-0.6B llama.cpp Q4_1 pp512

```bash
/home/yanxing/llama.cpp-spacemit/build/bin/llama-bench \
  -m /home/yanxing/Qwen3-0.6B-GGUF/qwen3-0.6b-q4_1.gguf \
  -p 512 -n 0 -r 5 -t 8
```

### 3.4 Qwen3-0.6B llama.cpp Q4_1 tg128

```bash
/home/yanxing/llama.cpp-spacemit/build/bin/llama-bench \
  -m /home/yanxing/Qwen3-0.6B-GGUF/qwen3-0.6b-q4_1.gguf \
  -p 0 -n 128 -r 5 -t 8
```

### 3.5 跨模型命令差异

Qwen3-1.7B 沿用 Qwen3-0.6B 的执行模式，只替换模型路径：

| 场景 | MNN 参数 | llama.cpp 参数 |
|---|---|---|
| pp512 | `-p 512 -n 0 -rep 5 -t 8 -load false -fa 1 -kv false` | `-p 512 -n 0 -r 5 -t 8` |
| tg128 | `-p 1 -n 128 -rep 5 -t 8 -load false -fa 0 -kv true` | `-p 0 -n 128 -r 5 -t 8` |

Qwen3.5 是混合 LinearAttention/full Attention 模型，正式测试使用模型适配后的执行模式：

| 场景 | MNN 参数 | llama.cpp 参数 |
|---|---|---|
| pp512 | `-p 512 -n 0 -rep 5 -t 8 -load false -fa 0 -kv false` | `-p 512 -n 0 -r 5 -t 8` |
| tg128 | `-p 1 -n 128 -rep 5 -t 8 -load false -fa 0 -kv true` | `-p 0 -n 128 -r 5 -t 8` |

IME2 路径由构建时的 `MNN_USE_SPACEMIT_IME2` 宏决定，运行命令不设置也不读取任何
IME2 调优环境变量。`-n 0` 直接使用 `llm_bench` 原生的 prefill-only 口径。`-fa` 是 MNN
的模型专属执行选择；Qwen3-0.6B/1.7B 与 Qwen3.5 不能描述为
使用完全相同的 Attention fast-path。

### 3.6 采样方法

每个场景采用三个新进程轮次，并在 MNN 与 llama.cpp 之间交替运行：

- MNN 每轮内部 `-rep 5`
- llama.cpp 每轮内部 `-r 5`
- Qwen3-1.7B 和 Qwen3.5 明确采用 AB/BA/AB 顺序：
  - Round 1：MNN → llama.cpp
  - Round 2：llama.cpp → MNN
  - Round 3：MNN → llama.cpp
- 最终结论使用三个进程结果的算术平均值
- 同时报告全部原始值和最差对最好结果
- warm-up 不计入正式结果
- 每个新增模型正式进程前后记录时间、CPU 频率和温度

模型加载时间不计入吞吐量。

## 4. 最终性能数据

### 4.1 Qwen3-0.6B 三轮原始结果

| 轮次 | MNN pp512 | llama.cpp pp512 | MNN tg128 decode | llama.cpp tg128 |
|---:|---:|---:|---:|---:|
| 1 | 375.24 ± 3.23 | 356.71 ± 0.11 | 54.75 ± 0.17 | 53.65 ± 0.02 |
| 2 | 376.78 ± 3.46 | 357.14 ± 0.14 | 53.93 ± 0.60 | 53.72 ± 0.02 |
| 3 | 391.69 ± 3.02 | 356.68 ± 0.05 | 54.79 ± 0.05 | 53.73 ± 0.00 |

单位均为 tok/s。表中的 `±` 是单个进程内部 5 次重复的统计结果。

### 4.2 Qwen3-0.6B 汇总

| 场景 | MNN 三轮数据 | llama.cpp 三轮数据 | MNN 均值 | llama.cpp 均值 | MNN 提升 |
|---|---|---|---:|---:|---:|
| pp512 | 375.24 / 376.78 / 391.69 | 356.71 / 357.14 / 356.68 | 381.24 | 356.84 | **+6.84%** |
| tg128 | 54.75 / 53.93 / 54.79 | 53.65 / 53.72 / 53.73 | 54.49 | 53.70 | **+1.47%** |

计算方式：

```text
pp512 = (381.236667 / 356.843333 - 1) * 100% = 6.8359%
tg128 = (54.49 / 53.70 - 1) * 100% = 1.4711%
```

运行时报告的模型文件大小分别为：

- MNN：320.16 MiB
- llama.cpp Q4_1：477.20 MiB

文件大小差异来自模型容器、量化 block 和元数据布局差异，不参与 tok/s 提升比例计算。

### 4.3 Qwen3-1.7B 跨模型验证

三轮原始结果：

| 轮次 | MNN pp512 | llama.cpp pp512 | MNN tg128 decode | llama.cpp tg128 |
|---:|---:|---:|---:|---:|
| 1 | 167.44 ± 0.81 | 167.90 ± 0.03 | 22.07 ± 0.01 | 23.16 ± 0.00 |
| 2 | 167.71 ± 0.44 | 167.97 ± 0.03 | 22.08 ± 0.01 | 23.12 ± 0.01 |
| 3 | 167.02 ± 0.59 | 167.69 ± 0.01 | 22.04 ± 0.06 | 23.13 ± 0.01 |

汇总：

| 场景 | MNN 三轮均值 | MNN 范围 | llama.cpp 三轮均值 | llama.cpp 范围 | MNN 相对变化 |
|---|---:|---:|---:|---:|---:|
| pp512 | 167.39 | 167.02～167.71 | 167.85 | 167.69～167.97 | **-0.28%** |
| tg128 | 22.06 | 22.04～22.08 | 23.14 | 23.12～23.16 | **-4.64%** |

计算方式：

```text
pp512 = (167.390000 / 167.853333 - 1) * 100% = -0.2760%
tg128 = (22.063333 / 23.136667 - 1) * 100% = -4.6391%
```

最保守的“最差 MNN 对最好 llama.cpp”为：

- pp512：`167.02 / 167.97 - 1 = -0.57%`
- tg128：`22.04 / 23.16 - 1 = -4.84%`

Qwen3-1.7B 的 pp512 可以视为同档，但 tg128 的差距稳定存在，不能用进程波动解释。运行时报告的模型文件大小为：

- MNN：923.82 MiB
- llama.cpp Q4_1：1.24 GiB

### 4.4 Qwen3.5 跨模型验证

#### 4.4.1 Qwen3.5-0.8B 三轮原始结果

| 轮次 | MNN pp512 | llama.cpp pp512 | MNN tg128 decode | llama.cpp tg128 |
|---:|---:|---:|---:|---:|
| 1 | 128.70 ± 0.54 | 104.51 ± 0.27 | 33.95 ± 0.01 | 28.46 ± 0.01 |
| 2 | 126.00 ± 0.72 | 104.52 ± 0.22 | 33.51 ± 0.61 | 28.53 ± 0.01 |
| 3 | 128.02 ± 0.45 | 105.79 ± 0.17 | 34.09 ± 0.02 | 28.41 ± 0.02 |

#### 4.4.2 Qwen3.5-2B 三轮原始结果

| 轮次 | MNN pp512 | llama.cpp pp512 | MNN tg128 decode | llama.cpp tg128 |
|---:|---:|---:|---:|---:|
| 1 | 86.10 ± 0.18 | 78.23 ± 0.19 | 17.70 ± 0.01 | 16.11 ± 0.00 |
| 2 | 84.90 ± 0.31 | 79.28 ± 0.07 | 17.69 ± 0.00 | 16.18 ± 0.00 |
| 3 | 85.79 ± 0.36 | 79.36 ± 0.18 | 17.65 ± 0.00 | 16.13 ± 0.00 |

#### 4.4.3 Qwen3.5 汇总

| 模型 | 场景 | MNN 三轮均值 | MNN 范围 | llama.cpp 三轮均值 | llama.cpp 范围 | MNN 提升 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | pp512 | 127.57 | 126.00～128.70 | 104.94 | 104.51～105.79 | **+21.57%** |
| Qwen3.5-0.8B | tg128 | 33.85 | 33.51～34.09 | 28.47 | 28.41～28.53 | **+18.91%** |
| Qwen3.5-2B | pp512 | 85.60 | 84.90～86.10 | 78.96 | 78.23～79.36 | **+8.41%** |
| Qwen3.5-2B | tg128 | 17.68 | 17.65～17.70 | 16.14 | 16.11～16.18 | **+9.54%** |

计算方式：

```text
Qwen3.5-0.8B pp512 = (127.573333 / 104.940000 - 1) * 100% = +21.5679%
Qwen3.5-0.8B tg128 = (33.850000 / 28.466667 - 1) * 100% = +18.9110%
Qwen3.5-2B   pp512 = (85.596667 / 78.956667 - 1) * 100% = +8.4097%
Qwen3.5-2B   tg128 = (17.680000 / 16.140000 - 1) * 100% = +9.5415%
```

MNN 最差新进程轮次仍高于 llama.cpp 最好轮次：

- Qwen3.5-0.8B pp512：`126.00 / 105.79 - 1 = +19.10%`
- Qwen3.5-0.8B tg128：`33.51 / 28.53 - 1 = +17.46%`
- Qwen3.5-2B pp512：`84.90 / 79.36 - 1 = +6.98%`
- Qwen3.5-2B tg128：`17.65 / 16.18 - 1 = +9.09%`

运行时报告的文本模型文件大小为：

| 模型 | MNN `llm.mnn.weight` | llama.cpp Q4_1 |
|---|---:|---:|
| Qwen3.5-0.8B | 403.75 MiB | 497.43 MiB |
| Qwen3.5-2B | 1009.98 MiB | 1.19 GiB |

MNN 的 `visual.mnn` 没有进入上述大小或 benchmark；本次 llama.cpp Q4_1 量化器则保留 token embedding 为 Q6_K。
这些格式差异意味着文件大小只能用于描述产物，不能作为性能结论的证据。

0.8B 到 2B 时 MNN 的领先幅度有所收窄。一个合理但尚未由 profile 证明的解释是：两个 Qwen3.5 模型的
LinearAttention recurrence 核心维度相同，而 2B 的 Linear/MLP 权重更大，使端到端执行更偏向权重带宽和 GEMM。
本文只将其作为待验证推测，不作为 benchmark 结论。

### 4.5 新增模型的频率、日志与产物指纹

Qwen3-1.7B 和 Qwen3.5 的全部正式日志均包含命令标签及 pre/post 快照：

- 所有采样点的 CPU 频率均为 1.8 GHz。
- Qwen3-1.7B 的温度采样范围约为 54～63°C。
- Qwen3.5 的温度采样范围约为 54～63°C。
- warm-up 日志没有计入任何正式均值。

正式日志路径见第 2.1 节。关键产物 SHA-256：

| 产物 | SHA-256 |
|---|---|
| Qwen3-1.7B MNN `llm.mnn.weight` | `c23fbb42e2c7f0711af5baee1807aded580a84ef54c3a337123847568f254131` |
| Qwen3-1.7B llama.cpp Q4_1 | `eb5880ceae8368ab666bf1e94f107170e01871774a9e465ff329cd2c41b97f29` |
| Qwen3.5-0.8B MNN `llm.mnn.weight` | `c821b6d61b4b0b16f6d81dec5f9f73011e673731878bd28494d75a420caf17fe` |
| Qwen3.5-0.8B llama.cpp Q4_1 | `d336b08c601395d64f279af7da34ab023a583f8cd78e92f20d9fe1f37a78c4a5` |
| Qwen3.5-2B MNN `llm.mnn.weight` | `ffe324a8a4f3ad04c0be9bd9929a8dacb9020f61b7aa411cf2c81cb59215a369` |
| Qwen3.5-2B llama.cpp Q4_1 | `481d33314887f8e3098fb3f57fd780fb368c00598aad5885531c054832eda081` |
| MNN `llm_bench`（正式三轮测试） | `7bf66f4cc2cd86543eaf9e573258b4327515a53e168bb5aff524b434c9ad47a5` |
| K3 llama.cpp `llama-bench` | `72e4a8351f2b689e36d23169043ed61ba908e57abec6b2495cd4fecc7ac18376` |

Qwen3.5 的 MNN 与 llama.cpp 短生成均输出连贯中文；Qwen3-1.7B MNN 短生成也通过。
llama.cpp 的正式 benchmark 每轮均检测到 IME2/TCM，但 `/dev/tcm_sync_mem` 不可用，
TCM 同步 barrier 回退到 heap；所有轮次行为一致且 benchmark 正常完成。

### 4.6 单编译宏清理后的性能回归

将 IME2 路由收敛到单一编译宏、删除运行时环境变量解析和永久关闭的 profile/trace 计时后，
在同一块 K3 上重新构建。以下命令均没有设置任何 `MNN_SPACEMIT_IME2*` 或 `MNN_RVV_*`
调优环境变量。

Qwen3-0.6B 使用三个新进程、每进程内部重复五次：

| 场景 | 新轮次 1 | 新轮次 2 | 新轮次 3 | 新均值 | `f4ffc5e487` 历史均值 | 相对变化 |
|---|---:|---:|---:|---:|---:|---:|
| pp512 | 395.41 | 405.86 | 394.96 | **398.74** | 381.24 | **+4.59%** |
| tg128 | 55.08 | 54.88 | 54.98 | **54.98** | 54.49 | **+0.90%** |

为减少温频差异，还在重新构建前用同机旧二进制各跑了一个五次重复进程：

| 场景 | 同机旧二进制 | 单宏新版本均值 | 相对变化 |
|---|---:|---:|---:|
| pp512 | 397.28 | 398.74 | **+0.37%** |
| tg128 | 53.19 | 54.98 | **+3.37%** |

另外三个模型各运行一个五次重复进程作为跨模型回归检查；该表不是新的三进程正式均值：

| 模型 | 场景 | 单宏版本 | 历史 MNN 三轮均值 | 相对变化 |
|---|---|---:|---:|---:|
| Qwen3-1.7B | pp512 | 169.12 | 167.39 | +1.03% |
| Qwen3-1.7B | tg128 | 22.37 | 22.06 | +1.41% |
| Qwen3.5-0.8B | pp512 | 127.92 | 127.57 | +0.27% |
| Qwen3.5-0.8B | tg128 | 34.67 | 33.85 | +2.42% |
| Qwen3.5-2B | pp512 | 86.44 | 85.60 | +0.98% |
| Qwen3.5-2B | tg128 | 18.08 | 17.68 | +2.26% |

本轮日志保存在 `/home/yanxing/bench-ime2-macro-cleanup`。动态链接后的
`libMNN.so` SHA-256 为
`8631ec2e8663f537c556ed814ff2e999dbea2b324136023527a90f17b2da3e78`；
该轮 `llm_bench` 可执行文件相对正式三轮测试未变化，SHA-256 仍为
`7bf66f4cc2cd86543eaf9e573258b4327515a53e168bb5aff524b434c9ad47a5`。

随后一个交付候选又移除了只改变 `--profile` 逐节点排序、未被正式命令启用的 `Profiler.cpp`
修改。按该阶段源码重建后的 `llm_bench` SHA-256 为
`1bde1bd0795e1a4010cb183524f42d8bf35eab43d1cc3742cd478d350ba83c70`。
该阶段二进制各运行一个五次重复进程，Qwen3-0.6B pp512 为
`395.69 ± 3.02 tok/s`，tg128 decode 为 `55.22 ± 0.12 tok/s`，均处于上述单宏回归范围；
这两个单进程结果仅用于验证该阶段工具二进制，不替代正式三进程比较。

### 4.7 第一阶段 Attention RVV 分层重构 A/B（历史）

将 RISC-V/K3 的门禁、TCM adapter 和 kernel 注册从 `CPUAttention.cpp` 拆到独立 RVV
实现后，使用两个隔离 worktree 和 build 目录做源码 A/B。Baseline 为提交 `720f69d641`，
Candidate 只增加本节记录的 Attention 分层重构；`ldd` 确认两侧分别加载各自的
`libMNN.so`、`libllm.so` 和 `libMNN_Express.so`。

测试在单个持久 SSH 会话内执行。开发板后台登录服务稳定后，使用
Baseline→Candidate→Candidate→Baseline 的对称顺序抵消随时间变化的系统负载：

| 场景 | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 | Baseline 均值 | Candidate 均值 | 相对变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp512 | 352.69 ± 2.87 | 362.44 ± 3.04 | 370.17 ± 3.24 | 381.99 ± 4.48 | 367.34 | 366.31 | -0.28% |
| tg128 | 54.16 ± 0.38 | 55.01 ± 0.36 | 54.08 ± 1.29 | 54.99 ± 0.04 | 54.575 | 54.545 | -0.055% |

pp512 存在明显的进程间时间漂移，但对称均值差异小于单进程内部波动；tg128 的差异
仅为 0.055%。全程 CPU 频率保持 2.2/1.8 GHz，最高温度约 67°C，无降频。
因此结论是分层重构没有可测性能回退，不将上述噪声内变化计为新的性能提升或下降。
Candidate 的 16-token 短生成同时通过。

本节只覆盖提交 `720f69d641` 后的第一阶段 Attention 拆分；后续函数表方案的历史回归
见下一节，最终 Execution 子类解耦方案见第 4.9 节。

### 4.8 函数表三层重构后的历史回归

最终候选在远端独立 worktree `/tmp/AliNNPrivate-ime2-redesign` 和独立 build
`/tmp/AliNNPrivate-ime2-redesign-build` 中重新配置、全量编译和运行，没有修改
`/home/yanxing/AliNNPrivate` 的现有工作区。

Qwen3-0.6B 使用 `llm_bench` 在持久 SSH 会话中运行三个五次重复进程。
单独 SSH 登录会短暂拉起远端 user-systemd、pipewire 和 python 服务，因此相关受干扰结果
被丢弃。三层分层完成后首先得到：

| 场景 | 三个进程 | 均值 | 第 4.2 节重构前正式均值 | 相对变化 |
|---|---|---:|---:|---:|
| pp512 | 389.04 ± 3.27 / 391.82 ± 4.86 / 367.89 ± 1.79 | **382.92** | 381.24 | **+0.44%** |
| tg128 decode | 54.58 ± 0.06 / 54.57 ± 0.04 / 54.69 ± 0.02 | **54.61** | 54.49 | **+0.23%** |

资源终审随后发现旧 resource 支持执行期 rebind，且 cached-mmap 会绕过绑定。交付代码将
resource 改为一次性绑定 immutable weight，并用 resource-owned cache + TLS weak-owner/
aliasing-handle 保证生命周期；同时移除了 RV64 热查找中的多余 acquire fence。按这份最终
源码重新构建后又运行三个成功的新进程：

| 场景 | 三个交付进程 | 交付均值 | 第 4.2 节重构前正式均值 | 相对变化 |
|---|---|---:|---:|---:|
| pp512 | 365.94 ± 1.31 / 384.07 ± 4.90 / 378.64 ± 2.92 | **376.22** | 381.24 | **-1.32%** |
| tg128 decode | 54.51 ± 0.08 / 54.15 ± 0.04 / 54.43 ± 0.03 | **54.36** | 54.49 | **-0.23%** |

pp512 仍存在明显的进程间漂移，交付均值比前一批低 1.75%；tg128 相对前一批低 0.46%，
相对重构前正式均值低 0.23%，没有出现结构性性能回退。若只参考第 4.2 节 llama.cpp
正式均值，则交付代码 pp512 与 tg128 分别高 `5.43%` 和 `1.24%`。本轮没有同步重跑
llama.cpp，因此这些百分比只用于确认既有结论，不替代第 4.1～4.2 节的双引擎交替正式比较。

最终 resource-safe TLS 弱缓存还与历史 `720f69d641` 二进制在同一持久会话中做了穿插
控制：新版本 tg128 为 54.58/54.57 tok/s，控制版本为 54.38 tok/s；新版本 pp512
为 389.04/391.82 tok/s，控制版本为 369.04 tok/s。该控制只用于排除重构回退，不计为
新的性能收益。

最终 resource/TLS 代码的 8-token IME2 短生成输出连贯中文；此前 16-token 验证同样通过。
随后将 `MNN_RVV_SPACEMIT_IME2` 关闭，重新编译并链接
标准 RVV 版本，4-token 短生成同样输出正确。标准 RVV 版本速度较慢是预期行为，本轮只验证
回退路径的链接和运行正确性。

最终构建 flags 也做了直接核对：

- `MNNCPU`：只有 `MNN_USE_RVV`，没有 `MNN_USE_SPACEMIT_IME2` 或 `xsmtvdotii`
- `MNNRVV`：只有 `MNN_USE_RVV`，使用标准 base march
- `MNNSpacemitIme2Runtime`：定义 `MNN_USE_RVV` 和 `MNN_USE_SPACEMIT_IME2`，
  使用标准 base march
- `MNNSpacemitIme2`：定义相同两个宏，只在 kernel target 追加 `_xsmtvdotii`

因此这轮结果同时验证了当时函数表方案的功能回归；最终 Execution 子类解耦回归另见下一节。
正式 MNN 与 llama.cpp 性能结论仍以第 1 节和第 4.1～4.5 节的三进程交替数据为准。

### 4.9 Execution 子类解耦回归

最终结构删除 Linear/Attention 的 config/args/state 镜像函数表，改为
`CPUExtension` 工厂创建 RVV/IME2 Execution 子类。K3 重构前后结果如下：

| 场景 | 重构前 | Execution 子类版 | 相对变化 |
|---|---:|---:|---:|
| pp64 | 503.35 ± 0.34 | 500.65 ± 4.80 tok/s | -0.54% |
| pp256 | 436.32 ± 6.72 | 434.04 ± 0.62 tok/s | -0.52% |
| pp1024 | 180.17 ± 0.45 | 182.67 ± 0.55 tok/s | +1.39% |

三项 prefill 均处于 ±3% 验收范围。pp512 在板端负载较低时为
374.95 ± 2.38 tok/s，与第 4.8 节最终函数表版本 376.22 tok/s 基本一致；负载继续升高后
同一二进制降至 350.45 ± 2.05 tok/s，因此不将该低值解释为代码回退。

decode 验证期间，旧版 control 的绝对速度也从阶段基线 54.58 tok/s 降到约
51.5 tok/s。为排除无关系统负载，采用同一持久会话内的 candidate/control 配对比较：

| 轮次 | Execution 子类版 | 同机旧版 Control | 相对变化 |
|---|---:|---:|---:|
| 1 | 50.96 ± 0.53 | 51.52 ± 0.17 tok/s | -1.09% |
| 2 | 51.63 ± 0.02 | 51.56 ± 0.03 tok/s | +0.14% |
| 3 | 50.78 ± 0.94 | 51.36 ± 0.19 tok/s | -1.13% |
| 三轮均值 | 51.1233 | 51.4800 tok/s | **-0.69%** |

prefill/decode 专用路径均用一次性日志确认实际命中，随后已移除诊断代码。最终源码在
macOS arm64 上完成 388/388 全量单测和 8-token 短生成；K3 IME2 ON 的完整链接和
8-token 短生成也已通过。最后增加的 factory 空指针短路及 LOW_MEMORY=OFF 源文件裁剪
已通过本机编译和静态宏组合终审；补跑最新 K3 二进制时开发板网络暂时不可达，待恢复后
只需补记该轮 smoke，不影响上述已完成的 kernel/Execution A/B 结论。

## 5. 优化过程

最终交付前，本分支相对 `e9398e831c` 的 K3 优化由以下开发提交逐步形成；
交付时这些历史已压缩为一个提交：

| squash 前提交 | 内容 |
|---|---|
| `32f493b780` | 增加 K3 SpacemiT IME2 int8/int4 GEMM 基础路径 |
| `ac3668b81d` | 优化 IME2 decode Linear 路径 |
| `515cbdafb2` | 增加 decode pipeline、Attention direct-matvec 和 scratch 优化 |
| `f4ffc5e487` | 完成非对称 W4B64 prefill/decode、Attention 和 KV Cache 优化 |

以下章节描述以 Qwen3-0.6B 为目标完成的实现。Linear/MatMul、通用 decode Attention 和 KV Cache 改动可被其他模型复用，
但 fused Attention 等路径带有明确 shape 门禁；第 4 节的跨模型端到端结果不代表所有模型走过完全相同的 kernel 集合。

早期探索阶段曾记录过 pp128 的旧路径约 222～225 tok/s、第一阶段 IME2 优化约 264 tok/s。这些数据的模型、量化和输入口径没有全部按最终协议锁定，只用于定位瓶颈，不作为本报告的最终性能结论。

### 5.1 IME2 Linear/MatMul 专用路径

为 K3 增加 IME2 int8 × int4 GEMM、权重重排和 executor 路由，使 LLM Linear 不再依赖通用 RVV W4 路径。

核心工作包括：

- K3 IME2 I8/I4 本地 kernel
- W4 权重预打包和缓存
- hierarchical K32/K256 数据布局
- 针对 prefill 的 M4 路径
- 针对 decode 的 M1 路径
- 按输出通道分组的 worker 调度
- 不满足 shape、VLEN 或量化条件时回退原实现

最终结构通过 `CoreFunctions::extension` 中的架构无关工厂选择执行器。只有 K3
registration 安装工厂且算子满足 W4、动态量化和 GEMM unit 等基础条件时，才创建
`SpacemitIme2ConvInt8Executor`；其他 CPU 和普通 RVV 继续创建原
`DenseConvInt8TiledExecutor`。

全部 K3 shape 门禁、pack-scale 融合、prefill/decode 调度及 IME2 resource 都位于
`riscv/rvv/spacemit_ime2/` 的子类中。子类 `onExecute` 先尝试专用路径，未命中或执行失败
时直接调用基类实现。通用 `ConvInt8TiledExecutor` 不再包含 RISC-V/IME2 宏、专用参数
镜像或热路径分支，只保留架构中性的工厂接入、resize hook，以及一个 weight reorder
完成回调。

weight reorder 回调解决了通用权重重排与 IME2 预打包的时序关系：普通模型只有在
main/branch 权重全部重排成功后，才在同一个初始化任务中 prepare；任一重排失败都会
禁用专用 resource。cached-mmap 模型则在缓存布局确认后同步 prepare。回调只捕获共享
resource，不捕获 executor 的 `this`，避免异步初始化期间出现悬空对象。

每个 IME2 executor/clone 共享持有一个 `LinearResource`。IME2 的 packed-B 和
非对称 residual cache 全部属于该 resource；每个 resource 只允许绑定一个 immutable
STATIC weight，异地址重复绑定会失败，不存在执行期 rebind、cache clear 或 generation
切换。普通模型在 weight reorder 全部完成后 prepare，cached-mmap 模型在提前返回前
prepare。最后一个 clone 释放时统一销毁，不再通过裸 weight 地址维护全局长生命周期缓存。

预打包阶段只创建生产 prefill 实际使用的 fused-residual HP 布局。decode 专用
asymmetric-pair 布局仅在符合门禁并首次执行时按需创建；旧的 standard、普通 HP、
symmetric 和额外 residual 布局不再一次性全部常驻。这样既保留 prefill 首次执行前的
预热，也避免原 eager-prepack 同时复制多份无用权重。

decode 热路径使用 resource-safe 的 TLS 索引 asymmetric-pair packed-B 和
weight-bias residual。TLS 保存 weak owner 和只读 raw value；命中时用 control-block
identity 防止 resource 地址复用造成 ABA，再返回以 resource 为 owner 的 aliasing
`shared_ptr`。TLS 本身不强持有数据，最后一个 `ResourceInt8` 析构仍会释放 packed
weight；当前 resource cache 只增不删，因此 alias 生命周期内 raw value 保持有效。

### 5.2 将动态量化 scale 合入 A pack

原路径先执行一次 `BatchSymDynamicQuantScaleOnly`，计算每行激活的 `absmax`、量化 scale 和反量化 scale，然后再次遍历输入完成 A pack。

新增：

```text
MNNSpacemitIme2PackFloatAHpStridedRowsRangeDynamicQuant
```

该函数在 A pack 阶段一次完成：

1. 每行 `absmax`
2. `quantScale = 127 / absmax`
3. `inputScale = absmax / 127`
4. int8 A pack
5. 非对称 W4 修正所需的 `srcKernelSum`

K3 IME2 专用构建固定启用这一路径。它减少一次完整激活遍历和独立 worker pass，
但不会改变权重的非对称量化属性。

### 5.3 Strided Linear 调度

旧实现会将多个 M tile 拆成较多独立 worker job，造成：

- worker dispatch 次数偏多
- 小任务同步开销占比高
- A pack、GEMM 和 post-process 之间的局部性较差

K3 IME2 专用构建固定使用 strided Linear 调度：一个 worker 按固定 row stride
连续处理多个 M4 tile，并复用已经准备好的 post 参数和源数据布局。

### 5.4 非对称 W4B64 fused-residual kernel

prefill 主路径使用 blk258 hierarchical kernel：

- 保留 W4B64 block scale
- 精确计算非对称 zero-point/source-sum residual
- 将 residual correction 融入 IME2 累加
- 固定 M stride 为 4

K3 IME2 专用构建固定启用精确 fused-residual。

这里的 residual 指非对称权重量化修正，不是将权重近似成 centered/symmetric W4。对不能精确满足 fast-path 条件的权重和 shape，仍回退旧路径。

### 5.5 Direct-C4 epilogue

prefill 的主要新收益来自 M4 direct-C4 epilogue：

```text
MNNSpacemitIme2GemmI8I4HpM4DirectC4Local
```

blk258 kernel 在寄存器中已经持有四行输出以及完整非对称 W4B64 residual。新 epilogue 继续在寄存器中完成：

1. 每行 activation scale
2. bias
3. 可选 min/max clamp
4. C4 layout 重排
5. 直接写最终输出 Tensor

这样避免了：

- 写入完整临时 `cBuffer`
- post-process 再次读取 `cBuffer`
- 独立的 layout conversion
- 额外的中间结果带宽

该路径在 K3 IME2 专用构建中固定启用，但仍带有严格的 blk258、M4、N 对齐、
FP32/C4 和 post 参数门禁；任一条件不满足即回退原 GEMM + post 路径。

### 5.6 Decode 非对称 pair kernel

decode 的 `M=1` 与 prefill 的 `M=4` 特征不同，主要受权重带宽、pack 和小任务调度影响。

最终 decode 路径包括：

- 从原始 FP32 activation 直接生成 hierarchical HP A layout
- 为 W4B64 非对称权重生成 exact asym-pair packed-B layout
- blk261 M1 native IME2 kernel
- 成对处理一个 block64 内的两个 K32 子块，复用 block64 scale/correction 元数据
- 在 kernel 内精确完成 dot、block scale 和非对称 correction
- worker 内完成 bias/post，减少额外 dispatch

K3 IME2 专用构建固定启用 FP32 hierarchical pack 和非对称 M1 pair kernel。
若 shape、block 数、输出 tail、VLEN 或 packed weight 条件不满足，执行器会回退通用 decode 路径。

### 5.7 Decode Attention direct-matvec 与 scratch 控制

decode Attention 的 query 行数为 1，不需要为通用 batch MatMul 保留全部临时 Tensor。

优化后：

- QK 与 PV 使用 RVV `WithAStride` direct-matvec
- tg128 的 QK score 优先复用 `CPUAttention` 已有的 `mPackQ` workspace
- workspace 不足时，由 `MNNRvvAttention` 实例持有一个 decode scratch
- 最大 4096 context 的持久 scratch 最多为 128 KiB/层
- 28 层的最坏常驻上限约 3.5 MiB
- 更长 context 使用本轮临时申请并在结束时释放的 transient buffer，避免每层常驻大块内存

按旧分配模型每层约 `576 * context` 字节计算，28 层、4K/8K/32K context 的估算常驻 scratch
约为 63/126/504 MiB。新设计避免了随最大 context 在每层重复放大。

direct-matvec 属于通用 RVV 优化，不依赖 IME2 指令；满足 shape、layout 和 scratch 条件时启用，
否则保留原 Attention 路径作为 fallback。

为了避免通用 `CPUAttention` 混入架构宏，RVV 的 direct-matvec 适用条件、
`WithAStride` kernel 入口和 scratch 生命周期都下沉到 `MNNRvvAttention` 子类。
scratch 是 Execution 实例成员，在 resize 和析构时释放，不进入全局函数表或通用 CPU
类成员。

### 5.8 标准 RVV decode 与 K3 fused Attention 分层

Attention 最终按 Execution 继承关系拆成三层：

```text
CPUAttention
  -> MNNRvvAttention：decode direct-matvec
       -> MNNSpacemitIme2Attention：K3 fused Attention
```

具体职责如下：

- `rvv/MNNRvvAttentionFunctions.cpp`：只实现标准 RVV decode direct-matvec、
  per-Execution scratch 和 RVV 子类工厂，不包含 K3/TCM 逻辑。
- `rvv/spacemit_ime2/MNNSpacemitIme2AttentionFunctions.cpp`：实现 K3 fused
  shape/layout 门禁、TCM task adapter 和 pair-head → single-head 回退；fused 未命中或
  失败后调用父类的标准 RVV 路径。
- `rvv/spacemit_ime2/MNNSpacemitIme2AttentionKernels.cpp`：实现 K3-only online
  softmax、single-head 和 GQA group=2 pair-head fused kernel。
- `MNNSpacemitIme2RunTcmTasks` 与 IME2 worker/TCM runtime 共享，定义在 vendor
  `MNNSpacemitIme2GemmInt8.cpp` 中。

`CPUExtension::createAttentionExecution` 决定创建通用、RVV 或 IME2 子类。通用
`CPUAttention.cpp` 在 KV update 和 mask/shape 解析后只调用一次架构无关的窄虚函数
hook；之所以不直接在子类重写整个 `onExecute`，是为了避免复制通用 mask、量化和 KV
更新逻辑，以及失败回退时重复写 KV。子类只有在完整 Attention 输出已经写完时才返回
`true`；返回 `false` 后才分配原通用临时 Tensor 并执行原有 kernel 路径。通用文件中
没有 RISC-V/IME2 宏、专用符号、TCM 常量或 K3 shape 门禁。

Qwen3-0.6B 的两个 query head 共享一个 KV head。pair-head kernel 一次装入共享 K/V，并计算两个 query head，减少 K/V pack 和读取。

最终 fused Attention 只在以下严格条件下启用：

- 构建时定义 `MNN_USE_SPACEMIT_IME2`
- RISC-V RVV，`vlenb == 128`
- FP32、C4
- 16 个 query head、8 个 KV head
- head dimension 128
- 8 个计算线程
- causal flash attention
- 无 attention sinks
- 无 KV 量化
- `seqLen == kvSeqLen`
- seqLen 为 64 的倍数，范围为 64～512
- KV block 为 64
- 无 padding

完整回退链路为：

```text
K3 fused Attention
  -> 标准 RVV decode direct-matvec
       -> 通用 CPU Attention
```

每一层只在自己完整处理输出时返回成功，不会让通用 CPU 路径理解架构内部状态。

Qwen3.5 full Attention 的 8Q/2KV/head dimension 256 以及混合 LinearAttention 结构不满足上述门禁，
因此第 4.4 节的
Qwen3.5 性能不能归因于这个 Qwen3-0.6B pair-head fused Attention kernel。

### 5.9 KV Cache pack 与更新

FP32、非量化 KV Cache 路径增加：

- Key 使用 `MNNPackCUnit` 批量打包
- C4 Value 按连续 block 使用 `memcpy`
- RISC-V 上按 KV head 并行更新

这些优化减少逐 token、逐元素地址计算和串行 KV head 更新开销。路径仅在 FP32、非量化 KV Cache 和满足布局条件时启用。

### 5.10 单编译宏与运行时门禁

K3 IME2 专用实现只由 `MNN_USE_SPACEMIT_IME2` 这一 C++ 编译宏决定是否编入并启用。
CMake 配置项 `MNN_RVV_SPACEMIT_IME2=ON` 会创建两个独立 OBJECT target：

- `MNNSpacemitIme2Runtime`：使用标准 RVV base march，编译 Execution 子类和 K3
  registration，不生成 vendor 指令。
- `MNNSpacemitIme2`：只编译 IME2 Attention/Linear kernel，使用
  `${MNN_RVV_BASE_MARCH}_xsmtvdotii` 和 vendor 编译选项。

两个 target 都 PRIVATE 定义同一个 `MNN_USE_SPACEMIT_IME2` 宏；`MNNCPU` 只看到
架构无关的 `CPUExtension` 工厂接口，标准 `MNNRVV` 始终使用移除 `_xsmtvdotii`
的 base march，二者均不定义 IME2 宏。ON/OFF 构建通过 CMake 互斥选择标准或
vendor registration；标准 RVV W8 kernel 始终保留为 fallback，vendor W8/W4
使用独立符号并由函数表覆盖，不再用同名符号替换通用实现。
`CommonOptFunction.cpp` 和 `Int8FunctionsOpt.cpp` 只调用稳定的
`MNNRvvInitialize*` 注册入口，不引用 Spacemit 实现名。

IME2 Linear、Attention 和本地 kernel 不读取 `MNN_SPACEMIT_IME2*`、`MNN_RVV_*`
或其他调优环境变量。静态扫描确认本次删除 91 个 `getenv` 调用，新增为 0；
旧宏 `MNN_USE_SPACEMIT_IME2_ASM` 也已完全移除。

运行时仍保留 shape、layout、空指针、VLEN、量化格式、packed weight 和 TCM 可用性等
正确性/资源门禁；条件不满足时自动回退通用实现。通用 RVV decode direct-matvec
仍由 RVV 能力和 shape 门禁控制，不被错误绑定到 K3 IME2 宏。

## 6. 独立 IME2 target、单宏与固定路径

K3 构建配置入口为：

```bash
cmake ... -DMNN_RVV_SPACEMIT_IME2=ON
```

该选项只决定是否加入独立 runtime/kernel OBJECT target 和互斥 registration。普通 RVV
构建使用 `OFF`，不会编译 `rvv/spacemit_ime2/` 下的源文件，也不需要 K3 指令扩展。

运行 `llm_demo` 或 `llm_bench` 时不需要任何 IME2 环境变量。满足各自运行时门禁后，
专用构建固定使用以下已验证路径：

- `SpacemitIme2ConvInt8Executor` Linear 专用执行器
- FP32 activation 直接进入 A pack，并在 pack 中计算动态 scale
- strided M4 worker 调度
- 精确非对称 W4B64 fused-residual
- prefill direct-C4 epilogue
- decode FP32 hierarchical pack 与非对称 M1 pair kernel
- 严格 shape 门禁下的 K3 fused Attention

生产构建固定关闭 fused-linear、TCM decode ping-pong pipeline 和 symmetric-W4 实验路径；
它们不能通过环境变量重新启用，后续实验需要修改源码并重新编译。

TCM decode pipeline 的实验思路是把两个 worker 分为 copy 与 compute 角色；每个 worker 持有自己的 TCM buffer，
barrier 用于交替 copy/compute 阶段，而不是交换 buffer 指针。源码仍保留实验实现，
但生产配置固定关闭，报告中的性能提升不依赖该双缓冲路径。

fused Attention 使用 TCM task pool 和每 worker scratch，是对 TCM 的有效利用，但不等同于上述 Linear packed-B 双缓冲实验。

## 7. 正确性与工程验证

### 7.1 Direct-C4

开发阶段在真实模型调用上同时计算 direct-C4 与旧 epilogue：

```text
checked calls: 16
mismatched_c4: 0
max_abs: 0
```

验证完成后，自检和 trace 代码已从生产路径移除。

### 7.2 Decode 非对称 pair kernel

标量 oracle 分别核对：

- 原始 dot
- block scale
- zero-point/source-sum correction
- bias/post 输出

目标用例达到逐位一致。短生成测试中，优化路径与相同 fused-residual 配置下关闭 direct-C4/fused Attention 的参考路径输出一致。

### 7.3 Fused Attention

K3 harness 覆盖 seq64、seq128 和 seq512：

- `vlenb = 128`
- pair kernel 全部返回成功
- seq128 single-head 与 pair-head 逐位一致
- 对 FP64 参考的最大绝对误差：`6.65548e-4`
- RMSE：`2.64e-5`～`5.75e-5`
- 无 NaN/Inf
- `RESULT failures=0`

### 7.4 构建与提交检查

- 精确源码同步到 K3 隔离 worktree，未修改远端现有工作区
- `MNN_RVV_SPACEMIT_IME2=ON` 的 K3 完整构建成功，`libMNN.so`、`libllm.so`、
  `llm_bench` 和 `llm_demo` 均链接成功
- 实际 flags 确认 `MNNSpacemitIme2Runtime` 使用 base march，
  `MNNSpacemitIme2` 单独带 `_xsmtvdotii`；二者定义同一个 IME2 宏，
  `MNNCPU` 与标准 `MNNRVV` 均不带 vendor 宏/ISA
- `MNN_RVV_SPACEMIT_IME2=OFF` 的标准 RVV `MNNRVV`、`MNNCPU`、`libMNN.so`
  和 `llm_demo` 构建、链接及 4-token 短生成通过
- 本地 ARM 的 `MNNCPU`、`llm_bench`、`llm_demo` 和 `run_test.out` 构建通过
- `op/attention`、`op/attention_c4`、`op/attention_c4_tail` 回归通过
- 整个 `rvv/spacemit_ime2/` 目录中环境变量读取 API 为 0
- 最终 resource/TLS 代码 Qwen3-0.6B 8-token 短生成、pp512 和 tg128 回归通过；
  此前 16-token 验证同样通过
- 优化历史中的 `./test.sh static`、四个模型短生成和 Qwen3-0.6B 2800-token
  长 prompt 验证仍保留；本节不把它们冒充为 2026-07-29 的重新执行结果
- `git diff --check` 通过
- squash 前性能基线提交：`f4ffc5e487`
- K3 性能优化主体提交：`b083c45d6c`；Execution 子类解耦为后续结构重构

## 8. 关键实现文件

| 文件 | 主要内容 |
|---|---|
| `source/backend/cpu/riscv/CMakeLists.txt` | 分离标准 `MNNRVV`、base-march runtime 与 vendor-ISA kernel target |
| `source/backend/cpu/compute/CPUExtension.hpp` | 架构无关的 Linear/Attention Execution 工厂 |
| `source/backend/cpu/compute/ConvInt8TiledExecutor.cpp/.hpp` | 通用 Linear fallback、resize hook 和 weight reorder 完成回调 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2ConvInt8Executor.cpp/.hpp` | K3 Linear 子类、eligibility、pack-scale、prefill 和 direct decode |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2GemmInt8.cpp` | K3 pack/GEMM/cache/worker/TCM/W8/W4 实现 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2GemmI8I4Local.cpp` | K3 IME2 本地汇编 kernel、M4 direct-C4、M1 asym-pair |
| `source/backend/cpu/riscv/rvv/MNNGemmInt8AddBiasScale_16x4_Unit_RVV.cpp` | 标准 RVV W8 kernel，始终保留为通用 fallback |
| `source/backend/cpu/CPUAttention.cpp/.hpp` | 架构中性的窄虚函数 hook、工厂分发和原通用 fallback |
| `source/backend/cpu/riscv/rvv/MNNRvvAttention.hpp` | 标准 RVV Attention 子类接口和 per-Execution scratch |
| `source/backend/cpu/riscv/rvv/MNNRvvAttentionFunctions.cpp` | 标准 RVV decode direct-matvec 子类实现 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2Attention.hpp` | K3 Attention 子类接口 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2AttentionFunctions.cpp` | K3 fused 实现、TCM 门禁及向 RVV 父类回退 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2AttentionKernels.cpp` | K3 online softmax、single/pair fused Attention kernel |
| `source/backend/cpu/riscv/rvv/MNNRvvFastPathRegistration.cpp` | 标准 RVV 注册入口 |
| `source/backend/cpu/riscv/rvv/spacemit_ime2/MNNSpacemitIme2FastPathRegistration.cpp` | K3 vendor 注册入口和 W8/W4 覆盖 |
| `source/backend/cpu/riscv/rvv/MNNRvvFastPathUtils.hpp` | RVV 子类侧同步并行调度 helper |
| `source/backend/cpu/CPUKVCacheManager.cpp` | FP32 KV Cache 快速 pack 和 KV head 并行更新 |
| `source/backend/cpu/compute/CommonOptFunction.{h,cpp}` | `CPUExtension`/KV 并发字段和稳定 RVV 注册调用 |
| `source/backend/cpu/compute/Int8FunctionsOpt.cpp` | 标准 RVV int8 初始化和稳定 vendor override 注册调用 |

## 9. 回退与问题定位

生产版本不再支持通过环境变量逐项关闭 IME2 子路径。

整体对照应使用两个独立 build 目录：

- K3 IME2 构建：`MNN_RVV_SPACEMIT_IME2=ON`
- 通用 RVV 构建：`MNN_RVV_SPACEMIT_IME2=OFF`

单个子路径的问题应通过 kernel harness 或独立诊断分支定位，不重新引入运行时环境开关。
性能回归仍应使用第 3 节中对应模型的 pp512/tg128 参数运行至少三个新进程轮次。

## 10. 局限与后续方向

1. Qwen3-0.6B pp512 的 MNN 进程间波动高于 llama.cpp，且该批早期数据没有完整温频快照。
   Qwen3-1.7B/Qwen3.5 已补齐逐进程日志和 pre/post 快照，但仍不是连续频率监控。
2. fused Attention 当前门禁针对 Qwen3-0.6B 的具体 head/GQA/sequence 形状；Qwen3.5 明确不满足该门禁，
   因此跨模型结果必须按端到端执行路径解释。
3. Qwen3-0.6B tg128 的均值优势只有 1.47%；Qwen3-1.7B tg128 仍落后 4.64%。
   Qwen3.5 虽在 0.8B/2B 四项中领先，也不能外推到更长 context、视觉输入、MTP 或其他模型架构。
4. MNN W4B64 与 llama.cpp Q4_1 的 block、embedding、容器和元数据布局不同。本报告比较的是各引擎的正式
   低比特端到端路径，不是相同权重字节布局上的单 kernel 对照。
5. llama.cpp 各轮的 `/dev/tcm_sync_mem` 均不可用，TCM 同步 barrier 回退到 heap。该状态在所有轮次一致，
   但如果后续系统镜像提供该设备，需要重新测试，不能直接沿用当前数值。
6. 下一阶段可继续研究：
   - 优先 profile Qwen3-1.7B tg128，定位其稳定的 4.64% decode 差距
   - persistent worker 与更低成本的 decode dispatch
   - packed-B 的 TCM ping-pong 流水，并确保 copy 与 IME2 compute 真正并行
   - 更长 context 的 decode Attention/KV 带宽优化
   - 将 fused Attention 扩展到更多 head 数、GQA ratio 和 sequence 长度
   - 在固定频率、固定温度和连续频率采样下进行更长时间的统计测试
