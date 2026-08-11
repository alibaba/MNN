# RISC-V 远端开发板验证

## 目录

1. 基本原则
2. 连接与工作区检查
3. 构建矩阵
4. 正确性矩阵
5. 性能实验
6. 日志与报告

## 1. 基本原则

本地负责阅读、修改和提交代码，远端 RISC-V 板负责 ISA 编译、运行和性能验证。把主机名、代理、
用户名、仓库目录、build 目录和模型目录作为会话参数，不写入 Skill 或对外文档。

执行远端操作前：

1. 使用用户提供或已配置的 SSH 入口；
2. 确认远端仓库与 build 目录；
3. 只读检查远端分支、HEAD 和工作区；
4. 发现远端已有修改时保留它们，使用隔离 worktree/clone 或明确的文件同步范围；
5. 不运行 `git reset --hard`、`git clean` 或覆盖式全目录同步。

连接不稳定时使用用户已经给出的备用连接方式；不要把代理地址或凭据提交到仓库。

## 2. 连接与工作区检查

使用占位符组织命令：

```bash
ssh <RISCV_HOST> 'uname -m'
ssh <RISCV_HOST> 'git -C <REMOTE_REPO> status --short --branch'
ssh <RISCV_HOST> 'git -C <REMOTE_REPO> rev-parse HEAD'
```

预期 `uname -m` 为 `riscv64`。同时核对：

- 编译器版本与 vendor ISA 支持；
- `lscpu`/runtime 报告的核心和 VLEN；
- TCM runtime/device 是否可用；
- 当前 CPU governor、频率和系统负载；
- 目标模型与量化配置。

同步代码优先使用可审计方式：

- 已授权 push 时，在远端拉取明确提交；
- 尚未提交时，只同步本次修改的显式文件列表；
- 需要隔离时创建独立 worktree，不覆盖用户远端工作区。

同步后用 hash 或 `git diff --check` 确认远端源码与候选版本一致。

## 3. 构建矩阵

复用现有、已配置的 build 目录时，只执行增量构建：

```bash
cmake --build <IME2_BUILD> -j <JOBS>
```

需要新配置时，分别保留两个目录：

```text
Vendor build: MNN_RVV_SPACEMIT_IME2=ON
Pure RVV build: MNN_RVV_SPACEMIT_IME2=OFF
```

不要在同一 build 目录来回切换选项后直接比较性能。检查：

- vendor kernel object 带厂商 ISA 编译参数；
- runtime/Execution object 只使用它实际需要的 ISA；
- 标准 RVV object 不带 vendor extension；
- `libMNN`、LLM 库、`llm_demo`、`llm_bench` 均成功链接；
- OFF 变体没有 vendor symbol 或未解析引用。

构建后检查目标程序和动态库的真实路径、时间戳及 loader 解析结果，确认 benchmark 加载的是
本次 build 产物，而不是旧目录、系统安装或另一变体中的库。必要时结合 `realpath`、`stat`、
`readelf -d` 或目标平台等价工具核对。

本地 ARM/x86 编译适合发现通用接口污染，但不能替代纯 RVV 构建。

## 4. 正确性矩阵

按成本从低到高运行：

| 层级 | 必测内容 |
|---|---|
| Kernel | 主 tile、K/N tail、signedness、scale/offset、direct output |
| Op | DenseConv/MatMul/Attention、C4、tail、线程分片 |
| Vendor smoke | 短 prompt、固定采样、若干 decode token |
| Pure RVV smoke | OFF 变体运行相同 prompt |
| Long prompt | 跨过 prefill/Attention 分支阈值 |
| Cross-model | 至少一个小模型和一个较大/不同结构模型 |

量化 kernel 优先使用逐层比较：

```text
packed input
  -> integer accumulator
  -> dequantized FP32
  -> zero-point correction
  -> epilogue/output layout
```

TCM 路径开发阶段可让同一 tile 同时执行 TCM 与 DRAM kernel 并逐位比较。验证完成后删除重复计算
和诊断输出。

模型输出异常时固定采样方式，比较 fallback、纯 RVV 和 FP16/FP32，先区分 kernel 错误、
量化误差与采样波动。

## 5. 性能实验

使用 `llm_bench` 测量 MNN 端到端性能。命令使用占位路径：

```bash
# Prefill
<BUILD>/llm_bench \
  -m <MODEL_CONFIG> \
  -p <PROMPT_TOKENS> -n 0 -rep <REPEAT> -t <THREADS> \
  -load false

# Decode
<BUILD>/llm_bench \
  -m <MODEL_CONFIG> \
  -p 1 -n <GENERATE_TOKENS> -rep <REPEAT> -t <THREADS> \
  -load false
```

使用 benchmark 原生参数选择 prefill/decode，不增加生产代码中的 `getenv` benchmark 开关。
模型专属 Attention/KV 参数以当前 `llm_bench --help` 和执行图为准，不照抄另一模型。

实验规则：

1. 固定二进制、模型、量化、线程、输入长度和 runtime 选项；
2. 每个候选与基线至少启动多个新进程；
3. 尽量使用 A/B、B/A、A/B 交替顺序；
4. 每个进程内部重复多次，但不要把进程内低方差当成跨进程稳定；
5. 记录每轮均值、范围、频率、温度和系统负载；
6. 同时报告 prefill 与 decode；
7. 做线程数或门槛 sweep 时一次只改变一个变量；
8. 性能提升后重新运行正确性 smoke。

分析 decode 时记录模型权重与元数据的实际读取字节数，计算：

```text
effective_bandwidth = bytes_per_token * tokens_per_second
```

分析 prefill 时同时记录 pack/动态量化时间和 GEMM 时间，防止 kernel 变快但前处理抵消收益。

对 TCM/DMA 流水分别记录：

- copy 时间；
- compute 时间；
- barrier/dispatch 时间；
- 流水总时间；
- DRAM fallback 时间。

只有总时间稳定下降且结果一致才保留。

## 6. 日志与报告

保留：

- 源码版本或候选标识；
- 构建选项；
- 模型量化格式；
- benchmark 完整参数；
- 多进程原始数据和汇总；
- 正确性测试结果；
- 频率/温度等限制条件；
- 回退条件和未覆盖场景。

内部开发记录可以引用远端路径和日志；对外报告必须移除：

- SSH 主机、代理和用户名；
- 内部仓库/build/model 路径；
- 内部提交或评审编号；
- 未公开硬件信息；
- 无法由公开口径支持的竞品结论。

提交前运行敏感信息扫描，并人工检查命令块、表格脚注和图片说明。
