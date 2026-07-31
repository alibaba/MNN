---
name: riscv-cpu-optimize
description: MNN RISC-V CPU 算子与 LLM 性能优化工作流，覆盖标准 RVV、厂商矩阵扩展（如 SpacemiT IME/IME2）、低比特 GEMM/GEMV/Linear/MatMul、Attention、KV Cache、数据重排、TCM/local-memory 流水和远端开发板验证。用户要求优化或评审 RISC-V/RVV kernel、prefill/decode 性能、量化计算、架构分层、远端板端 benchmark，或排查 RISC-V 专用路径性能回归时使用。
---

# MNN RISC-V CPU 性能优化

> **边界**：不要读取、修改或依赖 `schema/private/` 与 `source/internal/`。

## 按需读取

- 涉及 RVV、IME2、W4B64、TCM、异构核或 kernel 设计时，先完整读取
  [`references/rvv-ime2.md`](references/rvv-ime2.md)。
- 涉及 SSH 开发板、交叉编译、正确性回归或性能报告时，先完整读取
  [`references/remote-validation.md`](references/remote-validation.md)。
- 需要调整测试阶段或 CI 时，再使用 `skills/test-ci/SKILL.md`。
- 遇到正确性回归时，再使用 `skills/general-debug/SKILL.md`。

## 核心约束

1. **先分清三层实现**：区分通用 CPU、标准 RVV、厂商 ISA。厂商指令、TCM 和 shape
   门禁只进入厂商 target；标准 RVV 始终保留为可独立构建、可运行的 fallback。
2. **先正确再加速**：先建立标量 oracle、原路径对照和模型级 sanity，再改 pack、kernel、
   线程或存储流水。
3. **把 pack 与 kernel 视为同一 ABI**：同时核对 tile、stride、signedness、scale、
   zero point、tail、输出 layout 和函数注册；不要单独替换其中一侧。
4. **分别优化 prefill 与 decode**：prefill 优先提高数据复用和矩阵单元利用率；decode
   优先减少权重流量、dispatch、同步和后处理。不要假设同一 kernel 或线程数同时最优。
5. **让 Executor 只负责编排**：把 ISA kernel、TCM runtime 和 vendor shape 规则下沉到
   架构目录。通用文件只保留稳定的扩展入口和 fallback，不混入厂商宏或数据布局知识。
6. **用构建能力控制厂商路径**：使用单一构建选项/编译宏隔离 vendor target；不要为生产
   路径增加 `getenv` 调优开关。运行时只保留 shape、layout、资源和正确性门禁。
7. **以目标板为最终依据**：本机编译只能发现通用错误。ISA、VLEN、核拓扑、TCM 和持续
   带宽必须在目标硬件验证。
8. **保护通用平台**：验证厂商 ON、纯 RVV OFF 两个变体；不能只证明 IME2 构建可用。
9. **控制修改面**：避免无关格式化和顺手重构。先用独立子类、工厂或窄回调隔离架构差异，
   只有通用语义确实变化时才修改通用 CPU 实现。
10. **数据驱动验收**：同时记录正确性、端到端 tok/s、热点占比、有效带宽和波动范围；
    不以单次最好成绩或理论 TOPS 代替结论。

## 工作流

### 0. 锁定目标与现状

开始修改前写清以下矩阵：

| 维度 | 必须确认 |
|---|---|
| 入口 | Execution、函数表、注册入口、最终 kernel symbol |
| 数据 | dtype、量化位宽、block size、对称/非对称、scale/offset 类型 |
| shape | prefill/decode、M/N/K、tail、head/GQA、context 范围 |
| ISA | 标准 RVV、VLEN/SEW/LMUL、厂商扩展与编译参数 |
| 存储 | pack layout、cache/TCM、临时 buffer、读写字节数 |
| 并发 | 线程池、核亲和性、共享计算单元、barrier/dispatch 次数 |
| 验收 | op test、短生成、长 prompt、纯 RVV smoke、端到端 benchmark |

执行：

1. 用 `rg` 从 op/Execution 追到注册、packer 和 kernel，确认真实热路径。
2. 记录当前分支、相关工作区修改和远端状态，避免覆盖用户改动。
3. 固定模型、线程数、输入长度、量化格式和 benchmark 参数。
4. 获取至少一组未改代码的正确性与性能基线。

### 1. 判断瓶颈

先 profile，再选择方案：

| 现象 | 优先检查 |
|---|---|
| prefill 慢 | pack/动态量化遍数、M tile、权重复用、矩阵单元利用率 |
| decode 慢 | packed weight 字节数、持续带宽、dispatch/barrier、epilogue |
| kernel 快但模型不快 | 调用次数、Attention/KV、layout conversion、线程池 |
| 增加线程反而慢 | 共享矩阵单元、内存带宽、核拓扑、同步成本 |
| TCM 无收益 | 工作集、copy/compute 是否重叠、启动成本、真实 TCM 可用性 |
| 数值只在 vendor 路径错 | pack ABI、signedness、scale/zero-point 修正、tail |

估算 roofline：

```text
decode tokens/s 上限 ≈ 持续有效带宽 / 每 token 必读权重与元数据字节数
```

不要把接口峰值带宽、稀疏 TOPS 或单条指令峰值当成模型可达到的吞吐。

### 2. 建立正确性阶梯

按以下层级逐步对拍：

1. 标量数学公式。
2. 单个 quant block、tile 和 output group。
3. pack 后输入、整数 accumulator、dequant、zero-point correction、epilogue。
4. 主 kernel 与 remain/tail。
5. 原路径与新路径的算子输出。
6. 短生成和跨 prefill 门槛的长 prompt。

低比特非对称量化必须单独核对：

- activation 动态量化不等于权重对称量化；
- block scale 不等于 zero-point 修正；
- `sum(A)`、offset/residual、bias/clamp 的融合顺序；
- FP16 小计与 FP32 累加边界；
- 多线程 output chunk 的 metadata 偏移。

临时双算或诊断可以用于开发，但在最终实现中移除。

### 3. 设计最小专用路径

优先采用以下分层：

```text
通用 CPU Execution
  -> 标准 RVV Execution / 函数表
       -> vendor Execution / kernel target
```

实现时：

1. 让通用层解析通用参数、管理通用资源并提供 fallback。
2. 让 RVV 层处理 RVV 能力、通用向量 kernel 和 per-Execution scratch。
3. 让 vendor 层处理专用 layout、TCM、核拓扑、shape 门禁和厂商指令。
4. 让专用路径只有在完整写完输出后返回成功；失败时安全回退，避免重复更新 KV Cache
   或重复执行有副作用的前置逻辑。
5. 让 weight reorder 是否支持由 vendor 回调根据完整 layout 参数判断，不把子类知识泄漏
   到通用构造函数。

### 4. 实现和逐步验收

每次只引入一个可测变化：

1. pack/layout；
2. 最小 kernel；
3. prefill/decode 调度；
4. epilogue/direct output；
5. TCM 或异步搬运；
6. 更大 shape 与 tail。

每一步都运行最小正确性测试和相邻 A/B。性能没有稳定收益时回退该步骤，不依赖无效日志、
padding 或二进制偶然布局维持成绩。

### 5. 完整验证

至少覆盖：

- vendor 构建与链接；
- 标准 RVV 构建与链接；
- kernel/op 的主形状和 tail；
- vendor 路径短生成；
- 标准 RVV 短生成；
- 跨 prefill 分支阈值的长 prompt；
- 目标模型 prefill 与 decode；
- 不满足门禁时的 fallback；
- `git diff --check` 和厂商目录外的污染检查。

性能测试使用多个新进程，尽量交替运行基线与候选版本，并记录均值、范围、线程数、频率和
温度。只有 Markdown 改动时不需要重新构建，但仍检查格式、路径和敏感信息。

### 6. 交付

交付前说明：

- 修改了哪些层、为何不能复用通用路径；
- 哪些 shape/格式命中，哪些自动回退；
- 正确性和性能各运行了什么；
- 纯 RVV 是否验证；
- 仍受计算、带宽还是同步限制；
- 是否存在目标板、运行时或模型条件。

对外文档只保留可公开复现的配置和数据，不写 SSH 别名、用户名、内部路径、提交 ID 或
未公开硬件细节。
