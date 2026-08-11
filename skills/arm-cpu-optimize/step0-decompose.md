# 步骤 0：计算拆解与路径确认

> **目标**：在写代码前确认优化入口、数据布局、已有函数可复用性和必须覆盖的正确性路径。
>
> **适用场景**：普通 ARM CPU 算子优化、低 bit GEMM/GEMV kernel、dispatch/pack review。

---

## 0.1 先确认目标路径

记录这次任务实际覆盖的路径，不要默认所有 ISA 和所有 shape 都相关。

| 项目 | 需要写清楚 |
|------|------------|
| 入口 | executor、CoreFunctions 指针、asm symbol、packer |
| 数据 | FP32/FP16/INT8/w2/w3/w4，per-channel/per-block |
| shape | E=1 decode、E>1 prefill、block32、block64、tail、OC split |
| ISA | NEON、SDOT、I8MM、SME2、fallback、runtime disable flag |
| 后处理 | bias、scale、zero point、add-dst、fp32 min/max、ReLU/ReLU6 |
| 验收 | op test、模型 prompt、roofline/speed test |

低 bit LLM kernel 尤其要明确：权重 pack layout、cell stride、metadata stride、block size、是否有 padding。

---

## 0.2 读取最少必要代码

优先读取和当前入口直接相关的文件：

- `source/backend/cpu/compute/CommonOptFunction.h`：函数指针、参数结构、pack mode。
- 当前 executor 或 packer 文件：确认调度、线程划分、metadata 布局。
- 当前 ISA 的 asm 文件：确认真实寄存器、tile、tail、postprocess。
- `source/backend/cpu/CPUAttention.cpp`：仅在需要 CoreFunctions 复用范式时参考。

不要读取或依赖 `schema/private/`、`source/internal/`。

---

## 0.3 判断是否已有函数可复用

普通算子优先复用 CoreFunctions；低 bit kernel 则先确认已有 kernel/packer 是否只需修正或特化。

| 代码模式 | 优先考虑 |
|----------|----------|
| 大规模 GEMM | `MNNPackedMatMul` / `MNNPackedMatMulRemain` |
| E=1 GEMV / decode | `MNNComputeMatMulForE_1` |
| scale + bias | `MNNScaleAndAddBias` / `MNNScaleAndAddBiasScalar` |
| softmax / norm / activation | `MNNSoftmax` / `MNNNorm` / `MNNExp` / `MNNSiLu` |
| NC4/NC8 或 MatMul pack | `MNNPackCUnit` / `MNNUnpackCUnit` / MatMul pack helpers |
| 没有函数覆盖的核心热点 | C++ oracle -> C++ SIMD 模拟 -> asm |

替换前必须确认：

- 数学语义完全一致，不只是函数名相似。
- layout、转置、stride、tail 和输出格式匹配。
- `dst == src` 时该函数是否 in-place 安全；不确定就读实现或写小测试。
- 小 shape 是否值得付出 pack/function call 开销。

---

## 0.4 建立 correctness oracle 计划

复杂 kernel 不要直接写 asm。先决定 oracle 怎么做：

| 层级 | 用途 |
|------|------|
| C++ 标量 oracle | 固定数学语义、pack 解码、metadata 读取 |
| C++ SIMD/寄存器模拟 | 固定 `sdot`/`smmla` lane 分组、unpack 顺序、tail |
| asm kernel | 只迁移已经验证过的语义 |

低 bit kernel 的 oracle 至少覆盖：

- 单 block32 和 block64。
- 单 OC group 和 `tId>0` 的 OC chunk。
- fp32 min/max 非空路径。
- multi-block 累加路径。
- i8mm 和 sdot 路径（分别在对应能力的设备或构建配置上验证）。

---

## 0.5 输出本步骤结论

用短表即可，不需要长报告。

```markdown
## 路径结论

入口：
ISA：
shape：
pack/cell stride：
后处理：

## 复用判断

| 逻辑 | 复用现有函数/保留/新增 kernel | 原因 |
|------|-------------------------------|------|

## correctness oracle

| 对比点 | 方法 | 覆盖 case |
|--------|------|-----------|
```

---

## 通过标准

- 已确认当前任务的真实入口和 ISA 路径。
- 已确认 pack/cell stride/metadata stride。
- 已判断哪些逻辑复用 CoreFunctions，哪些必须新增或修改 kernel。
- 已有 C++ 标量 oracle 或明确了如何补 oracle。
- 已列出必须跑的正确性和性能验证命令。
