# RVV、IME2 与低比特 kernel 参考

## 目录

1. 实现分层
2. RVV 检查表
3. IME2 编程模型
4. 非对称 W4B64
5. Prefill 与 decode
6. TCM 与共享资源
7. Attention 与 KV Cache
8. 常见错误

## 1. 实现分层

始终区分：

| 层 | 可依赖能力 | 典型职责 |
|---|---|---|
| 通用 CPU | 标量、通用线程与 Tensor layout | 参数解析、通用 buffer、fallback |
| 标准 RVV | RVV 1.0 与运行时 VLEN | 向量量化、归约、pack、通用 GEMM/Attention |
| Vendor | 专用编译 target 与运行时资源 | IME2 kernel、TCM、核拓扑、专用 layout |

保持 vendor 指令源文件使用独立编译参数。不要让普通 RVV object 携带 vendor ISA，也不要
用同名符号静默替换标准实现。通过稳定注册入口或函数表覆盖。

在 MNN 中优先从这些位置定位当前实现，不依赖固定行号：

```text
source/backend/cpu/compute/
source/backend/cpu/riscv/
source/backend/cpu/riscv/rvv/
source/backend/cpu/riscv/rvv/spacemit_ime2/
```

使用 `rg` 确认当前 symbol、CMake target、编译宏和 fallback，不假设文档中的历史名称仍有效。

## 2. RVV 检查表

实现 RVV kernel 时逐项确认：

- 运行时 `vlenb` 与目标 kernel 的最低要求；
- `SEW`、`LMUL`、有效 `VL` 和寄存器组占用；
- `vsetvl` 是否在内层循环重复；
- tail-undisturbed/tail-agnostic 与 mask 语义；
- widening/narrowing、signedness 和饱和/舍入模式；
- segment/strided load 是否真正匹配内存布局；
- unroll 后是否因 LMUL 造成寄存器溢出或 spill；
- VLEN=256 的纯 RVV 路径是否被错误喂给要求 VLEN=1024 的 kernel；
- 主循环和 tail 是否读取相同 metadata；
- intrinsic、内联汇编和编译器自动向量化是否生成预期指令。

1024-bit 向量寄存器宽度不代表每个普通 RVV 算术操作都具有 1024-bit/cycle 吞吐。分析
性能时区分架构 VLEN、执行管线宽度、load channel 和矩阵单元吞吐。

## 3. IME2 编程模型

SpacemiT IME2 复用 RVV 向量寄存器表示矩阵 tile，不引入独立 matrix register file。公开
指令类别包括：

- `smt.vmadot*`：整数矩阵乘加；
- `smt.vmadot.sp*`：结构化稀疏；
- `smt.vmadot.hp*`：整数点积与 block scale；
- `smt.vfwmadot*`：FP16/BF16 到 FP32；
- `smt.vpack/vupack/vnpack*`：矩阵布局与 nibble 重排。

先从当前编译器、指令规范和现有 kernel 核对确切助记符、tile 和 operand signedness，不凭名称
猜测语义。

IME2 基础整数矩阵指令使用同位宽输入。实现逻辑上的 INT8 activation × INT4 weight 时，
不要声称存在原生混合位宽指令；可把 INT8 activation 拆为有符号高半字节和无符号低半字节，
分别点积后合并。

两路输入的 signedness 必须和拆分方式一致。用标量公式验证：

```text
a_int8 = 16 * a_hi_signed + a_lo_unsigned
dot(a_int8, w_u4) = 16 * dot(a_hi_signed, w_u4) + dot(a_lo_unsigned, w_u4)
```

## 4. 非对称 W4B64

区分：

- 权重：每 64 个值一组的非对称 4-bit 量化；
- 激活：运行时动态对称 INT8 量化；
- 计算：整数点积、weight offset 修正、scale 和后处理。

概念公式：

```text
y_block =
    activation_scale * weight_scale
    * (dot(qA, qW) - weight_zero_point * sum(qA))
```

具体实现可能把 zero point 转成 offset/residual，并把 `sum(qA)`、scale 和 correction 写入
packed metadata。无论如何，标量 oracle 必须从实际 packed layout 读取并复现 kernel 的运算
顺序。

优化 pack 时同时核对：

- block64 是否被拆成多个硬件 K tile；
- A/B 的 row-major/column-major 要求；
- scale、offset、row sum 的精度和对齐；
- super-block stride；
- output channel group 和线程分片；
- remain/tail 的 metadata 地址。

## 5. Prefill 与 decode

### Prefill

优先尝试：

- 多行 M tile，提高权重复用；
- 合并 absmax、动态量化、A pack 和 `sum(A)`；
- strided-row 或连续 row-block 调度，减少细任务 dispatch；
- register blocking；
- direct-layout epilogue，避免中间 C buffer 和第二次转换。

先确认 activation 行数足以摊薄 pack 和 barrier。小 M 保留轻量路径。

### Decode

优先尝试：

- M1/GEMV 专用 kernel；
- 连续输出 panel 分片，形成顺序权重访问；
- persistent worker，减少每层 dispatch；
- direct output，把 scale、bias/clamp 和最终 layout 写入融合；
- 减少 packed-B 元数据与冗余读取；
- 在达到计算峰值前先测持续有效内存带宽。

不要默认使用全部核心。增加 worker 可能只增加共享矩阵单元争抢、DRAM 竞争和 barrier 成本。

## 6. TCM 与共享资源

把 TCM 当作显式管理的 scratchpad，而不是自动缓存：

1. 从运行时查询 block 数量和容量；
2. 拒绝模拟 TCM 或无效映射；
3. 在容量不足、工作集过小或 runtime 不可用时回退；
4. 明确 buffer 的软件所有权和底层 bank 的硬件共享关系；
5. 先验证 DRAM 与 TCM kernel 数值一致，再测流水；
6. 区分“复制到 TCM”与“复制和计算真正重叠”。

worker-pair 双缓冲可以让一名 worker 计算当前 tile、另一名准备下一 tile，再交换角色。它不等于
单个 worker 天然拥有两个物理 TCM bank。barrier 只同步阶段，不应交换或破坏 buffer 所有权。

当前若使用 RVV load/store 搬运，应如实描述；只有实际发出并验证异步 DMA 后才能称为 AI-DMA
流水。DMA 方案还要核对启动延迟、完成同步、对齐、容量和尾块。

## 7. Attention 与 KV Cache

不要因为文件位于 vendor target 就把所有计算称为 IME2。Attention 的 QK、online softmax、
PV 可能主要使用标准 RVV FP16/FP32 FMA、归约和转换，TCM 只提供 scratch。

设计 fused Attention 时：

- 把 mask、GQA、head dimension、sequence、KV quant 和 layout 门禁放在 vendor 子类；
- 在 KV Cache 更新后只调用一次窄 fast-path hook；
- fast path 完整写完输出才返回成功；
- 失败后回退 RVV，再回退通用 CPU；
- 避免复制完整的通用 `onExecute`；
- 按 Execution 管理 scratch，避免每层按最大 context 常驻放大。

KV Cache 优化必须分别验证 FP32/FP16、量化/非量化、连续/非连续布局和多线程更新。只在已验证
条件启用批量 pack、`memcpy` 或并行更新。

## 8. 常见错误

| 错误 | 修正 |
|---|---|
| 把动态激活量化写成对称权重 | 分别记录 W 与 A 的量化定义 |
| 把 IME2 block scale 当 zero-point 修正 | 单独验证 offset 与 `sum(A)` |
| 把 60 TOPS 稀疏峰值用于 dense W4 | 使用匹配的 dense/sparse 口径 |
| 把 LPDDR 接口峰值当持续带宽 | 用板端 microbenchmark 或模型字节数反推 |
| 只验证 vendor ON 构建 | 再编译、运行纯 RVV OFF 变体 |
| vendor 逻辑混入通用 Execution | 下沉到子类、窄回调或独立函数表 |
| TCM copy 有执行就称双缓冲 | 证明 copy/compute 时间线重叠 |
| 增加线程就假设更快 | 做线程数 sweep 并检查共享单元和带宽 |
| 修改格式掩盖功能 diff | 只保留必要功能行，提交前检查 diff |
