# RISC-V kernel 实现参考

> **何时读**：要在 `source/backend/cpu/riscv/` 下写一条新 kernel（RVV intrinsic、内联汇编，
> 或 SpacemiT IME2 厂商矩阵扩展）、给标准 RVV 补一个已有函数的实现、或给 vendor 路径加 fast path 之前。
> **本文只回答「怎么写对、怎么被选中」**；「运行时到底走了哪条路径、为什么慢」属于诊断，
> 在 [`../../optimize/arch/riscv.md`](../../optimize/arch/riscv.md)（三条路径矩阵、构建门 / 运行时门自证、
> decode roofline 都在那里，本文不重复）。AArch64 侧对应 [`arm.md`](arm.md)，x86_64 侧对应 [`x86_64.md`](x86_64.md)。
>
> **板端交叉编译、正确性回归与性能实验纪律**在
> [`../../shared/riscv-remote-validation.md`](../../shared/riscv-remote-validation.md)——本文不写命令。
>
> **命名**：目录名 `source/backend/cpu/riscv/`（下分 `rvv/` 与 `rvv/spacemit_ime2/`），
> CMake 选项名 `MNN_USE_RVV` / `MNN_RVV_SPACEMIT_IME2` / `MNN_RVV_MARCH` / `MNN_RVV_FAST_MATH`
> 与宏名 `MNN_USE_SPACEMIT_IME2` 保持字面写法。平台术语写 RISC-V，向量扩展写 RVV。

## 〇、三层实现划分：新代码落在哪个 target

写第一行代码前先定位层次，因为**层次直接决定文件放哪、编译参数是什么**：

| 层 | 可依赖能力 | 放哪 | object lib | ISA 编译参数 |
|---|---|---|---|---|
| 通用 CPU | 标量、通用线程与 Tensor layout | `source/backend/cpu/` | `MNNCPU` | 无 |
| 标准 RVV | RVV 1.0 与**运行时** VLEN | `riscv/rvv/*.cpp` | `MNNRVV` | `-march=${MNN_RVV_BASE_MARCH}`（默认 `rv64gcv`）`-mabi=lp64d` |
| Vendor runtime | 专用宏 + 通用 ISA | `riscv/rvv/spacemit_ime2/`（Attention / Executor / 注册） | `MNNSpacemitIme2Runtime` | 同上 **+ `-DMNN_USE_SPACEMIT_IME2`** |
| Vendor kernel | 厂商矩阵指令 | `riscv/rvv/spacemit_ime2/`（GemmInt8 / GemmI8I4Local / AttentionKernels） | `MNNSpacemitIme2` | 上面 **+ `_xsmtvdotii`** + `-fno-stack-protector` |

三条硬约束（都在 `riscv/CMakeLists.txt` 里可查）：

1. **只有 `MNNSpacemitIme2` 这一个 object lib 带 `_xsmtvdotii`。**
   CMake 用 `string(REPLACE "_xsmtvdotii" "" MNN_RVV_BASE_MARCH ...)` 把厂商扩展从基线 ISA 串里剥掉，
   即使旧 build cache 的 `MNN_RVV_MARCH` 还带着合并写法。
   **不要把厂商指令写进 `rvv/` 或 runtime lib 的文件**——那不是"顺手放一起"，是构建隔离被破坏。
2. **新增 vendor 文件要自己决定进哪个 lib**：`MNN_SPACEMIT_IME2_RUNTIME_SRC` 与 `MNN_SPACEMIT_IME2_SRC`
   是两份**显式列举**的文件名列表（不是 GLOB）。写了新文件不加进去 = 静默不参与编译。
   `rvv/` 侧相反，是 `FILE(GLOB ...)`，新文件自动进 `MNNRVV`。
3. **标准 RVV 必须始终是可独立构建、可运行的 fallback。**
   `MNN_RVV_SPACEMIT_IME2=OFF` 的构建要能编过且跑对，这是 vendor 路径出问题时唯一的对照物。

## 一、注册面：两个 fast-path TU 定义**同名符号**

这是 RISC-V 侧与 ARM / x86_64 最不一样的一点，写之前必须知道：

```
rvv/MNNRvvFastPathRegistration.cpp                    ┐ 两者定义同一对符号：
rvv/spacemit_ime2/MNNSpacemitIme2FastPathRegistration.cpp ┘  MNNRvvInitializeFastPathFunctions(CoreFunctions*)
                                                             MNNRvvInitializeInt8FastPathFunctions(CoreInt8Functions*)
```

`MNN_RVV_SPACEMIT_IME2=ON` 时 CMake 把 `rvv/MNNRvvFastPathRegistration.cpp` 从 `MNNRVV` 源列表里
`REMOVE_ITEM`，由 vendor TU 提供这对入口。所以：

- **这是构建期互斥（ODR），不是运行时二选一。** 想加"第三种 vendor"不能再加一个同名 TU，
  必须先把这层改成真正的运行时派发，否则链接期重复定义。
- 调用点在 `compute/CommonOptFunction.cpp`（float 侧，`#if defined(__riscv) && defined(MNN_USE_RVV)`
  的 `supportRVV` 块内）与 `compute/Int8FunctionsOpt.cpp`（int8 侧，`#ifdef __riscv` + `#ifdef MNN_USE_RVV`
  双层构建门内）。两处都在**基表上逐字段覆盖**，RISC-V 没有第二张函数表、也没有第二个 Backend。
- 推论：**漏覆盖一个字段的后果是退回标量实现——慢但对**，和 x86_64 同类，与 ARM fp16（arm82 有独立表，
  漏字段是 `nullptr` / 崩溃）不同。所以 RISC-V 上出数值错，不要花时间核对函数表，去查 §二 的 pack/ABI
  与 §五 / §八 的门禁。三侧完整对照见 [`../../SKILL.md`](../../SKILL.md)「三侧不同构对照表」。

注册块的位置约束（**这条最容易踩**）：int8 那块在嵌套子表快照之前，所以
`core->int8MatmulRelatedFunctions.eP = 8` 的手工赋值才生效。
往这块**后面**加新 ISA 分支会被快照覆盖掉，机制见
[`../dispatch-and-register.md`](../dispatch-and-register.md) §4.1。

`CoreFunctions::extension`（`compute/CPUExtension.hpp`）上挂着 `createInt8GemmExecution` 与
`createAttentionExecution` 两个工厂。RVV 档只挂 Attention；vendor 档在 `MNN_LOW_MEMORY` 下还挂
int8 gemm Execution，它参与 `ConvInt8TiledExecutor.cpp` 的 `preferLinearPlaneSplit` 判断、
进而影响 `mSplitByOc`——改 RVV/vendor 路径时这条线索容易漏。

## 二、pack / ABI：RVV int8 档同一个 tile 的三处表述必须同改

`Int8FunctionsOpt.cpp` 的 `supportRVV` 块里同时改了三个量，三处都引用同一个宏
`GEMM_INT8_DST_XUNIT_RVV`（定义在 `Int8FunctionsOpt.h`，与 `*_ARM82` / `*_ARM86` / `*_SME2` 同构）：

| 量 | RVV 档取值 |
|---|---|
| `MNNGetGemmUnit` → `MNNGetGemmUnitRVV` | `UNIT = GEMM_INT8_UNIT`，`SRC_UNIT = GEMM_INT8_SRC_UNIT`，**`DST_XUNIT = GEMM_INT8_DST_XUNIT_RVV`** |
| `MNNPackC4Int8ForMatMul_A` | `_ArmBasicMNNPackC4ForMatMul_A<GEMM_INT8_DST_XUNIT_RVV, GEMM_INT8_SRC_UNIT, GEMM_INT8_UNIT>` |
| `int8MatmulRelatedFunctions.eP` | `GEMM_INT8_DST_XUNIT_RVV` |

三者是**同一个 tile 的三种表述**（kernel 认的 tile、packer 摆的 tile、上层排任务用的 eP），
改一个不改另外两个就是静默错数。**新增一层 ISA 档时先在 `Int8FunctionsOpt.h` 立好这一档的
`GEMM_INT8_{UNIT,SRC_UNIT,DST_XUNIT}_<档名>`，三处只准引用宏、不写字面量**——这是本类缺陷唯一的
结构性防线，靠「记得三处同改」防不住。
这五个同源量（tile / packer / weight reorder / cell stride / kernel 指针）
与 `QuanPostTreatParameters` 后处理 ABI 的完整说明在 [`../pack-and-abi.md`](../pack-and-abi.md) §一，
**不在本文重复**。

## 三、IME2 编程模型：真实助记符与 signedness

IME2 **复用 RVV 向量寄存器**表示矩阵 tile，不引入独立的 matrix register file。
仓库里实际写出的助记符（`spacemit_ime2/MNNSpacemitIme2GemmI8I4Local.cpp` 的内联汇编）是**裸助记符**：

| 助记符 | 用途 |
|---|---|
| `vmadotu` / `vmadotsu` | 整数矩阵乘加。`u` = 两路无符号，`su` = 第一路有符号、第二路无符号 |
| `vmadotu.hp` / `vmadotsu.hp` | 带 block scale 的高精度点积变体，多两个 operand（scale 向量 + 索引） |
| `vpack.vv` / `vupack.vv` | 矩阵布局重排 / nibble 展开，第三操作数是 stage 号 |

**写代码时以编译器实际接受的助记符和现有 kernel 为准，不要凭指令手册里的带前缀写法猜。**
新加指令先在目标板上用最小 `.S` 或内联汇编试编译通过再往 kernel 里放。

**基础整数矩阵指令是同位宽输入**：不存在原生 INT8×INT4 混合位宽指令。
逻辑上的「INT8 activation × INT4 weight」是把 INT8 activation 拆成有符号高半字节与无符号低半字节，
分别点积再合并——这正是 kernel 里 `vmadotsu`（高半，有符号）与 `vmadotu`（低半，无符号）成对出现的原因。

进汇编前用标量恒等式先验一遍拆分与 signedness 的一致性：

```text
a_int8 = 16 * a_hi_signed + a_lo_unsigned
dot(a_int8, w_u4) = 16 * dot(a_hi_signed, w_u4) + dot(a_lo_unsigned, w_u4)
```

**两路输入的 signedness 必须和拆分方式严格对应**，接反了在小数值上仍然对、只在特定符号组合上错，
是最难查的一类。分层比较点（unpack int → int32 累加器 → dequant fp32 → dst）见
[`../correctness-gate.md`](../correctness-gate.md) §2.1。
进汇编前必答的五个寄存器 live range 问题与 ISA 无关，RVV 同样适用，见 [`arm.md`](arm.md) §3.5。

## 四、`blkLen` 是**变体选择器**，不是块长度

`MNNSpacemitIme2GemmI8I4Local()`（`spacemit_ime2/MNNSpacemitIme2GemmI8I4Local.cpp`）的第一个参数
名叫 `blkLen`，但 256 之上的值**不是更长的 K 块，而是编码了 packed layout 与融合方式的变体号**：

| `blkLen` | 含义 |
|---|---|
| 256 | 基准 block64 W4 布局 |
| 257 | i4×i4（A 也是 4-bit）高精度变体，M≥4 |
| 258 | 融合 residual 的 M4 变体，另有 direct-C4 epilogue 入口 |
| 259 | centered 变体，仅 M1 |
| 260 | fixed A-scale 的 M4 变体 |
| 261 | 非对称 pair 的 M1 变体 |
| 其它 | 落到通用 `M1` / `M4` 路径 |

两条调用契约：

- **返回值是"本次处理的行数"，`0` 表示拒绝。** 每个变体入口都先做一遍
  `countM` / `countN % 32` / `kBlocks` / `quantBZp` / VLEN 的门禁，任一不满足就 `return 0`
  让调用方回退。加新变体必须沿用这个 **fail closed** 约定——不要写成"尽量算一部分"。
- **加新变体就是占一个新号**，同时要改 pack 侧写出的 layout 和调用侧传的号，两边同改。
  号与 layout 的对应关系没有第二处记录，只有这张分派表和 pack 代码，改之前先读全。
- **fail closed 的判据是"能力"，不是"M 能不能整除"。** 不要用 `M % tile == 0` 把整条 vendor
  fast path 挡掉——那会让所有非整数倍 shape 静默退回通用路径。正确形状是：主 kernel 吃完整 tile，
  剩余行交给 tail kernel，**两者用同一套 packed-A/B ABI**（tail 与主循环的 metadata 约定见 §六）。
  把 prefill 切成固定长度（1024 / 2048）只能作为长输入的独立实验：它会给短输入凭空加计算量，
  也替代不了 Execution 复用。

## 五、VLEN 是硬门：`vlenb != 128` 必须 fail closed

IME2 的 N32 汇编 kernel 是**按 VLENB=128（VLEN=1024）写死**的，不是"VLEN 越大越快"：

- vendor kernel 里用 `asm volatile("csrr %0, vlenb" ...)` 或 `__riscv_vlenb()` 取运行时值，
  `!= 128` 直接 `return 0`（GEMM）或跳过 fast path（Attention kernels）。
- **标量 oracle 变体不设这道门**，任何核上都能跑——这是分层比较的基础，别顺手给它也加上 VLEN 检查。
- 反向也成立：**VLEN=256 的纯 RVV 路径绝不能被喂给要求 VLEN=1024 的 kernel**。
  新写 vendor kernel 时，门禁要写在 kernel 入口而不是只写在调用方，因为调用方可能有多条。

标准 RVV kernel 相反，**必须对运行时 VLEN 通用**：用 `vsetvl` 拿到有效 `VL`，
不要把 `vlenb` 的某个具体值编进循环结构。

## 六、非对称 W4B64 的 packed metadata

先把三件事分开，混起来就一定错：

- **权重**：每 64 个值一组的非对称 4-bit 量化（block64）；
- **激活**：运行时动态**对称** INT8 量化；
- **计算**：整数点积 + weight offset 修正 + scale + 后处理。

概念公式：

```text
y_block = activation_scale * weight_scale * (dot(qA, qW) - weight_zero_point * sum(qA))
```

实现里 zero point 可能已被转成 offset/residual，`sum(qA)`、scale、correction 都写进了 packed metadata
（可参照 `MNNSpacemitIme2GemmI4I4HpRef` 这类 `*Ref` 标量变体，它显式算出
`bSuperBlockStride` / `bTileStride` / `aSubBlockStride` / `aBlockStride`，
并从 A block 尾部取 `aSum` 与 `aScaleAvg`）。

**标量 oracle 必须从实际 packed layout 读、并复现 kernel 的运算顺序**，
自己另写一份"数学上等价"的参考实现验不出 layout 错。

改 pack 时同时核对：block64 被拆成几个硬件 K tile；A/B 的 row-major/column-major 要求；
scale / offset / row sum 的精度与对齐；super-block stride；output channel group 与线程分片；
**remain/tail 读的 metadata 地址是否和主循环同一份**（主循环对、tail 错是高频缺陷）。

## 七、TCM：显式管理的 scratchpad，不是缓存

TCM runtime 是**运行期 `dlopen` 的外部库**，不是链接依赖：
`MNNSpacemitIme2GemmInt8.cpp` 里 `dlopen("libspine_tcm.so", ...)`，再 `dlsym` 出
`spine_tcm_runtime_is_available` / `_layout_info` / `_mem_get` / `_mem_free`。

写 TCM 路径的五条纪律：

1. **容量、可用性都从 runtime 查**，不模拟 TCM、不假设映射成功；
   容量不足、工作集过小、runtime 缺失都要能回退到 DRAM kernel。
2. **acquire/release 必须配对到所有退出分支**，包括门禁失败的早退。
3. **先证明 DRAM kernel 与 TCM kernel 数值逐位一致，再谈流水。**
   开发阶段可让同一 tile 两条都算一遍逐位比较；验证完删掉重复计算和诊断输出。
4. **worker-pair 双缓冲 ≠ 单个 worker 有两个 bank。** 是一名 worker 算当前 tile、
   另一名准备下一 tile 然后换角色；barrier（如 `tcmReady` 这种 acquire/release 原子标志）
   只同步阶段，不应交换或破坏 buffer 所有权。
5. **"复制到 TCM"不等于"复制和计算真正重叠"。** 当前若用 RVV load/store 搬运就如实写 RVV 搬运，
   只有实际发出并验证了异步 DMA 才能称 DMA 流水；DMA 还要另外核对启动延迟、完成同步、对齐、容量和尾块。

vendor 路径的调优开关一律用 `static constexpr bool ...Enabled()` 编译期常量
（`MNNSpacemitIme2TcmEnabled()` 之类），**不要加 `getenv`**：这些是构建能力隔离，不是用户配置。
env 机制的选择依据见 [`../../shared/env-registry.md`](../../shared/env-registry.md)。

## 八、Attention fast path：三层虚函数 override 链

```
CPUAttention::tryExecuteFastPath()                  基类，默认 false
  └─ MNNRvvAttention::tryExecuteFastPath()          标准 RVV：decode（seqLen==1）路径
       └─ MNNSpacemitIme2Attention::tryExecuteFastPath()
              自己的门禁通过 → 跑 IME2 fused kernel
              否则 return MNNRvvAttention::tryExecuteFastPath(...)   ← 必须显式委托
```

调用点只有一处：`CPUAttention.cpp` 在 **KV Cache 更新之后**调一次 `tryExecuteFastPath`，
返回 `true` 表示输出已完整写完。四条契约：

- **门禁写全，写在子类。** 现有实现把 `mUseFlashAttention`、KV 量化模式、`mBytes` / `mPack` / `hP` / `lP`、
  head 数与 GQA 分组、`mThreadNum`、`seqLen` / `kvSeqLen` / `mKvBlockSize` / `mHeadDim`、
  `paddingLength`、`lowerTriangular`、`hasSinks`、`qScale`、`directC4Output`、指针非空
  全部串在一个条件里。**条件宁可窄**——漏一条就是错数，多一条只是少一次加速。
- **fast path 要么完整写完输出再返回 `true`，要么什么都不改再返回 `false`。**
  中途失败（scratch 申请不到、并行任务未全部完成）必须回到 `false` 且不留半成品输出。
- **失败后逐层回退**：vendor → 标准 RVV → 通用 `onExecute`。
  子类不要复制通用 `onExecute` 的算法，只加门禁和自己的 kernel。
- **scratch 按 Execution 管理，不按最大 context 常驻**。现有实现对小 `mKvBlockSize` 用持久 scratch、
  超阈值改用一次性 `Tensor`，避免每层都按最大 context 放大。

vendor 档还会把 `core->MNNSoftmax` 整体换成自己的 online-softmax 实现（同签名，含
`runningMax` / `runningSum` / `updateScale`）。**换掉的是全局字段，不只是 Attention 内部用**——
替换实现必须对该签名的所有调用点成立，不能只在 Attention 的 shape 下正确。

KV Cache 侧的优化（`kvUpdateConcurrent = true` 之类）必须分别验证 FP32/FP16、量化/非量化、
连续/非连续布局、多线程更新，只在已验证条件下启用批量 pack / `memcpy` / 并行更新。

## 九、标准 RVV 逐项检查表

写或改 `rvv/` 下的 kernel 时逐项过：

- 运行时 `vlenb` 与本 kernel 的最低要求（§五）；
- `SEW`、`LMUL`、有效 `VL` 与寄存器组占用；
- `vsetvl` 是否被塞进了内层循环；
- tail-undisturbed / tail-agnostic 与 mask 语义是不是你以为的那个；
- widening / narrowing、signedness、饱和与舍入模式；
- segment / strided load 是否真的匹配内存布局（而不是"看起来对"）；
- unroll 之后 LMUL 是否导致寄存器溢出 / spill；
- **主循环和 tail 是否读同一份 metadata**；
- intrinsic、内联汇编与编译器自动向量化是否真的生成了预期指令（反汇编确认，别只看快了）。

`MNN_RVV_FAST_MATH=ON` 会给 `MNNRVV` 加 `-ffast-math`。**它会改变浮点语义**，
做数值对照时先确认两侧这个开关一致，否则差异归因会跑偏。

## 十、常见错误

| 错误 | 修正 |
|---|---|
| 厂商指令写进 `rvv/` 或 runtime lib | 只有 `MNNSpacemitIme2` lib 带 `_xsmtvdotii`，见 §〇 |
| 新增 vendor 文件没加进 CMake 列表 | vendor 侧是显式列举不是 GLOB，见 §〇 |
| 再加一个同名 fast-path 注册 TU | 构建期 ODR 互斥，加第三种前先改成运行时派发，见 §一 |
| 新 ISA 分支加在 int8 注册块之后 | 会被嵌套子表快照覆盖，见 §一 |
| 只改 `MNNGetGemmUnit` 不改 packer / `eP` | 同一个 tile 的三处表述同改，见 §二 |
| 假设有原生 INT8×INT4 混合位宽指令 | 拆高低半字节，signedness 与拆分方式对应，见 §三 |
| 加 `blkLen` 变体只改 kernel 不改 pack | 号与 layout 同改，见 §四 |
| 用 `M % tile == 0` 门禁整条 fast path | 主 kernel 吃完整 tile + tail kernel 同 ABI 收尾，见 §四 |
| kernel 门禁只写在调用方 | 写在 kernel 入口并 `return 0` fail closed，见 §四/§五 |
| 给标量 oracle 也加 VLEN 检查 | oracle 必须任何核可跑，见 §五 |
| 自己另写"数学等价"的参考实现 | oracle 必须从实际 packed layout 读并复现运算顺序，见 §六 |
| tail 读的 metadata 地址与主循环不同 | 主循环对 tail 错是高频缺陷，见 §六 |
| TCM copy 有执行就称双缓冲 / DMA 流水 | 证明 copy/compute 时间线重叠、DMA 真的发出，见 §七 |
| 给 vendor 路径加 `getenv` 调优开关 | 用 `constexpr` 编译期常量，见 §七 |
| fast path 中途失败留下半成品输出 | 要么写完返回 `true`，要么不改返回 `false`，见 §八 |
| 子类复制通用 `onExecute` | 只加门禁与自己的 kernel，失败逐层回退，见 §八 |
| 改格式掩盖功能 diff | 只保留必要功能行，提交前逐行看 diff |
