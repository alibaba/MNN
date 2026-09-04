# pack 与 kernel 的 ABI 契约

> **何时读**：改 pack layout、tile 参数（UNIT / SRC_UNIT / DST_XUNIT）、cell stride、weight reorder、
> 低 bit metadata 步进，或新写一个需要打包输入的 kernel 之前。
> **本文的失效模式最特殊**：契约破了通常**不崩、不报错、单测也能过**，只是模型输出质量变差。
> 所以这里的每一条都是"改动前必须逐项核对"，不是"出问题再回来查"。

## 一、契约是什么：五个必须同源的量

一个 packed matmul 由五个量共同定义，它们**必须来自同一个源**，任何一个单独改都会破坏契约：

| 量 | 谁定义 | 谁消费 |
|---|---|---|
| **tile**（UNIT / SRC_UNIT / DST_XUNIT） | `MNNGetGemmUnit` 的某个变体 | executor 算 buffer 尺寸、packer 模板参数、kernel 内部循环 |
| **packer**（A 矩阵重排） | `MNNPackC4Int8ForMatMul_A` / `MNNPackC4ForMatMul_A` | kernel 按这个 layout 读 src |
| **weight reorder**（B 矩阵重排） | `reorderWeight()` / `MNNReorderWeightInt4` / online reorder | kernel 按这个 layout 读 weight |
| **cell stride**（每个 packed cell 的字节数） | executor 的指针算术 | 多线程 OC 分片的 `weightPtr` 偏移 |
| **kernel 指针** | 派发表注册 | 实际执行 |

**改任何一个，就要把五个一起过一遍。** 这是本 skill 的铁律 4，也是本仓库最难发现的一类 bug：
把 SME2 的 packer 喂给 i8mm/sdot kernel，形状仍然"合法"，只是算错。

> **两个数字不要混**：这里的**五个量**是**概念**——必须彼此同源的对象；
> §四 的**七处**是**动作**——要落笔改的代码位置（五个量里的 tile 和 kernel 指针各展开成多处）。
> 全树引用这份契约时统一说「五个同源量」（概念）或「§四 的七处同改」（动作），
> 不要再出现「三件套」这类另起的计数。这两个定义全树只在本文，其他文档一律引用。

现有 tile 取值（int8）：

| 档位 | UNIT | SRC_UNIT | DST_XUNIT | 宏 | getter |
|---|---|---|---|---|---|
| 无 sdot（aarch64） | 4 | 16 | 4 | `GEMM_INT8_*`（`Int8FunctionsOpt.h`） | `MNNGetGemmUnit`（`Int8FunctionsOpt.cpp`） |
| 无 sdot（arm32） | 4 | 16 | **2** | 同上（`#else` 分支） | 同上 |
| SDOT | 8 | 4 | 12 | `*_ARM82` | `MNNGetGemmUnitSdot` |
| I8MM | 8 | 8 | 10 | `*_ARM86` | `MNNGetGemmUnitI8mm` |
| SME2 | 32 | 4 | 16 | `*_SME2` | `MNNGetGemmUnitSme2_HP32` |
| RVV | 4 | 16 | **8** | 复用 `GEMM_INT8_*` + `GEMM_INT8_DST_XUNIT_RVV` | `MNNGetGemmUnitRVV` |
| SSE / AVX2 / AVX512 | 见 x86_64 侧 | | | `avx/GemmInt8.cpp`、`avx512/GemmInt8Macro.h` | `_AVX2_MNNGetGemmUnit`（`avx/GemmInt8.cpp`）等 |

x86_64 各档的具体取值见 [`arch/x86_64.md`](arch/x86_64.md) §四。

## 二、tile 契约的四个陷阱

### 2.1 tile 的唯一数字来源是宏（x86_64 侧例外）

ARM / RISC-V 侧的 getter（`MNNGetGemmUnit` / `...Sdot` / `...I8mm` / `...Sme2_HP32` / `...RVV`，
`Int8FunctionsOpt.cpp`）与 `int8MatmulRelatedFunctions.eP` **都只引用 `Int8FunctionsOpt.h` 的分层宏**
（`GEMM_INT8_*` / `*_ARM82` / `*_ARM86` / `*_SME2` / `GEMM_INT8_DST_XUNIT_RVV`），不写字面量。
改 tile 只需改宏，getter 与 `eP` 自动跟随。

- **新增一层 ISA 时**：先在 `Int8FunctionsOpt.h` 立好该档的
  `GEMM_INT8_{UNIT,SRC_UNIT,DST_XUNIT}_<档名>`，getter / packer 模板参数 / `eP` 三处只准引用宏。
  写字面量会让这条防线失效，而且**不会有任何编译错误**。
- **预检**：改宏值时 grep 宏名，找出所有直接用宏的地方
  （例如 `ConvInt8TiledExecutor.cpp` 就直接拿 `GEMM_INT8_DST_XUNIT_SME2` 与 getter 返回值比较——
  这类比较依赖两边同源，见 §2.2）。
- **x86_64 侧不受这条保护**：`AVX2Functions.cpp` 的 `int8MatmulRelatedFunctions.eP` 仍是字面量，
  见 [`arch/x86_64.md`](arch/x86_64.md) §6.7。

### 2.2 `DST_XUNIT` 被当成 ISA 身份使用

`ConvInt8TiledExecutor.cpp`：

```cpp
mOnlineReorderWeightSme = (weightOnlineReorderOption > 0 && DST_XUNITMain == GEMM_INT8_DST_XUNIT_SME2);
```

判断"我是不是跑在 SME2 上"用的是 **tile 值相等**，不是能力位。

- **后果**：新增一层 ISA 如果 `DST_XUNIT` 恰好也等于 16，会被误认成 SME2，走上 online reorder 路径。
- **预检**：新增档位时，先 grep `GEMM_INT8_DST_XUNIT_` 找出所有"用 tile 值判身份"的地方，
  确认新取值不与已有档位碰撞；碰撞了就改判据（用能力位），不要改 tile 迁就它。

### 2.3 `MNNGetGemmUnit` 不是 UNIT 的唯一来源

SME2 decode 路径在调用 getter **之后**把 `UNITMain` 覆盖成 128
（`ConvInt8TiledExecutor.cpp`，`GEMM_INT8_UNIT_SME2_128 = 128`，`Int8FunctionsOpt.h`），
后续所有 `ROUND_UP(oc, UNITMain)`、weight 长度、`shapeMain`都按 128 走。
`CommonOptFunction.cpp` 里的 online reorder 也直接用这个宏。

- **后果**：只读 `MNNGetGemmUnit` 推断 buffer 布局，在 SME2 decode 上会算错。
- **预检**：算 weight buffer / OC 对齐时，**读 executor 里那个可能被覆盖过的局部变量**，
  不要重新调 getter。

### 2.4 `eP` 与 `DST_XUNIT` 是两条语句、一个数字来源

`MatmulRelatedFunctions::eP`（`CommonOptFunction.h`）与 getter 的 `DST_XUNIT`
在语义上是同一个东西，由**两条独立语句**赋值；ARM / RISC-V 侧两条语句都引用同一个分层宏（§2.1），
所以这一侧不会再漂。仍要注意两点：

- x86_64 侧 `eP` 是字面量，两边没有共同来源（[`arch/x86_64.md`](arch/x86_64.md) §6.7）。
- `int eP;` 是这个 struct 里**唯一没有默认初值**的成员（其余都写了 `= nullptr`），漏赋值就是读未初始化值。
- **预检**：新增一层 ISA 后，打印 `MNNGetGemmUnit` 三个值与 `int8MatmulRelatedFunctions.eP`，
  确认 `eP == DST_XUNIT`。位置要求（写在哪个 snapshot 之前）见
  [`dispatch-and-register.md`](dispatch-and-register.md) §四。

## 三、低 bit 权重：cell stride 与 metadata 步进

低 bit（w2 / w3 / w4）的错误几乎全部集中在**指针步进**上。

### 3.1 cell stride 必须是真实 packed 字节数

- **规则**：OC 分线程时 `weightPtr` 的偏移必须用**每个 packed cell 实际占用的字节数**，
  **不是** useful payload 的比例（例如不能用 "4bit ⇒ 字节数减半" 这种推导）。
  如果每个 cell 有 padding，**kernel 和 packer 都要按 padded stride 前进**。
- **症状**：`tId == 0` 正确，`tId > 0` 的 OC chunk 错；或者只在某些 OC 数量下错。
- **预检**：正确性用例必须覆盖 `tId > 0` 的 chunk 和 `mSplitByOc == true`。
  标量 oracle 也要按真实 stride 读，不能用理想化公式（见 [`correctness-gate.md`](correctness-gate.md) §2.1）。

### 3.2 block metadata 按 block 粒度分别确认

block32 / block64 / per-channel 三种 metadata 步进不同。写 kernel 时：

- 每种粒度单独确认 scale / zero point / kernelSum 的步进；
- **整除门限**要显式检查（`block % hP`、`chunk % pack`），不满足时有明确的退回路径；
- 量化路径里不要写死旧的 block 值。

### 3.3 逻辑分块 ≠ 物理 chunk

这是同一个陷阱的另一面，在 KV cache 上出过事故（长 prompt 才暴露）：把逻辑块放宽而物理 chunk 不变，
地址公式必须写成 **chunk 索引 + chunk 内行偏移 + extra stride** 三段，不能当成平坦数组。
详见 [`cpu/optimize/layout-and-memory.md`](../optimize/layout-and-memory.md) §二。

## 四、packer 与 kernel 的模板参数就是契约本身

A 矩阵 packer 的模板参数**直接是 tile**：

```cpp
// Int8FunctionsOpt.cpp 附近
_ArmBasicMNNPackC4ForMatMul_A<GEMM_INT8_DST_XUNIT_SME2, GEMM_INT8_SRC_UNIT_SME2, GEMM_INT8_UNIT_SME2>
```

- **机制**：packer 与 kernel 各自按自己那套 tile 读写。两者不匹配时，形状仍然"合法"，
  没有断言会触发。
- **预检**：改 pack mode / UNIT / SRC_UNIT / DST_XUNIT / `MNNGetGemmUnit` 时，一次改全这七处：
  1. packer 模板参数；
  2. cell stride（真实 packed 字节数）；
  3. weight reorder（`ConvInt8TiledExecutor.cpp` 的 `reorderWeight()`、`packWeightAndQuantInfo()`）；
  4. mixed / online reorder 的选择判据；
  5. kernel 注册（含 `Int8GemmKernel` / `_Fast` / `_W4` / `_W2` / `_W3` 各变体）；
  6. `MatmulRelatedFunctions::eP`；
  7. **所有 `MNNGetGemmUnit` 消费者**（见 §六）。

## 五、后处理 ABI：`QuanPostTreatParameters`

int8 kernel 的后处理参数是一个结构体（`Int8FunctionsOpt.h`），字段包括
`scale`、`biasFloat`、`maxValue`/`minValue`、`useInt8`、`roundValuePos`/`roundValueNeg`、
`srcKernelSum`、`weightKernelSum`、`fp32minmax`、`blockNum`、`bias`、
`inputScale`/`inputBias`、`accumBuffer`、`indices`。

三条硬性要求：

1. **`useInt8` 决定输出类型**（默认 1 = 输出 int8，否则输出 fp32）。
   新 kernel 必须两条都实现或明确拒绝，不能只测一条。
2. **`fp32minmax` 可能为 `nullptr`**。`nullptr` 分支和非 `nullptr` 分支是两条 postprocess 路径，
   寄存器 live range 表必须覆盖**两条**（这是 hoist 常量最容易漏的分支，见 [`arch/arm.md`](arch/arm.md) §3.5）。
3. **新增字段要在结构体声明处写默认值**。已有字段大多带默认值（`useInt8 = 1`、`blockNum = 1`、
   `bias = nullptr` 等），但不是全部——`scale`、`biasFloat`、`maxValue`、`minValue`、
   `srcKernelSum`、`weightKernelSum`、`fp32minmax` 都没有。构造点漏填就是不确定值。

fp16 与 fp32 的后处理是**不同的 kernel**（`MNNGemmInt8AddBiasScale_*_Unit_FP16` 系列 vs 非 FP16 系列，
`CommonOptFunction.h`），**互相不能推断正确性**。

## 六、`MNNGetGemmUnit` 的全部消费者

改 tile 时必须逐个检查（`source/backend/cpu/` 下，HEAD 实测）：

| 消费者 | 坐标 |
|---|---|
| Attention | `CPUAttention.cpp` |
| KV cache | `CPUKVCacheManager.cpp` |
| Conv 通用 | `CPUConvolution.cpp` |
| Conv tiled（float） | `compute/ConvolutionTiledExecutor.cpp` |
| Conv int8 tiled | `compute/ConvInt8TiledExecutor.cpp` |
| Idst conv int8 | `compute/IdstConvolutionInt8.cpp` |
| Conv int8 winograd | `compute/ConvInt8Winograd.cpp` |
| RISC-V 厂商 executor | `riscv/rvv/spacemit_ime2/MNNSpacemitIme2ConvInt8Executor.cpp` |

注意 `ConvInt8TiledExecutor.cpp` 里相邻的两处读的是**两张不同的表**
（`mRelatedFunctions` 与 `mArm82Functions`），在 i8mm 机器上返回值不同——这是设计意图，不是 bug，
原因见 [`cpu/optimize/arch/arm.md`](../optimize/arch/arm.md) §2.2 结论 2。

上表只列**读取方**。x86_64 侧另有两处**转抄方**——把基表的 `MNNGetGemmUnit` 指针拷进嵌套子表
（`x86_x64/AVX2Functions.cpp`、`x86_x64/FunctionDispatcher.cpp` 的
`int8MatmulRelatedFunctions.MNNGetGemmUnit = ...`）。它们不消费返回值，但新增一档 ISA 时
漏掉转抄，Executor 拿到的就是上一档的 getter，症状与 §四 的漏改完全一样。

## 七、改动前自查表

改 pack / tile / stride 之前，逐条打勾：

- [ ] 该档的 tile 宏已在 `Int8FunctionsOpt.h` 立好，getter / packer 模板参数 / `eP` 都只引用宏、无字面量（§2.1）
- [ ] 新 tile 值不与已有档位碰撞，不会被"用 tile 值判身份"的代码误认（§2.2）
- [ ] buffer 尺寸读的是 executor 里可能被覆盖过的局部变量，不是重新调 getter（§2.3）
- [ ] `eP == DST_XUNIT` 已运行时打印验证（§2.4）
- [ ] cell stride 用的是真实 packed 字节数，含 padding（§3.1）
- [ ] block32 / block64 / per-channel 三种 metadata 步进分别确认（§3.2）
- [ ] packer 模板参数、cell stride、weight reorder、online reorder 判据、kernel 注册、`eP`、所有消费者七处同改（§四、§六）
- [ ] `useInt8` 两条路径、`fp32minmax == nullptr` 两条分支都实现并测过（§五）
- [ ] fp16 与 fp32 分别跑过（§五）
- [ ] `tId > 0` 的 OC chunk 与 `mSplitByOc == true` 覆盖到了（§3.1）
