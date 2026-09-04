# 派发与注册：让你的 kernel 真正被选中

> 本文回答的是**「怎么把 kernel 挂进去」**。
> 「运行时到底挑了哪条路 / 为什么降级了 / 能力位为什么是 0」属于诊断问题，见 [`cpu/optimize/diagnose-and-route.md`](../optimize/diagnose-and-route.md)。
>
> 本文的坐标一律写「文件名 + 函数名」，不写行号——函数名在重构中比行号稳定得多。

写完 kernel 后最常见的两种失败**都不报错**：

1. kernel 根本没被调用，你测到的是旧实现的性能（看起来「优化无效」）。
2. kernel 被调用了，但配套的 pack / tile 没跟着换，结果是**能跑、不崩、单测过、只是模型质量变差**。

第 2 类由 [`pack-and-abi.md`](pack-and-abi.md) 负责。本文负责第 1 类，以及注册动作本身的原子性。

---

## 一、CPU 后端的表结构：一句话地图

CPU 后端把「函数指针」组织成**三层**，每层的替换时机完全不同：

| 层 | 实体 | 何时确定 | 谁替换 |
|----|------|----------|--------|
| L1 基表 | 全局唯一的 `CoreFunctions` / `CoreInt8Functions`，由 `MNNGetCoreFunctions()` / `MNNGetInt8CoreFunctions()` 取得 | 进程内**一次性**初始化 | `MNNCoreFunctionInit()`（`CommonOptFunction.cpp`）、`MNNCoreInt8FunctionInit()`（`Int8FunctionsOpt.cpp`） |
| L2 二级表 | 另一个完整的 `CoreFunctions` 实例：`AVX2Functions`、`BF16Functions`、`Arm82Functions` | 同样是进程内一次性，但**先拷贝基表再打补丁** | 各自的 `init()` |
| L3 嵌套子表 | `CoreFunctions::int8MatmulRelatedFunctions` / `arm82MatmulRelatedFunctions`（`CommonOptFunction.h`），类型 `MatmulRelatedFunctions` | 在 L1/L2 init **末尾快照** | 见 §三 |

Executor 拿到的不是基表，而是「当前 Backend 选中的那张表」：

```cpp
// CPUBackend.cpp（默认构造）
mRelatedFunctions = &core->int8MatmulRelatedFunctions;
// CPUBackend.cpp（Arm82Backend 分支）
res->mRelatedFunctions = &(res->functions()->int8MatmulRelatedFunctions);
```

**所以：改了基表不等于改了 Executor 实际用的表。** 三层里任何一层漏改，行为就分叉。

---

## 二、构建门 vs 运行门：两个独立维度

一条 ISA 路径能生效，必须**同时**过两道门。混淆这两者是「本机没效果」的头号原因。

| | 构建门（编译期） | 运行门（运行期） |
|---|---|---|
| 形态 | `#ifdef` / CMake `option()` / `-march` `-m` 标志 | `if (core->supportXxx)` / `cpuFlags` 探测 |
| 例子 | `__aarch64__`、`MNN_SME2`、`MNN_USE_ARMV82`、`MNN_LOW_MEMORY`、`MNN_USE_SSE`、`MNN_USE_AVX`、`MNN_AVX2`、`MNN_AVX512`、`MNN_AVX512_VNNI` | `supportSDot`、`supportI8mm`、`supportSME2`、`supportFp16arith`、`supportRVV`（`CommonOptFunction.h`） |
| 失败表现 | 代码整段不参与编译，符号不存在 | 代码编进去了但分支不进，指针停在上一层 |
| 怎么自证 | `nm` / `strings` 查符号；看 CMake 输出的编译命令 | 打印能力位；或用 `MNN_CPU_TARGET` 强制降级做 A/B |

两个高频陷阱：

- **`MNN_LOW_MEMORY` 是构建门，不是运行门。** `Int8FunctionsOpt.cpp` 里 W4/W2/W3 的 kernel 赋值大量包在 `#if defined(MNN_LOW_MEMORY)` 内。不开这个宏，低 bit kernel 指针根本没被赋值，模型跑起来会走别的路径或直接 nullptr。低 bit 相关任何实验，先确认 `-DMNN_LOW_MEMORY=ON`。
- **`MNN_USE_ARMV82` 与 `supportFp16arith` 是两件事。** 前者决定 fp16 kernel 是否编译，后者决定运行时是否选 `Arm82Backend`。再加上 `precision == Precision_Low` 这个**用户请求**（`CPUBackend.cpp`），三者缺一就没有 fp16 路径。

---

## 三、二级表构造：安全写法与唯一的例外

### 3.1 安全模式：整体拷贝后打补丁

`AVX2Functions` 和 `BF16Functions` 都是这个写法：

```cpp
// x86_x64/AVX2Functions.cpp
bool AVX2Functions::init(int cpuFlags) {
    gAVX2CoreFunctions = new CoreFunctions;
    auto coreFunction = gAVX2CoreFunctions;
    gAVX2CoreInt8Functions = new CoreInt8Functions;
    // Init default functions —— 关键：先整体拷贝基表
    *coreFunction = *MNNGetCoreFunctions();
    *gAVX2CoreInt8Functions = *MNNGetInt8CoreFunctions();
    _AVX_MNNInt8FunctionInit(gAVX2CoreInt8Functions);
    // Init AVX2 —— 再逐项覆盖需要改的
    coreFunction->MNNGetMatMulPackMode = _MNNGetMatMulPackMode;
    geP = 24; glP = 1;
    ...
```

```cpp
// cpu/bf16/BF16Functions.cpp
gInstance = new CoreFunctions;
*gInstance = *MNNGetCoreFunctions();
```

这个写法的性质：**基表新增字段自动继承**。你往 `CoreFunctions` 加一个新字段并在基表里赋值，AVX2 / BF16 侧无需改动就能拿到，最坏情况是「用了通用实现，慢但对」。

### 3.2 唯一例外：`Arm82Functions` 是逐字段赋值

```cpp
// source/backend/arm82/Arm82Functions.cpp
bool Arm82Functions::init() {
    auto origin = MNNGetCoreFunctions();
#define FUNC_PTR_ASSIGN(dst, src) dst = (decltype(dst))(src)
    gInstance = new CoreFunctions;              // ← 没有 *gInstance = *origin;
    gArm82CoreInt8Functions = new CoreInt8Functions;
    *gArm82CoreInt8Functions = *MNNGetInt8CoreFunctions();   // int8 表倒是整体拷贝
    gInstance->int8MatmulRelatedFunctions = origin->int8MatmulRelatedFunctions;  // 嵌套子表也整体拷贝
    ...
```

后面用约 120 处 `FUNC_PTR_ASSIGN` **逐字段**填 `gInstance`。

> 注意路径：是 **`source/backend/arm82/Arm82Functions.cpp`**，与 `source/backend/cpu/` 平级，**不在 cpu 目录下**。老文档写成 `arm82/Arm82Functions.cpp` 容易误导。

为什么这是个坑，取决于 `new CoreFunctions` 的语义：

- `new CoreFunctions` 是**默认初始化**，不是零初始化。
- 有 NSDMI（声明处 `= nullptr` / `= false` / `= 0`）的成员会拿到默认值；**没有的成员是不确定值**。

对照 `CommonOptFunction.h`：`supportFp16arith = false`、`MNNAbsMax ... = nullptr` 这类有默认值；而 `MNNGetMatMulPackMode`、`MNNPackC4ForMatMul_A`、`MNNPackedMatMul` 这批**没有**。

于是：

| 情况 | x86_64（AVX2/BF16） | ARM（arm82） |
|------|--------------------|-------------|
| 基表加了新字段，二级表没同步 | 继承基表值 → 走通用实现，**慢但对** | 若字段有 NSDMI → `nullptr`；若没有 → **栈/堆垃圾值** |
| 症状 | benchmark 没提升 | 崩溃，或更糟：随机数值错误、`-O` 级别不同表现不同 |

**铁律：往 `CoreFunctions` 加字段，必须在声明处写 `= nullptr`（或合适默认值），并同步检查 `Arm82Functions::init()` 是否需要赋值。** 声明处的默认值是唯一能同时保护三张表的手段。

### 3.3 改动 `CoreFunctions` 字段签名的连带面

改一个已有函数指针的签名（加参数、改类型），要同时扫：

1. 基表赋值处（`CommonOptFunction.cpp` / `Int8FunctionsOpt.cpp`）。
2. 三张二级表（`AVX2Functions.cpp`、`BF16Functions.cpp`、`Arm82Functions.cpp`）。`FUNC_PTR_ASSIGN` 里有 `decltype` 强转——**它会把签名不匹配静默转过去，编译不报错**，这是 arm82 侧最危险的地方。
3. 嵌套子表 `MatmulRelatedFunctions`（`CommonOptFunction.h`）里的同名字段。
4. 所有 asm 实现的实参约定（改参数个数就是改 ABI，见 [`arch/arm.md`](arch/arm.md)）。

第 2 条值得单独强调：普通赋值签名不匹配会编译失败，`FUNC_PTR_ASSIGN` 的 `decltype` 强转把这道保护拆掉了。arm82 侧改签名后**必须人工核对每一处**，不能依赖编译器。

---

## 四、嵌套子表的快照时序：新 ISA 必须插在正确位置

`MatmulRelatedFunctions` 不是引用，是**值拷贝的快照**。init 函数末尾把 `gCoreFunc` 的指针「拓印」进去。所以**你的赋值语句相对快照点的位置，决定它是否生效**。

### 4.1 `Int8FunctionsOpt.cpp` 的时序（`MNNCoreInt8FunctionInit()` 内，自上而下）

| 顺序 | 内容 |
|----|------|
| 1 | 基线 `int8MatmulRelatedFunctions.eP = GEMM_INT8_DST_XUNIT` |
| 2 | `if (core->supportSDot) { ... }`，内含 `eP = ..._ARM82`、**`arm82MatmulRelatedFunctions` 快照** |
| 3 | `if (core->supportI8mm) { ... }`，内含 `eP = ..._ARM86` |
| 4 | `#endif // __aarch64__` |
| **5** | **快照 #1** → `int8MatmulRelatedFunctions` 拓印 12 个字段 |
| 6 | `if (core->supportSME2) { ... }`，内含 `eP = ..._SME2` |
| 7 | `if (core->supportRVV) { ... }`，内含 `eP = 8` |
| **8** | **快照 #2** → 同样 12 个字段，**覆盖快照 #1** |

两条可操作结论：

- **新增 ISA 分支必须写在快照 #2 之前。** 写在之后，`gCoreFunc` 是对的，但 Executor 通过 `mRelatedFunctions` 拿到的是旧指针——kernel 存在、被注册、就是不被调用。
- **`eP` 必须每个分支手工赋值。** 快照块只拷函数指针，`eP` 不是 `gCoreFunc` 的成员（它是 `MatmulRelatedFunctions` 独有的，声明在 `CommonOptFunction.h`，且是该结构体里唯一没有默认值的成员）。忘了赋值 → 沿用上一层的 `eP` → tile 与 kernel 不匹配 → 静默算错。上表第 6、7 步的 SME2 / RVV 分支就是各自补这一句。

### 4.2 快照 #1 看着冗余，但不能删

两次快照拷的是同一批字段，#2 完全覆盖 #1。看起来 #1 是死代码——**但插在两者之间的 SME2 / RVV 分支正是依赖 #2 才生效的**。同构的陷阱在 arm82 侧更明显：

```cpp
// Arm82Functions.cpp 第一次快照（看似冗余）
{ gInstance->int8MatmulRelatedFunctions.MNNPackC4Int8ForMatMul_A = gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A; ... }

// 紧接着的 SME2 分支，改了 packer
if (origin->supportSME2) {
    gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _Arm82MNNPackC4ForMatMul_A<16, 4>;
    ...
}

// 第二次快照 —— 删掉它，SME2 的 <16,4> packer 就永远进不了子表
gInstance->int8MatmulRelatedFunctions.MNNPackC4Int8ForMatMul_A = gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A;
```

**别把「重复的快照块」当冗余清理掉。** 判断标准是「两次快照之间有没有分支改过被拷的字段」，而不是「代码看起来一样」。`CommonOptFunction.cpp` 的浮点侧快照同理。

### 4.3 `arm82MatmulRelatedFunctions` 是刻意冻结在 SDOT 档的

它只在 `supportSDot` 分支内快照一次，**之后 i8mm / SME2 都不再更新它**。这不是 bug：混合 kernel 路径（`ConvInt8TiledExecutor` 的 `mArm82Functions`，见 `ConvInt8TiledExecutor.cpp`）需要的就是「SDOT 档 (12,4,8) 的那一套」，用来和主档（SME2 128 档）配对做 OC 拆分。

所以：**给 `arm82MatmulRelatedFunctions` 加字段，要在 `supportSDot` 那个分支块里赋值，而不是在末尾快照里。** 反过来，如果你在 i8mm 分支里改了某个指针并期待混合路径也变——不会变。

---

## 五、注册一条新 ISA 路径的完整清单

按顺序做，每步都有独立的可验证结果。

| # | 动作 | 验证手段 |
|---|------|----------|
| 1 | CMake 里加 `option()` + 编译标志 + 源文件列表 | 编译命令里能看到目标 `-march`/`-m` 标志 |
| 2 | 能力位：在 CPU 特性探测处新增 `supportXxx`，并在 `CoreFunctions` 声明处给 `= false` | 打印能力位，目标机为 true、非目标机为 false |
| 3 | kernel 实现 + `MNNGetGemmUnitXxx` getter | 单独 unit test 直调 kernel，对齐 C++ oracle |
| 4 | 在 `MNNCoreInt8FunctionInit()` 里加 `if (core->supportXxx) { ... }`，**位置在快照 #2 之前** | 见步 6 |
| 5 | 该分支内同时改：`MNNGetGemmUnit`、`MNNPackC4Int8ForMatMul_A`、各 `Int8GemmKernel*`、`MNNSumByAxisLForMatmul_A`、**`eP`** | 逐项对照 [`pack-and-abi.md`](pack-and-abi.md) §四 的七项同改清单 |
| 6 | 确认 Executor 真的拿到了新指针 | 临时在 kernel 入口打一行日志 / 下断点；或对比新旧 tile 值 |
| 7 | 二级表：若 fp16 路径也要走新 ISA，同步 `Arm82Functions::init()`（逐字段！） | fp16 与 fp32 分别跑同一组用例 |
| 8 | 回退可达性：目标机 + 非目标机各跑一遍 | 非目标机结果必须与改动前**逐位一致** |
| 9 | 混合 kernel 影响面：新 ISA 是否会被误认成 SME2 | 查 `DST_XUNIT` 是否撞 16（见 `pack-and-abi.md` §2.2） |

第 8 步不能省。**「在目标机上更快了」和「在非目标机上没变」是两个独立结论**，只验证前者是最常见的交付缺陷。

---

## 六、注册面的命名与结构陷阱

- 注册用的 getter 名（`MNNGetGemmUnitSdot` / `..._I8mm` / `..._Sme2_HP32` / `..._RVV`）与宏名（`GEMM_INT8_*_ARM82` / `_ARM86` / `_SME2`）是**两套独立命名**，靠 getter 名 grep 不到宏、反之亦然，找齐一档的所有落点要两个名字都搜。数值本身已由宏统一，见 `pack-and-abi.md` §2.1。
- 「有 getter 就有路径」的推断不成立：确认一条路径是否活着，要 grep **赋值点**（`gCoreFunc->MNNGetGemmUnit = ...`）而不是定义点。
- x86_64 侧的第二层不叫 `supportXxx` 而是 `AVX2Backend::isValid()`（`CPUBackend.cpp`）——**能力探测和 Backend 选择合并了**。而且 `MNN_CPU_USE_DEFAULT_BACKEND` 分支在 `isValid()` 之前 `break`，所以设了这个 flag 就永远拿不到 AVX2 路径。**三侧结构不同构，别互相套**——完整差异见 [`cpu/SKILL.md`](../SKILL.md)「三侧不同构对照表」。

---

## 七、相关文档

- 配套的 pack / tile / ABI 契约：[`pack-and-abi.md`](pack-and-abi.md)
- 什么时候才该下沉到 asm（四级阶梯与退出条件）：[`SKILL.md`](SKILL.md) 铁律 1
- 交付前的跨 ISA 验证闸门：[`correctness-gate.md`](correctness-gate.md)
- 「运行时到底选了哪条路」的诊断：[`cpu/optimize/diagnose-and-route.md`](../optimize/diagnose-and-route.md)
- ARM 侧路径全景与降级自证（诊断面）：[`cpu/optimize/arch/arm.md`](../optimize/arch/arm.md)
- x86_64 侧路径全景与降级自证（诊断面）：[`cpu/optimize/arch/x86_64.md`](../optimize/arch/x86_64.md)
