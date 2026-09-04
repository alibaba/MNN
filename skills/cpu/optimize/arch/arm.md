# ARM 侧路径与派发：诊断面

> **何时读**：在 ARM CPU 上做性能归因，需要先确认「我到底跑在哪条路径上」的时候。
> 这是 [`optimize/`](../SKILL.md) 分支的 **L4（Dispatch / 函数表）诊断面**。
>
> **不在本文**：
> - 怎么把 kernel 挂进表、怎么写 ARM 汇编 → [`../../kernel/arch/arm.md`](../../kernel/arch/arm.md)、
>   [`../../kernel/dispatch-and-register.md`](../../kernel/dispatch-and-register.md)
> - tile / pack / stride 契约 → [`../../kernel/pack-and-abi.md`](../../kernel/pack-and-abi.md)
> - 与 x86_64 / RISC-V 的结构差异 → [`../../SKILL.md`](../../SKILL.md)「三侧不同构对照表」（**只有那一份**，不要在本文再抄）
> - 命令与验证矩阵 → [`../../shared/build-test-and-benchmark.md`](../../shared/build-test-and-benchmark.md)
> - env 开关语义 → [`../../shared/env-registry.md`](../../shared/env-registry.md)

## 一、路径矩阵

ARM 侧的路径由**两条正交的轴**决定，不是一条链：

- **架构级别轴**（§1.1）：决定 int8 tile、以及哪些 kernel 覆盖基表。
- **精度轴**（§1.2）：决定 `bytes` / `float pack` / 走哪个 Backend 与哪张表。

两条轴**相乘**。fp16 不是「SME2 之后的第五档 ISA」，它是**任意架构级别下的另一种精度**——
把这两条轴串成一条，是 ARM 侧归因最常见的结构性错误。

### 1.1 架构级别轴：决定 int8 tile

| 架构级别 | 新增可用指令 | MNN 内部后缀 | 构建门 | 运行时门 | int8 tile (hP/UNIT, lP/SRC_UNIT, eP/DST_XUNIT) |
|---|---|---|---|---|---|
| **aarch32（Armv7 / 32 位）** | 只有 NEON | 无 | — | — | (4, 16, **2**) |
| **aarch64，低于 v8.2** | 只有 NEON | 无 | — | 基线，恒可达 | (4, 16, **4**) |
| **Armv8.2** | `sdot` | `*_ARM82` | `__aarch64__` | `supportSDot`（`MNNGetCPUInfo->dot`） | (8, 4, 12) |
| **Armv8.6** | `+ smmla`（i8mm） | `*_ARM86` | `__aarch64__` | `supportI8mm` | (8, 8, 10) |
| **Armv9（SME2 为 v9.2）** | `+ SME2` | `*_SME2` | `MNN_SME2` | `supportSME2`（另看 `smeCoreNumber`） | (32, 4, 16)；decode-max 另有 HP128 |

四条要点：

1. **这条轴是累积的，不是四套独立实现。** v8.6 机器上 sdot 同样可用，v9 机器上 sdot / smmla 都可用；
 派发也照这个顺序累积覆盖同一张基表（§2.2）。所以「我在 i8mm 机器上」**不等于**
 「这个 op 走了 i8mm kernel」，反过来「跑着一个 ARM82 后缀的函数」也不等于档位掉了。
2. **MNN 只看运行时能力位，不读架构级别声明。** 上表的级别是通常对应关系；
 sdot / i8mm / SME2 在成为对应级别的必选特性之前都是 optional 扩展，
 所以标称 v8.2 的芯片可能没有 sdot。检测有三条路径且会叠加：
 Linux/aarch64 走 HWCAP/HWCAP2（`CPURuntime.cpp`）、Darwin 走 `sysctlbyname`、
 Android 另外叠一层 MIDR 白名单启发式（组合处）。**实测能力位才是准的。**
3. **aarch32 与 aarch64 基线的差别只有 `DST_XUNIT`（2 vs 4）**，`Int8FunctionsOpt.h` 按
 `__aarch64__` 分档。这条界线**不是「有没有 dot」**——sdot 及以上整段在
 `#if defined(__aarch64__)` 里（`Int8FunctionsOpt.cpp` 中 `supportSDot` / `supportI8mm` 同处一个
 `#if defined(__aarch64__)` 块，SME2 段另起一个 `#ifdef __aarch64__`），
 arm32 即使能力位报了 `dot=true`（MIDR 白名单会这么报）也拿不到对应 kernel。
4. **低 bit kernel 另有构建门。** w4/w2/w3 的 `*_ARM82` / `*_ARM86` 变体在 `MNN_LOW_MEMORY` 里
 （`Int8FunctionsOpt.cpp` 的 `#ifdef MNN_LOW_MEMORY` 段，sdot / i8mm 分支内还各自嵌一层
 `#if defined(MNN_LOW_MEMORY)`），其 fp16 出口再套一层 `MNN_USE_ARMV82`。
 少任一个宏，架构级别对了也没有那组 kernel——这是「机器够新但没走上低 bit 快路」的第一嫌疑。

int8 tile 常量定义在 `compute/Int8FunctionsOpt.h`（`GEMM_INT8_*` / `*_ARM82` / `*_ARM86` / `*_SME2`），
取值函数在 `Int8FunctionsOpt.cpp`。这三个值与 packer、weight reorder、buffer stride 的同源契约
见 [`../../kernel/pack-and-abi.md`](../../kernel/pack-and-abi.md) §一、§二。

### 1.2 精度轴：决定 bytes / pack / 哪张表

| 精度 | 表 / Backend | bytes | float pack | int8 档从哪来 | 构建门 | 运行时门 |
|---|---|---|---|---|---|---|
| **fp32** | 基表 `gCoreFunction` | 4 | 4 | 就是 §1.1 当前那一档 | — | 默认 |
| **fp16** | **第二张表 + 第二个 Backend**：`Arm82Functions::init` / `Arm82Backend` | **2** | **8** | **继承** §1.1 当前档，只换 A packer | `MNN_USE_ARMV82` | `supportFp16arith && precision == Precision_Low`（`CPUBackend.cpp`） |
| **bf16** | 第三张表（整体拷贝基表后覆盖） | 2 | — | — | `MNN_SUPPORT_BF16` | `precision == Precision_Low_BF16` |

**fp16 为什么不算一个 ISA 档**：`Arm82Functions::init` 第一件事就是
`gInstance->int8MatmulRelatedFunctions = origin->int8MatmulRelatedFunctions`（`Arm82Functions.cpp`），
之后只按 `origin` 的能力位换 A 矩阵 packer（同文件：sdot → `_Arm82MNNPackC4ForMatMul_A<12,4>`，
i8mm → `_ArmBasicMNNPackC4ForMatMul_A_L8<10,8>`）。因此：

- fp16 的 int8 tile 与**同一台机器上的 fp32 完全相同**，§1.1 那张表不用为 fp16 再查一遍；
- fp16 的 int8 GEMM kernel（`MNNGemmInt8AddBiasScale_*_Unit_FP16`）**不在**第二张表里，
 而是 int8 子表里的另一组字段，在 sdot / i8mm / SME2 各分支中分别赋值
 （`Int8FunctionsOpt.cpp`）；
- 但 float 侧（pack=8、bytes=2、所有 `*FP16` 算子）确实是独立的第二张表，
 **fp32 正确推不出 fp16 正确**，反之亦然。验证要求见 §五。

精度轴与架构级别正交也包括 aarch32：`Arm82Functions` 的构建门是 `__ANDROID__ || __aarch64__`
（`Arm82Functions.cpp`），32 位 Android 上只要 `fp16arith` 能力位为真（MIDR 白名单，
`CPURuntime.cpp`）就能走 fp16 float 路径，而 int8 仍停在 aarch32 那一档。

## 二、诊断需要的结构常识

### 2.1 一张基表 + 三张二级表

```
MNNCoreFunctionInit()  →  gCoreFunction（基表，fp32/NEON，pack=4，bytes=4）
                            ├─ Arm82Functions::init()  → gInstance      （fp16, pack=8, bytes=2）
                            ├─ BF16Functions::init()   → gInstance      （bf16）
                            └─（x86_64 侧才有 AVX2Functions）
MNNCoreInt8FunctionInit() → gCoreFunc（int8 基表）+ 两张嵌套子表
                            ├─ int8MatmulRelatedFunctions   （当前 ISA 最高档）
                            └─ arm82MatmulRelatedFunctions  （sdot 世代的那一套）
```

`CPUBackend::onCreate`（`CPUBackend.cpp`）按顺序挑选：fp16 → bf16 →
`MNN_CPU_USE_DEFAULT_BACKEND` → （x86_64 的 AVX2）→ 基础 CPUBackend。

`mRelatedFunctions` 指向哪张嵌套子表由 `CPUBackend.cpp` 决定；
executor 侧在 `ConvInt8TiledExecutor.cpp` 把它和 `arm82MatmulRelatedFunctions`
一起**按值拷进自己的成员**。

> **对归因的含义**：executor 里看到的函数指针是 **resize 时的快照**。
> 运行中改基表不会反映到已 resize 的 executor 上；反过来，「op 慢」如果发生在 resize 之后，
> 换表类的推断都不成立。

### 2.2 累积覆盖带来的两个诊断结论

`MNNCoreInt8FunctionInit()`（`Int8FunctionsOpt.cpp` ~2650-2840）里 SDOT / I8MM / SME2 / RVV 是
**顺序 `if`，不是 `else if`**（完整时序与 snapshot 位置见
[`../../kernel/dispatch-and-register.md`](../../kernel/dispatch-and-register.md) §四）。归因时用得上的是两条：

1. **「i8mm 路径」实际是 SDOT ∪ I8MM 的叠加。** 在 i8mm 机器上 SDOT 分支也执行过，
   i8mm 没覆盖的字段保留 SDOT 的值（例如 `ConvDepthwise3x3LineInt8_ARM82`）。
   所以看到「i8mm 机器上跑着一个 ARM82 后缀的函数」不是配置错误。
2. **两处 `Int8GemmKernel` 不一样不是 bug。** `arm82MatmulRelatedFunctions` 只在 SDOT 分支里被填，
   i8mm / SME2 都不更新它——这是混合 kernel 路径刻意要的「SDOT 档那一套」。
   在 i8mm 机器上 `int8MatmulRelatedFunctions.Int8GemmKernel` 是 ARMV86 版，
   `arm82MatmulRelatedFunctions.Int8GemmKernel` 仍是 ARMV82 版。
3. **`*_DecodeMax` 字段只有 SME2 分支填**（`Int8FunctionsOpt.cpp`），下面没有回退档。
 所以「SME2 机器上 decode 变快」与「i8mm 机器上同一改动无效」可以同时成立，不是测错。

## 三、自证：我到底跑在哪条路径上

唯一的运行时开关是 `MNN_CPU_TARGET`，它按 §1.1 的架构级别逐级降档
（`CommonOptFunction.cpp`）：

| 档位 | 等效于 §1.1 的哪一级 | 放行的能力位 |
|---|---|---|
| `0` | aarch64 低于 v8.2 | 全关（**连 fp16 一起关**） |
| `1` | Armv8.2 | `fp16arith` + `sdot` |
| `2` | Armv8.6 | 再加 `i8mm` |
| `3` | Armv9.2 | 再加 `sme2`（否则 `smeCoreNumber` 一并清零） |

**这个开关同时压两条轴**：`fp16arith` 和 `sdot` 共用 `target >= 1`，
所以它不是纯粹的架构级别降档——想单独固定精度轴要用 `precision`（见 §五），不要指望用 `MNN_CPU_TARGET` 分离两者。

降档时打印：

```
MNN_CPU_TARGET=%d effective ARM features: fp16=%d, i8sdot=%d, i8mm=%d, sme2=%d
```

**`#ifdef MNN_PIPELINE_PROFILE` 包住的不只是打印，而是 `getenv` + 能力位屏蔽整段**
（`CommonOptFunction.cpp`）。这个宏没有 `option()` 声明，默认构建里不存在，
所以**默认构建下 `MNN_CPU_TARGET` 是彻底空操作——降级根本不发生**，不是「降级生效了只是没回显」。
必须 `cmake -DMNN_PIPELINE_PROFILE=ON` 重建才有降档能力；「我降档了但性能没变」的第一嫌疑就是这个。
该宏在 CPU 侧只用于这段代码（`cpu/CMakeLists.txt` 只把它加给 `MNNCPU`），
不引入计时开销，可以放心开着做路径覆盖测试。其余开关见
[`../../shared/env-registry.md`](../../shared/env-registry.md)。

能力位对了还不够，还要确认**这一次 op 真的落在你以为的 kernel 上**。可靠做法是在
`ConvInt8TiledExecutor` 选完 `mGemmKernel` 后临时打印函数指针，或在候选 kernel 里各插一次
一次性打印——**不要用「CPU 支持 i8mm」推断「这个 op 走了 i8mm」**。

## 四、ARM 侧事故台账

规则写在层文档里，本表只做「症状 → 去哪」的索引。

| 事故 | 提交 | 症状 | 规则在哪 |
|---|---|---|---|
| 线程数被烘焙进权重打包边界 | `812e1bed34` | 只在某些线程档位错（t5–t8 输出全废，t1–t4 正常）；尾部无人计算 + 越界写 | [`../runtime-and-scheduling.md`](../runtime-and-scheduling.md) §2.2 |
| 逻辑分块粒度 ≠ 物理 chunk | `0fd8efff1e` | 短 prompt 全对，kv 超过 chunk 尺寸后乱码 | [`../layout-and-memory.md`](../layout-and-memory.md) §二 |
| 缓存扩容使 peak RSS +190MB | `bb6bdcf827` | 吞吐正常，peak RSS 莫名上涨 | [`../layout-and-memory.md`](../layout-and-memory.md) §四 |
| 删掉「看起来多余的」memset | `bb6bdcf827`（同一改动） | 只在多线程 + 特定 chunk 尺寸下乱码 | [`../layout-and-memory.md`](../layout-and-memory.md) §五 |
| ThreadPool 自旋/空闲策略 | `142f294b0c`、`502dc4511b` | 全 worker 停在 `__psynch_cvwait` 死锁；或 decode worker 睡死 kv2048 -13%；或异构核 prefill 悬崖 | [`../runtime-and-scheduling.md`](../runtime-and-scheduling.md) §1.4 |

这五条的共同点：**都不崩在改动点上，都需要特定维度（线程数 / prompt 长度 / chunk 尺寸）才复现。**
所以 ARM 侧任何涉及线程数或分块的改动，验证必须跨维度，不能用一档代表全部（见 §五）。

## 五、ARM 侧的降档与精度覆盖

通用命令、argv 顺序、测试名匹配规则、结果记录格式在
[`../../shared/build-test-and-benchmark.md`](../../shared/build-test-and-benchmark.md)。
本节只给 ARM 独有的两条维度，正好就是 §一 的两条轴：

**精度轴**——fp32 与 fp16 是两张表、两套 float kernel，互相不能推断：

```bash
./run_test.out op/lowMemory/blockConv 0 1 4      # precision=1 → fp32（基表）
./run_test.out op/lowMemory/blockConv 0 2 4      # precision=2 → fp16（Arm82 表）
```

**架构级别轴**——`MNN_CPU_TARGET=0..3` 在同一台机器上跑齐 §1.1 的四级（aarch64 基线 /
v8.2 sdot / v8.6 i8mm / v9.2 SME2），**需 `-DMNN_PIPELINE_PROFILE=ON`**（见 §三）。
x86_64 侧是 0..4 且含义不同，见 [`x86_64.md`](x86_64.md)。

两条轴要**交叉**跑，不是各跑一遍：`MNN_CPU_TARGET=0` 会把 fp16 一起关掉（§三），
所以「fp16 × 基线档」这一格拿不到，能覆盖的是 target 1/2/3 各配 precision 1 和 2。

线程数维度按 §四 的台账处理：1 / 4 / 超过 P 核数各一档。

## 六、诊断面代码坐标速查

只列**归因时要读的**坐标。注册面、kernel 选择、asm 风格在
[`../../kernel/arch/arm.md`](../../kernel/arch/arm.md)。

| 主题 | 坐标 |
|---|---|
| `CoreFunctions` 定义 | `compute/CommonOptFunction.h`（能力位、嵌套 `int8MatmulRelatedFunctions` / `arm82MatmulRelatedFunctions`） |
| 基表初始化 / 能力位 | `compute/CommonOptFunction.cpp`（能力位来自 `MNNGetCPUInfo()`，`MNN_CPU_TARGET` 覆盖 + 打印） |
| int8 表与 ISA 分层 | `compute/Int8FunctionsOpt.cpp`；tile 常量 `Int8FunctionsOpt.h`；取值函数 `MNNGetGemmUnit*` |
| Backend 选择顺序 | `CPUBackend.cpp`（fp16 → bf16 → DEFAULT_BACKEND → AVX2 → 基础） |
| executor 侧表快照 | `ConvInt8TiledExecutor.cpp`；来源 `CPUBackend.cpp` |
| 线程数与工作划分 | `CPUBackend.cpp` 的 `computeThreadNumber()`、`computeDivideSizes()` |
| SME/NEON 非对称划分 | `ConvInt8TiledExecutor.cpp`；`mDivides` 声明在同名 `.hpp` |
| 权重打包 | `ConvInt8TiledExecutor.cpp` 的 `reorderWeight()`、`packWeightAndQuantInfo()` |
| ThreadPool 等待策略 | `cpu/ThreadPool.cpp`：常量与 `MNNThreadPoolRelax`、worker 循环（`ThreadPool::init()` 里的线程 lambda）、主线程等待（`ThreadPool::enqueueInternal()` 尾部的 `do-while`） |
| 并发宏 | `source/core/Concurrency.h` 的 `MNN_CONCURRENCY_BEGIN` / `MNN_CONCURRENCY_END` / `MNN_CONCURRENCY_ENQUEUE`（按 OpenMP / GCD / ThreadPool 三种后端各有一套定义） |
