# CPU 侧正确性 bug：跨层交界处的坑

> **何时读**：CPU 后端上「结果不对」而不是「慢」。尤其是**不崩、单测过、只是质量变差**，
> 或**只在某个线程档 / 某个长度 / 某条 ISA 上错**。做完性能改动后先过一遍本文的 §五。
>
> **不在本文**：框架级、跨后端通用的正确性 bug 方法论（内存别名、量化导出损坏、fp16 表示能力、
> 持久化缓存、未初始化内存、多线程非确定）在 [`general-debug`](../../general-debug/SKILL.md)——
> 那边按症状分册，先在它的入口分流表选册；
> ISA 侧的具体实例见 [`arch/arm.md`](arch/arm.md) §四（事故台账）/ [`arch/x86_64.md`](arch/x86_64.md) §四（归因陷阱）；
> 性能归因在 [`diagnose-and-route.md`](diagnose-and-route.md)。

---

## 零、这一册的边界：先决定该读哪本

本文只收**一类** bug：**根因落在 CPU 后端的层与层交界处**——某一层改了参数或粒度，
另一层还按旧约定取址 / 分片 / 选 kernel。它们的共同特征是**两侧单看都自洽**，
所以 code review 和单测都抓不到。

先做这个分流，选错本子会白读一遍：

| 现象 | 去哪 |
|------|------|
| 同一输入**每次跑结果不同** | [`general-debug/nondeterminism.md`](../../general-debug/nondeterminism.md) §9（未初始化内存）与 §10（CPU 多线程非确定）。**不要**从本文开始 |
| 所有后端（CPU 和 Metal/GPU）**一致地**错 | [`general-debug/export-and-quant.md`](../../general-debug/export-and-quant.md) §2（导出侧权重损坏）。CPU 只是复现载体，不是根因 |
| 长 prompt 才错、`precision: high`（fp32）就对 | [`general-debug/fp16-range.md`](../../general-debug/fp16-range.md) §5（fp16 表示能力） |
| 只有 CPU 错、Metal 对，且改过 `onResize` 的 buffer 分配 | [`general-debug/memory-aliasing.md`](../../general-debug/memory-aliasing.md) §1（内存别名 / 生命周期） |
| **改过线程数决策 / 分片 / pack / tile / 分块粒度 / 二级函数表**之后开始错 | **本文** |
| 只在**部分线程档**、**部分长度**、**部分 ISA**、**部分 precision** 上错 | **本文** §一 |

> [`general-debug/nondeterminism.md`](../../general-debug/nondeterminism.md) 的 §9.4 和 §10.4 已经是两个 CPU 案例（`DenseConvInt8TiledExecutor` 拷贝构造漏拷
> `mMixedKernel`；t≥4 贪心逐 run 分叉）。本文不重复它们——那两类的判别特征是「同二进制多次跑不一样」，
> 与本文「稳定地错在某个维度上」正好互补。

---

## 一、症状 → 优先怀疑（主路由表）

**判别维度比症状本身信息量大。** 先问「在哪一档对、在哪一档错」，再去看代码。

本表是**事后定位**这一面：已经错了，从「哪一档错」倒推到哪一层。
**交付前**的对应面是 [`cpu/kernel/correctness-gate.md`](../kernel/correctness-gate.md) §三
（症状 → 哪条轴没覆盖 → 怎么把它跑出来）。同一症状两边各有一行，但机制与预检只写在本文。

| 症状 | 交界 | 优先怀疑 | 章节 |
|------|------|---------|------|
| t1–t4 对，t5+ 全废；或尾部输出没人算、越界写 | L1↔L2/L3 | 线程数被烘焙进打包边界或分片数组 | [§2.1](#21-c1构造期烘焙-vs-运行期变量) |
| 短输入对，长度超过某个值之后开始乱 | L2↔L3 | 逻辑分块粒度 ≠ 物理 chunk，地址公式没同步 | [§2.2](#22-c2逻辑粒度--物理粒度) |
| **不崩、单测全过，只是模型输出质量变差** | L3↔L4 | packer / kernel / cell stride 不配套 | [§2.3](#23-c3五个同源量不配套) |
| fp32 对、fp16（`precision: low`）崩或全零；行为随 `-O` 等级抖动 | L4↔L5 | 二级 `CoreFunctions` 表漏字段 | [§2.4](#24-c4二级函数表漏字段) |
| 删掉一句"看起来多余的" memset 后，只在多线程 + 某个 chunk 尺寸下错 | L2↔L3 | 隐式清零契约无人声明 | [§2.5](#25-c5隐式清零与扩容耦合) |
| 一条 ISA 对，另一条错；或 fp16 对 fp32 错 | L4↔L5 | 用一格代表全部的覆盖率假设 | [§2.6](#26-c6覆盖率假设ISA--precision--线程--长度) |
| prefill 对、decode 错（或反之） | L2 | decode 是 `E == 1`，走的量化路径和 tile 都不同 | [§2.6](#26-c6覆盖率假设ISA--precision--线程--长度) |
| 单测过但 LLM 输出退化 | — | op 单测的形状分布系统性漏掉 decode，见 [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) §五 |

---

## 二、六类跨层不一致

每类给三件事：**机制**（为什么两侧单看都自洽）、**证据**（真实提交）、**预检**（改之前查什么）。

### 2.1 C1：构造期烘焙 vs 运行期变量

- **机制**：一个量在**构造期**按某个参数算好并持久化，该参数在**resize 期**又被改了，
  但持久化的结果不重算。最典型的参数是线程数——它会被 `computeThreadNumber()`
  （`CPUBackend.cpp`）cap 掉：

  ```cpp
  int CPUBackend::computeThreadNumber(int workItems) const {
      int perfCores = mCoreFunctions->perfCoreNumber;
      if (workItems > 1 && perfCores > 0 && mThreadNumber > perfCores) {
          return perfCores;      // ← 返回值 < mThreadNumber
      }
      return mThreadNumber;
  }
  ```

  于是同一次 forward 里存在**两个不同的"线程数"**：cap 后的实际并发度，和 `mThreadNumber`。
  `computeDivideSizes(size, dst, avgDiv, threads)`（`CPUBackend.cpp`）如果拿到的是后者，
  而调用方按前者开数组，就同时得到**越界写**和**尾部无人计算**两个后果。
- **证据**：`812e1bed34` *[CPU:Bugfix] Fix weight-pack boundary and work division under capped threads*。
  权重打包边界在构造时按 `threadNumber()` 烘焙（`ConvInt8TiledExecutor.cpp` 的 `reorderWeight()`、
  `packWeightAndQuantInfo()`），线程数被 cap 后重算却不重打包 → SME 线程读到 NEON 布局。
- **预检**：列出所有"以线程数为参数、结果被持久化或跨阶段复用"的量，逐个确认它与**实际并发度同源**：
  打包边界（`mOcMain` / `ocMainThreads`）、`mDivides`、按线程数开的数组与 scratch buffer。
  同一个判据要在两个地方都用 cap 后的值，不能一处用 cap 后、一处用 `mThreadNumber`。
- **测哪一档**：cap 只在 `mThreadNumber > perfCores` 时生效，所以**必须测一档超过 P 核数的线程数**。
  只测 t1/t4 这类 bug 永远不出现。另外注意 `perfCoreNumber` 目前**只在 Apple 平台被赋值**
  （`CPURuntime.cpp`），Android/Linux 上 cap 根本不发生——**同一份代码在 Mac 上错、
  在手机上对，不代表手机上是安全的**，只是那台机器没进这条分支。
- 详见 [`runtime-and-scheduling.md`](runtime-and-scheduling.md) §1.2 / §2.2；ARM 侧实例 [`arch/arm.md`](arch/arm.md) §四。

### 2.2 C2：逻辑粒度 ≠ 物理粒度

- **机制**：分块有两个独立的数——**逻辑块**（一次算多少行）与**物理 chunk**（缓存实际按多少行分段存）。
  地址公式必须跟**物理** chunk 走。只放宽逻辑块、不改地址公式，越过第一个 chunk 边界之后就读错行；
  而这在短输入下根本不会发生。
- **证据**：`0fd8efff1e` *[CPU:Perf] Widen single-thread decode flash kv block to 256 rows* —— 逻辑块放宽到
  256、物理布局仍是 64 行 chunk，`kv > 64` 之后读错行，**靠长 prompt 金丝雀才捕获**。
  反向教训：把物理 chunk 也放大到 256、逻辑块保持 64，在 t4 掉约 4%（物理行距变 4 倍，L2/TLB 压力）。
  两个数得分开调。
- **现状参考**：`compute/CommonOptFunction.h` `MNN_FLASH_ATTENTION_BLOCK_SIZE 64`、
  `MNN_FLASH_ATTENTION_BLOCK_DECODE 2048`（后者的声明注释里带实测数字）；
  物理 chunk 的选择在 `CPUKVCacheManager.hpp flashAttentionChunkKv()`——注意它的启用条件
  **很窄**（`threadNumber() == 1` 且 V 未量化），三个条件里任何一个不满足就退回 64。
  注释里明写了原因：多线程下宽 chunk 反而回退，而 **V-int8 的 PV 调用点写死了
  `MNN_FLASH_ATTENTION_BLOCK_SIZE`**——这就是"另一处写死了旧值"的活标本。
- **预检**（改任何分块粒度前逐条过）：
  1. 物理 chunk 尺寸是否同步；
  2. 地址公式（chunk 索引 + chunk 内偏移 + `bExtraStride`）是否按物理 chunk 重算；
  3. 整除门限（`block % hP`、`chunk % mPack`）是否仍成立；
  4. 量化路径 / 另一条 precision 路径里是否**写死了旧常量**（grep 那个宏名，看还有几个调用点）；
  5. 是否与 threadNumber 联动（若联动，回到 §2.1）。
- **测哪一档**：正确性用例**必须覆盖跨过 chunk 边界的长度**。只测 kv < chunk 等于没测。
- 详见 [`layout-and-memory.md`](layout-and-memory.md) §二；ARM 侧实例 [`arch/arm.md`](arch/arm.md) §四。

### 2.3 C3：五个同源量不配套

- **机制**：这是**最难发现的一类**，因为形状仍然"合法"。packer 的 tile 是**模板参数**，
  例如 `Int8FunctionsOpt.cpp` 的
  `_ArmBasicMNNPackC4ForMatMul_A<GEMM_INT8_DST_XUNIT_SME2, GEMM_INT8_SRC_UNIT_SME2, GEMM_INT8_UNIT_SME2>`。
  把 SME2 的 packer 喂给 i8mm/sdot kernel，不会崩、不会报维度错，只是数据摆错位置。
- **症状特征**：**能跑、不崩、op 单测过，只有模型输出质量变差**。所以这类 bug 的唯一可靠门禁是
  **模型级 sanity**（固定 prompt 的 greedy 输出对拍），不是单测。
- **另一个常见变体**：低 bit 的 cell stride 用了"有效载荷比例"而不是**真实 packed cell 字节数**。
  w4 每字节两个权重，看似 stride 折半，但 packed cell 里还有量化参数与对齐填充，
  真实字节数不等于 `useful_bits / 8`。
- **预检**：改 pack mode / `UNIT` / `SRC_UNIT` / `DST_XUNIT` / `MNNGetGemmUnit` 时**一次改全**——
 改动点清单在 [`cpu/kernel/pack-and-abi.md`](../kernel/pack-and-abi.md) §四（七处同改），
 必须同源的五个量在同一份文档 §一。本文不重列，那边是唯一源。
 两个最容易漏的：所有 `MNNGetGemmUnit` 的消费者（`CPUAttention.cpp`、`CPUKVCacheManager.cpp`、
 `ConvolutionTiledExecutor`、`IdstConvolutionInt8`），以及新增 ISA 层时**手动设**
 `int8MatmulRelatedFunctions.eP`——它不从 `MNNGetGemmUnit` 推导。
- 详见 [`cpu/kernel/pack-and-abi.md`](../kernel/pack-and-abi.md) §四（packer 与 kernel 的模板参数契约）；
  新增 ISA 层时 `eP` 的手动赋值见 [`cpu/kernel/dispatch-and-register.md`](../kernel/dispatch-and-register.md) §四。

### 2.4 C4：二级函数表漏字段

- **机制**：`new CoreFunctions` 是 **default-initialization，不是零初始化**。
  只有带 NSDMI（`= nullptr` / `= false`）的成员才确定为空；其余是**不确定值**。
  三张 float 二级表里，x86_64 的 `AVX2Functions.cpp` 和 bf16 的 `BF16Functions.cpp`
  都在 `new` 之后立刻整体拷贝基表（安全）；**只有 arm82 的 float 表是逐字段赋值**
  （`Arm82Functions.cpp` 之后约 120 条 `FUNC_PTR_ASSIGN`）。
- **两种故障形态**：漏赋值的字段若带 NSDMI → 确定性 `nullptr`（要么空指针崩，要么被 `if (ptr)`
  悄悄跳过、静默走回退路径）；若不带 NSDMI → 跳野地址，**行为随构建与优化等级抖动**。
  后者最容易被误判成逻辑 bug 或"编译器问题"。
- **预检**：改 `CoreFunctions` 结构后，逐一核对 arm82 / bf16 / x86_64 / riscv 四处二级表构造。
  新增字段时**在声明处就写 `= nullptr`**——代价为零，把"野指针"降级为"确定的空指针"。
  验证手段是**运行时打印 `backend->functions()->新字段` 的实际值**，
  不要只确认"我在 init 里写了赋值语句"。
- **相关的同类错误**：executor 的**拷贝构造函数漏拷成员**（`general-debug/nondeterminism.md` §9.4 的
  `mMixedKernel`）机制上属于同一族——"新增字段没同步到所有构造路径"。改动面清单要把
  拷贝 ctor 一起算进去。
- 详见 [`cpu/kernel/dispatch-and-register.md`](../kernel/dispatch-and-register.md) §3.2（arm82 是唯一逐字段赋值的二级表）；
  表结构与派发全景在 [`arch/arm.md`](arch/arm.md) §二 / [`arch/x86_64.md`](arch/x86_64.md) §二。

### 2.5 C5：隐式清零与扩容耦合

- **机制**：某段代码依赖"这块内存已被清零"，但这个契约**没有任何地方声明**。
  删掉那句 memset 之后，只在特定组合（多线程 × 某个 chunk 尺寸 × 某条量化路径）下才暴露。
  而"改成写时补零尾行"这种看似更省的修法，又会**跨 tile 溢出写坏下一个 tile**。
- **证据**：`bb6bdcf827` *[CPU:Perf] Widen quant-KV (-qa 1) decode attention blocks to the float-KV tiers*
  ——同一个提交里踩了两次：去掉 memset 在多线程 64-chunk 下直接乱码；
  直接复用宽 V chunk 使 peak RSS +190MB。
- **预检**：删任何 memset 之前，**先找出谁依赖这块内存已被清零**并写进注释；
  然后在多线程 × 多 chunk 尺寸 × 量化/非量化上交叉验证。找不出依赖方就不要删。
- **附带的内存问题不是正确性问题但同源**：peak RSS 由**分配器 free-list 的复用形态**决定，
  与 buffer 大小不是单调关系。动扩容路径前用 fresh-alloc 埋点**实测**，
  不要凭 realloc 的触发条件推断——这一条已经误判过一次。
 详见 [`layout-and-memory.md`](layout-and-memory.md) §四。

### 2.6 C6：覆盖率假设（ISA × precision × 线程 × 长度）

这不是一类 bug，而是**让前五类逃过门禁的原因**。

会切换代码路径的轴有几条、每条必测哪些取值、源码里对应哪个 `if`，
全在 [`cpu/kernel/correctness-gate.md`](../kernel/correctness-gate.md) §一（**唯一源，本文不重列**）。
本文只补一张反向映射——**某条轴没扫，会漏掉本文的哪一类**：

| 没扫的轴 | 会漏掉 |
|------|--------|
| **线程数**（尤其没测超过 P 核数的档） | §2.1 的 cap 烘焙、§2.5 的清零依赖 |
| **长度 / shape**（没跨过 chunk 边界） | §2.2 的物理 chunk 地址公式 |
| **precision**（只跑了一档） | §2.4 的 arm82 逐字段二级表漏赋值 |
| **ISA**（只跑了目标档） | §2.3 的 packer/kernel 错配、§2.4 的快照时序 |

**"只测 prefill 或只测单线程，覆盖率是结构性不足"** ——不是覆盖得少，是整类 bug 落在测不到的格子里。
另外 op 单测的形状分布系统性偏向 prefill，所以 LLM 低 bit 改动**必须**加模型级 greedy 对拍。
命令与验证矩阵模板见 [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md) §五。

> ⚠️ 两个会让门禁本身失效的陷阱：`run_test.out` 的 argv 顺序是
> `<name> <backend> <precision> <thread> ...`，**错位不报错，只会静默测了别的组合**；
> argv[1] 是**单个前缀**匹配，**不支持逗号分隔多个 key**（逗号只属于 `MNN_TEST_SKIP`，且是精确名匹配）。
> 判定要**看 `passed` 计数，不要看退出码**——名字打错时"零个用例通过"也可能是 0 退出。

---

## 三、定位流程：从"哪一档错"倒推到哪一层

CPU 侧不需要每次都上 debugger。**扫维度比读代码便宜得多**，而且直接指向层。

**Step 1：先固定一个稳定复现的最小组合。** 若同一组合下结果不稳定，立刻转
[`general-debug/nondeterminism.md`](../../general-debug/nondeterminism.md) §9/§10——那是另一类根因。

**Step 2：沿四个维度各扫一遍，记下"对/错"的分界点。** 这一步的产出是**分界点**，不是结论：

| 分界点落在 | 指向 |
|-----------|------|
| 某个线程数之上 | §2.1（先确认那个档位是否越过了 `perfCores`） |
| 某个长度之上，且该长度接近 chunk 尺寸 | §2.2 |
| 某条 precision | §2.4（fp16 → 查 arm82 逐字段表） |
| 某条 ISA | §2.3 或 §2.4 |
| **哪一档都错，但只是质量差** | §2.3 —— 这类没有分界点，这本身就是判据 |

**Step 3：用替换法确认，不要用推理法。** 把怀疑的一环换成朴素实现或空操作，看结果怎么变。
典型替换点：把新 packer 换回旧 packer、把物理 chunk 强制回 64、把线程数硬编码成 1、
把二级表的新字段强制指向基表实现。**一次只换一件。**

**Step 4：在改代码前，先让那个"不变量"可观测。** 打印实际的 stride / chunk / 线程数 /
函数指针值，确认它就是你以为的值。这一步经常直接结束排查——
「我在 init 里写了赋值」和「运行时那个指针真的是它」是两件事。

---

## 四、与性能改动的关系：这一册为什么挂在 `cpu/optimize` 下

上面六类**全部由性能改动引入**，无一例外：放宽分块、换 tile、加 ISA 层、调线程策略、省一次 memset。
所以它们的正确的归属不是"debug 技巧"，而是**性能改动的必查项**。

一条元规则：**性能改动的验证矩阵必须覆盖它所触碰的那个维度的两侧。**
放宽了分块 → 必须测跨边界长度；改了线程决策 → 必须测超过 P 核数的档；
动了二级表 → 必须两个 precision 都跑；换了 packer → 必须做模型级对拍。
**改哪个维度就测那个维度**，这比"全都测一遍"更可行也更有效。

---

## 五、改动面 → 必查条目（前瞻索引）

提交前对照。左列是你**动了什么**，右列是**必须一起查的东西**。

| 你动了 | 必查 |
|--------|------|
| 线程数决策 / cap / 大小核策略 | 所有以线程数为参数且被持久化的量（打包边界、`mDivides`、按线程开的数组）是否与实际并发度同源（§2.1）；ThreadPool 等待策略的四条要点（[`runtime-and-scheduling.md`](runtime-and-scheduling.md) §1.4）；测一档 > P 核数 |
| 分片 / tiling / 划分轴 | `computeDivideSizes` 写的是**前缀和边界不是长度**；`multiThreadDivide` 用的是**原始** `threadNumber()`；三种划分工具语义不同（[`runtime-and-scheduling.md`](runtime-and-scheduling.md) §2.1） |
| 逻辑分块粒度 | 物理 chunk、地址公式、整除门限、**另一条路径里写死的旧常量**、是否与 threadNumber 联动（§2.2）；测跨 chunk 边界长度 |
| 物理 chunk 尺寸 | 同上，外加多线程性能回归（宽 chunk 在多线程上可能反而变慢） |
| pack / `UNIT` / `SRC_UNIT` / `DST_XUNIT` / `MNNGetGemmUnit` | [`cpu/kernel/pack-and-abi.md`](../kernel/pack-and-abi.md) §四 的七处同改逐条打勾（§2.3）；cell stride 用**真实 packed 字节数**；加模型级对拍 |
| 新增 ISA 层 / 新 `.S` | 写在 int8 表 snapshot 之前；手动设 `eP`；build list 是否含新文件；函数指针只在能力位满足时注册；降档回退路径可达（§2.3/§2.6） |
| `CoreFunctions` 结构（加/改字段） | arm82 / bf16 / x86_64 / riscv 四处二级表；声明处写 `= nullptr`；运行时打印实际指针值（§2.4） |
| Executor 新增成员变量 | **拷贝构造函数**是否同步（§2.4 末） |
| 缓存扩容 / realloc / memset | 谁依赖清零（写进注释）；多线程 × 多 chunk × 量化/非量化交叉验证；fresh-alloc 埋点实测 peak RSS（§2.5） |
| 融合算子拆成复用现成 kernel | 访存遍数是否变多（性能净亏）；优先扩签名保留融合语义，扩签名时回到"改 `CoreFunctions` 结构"那一行 |
| 任何低 bit / LLM 相关改动 | op 单测**不够**，必须加固定 prompt 的 greedy 对拍（§2.6） |

---

## 六、坐标速查

| 主题 | 坐标 |
|---|---|
| 线程数 cap | `CPUBackend.cpp computeThreadNumber()`；`perfCoreNumber` 赋值 `CPURuntime.cpp`（**Apple only**） |
| 工作划分 | `CPUBackend.cpp computeDivideSizes()`（写前缀和边界） |
| 权重打包边界 | `ConvInt8TiledExecutor.cpp` 的 `reorderWeight()`、`packWeightAndQuantInfo()` |
| prefill/decode 路径分叉 | `ConvInt8TiledExecutor.cpp`（`mUseBatchQuan`） |
| kernel 选择（架构分叉） | `ConvInt8TiledExecutor.cpp`（x86_64 `nbits() <= 7` / ARM `OVERFLOW_AWARE`；`Int8GemmKernel_W4` **只在 ARM 分支**） |
| packer 模板参数 | `compute/Int8FunctionsOpt.cpp`；tile 常量 `Int8FunctionsOpt.h`；int8 表 snapshot 在 `MNNCoreInt8FunctionInit()` 里做两次（第二次覆盖第一次） |
| 分块常量 | `compute/CommonOptFunction.h`：`MNN_FLASH_ATTENTION_BLOCK_SIZE`（64）、`MNN_FLASH_ATTENTION_BLOCK_DECODE`（2048，声明注释里带实测数字） |
| 物理 chunk 选择 | `CPUKVCacheManager.hpp flashAttentionChunkKv()`（启用条件很窄；注释里点明 V-int8 调用点写死旧常量） |
| ⚠ arm82 逐字段二级表 | `arm82/Arm82Functions.cpp` + 后续约 120 条 `FUNC_PTR_ASSIGN` |
| ✅ 对照：整体拷贝的安全写法 | `x86_x64/AVX2Functions.cpp`、`cpu/bf16/BF16Functions.cpp` |
| `CoreFunctions` 定义（看哪些字段有 NSDMI） | `compute/CommonOptFunction.h` |
| 单测 argv 与名字匹配 | `test/main.cpp`；`test/MNNTestSuite.cpp` 的 `MNNTestSuite::run()`：单前缀匹配 `test->name.find(prefix) == 0`、跳过按 `MNN_TEST_SKIP` 里的精确名 |

---

## 七、新增类别

发现新的跨层不一致类别时：在 §一 路由表加一行，在 §二 按 `C<N>` 递增加一小节（机制 / 证据 / 预检 /
测哪一档），并在 §五 改动面索引里补对应的行。**证据必须是真实提交或可复现的实验**，
不要写"据说"。如果发现的是**框架级、跨后端**的类别，去
[`general-debug`](../../general-debug/SKILL.md) 新开一份分册并同步它的分流表，不要放在这里。
