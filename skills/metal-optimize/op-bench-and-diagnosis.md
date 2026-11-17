# 用 op 单测 + 对手基准定位性能问题(诊断流程)

> **何时读**:e2e 比对手慢，但不知道慢在哪;或者已知某个 kernel 慢，想知道"还有多少空间、
> 该往哪个方向改、什么时候该收工"。
>
> **本文只讲流程**(怎么发现问题 → 怎么找到真瓶颈 → 怎么定方案 → 怎么收工)。
> 具体 kernel 手段见 [`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md) §2.5。

> ## ⛔ 本文件的记录规则
>
> 1. **只写流程与判据**:怎么发现问题、怎么排除伪信号、怎么定方案、什么时候收工。
>    **不写任何性能数字**(百分比、ms/µs、GB/s、TFLOPS、tok/s、倍数、配对计数、扫点表)。
> 2. **不写溯源**:日期、commit / 分支标号、设备型号与模型名,只在"这条判据仅对某类设备或
>    形状成立"时以**类别**出现(如"tensor-coop 设备"、"短 ctx"),不写具体机型、模型与时间。
> 3. **案例只留可复用的那一句**:§9 的案例表写"症状 → 诊断手段 → 动作 → 定性结果",
>    结果栏只写方向与机制(领先 / 判负 / 为什么),不写幅度,不留实验档案。
> 4. **陷阱与验证方法要留**:会让人重复踩的口径错误、顺序偏置、仪器差异是操作知识,
>    写"触发条件 + 规矩 + 怎么验",不写事故经过。
> 5. 环境变量只在本文件中作为**路径开关**被引用;名称、默认值、语义的唯一登记处是
>    [`env-registry.md`](./env-registry.md)。

## 0. 核心洞察:为什么必须"最小单元 + 对手"两件一起上

两件事各解决一个独立的问题,缺一个流程就走不通:

| | 解决什么 | 缺了它会怎样 |
|---|---|---|
| **op 单测(最小单元)** | **信噪比**。目标 kernel 在单测里占满被测时间,ΔK 满幅可见 | e2e 本身的抖动就有百分之几,占比小的 kernel 收益被稀释掉 ⇒ 真收益埋在噪声里,正负号由当天热漂移决定 |
| **对手基准** | **绝对标尺**。告诉你"这个数字是不是已经到硬件极限" | 只有自己的历史数字 ⇒ 「已优化不少」和「还差几倍」长得一样,无法决定要不要继续投入 |

**代价已实际发生(两个方向都翻过车)**:

- 只看 e2e:`FATC_QK_K32` + `FATC_O_CT` 曾被 e2e A/B 判为「证伪」并默认关闭,
  次日改用 op 单测复测才发现是明显的真收益,e2e 随后也复现了。
  一个真收益被雪藏一天,还附带了一个错误的机制解释。
- 只看 op 单测:`gateup ROW2_SPLITK` micro 明显为正、e2e 却是负的(§2.1.10)。

⇒ 结论不是"哪个更可信",而是 **两个尺度分工不同**,见 §4 的分工表。

## 1. 第一步:造 op 单测(最小单元)

放在 `test/speed/*.cpp`,用 `MNNTestSuiteRegister` 注册成 `speed/Xxx`。**五条构造要求**:

1. **形状照抄真实模型,不要用整数漂亮的形状。** `LlmAttentionSpeed.cpp` 直接写死
   q=16 / kv=8 / head_dim=128 / 28 层,并用 `MNN_QWEN3_MODEL` 切 0.6B / 4B / 2B。
   形状决定走哪条 kernel 分支,漂亮形状会把你测到另一条路径上去。还要按真实图核对融合前缀、后缀及出现次数，首层无 LN 等例外必须单独计组。模型选择参数可能静默回退到默认项，采纳结果前必须核验输出的实际模型、形状与权重足迹。
2. **一次 `onForward` 覆盖整个模型的层数,再除回 per-layer。** 单层的 GPU 工作量太小,
   launch 噪声占比过高;28 层一起发射才接近真实调度密度。报告值 = wall / loop / layers。
3. **报"图形无关的品质因子",不是 ms。** 带宽 bound 的 op 报 **GB/s**(decode attention
   报 KV GB/s),算力 bound 的报 **TFLOPS**(prefill attention 报 causal TFLOPS)。
   ms 无法跨 ctx / 跨设备比,GB/s 可以直接和硬件峰值比 ⇒ **单测自己就能回答"还剩多少空间"**。
4. **规模要能扫,而且是 env 扫,不是改代码。** `MNN_ATTN_CTX` / `MNN_ATTN_SEQ` /
   `MNN_ATTN_ROUNDS`。**规模扫描是后面固定/流式分解的输入**,没有它 §3.2 做不了。
5. **第一轮必须是不计时 warmup**,输出打成 `r-1` / `r0` / `r1` 每轮一行 + `BEST` 汇总行,
   便于脚本 `awk` 取 min。

> ⚠️ **`./run_test.out <name> <backend> <precision>`,末位 precision 必须是 `2`(Low=fp16)。**
> `1` 是 fp32,会让所有 fp16-only kernel **静默**退回慢路径——不报错、不打 banner,
> 只是数字掉一大档。跑之前先 grep 一眼 kernel active banner。

## 2. 第二步:造对手镜像

> 🔒 **硬规则:op 耗时对比只能来自 op 单测,且对手侧必须复现融合算子的同一段「头到尾」。**
> MNN 的 `FusedLinear` / `RoPE` 是融合结果,对手侧若只测其中一段(比如只测 rope 不测
> q_norm/k_norm、只测 GEMM 不测 rms-norm 前缀和 silu-mul epilogue),比的就不是同一件事。
> 整模型 e2e 与 `llm_bench --profile` 的模型内数字**只能用于自查"有没有 op 之外的时间"**
> (§3.5),**绝不能充当 op 对比**——两侧仪器不同,比出来的是仪器。
> 需要 e2e 时,e2e 的用法是:`e2e per-layer − Σ(自家单测)` = 自家 op 之外的每层开销,
> 两侧各算一遍再比这个残差。

对手镜像放在 `~/mlx-bench/*_mlx.py`,和 C++ 单测**同名同形状同品质因子**,直接对着输出表读。
已有 6 份:`qwen3_attn_decode_mlx.py`、`qwen3_attn_prefill_mlx.py`、
`qwen3_decode_linear_mlx.py`、`qwen3_prefill_linear_mlx.py`、`qwen3_fused_linear_mlx.py`、
`qwen3_rope_mlx.py`(另有 e2e 侧 `mlx_pg.py`、`qwen3_linear_attn_mlx.py`)。

> ⚠️ **这些镜像不在仓库里**:它们只依赖外部参照实现的框架、与 MNN 代码无关,
> 已从 `test/speed/` 移到 **`~/mlx-bench/`** 且不跟踪(那里有 `README.md` 列出对应关系)。
> **解释器必须用 `~/.venv-py312/bin/python`**——系统 `python3` 里没装对手框架,
> 直接跑会 `ModuleNotFoundError`。旧记录里可能仍写着旧路径 `test/speed/*_mlx.py` 和 `python3`,
> 按本节换算。

### 2.1 四项可比性核对(动手比之前必须过一遍)

**MMA/FLOP 条数、tile 形状、grid 规模、稀疏/提前退出策略。** 四项不一致时数字不可比,
差距会被算到错误的原因上。补充两项本次踩到的:

5. **两边是否做了同一件事的全部。** 对手镜像最初只对着预建 KV cache 做 SDPA,
   **从不 append 新 token**,而 MNN 每步必付一次 cache 写 ⇒ 基准少收了对手的钱。
6. **两边的 loop 结构是否等价。** 见下条陷阱。
7. **两边的 repeat / 统计口径是否逐字一致。** 两项都要核:**丢不丢 warmup**、
   **报 mean 还是 max**。曾发现对手镜像驱动一直是全部保留 + 报 `max`,
   而 `llm_bench` 是 `for (i=0; i<nRepeat+1; ++i)` + `if (i>0)` 丢首轮 + 报 `mean ± std`
   ⇒ **拿对手的 max 比 MNN 的 mean,白送对手约一个 std**。当轮对手的 rep0 在每一格
   都是编译离群点(比后续 rep 差一倍以上),这个偏置足以凭空造出或抹掉一个结论。
   **写镜像驱动时先读对手 harness 的 repeat 循环,再写自己的。**

### 2.2 ⚠️ 惰性图陷阱(任何 lazy graph 框架)

**把 N 步攒成一张图再 eval,会改变 in-place donation 语义。**

给对手镜像补 append(照对手框架 KV cache 的 `update_and_fetch` 写
`k_cache[..., ctx:ctx+1, :] = k1`)后,开销**随 ctx 线性增长**——即每步**整份 cache 重拷**,
比 MNN 的单 token 写贵一个数量级以上。

**根因是 harness,不是对手框架**:benchmark 把多步串成一张惰性图,
N 个 `slice_update` 链式依赖 ⇒ 前一版本仍活在图里 ⇒ **buffer donation 失效,退化成整份拷贝**。
改成每步 eval(`--loop 1`)后 append 几乎免费。

⇒ **要量"每步"成本就必须每步 eval**,并单独评估 sync 噪声(loop=1 时 sync 噪声会反过来主导)。

⚠️ **上面这条 `--loop 1` 的结论(「append 近乎免费、MNN 的 copy dispatch 是真实
劣势」)后来被修正为偏低**:loop=1 每步的 sync 会把两臂的差压扁。
反过来,"每个入队迭代各给一套 cache"虽然避开了 copy-on-write,却引入**冷足迹**惩罚
(且随 ctx 放大),会**高估**对手。

**两头都避开的做法 —— A/B 双 cache**:批量图里每层备两块 cache,**写 A、SDPA 只读 B**,
两臂都分配 A 和 B,唯一差别是那几次 slice 写。A 从不被读 ⇒ 不触发 copy-on-write;
批量 ⇒ 无 sync 主导;足迹相同 ⇒ 无冷热偏差。这样量出的对手 append 开销
**绝对值不随 ctx 变化 ⇒ 确认是 O(1) dispatch 开销**。校验:该法的 "SDPA only" 一列
与默认镜像逐格吻合。
**结论翻转**:补上这一次 dispatch 后,原先记在 attention 头上的 e2e 缺口随之撤回。

**识别信号:开销随规模线性增长,而你以为它是 O(1)。** 见到这个立刻怀疑图结构,别急着下结论。

### 2.3 ⚠️ CSE guard 陷阱:不要扰动输入,要扰动参数

§1 第 2 条要求"一次 `onForward` 覆盖整个模型的层数",于是必须让 N 层不被
公共子表达式消除(CSE)折叠成 1 层。attention 单测用的写法是给每层输入乘一个不同的标量:

```cpp
auto qL = q * _Scalar<float>(1.0f + 0.001f * (float)l);   // ❌ 对带宽 bound 的 op 是灾难
```

**这个乘法本身就是一趟完整的读+写。** 给 RoPE 抄这一招时,每层为 q 张量白付两倍于
它自身大小的访存——**和被测 op 的工作量同一量级**,直接把有效带宽腰斩,
并让 MNN 在对手对比表里凭空落后。

**正确做法:扰动参数,不扰动数据流。** RoPE 单测改成逐层给不同的 norm gamma:

```cpp
const float gamma = 1.0f + 0.001f * (float)l;             // ✅ 编译期常量,零运行时成本
param->q_norm->gamma = std::vector<float>(headDim, gamma);
```

权重 / gamma / scale 这类参数是 const buffer,逐层不同即可阻止折叠,而且不增加任何访存。

⇒ **判据:CSE guard 的成本必须远小于被测 op。** 算力 bound 的 op(GEMM)对输入乘法不敏感,
带宽 bound 的 op(RoPE / norm / elementwise)必须走参数侧。加 guard 后先看品质因子——
**GB/s 明显低于硬件峰值的一半时,先怀疑自己的 harness,再怀疑 kernel。**

## 3. 第三步:读差距——三种诊断,按问题选

### 3.0 ⚠️ 先做预算对账:把所有 op 的 Δ 加起来,对上 e2e 缺口

**在挑哪个 op 去优化之前,先把每组的 Δ 求和,看能不能凑出 e2e 的缺口。**
不做这一步,你会去优化一个**方向正确但权重被高估**的项 —— 单看比值,任何落后的 op
都像值得动;只有和 e2e 对上账,才知道它值几个点。

一次 decode 对账(pg512,128,base 档)的形态:

| 项 | Δ(MNN − 对手) |
|---|---|
| linear(5 组求和) | **领先** |
| attention ×N | **领先** |
| 其余未测部分(相减得到) | **落后,且等于缺口的全部**(⚠️ 随轮次变,见下) |
| **合计** | **= 该轮 e2e 实测缺口 ✓** |

结论直接反转了投入方向:**两个被测过的大项都是领先项** —— linear 里那个唯一落后的
`o_proj` 即使全额追回也只值 e2e 很小一截;而缺口 **全部落在从没被测过的残差**里。
在此之前几轮 op 级优化都花在 linear 上。

⚠️ attention 那一格最初记的是**落后、占缺口的大半**,一度被当成第二大待优化项 ——
**其实是镜像少了一次 KV 写 dispatch**(§2.1 第 5 条 / §2.2)。修正后它变成领先项,
残差随之明显变大。**对账表里任何一格的口径错误,都会等量地转移到「残差」那一格**,
所以每一格都要能独立复现,否则残差只是误差的垃圾桶。

⚠️ **残差不是常量,别跨轮次搬用**。它是「e2e 缺口 − 已测 op Δ」相减得到的,
所以跟着**那一轮自己的 e2e 缺口**走:换一轮重测,同一格的缺口可以差好几倍,残差也跟着差。
方向不变(两个已测组都领先、缺口全在未测部分),但量级的相对不确定度很大,
**必须按区间引用**。推论:**e2e 缺口与 op Δ 必须在同一轮里测**,
否则相减的是两轮各自的热漂移之和。

三条纪律:

- **口径要对齐到同一个 wall**。`llm_bench` 的 decode speed **不含采样**,对手侧的
  `generation_tps` 把 argmax 算在生成循环里 —— 直接对比会把缺口少报一截。
  **用两侧都含采样的 wall 值。**
- **op 预算的绝对值不可引用,只有差值有意义**。op bench 隔离测 best-of-N,
  真实 e2e 里算子背靠背争带宽 ⇒ 两侧预算都系统性偏低、残差都被抬高。
- **「相减得到的残差」要标注成未测项**,不要当成已归因。它里面混着 embedding gather、
  RoPE、residual add、final norm、采样,以及每 token 上百次 dispatch 的框架开销。

拿到「MNN 比对手慢一截」这种数字后,**它本身不能转成任何动作**。要先拆。

### 3.1 绝对标尺:先问"这是差距还是物理极限"

把品质因子和硬件峰值比:小 GEMV 离峰值带宽还远、lm_head 已经接近
⇒ 小 GEMV 有 headroom、lm_head 没有。**对手数字的第一用途是校准"可达值",不是排名。**

### 3.2 固定/流式分解:差距随规模变化时,第一件事

**拿两个干净的规模点拟合 `t = a + b·N`**,分出**每次固定开销 a** 和**流式速率 b**。
本次用最小与最大两个 ctx 的 min:

| | 固定 a | 流式 b(折算带宽) |
|---|---|---|
| MNN | 大 | 略高 |
| 对手 | 小 | 略低 |

⇒ **MNN 的内层流式已经略胜对手,残余差距全部在 a。** 这一步的价值是**直接砍掉一整类方向**:
所有内层 ILP / 访存合并 / 批量预发射的手段都不用试了(事后 `SDPA_KVB` 批量预发射实测
在长 ctx 上为负,正好验证)。

**自检**:把拟合回代到第三个规模点。对手侧的预测与实测很准;MNN 侧偏松
⇒ 说明小规模那个点噪声大,结论要打折。**回代不准就别用这个拟合下强结论。**

**⚠️ 能直接造出近零规模点时,不要外推截距。** 定位 `o_proj` 时我曾从 4 个带噪点
拟合出「MNN 的 per-dispatch 固定开销高于对手」—— **符号是反的**。加两个近零权重探针
(`diag_fixed_x28` / `diag_fixed_x112`,把 oc 缩到极小 ⇒ 流式项可忽略)直接读,
**MNN 的地板反而更低**。GEMV/GEMM 这类形状可以自由缩到近零,**几分钟的探针胜过任何拟合**;
`x112` 那一臂还顺带验了「固定开销随 dispatch 数线性」(否则是 per-onForward 常量,
诊断完全不同)。拟合只留给规模无法缩到零的场景(如 ctx)。

### 3.3 消融阶梯:差距在 kernel 内部时

逐类删工作读差值,得到分项成本表和**地板值**,然后**拿地板值和对手比,不要拿总时间比**。
完整方法(含防 DCE、四条纪律)见 `kernel-dev-and-optimize.md` §2.0「归因方法」。

本次用它给 copy dispatch 定价:临时 `MNN_METAL_ATTN_SKIP_COPY` 消融 ⇒ 只解释缺口的
一小部分。**我原本预测它是主因,证伪。**
⇒ 消融的价值一半在证伪自己的假设。

### 3.4 ⚠️ 不要写"host vs GPU 拆分"探针

在 timer 中间插一刀量 enqueue 时间:长 ctx 下会报出几十 µs/layer 的 "host",
实则是命令队列满之后的 **GPU 背压**,不是 CPU 开销。**这种探针无法分离 host 与 GPU。**
要区分只能用 GPU busy vs wall(§2.0)。

### 3.5 GPU busy vs wall 的具体做法:`-DMNN_SESSION_CPU_TRACE`

这是**唯一**能把真实 GPU 时间和提交气泡分开的工具,不是 env 而是**编译宏**
(`MetalBackend.hpp` 的 `struct MetalCpuTrace`,单独开一个 build 目录编)。它在
command buffer 的 completion handler 里累计 `GPUEndTime − GPUStartTime`,同时记
相邻 buffer 之间的 **gap**,进程退出时打印;另有 `encodeNs/encodeOps`、`commitNs`、
按站点分类的 `waitSiteNs[4]`。

读法:**GPU busy / forward 次数 vs 单次 wall**。判 `o_proj` 到底是不是 GPU-bound 时,
busy 除以 forward 次数正好对上单次 wall ⇒ **计时区间内 GPU 是满的**,那些巨大的 gap
全落在计时区间外(建图/预热)。
⇒ 排除了 CPU encode 与提交气泡,剩下的只能是 kernel 本身。

⚠️ **gap 的总量会很大且容易误读**,一定要看它落在计时区间内还是外;
harness 侧可以另外把 enqueue 与 wait 拆开(`LlmLinearSpeed.cpp` 的
`enq=... (%)` 输出)当粗筛——**GPU-bound 的组 enqueue 应远小于总时间**,
但它只能当粗筛,定论仍靠本节的 busy vs wall。

### 3.6 不变量判据:先问"这个改动增加了总在途字节吗"

窄 dispatch 打不满带宽时,最容易想到的一整类改动是**重新分组同一份工作**
(拆 K、一个 simdgroup 多带几行、换 lane 划分)。这类改动**全都不该有收益**,因为

> 总在途字节数 =(参与的 simdgroup 数)×(每 simdgroup 的 load 数)
> **在同一份工作的任何重分组下都是不变量。**

`o_proj`/`down_proj` 上跨三个时间点试了三个变体,全部落在噪声量级且多数为负:
`SPLIT_K_2`、`SPLIT_K_SHUFFLE`(已删码)、`ROW_2`(宏现名 `GEMV_SG_OC_QUADS_2`)。
事后看它们本来就该全负。**遇到这类问题先算这个不变量,没增加就别写 kernel。**

真正能动的只有两类:**提高每次 dispatch 的字节量**(融进邻居 op),或
**加宽每 lane 的 load**(同样的请求数搬更多字节)。

⚠️ 反过来也要小心:**别把一个机制从它成立的路径照搬到另一条路径**。
`ROW_2` 在**融合** GEMV 上确实赚,但那是因为融合体有可共享的 LN/input
前导;plain GEMV 里权重读就是全部流量,双行只是重分组 ⇒ 判负。
(详见 `kernel-dev-and-optimize.md` §2.1.6 的边界说明。)

### 3.7 ⚠️ `llm_bench --profile` 只能做「同 `-rep` 的两 pp 差分」

per-op 同步的 profiler 有三个叠加缺陷,**直接读它的绝对值或百分比一定错**:

1. **`Avg(ms)` / `Called times` 在 `loops=1` 下是累计和,不是均值。**
   见 `tools/cpp/Profiler.cpp`:`costTime` / `calledTimes` 逐次累加,打印时除以 `loops`。
2. **调用次数里混了 load-time warmup 的 decode step。** 按 `-rep` 扫会发现 Attention 的
   调用次数随之变化:**每个 prefill pass 只贡献"每层一次"**,剩下的大头全是 seq=1 的
   decode。**大部分调用次数不是你要测的东西。**
   识别信号:`Called times` **与 pp 无关**(换 pp 读数不变)。
3. **每个被 profile 的 op 付一次 `wait(MAP_TENSOR_READ, true)` 的 flush 常数。**
   两点拟合出的单次常数 × 调用次数,正好等于 profiled 总时长与真实总时长之差。
   ⇒ **调用多、单次工作量小的 op(RoPE、BinaryOp)被系统性抬起来。**

**正确读法:固定同一个 `-rep`,拿两个 pp 相减。** baseline 调用次数与 decode 工作量
逐项完全相同,精确抵消,余下就是 `N_pass × (w(pp2) − w(pp1))`;再按 `w ∝ seq`
(GEMM/RoPE)或 `w ∝ seq²`(causal attention)外推到目标 seq。

一个 4B 长 pp 的实例:原始表报的"非 GEMM 残差"里 RoPE 占最大头;差分还原后**真实残差
只有原始读数的零头**,其中 RoPE 对照带宽下限(q + k 的读写字节 / 峰值带宽)已在最优附近。
**若照原始表去优化 RoPE,会花整轮时间追一个不存在的缺口。**

但**差分只够用来"排除",不够用来"定价"**:相减得到的区间很宽,而后来补的
`speed/LlmRoPE` 单测给出的值落在区间内,方向对,
**但只有单测能拿去和对手比,并回答"离单趟下限还剩多少倍"这种可行动的问题**。

⇒ **本节的读数都只是 MNN 自查:回答"时间有没有落在已知 op 之外"。**
`--profile` 的模型内 per-layer 值
**不能与对手框架的 op 数字并列成表**——两侧仪器不同,比出来的是仪器(见 §2 硬规则)。

⇒ 配合 §3.1:**先用带宽/峰值给这个 op 估一个上界,再决定要不要相信 profiler 的百分比;
真要定价就去补单测。**

## 4. 第四步:定位到旋钮后怎么标定(尺度分工)

| 改动类型 | 标定尺度 | 理由 |
|---|---|---|
| **路径开关**(走不走 kernel X / 阈值) | **e2e 为准** | 防"kernel 级快 / e2e 平"的伪收益;`ROW2_SPLITK` 就是 micro 为正、e2e 为负 |
| **已确认在关键路径上的 kernel 内部参数**(nsg / tile / batch) | **先 op 单测标定,再 e2e 验证不回归** | e2e 分辨不出这一档;decode attention 的 nsg 默认值就因为标在 e2e 上而**长期是错的** |

第二类的教训值得展开:

> decode attention 的 nsg 默认值曾来自一次 e2e 扫描(长 pp 下大 nsg 胜出),
> 另一档直接继承了这个默认值。改用 op 单测(attention 占满被测时间)复测,
> **nsg16 在每个 ctx 都胜**,且 ctx 越短幅度越大。落地后 e2e decode 把对对手的落后逆转成领先。

**旋钮标定的两条附加纪律**:

- **组合扫,不要单旋钮扫。** A 的停顿盖住 B 的收益时,单独扫 B 会把正向项判负。
  已判负的旋钮,同 kernel 有新正向项落地后**要复测**。
- **扫完要确认是尖峰还是平台。** 胜出档的两侧邻居都明显更差 ⇒ 尖峰,旋钮已耗尽,别再扫。
  顺带:最小档变慢的方向与"短 ctx 该少开 simdgroup 省归约"的直觉**相反**——
  grid 只有 `(1, B*H, 1)` 个 threadgroup,短 ctx 的瓶颈是**并行度不足**。
  **旋钮的胜负方向本身是机制证据,要读它。**

## 5. 第五步:验证

1. **先过正确性 oracle,再信任何数字。** 错误 kernel 一样能变快。
   本次:5 个 attention op 用例(`op/attention` / `_c4` / `_c4_tail` / `_hd256` / `_prefill`)。
2. **⚠️ 改归约顺序的旋钮不能用 byte-compare 当判据。** nsg 改变跨 simdgroup 求和顺序 ⇒
   输出是另一个合法求和序,**不与旧版逐字节一致**。此时 oracle 只能是带容差的用例。
3. **e2e 的作用是"排除副作用",不是"发现收益"。** 本次同时验了 4096,128 无回归。
4. **无关失败要用零重建基线证明是既有问题。** 全 `op` 套跑到
   `op/unary/erfinvInt8` SIGKILL(exit 137)。**不要假设与自己无关**——
   把 nsg 默认值临时改回旧值(**不重建**)重跑整套,
   两次 exit code、到达测试数、失败集合**逐字节相同** ⇒ 证明与本次改动无关(累积内存压力)。
   **env 覆盖是最廉价的 A/B 基线,凡是能用 env 回到旧行为的改动都该这么验。**

## 6. 第六步:收工判据

**不是"没想法了"就收工,要能说出剩余差距的机制、量级和它被什么挡住。**

本次收工时的账(可以照这个格式写):

- 流式 b 已胜对手 ⇒ 内层无空间。
- 残余 = 每层的固定开销,其中 copy dispatch 已单独量出占一小部分,
  余下是 per-layer host/dispatch。
- 要再动只剩结构改动(把 KV 写融进上游 RoPE kernel 省一次 dispatch,或 `MTLDispatchTypeConcurrent`),
  **e2e 上限不到一个点**(attention 只占 decode step 的一小部分,再乘最好情况)。
- e2e 侧已**领先**对手。
- ⇒ 按 §2.0 纪律,**op 级差距在 e2e 上兑现不出来就不追**,收工。

**归因于"编译器已经优化得很好了"一律视为未完成。**

**证伪也是成果**:中性/负收益要连方法一起写进 commit / PR,防止下一轮重复投入;
只有能提炼成**跨形状仍成立的判据**时才进 skill(见 [`SKILL.md`](./SKILL.md) 记录规则)。
本轮判负 4 条(`SDPA_KVB`、copy dispatch 定价、
host/GPU 探针不可用、对手 append 建模不可用)。

## 7. 度量卫生(不遵守则以上全部无效)

- **先核对开关的实际控制范围**：关闭加载特化可能同时改变融合路由、split-K 和 staging。
  对照两臂的 pipeline 宏及 grid/threadgroup，再把这些变化分别消融；组合差值不能直接归因于加载宽度。
- **交替配对**(A/B、B/A),**丢弃每臂第一轮 warmup**,**轮内取 min**,
  **3 次同向且区间不重叠才算信号**。
- **串行 sweep 会被顺序/热度污染**:同一条臂串行扫下来,绝对值几分钟内就能单向漂出
  一个明显台阶。固定"先跑某一档"的扫描 ⇒ 系统性偏向那一档。
- **先证明开关真的传进去了**:`#ifdef` 里塞 `#error` 确认只有开开关时 pipeline 编译失败,
  再撤掉。「改了没效果」和「改的代码根本没进 shader」在数字上完全一样。
- **zsh 不对未加引号的参数展开分词**:多变量用 `${=var}` 或写 `A=1 B=2 ./bench` 前缀
  (SKILL.md 原则 7b)。注意 `${=var}` 是 **zsh 专属**,bash 里报 `bad substitution`。
- 手工替换 `libMNN.dylib` 后必须 `codesign -f -s -`,否则进程被 SIGKILL 而脚本只记下一个 0。
- **llm_bench / llm_demo 是薄启动器,Metal 代码全在 dylib 里**:双二进制 A/B 必须给每臂各存
  一整套 dylib(`libMNN` / `libllm` / `libMNN_Express`,注意 `libMNN_Express` 在 `build/express/`)
  并按臂换入 `build/` 再跑——只复制可执行文件会让两臂加载同一份库(rpath 指向 build 绝对路径),
  A/B 静默失效。整份构建产物的 cp 自带签名,无需重签。可执行文件要在 `build/` 里跑(本机从
  /tmp 直接执行吃 SIGKILL、零输出),脚本本身放 /tmp 没事。
- **Bash 工具的工作目录会跨命令保持**,`cd build` 连着第二次 `cd build` 会失败并让
  `&&` 链静默跳过前一个测试 ⇒ **脚本里一律用绝对路径**;git 命令的 pathspec 也按 cwd 解析,
  链里写 `git -C <repo>`,不要依赖当前目录。
- **跨引擎 e2e 的每一格都要反序复核**:一轮全量矩阵里的 decode 负项做反序,**多数翻符号**。
  「MNN 先对手后」在小差距上有一个**系统性顺序偏置**,量级与要判的缺口同阶,
  所以 **小差距的格子,单向顺序的结论一律不成立**。反序只要多花一倍机时,
  却是「真缺口」与「顺序伪影」之间唯一的判据。
- **「反序复核后判真」有保质期,只绑定当时那个 build**:早先某一轮判真的两个 decode
  缺口,在后来的 nsg 分档修复之后全部翻正。**引用任何历史缺口
  之前先确认它测在哪个 commit 上**,否则会去追一个已经被别人修掉的问题。
- **每一轮都带一个「空臂对照」**:让至少一个 group 的代码在两臂**逐字节相同**(比如开关
  只作用于 kernel 的一条分支,其余 group 走别的 kernel),它读出的 Δ 就是这台机器**当下**
  的噪声底。一轮 probe 正是靠它同时抓到两件事:(a) 固定 A 先跑的顺序偏差——对照组
  每一对都读出同向的非零差;(b) 修掉顺序后噪声底依然不可忽略,且逐对符号混杂
  ⇒ 同一轮里被测组只有超出这条线才算信号。**没有空臂时,"每对都同向"完全
  可能是顺序伪影**——它看起来和真信号一模一样。
- **空臂可以「白捡」**:扫因子/档位时,如果某个现存 env 开关的 off 态恰好和某一档路由**逐字节相同**,
  就把它当空臂一起排进轮转,不用另造对照——两臂读数的差就是当轮噪声底,只有明显超过它的差值才敢判
  为真信号。**排臂前先跑一次指纹取证,确认哪两臂真的等价。**
  ⚠️ **取证要覆盖开关能碰到的全部路由,不只是被测那条**:同一个开关换一个模型就可能不是单变量了
  ——例如某模型同时有 gate/up 与四投影组两条窄路由,关掉"窄加载调度"会**连带**改掉另一条,于是
  这个"空臂"与被测臂差出远超噪声底的一截。当时只 grep 了被测那两条 pipeline 就默认等价,是靠
  "两臂差得离谱"才发现的。**空臂不成立时它会伪装成一个巨大的假信号。**
  注:能白捡的空臂取决于当时现存的开关;开关被删掉后要另找,或临时加一个测量用旋钮、测完删掉。
- **每条 arm 的 per-rep 原始输出必须落盘,只留均值等于放弃事后审计能力**:
  一次 prefill 的方向翻转,靠的正是比对两轮的逐 rep 值才判出「是 MNN 臂
  被持续负载压低、而非回归」——只有均值时这条结论是「判不了」。

### 配对脚本模板

```zsh
#!/bin/zsh
# usage: KNOB_NAME=<被测 env 名> pair.sh <ctx> <valA> <valB> <pairs>
# 现存开关的名称与语义见 env-registry.md
ctx=$1; a=$2; b=$3; pairs=${4:-3}
cd /Users/jiuqi/AliNNPrivate/build || exit 1
run() {
  env $KNOB_NAME=$1 MNN_ATTN_CTX=$ctx MNN_ATTN_ROUNDS=3 \
    ./run_test.out speed/LlmAttnDecode 1 2 2>&1 \
  | awk '/^r[0-9] /{if($4>0&&(m==0||$4<m))m=$4} END{printf "%.5f", m}'
}
run $a >/dev/null; run $b >/dev/null      # 每臂丢一轮 warmup
for i in $(seq 1 $pairs); do
  # 真的交替:奇数对 B 先跑。固定 A 先跑会给「两臂代码逐字节相同」的
  # 对照组读出非零差,即把对内漂移全记在一条臂上。
  if [ $((i % 2)) -eq 1 ]; then ra=$(run $a); rb=$(run $b)
  else                          rb=$(run $b); ra=$(run $a); fi
  echo "p$i A=$ra B=$rb delta=$(echo "$ra $rb" | awk '{printf "%+.2f%%", 100*($1-$2)/$1}')"
done
```

对手对比脚本同形:两个 arm 换成 MNN 单测和对手镜像,各自 `awk` 取 min 后交替跑。

## 8. 现成资产

| 资产 | 位置 | 备注 |
|---|---|---|
| op 速度单测 | `test/speed/*.cpp` | attention / GEMV / GEMM / LinearAttention / FusedLinear / Conv 等 22 份 |
| 对手镜像 | `~/mlx-bench/qwen3_*_mlx.py`(**不在仓库里**,只依赖对手框架) | 6 份,与对应 C++ 单测同形状同品质因子;e2e 侧另有 `~/mlx-bench/mlx_pg.py`。旧记录里的路径换算见 §2 |
| attention decode | `./run_test.out speed/LlmAttnDecode 1 2` | env:`MNN_ATTN_CTX` / `MNN_ATTN_ROUNDS` / `MNN_QWEN3_MODEL` |
| attention decode 对手 | `qwen3_attn_decode_mlx.py --ctx N` | **与 MNN 比必须加 `--kv-write`**,否则镜像少一次 KV 写 dispatch(§2.2) |
| attention prefill | `./run_test.out speed/LlmAttn 1 2` | env:`MNN_ATTN_SEQ` / `MNN_ATTN_ROUNDS` |
| decode GEMV 诊断组 | `MNN_DECODE_GEMV_DIAG=1 ./run_test.out speed/LlmDecodeGemv 1 2` | 在四模型真实形状之外加诊断轴:`diag_i2k_oc*` / `diag_i*k_oc1024`(总字节固定 33 MB,分别沿 oc 与 ic 扫 bytes-per-dispatch)、`diag_fixed_x28/x112`(近零权重,直读 per-dispatch 固定开销并验线性)。`MNN_DECODE_GEMV_GROUP` 按子串筛组;对手侧 `qwen3_decode_linear_mlx.py --diag` 同形状 |
| e2e 全量矩阵 | `tmp/mnn_vs_mlx_matrix2.sh [precool] [out]` | 4 模型 × pg{512,2048,4096},每 arm 前 90s 预冷,MNN 先对手后;**每条 arm 的原始 stdout 落 `$out.log`**(事后审热漂移的唯一依据)。配套 `tmp/rev_check.sh` 做反序复核 —— **任何负项都必须过一遍它**(§7) |
| 对比 / 配对脚本 | `tmp/*.sh`(未跟踪的 scratch) | `attn_decode_vs_mlx.sh`、`nsg_pair.sh`、`mnn_vs_mlx_e2e.sh` 等 |

## 9. 案例总表(症状 → 诊断 → 动作 → 结果)

| 案例 | 症状 | 诊断手段 | 动作 | 结果 |
|---|---|---|---|---|
| **decode SDPA nsg** | op 级短 ctx 明显落后对手 | 先证伪路径归因(见下),再 op 单测扫 nsg | 非 tensor-API 默认改小一档 | op 级由落后转领先,e2e 短 pp 随之反超对手 |
| **短 ctx 残余差距** | 落地后短 ctx 仍落后 | **固定/流式分解** | 三条路全试,全负 ⇒ 收工 | 流式项已胜对手,残余是每 dispatch 的固定开销,折算到 e2e 上限很小 ⇒ **不追** |
| **FATC_QK_K32 + O_CT** | e2e A/B 判负「证伪」 | 改用 op 单测(kernel 占满被测时间) | 默认打开 | op 与 e2e 长 pp 双双为正——**一个真收益曾被 e2e 雪藏** |
| **SDPA_KVB**(证伪) | 「批量预发射在 FA-SG 上有效」 | op 单测配对 | 不采纳 | 长 ctx 判负:该 kernel 的 V 行读**本来已 hoist**,不阻塞 ⇒ 批量只多付寄存器 |
| **gateup ROW2_SPLITK**(证伪) | micro 明显为正 | 回 e2e 验证 | 不采纳 | e2e 为负 ⇒ 路径类开关必须认 e2e |
| **decode 预算对账**(§3.0) | e2e decode 落后对手,不知道该动谁 | 五组 linear + attention 逐组测,**Δ 求和对上 e2e** | 把投入从 linear 移开 | 两个已测大项都领先,缺口全在未测残差里 ⇒ **已测项不是杠杆** |
| **attention 假缺口**(证伪)(§2.2) | 对账里 attention 占缺口大半,像第二大杠杆 | 发现镜像**从不写 KV cache**,而 MNN 每层派发一次 copy kernel;用 **A/B 双 cache** 干净量出 append 成本 | 撤回该项,镜像补 `--kv-write` | append 是 O(1) 的每层常量;补齐后 attention 转为领先项,长 ctx 反超 ⇒ **一个「主要缺口」整个不存在** |
| **o_proj 剩余缺口**(§3.2/3.5/3.6) | 同形状下带宽品质因子低于对手 | 近零探针排除固定开销 → `MNN_SESSION_CPU_TRACE` 排除提交气泡 → 沿 oc 与 ic 双向扫 bytes-per-dispatch | **收工,不动 kernel** | 品质因子是 bytes-per-dispatch 的纯函数(两条扫描重合),机制是延迟暴露且**对重分组不变**;把 `ROW_2` 搬到 plain 路径判负已回退 |

### 附:诊断前先证伪"路径归因"

本轮起手时按文档以为热点在 `QK_QSPLIT`,**实际那条路已是死代码**
(`sDecodeFusedThresh = 2` ⇒ 单 pass SDPA 覆盖所有 kv≥2 的 decode,
`mSdpaSinglePass` 无条件清掉 `mDecodeQkSoftmax`)。
把 SDPA 路径关掉实测明显变慢 ⇒ 证明活跃路径确实是 SDPA。

**教训:动手前先用 env 开关证明你以为的热点真的在跑。文档会过期,`if` 条件不会。**
