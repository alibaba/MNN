# 用 op 单测 + 对手基准定位性能问题(诊断流程)

> **何时读**:e2e 比对手慢，但不知道慢在哪;或者已知某个 kernel 慢，想知道"还有多少空间、
> 该往哪个方向改、什么时候该收工"。
>
> **本文只讲流程**(怎么发现问题 → 怎么找到真瓶颈 → 怎么定方案 → 怎么收工)。
> 具体 kernel 手段见 [`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md) §2.5。

## 0. 核心洞察:为什么必须"最小单元 + 对手"两件一起上

两件事各解决一个独立的问题,缺一个流程就走不通:

| | 解决什么 | 缺了它会怎样 |
|---|---|---|
| **op 单测(最小单元)** | **信噪比**。目标 kernel 在单测里占 100% 被测时间,ΔK 满幅可见 | e2e 抖动 ±2~3%,占比 17% 的 kernel 收益被除以 6 ⇒ 真收益埋在噪声里,正负号由当天热漂移决定 |
| **对手基准** | **绝对标尺**。告诉你"这个数字是不是已经到硬件极限" | 只有自己的历史数字 ⇒ 「已优化 20%」和「还差 3 倍」长得一样,无法决定要不要继续投入 |

**代价已实际发生(两个方向都翻过车)**:

- 只看 e2e:`FATC_QK_K32` + `FATC_O_CT` 被 e2e A/B 判为「−6~−11%,证伪」并默认关闭,
  次日改用 op 单测复测,组合是 **+16.4%**,e2e 复现 pp4096 **+5.8%**。
  一个真收益被雪藏一天,还附带了一个错误的机制解释。
- 只看 op 单测:`gateup ROW2_SPLITK` micro **+17%**,e2e **−1.3%**(§2.1.10)。

⇒ 结论不是"哪个更可信",而是 **两个尺度分工不同**,见 §4 的分工表。

## 1. 第一步:造 op 单测(最小单元)

放在 `test/speed/*.cpp`,用 `MNNTestSuiteRegister` 注册成 `speed/Xxx`。**五条构造要求**:

1. **形状照抄真实模型,不要用整数漂亮的形状。** `LlmAttentionSpeed.cpp` 直接写死
   q=16 / kv=8 / head_dim=128 / 28 层,并用 `MNN_QWEN3_MODEL` 切 0.6B / 4B / 2B。
   形状决定走哪条 kernel 分支,漂亮形状会把你测到另一条路径上去。
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
> 只是数字变成 1/4 档(M5 seq4096:9.9 → 47 ms/layer)。跑之前先 grep 一眼 kernel active banner。

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
   **报 mean 还是 max**。2026-08-31 发现 `tmp/mlx_pg.py` 一直是全部保留 + 报 `max`,
   而 `llm_bench` 是 `for (i=0; i<nRepeat+1; ++i)` + `if (i>0)` 丢首轮 + 报 `mean ± std`
   ⇒ **拿对手的 max 比 MNN 的 mean,白送对手约一个 std**。当轮对手 rep0 在 12/12 格
   都是编译离群点(0.6B pp512 rep0=3892 vs rep1/2≈9000,差 2.3 倍),这个偏置足以
   凭空造出或抹掉一个 5% 结论。**写镜像驱动时先读对手 harness 的 repeat 循环,再写自己的。**

### 2.2 ⚠️ 惰性图陷阱(任何 lazy graph 框架)

**把 N 步攒成一张图再 eval,会改变 in-place donation 语义。**

给对手镜像补 append(照对手框架 KV cache 的 `update_and_fetch` 写
`k_cache[..., ctx:ctx+1, :] = k1`)后,开销**随 ctx 线性增长**:
ctx128 +0.0136 / ctx512 +0.0446 / ctx2048 +0.185 ms —— 即每步**整份 cache 重拷**,
比 MNN 的单 token 写贵 ~30×。

**根因是 harness,不是对手框架**:benchmark 把 `loop=32` 步串成一张惰性图,
32 个 `slice_update` 链式依赖 ⇒ 前一版本仍活在图里 ⇒ **buffer donation 失效,退化成整份拷贝**。
改成每步 eval(`--loop 1`)后 append 几乎免费(ctx128 0.0300 vs 0.0287)。

⇒ **要量"每步"成本就必须每步 eval**,并单独评估 sync 噪声(loop=1 时 sync 噪声会反过来主导)。

⚠️ **上面这条 `--loop 1` 的结论(「append 近乎免费 ~1.3 µs、MNN 的 copy dispatch 是真实
劣势」)已于 2026-08-31 修正为偏低**:loop=1 每步 ~200 µs 的 sync 会把两臂的差压扁。
反过来,"每个入队迭代各给一套 cache"虽然避开了 copy-on-write,却引入**冷足迹**惩罚
(~+1 µs/层且随 ctx 放大),会**高估**对手。

**两头都避开的做法 —— A/B 双 cache**:批量图里每层备两块 cache,**写 A、SDPA 只读 B**,
两臂都分配 A 和 B,唯一差别是那几次 slice 写。A 从不被读 ⇒ 不触发 copy-on-write;
批量 ⇒ 无 sync 主导;足迹相同 ⇒ 无冷热偏差。实测对手的 append
**+4.73 / +4.15 / +3.66 / +2.85 µs/层**(ctx 256/512/640/1024)——**绝对值恒定,
确认是 O(1) dispatch 开销**(2 次 slice 写 × ~2 µs)。校验:该法的 "SDPA only" 一列
与默认镜像逐格吻合(ctx512 23.21 vs 23.2)。
**结论翻转**:补上这一次 dispatch 后 ctx512 由「MNN 慢 17.2%」变成 **−0.6%(平手)**,
原先记在 attention 头上的 +83 µs e2e 缺口随之撤回。

**识别信号:开销随规模线性增长,而你以为它是 O(1)。** 见到这个立刻怀疑图结构,别急着下结论。

### 2.3 ⚠️ CSE guard 陷阱:不要扰动输入,要扰动参数

§1 第 2 条要求"一次 `onForward` 覆盖整个模型的层数",于是必须让 N 层不被
公共子表达式消除(CSE)折叠成 1 层。attention 单测用的写法是给每层输入乘一个不同的标量:

```cpp
auto qL = q * _Scalar<float>(1.0f + 0.001f * (float)l);   // ❌ 对带宽 bound 的 op 是灾难
```

**这个乘法本身就是一趟完整的读+写。** 给 RoPE 抄这一招时,q `[4096,32,128]` fp16 = 33.5 MB,
每层白付 ~67 MB 的访存——**和被测 op 的工作量同一量级**,直接把 1.10 ms/层测成
1.98 ms/层(43.5 vs 77.9 GB/s),并让 MNN 在对手对比表里凭空落后。

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

2026-08-31 Qwen3-0.6B decode(pg512,128,M4 base)的账:

| 项 | Δ(MNN − 对手) |
|---|---|
| linear(5 组求和) | **−73 µs(MNN 领先)** |
| attention ×28 | **−30 µs(MNN 领先)** |
| 其余未测部分(相减得到) | **+211 µs**(⚠️ 随轮次变,见下) |
| **合计** | **+108 µs = 该轮 e2e 实测缺口 ✓** |

结论直接反转了投入方向:**两个被测过的大项都是领先项** —— linear 里那个落后 11% 的
`o_proj` 全额追回也只值 1.3% e2e;而缺口 **100% 落在从没被测过的残差**里。
在此之前几轮 op 级优化都花在 linear 上。

⚠️ attention 那一格最初记的是 **+83 µs**,占缺口 77%,一度被当成第二大待优化项 ——
**其实是镜像少了一次 KV 写 dispatch**(§2.1 第 5 条 / §2.2)。修正后它变成领先 30 µs,
残差从 98 涨到 211。**对账表里任何一格的口径错误,都会等量地转移到「残差」那一格**,
所以每一格都要能独立复现,否则残差只是误差的垃圾桶。

⚠️ **211 µs 不是常量,别跨轮次搬用**。残差是「e2e 缺口 − 已测 op Δ」相减得到的,
所以它跟着**那一轮自己的 e2e 缺口**走:前一轮同轮测到的缺口是 99~108 µs ⇒ 残差 211;
后一轮全量矩阵(含反序复核)同一格的缺口只有 **25~52 µs** ⇒ 残差
**128~155 µs**。方向不变(两个已测组都领先、缺口全在未测部分),但量级的相对不确定度
达 30~40%,**必须按区间引用**。推论:**e2e 缺口与 op Δ 必须在同一轮里测**,
否则相减的是两轮各自的热漂移之和。

三条纪律:

- **口径要对齐到同一个 wall**。`llm_bench` 的 decode speed **不含采样**,对手侧的
  `generation_tps` 把 argmax 算在生成循环里 —— 直接对比会把缺口少报约 9 µs
  (2.1% vs 实际 2.30%)。**用两侧都含采样的 wall 值。**
- **op 预算的绝对值不可引用,只有差值有意义**。op bench 隔离测 best-of-N,
  真实 e2e 里算子背靠背争带宽 ⇒ 两侧预算都系统性偏低、残差都被抬高。
- **「相减得到的残差」要标注成未测项**,不要当成已归因。它里面混着 embedding gather、
  RoPE、residual add、final norm、采样,以及每 token 上百次 dispatch 的框架开销。

拿到「MNN 0.0276 vs 对手 0.0243」这种数字后,**它本身不能转成任何动作**。要先拆。

### 3.1 绝对标尺:先问"这是差距还是物理极限"

把品质因子和硬件峰值比。M4 Pro 峰值 ~273 GB/s:小 GEMV 只到 41-52%,lm_head 到 81%
⇒ 小 GEMV 有 headroom、lm_head 没有。**对手数字的第一用途是校准"可达值",不是排名。**

### 3.2 固定/流式分解:差距随规模变化时,第一件事

**拿两个干净的规模点拟合 `t = a + b·N`**,分出**每次固定开销 a** 和**流式速率 b**。
本次用 ctx128 / ctx4096 的 min:

| | 固定 a | 流式 b | 折算带宽 |
|---|---|---|---|
| MNN | **9.5 µs/layer** | 0.0404 µs/tok | 101 GB/s |
| 对手 | **2.5 µs/layer** | 0.0419 µs/tok | 98 GB/s |

⇒ **MNN 的内层流式已经略胜对手,残余差距 100% 在 a。** 这一步的价值是**直接砍掉一整类方向**:
所有内层 ILP / 访存合并 / 批量预发射的手段都不用试了(事后 `SDPA_KVB` 批量预发射实测
ctx4096 **−9~−10%**,正好验证)。

**自检**:把拟合回代到第三个规模点。对手预测 ctx512 = 24.0 vs 实测 24.3-24.8(很准);
MNN 预测 30.2 vs 实测 27.6-28.9(偏松 ⇒ 说明 ctx128 那个点噪声大,结论要打折)。
**回代不准就别用这个拟合下强结论。**

**⚠️ 能直接造出近零规模点时,不要外推截距。** 2026-08-31 定位 `o_proj` 时我从 4 个带噪点
拟合出「MNN per-dispatch 固定开销 1.5 µs vs 对手 0.46 µs」,据此断言 MNN 的固定开销更高
—— **符号是反的**。加两个近零权重探针(`diag_fixed_x28` / `diag_fixed_x112`,oc=64
⇒ 0.037 MB/层,流式项 ~0.4 µs)直接读:MNN **2.21 / 2.15 µs**、对手 **2.50 / 2.70**,
**MNN 的地板反而更低**。GEMV/GEMM 这类形状可以自由缩到近零,**几分钟的探针胜过任何拟合**;
`x112` 那一臂还顺带验了「固定开销随 dispatch 数线性」(否则是 per-onForward 常量,
诊断完全不同)。拟合只留给规模无法缩到零的场景(如 ctx)。

### 3.3 消融阶梯:差距在 kernel 内部时

逐类删工作读差值,得到分项成本表和**地板值**,然后**拿地板值和对手比,不要拿总时间比**。
完整方法(含防 DCE、四条纪律)见 `kernel-dev-and-optimize.md` §2.0「归因方法」。

本次用它给 copy dispatch 定价:临时 `MNN_METAL_ATTN_SKIP_COPY` 消融 ⇒ ctx512
−0.4/−4.8/−5.9%,约 **1.5 µs**,只解释 7 µs 缺口的 1/5。**我原本预测它是主因,证伪。**
⇒ 消融的价值一半在证伪自己的假设。

### 3.4 ⚠️ 不要写"host vs GPU 拆分"探针

在 timer 中间插一刀量 enqueue 时间:ctx4096 报 57-60 µs/layer "host",
实则是命令队列满之后的 **GPU 背压**,不是 CPU 开销。**这种探针无法分离 host 与 GPU。**
要区分只能用 GPU busy vs wall(§2.0)。

### 3.5 GPU busy vs wall 的具体做法:`-DMNN_SESSION_CPU_TRACE`

这是**唯一**能把真实 GPU 时间和提交气泡分开的工具,不是 env 而是**编译宏**
(`MetalBackend.hpp` 的 `struct MetalCpuTrace`,单独开一个 build 目录编)。它在
command buffer 的 completion handler 里累计 `GPUEndTime − GPUStartTime`,同时记
相邻 buffer 之间的 **gap**,进程退出时打印;另有 `encodeNs/encodeOps`、`commitNs`、
按站点分类的 `waitSiteNs[4]`。

读法:**GPU busy / forward 次数 vs 单次 wall**。2026-08-31 判 `o_proj` 到底是不是
GPU-bound:busy 100.35 ms / 255 forwards = **0.394 ms/forward**,对上 0.380~0.394 ms
的 wall ⇒ **计时区间内 GPU 是满的**,那 141 ms 的 gap 全落在计时区间外(建图/预热)。
⇒ 排除了 CPU encode 与提交气泡,剩下的只能是 kernel 本身。

⚠️ **gap 的总量会很大且容易误读**,一定要看它落在计时区间内还是外;
harness 侧可以另外把 enqueue 与 wait 拆开(`LlmLinearSpeed.cpp` 的
`enq=... (%)` 输出)当粗筛——**GPU-bound 的组 enqueue 应远小于总时间**
(那次是 38~41%,3 倍余量),但它只能当粗筛,定论仍靠本节的 busy vs wall。

### 3.6 不变量判据:先问"这个改动增加了总在途字节吗"

窄 dispatch 打不满带宽时,最容易想到的一整类改动是**重新分组同一份工作**
(拆 K、一个 simdgroup 多带几行、换 lane 划分)。这类改动**全都不该有收益**,因为

> 总在途字节数 =(参与的 simdgroup 数)×(每 simdgroup 的 load 数)
> **在同一份工作的任何重分组下都是不变量。**

`o_proj`/`down_proj` 上跨三个时间点试了三个变体,全部 ≤2% 且多数为负:
`SPLIT_K_2`(+1.8% / −3.3%)、`SPLIT_K_SHUFFLE`(−6% / −3%，2026-09-02 已删码)、
`ROW_2`(−1~4% / −3.5%，宏已改名 `GEMV_2OCQUAD_PER_SG`)。
事后看它们本来就该全负。**遇到这类问题先算这个不变量,没增加就别写 kernel。**

真正能动的只有两类:**提高每次 dispatch 的字节量**(融进邻居 op),或
**加宽每 lane 的 load**(同样的请求数搬更多字节)。

⚠️ 反过来也要小心:**别把一个机制从它成立的路径照搬到另一条路径**。
`ROW_2` 在**融合** GEMV 上确实赚 +2.3~2.8%,但那是因为融合体有可共享的 LN/input
前导;plain GEMV 里权重读就是全部流量,双行只是重分组 ⇒ 判负。
(详见 `kernel-dev-and-optimize.md` §2.1.6 的边界说明。)

### 3.7 ⚠️ `llm_bench --profile` 只能做「同 `-rep` 的两 pp 差分」

per-op 同步的 profiler 有三个叠加缺陷,**直接读它的绝对值或百分比一定错**:

1. **`Avg(ms)` / `Called times` 在 `loops=1` 下是累计和,不是均值。**
   见 `tools/cpp/Profiler.cpp`:`costTime` / `calledTimes` 逐次累加,打印时除以 `loops`。
2. **调用次数里混了 load-time warmup 的 decode step。** 4B 按 `-rep` 扫
   0/1/3 → Attention 828/864/936,**每个 prefill pass 只贡献 36 次(每层一次)**,
   剩下 792 次 = 22 pass × 36 层全是 seq=1 的 decode。**3/4 的调用次数不是你要测的东西。**
   识别信号:`Called times` **与 pp 无关**(pp1024 和 pp4096 都是 864/1752/1728)。
3. **每个被 profile 的 op 付一次 `wait(MAP_TENSOR_READ, true)` 的 flush 常数。**
   两点拟合 ≈**0.4 ms/call**;5880 次调用 ≈ 2.4 s,正好等于 profiled 10.34 s 与真实
   7.83 s 的差。⇒ **调用多、单次工作量小的 op(RoPE 864 次、BinaryOp 384 次)被系统性抬起来。**

**正确读法:固定同一个 `-rep`,拿两个 pp 相减。** baseline 调用次数与 decode 工作量
逐项完全相同,精确抵消,余下就是 `N_pass × (w(pp2) − w(pp1))`;再按 `w ∝ seq`
(GEMM/RoPE)或 `w ∝ seq²`(causal attention)外推到目标 seq。

2026-08-31 4B pp4096 实例:原始表报"非 GEMM 残差 5.06%"(RoPE 3.10 + BinaryOp 1.05 +
Raster 0.49 + While 0.28 + LayerNorm 0.15);差分还原后**真实残差 ≈1.4%**,
其中 RoPE 落在 0.79~1.17 ms/层,对照带宽下限(q 33.5 MB + k 8.4 MB,读+写 ≈84 MB
@~100 GB/s ⇒ ≈0.8 ms/层)已在最优附近。
**若照原始表去优化 RoPE,会花整轮时间追一个不存在的 3.10%。**

但**差分只够用来"排除",不够用来"定价"**:0.79~1.17 是 1.5x 宽的区间,而后来补的
`speed/LlmRoPE` 单测给出 1.104 ms/层(77.9 GB/s)——落在区间内,方向对,
**但只有单测能拿去和对手比,并回答"还剩 1.31x 单趟下限"这种可行动的问题**。

⇒ **本节所有数字都只是 MNN 自查:回答"时间有没有落在已知 op 之外"。**
`--profile` 的模型内 per-layer 值(FusedLinear 63.7 / Conv 33.2 / Attention 16.9)
**不能与对手框架的 op 数字并列成表**——两侧仪器不同,比出来的是仪器(见 §2 硬规则)。

⇒ 配合 §3.1:**先用带宽/峰值给这个 op 估一个上界,再决定要不要相信 profiler 的百分比;
真要定价就去补单测。**

## 4. 第四步:定位到旋钮后怎么标定(尺度分工)

| 改动类型 | 标定尺度 | 理由 |
|---|---|---|
| **路径开关**(走不走 kernel X / 阈值) | **e2e 为准** | 防"kernel 级快 / e2e 平"的伪收益;`ROW2_SPLITK` micro +17% / e2e −1.3% |
| **已确认在关键路径上的 kernel 内部参数**(nsg / tile / batch) | **先 op 单测标定,再 e2e 验证不回归** | e2e 分辨不出这一档;`DECODE_SDPA_NSG` 的 M4 默认值因为标在 e2e 上而**错了一个月** |

第二类的教训值得展开,它是本轮最大的单项收益来源:

> `MNN_METAL_DECODE_SDPA_NSG` 默认 32,来自一次 M5 的 e2e sweep(p2048 nsg32 179.3 >
> nsg16 167.9 > nsg8 166.4)。M4 分档直接继承。改用 op 单测(attention = 100% 被测时间)
> 复测,**nsg16 在每个 ctx 都胜**:ctx512 **+7.61/+11.55/+12.81%**、ctx4096 +1.9~4.0%、
> ctx8192 +1.8~3.5%。落地后 e2e p512 decode 203.4 → **206.7 t/s**,把对对手的 −1.6% 逆转成 +0.7%。

**旋钮标定的两条附加纪律**:

- **组合扫,不要单旋钮扫。** A 的停顿盖住 B 的收益时,单独扫 B 会把正向项判负。
  已判负的旋钮,同 kernel 有新正向项落地后**要复测**。
- **扫完要确认是尖峰还是平台。** nsg16 在 ctx128 上两侧邻居都差 ~20%(nsg8 −19~−25%、
  nsg32 −20~−28%、nsg4 −74~−78%)⇒ 尖峰,旋钮已耗尽,别再扫。
  顺带:nsg4 变慢的方向与"短 ctx 该少开 simdgroup 省归约"的直觉**相反**——
  grid 只有 `(1, B*H, 1)` = 16 个 threadgroup,短 ctx 的瓶颈是**并行度不足**。
  **旋钮的胜负方向本身是机制证据,要读它。**

## 5. 第五步:验证

1. **先过正确性 oracle,再信任何数字。** 错误 kernel 一样能变快。
   本次:5 个 attention op 用例(`op/attention` / `_c4` / `_c4_tail` / `_hd256` / `_prefill`)。
2. **⚠️ 改归约顺序的旋钮不能用 byte-compare 当判据。** nsg 改变跨 simdgroup 求和顺序 ⇒
   输出是另一个合法求和序,**不与旧版逐字节一致**。此时 oracle 只能是带容差的用例。
3. **e2e 的作用是"排除副作用",不是"发现收益"。** 本次同时验了 4096,128 无回归。
4. **无关失败要用零重建基线证明是既有问题。** 全 `op` 套跑到
   `op/unary/erfinvInt8` SIGKILL(exit 137)。**不要假设与自己无关**——
   用 `MNN_METAL_DECODE_SDPA_NSG=32` 把默认值改回旧值(**不重建**)重跑整套,
   两次 exit code、到达测试数、失败集合**逐字节相同** ⇒ 证明与本次改动无关(累积内存压力)。
   **env 覆盖是最廉价的 A/B 基线,凡是能用 env 回到旧行为的改动都该这么验。**

## 6. 第六步:收工判据

**不是"没想法了"就收工,要能说出剩余差距的机制、量级和它被什么挡住。**

本次收工时的账(可以照这个格式写):

- 流式 b 已胜对手(101 vs 98 GB/s)⇒ 内层无空间。
- 残余 = ~7 µs/layer 固定开销,其中 copy dispatch ~1.5 µs(已量),
  余 ~5.5 µs 是 per-layer host/dispatch。
- 要再动只剩结构改动(把 KV 写融进上游 RoPE kernel 省一次 dispatch,或 `MTLDispatchTypeConcurrent`),
  **e2e 上限 ~0.8%**(ctx512 attention 占 decode step ~16%,×最好情况)。
- e2e p512 MNN 206.7 已**领先**对手 205.7。
- ⇒ 按 §2.0 纪律,**op 级差距在 e2e 上兑现不出来就不追**,收工。

**归因于"编译器已经优化得很好了"一律视为未完成。**

**证伪也是成果**:中性/负收益要连方法一起写进 commit / PR,防止下一轮重复投入;
只有能提炼成**跨形状仍成立的判据**时才进 skill(见 [`SKILL.md`](./SKILL.md) 记录规则)。
本轮判负 4 条(`SDPA_KVB`、copy dispatch 定价、
host/GPU 探针不可用、对手 append 建模不可用)。

## 7. 度量卫生(不遵守则以上全部无效)

- **交替配对**(A/B、B/A),**丢弃每臂第一轮 warmup**,**轮内取 min**,
  **3 次同向且区间不重叠才算信号**。
- **串行 sweep 会被顺序/热度污染**:KVB 串行扫的绝对值几分钟内从 0.0277-0.0289
  漂到 0.0304-0.0333。第一轮 nsg 扫描总是先跑 nsg16 ⇒ 系统性偏向。
- **先证明开关真的传进去了**:`#ifdef` 里塞 `#error` 确认只有开开关时 pipeline 编译失败,
  再撤掉。「改了没效果」和「改的代码根本没进 shader」在数字上完全一样。
- **zsh 不对未加引号的参数展开分词**:多变量用 `${=var}` 或写 `A=1 B=2 ./bench` 前缀
  (SKILL.md 原则 7b)。注意 `${=var}` 是 **zsh 专属**,bash 里报 `bad substitution`。
- 手工替换 `libMNN.dylib` 后必须 `codesign -f -s -`,否则进程被 SIGKILL 而脚本只记下一个 0。
- **Bash 工具的工作目录会跨命令保持**,`cd build` 连着第二次 `cd build` 会失败并让
  `&&` 链静默跳过前一个测试 ⇒ **脚本里一律用绝对路径**。
- **跨引擎 e2e 的每一格都要反序复核**:2026-08-31 全量矩阵里三个 decode 负项做反序,
  **2/3 翻符号**。「MNN 先对手后」在小差距上有约 **1~1.5% 的系统性顺序偏置**,
  所以 **|Δ| < 5% 的格子,单向顺序的结论一律不成立**。反序只要多花一倍机时,
  却是「真缺口」与「顺序伪影」之间唯一的判据。
- **「反序复核后判真」有保质期,只绑定当时那个 build**:早先某一轮判真的两个 decode
  缺口(4B pp4096 / pp512),在后来的 nsg 分档修复之后全部翻正。**引用任何历史缺口
  之前先确认它测在哪个 commit 上**,否则会去追一个已经被别人修掉的问题。
- **每一轮都带一个「空臂对照」**:让至少一个 group 的代码在两臂**逐字节相同**(比如开关
  只作用于 kernel 的一条分支,其余 group 走别的 kernel),它读出的 Δ 就是这台机器**当下**
  的噪声底。2026-09-02 的 W16 probe 正是靠它同时抓到两件事:(a) 固定 A 先跑的顺序偏差
  (对照组 -3.65%,0/3「一致」);(b) 修掉顺序后噪声底仍有 **±2%**(对照组 0.00~-2.25%,
  逐对符号混杂)⇒ 同一轮里被测组只有超出这条线才算信号。**没有空臂时,"3/3 同向"完全
  可能是顺序伪影**——它看起来和真信号一模一样。
- **每条 arm 的 per-rep 原始输出必须落盘,只留均值等于放弃事后审计能力**:
  Qwen3.5-2B pg512 prefill 的方向翻转,靠的正是比对两轮的逐 rep 值才判出「是 MNN 臂
  被持续负载压低、而非回归」——只有均值时这条结论是「判不了」。

### 配对脚本模板

```zsh
#!/bin/zsh
# usage: pair.sh <ctx> <valA> <valB> <pairs>
ctx=$1; a=$2; b=$3; pairs=${4:-3}
cd /Users/jiuqi/AliNNPrivate/build || exit 1
run() {
  env MNN_METAL_DECODE_SDPA_NSG=$1 MNN_ATTN_CTX=$ctx MNN_ATTN_ROUNDS=3 \
    ./run_test.out speed/LlmAttnDecode 1 2 2>&1 \
  | awk '/^r[0-9] /{if($4>0&&(m==0||$4<m))m=$4} END{printf "%.5f", m}'
}
run $a >/dev/null; run $b >/dev/null      # 每臂丢一轮 warmup
for i in $(seq 1 $pairs); do
  # 真的交替:奇数对 B 先跑。固定 A 先跑实测会给「两臂代码逐字节相同」的
  # 对照组读出 -3.65%(2026-09-02 W16 probe),即把对内漂移全记在一条臂上。
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
| 对手镜像 | `~/mlx-bench/qwen3_*_mlx.py`(**不在仓库里**,只依赖对手框架) | 6 份,与对应 C++ 单测同形状同品质因子;e2e 侧另有 `~/mlx-bench/mlx_pg.py`。2026-09-03 前的记录写作 `test/speed/*_mlx.py` |
| attention decode | `./run_test.out speed/LlmAttnDecode 1 2` | env:`MNN_ATTN_CTX` / `MNN_ATTN_ROUNDS` / `MNN_QWEN3_MODEL` |
| attention decode 对手 | `qwen3_attn_decode_mlx.py --ctx N` | **与 MNN 比必须加 `--kv-write`**,否则镜像少一次 KV 写 dispatch(§2.2) |
| attention prefill | `./run_test.out speed/LlmAttn 1 2` | env:`MNN_ATTN_SEQ` / `MNN_ATTN_ROUNDS` |
| decode GEMV 诊断组 | `MNN_DECODE_GEMV_DIAG=1 ./run_test.out speed/LlmDecodeGemv 1 2` | 在四模型真实形状之外加诊断轴:`diag_i2k_oc*` / `diag_i*k_oc1024`(总字节固定 33 MB,分别沿 oc 与 ic 扫 bytes-per-dispatch)、`diag_fixed_x28/x112`(近零权重,直读 per-dispatch 固定开销并验线性)。`MNN_DECODE_GEMV_GROUP` 按子串筛组;对手侧 `qwen3_decode_linear_mlx.py --diag` 同形状 |
| e2e 全量矩阵 | `tmp/mnn_vs_mlx_matrix2.sh [precool] [out]` | 4 模型 × pg{512,2048,4096},每 arm 前 90s 预冷,MNN 先对手后;**每条 arm 的原始 stdout 落 `$out.log`**(事后审热漂移的唯一依据)。配套 `tmp/rev_check.sh` 做反序复核 —— **任何负项都必须过一遍它**(§7) |
| 对比 / 配对脚本 | `tmp/*.sh`(未跟踪的 scratch) | `attn_decode_vs_mlx.sh`、`nsg_pair.sh`、`mnn_vs_mlx_e2e.sh` 等 |

## 9. 案例总表(症状 → 诊断 → 动作 → 结果)

| 案例 | 症状 | 诊断手段 | 动作 | 结果 |
|---|---|---|---|---|
| **decode SDPA NSG**(v19) | op 级 ctx512 比对手慢 ~30% | 先证伪路径归因(见下),再 op 单测扫 nsg | 非 tensor-API 默认 32 → **16** | op ctx512 **+7.6~12.8%**;e2e p512 203.4 → **206.7 t/s**,反超对手 |
| **短 ctx 残余差距** | 落地后 ctx512 仍差 ~15% | **固定/流式分解** | 三条路全试,全负 ⇒ 收工 | 证明流式已胜对手,残余是 ~7 µs 固定开销,e2e 上限 0.8% ⇒ **不追** |
| **FATC_QK_K32 + O_CT** | e2e A/B 判 −6~−11%「证伪」 | 改用 op 单测(kernel 占 100%) | 默认打开 | op **+16.4%**,e2e pp4096 **+5.8%**——**一个真收益曾被 e2e 雪藏一天** |
| **SDPA_KVB**(证伪) | 「批量预发射在 FA-SG 上有效」 | op 单测配对 | 不采纳 | ctx4096 **−9~−10%**:该 kernel 的 V 行读**本来已 hoist**,不阻塞 ⇒ 批量只多付寄存器 |
| **gateup ROW2_SPLITK**(证伪) | micro **+17%** | 回 e2e 验证 | 不采纳 | e2e **−1.3%** ⇒ 路径类开关必须认 e2e |
| **decode 预算对账**(§3.0) | e2e decode 落后对手 2.3%,不知道该动谁 | 五组 linear + attention 逐组测,**Δ 求和对上 e2e** | 把投入从 linear 移开 | linear 合计 **−73 µs(领先)**、attention **−30(修正后)**、未测残差 **+211** = 108 µs ✓ ⇒ **两个已测项都领先,缺口全在未测部分** |
| **attention +83 µs(证伪)**(§2.2) | 对账里 attention 占缺口 77%,像第二大杠杆 | 发现镜像**从不写 KV cache**,而 MNN 每层派发一次 copy kernel;用 **A/B 双 cache** 干净量出 append 成本 | 撤回该项,镜像补 `--kv-write` | append **+2.9~4.7 µs/层**(O(1));ctx512 由 **+17.2% → −0.6%**,ctx640/1024 MNN 反超 **7%/18%** ⇒ **一个占 77% 的「主要缺口」整个不存在** |
| **o_proj 剩余 30 µs**(§3.2/3.5/3.6) | 同形状 89 vs 对手 99 GB/s | 近零探针排除固定开销 → `MNN_SESSION_CPU_TRACE` 排除提交气泡 → 沿 oc 与 ic 双向扫 bytes-per-dispatch | **收工,不动 kernel** | GB/s 是 bytes-per-dispatch 的纯函数(两条扫描重合),机制是延迟暴露且**对重分组不变**;`ROW_2` 搬到 plain 路径 **−1~4%(3/3)** 已回退 |

### 附:诊断前先证伪"路径归因"

本轮起手时按文档以为热点在 `QK_QSPLIT`,**实际那条路已是死代码**
(`sDecodeFusedThresh = 2` ⇒ 单 pass SDPA 覆盖所有 kv≥2 的 decode,
`mSdpaSinglePass` 无条件清掉 `mDecodeQkSoftmax`)。
用 `MNN_METAL_DECODE_SDPA=0` 关掉 SDPA:ctx512 从 0.0305 掉到 0.0520
⇒ 证明活跃路径确实是 SDPA。

**教训:动手前先用 env 开关证明你以为的热点真的在跑。文档会过期,`if` 条件不会。**
