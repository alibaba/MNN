---
name: metal-optimize
description: MNN Metal 后端 op/kernel 开发与优化入口。索引各份 sub-doc：性能问题诊断流程（op 单测 + 对手基准定位真瓶颈，优化任务第一站）、kernel 开发规范与优化知识库（命名/写法/GEMV/GEMM/attention + 手段方法论）、算子融合全链路（导出图→converter→Metal 单 dispatch + 融合方法论）、运行时调度（fence/content-cache/H2D/replay + 调度方法论）、构建测试基线、env 开关注册表。根据当前任务选择性阅读对应 sub-doc。
---

# MNN Metal 优化 Skill（索引）

> **触发**：新增或修改 Metal kernel / shader / dispatcher；LLM decode/prefill 性能优化；
> 算子融合（导出侧声明 + 后端单 dispatch）；per-op profiling 定位瓶颈；跑 Metal LLM 测试或对拍。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## 使用方式

**先按任务类型定位到具体 sub-doc**，本文件只做索引和路由，不承载技术内容。

## 记录规则（默认不记录）

> **每次实验的结果不需要写进 skill。** skill 是给下一个人用的操作知识，不是实验流水账。绝大多数 A/B 跑完就完了，写在 commit message 和 PR 描述里即可。

**只有满足下列之一才值得写进 skill，且只写"可复用的那一句"，不搬数据：**

| 值得写 | 写到哪 | 写成什么 |
|---|---|---|
| 一条**跨形状/跨芯片仍然成立的判据**（例："coop-tensor 路径加宽 K 赚、加宽 N 亏，因为 destination capacity 只跟 N 走"）| 对应 sub-doc 的方法论节 | 一句原理 + 适用边界，不带百分比 |
| 一个**会让人重复踩的陷阱**（编译期宏四处同步、内存别名、`run_test.out` 用 precision 1 静默关掉 fp16 kernel、zsh 不分词导致开关没生效）| `kernel-dev-and-optimize.md` §1.6 或本文件"通用原则速览" | 症状 + 根因 + 规避写法 |
| 一个**新增/删除/改名的 env 开关** | `env-registry.md` | 只写名称 / 默认值 / 功能 / 生效条件 |
| 一条**新的路由或默认值**（哪种 shape 走哪条 kernel）| `kernel-dev-and-optimize.md` 路由速查 | 条件 → 走哪条 kernel |
| 一个**已证伪且别人很可能重试的方向** | 只在能提炼成**可复用判据**时，写进对应手段的「适用边界」 | "这类手段在 Z 条件下不成立，因为 Y"——**不留实验档案** |

**不要写进任何 skill 文件：**

- 单次 A/B 的百分比、ms、GB/s、tok/s、配对明细、扫点表、per-rep 方差；
- 标定过程叙事（先测什么后测什么、哪次翻案、哪次是漂移伪影）；
- 只在一台机器一个模型上成立的数字；
- 已回退的实验代码的实现细节。

**`kernel-dev-and-optimize.md` 有额外的硬性约束**：它记的是**每个手段为什么能赚**——赚在哪一类资源上（带宽 / 并行度 / 指令 / 同步）、什么条件下成立、有什么陷阱。**不写任何性能数字**（百分比、ms、GB/s、TFLOPS、tok/s、扫点表、配对明细），**不写日期 / commit hash / 具体机型与模型名**（设备与形状只以"类别"出现，如"tensor-coop 设备"、"GQA group 较大的模型"），**不写反例存档与标定叙事**。详见该文件开头的记录规则。

**`runtime-scheduling.md` 只记录优化方法与原理**：保留开销来源、适用条件、安全边界、代码入口与验证方式；**不写任何实测数据或实验叙事**，包括耗时、吞吐、占比、实测计数、A/B 结果及测试通过数量。详见该文件开头的记录规则。

**`env-registry.md` 有额外的硬性约束**：它只是"当前存在哪些开关、默认什么、干什么、什么条件下生效"的查询表。**任何环境变量带来的性能收益或损失，一律不许写进那个文件**——包括百分比、配对结果、转正/证伪结论、commit hash、日期、设备与模型名。详见该文件开头的约束清单。

## Sub-doc 结构

| 文件 | 何时阅读 | 内容 |
|---|---|---|
| **[`op-bench-and-diagnosis.md`](./op-bench-and-diagnosis.md)** | **比对手慢但不知道慢在哪**;或已知某 kernel 慢，想知道"还剩多少空间 / 往哪改 / 何时收工"——**任何优化任务的第一站** | 诊断流程:为什么必须「op 单测(信噪比)+ 对手基准(绝对标尺)」两件一起上;怎么造 op 单测(5 条构造要求、品质因子选 GB/s 还是 TFLOPS)；怎么造对手镜像(6 项可比性核对、**惰性图 donation 陷阱**)；三种诊断的选用(绝对标尺 / **固定-流式分解 `t=a+b·N`** / 消融阶梯)；**标定尺度分工表**(路径开关认 e2e、kernel 内部旋钮认 op 单测)；验证(改归约顺序不能 byte-compare、**env 覆盖当零重建基线**)；收工判据;度量卫生 + 配对脚本模板;现成资产清单;5 个案例总表 |
| **[`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md)** | 写或改任何 Metal kernel **之前**都先读第一部分；做 kernel 层性能优化时读第二部分 | **第一部分 开发规范**：核心原则、dispatcher 结构、shader 组织与 kernel 命名约定、**变量/宏名必须等于真实物理量**、编译期宏与 pipeline 缓存 key 的四处同步、Execution 骨架与注册、9 个通用陷阱（A–I）、packed weight 设计、修改流程、正确性验证、tensor API cooperative tensor 布局、Apple GPU 杠杆。<br>**第二部分 优化知识库（只写机制与原理，不含数字 / 日期 / 反例存档）**：优化总纲与 decode 画像、**kernel 内部消融阶梯（逐类删工作定位内部瓶颈 + 地板值 vs 对手）**、GEMV（deferred dequant / 双 SG / pre-scaling / split-K / `GEMV_2OCQUAD_PER_SG` / W2W3 / 短序列缺口）、GEMM（fused Q4 / M64 / in-shader 阈值）、Attention（causal-tri / FA / tensor-API tc / 单 pass SDPA / `SDPA_QH_PER_TG` / QSPLIT / GQA / 路由速查）、其他 kernel（RMSNorm 小 batch / LinearAttention scan 演进 / gated norm）、§2.5 kernel 手段方法论（原理/适用/陷阱/验证，选题与方案评审用） |
| **[`graph-fusion.md`](./graph-fusion.md)** | 要让多个算子合成一次 dispatch；改 `FusedLinear` / `GatedRMSNorm`；排查"融合没命中 / 融合后输出错" | 融合全链路：Python 导出期声明分组 → converter LN 吸收 → geometry 兜底拆分 → Metal `setupFusion` 装配 leader/follower。含内存别名铁律与 STATIC re-home、链式门控依赖、`GatedRMSNorm` 独立链路、已删除的后端图匹配历史、排查清单、§9 融合方法论（同构投影合并 / 打包 grid / 正确性门槛） |
| **[`runtime-scheduling.md`](./runtime-scheduling.md)** | 怀疑 decode 有 CPU 阻塞 / GPU 空泡；改 resize 时机、commit 节奏、H2D、Encode Replay | per-backend fence、content-cache、队内 H2D 上传、采样去框架化（host ArgMax/topK）、完成等待自旋、commit cadence、Encode Replay（安全模型 / attention 与 LinearAttention 接入 / KV 悬垂指针坑）、调度类改动的验证套路、§9 调度方法论（同步点治理 / 自动阈值 / 开关收敛纪律） |
| **[`build-and-test.md`](./build-and-test.md)** | 改完代码要 build / 跑测试 / 对拍 | cmake 编译命令、模型导出命令、性能测试命令、CPU/Metal 对拍 |
| **[`env-registry.md`](./env-registry.md)** | 查 / 新增 / 删除 Metal 相关环境变量开关 | env 集中登记，**只有名称 / 默认值 / 功能 / 生效条件四列**：attention prefill、attention decode、LayerNorm、decode GEMV、量化 GEMM、融合、调度、profiling。含新增开关规范。⛔ 禁止写入任何性能数字与实验叙事 |

## 快速任务→sub-doc 索引

| 想做的事 | 优先读 |
|---|---|
| 新加一个 Metal op / kernel（该叫什么名、写在哪、怎么注册）| `kernel-dev-and-optimize.md` §1.3 / §1.5 |
| 加一个编译期变体宏 | `kernel-dev-and-optimize.md` §1.4（**四处必须同步**）|
| 新加 quant bit / 改 dispatcher 路径 | `kernel-dev-and-optimize.md` §1.2 / §1.6 / §1.7 |
| 想知道 Metal 的坑（宏 alias / weight byte order / getDequantScale coef）| `kernel-dev-and-optimize.md` §1.6 |
| Apple GPU 优化杠杆选择（sg_matrix / sg_reduce / tensor API）| `kernel-dev-and-optimize.md` §1.11 / §1.10 |
| **比对手慢，但不知道慢在哪** | `op-bench-and-diagnosis.md`——**优化任务第一站**，含完整诊断流程 |
| **要新建一个 op 级 speed 单测 / 对手镜像** | `op-bench-and-diagnosis.md` §1 / §2（5 条构造要求 + 6 项可比性核对）|
| **差距随规模（ctx/seq）变化，不知道是固定开销还是流式速率** | `op-bench-and-diagnosis.md` §3.2 固定/流式分解 `t = a + b·N` |
| **旋钮该在 op 单测还是 e2e 上标定** | `op-bench-and-diagnosis.md` §4 尺度分工表 |
| **决定要不要投入某个优化方向（先看这条）** | `kernel-dev-and-optimize.md` §2.0——先自测 GPU busy vs wall；若已 GPU-bound 且 occupancy 受限，<5us 级 GPU 节省不兑现为 wall |
| **已知某 kernel 慢，但不知道它内部哪一段慢** | `kernel-dev-and-optimize.md` §2.0「kernel 内部消融阶梯」——逐类删工作读差值，得到分项成本表 + 地板值 |
| 优化选题 / 方案评审 / 复盘优化方向 | 方法论三节：`kernel-dev-and-optimize.md` §2.5、`graph-fusion.md` §9、`runtime-scheduling.md` §9（只讲原理/适用/陷阱/验证，不含数字）|
| GEMV 优化（decode 主战场）| `kernel-dev-and-optimize.md` §2.1 |
| GEMM / prefill 优化 | `kernel-dev-and-optimize.md` §2.2 |
| Attention 优化 / 想知道当前走哪条路径 | `kernel-dev-and-optimize.md` §2.3（§2.3.8 是路由速查）|
| 改 `prefill_flash_attn_tc`（coop-tensor prefill attention）| 机制见 `kernel-dev-and-optimize.md` §2.3.4；FATC 开关默认值与生效条件见 `env-registry.md` |
| LinearAttention（Qwen3.5 gated delta rule）| `kernel-dev-and-optimize.md` §2.4.2 / §2.4.3 |
| 算子融合：为什么没命中 / 融合后输出错 | `graph-fusion.md` §8 排查清单 |
| 新模型结构要加融合 | `graph-fusion.md` §1（导出侧声明）+ §4（后端装配）|
| 融合后输出逐次不同 | `graph-fusion.md` §4.3 内存别名 |
| decode 每 token 的 CPU 阻塞 / 同步开销 | `runtime-scheduling.md` |
| Encode Replay 相关（新 op 要不要接入 / 为什么被 ban）| `runtime-scheduling.md` §7 |
| cmake 编译选项 / 模型导出命令 | `build-and-test.md` |
| 查某个 env 开关的默认值和语义 | `env-registry.md` |

### 选题决策路径

```
瓶颈在哪？
├─ 还不知道 / 只知道"比对手慢"          → op-bench-and-diagnosis.md（造 op 单测 + 对手镜像，再做固定/流式分解）
├─ 每 token 固定开销大（dispatch 多 / CPU 段长） → 融合方法论 graph-fusion.md §9 / 调度方法论 runtime-scheduling.md §9
├─ 权重带宽未吃满（decode GEMV）               → kernel 方法论 §2.5.3/§2.5.4
├─ attention 随 KV 变慢                        → kernel 方法论 §2.5.4/§2.5.5/§2.5.8
├─ GPU→CPU 同步密集                            → runtime-scheduling.md §9.2/§9.1
├─ 知道是哪个 kernel，不知道它内部哪段慢       → 消融阶梯 kernel-dev-and-optimize.md §2.0
└─ 不确定                                      → 先 profile（§2.0 GPU busy vs wall）
```

## 通用原则速览（细节见 `kernel-dev-and-optimize.md`）

1. **shader 是嵌入的 C++ 字符串**（`R"metal(...)metal";` 在 `*Shader.hpp` 里），不是独立 `.metal` 文件；公共头靠**字符串拼接**共享。改完直接 make。
2. **变体用 `preprocessorMacros`**，不用 function constants。加宏必须同步改四处：shader `#ifdef`、pipeline 缓存 key、宏字典、`onResize` 的 grid/threadgroup。
3. **dispatcher 要先摸清**：一个 op 常有多条 kernel，扩之前先决定支持哪几条 + 让其他路径显式 fallback。
4. **Apple GPU ≠ Android**：M3/M4/M5 之间都不能互推，更不能推 Vulkan/OpenCL。设备分档看 `architecture.name` / `isSupportTensorApi()`。
5. **正确性 oracle 先于性能**：fp32（`precision: high`）bit-identical 是最强证据；fp16 greedy 对拍是次强。**token 级一致不等于 bit 级一致**。错误 kernel 一样能"变快"，对拍通过之前测出的收益都不可信。
6. **融合必查内存别名**：把前驱折进后继时，前驱的输入可能已被分配器复用为后继的输出。
7. **A/B 必须交替配对**：热态漂移能造出 3 倍虚假收益；profile build 的绝对数字是伪影。
7b. **zsh 不对未加引号的参数展开分词**。`run() { env $3 ./bench; }` 传 `"A=1 B=2"` 时，`env` 会把整串当成**一个**赋值（`A="1 B=2"`），第二个变量**静默丢失**——A/B 看起来"没差异"，实则探针根本没打开。本次就因此误判过一轮。多变量用 `${=3}` 强制分词，或直接写 `A=1 B=2 ./bench` 前缀。**任何"改动没效果"的结论，先确认开关真的传进去了。**
8. **先定位瓶颈段再动手**：GPU busy vs wall、kernel 计时 vs e2e、带宽兑现率是三种不同口径；优化必须对准真正的那一段，否则 kernel 变快 e2e 不动。
8b. **kernel 内部要归因就逐类删工作，别正向猜**：临时开关删掉一类工作（输出故意错），一路删到只剩 MMA/FMA，读相邻差值得到分项成本表和**地板值**。然后**拿地板值和对手比，不要拿总时间比**——对手 ≈ 你的地板 ⇒ 差距 100% 在重叠效率（该做批量预发射 / 提高计算:访存比）；对手 < 你的地板 ⇒ 差距在算法与工作量，改重叠白费力。见 `kernel-dev-and-optimize.md` §2.0。探针必须防 DCE（删一小段却省掉半个 kernel = 下游被整体消掉，不是发现）。
8c. **标定尺度按开关类型分**：**路径开关**（走不走 kernel X）认 e2e——防"kernel 级快 / e2e 平"的伪收益；**已确认在关键路径上的 kernel 内部参数**（nsg / tile / batch）反过来——**先用该 kernel 占 100% 的 op test 标定，再用 e2e 验证无回退**。否则会把 e2e 分辨不出的旋钮标错：`MNN_METAL_DECODE_SDPA_NSG` 的某一档默认值就因为标在 e2e 上而长期是错的——attention 在那个场景只占 e2e 一小片，e2e 根本分辨不出这个旋钮；改用 op test 一测就分出胜负。
8d. **对手基准要先做两件事:固定/流式分解 + 确认惰性图没改语义**。(a) 差距随规模变化时，先拿两个干净规模点拟合 `t = a + b·N` 分出**每次固定开销 a** 和**流式速率 b**——曾据此发现 MNN decode attention 的 b 已胜对手基准，差距全在 a，于是所有内层 ILP 方向都不用试了。(b) 惰性图框架里，**把 N 步攒成一张图再 eval 会改变 in-place donation 语义**：原地 slice 更新退化成整份拷贝，我据此把对手基准的 KV append 误判成 O(ctx)（比真实贵 ~30×）。量"每步"成本必须**每步 eval**。
9. **布局是全局不变量**：数据布局改动牵动所有读写方，必须原子落地，中间状态不可运行。
9b. **simdgroup MMA 的两个乘数都必须是 half**，只有累加器用 float。把任一乘数提升成 float 会掉到 float MMA 速率——fused attention 的 PV 曾因此慢 1.4×（见 kernel 方法论 §2.3.9）。
10. **证伪不建档，只留可复用判据**：负向/中性实验绝大多数跑完就完了（写在 commit / PR 里即可）。只有当它能提炼成一句**跨形状仍成立的判据**时才进 skill，且写成对应手段的「适用边界」——"这类手段在 Z 条件下不成立，因为 Y"。百分比、配对明细、翻案与标定叙事一律不进 skill（见上文"记录规则"）。
11. **名字必须等于真实物理量**：改任何后端算子 / kernel 代码时，**先核对沿途变量、成员、宏、env 名是否就是它实际装的那个量**（不只是自己新加的名字）。名字骗人不会报错——编译、对拍、单测全都通过，但基于它写的分档、阈值、门控会一起错。最贵的一次是把分档自变量写成"head 数"，而 kernel 真正的并行度是 dispatch 的 threadgroup 数（`batch*head_num/qh`），默认值换个模型就反向。名字与索引算术不符就改名（宏名 / 缓存 key / 宏字典三处同步）。见 `kernel-dev-and-optimize.md` §1.3「变量 / 字段 / 宏名必须等于真实物理量」。

## 相关 Skills

- `skills/general-debug/` — 正确性 bug / 回归诊断入口（按症状分册）；Metal 常用的两册是
  `general-debug/memory-aliasing.md`（§1 内存别名 / 生命周期，含融合引入的别名竞争）与
  `general-debug/gpu-oob.md`（§6 shader 越界 / command buffer 故障）
- `skills/opencl-optimize/`、`skills/vulkan-optimize/`、`skills/cpu/` — 其他后端
- `skills/support-new-llm/` — 新增 LLM 模型的完整流程
- `skills/test-ci/` — 单测 / 回归测试
