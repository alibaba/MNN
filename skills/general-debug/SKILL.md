---
name: bugfix
description: MNN 各类正确性/回归 bug 的排查入口，按 bug 类别分册组织，本文件只做症状分流。分册：内存别名与生命周期（arena reuse、`MemChunk`、融合引入的别名竞争）、量化误差与导出侧权重损坏（低 bit 打包、导出分块、PyTorch MPS/CUDA 大张量静默错误）、fp16 表示能力不足（长序列复读、position 塌缩，以及「实时计算→预计算查表」重构的三类陷阱）、GPU shader 越界与 command buffer 故障、后端 kernel 隐式假设违反（causal mask、layout 约定）、持久化缓存误信（weight-mmap sync 自我污染、跨模型缓存复用）、逐 run 不同的非确定性（未初始化内存/堆垃圾依赖、多线程动态分发×异构 kernel）。用户报告 MNN 输出乱码/退化、单测或 golden 对不上、改动后回归、换后端结果不同、开某开关才错、结果每次跑都不一样时使用。
---

# MNN Bugfix 排查 Skill（入口）

> **触发**：MNN 中出现正确性 bug、单测/golden 对不上、回归；或做完改动（新 op、fusion、图 pass、
> 后端 kernel、量化导出）后行为异常。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

本文件**只做分流，不承载技术内容**：先用下面两张表决定该读哪一份分册（或该转去哪个 skill），
再整份读那一份。分册之间互不依赖，可独立阅读。

## 第一步：这个 bug 是不是本 skill 的

| 情况 | 去哪 |
|---|---|
| **QNN / NPU（高通 HTP）**：结果不对/精度差、报 `1002/6000/1003/6004`、`graphFinalize/graphExecute` 失败、`validateOpConfig failed`、某算子 QNN 不支持、LLM 在 NPU 上乱码 | [`qnn-debug`](../qnn-debug/SKILL.md)（QNN 两条执行路径、中间张量 dump、误差模式与算子约束速查），不要在这里从头排查 |
| **CPU 性能改动之后稳定地错在某一档**：只有 t4+ 错、只有超过某长度错、只有某条 precision/ISA 错，或不崩不报错只是模型输出质量变差 | [`cpu/optimize/bugfix.md`](../cpu/optimize/bugfix.md)（CPU 五层交界处的六类不一致 + 「改动面 → 必查条目」索引） |
| **崩溃**（SIGSEGV / abort / 真机 crash 日志） | [`crash-debug`](../crash-debug/SKILL.md) |
| 其余：跨后端通用的**正确性**方法论 | 本 skill，继续看第二步 |

**与 CPU 分册的判别口径**：那本收「稳定地错在某一档」，本 skill 收「同一输入每次跑都不一样」
（[`nondeterminism.md`](nondeterminism.md)）与「与层无关的框架级根因」。同一组合下结果不可复现，留在本 skill。

## 第二步：症状 → 分册

| 分册 | 典型症状（强判别信号加粗） | 旧编号 |
|---|---|---|
| [`memory-aliasing.md`](memory-aliasing.md) | 数值错乱、乱码 token、NaN，但指针地址都合法、单 op 单跑对；**一个后端错另一个后端对**；关掉某个新加的优化/融合就好；加 printf 或改 buffer size 就"好了"；**反复创建/销毁后物理内存按固定步长线性增长** | §1 |
| [`export-and-quant.md`](export-and-quant.md) | 低 bit（Q4）乱码而 Q8 正常；**所有推理后端一致地错**；torch 侧 `--test` 正常；只有大 vocab / 大模型触发 | §2 |
| [`fp16-range.md`](fp16-range.md) | 长 prompt 输出重复/漂移，短 prompt 正常；**fp32（`precision: high`）对、fp16 错**且所有 fp16 后端一致地错；**出错阈值恰是 2 的幂**（2048/4096） | §5 |
| [`gpu-oob.md`](gpu-oob.md) | `[METAL] command buffer error` 后**速度假快数百倍**；只在某 shape 阈值之上触发；关某条 kernel 路径 env 后消失；同一越界在别的模型上表现为静默数值损坏 | §6 |
| [`kernel-assumptions.md`](kernel-assumptions.md) | 某**类**模型（SWA / prefix LM / bidirectional）静默乱码，标准 causal LLM 正常；调一个"看起来无关"的性能开关就好 | §7 |
| [`stale-cache.md`](stale-cache.md) | 开 `use_mmap` 才乱码（单字符刷屏）；**App 内错、`llm_demo` 对**；清缓存目录或换 `tmp_path` 就好；换模型不换目录后乱码 | §8 |
| [`nondeterminism.md`](nondeterminism.md) | **同一输入每次跑结果都不一样**：§9 输出在正常/乱码间漂移、加无关代码就"好了"、只在某线程数触发、大片 ±65504 或 `!!!`、bisect 结论不稳定；§10 内容连贯但逐 run 不同（贪心解码仍分叉） | §9、§10 |

> **旧编号一列是给历史引用用的**：其他 skill 与复盘记录里写的「general-debug §9 / §10」等指的就是
> 这里对应的分册。分册内部的小节号（§1.3、§5.6、§9.4 …）保持不变，可以继续直接引用。
> 历史编号 §3（并发 / 线程竞争）与 §4（图优化回归）**没有独立分册**：并发类的两个已入库案例分别是
> [`memory-aliasing.md`](memory-aliasing.md) §1.6（GPU 单 dispatch 内 threadgroup 竞争，含逐 op commit 二分法）
> 与 [`nondeterminism.md`](nondeterminism.md) §10（CPU 动态分发 × 异构 kernel）。
>
> 尚未入库：**图优化回归**（某个 converter pass 之后跑错、disable 该 pass 就正常）。
> 首次复现时按下面的维护约定新开一份分册，编号从 **§11** 起——§3/§4 是历史空号，不再复用。

**症状横跨多类时**的推荐顺序：先做「所有后端是否一致地错」的分流（一致 → `export-and-quant`，
不一致 → `memory-aliasing`），再做「fp32 是否也错」（fp32 对 → `fp16-range`），
最后才怀疑 kernel 假设与缓存。**「每次跑都不一样」优先于一切**——非确定性会让前面所有对拍结论作废，
先去 `nondeterminism.md` 把它压成确定性问题。

## 通用排查原则（所有分册共用）

1. **先复现最小化**：定住能稳定复现的最小 case（最短 prompt、最小 shape、单线程），然后逐步添加变量。
2. **两向 A/B**：换后端、换开关、换编译选项、换线程数 —— 任一维度上"这边错那边对"都是强线索。
3. **别信"这块应该是独立的"，用直接观察去证明**：地址、值、时序都要用打印/断点去看，不要靠推理。
4. **修改先做假设，验证后再改代码**。改一版跑一版，避免"多改一起观察但不知道哪个生效"。
5. **改完记录**：如果本次 bug 有可复用的教训，追加到对应分册的"参考案例"；如果发现新的 bug 类别，
   新开一份分册并更新本文的分流表。
6. **导出/序列化崩溃先查 I/O 边界**：用回溯定位第一个文件写入点；converter 入口先创建并验证父目录，
   `fopen` 失败必须立即返回，禁止把空 `FILE*` 传给 `fwrite`。不要把这种晚发的 SIGSEGV 误判成模型或后端问题。
7. **区分建图与执行阶段的资源绑定**：对外部 backend API，先确认 graph tensor 创建时允许哪些字段，
   client buffer、memory handle 等运行时资源只在 API 要求的阶段绑定；在至少两个 SDK/SoC 组合上检查
   建图日志，不能以某个版本的宽容行为代替契约。

**对拍纪律**（每个分册都会踩）：hash 输出前剔除 `cost time` 等计时行；对拍前预热一次（首次运行
会重建 pipeline cache）；`llm_demo` 强制共享 `tmp/` 缓存目录，对拍前 `rm -rf tmp`；
确定性测试必须显式钉死 sampler（`"sampler_type":"greedy"`），默认 `mixed` 本身就是非确定采样。

## 目录结构

```
skills/general-debug/
├── SKILL.md                 ← 本文件，症状分流 + 通用原则
├── memory-aliasing.md       §1 内存别名 / 生命周期（arena reuse、融合别名竞争、fp32 当 oracle）
├── export-and-quant.md      §2 量化误差 / 导出侧权重损坏（量化 bisect、离线反量化比对）
├── fp16-range.md            §5 fp16 表示能力不足（值域改写、查表化重构的三类陷阱）
├── gpu-oob.md               §6 GPU shader 越界 / command buffer 故障
├── kernel-assumptions.md    §7 后端 kernel 隐式假设违反（causal mask、layout 约定）
├── stale-cache.md           §8 持久化缓存误信（weight-mmap sync 自我污染、跨模型污染）
└── nondeterminism.md        §9 未初始化内存 / 堆垃圾依赖 + §10 多线程数值非确定
```

## 维护约定

- 每份分册按同一骨架写：`触发` → `核心心法` → `相关背景`（可选）→ `排查流程`（Step 化）→
  `常见对照表：症状 → 优先怀疑` → `参考案例` → `相关文件索引`。
- 新增类别 = **新开一份分册**（不要往已有分册里塞进第二类根因），文件名用症状而非编号，
  并在本文的分流表与目录结构里各补一行；分册内部小节号沿用新分册的编号段。
- 参考案例写"排查路径 + 根因 + 修复 + 避坑要点"，其中**排除项与它们为什么误导**必须写，
  否则下一个人会重跑同样的死路。
- 结论上提到分册正文，过程与数字留在案例段；性能相关的过程数据不进本仓，
  归到对应后端 skill 的结论文档或外部台账，不要挤进本 skill。
- 复盘走 [`retrospective`](../retrospective/SKILL.md)。
