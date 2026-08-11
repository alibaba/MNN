# Metal 运行时调度与管线同步

> **读这份文档的时机**：decode 每 token 有可疑的 CPU 阻塞 / GPU 空泡；改 resize 时机、
> commit 节奏、H2D 上传、Encode Replay；想知道"CPU 侧优化还有没有空间"。
>
> **现状**：CPU-GPU 串行化已被逐项消除（fence → content-cache → 队内 H2D →
> 设备端采样 → replay）。剩余空间是否还值得投入，取决于具体机型 / 模型档，
> 先按 §0 的方法自测再决定。
>
> 相关文档：kernel 优化见 [`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md)；
> 算子融合见 [`graph-fusion.md`](./graph-fusion.md)；env 开关见 [`env-registry.md`](./env-registry.md)。

---

## 0. 先测 GPU busy vs wall，再决定要不要投入

**方法**：`-DMNN_METAL_OP_PROFILE=ON` 单独 build，量 GPU busy/token，与**生产 build**
（非 profile build）同期的 wall/token 对比。GPU busy 逼近 wall ⇒ 已 GPU-bound，
调度层没得挖；差距大 ⇒ 还有 CPU-GPU 串行化可消。

下表是**一次样本**（0.6B / M4 Pro，p512 稳态 60 forward 平均），**不同机型 / 模型档结论不同，必须自测**：

| 指标 | 实测 |
|---|---|
| GPU dispatch / token | 266 |
| **GPU busy / token** | **2950 us** |
| 生产 wall / token（同期 341-348 t/s）| **2874-2933 us** |

该样本上 GPU busy ≈ 生产 wall ⇒ 这台机器上的这个模型档已基本 GPU-bound。
完整画像（GEMV 占 70%、各 dispatch 的带宽利用率）见
[`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md) §2.0。

**一旦自测确认 GPU-bound**，两条投入判据随之成立：

1. **<5us 级的 GPU 节省不再兑现为 wall**（该样本上砍 28 个 dispatch 只有 +0.7%；
   去掉 per-token 同步 ≈0；`COMMIT_NUM=50` 让 busy 少 42ms 而 wall 不动）。
2. wall 收益只剩两类：**CPU-GPU 串行化消除**（本文档 §1/§2/§3 属于这类）
   与**大体量 GPU 节省**（kernel 层 / 融合层）。

⚠️ **早期"单流 GPU 利用率仅 ~60%""空泡 ~29%"的解读是错的**：
双实例并发 1.41× 不代表有 29% 空闲时间，而是**单实例 kernel 填不满 GPU（occupancy 不足）**，
两实例共同调度才喂满机器——是 occupancy 效应，不是 idle-gap 效应。
profile build 内测得的 17% "idle" 是测量伪影。

⚠️ **按模型档分别评估**：4B 及以上不受管线约束（GEMV 占 67%、GPU busy 逼近 wall，Sync 仅 9.5%），
GPU 优化直接兑现；小模型才有管线问题。

---

## 1. `onResizeBegin` per-backend fence（默认开）

发现 `9356700fb` → 落地 `7f186691a`。

**根因链**（用 resize trace 探针定位）：steady decode 时大图输入既不脏也不重排；
每 token 强制 resize 的其实是那个 **1-op 的 logits `StridedSlice` 子模块**
（`mUseContentInputs=true`）。代价不在 resize 本身，而在
`MetalBackend::onResizeBegin` 的**无条件全队列 wait 排空**——
它每 token 在中段把主图的 GPU 工作与 lm_head 强行串行化。
这才是 60% GPU 利用率、~1.2ms/token "resize" 段的真正机制。

**修复**：改成 `waitOwnInflight()`，只等本 backend 自己最后一次 commit
（`mLastOwnCommandBuffer`）。语义正确性论证：allocator reset 只需要本 backend
自己的在飞工作完成，别的 backend 的工作与本 allocator 无关。

**实测**：0.6B decode **+4% 稳定**（greedy 逐字一致；fence 与 `none` 等价，
说明收益全额保留）；4B tg128 持平；pp512 微升。

**开关**：`MNN_METAL_RESIZE_WAIT=local`（默认，隐式）/ `global`（旧行为，回退用）/
`none`（完全跳过，实验，不安全）。

> ⚠️ **方法论教训**：本机 tg128 存在**热态双峰**（~280 段 vs ~320 段），跨时段 A/B 完全污染。
> 先前 NOWAIT 测出的 "+14%" 实为跨热态高估，真实只有 +4%。
> 任何管线类优化都必须**交替配对**测。

---

## 2. Content-cache：内容依赖子模块的 resize 缓存（Core 层，默认开）

`1ef0ace93`。这是 Core 层改动，**全后端受益**。

**机制**：logits-slice 子模块的 shape 依赖控制张量的**内容**（`logits_index`，
decode 期间恒为 -1），所以每 token 都要全量 resize。
`StaticModule::_resize` 对 content-for-shape 输入（整型控制张量）做**内容字节比对缓存**——
内容未变就跳过 resize（每 token 一次 4 字节 `memcmp`）。

安全边界：浮点输入（embeds / mask）不参与比对；>4KB 或非 host 可读的输入自动回退 always-resize。

**实测**：resize 段 **317ms → 6ms**（re-encode 131 → 6 次，只剩真实的 prefill↔decode 转换点）；
tg128 +1%（fence 已经吃掉大部分 wall 收益，本项的价值是把 CPU 侧 resize 成本**结构性归零**）；
4B 77.3（当时新高）；386/386 单测。`MNN_LLM_CONTENT_RESIZE_ALWAYS=1` 回退。

**决策记录：为什么不在导出端做 shape-static 根治**
`logits_index` 是有意的单图多模态设计（all-logits=0 / last-logit=-1 / spec-decode 变长）。
根治要么拆图要么双输出，而 hidden-states 切片放在 lm_head **之前**的目的正是省掉
all-token 的 lm_head 计算——双输出与这个目的自相矛盾。
content-cache 之后残余价值 ≈ 0（16us/call × 6 次合法转换点），**不单独立项**。

---

## 3. 队内 H2D 上传（staging ring + queue-ordered blit，默认开）

`757610526`。

**机制**：每 token 的输入上传从"drain 全 GPU + 直写"改为
**staging ring 槽位（命令缓冲租约复用）+ 队内 blit**。
队列顺序本身就保证了读写安全，per-token drain 归零。

**实测**：0.6B decode **+5.2%**（4 轮配对）、3.5-2B +1.6%、4B +1%，pp512 无回归，388/388 单测。
`MNN_METAL_H2D_QUEUED=0` 回退。

> 这一项和 §1 是同一类：**把"为了安全而 drain"换成"靠队列顺序保证安全"**。
> 遇到任何 `waitUntilCompleted` 都先问一句：这个等待到底在保护什么？能不能用队内顺序替代？

---

## 4. 设备端采样（ArgMax / TopKV2）

`7b2c8bfcd8`（`transformers/llm/engine/src/sampler.cpp`）。

**动机**：CPU trace 显示 `wait[copyD2H]` 2.7ms/token **全部**来自 logits 回读
（Qwen3 vocab ~600KB/token）。

- **greedy**：`Express::_ArgMax(logits, -1)` 在设备端算 argmax，跨设备边界只回读 **4 字节 index**。
  `MetalArgMax` 的 first-max tie-break 与原来的 CPU 循环一致。
- **mixed + topK 前置**：`_TopKV2` 设备端取 top-k values/indices，
  后续 pipeline 步骤在 k 大小的子集上跑（`SamplerState.is_subset`）。
  **精确等价的前提**：topK 必须是第一个有效过滤步 ⇒ 要求 `logit_bias` / `banned_tokens` 为空，
  且 topK 在 `mixedSamplers` 首位（或首位是 no-op penalty，即所有 penalty 系数为默认值）。
- 不满足前提时自动回退整份回读，语义不变。

⚠️ **`_TopKV2` 的 tie-break 未定义**——不能用它替代需要确定性 tie-break 的场景（`_ArgMax` 可以）。

---

## 5. Commit cadence

**当前状态**：由调用方的 `tuning()` 或 `MNN_METAL_COMMIT_NUM=N` 决定。

**标定结论**：默认 N=10 已最优。N=2 **-7%**，N=999 **-6%**；
补扫 20/30/50 全中性（MLX 用的 50-op 批次对 MNN 没有增益）。

**已移除的自动调优**（`ca8496a648`）：曾在 `llm.cpp` load 末尾对 Metal 跑
`tuning(OP_ENCODER_NUMBER, {10,20,40,80})`。**它从未生效过**——
调用点在 `mContext->status = RUNNING` 之前，`tuning()` 开头的 `CHECK_LLM_RUNNING`
判定 NOT_LOADED(-1) 直接 return，只留下一条 `[Error]: LLM in error state. Status: -1` 日志。
而且 llm_bench / llm_demo 本来就在 load 之后显式调 `tuning()`（候选集还更全），故直接删除。

> **值得记住的教训**：一段"看起来在调优"的代码，如果它的日志里有 Error 却没人看，
> 它可能从来没跑过。加自动调优时先确认调用点在状态机的正确位置。

---

## 6. Encode Replay（稳定 shape 前向录制重放，默认开）

`94a73ab19a`（基建）+ `405beb8aa4`（attention 接入）。
代码在 `source/backend/metal/MetalReplay.hpp` / `.mm`。

### 6.1 基本机制

当一个 `MetalExecution` 的输入/输出**设备地址**在连续调用之间保持一致时，
录制它的 encode（pipeline / buffer 绑定 / dispatch grid），
后续调用直接重放捕获的命令列表，**跳过 `onEncode` 的全部 CPU 逻辑**。

### 6.2 安全模型（这是本机制的核心）

- **每次重放前重新校验**：所有 tensor-backed 的绑定都对 tensor **当前**的 buffer+offset
  重新比对（`metalReplayValidate`）；任何不匹配就丢弃录制、退回正常 encode，并允许重录。
  KV-cache 扩容与 allocator 重排由此兜住。
- **豁免机制**：encode 依赖 per-token CPU 状态的 op 通过 `canRecordEncode()` 排除。
- **防抖**：连续 8 次重放失败的录制会被 **ban**，防止某个系统性 bail 的 hook 反复重录。
- **编译期**：`MNN_METAL_OP_PROFILE` build 下禁用（subpass encoder 切换无法建模）。

### 6.3 Attention 接入（最重的一条分支路径）

decode attention 的 encode 是最复杂的（路径决策 + shader key 选择 + branchy dispatch），
改为在稳定 token 上重放 + per-token 打补丁：

- 决策提取为 `_computePathFlags()`；参数写入拆成
  `_writeCopyParam` / `_writeQKVParam` / `_writeSoftmaxParam`，`onEncode` 与
  `onReplayUpdate` hook **共用同一份代码**（避免两条路径漂移）。
- `_pathSignature()` 指纹化所有会影响 kernel 变体 / 事件布局的结构 flag
  （含 kv≤128 的 SHORT_KV_128 变体、kv 相关的 `mQkvSimdReduce` 翻转）；变化即 bail + 重录。
- `onReplayUpdate` 重写 copy/QKV/softmax 的参数 buffer，并补丁录制事件里 kv 相关的
  grid/bytes（fused qk_softmax 的 local size、short-seq qk 的 grid 深度等），
  然后让 `pastLength` **恰好前进一次**。

⚠️ **KV-cache 悬垂指针坑**：KV 扩容会销毁旧的 cache tensor，
录制的绑定里就留下了**悬垂 tensor 指针**。
所以 `onReplayUpdate` 必须在 `metalReplayValidate` **之前**先比较 K/V tensor 的
**指针身份（绝不解引用）**；scale buffer 同理。

### 6.4 LinearAttention 接入

`canRecordEncode()` 从恒 false 改为 `seqLen==1 && gated_delta_rule`。

⚠️ 关键障碍：`Pipeline.cpp` 对 LinearAttention **每 token 强制 re-resize**，
`onResize` 每次重建 `mConvOut` ⇒ 录制绑定的是悬垂 `Tensor*`
（症状：每 token invalidate → 最后被 ban）。
修法：shape 不变时**保留 Tensor 对象**，外加一个 resize-generation 守卫在
`onReplayUpdate` 里 bail——**必须先于** `metalReplayEmit` 的解引用。

收益中性（decode 已 GPU-bound），价值在于消除一个结构性豁免。

### 6.5 实测与开关

| 项 | 数据 |
|---|---|
| 基建本身 | p512 +0.5%（价值主要在基建）|
| attention replay | p12 **+1.4%** / p2048 **+2.0%** |
| 正确性 | 0.6B greedy 238 token byte-identical；失效只发生在 kv=65/129 翻转 + 64-chunk KV 扩容，**0 ban** |

`MNN_METAL_DISABLE_REPLAY=1` 回退；`MNN_METAL_REPLAY_DEBUG=1` 打印 record / ban / invalidate 日志。

---

## 7. 调度类改动的验证套路

1. **先确认瓶颈真在 CPU 侧**。用 CPU trace（编译宏 `-DMNN_SESSION_CPU_TRACE`）看
   encode / commit / wait 的分段，别凭直觉。生产模式实测 op encode 只有 **0.92us/op**
   （0.16ms/token，5%）——**encode 很便宜，ICB 复用是死路**。
2. **交替配对 A/B**，绝不跨时段比。热态双峰能造出 3 倍虚假收益（§1 的教训）。
3. **greedy 逐字对拍**是调度类改动的黄金 oracle：调度改动不应该改任何数值，
   出现 token 差异就是 bug（不像 kernel 层融合那样有 fp16 rounding 借口）。
4. **回退开关必须留**：本文档每一项都有对应 env，这不是可选项——
   调度 bug 往往只在特定设备 / 特定 kv 长度上出现，现场没法重编译。
5. **不要用 profile build 的绝对数字**：counter sample buffer attachment 让 CPU op encode
   从 ~0.92us 涨到 4-20us（20×），制造出的 GPU idle 是**测量伪影**。
   profile build 只用来看**相对占比**。
