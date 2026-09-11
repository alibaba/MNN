# Metal 运行时调度与管线同步

> **读这份文档的时机**：decode 每 token 有可疑的 CPU 阻塞 / GPU 空泡；改 resize 时机、
> commit 节奏、H2D 上传、Encode Replay；判断 CPU 侧优化是否还有空间。
>
> **关注点**：per-backend fence、content-cache、队内 H2D、采样去框架化、完成等待和 encode replay。
> 先按 §0 定位关键路径，再选择优化手段，不按机型或模型规模直接下结论。
>
> 相关文档：kernel 优化见 [`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md)；
> 算子融合见 [`graph-fusion.md`](./graph-fusion.md)；env 开关见 [`env-registry.md`](./env-registry.md)。

> ## 本文件的记录规则
>
> 1. **只写优化方法与原理**：说明开销从哪里来、如何减少、适用条件、安全边界和验证方式，
>    保留对应源码文件与符号，方便定位实现。
> 2. **不写任何实测数据**：耗时、吞吐、百分比、占比、实测计数、A/B 表、方差、测试通过数量
>    与具体对拍结果一律不收录。代码中的索引、条件和安全上限不是实测数据，可按需解释。
> 3. **不写实验叙事**：日期、commit 溯源、机型与模型的跑分记录、标定过程和证伪流水账不收录。
>    失败经验只提炼为可复用的适用边界与陷阱，不保留某次实验的结论。
> 4. **验证方法要留**：可以说明怎样测量、怎样对拍，但不在这里记录测量结果。
>    环境变量的默认值与完整语义统一查 [`env-registry.md`](./env-registry.md)。

---

## 0. 先测 GPU busy vs wall，再决定要不要投入

**方法**：用 `-DMNN_METAL_OP_PROFILE=ON` 的独立构建定位 GPU 工作分布，
以生产构建的完整 token 周期为端到端基准，结合 CPU/GPU 时间线判断等待与重叠关系。
两边必须使用同一工作负载与计时范围；profile 构建会改变同步和编码成本，
**不能直接把两种构建的计时相减当作空泡**。

- **GPU 持续忙、CPU 编码被覆盖**：优先减少关键路径上的 GPU 工作，kernel 方法见
  [`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md) §2.0。
- **GPU 等待 CPU 提交**：检查 resize、encode、commit 与上传路径，定位哪里打断了流水。
- **GPU 已完成、CPU 仍未恢复**：检查完成通知与线程唤醒，而不是继续压缩 kernel 时间。

**投入判据**：只优化处于关键路径、或阻止 CPU/GPU 重叠的工作。
局部计时下降不保证 wall 下降；必须用生产构建的端到端配对验证是否兑现。

**不要混淆 occupancy 与 idle gap**：双实例并发吞吐增加，可能是多个实例共同填满了
单实例 kernel 未用足的执行资源，并不能据此反推单实例的空闲时间。
是否有调度空泡要看时间线，不能从并发加速比推导。

---

## 1. `onResizeBegin` per-backend fence

**开销来源**：内容依赖的 logits `StridedSlice` 子模块可能在 decode 中触发 resize。
如果 `MetalBackend::onResizeBegin` 无条件等待全队列排空，
一个子模块的内存管理就会把其他 backend 已提交的工作也串行化。

**方法**：在 `source/backend/metal/MetalBackend.mm` 中使用 `waitOwnInflight()`，
只等待本 backend 最近提交的 `mLastOwnCommandBuffer`。
allocator reset 需要保护的是本 backend 的在飞工作，不应扩大成无条件全队列等待。

**安全边界**：先确认被重置 allocator 管理哪些资源、哪些命令仍引用它们。
不能把局部 fence 简化为完全不等待；跳过等待可能让在飞命令访问已复用的内存。

**验证**：通过 resize trace 确认等待由哪个模块触发；检查 allocator 重置前的依赖，
再用交替 A/B 和输出对拍验证。不要跨时段比较不同热态的结果。
`MNN_METAL_RESIZE_WAIT` 可用于路径对照，取值与安全限制见 env 注册表。

---

## 2. Content-cache：内容依赖子模块的 resize 缓存

**机制**：logits-slice 的 shape 依赖控制张量 `logits_index` 的内容，而不只是输入 shape。
在 `express/module/StaticModule.cpp` 的 `StaticModule::_resize` 中，
对 content-for-shape 的整型控制输入缓存其内容字节；内容未变时跳过重复 resize。
这项机制位于 Core 层，不限于 Metal。

**安全边界**：

- 只比对参与 shape 计算的控制输入，不扫描 embeds / mask 等浮点数据。
- 输入必须 host 可读，且大小不超过内容缓存上限；否则继续按内容依赖执行 resize。
- 内容变化时必须重新 resize，覆盖 prefill/decode 切换、all-logits 与变长切片。

**为什么不简单固定导出 shape**：`logits_index` 承载 all-logits、last-logit 和
变长 decode 等语义；把 hidden-states 切片放在 lm_head 之前，是为了避免计算不需要的 logits。
拆图或双输出会改变这个取舍，不能只为消除 resize 而破坏单图多种调用方式。

**验证**：分别覆盖控制值不变、控制值变化与不可缓存输入；对照正常 resize 路径检查输出。
`MNN_LLM_CONTENT_RESIZE_ALWAYS` 可用于强制 resize 对照。

---

## 3. 队内 H2D 上传：staging ring + queue-ordered blit

**开销来源**：上传输入前先排空 GPU，再由 CPU 直写目标 buffer，会破坏 CPU/GPU 重叠。

**方法**：在 `source/backend/metal/MetalBackend.mm` 的上传路径中，
先写 staging ring 槽位，再把 blit 与消费该输入的计算按依赖顺序提交到队列。
槽位以命令缓冲完成状态管理复用，避免 CPU 提前覆写仍被 blit 读取的数据。

**安全边界**：队列顺序只保护队内有序操作；CPU 写 staging 内存还必须服从槽位租约。
不能在对应命令完成前复用槽位，也不能假设跨队列工作自动有序。

**验证**：覆盖连续上传、槽位循环复用与长短命令缓冲交替，检查输入和输出一致性；
通过时间线确认上传没有引入新的全队列 drain。`MNN_METAL_H2D_QUEUED` 用于路径对照。

> 遇到 `waitUntilCompleted`，先问它保护哪块资源、哪段读写依赖，
> 再判断能否用队内顺序与资源租约替代，不能只因为等待耗时就删除。

---

## 4. 采样路径去框架化

**先确认 executor 归属**：不能因为主图使用 Metal，就认为 sampler 的 Express 表达式也跑在 GPU。
`transformers/llm/engine/src/llm.cpp` 的 executor 配置与
`transformers/llm/engine/src/sampler.cpp` 的 logits 读取路径，是判断执行位置和同步边界的入口。

**greedy**：对已经 host 可读的 logits，使用原生 SIMD 循环，避免为一次归约创建临时 expr 会话。
当前 host 路径采用两段式 first-max：先求最大值，再找首个相等元素的下标，
保持与标量 first-max 相同的 tie-break。GPU argmax 是否适用，要看 logits 是否仍驻留设备、
能否减少同步；若 logits 已在 host，重新提交 GPU 工作可能只增加 dispatch 与等待。

**mixed + topK 前置**：`topKSubset` 使用 host 阈值扫描筛出候选子集，
通过 `SamplerState.is_subset` 保留原 token 下标映射，后续 temperature/topP 只处理子集。
前置的等价条件是 topK 为第一个有效过滤步骤：`logit_bias` / `banned_tokens` 为空，
且 topK 位于 `mixedSamplers` 首位，或前面的 penalty 是 no-op；否则不能提前裁剪。

**验证**：greedy 覆盖相等最大值、首尾下标与向量尾部；topK 覆盖边界相等值、下标映射
和不满足前置条件的回退路径。不要用未约定 tie-break 的 `_TopKV2` 充当确定性参照。

**计时盲区**：`transformers/llm/engine/src/speculative_decoding/generate.cpp` 的 decode 计时若在 `sample()`
之后启动，`decode_us` 就不包含采样成本。评估采样必须测完整 token 周期
（sample 入口到下一次 sample 入口）或整体 wall。

---

## 5. 完成等待自旋化：轮询状态代替阻塞唤醒

**机制**：CPU 采样必须等 logits 可读，等待本身不一定能消除；
但 `waitUntilCompleted` 的完成通知和线程唤醒，会推迟 CPU 观察到完成的时刻。
`source/backend/metal/MetalBackend.mm` 在等待路径中轮询 `buffer.status`，
期间调用 `sched_yield`；达到自旋时长上限后回退阻塞等待，避免长命令缓冲持续忙等。
`MNN_METAL_SPIN_WAIT` 用于与阻塞等待路径对照。

**适用边界与代价**：

- 自旋只减少完成后的感知延迟，不减少 GPU 工作量，也不绕过数据依赖。
- 等待期间消耗 CPU 调度资源；必须结合功耗与其他线程的竞争评估，不能只看延迟。
- 等待长度与完成后的唤醒成本不是同一个量，不能只凭平均等待长度决定是否自旋。
- 长等待必须有阻塞回退，完成状态和错误状态仍需正常处理。

**不要直接删除同步点**：它还可能承担内存复用的纪元边界。
若 CPU 持续提交而旧 buffer 仍在飞，动态池无法复用内存，分配压力上升；
tensor 地址随之变化又会使 encode replay 失效。
移除等待前必须同时证明资源回收、地址稳定和 CPU 读取结果的依赖仍然成立。

**预测式睡眠的陷阱**：先睡眠、再在预计完成点前自旋，依赖准确的定时与估计器。
系统可能合并短睡眠而越过完成点；若把睡过头的观测继续喂给 EWMA，
就会出现“观测等待变长 → 下次睡得更久”的正反馈。
估计器必须区分 GPU 完成时间与线程恢复时间，不能把自身引入的延迟当成 GPU 工作量。

**验证**：同时检查完整周期、尾延迟、逐轮离散程度与 CPU 占用，不能只比较均值。
覆盖短 decode、长 prefill、连续等待和错误完成状态，并与阻塞路径对拍。

> 与 §3 的区别：§3 用顺序依赖替代不必要的等待；本节降低必要等待的唤醒成本。
> 先问能不能不等，再问必须等待时能否更快观察到完成。

---

## 6. Commit cadence

**原理**：提交太频繁会增加 CPU 编码、提交与命令缓冲管理成本；
攒得太久又可能推迟 GPU 开始执行，减少 CPU/GPU 重叠。
目标不是最少的 command buffer，而是关键路径上的供给连续、依赖正确。

**方法**：由调用方 `tuning()` 或 `MNN_METAL_COMMIT_NUM` 调整批次，
在同一工作负载上交替测量生产 wall，并用时间线解释差异。
同时观察 command buffer 数量与它们之间的 GPU 间隙；不能仅凭数量判断切换是瓶颈，
也不能照搬其他引擎的批次或宣称某个固定值对所有模型最优。

**调用时机陷阱**：检查 `transformers/llm/engine/src/llm.cpp` 的状态机。
`tuning()` 受 `CHECK_LLM_RUNNING` 约束，必须在模型进入可运行状态后调用，
不能把 load 中提前返回的调用当作完成了调优。

**验证**：确认候选值真正生效、日志无状态错误，并覆盖 prefill/decode 的不同批次需求。
调优入口与普通调用方应避免重复执行同一组调优。

---

## 7. Encode Replay（稳定 shape 前向录制重放）

代码入口：`source/backend/metal/MetalReplay.hpp` / `MetalReplay.mm`。

### 7.1 基本机制

当一个 `MetalExecution` 的输入/输出设备绑定和执行路径稳定时，
录制 encode 事件（pipeline、buffer 绑定、dispatch grid），后续调用直接重放事件，
跳过 `onEncode` 中重复的 CPU 决策逻辑。
重放的是编码事件，不是重复提交已完成的 `MTLCommandBuffer`。

### 7.2 安全模型

- **每次重放前重新校验**：`metalReplayValidate` 比较 tensor 当前的 buffer 与 offset；
  不匹配就放弃重放、退回正常 encode，允许重新录制，覆盖 KV 扩容与 allocator 重排。
- **豁免机制**：无法由重放 hook 正确更新的 per-token CPU 状态，通过 `canRecordEncode()` 排除。
- **防抖**：连续失效达到上限后禁止该录制反复重试，避免重录本身成为开销。
- **编译边界**：`MNN_METAL_OP_PROFILE` 构建下禁用 replay，避免未建模的 subpass encoder 切换。

### 7.3 Attention 接入

`source/backend/metal/MetalAttention.mm` 将稳定的 encode 决策与每 token 参数更新分离：

- `_computePathFlags()` 计算路径；`_writeCopyParam` / `_writeQKVParam` / `_writeSoftmaxParam`
  写参数，`onEncode` 与 `onReplayUpdate` 共用，避免两套更新逻辑漂移。
- `_pathSignature()` 编码影响 kernel 变体与事件布局的结构 flag，
  包括短 KV 变体和 `mQkvSimdReduce` 等路径；指纹变化就退回正常 encode 并重录。
- `onReplayUpdate` 更新参数 buffer 和录制事件中随 KV 长度变化的 grid / bytes，
  保证 `pastLength` 恰好前进一次。

**KV-cache 悬垂指针陷阱**：扩容可能销毁旧 cache tensor。
`onReplayUpdate` 必须在 `metalReplayValidate` 解引用之前，先比较 K/V tensor 的指针身份；
不能通过读取旧对象来验证它是否存活。scale buffer 的替换也必须触发相应失效处理。

### 7.4 LinearAttention 接入

`source/backend/metal/MetalLinearAttention.mm` 的 `canRecordEncode()` 按
`seqLen==1 && gated_delta_rule` 限定重放路径。

**资源生命周期**：`Pipeline.cpp` 可能对 LinearAttention 每 token 强制 re-resize。
如果 `onResize` 每次重建 `mConvOut`，即使 shape 未变，录制绑定的 `Tensor*` 也会悬垂。
shape 不变时应保留 Tensor 对象，并在 `onReplayUpdate` 中检查 resize-generation；
失效判断必须先于 `metalReplayEmit` 的解引用。

### 7.5 验证与调试

用 `MNN_METAL_DISABLE_REPLAY` 对照正常 encode；
用 `MNN_METAL_REPLAY_DEBUG` 观察 record / ban / invalidate 的触发原因。
验证应覆盖稳定 shape、KV 扩容、路径阈值跨越、参数变化和重复 resize：
稳定路径能重放，变化路径能安全失效并重录，输出与正常 encode 一致。

---

## 8. 调度类改动的验证套路

1. **先确认瓶颈位置**：用 `MNN_SESSION_CPU_TRACE` 看 encode / commit / wait 分段，
   对照 GPU 时间线找关键路径；不能因为有编码开销就认定命令复用值得投入。
2. **交替配对 A/B**：保持工作负载、构建与计时口径一致，不跨时段比较；
   同时检查顺序偏差和热漂移，不把均值变化直接当成优化收益。
3. **输出对拍**：调度改动不应改变计算语义。固定输入与采样条件，检查数值与 token 输出；
   若基线本身不确定，先建立重复运行的误差范围，再判断是否由改动引入差异。
4. **验证回退路径**：用对应 env 对照正常路径，覆盖设备差异、KV 长度变化与资源扩容。
   现场诊断需要的开关及其生效条件以 env 注册表为准。
5. **隔离 profile 扰动**：counter sample buffer attachment 与额外同步会改变编码开销、
   GPU 间隙及相对占比。profile 用于定位，再用生产构建验证，不能用其绝对值预测 wall。

## 9. 调度优化方法论（原理 / 适用条件 / 陷阱 / 验证）

> 本节用于选题和方案评审；具体机制与代码入口见上文对应章节。

### 9.1 每 token 必经路径去框架化（采样）→ 机制详见 §4

- **原理**：小操作若每 token 都创建通用 expr 会话，建图、shape、调度与销毁可能比计算更重。
  对已有 host 数据使用原生 SIMD；考虑设备 kernel 时，先证明它能减少数据交接与同步。
- **排查**：确认 executor 归属，不把“主图在 GPU”当成“所有辅助操作在 GPU”。
- **边界**：替换实现必须保留 tie-break、过滤顺序和子集下标语义。
- **验证**：检查输出等价性，以完整 token 周期覆盖采样计时盲区。

### 9.2 GPU→CPU 同步点治理

- **原理**：lm_head → logits → CPU 采样包含数据依赖；GPU 计算完成与 CPU 恢复执行是两个时刻。
  减少同步次数、重叠可并行工作、降低必要等待的唤醒成本，分别对应不同的优化位置。
- **方法**：检查同步点前后的资源依赖，区分 kernel 工作、GPU 空闲与 CPU 唤醒延迟。
  只有位于关键路径上的节省才能兑现为 wall，不能因为存在同步就否定 GPU 优化。
- **验证**：用生产 e2e 配对判断收益，局部时间线只用于归因。

### 9.3 Encode Replay 与资源生命周期 → 机制详见 §7

- **原理**：稳定 decode 重放 encode 事件，省去重复 CPU 编码逻辑；
  前提是设备绑定、路径签名与 per-token 参数都符合重放契约。
- **陷阱**：resize 重建 Tensor 或参数 buffer，会使录制对象悬垂；
  内容更新与对象重建必须区分，无法保持稳定时应在解引用前失效。
- **方法**：新 op 接入前审查 resize 行为；重放相关崩溃或乱码先查资源重定位。

### 9.4 多路径自动阈值与降级链

- **原理**：同一算子的不同 kernel 路径存在性能交叉点；按设备类别与工作负载选择路径。
  attention 路由见 `kernel-dev-and-optimize.md` §2.3.8。
- **方法**：布局、读写方式或任一侧 kernel 改变后重新校准，不能沿用旧交叉点。
  阈值以下、磁盘 KV 与特殊 mask 等回退路径也要覆盖，不只验证主路径。
- **边界**：设备能力分档显式表达，不假设跨设备阈值一致；env 覆盖保留为对拍与调试通道。
- **验证**：阈值两侧分别对拍与 A/B，确认边界正确切换；实测值不写进本文件。

### 9.5 实验开关的收敛纪律

- **原理**：新优化先以默认关闭的开关隔离新旧路径，完成正确性验证和充分交替 A/B 后，
  再决定是否默认启用并收敛开关，避免未验证路径生效或无效组合长期累积。
- **方法**：必要的诊断与安全回退开关保留明确语义；没有保留价值的实验实现与开关一起移除。
  失败经验只提炼为适用边界，不保留实验档案。
- **登记**：名称、默认值、功能和生效条件统一放在 `env-registry.md`，不附实验数据。

### 9.6 decode wall 残差的定位套路（包围圈排除法）

适用于已经确认有端到端差距，但尚未确定开销归属的场景。

1. **相位切分**：用 `generate.cpp` 的 `DUMP_PROFILE_INFO` 将 token 周期分为
   forward / sample / tokenizer / stream，依据测量排除非关键段，不预设采样一定便宜。
2. **检查 GPU 间隙**：在路径与 shape 可比的前提下，对不同生成长度的 trace 做差分，
   分离 prefill 与稳态 decode；结合每 token 的 op / command buffer / wait 计数定位间隙。
   结构计数与耗时分开分析，数量多不等于成本高。
3. **有约束地做消融**：先列出同步保护的数据依赖与资源生命周期，再用隔离探针验证假设。
   检查 allocator 复用和 replay 命中是否同时改变；不能把删除等待后的结果全归因于等待本身。
   不可消除的数据依赖，应转向降低等待和唤醒成本（§5）。
4. **量化自洽再动手**：以“每 token 等待次数 × 可消除的单次唤醒成本”估算上限，
   与同轮完整周期残差对账，扣除已重叠工作，避免重复计费；对不上就继续切分相位。
5. **验证**：同一二进制 env 对照、交替 A/B、输出对拍，同时覆盖资源扩容与路径切换。

对每 token 的小操作，先分开测框架会话成本与实际计算成本，再决定是否去框架化；
采样的 executor、等价性与计时边界见 §4。
