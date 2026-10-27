# 逐 forward 的 host↔device 时序竞争：在飞 uniform 被覆写 / 未上传的输入被读取

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
> 本册收 **§11：CPU 在 GPU 异步执行窗口内碰了 device 侧内存**，两个对偶方向——写在飞的 uniform、
> 读还没上传的输入。
>
> **与相邻分册的判别**：
> - [`nondeterminism.md`](nondeterminism.md) §9 收"读到未初始化内存"，第一武器是 `MallocScribble`；
>   **本册在 scribble 下不复现**（数据不是未初始化，是"合法但过期/超前"）。这是最硬的分流判据。
> - [`concurrency.md`](concurrency.md) §3 收 host 侧共享所有权 / 引用计数竞争，与 device 无关。
> - [`memory-aliasing.md`](memory-aliasing.md) §1.6 收 **单个 dispatch 内部** threadgroup 之间的竞争；
>   本册是 **跨 command buffer**、CPU 与 GPU 之间的竞争。
> - [`gpu-oob.md`](gpu-oob.md) §6 收 shader 越界与 command buffer 故障；本册的 buffer 地址都合法。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## §11 触发

满足以下之一强烈怀疑本类：

- **单个 op 测试单跑 N 次全过，放进测试家族里跑就偶发失败**，且失败的是哪个用例随前置用例变化；
- `MallocScribble=1 MallocPreScribble=1` 下**仍然全过**（← 与 §9 的关键分野）；
- 错误值"合理但偏了 10~30%"，不是 NaN、不是 ±65504、不是乱码 token；
- 出错的数值对应**最早写入的那批数据**（例如 chunked prefill 的第一段、KV 的第 0 行）；
- 在两次 forward 之间插一次 GPU drain（读一次输出即可）就**稳定通过**；
- 增删一个看起来无关的前置测试用例，失败率就变；
- GPU 跑分逐 run 抖动明显大于其它后端，且抖动像是"某条快路径时有时无"。

## 核心心法

**Metal 的 encode 是 CPU 侧的、异步的：`onEncode` 里碰的每一个 host 可见地址，都可能在上一枚
command buffer 还在飞的时候被改，或者还没被 GPU 写进来。** `MTLResourceStorageModeShared` 让 CPU
能直接读写 device buffer，这既是便利也是陷阱——它把"什么时候安全"从驱动手里交给了你，而编译器
和 `metalReplayValidate` 都不会替你检查。

两条对偶的失败方向：

| 方向 | 机制 | 后果 |
|---|---|---|
| **CPU 写在飞读** | 持久 uniform buffer 每次 encode 就地覆写，上一枚 CB 的 kernel 还在读它 | 在飞的 dispatch 拿到**后一次** forward 的参数，写到错误偏移 |
| **CPU 读未上传** | 输入张量的 host→device 是 blit **编码进队列**的，encode 期读 `buf.contents` | 读到回收池槽位的**上一个租户**的数据 |

判据只有一句：**这块内存在 encode 这一刻，GPU 侧的读/写是否已经完成？** 答案是"不一定"就不能碰。

安全的只有两类：**加载期就就位的**（权重、常量、`copyTensorBytes` 拷的东西），和**每个 dispatch
独立的**（`setBytes` 的副本）。后端自有、但被上一枚 CB 引用着的持久 buffer **不属于**安全类。

## 相关背景

- `MetalBackend::onCopyBuffer` 三条路径：device→host 有 `wait(1)`；host→device **默认走 queued blit**
  （`MNN_METAL_H2D_QUEUED`，无 drain——这是刻意的，为了消掉每 decode token 一次的 CPU/GPU 串行）；
  只有 `MNN_METAL_H2D_QUEUED=0` 才回到 drain + 直写。**所以"读输入张量的值"在默认配置下必然不可靠。**
- `MetalBackend::wait(int traceSite = -1)`：traceSite 纯诊断（`MNN_SESSION_CPU_TRACE` 下 4 槽归因），
  传 -1 不归因。
- **replay 机制依赖持久 param buffer 可就地改写**（`MetalReplay.hpp` 头注释明说 attention 的
  "param-buffer contents" 属于 per-token CPU 状态）：录一次 dispatch，之后每 token 只改 buffer 内容。
  把 param 换成 `setBytes` 就**必须**同步改 `onReplayUpdate` 去 patch 录制下来的 `bytesArgs`，
  否则 replay 静默用旧参数。
- `metalReplayValidate` 只校验 **tensor-backed** 绑定；后端自有的 param buffer 不在校验范围内，
  换不出 staleness 信号——别指望它兜底。
- `setBytes` 上限 4KB，MNN 的 uniform 结构体都是几十字节量级，全部适用。

## 排查流程

**Step 1 — 确认是不是这一类（三个便宜的判别，别跳过）**

1. 单跑 vs 家族跑：单跑 10 次全过、家族里偶发 → 顺序/时序依赖。
2. `MallocScribble=1 MallocPreScribble=1` 下单跑：**仍全过 → 排除 §9**。scribble 会把未初始化读
   变成确定复现，所以它对 §9 有效、对本册无效。
3. 看错误值量级：合理但偏 10~30% → 像"某块输入是 0 或旧值"；NaN / ±65504 → 先回 §9 / §5。

**Step 2 — 用 drain 把窗口钉死**

在怀疑的两次 forward 之间强制一次 GPU 同步。最省事的办法是 `onForward` 之后 `readMap` 一次输出
（device→host 路径内部会 `wait`）。**用 env 开关控制它**，这样一个二进制能交替测两种情况、不用重建：

```cpp
if (getenv("MNN_TMP_NO_DRAIN") == nullptr) { out[0]->readMap<float>(); }
```

- 加 drain 后 N/N 通过、去掉后按比例失败 → **窗口确认**，且确认是 host↔device 时序而非逻辑错。
- ⚠️ drain 是诊断工具不是修复：它把 CPU/GPU 重新串行化，正是 queued upload 要消掉的东西。

**Step 3 — 枚举窗口里被 CPU 碰过的 device 内存**

```bash
grep -n "\.contents" source/backend/metal/<Op>.mm
```

逐个问三个问题：这块是**加载期就位的**？**后端自有且本 forward 独占的**？还是**可能被在飞 dispatch
引用的**？第三类就是嫌疑。同时检查有没有反向的：encode 期读**输入张量**的 device buffer。

**Step 4 — 定向验证嫌疑，一次只改一个**

只把嫌疑那一块换成安全形式（uniform → `setBytes`；读输入值 → 改读形状/元数据），其余不动，
重跑 Step 2 的 N 轮。失败率归零 = 实锤；仍在 = 换下一个嫌疑。**不要叠加改动**，否则无法归因。

**Step 5 — 修复后必须验 replay 与端到端**

单 op 测试往往走不到 replay（它需要前一次 encode 录制过）。**唯一可靠的验证是跑足够多 token 的
端到端解码并对拍文本**：`llm_demo` + 100 以上 decode token，与基线二进制交替跑、md5 对拍。
replay 参数没接对的话，第 2 个 token 起就会错。

## 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|---|---|
| 单跑全过、家族里偶发、**scribble 下也全过** | 本册（不是 §9） |
| 错误值合理但偏 10~30%，出错位置对应"最早写入的那批数据" | 在飞 uniform 被覆写：第一批 dispatch 拿到了后一批的参数 |
| 两次 forward 之间插 drain 就稳定通过 | 窗口确认，进 Step 3 |
| encode 期读输入张量的值，读到的是别的测试/别的张量的数据 | host→device blit 还没执行，读到回收池旧内容 |
| 某条性能快路径时有时无、跑分逐 run 抖动 | 快路径门控依赖了不可靠的 encode 期读取（正确性没坏，性能在漂） |
| 改了 param 传递方式后 decode 从第 2 个 token 起全错 | replay 还在按持久 buffer 的假设取参数，`bytesArgs` 没 patch |
| 只有 chunked / 分段 prefill 触发，单段 prefill 不触发 | 分段才有"上一枚 CB 在飞时开始下一次 encode"的窗口 |

## 参考案例

### 11.1 MetalAttention 三个 param buffer 被 encode 期覆写（2026-10-12）

**症状**：`op/attention_decode` 的 `decode chunked` 子用例在 Metal 上偶发失败，报
`mismatch at batch=0 head=0 dim=0: got=0.058996 exp=0.074895` + `decode chunked failed at decode
step 0 (kvLen=65, probe=0)`。单跑该测试 10/10 全过（scribble 下也全过），放进 `op/attention`
家族里跑约 40% 失败。origin/master 上同样失败，非新引入。

**排查路径**：Step 1 三个判别全部指向本册（单跑全过、scribble 全过、错误值合理但偏 21%、
`probe=0` 说明出错的是 KV 第 0 行即 chunked 第一段 prefill 写的那批）。Step 2 用 env 开关在两次
prefill 之间插 drain：**加 drain 12/12 通过，不加 3/12 失败**，窗口钉死在"prefill#1 在飞 × prefill#2
encode"。Step 3 枚举 `.contents`，发现 `mParamCopy`/`mParamQKV`/`mParamSoftmax` 三个持久 buffer
每次 encode 由 CPU 就地覆写、encoder 用 `setBuffer` 绑定，全程无同步。

**根因**：prefill#1 的 KV copy kernel 还在读 `mParamCopy`，prefill#2 的 `_writeCopyParam` 已把
`dst_k_offset` 改成 32 行的偏移 → prefill#1 的 K/V 被写到错误位置 → **KV 第 0 行始终是 0** →
step 0 的探针 query（`probe=0`）读到全零，注意力被摊平，输出偏 21%。

**修复**：三个结构体全部改 `setBytes`（10 个绑定点：8 QKV + 1 copy + 1 softmax），删除持久 buffer
与其 `newDeviceBuffer` 分配；`onReplayUpdate` 改为按 `(index, size)` patch 录制下来的 `bytesArgs`
（`CopyParam` 36B 与 `Param` 64B 同在 index 4 但不会出现在同一 event，`SoftmaxParam` 24B 在 index 2，
三者与 `patchIntBytes` 针对的 4 字节整数都不撞）。

**验证**：与纯净 origin/master 基线交替跑家族 25 轮——**基线 10/25 失败，修复后 0/25**；
Metal `llm_demo` 3 轮交替顺序、每轮 113 decode token、输出 md5 一致（Step 5）；性能中性且
**方差收窄**（数据归 metal-optimize 台账，不在本册）。

**排除项与它们为什么误导**（这一段是本案例最值钱的部分）：

1. **测试自己的注释指错了机制。** 原注释写"on KV growth the Metal cache manager memcpys old rows
   on the CPU, racing the first prefill's in-flight copy kernel"，我照着去查
   `MetalKVCacheManager::expandKVCacheInMem` 的 CPU memcpy。用常量算一遍就发现该路径在此序列里
   **根本不执行**：`mExpandChunk=64`、`mKvAlignNum=32` → prefill#1 扩到 `ROUND_UP(32+64,32)=96`，
   prefill#2（kv=64）与 decode（kv=65）都不再扩容；唯一那次扩容 `mPastLength==0` 故 `needCopy=false`。
   **教训：注释描述的机制也要用形状和常量算一遍是否真的可达**，否则会在一条死路上花很久。
2. **怀疑过全局 `gMeta` 状态在用例间泄漏。** grep 证明测试只写 `previous/remove/add` 且都显式重置，
   `KVMeta::sync()` 也清 `n_reserve/reserve`。排除。
3. **怀疑过对象布局变化**（新增 bool 成员移动了后续成员偏移，暴露潜伏的越界写）。做法：只保留
   .hpp 的成员声明、回退 .mm 的全部逻辑，重建后仍不失败。排除。
4. **怀疑过"执行次数敏感"**（跑得多就触发）。做法：在基线上把 dense 子用例重复到 32、48 次执行，
   都不失败。排除——**这一步很重要，因为它一度让我误判成"既有 flaky，与我无关"**。
5. **样本量不足导致的误判。** 前 6 轮交替里基线也偶发另一个用例（`attention_decode`），我据此
   一度认为失败是既有噪声。加大到"每个配置 4 次重复"后才看清：目标用例在我的二进制上 6/6 失败、
   基线 0/6。**偶发问题的对照必须带样本量，单次交替没有判别力。**

**避坑要点**：

- 新增/修改 uniform 传递方式时，先问"这块 buffer 会不会被上一枚 CB 引用"，再决定 `setBuffer` 还是
  `setBytes`。几十字节的结构体没有理由用持久 buffer。
- 改 param 传递方式**必须同时改 `onReplayUpdate`**，并用 100+ token 的端到端对拍验证（Step 5）。
- 这个模式在 Metal 后端是**面上的**：`newDeviceBuffer` 有 36 处分配、分布在 25 个文件。是否暴露
  取决于该 param 是否**每次 forward 都重写**——只在 resize 期写一次的不受影响。逐个审计时按这个
  判据筛。

### 11.2 MetalAttention 读 mask 哨兵值读到未上传的脏数据（2026-10-12）

**症状**：给 mask 约定加"双向全可见"哨兵时，把 `DEFAULT_MASK` 的门控挂到了"读 mask 张量的值判断
是不是双向哨兵"上，结果 `op/attention_c4` 从 0/6 失败变成 **17/17 失败**，报
`right: -5.6, compute: -5.25459`；而该用例**单跑通过**。

**排查路径**：Step 2 的变体——回退 .mm 只保留 .hpp 成员，不失败；恢复 .mm，失败 → 定位到 .mm 的
逻辑。再在读取点插桩打印读到的值：生产端明明写的 `0.0f`，读回来是
`-30000 / -5.6 / -5.28 / -0.0223 / 0.793 / 0.768 / 0.502 / 0.302`，167 次读取里 5 次是脏的。
其中 `-30000` 正好是 `AttentionTest.cpp` 里 mask 的常量，说明回收槽位的上一个租户就是前面某个
attention 测试的 mask——与"blit 还没执行"的诊断交叉印证。

**根因**：`_computePathFlags` 在 encode 期读 `inputs[3]` 的 `buf.contents`，但 host→device 是
queued blit，此刻目标 buffer 还是回收池的旧内容。原代码里这个读取只喂 `mCausalLayout`（**性能**
路径），所以脏读最多让那一层丢掉 CAUSAL_TRI / CAUSAL_BOUND / FA / faTc；`DEFAULT_MASK` 只看
`mKVCache`（形状推导，可靠），正确性无恙。把正确性决策挂到同一个读取上，就把性能隐患升级成了
错误结果。

**修复**：**取消读值**，改成纯形状判据（rank 0 = 因果，rank 1 且 `elementSize()==1` = 双向，
rank ≥2 = dense），CPU 与 Metal 共用 `core/AttentionMaskConvention.hpp` 里的同一对 helper。
形状元数据在 resize 期固定、不可能脏，于是 `mBidirectionalMask` 变得可信，`DEFAULT_MASK` 的门控
也就安全了。附带修掉了 `mCausalLayout` 的性能非确定性。

**排除项与它们为什么误导**：

1. **一度以为 Metal 侧"本来就对、不用改"。** 因为视觉编码器 `kv_cache=false`，`compilerShader`
   三个 mask 宏一个都不发，天然无 mask。这个结论对**视觉路径**成立，但据此认为"值读取无害"是错的：
   同一个读取在 `kv_cache=true` 的 LLM 路径上一直在悄悄决定快路径开关。
2. **差点把因果哨兵直接搬给视觉编码器。** `llm.cpp` 现成的 rank-0 标量 0 看起来就是"无 mask"，
   但 `isLowerTriangular==true` 的语义是**因果**（三角在 `MNNSoftmax` 里做，`_maskQK` 只 scale），
   搬过去会静默把 ViT 三角化——不崩、不 NaN，只是模型变笨。**必须读 `_maskQK` 的早退分支和
   `useMaskInSoftmax` 才能确定语义，不能靠名字。**
3. **CPU 侧的 rank-0 越界读。** 值 ≥ epsilon 的 rank-0 mask 会掉进 `_maskQK` 的 dense 路径，按
   `mask[row * maskCols + col]` 索引一个 1 元素缓冲区。改成形状判据后结构上不可能再发生。

**避坑要点**：

- **encode 期不要读输入张量的值。** 需要按输入内容分支时，找形状/元数据替代表示；实在没有，就把
  决策挪到数据确定就位之后，或者接受一次 drain 并明确它的性能代价。
- 区分**性能门控**与**正确性门控**：不可靠的信号只能喂前者。写代码时把这条不变量写进注释，
  否则下一个人会顺手把它用到正确性上（我就是那个人）。
- 一个后端里"唯一一处"做某件事的代码要格外怀疑：本案例中 `MetalAttention` 是整个 Metal 后端
  唯一在 encode 期读 per-forward 输入张量内容的地方，其余 `.contents` 全是自有 param buffer 或
  加载期权重。**没有同类可参照，往往意味着它本来就是错的。**

## 相关文件索引

| 文件 | 关注点 |
|---|---|
| `source/backend/metal/MetalBackend.mm` | `onCopyBuffer` 三条路径、queued H2D blit、`wait(traceSite)` |
| `source/backend/metal/MetalEnv.hpp` | `MNN_METAL_H2D_QUEUED` 等开关的解析与默认值 |
| `source/backend/metal/MetalReplay.hpp` | replay 的安全模型、`bytesArgs`、`metalReplayValidate` 只校验 tensor-backed 绑定 |
| `source/backend/metal/MetalAttention.mm` | `_makeCopyParam`/`_makeQKVParam`/`_makeSoftmaxParam`（setBytes 范式）、`_computePathFlags`（形状判据）、`onReplayUpdate`（bytesArgs patch 范式） |
| `source/core/AttentionMaskConvention.hpp` | mask 三分类契约，以及"为什么按形状不按值"的完整论证 |
| `test/op/AttentionTest.cpp` | `op/attention_decode` 的 chunked 子用例（11.1 的回归测试）、`op/attention_cache_bidirectional`（11.2 的回归测试，自证式：两种哨兵必须可分离） |
