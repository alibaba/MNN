# LLM Decode / Prefill 优化案例（Metal）

> **配套 SKILL.md 的 sub-doc**：把 [`kernel-basics.md`](./kernel-basics.md) 的方法论具象化到 LLM 场景的 11 个 Step，按投入产出比排序。每个 Step 独立可读、独立提交。

## 优化总纲

LLM decode 每步生成一个 token，核心链路：

```
RMSNorm → Q/K/V Linear(GEMV) → RoPE → Attention(QK+Softmax+AV) → O Linear(GEMV)
       → RMSNorm → Gate/Up Linear(GEMV) → SiLU*mul → Down Linear(GEMV) → Residual
```

其中 **GEMV（矩阵-向量乘）占 60-80% 时间**，是优化主战场；其次是 Attention 和 RMSNorm。

Prefill 路径中，**Attention 的三段中间物化（mTempQK / mTempSoftMax）** 是长 prompt 显存瓶颈；**flash-attention 融合**是主杠杆。

## 优化路线图

| 优先级 | 优化项 | 加速比 | 涉及文件 |
|--------|--------|--------|----------|
| P0 | Per-op GPU profiling 定位瓶颈 | — | `MetalBackend.mm/hpp`, `MetalDefine.h`, `MetalExecution.mm` |
| P1 | Q4 GEMV deferred dequant | 1.2-1.5x | `ConvSimdGroupShader.hpp`, `MetalConvolution1x1.mm/hpp` |
| P1 | Q4 GEMV 双 simdgroup + ushort4 | 1.1-1.3x | `ConvSimdGroupShader.hpp`, `MetalConvolution1x1.mm` |
| P1 | Fused prefill flash-attention (pp2048) | 1.03-1.09x | `MetalFlashAttnShader.hpp`, `MetalAttention.mm/hpp` |
| P2 | Q4 GEMV pre-scaling nibble extraction | 1.05-1.1x | `ConvSimdGroupShader.hpp` |
| P2 | Fused attention 支持 GQA group_size 2-8 | 1.1-1.2x | `MetalAttention.mm`, `MetalAttentionShader.hpp` |
| P2 | RMSNorm 小 batch kernel 选择 | 1.05x | `MetalLayerNorm.mm` |
| P2 | Gate/Up dual dispatch fusion | 减少 dispatch | `ConvSimdGroupShader.hpp`, `MetalConvolution1x1.*`, `MetalBinary.mm`, `MetalBackend.*` |
| P2 | QKV triple dispatch fusion | 减少 dispatch | `ConvSimdGroupShader.hpp`, `MetalConvolution1x1.*`, `MetalBackend.*`, `FuseTransformerC4.cpp` |
| P3 | RoPE fusion (inv_freq 直接传入) | 减少 op 数 | `MetalRope.mm`, `CPURoPE.cpp`, `custom_op.py`, `transformers.py` |
| P3 | RemoveDeadShapeOp pass | 减少 op 数 | `RemoveDeadShapeOp.cpp`, `PostConverter.cpp` |

---

## Step 1: Per-op GPU Profiling 定位瓶颈

### 原理

Metal GPU 执行是异步的，CPU 侧计时无法准确反映各 op 的 GPU 耗时。通过在每个 op 的 command buffer 前后插入 GPU timestamp，可以精确测量每个 op 的 GPU 执行时间。

### 实现

**关键文件**：
- `source/backend/metal/MetalDefine.h` — 定义 `MNN_METAL_OP_PROFILE` 宏
- `source/backend/metal/MetalBackend.hpp` — `OpProfiler` 类声明
- `source/backend/metal/MetalBackend.mm` — profiling 逻辑实现
- `source/backend/metal/MetalExecution.mm` — 每个 op 执行时注入 profiling

**开启方式**：
```bash
cmake .. -DMNN_METAL=ON -DMNN_METAL_OP_PROFILE=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j8 llm_demo
```

**使用**：运行 llm_demo 后会自动输出每个 op 的 GPU 耗时排序，快速定位瓶颈 op。

### 陷阱

- profiling 本身会引入 command buffer 切分开销，**绝对数字不准确**，但**相对排序可靠**
- 第一次 inference 有 pipeline 编译开销，**跳过前几个 token 再看数据**
- Fused op（如 FusedAttention）内部的多个 kernel 会合并计时

### 输出示例

```
=== Metal Op Profile (sorted by GPU time) ===
[  1] Conv1x1_GEMV  layer.0.attn.q_proj     2.15 ms  (18.2%)
[  2] Conv1x1_GEMV  layer.0.mlp.gate_proj   1.89 ms  (16.0%)
...
[Total GPU time] 11.82 ms
```

---

## Step 2: Q4 GEMV Deferred Dequantization

### 问题

标准 Q4 GEMV kernel 中，每个 simdgroup 线程在累积循环的内层同时做 **nibble 解包 + 反量化（乘 scale + bias）+ FMA 累积**。反量化涉及 fp16 乘加，是瓶颈。

### 优化思路

**延迟反量化**：在内层循环只做整数累积（int8 × int8 → int32），循环结束后一次性反量化：

```metal
// 旧：每步反量化
for (k) {
    half w = dequant(packed_w[k]);  // 每步 fp16 乘加
    sum += input[k] * w;
}

// 新：延迟反量化
int32_t isum = 0;
for (k) {
    int8_t w = unpack(packed_w[k]);  // 只做整数解包
    isum += int32_t(input_quant[k]) * int32_t(w);
}
sum = half(isum) * scale + bias;  // 循环外一次反量化
```

### 实现要点

**关键文件**：`source/backend/metal/ConvSimdGroupShader.hpp`

1. **输入也需要量化**：input 从 fp16 动态量化为 int8，在 host 端（`MetalConvolution1x1.mm`）分配量化 buffer 和 scale buffer
2. **双 buffer**：`mTempInput`（量化后 int8）+ `mInputScales`（per-row scale）
3. **kernel 内部**：先对 input 做 per-row absmax 量化，然后整数 GEMV，最后 `result = isum * input_scale * weight_scale + weight_bias * input_sum`
4. **input_sum 修正**：因为 weight 是非对称量化（有 zero point），需要额外累积 `sum(input_quant)` 用于 bias 修正

**Dispatcher 修改**（`MetalConvolution1x1.mm`）：
```cpp
// deferred dequant 条件：area=1 (decode) + supportSimdGroupReduce
if (mDequantScaleBias && dequantInShader && area <= 1 && supportSimdGroupReduce) {
    mUseDeferredDequant = true;
    // 分配额外 buffer...
}
```

### 性能数据

Qwen3-0.6B Q4, Mac M4:
- 标准 GEMV: decode ~140 tok/s
- Deferred dequant: decode ~180 tok/s (**+28%**)

---

## Step 3: 双 Simdgroup GEMV + ushort4 Vector Load

### 问题

单 simdgroup GEMV（g8 kernel）的 occupancy 受限于寄存器压力和 simdgroup 数量。同时 weight 读取粒度为 `uchar4`（4 bytes），未充分利用 memory controller 的 burst 能力。

### 优化思路

1. **双 simdgroup 并行**：一个 threadgroup 内 2 个 simdgroup 分别处理不同的 OC 范围，共享 input 数据（通过 threadgroup memory）
2. **ushort4 向量加载**：weight 用 `ushort4`（8 bytes）一次读取，相比 `uchar4` 减少一半的 load 指令数
3. **两者组合**：2 simdgroups × ushort4 = 单 threadgroup 处理更多 OC，减少 dispatch 次数

### 实现要点

**关键文件**：`source/backend/metal/ConvSimdGroupShader.hpp`

```metal
// 双 simdgroup kernel 结构
kernel void conv1x1_gemv_g8_deferred_sg2(
    ...
    uint sgid [[simdgroup_index_in_threadgroup]],  // 0 or 1
    uint lid [[thread_index_in_simdgroup]],
    ...
) {
    // 每个 simdgroup 处理 8 个 OC
    int oc_start = gid * 16 + sgid * 8;

    // 通过 threadgroup memory 共享 input
    threadgroup half shared_input[IC_CHUNK];
    if (sgid == 0) {
        // 协作加载 input 到 shared memory
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 各 simdgroup 独立做 GEMV
    ...
    // simd_sum reduction 在 simdgroup 内完成
}
```

**Dispatcher**（`MetalConvolution1x1.mm`）：
```cpp
// 双 simdgroup：threadgroup size = 64 (2 × 32 lanes)
pipeline.threadgroupSize = {64, 1, 1};
pipeline.groupSize = {(oc_4 + 1) / 2, 1, 1};
```

### 注意事项

- `ushort4` 读取需要 weight buffer 按 8 字节对齐
- 双 simdgroup 要求 OC 至少 16（2 × 8）
- 小 OC 情况（如 hidden_size=1024 的某些层）仍走单 simdgroup kernel

---

## Step 4: Pre-scaling Nibble Extraction

### 问题

Q4 weight 解包标准流程：
```metal
uchar4 packed = wt[idx];
char4 w;
w[0] = (packed[0] & 0xF) - 8;  // 取低 4 bit，减去 zero point
w[1] = (packed[0] >> 4) - 8;   // 取高 4 bit，减去 zero point
```

`>> 4` 和 `& 0xF` + `- 8` 在 GPU ALU 上各占一条指令。

### 优化思路

**Pre-scaling trick**：在 host 端 pack weight 时预乘一个系数，使得 nibble 提取可以用乘法替代 shift，同时隐式完成 zero point 减法：

```metal
// 新：用乘法同时完成解包和 zero point
ushort4 raw = as_type<ushort4>(wt[idx]);
// 低 nibble: (raw & 0x000F) * scale_factor 隐式包含 -8
// ��� nibble: (raw >> 4 & 0x000F) * scale_factor
```

具体实现中，将 `& 0xF` 和 `- 8` 合并为一条带预计算常量的 MAD 指令。

### 性能影响

约 **5%** 的 GEMV 提速，在 deferred dequant 基础上叠加。

---

## Step 5: Fused Attention GQA 扩展

### 问题

原有的 `decode_qk_softmax` fused kernel 只支持 `group_size = 1`（MHA）或硬编码的特定 group_size。Qwen3 等模型使用 GQA（Grouped Query Attention），`num_heads / num_kv_heads` 可以是 2、4、8 等。

### 优化

**扩展 fused kernel 支持 group_size 2-8**：

**关键文件**：
- `source/backend/metal/MetalAttentionShader.hpp` — shader 模板化 group_size
- `source/backend/metal/MetalAttention.mm` — dispatcher 按 group_size 选择 kernel

```metal
// 模板化 group_size
#define GROUP_SIZE N  // 编译时宏

kernel void decode_qk_softmax_gN(
    ...
) {
    // 每个 KV head 对应 GROUP_SIZE 个 Q heads
    int kv_head = head_idx / GROUP_SIZE;
    // 共享同一个 K 做 QK dot product
    ...
}
```

**Dispatcher**：
```cpp
int group_size = num_heads / num_kv_heads;
std::string kernel_name = "decode_qk_softmax_g" + std::to_string(group_size);
```

### 性能影响

对 GQA 模型（Qwen3 group_size=2, Llama3 group_size=4），避免了 Q 和 K 的显式 repeat_kv 拷贝，**decode attention 提速 10-20%**。

---

## Step 6: RMSNorm 小 Batch 优化

### 问题

LLM decode 时 RMSNorm 的 batch = 1（单 token），而默认 kernel 选择策略倾向选择大 batch 优化的 kernel（更多 threadgroup、更大 tile），小 batch 下 launch overhead 反而更大。

### 优化

**关键文件**：`source/backend/metal/MetalLayerNorm.mm`

```cpp
// 小 batch 时用更小的 threadgroup，减少 launch overhead
if (batch <= 4 && hidden_size <= 4096) {
    // 用单 threadgroup 处理整个 norm
    threadgroupSize = {256, 1, 1};
} else {
    // 大 batch 用多 threadgroup 并行
    threadgroupSize = {256, 1, 1};
    groupSize = {batch, 1, 1};
}
```

### 性能影响

Decode 下 RMSNorm **提速约 5%**，累积到整链路约 1% 加速。

---

## Step 7: RoPE Fusion 图优化

### 问题

标准 RoPE 实现需要多个独立 op：position_ids → inv_freq 计算 → cos/sin → 乘加。这些小 op 各自 launch 一次 GPU kernel，launch overhead 在 decode 时占比显著。

### 优化

**将 RoPE 融合为单个 op**，直接接收 `position_ids + inv_freq` 参数：

**关键文件**：
- `schema/default/MNN.fbs` — RoPE op 增加 `hasInvFreq` 属性
- `source/backend/metal/MetalRope.mm` — Metal kernel 直接接收 inv_freq 做 RoPE
- `source/backend/cpu/CPURoPE.cpp` — CPU 实现对齐
- `transformers/llm/export/utils/custom_op.py` — FusedRoPE 导出逻辑
- `transformers/llm/export/utils/transformers.py` — Attention 中 inv_freq 传递

**导出侧**（`custom_op.py`）：
```python
class FusedRoPE(torch.nn.Module):
    def __init__(self, head_dim, name, inv_freq=None):
        self.inv_freq = inv_freq  # 直接嵌入 inv_freq 列表

    def forward(self, q, k, position_ids, ...):
        # ONNX export 时导出为单个 fused op
        ...
```

**Runtime**（`MetalRope.mm`）：
```cpp
// 接收 position_ids + inv_freq，一个 kernel 完成 RoPE
if (mHasInvFreq) {
    // 直接用 inv_freq 计算 cos/sin 并应用
    encoder.setBuffer(mInvFreqBuffer);
    encoder.setBuffer(positionIdsBuffer);
    [encoder dispatchThreadgroups:...];
}
```

### 额外图优化：RemoveDeadShapeOp

**关键文件**：`tools/converter/source/optimizer/postconvert/RemoveDeadShapeOp.cpp`

Transformer fusion 后，部分 Shape/Gather/Unsqueeze op 的输出不再被使用，但仍保留在图中。`RemoveDeadShapeOp` pass 自动检测并移除这些死代码 op，减少 inference 时的 dispatch 开销。

---

## Step 8: Gate/Up Dual Dispatch Fusion

### 问题

LLM decode 的 MLP 层有两个结构相同的 projection（Gate 和 Up），它们共享相同的输入（RMSNorm 的输出）。标准执行中 gate 和 up 各自独立 dispatch 一次 GPU kernel，每次 dispatch 有固定开销（command encoder setup、pipeline switch 等）。

### 优化思路

**将 Gate 和 Up 两个 projection 合并为一次 dispatch**：
- 从 `MetalBinary` 的 MUL_SILU 输入关系发现 Gate/Up 配对
- Leader（gate）在 `gid.z = 2` 维度上同时 dispatch 两路
- Follower（up）在 `onEncode` 中直接返回

### 实现要点

**关键文件**：
- `source/backend/metal/MetalBinary.mm` — 从 MUL_SILU 输入发现 Gate/Up 关系
- `source/backend/metal/MetalBackend.mm/hpp` — Gate/Up 注册和匹配方法（`registerConv1x1ForOutput`）
- `source/backend/metal/MetalConvolution1x1.mm/hpp` — `setupGateUpFusion`、leader/follower 调度
- `source/backend/metal/ConvSimdGroupShader.hpp` — `GATE_UP_FUSED` shader 宏

**Dispatch 机制**：
```cpp
// Leader (gate): gid.z = 2
// z=0 → gate output/weight/bias/dequant
// z=1 → up output/weight/bias/dequant
// 共享 input 和 const buffer (gate 和 up 的 OC 相同)
```

**Buffer 绑定**（onEncode）：
```
index 0: input (共享)
index 1: gate output
index 2: const buffer
index 3-5: gate weight/bias/dequant
index 6: up output
index 7-9: up weight/bias/dequant
```

### 性能影响

每层减少一次 dispatch。对小模型（layer 数多但每层计算量小），dispatch 开销占比更高，收益更明显。

---

## Step 9: QKV Triple Dispatch Fusion

### 问题

LLM decode 的 Attention 层有三个 projection（Q/K/V），它们共享相同的输入（RMSNorm 的输出）。标准执行中 Q/K/V 各自独立 dispatch，每层 3 次 dispatch。

### 优化思路

**将 Q/K/V 三个 projection 合并为一次 dispatch**：
- 按输入 tensor 对 Q4 projection 建组，识别 GQA 形状（2 个 K/V OC 相同，Q OC >= K/V）
- Leader（Q）在 `gid.x` 上连续分段，一次 dispatch 覆盖三路
- 环境变量 `MNN_DISABLE_QKV_FUSION=1` 用于 A/B 对比

### 实现要点

**关键文件**：
- `source/backend/metal/MetalBackend.mm/hpp` — QKV 分组注册（`registerConv1x1ForQKV`）、匹配（`matchQKVFusions`）
- `source/backend/metal/MetalConvolution1x1.mm/hpp` — `setupQKVFusion`、三路 buffer 绑定、leader/follower 调度
- `source/backend/metal/ConvSimdGroupShader.hpp` — `QKV_FUSED` shader 宏
- `tools/converter/source/optimizer/postconvert/FuseTransformerC4.cpp` — `reorderQKVProjections` 图重排

**分组匹配条件**：
1. 恰好 3 个 projection 共享输入 tensor
2. GQA 形状：2 个较小 OC 相同（K/V），第 3 个不小于它们（Q）
3. 每路 output_channel <= 16384
4. 三路都走 `conv1x1_gemv_g4m1_2sg_wquant_sg` pipeline

**Dispatch 机制**：
```
gid.x 分段：
  [0, q_groups)                           → Q
  [q_groups, q_groups + k_groups)         → K
  [q_groups + k_groups, total_groups)     → V

shader 根据分段选择对应的 output/weight/bias/dequant/const buffer
```

**Buffer 绑定**（15 个 buffer）：
```
index 0:  input (共享)
index 1:  Q output
index 2:  Q const buffer
index 3-5: Q weight/bias/dequant
index 6:  K output
index 7-9: K weight/bias/dequant
index 10: V output
index 11-13: V weight/bias/dequant
index 14: segment info {q_groups, k_groups, k_oc_slice, v_oc_slice, k_scale_coef, v_scale_coef}
```

**Converter 图重排（`reorderQKVProjections`）**：
- 将同层 Q/K/V convolution 移到最早 projection 的位置，保持连续
- 必要原因：原图中 Q 分支的中间 op 可能复用尚未执行的 K/V 输出 buffer
- 重排后有助于 buffer 分配器为 Q/K/V 分配不同的 buffer

**Buffer Overlap 安全检查**：
- `matchQKVFusions` 在匹配后检查 Q/K/V 的 output buffer 是否重叠
- MNN 的 buffer 复用机制可能导致不同 tensor 共享同一 Metal buffer
- 若检测到重叠，安全跳过 fusion，回退到逐个 dispatch

### 陷阱

- **Buffer 复用**：MNN 在 `onResizeEnd` 中 `compute()` 分配 buffer 后，同一 Metal buffer ��能被多个 tensor 复用。`matchQKVFusions` 必须在 `compute()` 之后调用，检查实际分配结果
- **调用顺序**：`onResizeEnd` 中先 `compute()` 再 `matchQKVFusions()`，不能反过来
- **环境变量检查**：在 `matchQKVFusions` 入口处检查 `MNN_DISABLE_QKV_FUSION`

### 性能影响

每层减少两次 dispatch。需配合 converter 图重排使 buffer 分配器分配不重叠的 buffer 才能实际生效。

---

## Step 10: LLM 导出兼容性修复

### 问题：inv_freq RoPE 路径的 unsqueeze 错误

当模型显式指定 `head_dim`（如 Qwen3-0.6B 的 `head_dim=128, hidden_size=1024, num_heads=16`），使得 `head_dim ≠ hidden_size / num_heads` 时，ONNX 导出中 cos/sin 的维度变换错误。

**根因**：
```python
# 错误：unsqueeze(2).unsqueeze(1) 将 [seq, dim] → [seq, 1, dim, 1]
# dim 维度落在了 num_heads 的位置，导致广播失败
cos = torch.cat(..., dim=-1).unsqueeze(2).unsqueeze(1)

# 正确：unsqueeze(0).unsqueeze(2) 将 [seq, dim] → [1, seq, 1, dim]
# 与 query_states [bsz, seq, num_heads, head_dim] 正确广播
cos = torch.cat(..., dim=-1).unsqueeze(0).unsqueeze(2)
```

**关键文件**：`transformers/llm/export/utils/transformers.py`

**影响模型**：所有显式指定 `head_dim` 且 `head_dim ≠ hidden_size / num_heads` 的模型。

---

## Step 11: Fused Prefill Flash-Attention

### 问题

标准 MNN prefill attention 走三段 pipeline —— `prefill_qk` → `softmax_plane_sg` → `prefill_qkv`，每一段都通过 global memory 传递中间结果：

```
                      mTempQK             mTempSoftMax
Q, K --[QK matmul]---> [B*H, seq, kv] --[softmax]--> [B*H, seq, kv] --[PV matmul]--> Output
       (kernel 1)     write 128 MB     (kernel 2)     write 128 MB     (kernel 3)
                                        read 128 MB                     read 128 MB
```

对 Qwen3-0.6B pp2048（`head_dim=128, heads=16, kv_heads=8, seq=2048`），单层 attention 中间 buffer 各 128 MiB，一次前向仅这两块的 write+read 就有 ~512 MB 的显存流量。llama.cpp 用融合 flash-attn kernel（online softmax 状态放寄存器，O accumulator 放 threadgroup memory）省掉这些中间物化 —— MNN pp2048 因此比 llama.cpp 落后约 9%。

### 优化思路

融合 Q·K^T + online softmax + P·V 到一个 kernel 中，中间数据全部保留在 threadgroup memory 和寄存器：

- **Q_TILE=16, KV_TILE=32, NSG=4**（4 个 simdgroup, 128 线程）
- Grid `(ceil(seq_q/16), num_head*batch, 1)`：每个 threadgroup 处理一个 (b, h) 头下的 16 行 Q，流式扫过所有 KV
- 每 simdgroup 拥有 2 行 Q 的 `M` (running max) 和 `S` (running sum) 寄存器
- 每 KV 块 (32 列)：QK → 在线 softmax → 用同一段 P 做 PV → 累加到 `so` (O accumulator)

**Threadgroup memory 布局**（D=128 时 ~15 KB）：
- `sq[Q_TILE * HEAD_DIM]` half — Q 分块（cooperative load 一次）
- `sf[Q_TILE * KV_TILE]` float — QK 的 fp32 scratch
- `ss[Q_TILE * KV_TILE]` half — 归一化后的 P，用 half 存以便 half×half → float MMA
- `so[Q_TILE * HEAD_DIM]` float — O accumulator，在线 rescale

**关键文件**：
- `source/backend/metal/MetalFlashAttnShader.hpp` — 新文件，`gPrefillFlashAttn` shader 字符串
- `source/backend/metal/MetalAttention.hpp` — 新增 `mFlashAttnPrefill` flag 和 `mKernel_flashAttn` pipeline
- `source/backend/metal/MetalAttention.mm` — 开关门控、pipeline 编译、dispatch 分支

### 实施要点

1. **Eligibility 门控**（`onEncode`）：
   ```cpp
   // 两条路径任一开启即可：
   //   1. config: attention_mode / 8 >= 1  (与 CPU 侧约定一致)
   //   2. env:    MNN_ENABLE_FLASH_ATTN_PREFILL=1  (developer override)
   //      MNN_ENABLE_FLASH_ATTN_PREFILL=0 可强制关闭（A/B 用）
   int attentionOption = ...->hint().attentionOption;
   bool enableFromConfig = (attentionOption / 8) >= 1;
   const char* envStr = getenv("MNN_ENABLE_FLASH_ATTN_PREFILL");
   bool envForceOn  = envStr && envStr[0] == '1';
   bool envForceOff = envStr && envStr[0] == '0';
   bool enableFlashAttn = envForceOff ? false : (envForceOn || enableFromConfig);

   mFlashAttnPrefill = enableFlashAttn && supportSimdMatrix
                       && !mKvInDisk
                       && (mHeadDim == 64 || mHeadDim == 128 || mHeadDim == 256)
                       && (group_size in {1,2,4,8})
                       && !mShortSeq && mSeqLen >= 128;
   ```
   KV int8 量化（`mQuantKey / mQuantValue`）通过 QUANT_K/QUANT_V shader 分支支持，不再屏蔽 FA。

2. **Pipeline 编译**（`compilerShader`）：
   ```cpp
   std::vector<std::string> keys = {"prefill_flash_attn", ftype, group_str, "HEAD_DIM_" + head_dim_str};
   if (mHasMask)    keys.emplace_back("HAS_MASK");
   if (mOutputC4)   keys.emplace_back("ATTENTION_C4");
   if (mQuantKey)   keys.emplace_back("QUANT_K");
   if (mQuantValue) keys.emplace_back("QUANT_V");
   ```
   `ATTENTION_C4` 是关键 —— 见下方避坑要点。QUANT_K/V 各自独立宏，允许仅量化 K 或 V。

3. **Dispatch**（`onEncode` prefill 循环）：
   ```cpp
   if (mFlashAttnPrefill) {
       // grid = (ceil(seq/16), B*H, 1), threadgroup = (32, 4, 1) = 128 threads
       auto gl = std::make_pair(
           MTLSizeMake(UP_DIV(seqLenPiece, 16), mBatch * mNumHead, 1),
           MTLSizeMake(32, 4, 1));
       [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
       continue;   // 跳过原来的三段 pipeline
   }
   ```
   `continue` 跳过 `mDecodeQkSoftmax` / 标准 QK / softmax / PV 分支。

4. **Kernel 三步循环体**：
   - **QK step**：每 SG 负责一个 8×8 输出块（cols `sgitg*8..sgitg*8+7`）。用 `simdgroup_load(mQ, sq+k_step, HEAD_DIM)` 从 threadgroup 装 Q，`simdgroup_load(mK, K_ptr, stride, ulong2(0,0), true)` 直接从 device 装转置的 K。`half × half → float` MMA 累加到 `mQK`，最后 `simdgroup_store` 到 `sf`。
   - **Softmax step**：每 SG 用 32 lanes 各取 1 列，`simd_max` / `simd_sum` 做行 reduction。在线更新 `M[j], S[j]`，把 `vs = exp(s - M_new)` 转 half 写到 `ss`，再用 `ms = exp(M_old - M_new)` rescale `so`。
   - **PV step**：每 SG 负责 `NO_PER_SG = HEAD_DIM/8/NSG` 个 8×8 输出块（D=128 时 4 块）。从 `ss` 装 `mP`，从 `V` 直接装 `mV`（同样 `transpose=true`），MMA 累加到 `mO`，写回 `so`。

5. **在线 softmax 数值稳定性**：
   ```cpp
   float M_new = simd_max(fmax(M[j], s));
   float ms = (M[j] == -INFINITY) ? 0.0f : exp(M[j] - M_new);
   float vs = (s    == -INFINITY) ? 0.0f : exp(s    - M_new);
   S[j] = S[j] * ms + simd_sum(vs);
   M[j] = M_new;
   ```
   两个 `-INFINITY` 短路是必需的，避免 `exp(-inf - -inf)` = `exp(NaN)` = NaN 从初始状态或全 masked 行传播。

6. **KV int8 量化**（QUANT_K / QUANT_V）：D=256 下无法把整 tile 反量化到 threadgroup（会爆 32 KB），采用**每 k_step 分批 8×8 dequant**：
   - `sK[NSG*8*8]` = 512 字节；K 按 kv-token 行 dequant（同一 kv_pos 的 D 值共用 scale）
   - `sV[NSG*8*8]` = 512 字节；V 按 kv-token 列 dequant（同一 kv_pos 是不同 D 的同列）
   - `k_scales` / `v_scales` 是 `device ftype*`（fp16），**不是** `device float*`；错声明必乱码
   - 用 `threadgroup_barrier`，不是 `simdgroup_barrier`（8 lane 写、32 lane 读）
   - buffer(9) k_scales、buffer(10) v_scales，只在 QUANT_K/V 时绑定

### 避坑要点（踩过的坑）

1. **`ATTENTION_C4` 输出布局** — 最重要的坑。Qwen3 c4-head export 时 `mOutputC4=true`，此时 output tensor 的实际布局是 `[num_head*(head_dim/4), batch*seq_q, 4]`（NC4HW4-packed），而不是自然的 `[B, seq_q, H, D]`。如果不区分而按 `[B, seq_q, H, D]` 写，token 输出会**从第一步就乱码**（`" הוד\nmore\n are\n are ..."` 的经典 pattern），并且**代码逻辑看着完全正确**、地址访问全都在合法范围。

   正确的 epilogue offset：
   ```cpp
   #ifdef ATTENTION_C4
       int o_off = (h * (param.head_dim / 4) + (d / 4)) * 4 * param.batch * seq_q
                 + (b * seq_q + q_abs) * 4
                 + (d & 3);
   #else
       int o_off = ((b * seq_q + q_abs) * param.head_num + h) * param.head_dim + d;
   #endif
   ```

2. **不要用 threadgroup memory 中转 K/V**。初版实现里我预先把 K 排到 threadgroup 中 `[d, kv]` 布局、V 排成 `[kv, d]` 布局（怀疑 `simdgroup_load` 的 5 参 transpose flag 有 bug），结果虽然正确但 pp2048 掉 45%。真正的 bug 是 `ATTENTION_C4`，`simdgroup_load(..., ulong2(0,0), true)` 在 M4 上工作正常。改回直接 device load 后，pp2048 反超 baseline。

   经验：**先怀疑数据布局，再怀疑 Metal API**。转置 flag 大概率没问题，语义可查文档。

3. **mixed-dtype MMA 只有 all-half 或 all-float**。所以 P 必须以 half 存到 `ss`，softmax 里 `ss[...] = half(vs)` 从 fp32 vs 转下来。QK 输出必须先写 fp32 `sf`（不能直接写 half，会丢精度），softmax 读 fp32、算完再存 half 到 `ss` 供 PV MMA 使用。这就是为什么要两块 scratch (`sf` + `ss`) 而不能合并。

4. **各 SG 只 rescale 自己那 2 行 `so`，但 PV 读 `so` 的所有 8 行**。这靠 softmax→PV 之间那个 `threadgroup_barrier` 保证同步 —— barrier 后 4 个 SG 的 rescale 都完成，PV 读到的 8 行都是新值。少这个 barrier 就必错。

5. **正确性验证一定要 greedy sampling**（`temperature=0, top_k=1`），否则采样噪声掩盖数值差异。首版通过 `MNN_ENABLE_FLASH_ATTN_PREFILL=1` A/B 对比同 prompt 前 40-60 token 是否 byte-identical，确认后再看性能。

### 性能数据（Q_TILE=16, KV_TILE=32, NSG=4）

M4 Pro，W4-block32 Q4，Metal fp16，4 线程：

| Model | Metric | OFF (三段 pipeline) | ON (fused FA) | Δ |
|---|---|---:|---:|---:|
| Qwen3-0.6B | pp512 | 4889 tok/s | **5013 tok/s** | +2.5% |
| Qwen3-0.6B | **pp2048** | 3714 tok/s | **4149 tok/s** | **+11.7%** |
| Qwen3-0.6B | tg128 | 296 tok/s | 286 tok/s | noise |
| Qwen3-4B | pp512 | 681 tok/s | 695 tok/s | +2.1% |
| Qwen3-4B | **pp2048** | 613 tok/s | **642 tok/s** | **+4.7%** |
| Qwen3-4B | tg128 | 74.5 tok/s | 74.2 tok/s | noise |
| Qwen3-8B | pp512 (5 reps) | 248 tok/s | 255 tok/s | +3.1% |
| Qwen3-8B | **pp2048 (5 reps)** | 254 tok/s | **277 tok/s** | **+9.1%** |

vs llama.cpp Metal Q4_1 pp2048 gap：
- Qwen3-0.6B：0.907× → **1.019×**（**反超** llama.cpp）
- Qwen3-4B：0.924× → **0.980×**

Prefill 越长收益越大（pp2048 >> pp512）：省下的 mTempQK/mTempSoftMax 中间物化在长 prompt 下占比更大。tg128 不受影响（decode 走 mShortSeq 路径，不进 flash-attn）。

### 正确性验证（greedy sampling，全模型）

Metal fp16 + `attention_mode: 8` 与 baseline（Metal fp16 + 三段 pipeline）对拍，`temperature=0, top_k=1`。前 30 token byte-identical：

| Model | 256.txt | 512.txt | 1024.txt |
|---|:---:|:---:|:---:|
| Qwen3-0.6B (head_dim=128, GQA=2) | ✓ | ✓ | ✓ |
| Qwen3-4B (head_dim=128, GQA=4) | ✓ | ✓ | ✓ |
| Qwen3-8B (head_dim=128, GQA=4) | ✓ | ✓ | ✓ |

9/9 组合完全一致 —— fused flash-attn kernel 在数值上等价于原三段 pipeline，Qwen3 全系列（head_dim=128, GQA in {2,4}, 层数 28/36/36）都可用。Qwen3.5（head_dim=256）也通过 QUANT_K/V 支持了 FA + KV int8。

### 参数调优实验记录

初版 Q_TILE=8 得 pp2048 +8.6%（0.6B）。按投入产出比再扫描：

| 变体 | 0.6B pp2048 | 0.6B pp512 | 4B pp2048 | 4B pp512 | 结论 |
|---|---|---|---|---|---|
| Q_TILE=8, KV_TILE=32（初版）| +5.2% | +0.8% | +1.7% | +0.5% | 起点 |
| **Q_TILE=16, KV_TILE=32** | **+11.7%** | **+2.5%** | **+4.7%** | **+2.1%** | ✅ 采用 |
| Q_TILE=16, KV_TILE=64 | +3.6% | **−6.4%** ⚠ | +5.6% | −2.4% | ❌ 回退，pp512 变差 |
| Q_TILE=32 | — | — | — | — | 跳过：threadgroup mem ~30 KB 接近 32 KB 上限，occupancy 风险 |
| Mask block classifier | — | — | — | — | 跳过：现有 whole-tile causal early-exit 已达 causal 下三角 50% 理论上界（4160/8192 blocks） |

**关键教训**：
- **减少 K read 冗余（Q_TILE ↑）是长 prompt 上最有效的杠杆**：Grid.x 减半直接换 K 读半减，pp2048 收益翻倍。
- **减少 KV 循环次数（KV_TILE ↑）反而不好**：多出的累加器 + 每 iter 更多 V 读，在 M4 上抵消 iter 减少的收益，pp512 甚至回退。
- **whole-tile causal early-exit 已经足够**：对严格下三角的 causal mask，`kv_block > max_attend_k` 早退与 llama.cpp 的显式 block classifier 效果等价，不必再写额外 kernel。

### 后续优化方向

尝试后判断出的**收益/风险边界**（M4 Pro 上）：

1. **F=2 多头融合**（一个 TG 处理 2 个共 KV 的 Q head）：K read 冗余减半理论收益 ~15%，但 threadgroup memory 从 15 KB 涨到 30 KB（接近 32 KB 上限），occupancy 会从 2 TG/SM 降到 1 TG/SM。占用率减半可能抵消 K read 减半的收益，**净收益不确定**。需要严谨的 profile 数据驱动，不建议盲改。
2. **Barrier 优化**（去掉循环末 barrier）：正确性通过但 4B pp512 -14% 显著回退。barrier 起到 memory bandwidth 调度作用，**保留全部 barrier**。
3. **so 初始零填充跳过**：Metal threadgroup memory 初值未定义，可能是 NaN；后续 `so *= 0` 传播 NaN 到 padding rows。**真实正确性风险**，必须保留显式清零。
4. **NSG 4→8**：总 MMA 数守恒，仅 scheduling 变化，M4 上收益不明。跳过实施。
5. **Tensor API path**（M5/A19+）：`mpp::tensor_ops::matmul2d` 32×32 tile，参考 `prefill_qk_tensor`。当前 M4 无 tensor API，无法本地测。硬件到位后可尝试。

**M4 上 Q_TILE=16 KV_TILE=32 NSG=4 已接近 Metal flash-attn 甜蜜点**，进一步收益需要：
- Multi-head fusion（需要 profile 数据引导 threadgroup memory / occupancy trade-off）
- MNN base perf 改进（Conv1x1 GEMV / KVCache 等，超出 flash-attn 范畴）
- Tensor API（M5+）

### 与 llama.cpp Metal Q4_1 的最终对比（M4 Pro, 4 线程）

| Model | pp512 | pp2048 |
|---|:---:|:---:|
| Qwen3-0.6B | **1.014× ✅** | **1.025× ✅** |
| Qwen3-4B | 0.983× | 0.980× |
| Qwen3-8B | 0.911× | 0.948× |

0.6B 全面反超；4B 差 2%（噪声级）；8B 差 5-9%（**base MNN perf 问题**，非 flash-attn 范畴 —— OFF baseline 已经 0.94-0.92×）。
