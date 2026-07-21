# Metal Prefill 优化总结

## 概述

本次优化在 Apple Metal 后端上为 Qwen3 系列 LLM 推理带来 4B/8B 模型 prefill 性能提升 3.6-3.9%，decode 无退化。测试基于 Apple M4, Metal 4, Qwen3-4B/8B (W4-block32, transformer_c4)。

## 优化项

### 1. LayerNorm + Conv1x1 GEMV 融合（核心优化）

**问题**: 每个 Transformer 层的 decode 路径中，RMSNorm（含 residual add）和 Conv1x1 GEMV 是两个独立的 kernel dispatch。residual add + normalize 需要一次完整遍历输入数据，紧接着 Conv1x1 GEMV 再次读取同一数据。

**方案**: 将 RMSNorm 计算融合进 Conv1x1 GEMV kernel。GEMV kernel 在读取输入时同步完成 residual add 和归一化，消除单独的 LN dispatch。

**实现**:
- `MetalLayerNorm.mm`: `onResize` 中注册 binary RMSNorm 的 fusion info（hidden input、residual input、gamma、eps），通过 `backend->registerLayerNorm()` 注册到 MetalBackend
- `MetalBackend.hpp/mm`: 新增 `LayerNormFusionInfo` 结构和 `mLayernormMap`，`matchLNFusions()` 在 `onResizeEnd` 中将 LN 匹配到消费其输出的 Conv1x1 leader（QKV leader 或 GateUp leader）
- `MetalConvolution1x1.mm`: `setupLNFusion()` 创建带 `LN_FUSED` 宏的 fused pipeline；`bindLNBuffers()` 绑定 residual/gamma/eps 到 buffer 20-23
- `ConvSimdGroupShader.hpp`: `LN_FUSED` 宏下，GEMV kernel 在循环中计算 `hidden + residual`、RMSNorm（`simd_sum` 归约 + `rsqrt`）和 gamma 缩放，仅一个 threadgroup 写 `ln_residual_out` 避免竞争
- 融合后 `MetalLayerNorm::onEncode` 直接 return，跳过自身 dispatch

**效果**: 每层减少 1 次 kernel launch + 1 次输入数据遍历。对 decode 路径收益最大（GEMV 是访存密集型，减少数据搬运直接提升吞吐）。

### 2. QKV Follower Skip 恢复

**问题**: QKV follower 的 skip 逻辑被注释掉用于调试，导致 follower 仍然执行冗余的独立 dispatch。

**方案**: 恢复 `if (mIsQKVFollower) return;`，使 follower 跳过执行——Q/K/V 的计算已由 leader 的 fused dispatch 完成。

**文件**: `MetalConvolution1x1.mm`

### 3. 自适应 In-Shader Dequant（Prefill 路径）

**问题**: 非 Tensor-API 设备（M4 及以下）上 prefill 的 dequant 策略此前仅能通过环境变量强制开关。小权重用 in-shader dequant 会因 GEMM kernel 效率低而变慢，大权重用 outer-dequant 会有双 pass 开销。

**方案**: 自动按权重大小选择策略：
- `ic * oc > 4M`：使用 in-shader dequant（避免 dequant→fp16→GEMM 双 pass）
- `ic * oc <= 4M`：使用 outer-dequant + 优化的 fp GEMM kernel
- 环境变量 `MNN_METAL_PREFILL_INSHADER_DEQUANT=1/0` 仍可强制覆盖

**文件**: `MetalConvolution1x1.mm`

### 4. MetalRope `loadC4` 简化

**问题**: `loadC4` 有 `seqLen == 1` 的特殊分支和独立的 `c4Offset` 函数，逻辑冗余。

**方案**: 统一为单一公式 `tensor[(c4 * outerSize + token) * 4 + ci]`，删除 `c4Offset` 函数和 `xSeq` 临时变量，直接使用 `p.outerSize`。

**文件**: `MetalRope.mm`

### 5. CPU Attention C4 输出路径修复

**问题**: C4 输出路径中 `outputPacked` 的初始化位置错误（在计算之前设置），且非 C4 路径的逻辑不够清晰。

**方案**: C4 输出使用 `memcpy`（数据已在正确布局），非 C4 输出使用 `MNNUnpackCUnitTranspose`。

**文件**: `CPUAttention.cpp`

### 6. CPUKVCacheManager 微优化

预计算 `totalChannel = mKvNumHead * mHeadDim`，避免 `loadValue` lambda 中重复乘法。

**文件**: `CPUKVCacheManager.cpp`

### 7. 移除 FuseTransformerC4 中的 QKV 重排

**问题**: `reorderQKVProjections()` 在图优化阶段重排 Q/K/V 卷积顺序，但 QKV fusion 现在由 Metal 后端运行时匹配（`matchQKVFusions`），图阶段重排已无必要。

**方案**: 删除 `reorderQKVProjections()` 及其调用。

**文件**: `tools/converter/source/optimizer/postconvert/FuseTransformerC4.cpp`

## 性能数据

测试环境：Apple M4, 16GB, Metal 4, 4 线程, pp512 + tg128

| 模型 | 指标 | 优化前 | 优化后 | 变化 |
|------|------|--------|--------|------|
| Qwen3-0.6B | Metal prefill | 2957 | 2974 | +0.6% |
| Qwen3-0.6B | Metal decode | 178.34 | 178.35 | 0% |
| Qwen3-4B | Metal prefill | 412 | 427 | **+3.6%** |
| Qwen3-4B | Metal decode | 35.26 | 35.21 | 0% |
| Qwen3-8B | Metal prefill | 222 | 231 | **+3.9%** |
| Qwen3-8B | Metal decode | 19.89 | 19.91 | 0% |

Prefill 提升随模型增大而显著（4B/8B 更多的权重符合 in-shader dequant 阈值），decode 无退化（LN 融合减少的 dispatch 开销与 GEMV kernel 内增加的归一化计算相互抵消）。

## 涉及文件

| 文件 | 改动类型 |
|------|---------|
| `source/backend/metal/MetalLayerNorm.hpp` | 新增 `mIsFused` 标志 |
| `source/backend/metal/MetalLayerNorm.mm` | 注册 LN fusion info，融合时跳过 dispatch |
| `source/backend/metal/MetalBackend.hpp` | 新增 `LayerNormFusionInfo`、`mLayernormMap`、`registerLayerNorm`、`matchLNFusions` |
| `source/backend/metal/MetalBackend.mm` | 实现 `matchLNFusions`，更新 QKV fusion 注释 |
| `source/backend/metal/MetalConvolution1x1.hpp` | 新增 LN fusion 相关成员和方法声明 |
| `source/backend/metal/MetalConvolution1x1.mm` | `setupLNFusion`/`bindLNBuffers`，自适应 dequant，QKV follower skip 恢复 |
| `source/backend/metal/ConvSimdGroupShader.hpp` | `LN_FUSED` shader 宏实现 |
| `source/backend/metal/MetalRope.mm` | `loadC4` 简化 |
| `source/backend/cpu/CPUAttention.cpp` | C4 输出路径修复 |
| `source/backend/cpu/CPUKVCacheManager.cpp` | 预计算 totalChannel |
| `tools/converter/source/optimizer/postconvert/FuseTransformerC4.cpp` | 移除 `reorderQKVProjections` |
