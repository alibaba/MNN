# 算子融合：从导出图到 Metal 单 dispatch

> **读这份文档的时机**：想让若干个小算子合成一次 GPU dispatch；改 `FusedLinear` /
> `GatedRMSNorm` 相关代码；排查"融合没命中 / 融合后输出错"的问题；给新模型结构补融合。
>
> **一句话现状**：**Metal 后端已零图匹配**。融合分组全部由**导出图声明**，
> 后端只按声明去装配 leader/follower 单 dispatch。历史上后端里那套运行时模式匹配
> （`matchQKVFusions` / `matchLNFusions` / `matchLinearAttnGatedNormFolds` + 7 张注册表）
> 已在 2026-08-05/08-06 全部删除。
>
> 相关文档：kernel 本身怎么写见 [`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md)；
> 调度层（fence / replay / H2D）见 [`runtime-scheduling.md`](./runtime-scheduling.md)；
> env 开关见 [`env-registry.md`](./env-registry.md)。

---

## 0. 全链路总览

融合是**一条跨 Python 导出 / converter / geometry / 后端**的链，任何一环断掉都会静默回退到
未融合路径（正确但慢）。定位"为什么没融合"要按这个顺序查：

```
① Python 导出 (transformers/llm/export/)
   所有投影都是独立的 Linear（FakeLinear→rebuild 成带 pre/post Reshape+Convert 的
   量化 conv1x1）；导出图保持标准语义，不再发出 FusedLinear / GatedRMSNorm 自定义 op。
   mnn_converter.py::transformer_c4_args 把 fuse 开关透传给 MNNConvert。
        ↓
② Converter 图优化 (tools/converter/source/optimizer/postconvert/)
   FuseTransformerC4.cpp（融合算子的唯一创建者）
   ├─ fuseProjGroups()              构造 FusedLinear（管线第一步，见 §2）：
   │    fuseAttentionProjGroups        attention q/k/v(+输出门) → 3~4 成员
   │    fuseLinearAttentionProjGroups  linear-attn qkv/z/b/a → 4 成员
   │    fuseGateUpProjGroups           dense SwiGLU gate/up → act_silu_mul
   ├─ foldBinaryLnIntoFusedProj()   把前置 LayerNorm 吸进 FusedLinear（置 has_ln）
   ├─ fuseFusedGateUpOutputC4()     输出 C4 布局整理
   └─ GatedRMSNorm 折叠             linear-attention 区域匹配成功后，把 spelled-out 的
                                    Reshape/Cast*/RMSNorm/SILU/MUL 链就地构造成 GatedRMSNorm
                                    （旧模型里导出期发出的两种 op 仍被识别/加工）
        ↓
③ Geometry 兜底 (source/geometry/)
   GeometryFusedProj.cpp::_keepWhole()      Metal / OpenCL-buffer / Vulkan-buffer / CUDA 整体透传
                                            （后两者是组合执行，见 §3）；其余 ⇒ 拆回原始 conv1x1 图
   GeometryGatedRMSNorm.cpp                 谓词不过 ⇒ 分解成 LayerNorm + SILU + MUL
        ↓
④ Metal 运行时 (source/backend/metal/)
   MetalFusedProj.mm      setupFusion() 按成员顺序装配 leader/follower
   MetalConvolution1x1.mm setupGateUpFusion / setupQKVFusion / setupLNFusion（真正的单 dispatch）
   MetalGatedRMSNorm.mm   整 op 一个 kernel
```

**一条铁律贯穿全链**：融合把前驱折进后继时，**必须校验前驱的输入没有被分配器复用为后继的输出**。
详见 §4「内存别名」。

---

## 1. 导出侧：保持标准图，开关透传

导出侧**不再发出任何融合自定义 op**：投影都是独立 Linear，FusedLinear / GatedRMSNorm
全部由 MNNConvert 的 `FuseTransformerC4` 在 removeDupOps 阶段（MNN→MNN, optimizeLevel=1）构造。

### 1.1 `FusedLinear` op

一组共享同一输入的 conv1x1（投影）被打成一个 op，可选地把前置 LayerNorm 一起吸进来。

**导出开关**（`llmexport.py`，**默认全开**；`--disable_transformer_c4` 会把它们全部强制关闭；
`mnn_converter.py::transformer_c4_args` 把它们映射成 MNNConvert 的
`--transformerFuseQkvProj / --transformerFuseGateUpProj / --transformerFuseLnProj`；
`lora_split` 时 qkv/gateup 自动置 0，保持成员 conv 独立可匹配 lora 权重）：

```
--disable_fuse_qkv_proj        关掉 q/k/v（及 linear-attn 四路）分组
--disable_fuse_gate_up_proj    关掉 MLP gate/up 分组
--disable_fuse_ln_proj         关掉 LayerNorm 吸收
```

GatedRMSNorm 没有单独开关：由 `--transformerFuseC4` 统一控制（见 §5）。

⚠️ **旧模型（转换时未跑该 pass 的）在 Metal 上不享有融合**（实测 Qwen3.5-0.8B
decode dispatch 246 → 365）。**换收益的方式是重新转换模型，不是改后端**。

**成员语义**：`act_silu_mul == true` ⇒ 这组是 (gate, up)，运行时会追加 SiLU×Mul；
否则是 q/k/v（3 成员）或 q/k/v + 第四投影（4 成员：gated 变体的输出门，
或 linear-attention 的 qkv/z/b/a）。**运行时不做任何模式推断，只读成员顺序。**

**MoE 例外**：expert 投影的 post_reshape 是 2-D（`[-1, oc]`），天然不满足
`matchLinearPost` 的 3-D 约束，expert gate/up 不会被融合（与旧行为一致）。

### 1.2 `GatedRMSNorm` op（`OpType_GatedRMSNorm = 308`）

Qwen3.5 linear-attention 的输出门控段（RMSNorm × SiLU(z) 的双输入形态）。
**导出侧不再发出该 op**：图里保留标准的 Reshape/RMSNorm/SILU/MUL 链
（`transformers.py::LinearAttention.forward` 的 export 分支），由 converter 的
`FuseTransformerC4` 在 linear-attention 区域匹配成功后就地构造（见 §2、§5）。
参数复用 `LayerNorm` table，gamma/beta 沿用原 LayerNorm 的 external weight——
全程二进制往返，**天然规避了 §5.3 的 JSON 6 位截断坑**（bit-pattern hack 已删除）。

---

## 2. Converter 侧：构造 FusedLinear、LN 吸收与布局

`tools/converter/source/optimizer/postconvert/FuseTransformerC4.cpp`

| 函数 | 行 | 作用 |
|---|---|---|
| `fuseProjGroups()` | 管线第一步 | **构造 FusedLinear**：`fuseAttentionProjGroups`（RoPE q/k + attention v + 可选输出门投影）、`fuseLinearAttentionProjGroups`（qkv/z/b/a）、`fuseGateUpProjGroups`（SILU+MUL 两侧的等宽 conv 对）。把各成员 Convolution2D 的 external/bias/quan 参数**原样搬入** `FusedLinearParam.convs`（offset 不变，量化字节不变）；morph 到最早成员的 op 位置上，成员各自的 post Convert/Reshape 原样保留，多余的 pre 链删除。粒度开关：`--transformerFuseQkvProj / --transformerFuseGateUpProj`，`ln_fold` 位由 `--transformerFuseLnProj` 决定 |
| `foldBinaryLnIntoFusedProj()` | | 把 `FusedLinear` 前面的 residual-add + LayerNorm 折进去，置 `has_ln` |
| `fuseFusedGateUpOutputC4()` | | gate/up 输出的 C4 布局整理 |
| GatedRMSNorm 折叠 | `canFoldGatedNorm` + `matchHiddenBlockFromLinearAttention` 的 legacy 分支 | linear-attention 区域匹配成功后，把 Reshape/Cast*/RMSNorm/SILU/MUL 链就地 morph 成 `GatedRMSNorm`，并**删掉两侧 identity Reshape**（kernel 索引已吸收两次 C4 重排，输出直接喂 out_proj）；旧模型里导出期发出的 GatedRMSNorm op 走识别分支 |

**LN 折叠的唯一消费者约束**：LayerNorm 的输出必须只被这一组投影消费。
历史踩坑：attention 块的 LN 输出除 `FusedQKV` 外还被 `q_gate_proj` 消费（第 5 个消费者），
唯一消费者检查失败 ⇒ LN 折不进去。把 `q_gate_proj` 并入成 4 成员组后，
0.8B 上 `has_ln` 从 17/24 升到 **23/24**（仅 layer 0 无 residual add，符合预期）。

**成员顺序纯位置化**：运行时 `convs[i] → outputs[i]` 位置映射（gateup 约定 convs[0]=gate、
convs[1]=up，`out = up * silu(gate)`）。构造顺序沿用旧导出约定：q/k/v(/gate)、qkv/z/b/a、gate/up。

**late fusion 的收益**：融合发生在 FakeLinear rebuild/量化**之后**，per-op `--quant_config`
天然按成员独立生效；Pass 关闭或匹配失败时保留原始 conv 图，`lora_split` 直接置 0 两个 proj 开关。

---

## 3. Geometry 兜底：其余后端拆回去

`source/geometry/GeometryFusedProj.cpp:48 _keepWhole()`

四个后端把 `FusedLinear` 整体透传：
- **Metal**：原生融合 kernel（`nativeEnvelopeOk + allMembersAre1x1`）；
- **OpenCL-buffer**：组合执行容器 `FusedProjBufExecution`（env `MNN_OPENCL_FUSED_PROJ_DISABLE` 可关）；
- **Vulkan-buffer / CUDA**：组合执行容器（`VulkanFusedProj` / CUDA `FusedProjExecution`），
  经 `STATUS_SUPPORT_FUSED_PROJ` runtime status 握手（Vulkan image 变体与旧运行时答 0 自动关门），
  谓词是 `FusedProjCommon::compositeEnvelopeOk`（额外要求全部成员为 int 量化 type=1——
  fp16 权重成员在 Vulkan 会落到 `VulkanConvolutionImpl::create` 的不确定分支，保守走分解）。
  组合容器把成员 conv / 二元 RMSNorm / MUL_SILU 作为子执行驱动，数值与分解路径逐字节等价，
  保留整 op 是为将来收敛 kernel 数做准备。CUDA 无本仓 CI，生产启用前需 Linux+NVIDIA 人工验证；
  CUDA Memory_High（或未编译 MNN_LOW_MEMORY）时成员 conv 走 float 反量化路径，LLM 权重显存放大。

其它情况在 geometry 阶段把它**拆回原始的 conv1x1 子图**（外加 LayerNorm / SiLU-Mul）。
所以同一个 .mnn 模型在任何后端上跑得到完全相同的数值结果，只是没有融合收益。

`GeometryGatedRMSNorm` 同理：`OpCommonUtils::gatedRMSNormFusable` 通过才整 op 透传，
否则分解 `LayerNorm + SILU + MUL`。prefill 与非 Metal 后端都走分解。

> ⚠️ **geometry 的保留门与后端 Creator 的接受条件必须是同一个谓词**——这条坑见 §5.2，
> 是本仓踩过两次的同构 bug。OpenCL/Vulkan/CUDA 的 creator 与 `_keepWhole` 共用
> `FusedProjCommon` 里的谓词，新增拒绝条件必须加进谓词而不是 creator。

---

## 4. Metal 运行时：装配 leader/follower

### 4.1 时序：为什么在 `onResizeEnd`

```
MetalBackend::onResizeBegin()      清空 mFusedProjs 注册表
   ↓
每个 op 的 onResize()
   MetalFusedProj::onResize()      建子执行、算 shape、申请中间 buffer
      └─ 成功后记下 mProjOutputs / mLnHiddenIn / mLnResidualIn / mLnResidualOut
         并 backend->registerFusedProj(this)          （MetalFusedProj.mm:165-174）
   ↓
allocator 的 compute()             ★ 这里才把地址真正分配下去
   ↓
MetalBackend::onResizeEnd()        （MetalBackend.mm:1042）
   └─ 遍历 mFusedProjs，逐个调 setupFusion()
```

**必须等到 `onResizeEnd`** 的原因：`setupFusion` 里的**别名检测与 STATIC re-home 依赖
张量的实际地址**，而地址只有在 allocator `compute()` 之后才确定。

注册表在 `MetalBackend.hpp:346-394`：`FusedProjFusionHost` 抽象基类（唯一虚函数 `setupFusion()`）、
`registerFusedProj()`、`mFusedProjs`。**每次 resize 都重新注册**（`onResizeBegin` 会清空）。

### 4.2 `MetalFusedProj::setupFusion()`（`MetalFusedProj.mm:181-253`）

```cpp
// ① 融合 kernel 只存在于 decode-GEMV 管线；其它（含全部 prefill）保持逐成员 dispatch
for (auto &conv : mConvs) {
    if (!conv->is2sgDecodePipeline()) { return; }
}

// ② 分组直接来自导出的成员顺序，运行时不做任何模式推断
bool projFused = false;
if (mIsGateUp) {                                          // act_silu_mul
    if (mConvs.size() == 2 && !MetalEnv::get().gateUpFusionDisabled)
        projFused = mConvs[0]->setupGateUpFusion(mConvs[1].get(), mUp.get());
} else if (mConvs.size() >= 3 && !MetalEnv::get().qkvFusionDisabled) {
    projFused = (mConvs.size() == 3)
        ? mConvs[0]->setupQKVFusion(/* k,v */)
        : mConvs[0]->setupQKVFusion(/* k,v,w —— 第 4 投影 */);
}

// ③ LN 折叠必须以投影融合成功为前提（见 §4.4）
if (!projFused || !mHasLn || !mLn || MetalEnv::get().lnFusionDisabled) return;
if (!mLn->isNC4HW4() || !mLn->isRMSNormWithGammaBeta())                return;

// ④ 别名检测 + STATIC re-home（见 §4.3）
// ⑤ mConvs[0]->setupLNFusion(...) 成功后 mLn->setFused() 压掉 LayerNorm 自己的 dispatch
```

### 4.3 内存别名：融合最危险的一环

**问题**：LayerNorm 原本是一次独立的、更早的 dispatch，它读 residual 输入、写 normalized 输出。
分配器完全可以把某个投影的输出分配到 residual 输入的同一块地址上——
在"LN 先跑完、投影再跑"的顺序下这是合法的。
**融合之后两者进了同一个 kernel**，读和写就变成了同 dispatch 内的竞争。

这正是历史上 **LN×QKV_FUSED_P4 在 Qwen3.5-2B 上 decode 输出逐次不同**的根因
（当时未定位，以 `setupLNFusion` 硬门控排除 P4；2026-08-03 根因修复后门控已移除）。

**修法**（`MetalFusedProj.mm:226-248`）：

```cpp
std::vector<Tensor *> written = mProjOutputs;
if (mIsGateUp) {
    written = {mGate.get(), mUp.get(), mProjOutputs[0]};   // leader 写两半 + SiLU-mul 结果
}
for (auto *out : written) {
    if (!out) continue;
    if (!MetalBackend::tensorsOverlap(out, mLnHiddenIn) &&
        !MetalBackend::tensorsOverlap(out, mLnResidualIn)) continue;
    if (!backend->onAcquireBuffer(out, Backend::STATIC)) return;   // 失败就整个不折
}
// residualOutput 同样要查
```

- `Backend::STATIC` 池**永不被动态池复用** ⇒ re-home 之后地址不可能再与输入重叠。
- **re-home 失败就跳过融合**，绝不冒险。
- 同样的 STATIC re-home 也用在 QKV 融合的 k/v/w 输出上——融合后 k/v 提前写出，
  可能被动态池复用者覆写（见 `MetalConvolution1x1::setupQKVFusion` 的注释）。

> **写新融合时的自检**：列出这次融合的 kernel **写**哪些张量、**读**哪些张量，
> 两两跑 `tensorsOverlap`。只要有交集就 re-home 或放弃。
> 这个错误不会稳定复现——它表现为"偶尔输出不同"，比崩溃难查得多。

### 4.4 链式融合：每一级门控必须依赖上一级的成功

⚠️ **2026-08-10 修的 bug，教训值钱**：

`setupFusion` 曾**忽略** `setupGateUpFusion`/`setupQKVFusion` 的返回值，无条件接着做 LN 折叠。
LN 折叠会 `mLn->setFused()` 压掉 LayerNorm 自己的 dispatch，`mNormalized` 从此**无人写**；
投影融合一旦失败（env 关闭 / pipeline 创建失败 / 量化布局不一致 / STATIC re-home 失败），
非 leader 的 conv 仍各自 dispatch 去读 `mNormalized` → 输出错误。

复现：`MNN_METAL_DISABLE_QKV_FUSION=1`（LN 融合保持默认开）即让 0.8B 输出乱码。

修法：捕获返回值，`projFused == false` 直接跳过 LN 折叠。

> **通则**：链式融合的每一级门控必须**显式依赖上一级的成功结果**，
> 不能只依赖"通常会成功"。这条与 §5.2 的 geometry/Creator 谓词分叉是同构错误。

### 4.5 leader/follower 接口（`MetalConvolution1x1`）

融合的实际执行体仍在 `MetalConvolution1x1`（`.hpp:26-76`，实现 `.mm:83 / :192-357 / :365`）：

| 接口 | 语义 |
|---|---|
| `setupGateUpFusion(peer, peerOutput)` | `this` = gate（leader），`peer` = up（follower）|
| `setupQKVFusion(k,kOut, v,vOut, [w,wOut])` | 成员顺序第一个当 leader；默认使用 `maxGridX × 3/4` 矩形网格，一个 kernel 打全部投影；可通过 env 显式启用紧凑网格；**follower 的 `onEncode` 变 no-op** |
| `setupLNFusion(hiddenIn, residualIn, residualOut, gamma, eps)` | 把 residual-add + RMSNorm 折进 leader kernel 的前导 |
| `is2sgDecodePipeline()` | 是否走 `conv1x1_gemv_g4m1_2sg_wquant_sg` 管线——**融合的准入条件** |
| `isGateUpLeader/Follower()`、`isQKVLeader/Follower()` | 状态查询 |
| `getWeight/getBias/getDequantScale()` | leader 在融合 encode 时取 peer 的 buffer |

**kernel 侧**由编译宏区分变体：`GATE_UP_FUSED`（grid 加 z=2 维、强制 64 线程）、
`QKV_FUSED` / `QKV_FUSED_P4`（compact grid 用一维 TG 前缀和映射 projection；第 4 投影用 buffers 15-18、seg 6 floats）、
`LN_FUSED`。宏与 pipeline 缓存 key 的同步要求见
[`kernel-dev-and-optimize.md`](./kernel-dev-and-optimize.md) §1.4。

⚠️ **`ROW_2` 与融合正交但耦合**：三处 setup 里都是
`const bool row2 = !backend->isSupportTensorApi();`——**只在融合 leader 管线里编译，
且只对非 tensor-API 设备开**。`setupLNFusion` 里另有一处
`row2 && !mIsGateUpLeader && !mIsQKVLeader` 是 grid.x 减半的分支，不是宏门控，别混。

### 4.6 env 开关（`MetalEnv.hpp:140-142`）

| 开关 | 默认 | 说明 |
|---|---|---|
| `MNN_METAL_DISABLE_QKV_FUSION` | 开融合 | `=1` 关掉 q/k/v(/w) leader 装配 |
| `MNN_METAL_ENABLE_QKV_COMPACT_GRID` | 关（旧矩形网格） | `=1` 启用 `grid.x` 前缀和紧凑网格，其他融合保持不变；Mac M5 Pro 有收益，iPad M5 无收益 |
| `MNN_METAL_DISABLE_GATE_UP_FUSION` | 开融合 | `=1` 关掉 gate/up leader 装配 |
| `MNN_METAL_DISABLE_LN_FUSION` | LN 折入投影（两阶段关） | `=1` 改走独立 AddRMSNorm → 物化 normalized → 融合投影；Mac 有收益、iPad M5 无收益，故默认不启用两阶段（不影响 GatedRMSNorm）|

三个融合 DISABLE 开关语义 = 默认开，仅 `=1` 关闭；compact grid 是默认关的实验路径，
使用正向 ENABLE 开关。**所有 Metal env 走 `MetalEnv.hpp` 单一入口**，
后端代码禁止散落 `getenv`。目前唯一例外：`MetalLinearAttention.mm:689` 的
`MNN_METAL_DISABLE_LINEAR_ATTN_CONV_STATE_FUSION`。

---

## 5. `GatedRMSNorm`：另一条独立链路

Qwen3.5 linear-attention 输出门控段的融合，**不走 `FusedLinear`，也不受
`MNN_METAL_DISABLE_LN_FUSION` 影响**——它是一个独立 op（`OpType_GatedRMSNorm = 308`）。

### 5.1 链路

```
transformers.py::LinearAttention.forward   导出标准链：Reshape / RMSNorm / SILU / MUL
  → FuseTransformerC4  linear-attention 区域匹配成功后就地构造 GatedRMSNorm
       （canFoldGatedNorm 校验 RMSNorm/单 SILU/gamma 尺寸；LayerNorm 参数原样搬入，
        gamma/beta 保持 external weight 引用；同时删掉两侧 identity Reshape——
        kernel 索引已吸收两次 C4 重排，输出直接喂 out_proj。
        旧模型里导出期发出的 GatedRMSNorm op 仍被识别）
  → ShapeGatedRMSNorm / GeometryGatedRMSNorm
       OpCommonUtils::gatedRMSNormFusable 通过 ⇒ 整 op 透传给 Metal
       否则分解 LayerNorm + SILU + MUL（prefill 与非 Metal 后端都走分解）
       shape / fusable / Metal creator 均接受 external gamma（尺寸取 external[1]/4；
       非 mmap 时 createExecutionWithExternal 会在 creator 之前内联数据）
  → MetalGatedRMSNorm.mm  （kernel 设计见 kernel-dev-and-optimize.md §2.4.4）
```

**shape 契约**：x `[batch*heads, inside]`、z/out `[batch, heads*inside]`；decode 是 batch==1 特例。
⚠️ 首版硬编码 batch==1，prefill resize 直接 `Compute Shape Error`。

**开关**：converter 侧 `--transformerFuseC4`（llmexport 由 `--disable_transformer_c4` 传入，默认开）。
Pass 未开启或区域匹配失败时自然保留原始链。**运行时无 env 开关**。

### 5.2 ⚠️ geometry 保留门与 Creator 条件必须是同一个谓词

**2026-08-10 修的 bug**：首版 geometry 只判 `Metal && z batch==1`，
而 Creator 还会因 `inside%4!=0`、gamma/beta 缺失、非 useRMSNorm、非 NC4HW4、
设备不支持 simdgroup reduce 而拒绝。

此时 op 已经被 geometry 整体保留，**无法再退回分解**：

```
Creator 返回 nullptr → Pipeline 退到 backup CPU 后端
                     → CPU 没有 308 的 Execution
                     → "Create execution error : 308"
                     → Express 层 readMap() 为 null → SIGSEGV(139)
```

实测把合法用例的 `inside` 从 8 改成 6 即复现。

**修法**：条件收敛到 `OpCommonUtils::gatedRMSNormFusable`，**geometry 与 Creator 共用同一谓词**。
设备能力位由新增的 `STATUS_SUPPORT_SIMD_GROUP_REDUCE` runtime status 经
`GeometryComputer::Context::runtimeStatus` 透到 geometry
（Context 的 `allocBackend` 是 CPU backup，问不到 Metal 能力）。

> **新增任何 Creator 拒绝条件，都必须同步加进该谓词。**
> 这与 §4.4 的链式门控是同一类错误：**做保留决策的地方和做接受决策的地方必须看同一份条件**。

### 5.3 ⚠️ JSON 6 位截断陷阱（诊断方法论，不只是这个 op 的坑）

`MNNDump2Json` 打 float attr **只有 6 位有效数字**，gamma 128 个值里 95 个丢位
（`0.96484375` → `0.964844`）。后果：CPU fp32 从第 7 个 token 分叉；
而 **Metal fp32 对拍当时居然 token 全同**——旧融合读的是外部二进制权重（未截断），
新 op 读截断值，误差就在，只是该 prompt 上没翻转任何 greedy token。

> **教训**：跑文本对拍的"fp32 oracle"是 **token 级证据，不等于 bit 级证据**。
> 单一后端 + 单一 prompt 通过，不能下等价结论。
> 另外，排查这类问题**不能拿 `MNNDump2Json` 的输出做参照**——它显示时就已经截断了。

**当时的修复**：gamma 以 **fp32 位模式装进 int 列表**（`gamma_i` attr）过 JSON，
converter 侧 `struct.unpack` 还原。
**现状**：GatedRMSNorm 改由 converter 从原 LayerNorm 构造后，gamma 全程走 external
二进制权重，该 hack 已删除；本节保留作为诊断方法论。

---

## 6. 历史：后端图匹配已全部删除

2026-08-05 / 08-06 分三阶段把融合决策从"后端运行时猜"改成"导出图声明"。
写在这里是为了让人**看懂旧 commit、也别再往回走**。

| 已删除 | 原职责 | 现在由谁负责 |
|---|---|---|
| `matchQKVFusions` | 扫图找"同一输入恰好 3 个 conv1x1 消费者" | converter `fuseProjGroups` 构造的 `FusedLinear` 成员顺序 |
| `matchLNFusions` | 扫图找可折进投影的 LayerNorm | 导出侧 `ln_fold` + converter `foldBinaryLnIntoFusedProj` |
| `matchLinearAttnGateFolds` 的链匹配部分 | 找 gate/beta 链 | 导出期折叠；仅保留 STATIC re-home（改名 `applyLinearAttnGateFolds`）|
| `matchLinearAttnGatedNormFolds` | 找输出门控 7-op 链 | converter `FuseTransformerC4` 构造的 `GatedRMSNorm` op |
| 注册表 `mInputToConv1x1Group` / `mLayernormMap` | QKV/LN 匹配用 | `mFusedProjs`（按 op 注册，不是按图特征）|
| 注册表 `mOutputToConv1x1` | gate/up 从 `MetalBinary::onResize` 反查 | `setupFusion` 按成员顺序 |
| 注册表 `mRasterCopyMap` / `mCastMap` / `mRmsNormMap` / `mElemwiseMap` | gated-norm 匹配用 | 全删（`mElemwiseMap` 的唯一读者就是它）|
| `MetalRaster::setupGatedNormFusion/encodeGatedNorm/mGN*` | 在 Raster 上挂 gated-norm | `MetalGatedRMSNorm.mm` |
| Raster/Cast/Unary/Binary 的 `mFused` no-op | 被折叠者自我静音 | 不再需要（图上就没有这些 op 了）|

**保留**：`MetalGatedNormShader.hpp`（新 op 在用）、`MetalLayerNorm::mIsFused`
（fused-proj 的 LN 折叠用）、`MetalBackend::tensorsOverlap`（别名检测用）。

**至此 Metal 后端零图匹配。**

**为什么值得这么做**：
- 匹配器是**隐式契约**——导出图稍微变形（多一个消费者、插一个 ConvertTensor、
  `dimType` 换成 NHWC）就静默 miss，而且没有任何日志；
- 匹配顺序依赖 `onResize` 的执行时序，改一处会牵动别处（gate/up 曾经要从
  `MetalBinary::onResize` 反查 conv）；
- 每个后端都要重写一遍同样的模式识别。

**代价**：旧模型失去融合收益（输出不变），必须重导出。这是明确接受的取舍。

**⚠️ 唯一必须留在 `onResizeEnd` 的运行时动作**：`applyLinearAttnGateFolds` 的 STATIC re-home。
原因是 Pipeline 的 resize sweep（`_releaseTensor`）会释放 sweep 中提前获取的 STATIC home，
所以 re-home 必须晚于整个 sweep。

## 7. 导出侧的其它图优化

不属于 dispatch 融合，但同样是"在导出图上解决问题"的思路。

### 7.1 RoPE Fusion（`hasInvFreq`）

把 RoPE 的 cos/sin 计算折进导出图，运行时不再逐 token 重算。

⚠️ **inv_freq unsqueeze 陷阱**：

```python
# 错误：unsqueeze(2).unsqueeze(1) 把 [seq, dim] → [seq, 1, dim, 1]，dim 落到 num_heads 位置
# 正确：[seq, dim] → [1, seq, 1, dim]，与 query_states [bsz, seq, heads, head_dim] 正确广播
```

### 7.2 ❌ 导出期 QKV/GateUp **权重合并**（已闭合，勿重启）

与 §1 的"分组声明"不同，这个方案是把 q/k/v 的**权重张量物理拼成一个大矩阵**。
`MNN_EXPORT_MERGE_QKV` 整条链路（transformers.py 合并 + converter `fuseQkvPackedC4` +
RoPE/Attention 形状放行 + Metal `kBaseOffset`/`v_channel_offset`）已于 2026-07-24 **全部删除**。

**判负数据**：隔离测试 +1.1%，但**默认态（已有 leader/follower 融合）下 -4.1%**。
物理合并权重与 dispatch 融合是**互斥的两条路**，后者已经拿到了收益且不需要改权重布局。

### 7.3 RemoveDeadShapeOp 放宽

死 shape op 清理：op 数 1116 → 385，模型体积 **-33%**。
⚠️ **decode 性能 0 变化**——这些 op 本来就不在 GPU 时间线上。
价值在体积和图可读性，不要拿它当性能优化汇报。

---

## 8. 排查清单：融合没生效 / 融合后输出错

**没生效**（性能没变化、dispatch 数没降）：

1. **模型是不是新导出的？** 旧模型图里根本没有 `FusedLinear` / `GatedRMSNorm`。
   用 `MNNDump2Json` 看图里有没有这两个 op type。
2. **导出时有没有被 `--disable_fuse_*` / `--disable_transformer_c4` 关掉？**（默认全开，但脚本可能带了）
3. **是不是 prefill？** 融合 kernel 只在 decode-GEMV 管线存在（`is2sgDecodePipeline()`），
   prefill 全部逐成员 dispatch。
4. **`is2sgDecodePipeline()` 为什么 false？** 量化位宽 / oc 对齐 / pipeline 创建失败。
5. **env 关了吗？** `MNN_METAL_DISABLE_{QKV,GATE_UP,LN}_FUSION`。
6. **LN 没折？** 检查 converter 的 `has_ln` 是否置上（唯一消费者约束），
   以及运行时 `mLn->isNC4HW4() && isRMSNormWithGammaBeta()`。
7. **STATIC re-home 失败**（内存紧张时）会静默跳过融合。

**输出错 / 逐次不同**：

1. **先跑 `precision: high`（fp32）对拍**。fp32 bit-identical ⇒ 索引/布局/数学都对，
   问题在 fp16 rounding，多半可接受（本仓已合入的融合大多会改 fp16 token）。
2. **fp32 也不对 ⇒ 查内存别名**（§4.3）。特征是"输出偶尔不同"、
   对 `MNN_METAL_COMMIT_NUM=1` / `MNN_METAL_DISABLE_REPLAY=1` **不敏感**。
3. **稳定乱码 ⇒ 查链式门控**（§4.4）。用 env 逐项关融合二分：
   关掉某一项反而正确 ⇒ 那一项的前置依赖没检查。
4. **崩在 `Create execution error : <optype>` ⇒ geometry 保留门与 Creator 谓词分叉**（§5.2）。
5. **对拍前先预热一次**（pipeline cache 冷/热不可比），且**不要用 `MNNDump2Json` 的
   数值当参照**（6 位截断，§5.3）。
