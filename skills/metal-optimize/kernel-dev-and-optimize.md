# Metal Kernel 开发与优化

> **配套 SKILL.md 的 sub-doc**：新增 Metal kernel 的**命名、写法**，以及 kernel 层的**优化方法与知识库**。
>
> 结构：**第一部分 开发规范**（写任何新 kernel 前必读）；**第二部分 优化知识库**（GEMV / GEMM / Attention / 其他 kernel 的已落地优化、实测数据与避坑）。
>
> 融合（导出图声明 + 运行时单 dispatch）见 [`graph-fusion.md`](./graph-fusion.md)；管线同步 / replay / H2D 见 [`runtime-scheduling.md`](./runtime-scheduling.md)。

---

# 第一部分：Kernel 开发规范

## 1.1 核心原则

1. **shader 是嵌入的 C++ 字符串**。Metal kernel 写在 `*Shader.hpp` 里的 `R"metal(...)metal";` 字符串中（如 `ConvSimdGroupShader.hpp`），不是独立 `.metal` 文件。改完直接 `make` 就拿到，不需要 codegen。**字符串拼接顺序决定 `#define` 作用域**——「宏 alias 陷阱」特别依赖这点。
   - ⚠️ 旧路径 `source/backend/metal/shader/*.metal` 经 `makeshader.py` 生成 `AllShader.cpp/hpp`，属历史遗留（kernel 名形如 `main0`）。**新 kernel 一律走 `*Shader.hpp` 字符串**，不要往 `shader/` 里加。

2. **dispatcher 要先摸清**。Metal conv1x1 同一 op 常有多条 kernel（gemv 多种 / gemm 多种 / outer dequant），按 `area`、`oc`、`ic_4` 等 case 切。新加 quant bit 不可能一次扩完所有路径，**必须先决定支持哪几条 + 让其他路径不被 dispatch**。

3. **Apple GPU 是开发 + 主测平台**。Mac M3/M4/M5 数字真实可用，但**这条路不代表 Android**，Vulkan/OpenCL 上同算法行为可能完全不同；iPhone A 系列与 M 系列在调度 / occupancy 上差异显著，性能数字不能互推。

4. **正确性 oracle 先于性能**。CPU / `temperature=0` greedy 对拍前 N token 是黄金标准。

## 1.2 入口定位与 dispatcher 结构

```bash
grep -rn "OpType_<MyOp>" source/backend/metal/                    # Execution
grep -rn "kernel void <my_kernel>" source/backend/metal/*.hpp     # shader 字符串入口
```

低 bit 量化 conv 入口：`MetalConvolution1x1::onResize`。识别 quant：`mDequantBits ∈ {2,3,4,8}`（在 `MetalConvolutionCommon::loadWeight` 里设置）。`onResize` 把 `area` × `oc` × `dequantInShader` × simdgroup 能力组合分流到多条路径，扩 quant bit 前先把目标 shape 走的那条 case 标出来。

**当前 conv1x1 低 bit 量化路径**：

```
mDequantScaleBias && dequantInShader (area<128 或不支持 simdgroupMatrix)
  ├─ supportSimdGroupReduce && area <= short_seq=6   (decode-friendly)
  │    ├─ area > 1 → conv1x1_gemv_g4mN_wquant_sg
  │    ├─ oc > 16384 && oc_4 % 2 == 0 → conv1x1_gemv_g16_wquant_sg
  │    └─ else → conv1x1_gemv_g8_wquant_sg
  └─ supportSimdGroupMatrix && area > short_seq && oc > 8 → conv1x1_gemm_*_wquant_sg

mDequantScaleBias && !dequantInShader (area>=128 + simdgroupMatrix)
  → conv1x1_w_dequant + conv1x1_gemm_32x64_split_k_sg  (outer dequant + fp gemm)
```

新加 quant bit 时典型组合是 decode gemv path + prefill outer-dequant path，其他 gemv/gemm 实例 dispatcher 显式 fallback；一次性扩完所有 path 工作量太大。

## 1.3 shader 组织与命名约定

### 文件组织

新 kernel 写进对应的 `*Shader.hpp`，形式固定：

```cpp
static const char* gLinearAttnGatedNorm = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void linear_attn_gated_norm(...) { ... }
)metal";
```

- C++ 常量命名 `g` + 大驼峰（`gBasicConvPrefix`、`gPrefillFlashAttn`、`gDecodeSplitKV`）。
- 文件头统一 `#if MNN_METAL_ENABLED`；transformer 专用的再套 `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE`。
- **公共头通过 C++ 字符串拼接共享，不是 `#include`**：

  ```cpp
  // MetalConvolution1x1.mm:159
  std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
  ```

  `gBasicConvPrefix`（`ConvSimdGroupShader.hpp:11`）提供 `#include <metal_stdlib>`、`conv_activation_type`、`activate()`、`conv1x1_constants` 结构体，以及 `namespace MNN { uchar4x2 / char4x4 }` 等打包类型（:56-107）。写新 conv 类 kernel 就拼这个前缀，不要重复声明。
- `ftype` / `ftype2` / `ftype4` / `ftype4x4` 由编译宏注入（fp16 或 float），shader 里直接用。

### kernel 命名

小写下划线，段序：

```
算子_阶段_形状|tile_变体_后缀
```

| 段 | 约定 | 例 |
|---|---|---|
| 算子 | `conv1x1` / `prefill` / `decode` / `linear_attn` / `binary` | |
| 阶段 | `qk` / `qkv` / `softmax` / `prep` / `w_dequant` | `prefill_qk` |
| GEMM tile | `MxN` | `conv1x1_gemm_32x64_split_k_sg` |
| GEMV 分组 | `g<组数>m<每组行数>` + `<n>sg` | `conv1x1_gemv_g4m1_2sg` |
| 向量宽度 | `x1` / `x4` / `x16`；`c4` = NC4HW4；`c2` = vec2 | `binary_layernorm_c4_rms_sg` |
| 量化 | `wquant`（**位宽走编译宏，不进名字**）| `conv1x1_gemv_g8_wquant_sg` |
| 算法变体 | `split_k` / `rms` / `fused` / `align` / `chunk64` / `inplace` / `nax` | |
| 后缀 `_sg` | **依赖 simdgroup 原语**（需 `rt->supportSimdGroupReduce()` / `supportSimdGroupMatrix()`）| |
| 版本 | `_v2` / `_v4` = 算法迭代 | `linear_attn_gated_delta_rule_sg_v4` |

前缀谓词放最前（`binary_layernorm_c4_rms_sg` = binary 前导 + layernorm + C4 + RMS + simdgroup）。

**已知不一致（照抄前先确认）**：
- `decode_qk_softmax` 同名 kernel 在 `MetalAttentionShader.hpp:2167/2298/2461` 出现 3 次，靠 `#if` 互斥选择——同名不同体是允许的，但新 kernel 别学。
- `MetalAttentionShader.hpp:1143` 的 kernel 就叫 `copy`；`AllShader` 里的旧 kernel 叫 `main0`。
- `linear_attn_gated_norm` 用了 `simd_sum` 却没有 `_sg` 后缀。
- kernel 名与 Execution 类名**不要求**一致。

## 1.4 编译期变体：宏 + pipeline 缓存 key

代码库**没有任何 `MTLFunctionConstantValues`**，全部变体走 `MTLCompileOptions.preprocessorMacros`：

```objc
// MetalGatedRMSNorm.mm:83-88
MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
option.preprocessorMacros = @{
    @"ftype"      : @(ftype.c_str()),
    @"ftype4"     : @(ftype4.c_str()),
    @"SGS_PER_TG" : @(std::to_string(sgsPerTG).c_str()),
};
```

布尔开关用 `[dic setValue:@"1" forKey:@"XXX"]` 配 shader 里的 `#ifdef`；shader 中可调的取值型宏要写 `#ifndef / #define` 兜底默认值。

**pipeline 缓存 key 是 `std::vector<std::string>`**（`MetalBackend.hpp:101-106`，实现 `MetalBackend.mm:1609-1625`）：key[0] = kernel 名，key[1] = ftype，之后每个**生效的宏**依次 append。

```cpp
// MetalGatedRMSNorm.mm:79-80
std::vector<std::string> keys = {"linear_attn_gated_norm", ftype, "sgs" + std::to_string(sgsPerTG)};
```

`MetalAttention.mm:192-212` 是最规范的范式（多变体、互斥判断齐全）。

> ⚠️ **新增变体宏必须同步改 4 处，漏一处就是静默错误**：
> 1. shader 里的 `#ifdef` 分支（含 `#ifndef` 默认值）；
> 2. `keys.emplace_back(...)`——**漏了会取到别的变体的缓存 pipeline**；
> 3. `[dic setValue:... forKey:...]`——漏了宏根本没生效；
> 4. `onResize` 里因该宏改变的 grid / threadgroup 尺寸。
>
> 第 4 条例：`MetalConvolution1x1.mm:177-184`，`ROW_2` 把 grid.x 改成 `UP_DIV(slice,4)`，`GATE_UP_FUSED` 加 z=2 维并强制 64 线程。

其他约定：
- 取值型宏的 key 要带前缀区分（`"HEAD_DIM_" + head_dim_str`，避免与别的数字宏撞串）。
- 互斥的融合宏要显式挡住叠加（`MetalAttention.mm:198-201`）。
- 编译失败用 `pipelineCompileFailed(keys)` 记住，避免每次 resize 重试。

## 1.5 Execution 骨架与注册

继承 `MetalExecution`���`MetalExecution.hpp:19-24`），只需实现 `onResize` + `onEncode`。**最小范本：`MetalGatedRMSNorm.mm:37-167`**。

`onResize`（:58-111）标准流程：

```
算 shape → 不支持则 return NOT_SUPPORT
        → findPipeline(keys)；miss 则建宏 + makeComputePipelineWithSourceOption + insertPipeline
        → 校验 maxTotalThreadsPerThreadgroup
        → getConstBuffer 填参数
        → 存 mThreads
```

**buffer index 约定**：0 起是输入，紧接输出，然后 param，最后权重。
- `MetalGatedRMSNorm`：`0=x, 1=z, 2=out, 3=param, 4=gamma, 5=beta`
- 经典 conv：`0=in, 1=out, 2=const, 3=weights`

张量必须用 `MetalBackend::setTensor(tensor, encoder, idx)` 绑定（处理 buffer+offset）。

**grid 计算**：simdgroup kernel 手算，例如 `MTLSizeMake(1, UP_DIV(outside, sgsPerTG), 1)` × `MTLSizeMake(sgsPerTG*32, 1, 1)`；普通 elementwise 用 `computeBestGroupAndLocal`。

**注册**：`REGISTER_METAL_OP_CREATOR` 与 `REGISTER_METAL_OP_TRANSFORMER_CREATOR` 宏展开完全相同（`MetalBackend.hpp:483-491`），区别只在约定与调用点——transformer 版在 `MetalOPRegister.mm` 里被 `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` 包住。

CMake 门控掉的 kernel，`#else` 分支必须提供同名空函数，否则注册表链接失败：

```cpp
// MetalGatedRMSNorm.mm:168-171
#else
void ___MetalGatedRMSNormCreator__OpType_GatedRMSNorm__() {}
#endif
```

## 1.6 通用陷阱

### 陷阱 A：宏 alias 让 `#ifdef` 多分支同时为真

最严重的 Metal 坑。给"未扩展的 kernel"在新 quant bit 下编译过，常加 alias：

```c
#if defined(W_QUANT_2) && !defined(W_QUANT_4) && !defined(W_QUANT_8)
#define W_QUANT_4    // 让其它 kernel 还能编译
#endif
```

**坑**：alias 让 `#ifdef W_QUANT_4` 在你想扩展的那个 kernel 里**也**被命中。signature 里 `#ifdef W_QUANT_2` 第一个匹配（`uchar4* wt`），body 里 `#ifdef W_QUANT_4` 也匹配（`MNN::uchar4x2 w_int4 = xy_wt[z]`）→ 类型混淆 → 编译过 → 数值错。

修法：扩展的 kernel 里**所有相关 `#ifdef` 必须按 W_QUANT_2 → W_QUANT_3 → W_QUANT_4 → W_QUANT_8 顺序**，新 bit 优先匹配。signature 和 body 都要这个顺序，少一处都 sneaky 错。

**真实回归实例**：`20e5d03f3` 给 g8 kernel 加了 W2/W3 分支（signature 顺序正确），后来 `b71528f0d` 给 g8 body 加 W4 deferred 分支时把 `#ifdef W_QUANT_4` 放在 body 阶梯最前——alias 让 W2/3 编译同时命中 W4 body 分支（`uchar4x2` vs `uchar4*` 类型冲突）→ **所有 W2/3 decode pipeline 编译失败**，静默持续数周。教训：改 alias 字符串内任何 kernel 的 `#ifdef` 阶梯前，先确认该 kernel 是否在 W2/3 下编译。

### 陷阱 B：dispatcher 漏路径（lm_head）

LLM 的 lm_head conv（`oc = vocab_size ~150k`）走 `oc > 16384` 的特殊路径（如 `g16`）。新加 quant bit 没扩 g16 时，dispatch 还会进 g16 → 用错 layout 读 buffer → 数值错或 crash。

应对：dispatcher 选路写白名单，把没扩的路径强制 fallback 到已扩的（如 `g8`）。

### 陷阱 C：weightTransform 的多签名同步

`weightTransform(...)` 在 `MetalConvolutionCommon`、`MetalConvolutionWinograd`、`MetalConvolutionDepthwise` 都有 override。改签名（如加 `subBits` 参数）时这 3 处 + `.hpp` 共 4 处要同步，否则 build 报 `'override' but does not override any member function`。

### 陷阱 D：getDequantScale 的 `coef` fp16 范围补偿

Metal `getDequantScale` 用 `coef = 1000/max_data` 做 fp16 范围补偿（host 写 `s*coef`，shader `/coef`）。新加 quant bit 不要碰这个流程；scale/offset 的 originOffset 折叠**完全在 host alpha 写入时**完成，shader 一律按 signed 解出后 `signed_w * scale + bias` 即可，不要再折一次。

### 陷阱 E：tile 内 byte index 选择（OC vs K_inner）

W_QUANT_8 的 tile layout 是 `byte = ro * 4 + ri`（OC 外、K_inner 内），即 `xy_wt[z]` 取 16 字节 = 一个 (4 OC, 4 IC) tile，`w[i] = char4` 是 1 OC × 4 IC。

新 bit 的 packing 必须**镜像已有最高 bit kernel 的字节顺序**：byte i = OC i 的多个 IC，不能反过来变成"byte i = IC i 的多个 OC"。写反了 shader 照样编译通过、kernel 能跑，但输出乱码——只有 dump 第一个 op 的 weight 前几行和 CPU 对照才能发现。

### 陷阱 F：fp16 后端下 `Tensor::createDevice<float>` 只给一半字节

Metal 后端开 fp16（`useFp16InsteadFp32`）时，float 类型的 device tensor 按 **2 字节/元素**分配；若 shader 把这块 scratch 按 `device float*`（4B）读写，**buffer 实际只有需求的一半**。越界不会立刻炸：先静默踩别的内存（数值损坏），直到撞上未映射页才报 GPU 故障（`kIOGPUCommandBufferCallbackErrorInnocentVictim`，且真凶 buffer 常不在日志里）。

**前科（2026-07-29 split-KV 越界）**：`mTempSplitKV` 用 `createDevice<float>({B*H*32*(HD+2)})` 分配，shader 端 `device float* tmp_out`。分配 132KB、kernel 要 264KB ⇒ **nwg>16（kv>4096）必越界**。历史测试全 ≤kv4096（nwg=16 恰好贴边），Qwen3.5（HD=256）p4096+tg1000 才引爆；HD=128 模型同条件只是静默损坏不崩。

定位手段：`MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1` 直接报出 kernel 名与越界 offset。

**规矩**：shader 按 `device float` 消费的 scratch，host 一律 `createDevice<uint8_t>({bytes})` 按字节分配，公式写明 `* sizeof(float)`。检查现有代码时 grep `createDevice<float>` 与对应 shader 的指针类型是否一致。

## 1.7 Packed weight 设计

新加 quant bit 时**先固定 5 个量**：

| 量 | 解释 |
|---|---|
| tile = (IC_inner × OC_inner) | Metal conv1x1 一次原子访问的最小区块 |
| 字节/tile | 由 bit 决定，镜像已有 bit 的 stride |
| byte index 内的语义 | 与已有最高 bit kernel 的 byte ↔ (oc, ic) 映射保持一致 |
| bit 顺序 | 与 host packing / 跨后端约定一致 |
| signed/unsigned 存储 | 存 unsigned，shader 内减 offset 还原 signed |

**bit 不齐 32 位时的 split layout**（如 3bit）：低 2 bit 一段 + 高 1 bit 另一段，避免跨 word 边界。host packing 与 shader unpack 双向严格镜像。

**示例（w3 = 6B/tile）**：bytes 0..3 装低 2 bit（与 w2 layout 一致），bytes 4..5 装高 1 bit（byte 4 = OC{0,1} 的 high bit、upper nibble = OC even / lower = OC odd，byte 5 = OC{2,3}）。每个 nibble 内 bit `3-k` 对应 IC k 的 high bit。比"32 weights = 12B 跨 word 边界"方案更友好，shader 用一次 `vload8 + vload4` 就能取到一个 (4 IC × 8 OC) tile。

## 1.8 修改流程与检查清单

```bash
vi source/backend/metal/ConvSimdGroupShader.hpp      # 直接编辑 .hpp 里的字符串
cd build && cmake .. -DMNN_METAL=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j8 MNN llm_demo
```

**新加 `W_QUANT_N` 同步检查清单**：

| 位置 | 检查 |
|---|---|
| kernel signature | `#ifdef W_QUANT_N` 分支声明 `wt` 类型 |
| kernel body | `#ifdef W_QUANT_N` unpack 分支，**优先级在 W_QUANT_4 之前** |
| 宏 alias 块 | `#if (defined(W_QUANT_N) \|\| ...) && !defined(W_QUANT_4)` 让未扩展 kernel 编译过 |
| weightTransform | CPU pack 路径（`subBits == N` 分支）|
| `MetalConvolution1x1.mm` `mDequantBits` | `useIntN ? N : (int4Path ? 4 : 8)` |
| dispatcher 选路 | `mDequantBits == N` 时设 `W_QUANT_N` 宏，避开未扩展 path |
| 融合 setup 的 keys/dic | 三处 fusion setup 都要补新 bit，**漏改是静默退融合** |
| prefill（multi-token）| `(mDequantBits == N) && area > 1` 时 force `dequantInShader = false` |

**跨分支移植优化前先核对隐含前置**：性能提交可能基于已 squash 的基础设施提交开发，单看补丁看不出依赖。移植前用 `git log -S'<新增 override/API>' -- <目录>` 定位接口来源，确认目标分支是否已有实现；没有就只补最小前置提交，不要整文件采用来源分支的公共 registry/backend 文件。解决冲突后检查：核心实现与来源分支目标行为一致；被新 kernel 取代的旧 shader、pipeline 成员、dispatch 分支必须一起删除并用 `rg` 确认零引用。最后 clean rebuild + 至少一次真实 Metal 运行——**普通 C++ 编译发现不了运行时 Metal shader 编译错误**。

**编译错调试**：Metal 编译错在运行时打 log（`Warning: pipelineWithSource error`）：

| 错误 | 原因 |
|---|---|
| `use of undeclared identifier 'wt'` | 某 `#ifdef` 分支没声明 → alias 没设对，或新 bit 没补 signature |
| `no viable conversion from uchar4 to uchar4x2` | 多个 `#ifdef` 同时为真，body 命中错的分支（陷阱 A）|
| 编译过但乱码 | tile byte 顺序反了；或 dispatcher 漏路径 |

## 1.9 正确性验证

```bash
cd build && make -j8 llm_demo MNN
sed 's/"backend_type": "cpu"/"backend_type": "metal"/' <model>/config.json > <model>/config_mtl.json
DYLD_LIBRARY_PATH=build:build/express build/llm_demo <model>/config_mtl.json /tmp/prompt.txt
```

CPU / Metal 同 prompt + `temperature=0.0` 前 N 个 token 应一致（fp16 误差内）。**模型本身可能就坏**（小模型在低 bit 上量化退化常见），先用更大模型 baseline CPU 跑通，再验 Metal kernel。

数值偏差容忍：fp16 路径 abs < 1e-2 / rel < 5e-3；量化 dequant + fp16 abs < 1e-1。

**低 bit 专用 oracle**：低 bit 小模型输出常是乱码（W2 0.6B 即使 HQQ 也半乱码），而 CPU oracle 不可用（CPU int2/3 ARM kernel 在 Apple Silicon 上有 bug：W2 乱码、W3 SIGSEGV，未修）。可用 oracle = **`transformers/llm/export/mnn_quant_ref.py`**：从导出的 `.mnn.weight` 直接解码权重（header + MSB-first 位解包 + fp16 alpha）注入 HF 模型 greedy 生成，独立于 MNN 运行时。判据：
- 连贯模型（W3/W4）逐 token 一致；
- 乱码模型看共同前缀 + **噪声敏感度标定**：给 oracle logits 加 ε 噪声，若 1e-3 不翻转、1e-2 在与 Metal 分叉点相近位置翻转 ⇒ Metal 偏差是 O(1e-2) 精度级 = 合法；O(0.1+) 才是 bug。

⚠️ **对拍口径**：`llm_demo` 输出内嵌 `cost time` 行与末尾统计块，直接 hash 整个 stdout 会让**默认稳定配置也"每次不同"**。对拍必须先剔除计时行。

⚠️ **冷/热**：刚删 `mnn_cachefile.bin` 的第一次跑与后续不同（pipeline cache），所有对拍前先预热一次。

## 1.10 Metal tensor API：cooperative tensor 的逐元素布局

MNN 现有 tensor 用法（`prefill_qk_tensor`、`conv1x1_fused_q4_gemm_stage`）只用到「destination cooperative tensor + `run(sA,sB,cT)` + 一次性 `cT.store()`」。要写**融合 attention** 这类需要在两次 matmul 之间对中间结果做 online softmax 的 kernel，必须用到三个额外能力（已在 M5 / macOS 26.6 / `applegpu_g17g` 实测可编译）：

| 能力 | 说明 |
|---|---|
| `get_left_input_cooperative_tensor<A,B,C>()` / `get_right_...` | 让 A/B 也是寄存器 coop tensor，把上一次 matmul 的 destination 直接当下一次的 A（P 不落内存）|
| coop tensor 的 `operator[]` | 逐元素读写，softmax/mask 全在寄存器完成 |
| `run(ct_a, ct_b, ct_c)` | 三个操作数都是 coop tensor |
| `get_capacity()` / `get_multidimensional_index(i)` | 官方坐标查询（MLX 没用，它硬编码经验公式）|

⚠️ **硬约束**：input cooperative tensor 只允许单 simdgroup 作用域（`MPPTensorOpsMatMul2dImpl.h:3295`）⇒ 必须 `metal::execution_simdgroup`，每个 simdgroup 独立跑一个小 matmul（MLX NAX 用 **16×32×16**）。MNN 现有的 `execution_simdgroups<4>` + threadgroup source tensor 写法**无法表达 online softmax**。

⚠️ 能力探针在 `MetalBackend.mm` 的 `mSupportTensorCoopInput`（与 `mSupportTensorApi` 分开）。

**逐元素布局**（`matmul2d_descriptor(16,32,16, false, TB, true, multiply_accumulate)` + `execution_simdgroup`，实测 dump）：

```
qid = lane >> 2
fm  = (qid & 4) | ((lane >> 1) & 3)      // 0..7   慢轴基址
fn  = ((qid & 2) | (lane & 1)) * 4       // 0/4/8/12  快轴基址（每 lane 连续 4 个）
```

`get_multidimensional_index(i)` 返回 **(dim0, dim1) = (快轴, 慢轴)**：

| 操作数 | 形状（存储序）| capacity | 元素 i 的坐标 |
|---|---|---:|---|
| A（left, 不转置）| M=16 × K=16 | 8 | `K = fn + (i&3)`，`M = fm + (i>>2)*8` |
| B（right, `tb=true`）| N=32 × K=16 | 16 | `K = fn + (i&3)`，`N = fm + ((i>>2)&1)*8 + (i>>3)*16` |
| B（right, `tb=false`）| K=16 × N=32 | 16 | `N = fn + (i&3) + (i>>3)*16`，`K = fm + ((i>>2)&1)*8` |
| D（destination）| M=16 × N=32 | 16 | `N = fn + (i&3) + (i>>3)*16`，`M = fm + ((i>>2)&1)*8` |

推论（写融合 attention 直接用）：
- **每 lane 只持 2 个不同的 M 行**（`fm` 与 `fm+8`）⇒ online softmax 的 running max/sum 是 `float2`。
- **同一 M 行的 lane 只在 bit0 与 bit3 上不同** ⇒ 行归约 = `simd_shuffle_xor(v,1)` + `simd_shuffle_xor(v,8)`。
- `i>>3` 选第二个 16 宽 frag。

> ⚠️ Apple 文档明确说 coop tensor 布局是 implementation-defined（`MPPTensorOpsMatMul2d.h:216-225`）。上表是本机实测，**换设备/OS 需重跑 dump**。快速 dump：用 MLX 的 `mx.fast.metal_kernel` 跑一个 32 线程 kernel，把每 lane 每元素的 `get_multidimensional_index(i)` 写进输出 buffer 再打印。

## 1.11 Apple GPU 杠杆选择

### Metal 特有杠杆

- **simdgroup matrix（sg_matrix）for prefill**：`area > short_seq` + 支持 simdgroupMatrix 时走 `gemm_*_wquant_sg`，比 outer-dequant + fp gemm 快一档。sg_matrix kernel 每个 quant bit 单独实例化，新加 bit 想覆盖 prefill 必须扩它。
- **simdgroup reduce（sg_reduce）for decode**：`area == 1` 走 `gemv_*_wquant_sg`，依赖 `simd_sum`。WGS 通常 128（4 simdgroup × 32 lane）。
- **g4mN 模板化**：`conv1x1_gemv_g4mN_wquant_sg` 是 template `<int AREA_THREAD>`，按 area 实例化 N。
- **Metal4 tensor API**：M5+ 的 `matmul2d` + cooperative tensor（§1.10）。

---

# 第二部分：Kernel 优化知识库

> 案例以 LLM decode/prefill 为主战场（优化密度最高、数据最全），但 GEMM/GEMV/attention 技巧对 CNN / Diffusion 同样适用。

## 2.0 优化总纲

LLM decode 每步生成一个 token，核心链路：

```
RMSNorm → Q/K/V Linear(GEMV) → RoPE → Attention(QK+Softmax+AV) → O Linear(GEMV)
       → RMSNorm → Gate/Up Linear(GEMV) → SiLU*mul → Down Linear(GEMV) → Residual
```

- **Decode**：GEMV 占 60-80% 时间，是优化主战场；其次 Attention 和 RMSNorm。
- **Prefill**：GEMM 占 ~50%；Attention 三段中间物化（mTempQK / mTempSoftMax）是长 prompt 显存/带宽瓶颈，causal-bound / flash-attention 是主杠杆。
- **战略**：**Prefill 走 kernel 深化，decode 走管线深化**（decode 侧管线优化见 [`runtime-scheduling.md`](./runtime-scheduling.md)）。

### 关键定性结论：小模型 decode 是 GPU-bound 且 occupancy 受限

counter profiler 实测（0.6B / M4 Pro / p512 稳态 60 forward 平均）：

| 指标 | 实测 |
|---|---|
| GPU dispatch / token | **266**（层内 9 × 28 + 层外 ~28）|
| GPU busy / token | **2950 us** |
| 生产 wall / token（同期 341-348 t/s）| **2874-2933 us** |

**GPU busy ≈ 生产 wall ⇒ 生产 decode 基本 100% GPU-bound，没有可回收的空泡。** 早期"单流 GPU 利用率仅 ~60%"/"空泡 ~29%"的解读**是错的**：双实例并发 1.41× 不代表有 29% 空闲，而是**单实例 kernel 填不满 GPU（occupancy 不足）**——是 occupancy 效应，不是 idle-gap 效应。

decode GPU 时间去向（2950us/token）：

| op | 调用/token | us/token | 占比 |
|---|---:|---:|---:|
| gate_up 融合 GEMV | 28 | 624 | 21.2% |
| o_proj + down_proj GEMV | 56 | 579 | 19.6% |
| qkv 融合 GEMV | 28 | 502 | 17.0% |
| lm_head g16 | 1 | 352 | 11.9% |
| **GEMV 小计** | 113 | **2057** | **70%** |
| Attention qk_short + av + copy | 84 | 644 | 22% |
| RoPE | 28 | 127 | 4.3% |
| BinaryOp / LayerNorm / Raster / Cast / Unary | ~41 | ~120 | 4% |

换算成带宽——**这才是真正的剩余 headroom**（M4 Pro 峰值 ~273 GB/s）：

| dispatch | 权重量 | 实测带宽 | 峰值占比 |
|---|---|---:|---:|
| qkv 融合（17.9us/call）| 2.0 MB | 112 GB/s | 41% |
| o/down（10.3us/call）| ~1.3 MB | 125 GB/s | 46% |
| gate_up 融合（22.3us/call）| 3.1 MB | 141 GB/s | 52% |
| **lm_head（352us/call）** | **77.8 MB** | **221 GB/s** | **81%** |

小 GEMV 只到 41-52% 峰值，lm_head 到 81%——差别纯粹是**单 dispatch 体量**（2-3MB 撑不起 ramp-up 与足够在途读）。若小 GEMV 能达到 221 GB/s，GEMV 从 2057us → ~1210us，decode 从 341 → **~479 t/s**。SPLIT_K_2 + ROW_2 已把它从 85 推到 112-141 GB/s，**剩余 ~60% 空间是 occupancy / 在途读问题**（非布局：连续度已证无效；非图结构：DCE 实测 0 变化）。

> 4B 及以上不受管线约束（GEMV 占 67%、GPU busy 逼近 wall），GPU 优化直接兑现——**优化项要按模型档分别评估**。

## 2.1 GEMV（decode 主战场）

### 2.1.1 Q4 GEMV Deferred Dequantization（+28%，系列最大单项）

**问题**：标准 Q4 GEMV 在累积循环内层同时做 nibble 解包 + 反量化（乘 scale + bias）+ FMA。反量化的 fp16 乘加是瓶颈。

**优化**：延迟反量化——内层只做整数累积（int8 × int8 → int32），循环结束后一次性反量化：

```metal
// 旧：每步反量化
for (k) {
    half w = dequant(packed_w[k]);   // 每步 fp16 乘加
    sum += input[k] * w;
}

// 新：延迟反量化
int32_t isum = 0;
for (k) {
    int8_t w = unpack(packed_w[k]);  // 只做整数解包
    isum += int32_t(input_quant[k]) * int32_t(w);
}
sum = half(isum) * scale + bias;     // 循环外一次反量化
```

**实现要点**（`ConvSimdGroupShader.hpp` + `MetalConvolution1x1.mm`）：
1. **输入也要量化**：input 从 fp16 动态量化为 int8，host 端分配量化 buffer 与 scale buffer；
2. **双 buffer**：`mTempInput`（量化后 int8）+ `mInputScales`（per-row scale）；
3. kernel 内先对 input 做 per-row absmax 量化，然后整数 GEMV，最后
   `result = isum * input_scale * weight_scale + weight_bias * input_sum`；
4. **input_sum 修正**：weight 是非对称量化（有 zero point），需额外累积 `sum(input_quant)` 做 bias 修正。

**门控**：`area == 1`（decode）+ `supportSimdGroupReduce`。
**性能**：Qwen3-0.6B Q4 / M4：标准 GEMV ~140 tok/s → ~180 tok/s（**+28%**）。

### 2.1.2 双 Simdgroup GEMV + ushort4 向量加载（g4m1_2sg）

**问题**：单 simdgroup GEMV 的 occupancy 受寄存器压力与 simdgroup 数量限制；weight 读取粒度 `uchar4`（4B）未充分利用 burst。

**优化**：① 一个 threadgroup 内 2 个 simdgroup 分别处理不同 OC 范围，input 经 threadgroup memory 共享，TG 数减半；② weight 用 `ushort4`（8B）一次读取，load 指令数减半。

```metal
kernel void conv1x1_gemv_g8_deferred_sg2(
    ...
    uint sgid [[simdgroup_index_in_threadgroup]],  // 0 or 1
    uint lid  [[thread_index_in_simdgroup]], ...) {
    int oc_start = gid * 16 + sgid * 8;
    threadgroup half shared_input[IC_CHUNK];
    if (sgid == 0) { /* 协作加载 input */ }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 各 simdgroup 独立 GEMV，simd_sum reduction 在 simdgroup 内完成
}
```

**注意**：`ushort4` 需 weight buffer 8 字节对齐；双 simdgroup 要求 OC ≥ 16，小 OC 层仍走单 simdgroup kernel。

### 2.1.3 Pre-scaling Nibble Extraction（约 +5%，叠加在 deferred dequant 上）

标准解包的 `>> 4` / `& 0xF` / `- 8` 各占 ALU 指令。**Pre-scaling trick**：host 端 pack weight 时预乘系数，nibble 提取用乘法（MAD）替代 shift，同时隐式完成 zero point 减法——mask-only 解包，位权补偿预乘进输入。

### 2.1.4 GEMV 带宽效率画像与路线结论

同一 kernel `g4m1_2sg` 在不同 dispatch 权重下的带宽（M4 Pro，decode）：

| 场景 | 每 dispatch 权重 | 实测带宽 | 效率 |
|---|---|---|---|
| 0.6B 独立 GEMV（q/k/v/o/down）| ~0.8MB | **85GB/s** | 35% |
| 4B 独立 GEMV | ~5.9MB | 173GB/s | 71% |
| 4B gate_up fused | 28.8MB | 206GB/s | 84% |
| 4B lm_head g16 | 225MB | 226GB/s | 92%（到顶）|

⇒ **小权重 GEMV 是 latency-bound 不是 kernel-bound**：正确解法是减少 dispatch 数 / 增大单 dispatch 体量，不是继续调 kernel。

**路线结论**：g4m1_2sg 的 lane/TG 配比微调曾 4 次证伪（WIDE_MIDDLE / 4SG / unroll / split-K 早期版），但 **SPLIT_K_2 第 5 次成功**——关键差异是保留 pre-scaling 内循环、行内 K 流对半拆给 2 个 simdgroup（在途读加倍），而非改 lane 划分。M5 上 middle_step / VEC2 / VEC4 变体中性偏负。lm_head g16 已 182-226GB/s 接近上限，headroom 小。

### 2.1.5 Split-K Decode GEMV（SPLIT_K_2，+3.3~3.9% e2e，默认开）

`db13a99f4f`。

- **问题**：2sg GEMV 每行由 1 个 simdgroup 串行流式读整行，小投影（0.5-1.8MB 的 o/down/qkv）只有 88-137GB/s，而 87MB lm_head 达 252GB/s——小矩阵行数少、每行在途读不足，latency-limited。
- **机制**：`SPLIT_K_2` 保留 kernel 原有 pre-scaling 内循环，改为 4 simdgroup/threadgroup——每行的 quant block 对半拆给 2 个 simdgroup，各算半段部分和，经 threadgroup memory 合并，行内在途读加倍。
- **门控**（`MetalConvolution1x1.mm:753-769`）：只在 `area==1` 的**普通** decode GEMV 分支，条件 `gemvSplitK > 0 && oc%8==0 && blockSize%2==0`，dispatch `UP_DIV(oc,8) × 128 线程`。融合 leader 随后用自己的 2sg 管线覆盖。
  - **融合管线上扩展 SPLIT_K_2 已证伪**（p512 -3.3% / p2048 -1.8%），融合管线的正确解是 §2.1.6 ROW_2。
- **实测**（M4 Pro 0.6B）：tg256@p12 +3.9%、tg128@p512 +3.8%、tg128@p2048 +3.3%；greedy byte-identical。
- **反例存档**：先路由到现有 g8 kernel 的方案 e2e **-5%**——g8 的 nibble-unpack 内循环慢于 2sg 的 pre-scaling trick，**勿回退到该方向**。
- **开关**：`MNN_METAL_GEMV_SPLITK=0` 回退。

### 2.1.6 双行双流融合 GEMV（ROW_2，融合 dispatch 专属，auto 开）

融合 dispatch 上加 simdgroup 数证伪（-3%）后换思路：**加行内 ILP 而非加 simdgroup**——把 attention 内核 s0/s1 双流技巧搬到 GEMV。

- **机制**：`ROW_2` 编译变体（仅 gate_up/QKV/LN 融合管线；plain 路径继续走 SPLIT_K_2）——每 simdgroup 同时处理 2 个相邻 output slice，双 raw_dot 累加流共享同一次 input 读（LN 前导也共享），**无 barrier、无额外 simdgroup**；grid.x 减半（每 TG 4 slice）。第二行越界时别名到第一行（安全读，结果丢弃），QKV 各投影尾部由 per-row guard 处理。
- **实测**（M4 Pro 0.6B，冷机 2 轮交替 A/B）：p12 tg256 **+2.8%**（363.8→373.9）/ p512 **+2.4%**（337→345.3）/ p1024 **+2.3%**（301→308）/ p2048 **+1.2%**（256→259）——全 kv 段正收益（对比 SPLIT_K_2 融合版全负）。
- **门控**：`const bool row2 = !backend->isSupportTensorApi();`——三处 setup（`setupGateUpFusion` / `setupQKVFusion` / `setupLNFusion`）解析同一公式，保证 pipeline 宏与 grid 一致。M4/M5 双端标定完成后 `MNN_METAL_GEMV_ROW2` 三态开关已删除，设备门控即唯一逻辑。
- **正确性**：热稳定窗口 greedy **byte-identical**。开发中一度误判 DIFFERS——实为深度热节流下基线自身非确定。
- **为什么 ROW_2 赢而 SPLIT_K_2 输**：融合 kernel 的寄存器/占用余量吃不起翻倍的 simdgroup + barrier；双流方案在同一线程内加 ILP，占用不变，还顺带把 input/LN 读减半。

### 2.1.7 ⚠️ 短序列 GEMV 路径（area 2..16）从未优化

单 token 路径的候选已被逐一证伪，唯一剩下的数量级杠杆是"一次前向多算几个 token"（权重只读一次摊薄到 B 个 token）。本节量化该路径当前效率——**结论：空间很大，瓶颈明确是并行度而非算法**。

**e2e 实测**（M4 Pro，0.6B b64，`llm_bench -p B -n 1 -rep 6`；prefill 的 `area` 即 B，正是投机解码走的路径）：

| B | prefill tok/s | 每次前向 wall | vs B=1 | 理想 | 等效 decode 加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 349.1 | 2865 us | 1.00 | 1.00 | 1.00× |
| 2 | 490.4 | 4079 us | **1.42** | ~1.0 | 1.40× |
| 4 | 901.1 | 4439 us | **1.55** | ~1.0 | 2.58× |
| 8 | 1248.4 | 6408 us | 2.24 | ~1.0 | 3.58× |
| 16 | 1685.3 | 9494 us | 3.31 | ~1.0 | 4.83× |

decode GEMV 是权重带宽 bound，B 个 token 权重只读一次 ⇒ 理想 `cost(B) ≈ cost(1)`。实测 B=2 就要 1.42×，**没吃到应有摊薄**；且 **B=2 反常地比 B=4 更差**。

**归因**（counter profiler，us/call × 每前向调用数，只用相对量）：

| B | 走的 kernel | 每前向 GEMV | vs B=1 |
|---:|---|---:|---:|
| 1 | `qkv_fused/gate_up_fused/splitk2_gemv_g4m1_2sg` | 2229 us | 1.00 |
| 2 | `conv1x1_gemv_g4m2_wquant_sg` | 3939 us | **1.77** |
| 4 | `conv1x1_gemv_g4m4_wquant_sg` | 6419 us | **2.88** |

**根因（`MetalConvolution1x1.mm:616-644`）：短序列路径拿到的 simdgroup 并行度只有 decode 路径的一半，且完全没有融合。**

| | grid | threadgroup | SG/TG | 总 simdgroup |
|---|---|---|---|---|
| area==1（SPLIT_K_2）| `UP_DIV(oc,8)` | 128 线程 | 4 | **oc/2** |
| area 2..5（`g4mN`, piece=1）| `UP_DIV(oc,4)` | **32 线程** | **1** | **oc/4（减半）** |

- 这条路径**从未收到过单 token 路径的任何优化**：没有 SPLIT_K_2、没有 ROW_2，也拿不到 QKV/GateUp/LN 融合——`mIs2sgDecode` 与融合注册**只在 `area == 1` 分支设置**，所以 B≥2 时每层是 7 个独立投影 dispatch 而非 4 个融合 dispatch。
- 与 §2.0 画像自洽：小 GEMV 只到峰值 41-52%，是并行度/occupancy 受限；把 simdgroup 数砍半 ⇒ 时间近乎翻倍，正好解释 B=2 的 1.77×。
- ⇒ **可动项明确**：把 SPLIT_K_2 / 多 SG 并行度（以及融合）移植到 `conv1x1_gemv_g4mN`。目标把 `cost(2)` 从 1.77× 压到 ~1.1×。GEMV 占 decode ~70%，若达标，B=2 前向 wall 4079us → ~2900us，per-token 2040us → ~1450us。
- **前置依赖已就绪**：`transformers/llm/engine/src/speculative_decoding/` 已有 `lookahead` / `ngram` / `tokentree` / `eagle` / `mtp`——n-gram lookahead 不需要 draft 模型。实际收益 = 摊薄曲线 × 接受率，立项时需实测接受率。

### 2.1.8 W2/W3 decode 对齐 W4 优化栈

**背景**：2/3bit 支持（`20e5d03f3`）落地后 decode 实际是坏的——g8 body 的 W4 deferred 分支在 alias 下遮蔽 W2/W3 分支（陷阱 A 回归），所有 W2/3 decode pipeline 编译失败；且 2sg/SPLIT_K_2/融合/g16/SharedGather 全部只支持 4/8bit。另有**引擎级 bug**：`diskembedding.cpp` 对 2/3bit 误用 4bit nibble 解包（3bit 还越界读 → SIGSEGV），已修。

**实施**（每步 greedy 对拍过门）：
1. g8 body 阶梯重排 W2/W3→W4→W8（修复）+ g4mN 分支门控 4/8bit（非 sgMatrix 设备防越界）；
2. g8 W2/W3 deferred + pre-scaling：
   - **W2** 每 z 读 uchar4，mask `0xC0/0x30/0x0C/0x03` 配 in×(1/64, 1/16, 1/4, 1)，`adj = dbias - 2*scale`；
   - **W3** lo 面同 W2 + hi 面（tile bytes 4..5，nibble bit3=IC0..bit0=IC3）mask `0x8/0x4/0x2/0x1` 配 in×(1/2, 1, 2, 4)，`adj = dbias - 4*scale`；
   - pre-scale 全为 2 的幂 ⇒ fp 位精确。
3. 2sg kernel 6 处阶梯扩展（4 signature + ROW_2/plain body）+ dispatcher 解禁 + **3 个 fusion setup 的 keys/dic 补 W2/W3**（漏改是静默退融合，靠 profile subtag 检测）；
4. g16 lm_head 双行 W2/W3 分支（W3 row stride ×6）；
5. SharedGather（tied lm_head GatherV2 clone）W2/W3 分支——**onClone 门控与宏链必须同时落**，否则 2/3bit 静默编成 W8。

**实测**（M4 Pro 0.6B 交替配对）：decode tg128 **W3 +15%**（245→281）、**W2 +37%**（298→409）vs g8 路径；融合 kill-switch 验证 +3%（W3）；W4 全程字节一致。绝对值：W2 409 / W4 349 / W3 303 tok/s。prefill pp512 W3≈W4（5190，compute-bound）、W2 4248（-18%，`conv1x1_w_dequant` 的 W2 分支散字节读未优化，遗留项）。

**W3 < W4 decode 归因**：W3 tile 6B 非对齐，内环 6 标量 load + hi 面额外 8 mask（W4 是 1×ushort4 + 8 mask）；3×ushort load 变体实测**中性**（已回退）——与其他微调结论一致：0.6B 小 GEMV latency-bound，指令微调不兑现。结构性差距留待 4B+ 再评。

## 2.2 GEMM（prefill）

### 2.2.1 Fused Q4/Q8 GEMM（in-kernel 解包）+ M64 tile

`b71528f0d` 落地，`9ea642eed` 收敛开关。

- **机制**：tensor-API 设备（M5+）prefill 量化 conv 在 GEMM kernel 内解包反量化，省掉 dequant 预处理 dispatch + `mTempWeight` 分配（~4× 权重体积的带宽往返）；M64 tile 再省一半跨 TG 权重读冗余。
- **实测**：M64 tile M5 Qwen3-4B pp512 **+5.9%** / pp2048 **+6.8%**（greedy 前 20 token 逐字一致）。
- **W2/W3 标定**（2026-08-03 M5）：W2 +6~44% 全正；W3 4B +11~24%，但 **0.6B W3 fused pp2048 -3.6% 判负** ⇒ W3 额外要求 conv 权重 ≥4M 元素，小模型路由回 outer-dequant。
- **开关**：`MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI=1` 回退到 outer-dequant + fp GEMM（A/B 基线 / 紧急回滚）。非 M5 设备该开关是 no-op。

### 2.2.2 M64 sg_matrix GEMM 移植到 M4（收益低于预期，转自动策略）

`conv1x1_fused_q4_gemm_stage_m64` 的计算主体在 `#ifdef USE_METAL_TENSOR_OPS` 内，M4 sg_matrix 移植 = 从零写新 kernel（寄存器×2、threadgroup mem 翻倍、全新 index math）。

- **实测（M4）**：全场景 **+0.8~1.7%**（0.6B pp512/2048 +1.0/1.1%，4B outdeq +0.8~1.7%）；3 模型 greedy 一致；388/388 单测。**远低于 +10% 预期——M4 GEMM 瓶颈不在权重重复读**。
- **M3 Pro 标定**（配对 rep5 双向，0.6B）：pp512 **-1.4~-1.5%** / pp1024 中性 / pp2048 +0.8%，短 prompt 回归**否决全局默认开**。
- **处置（2026-07-31 定型）**：env 开关删除，改为 **arch-gen 设备分档自动策略**——`MetalBackend.mm` 解析 `MTLDevice.architecture.name`（`applegpu_g<gen><size>`）：gen≥16 且非 p 档（M4/M4 Pro/M4 Max Mac 及 M4 iPad）→ 走 `conv1x1_gemm_64x64_split_k_sg`；gen≤15（M1/M2/M3）与 phone p 档 → 32x64；macOS<14/iOS<17 无 architecture API → 保守关。
  - ⚠️ **family API 区分不了 M3/M4**（同 Apple9），必须用 `architecture.name`（MLX 同款做法）。
- **定量背景**：0.6B pp3312 的 outdeq_gemm ≈ 5.3 TFLOPS（fp16 峰 ~7.5），**prefill GEMM 已在 ~70% 算力峰值**，M64 上限本就 ~+10% e2e。

### 2.2.3 In-shader dequant 阈值改为面积相关（已默认生效）

4B 的 in-shader 4M 元素阈值在长 prompt 失效，outer-dequant 路径反而 pp1024 +3.4% / pp2048 +5.3%。改为 `area<512` 才走 in-shader（一行启发式，无新 env）：

- 4B pp2048 **+5.2%** / pp1024 +3.4% / 2B pp2048 +3.0%；pp256/512 无回归；峰值内存 +2MB；greedy 一致；388/388。
- ⚠️ **M3 验证为 merge blocking**（4M 启发式有 M3 回退前科）。
- 另：M4 上 in-shader wquant GEMM 与 outer-dequant 实测仅差 0.1%——两条路径都未到硬件上限。
- 开关：`MNN_METAL_PREFILL_INSHADER_DEQUANT_SGMATRIX`（仅非 tensor-API 设备生效）。

### 2.2.4 iOS 26.5 Metal4 tensor API 探测修复（`acc1afaab`）

- **问题**：MPP `matmul2d` 要求 M/N 至少一个 16 倍数、静态 K 16 倍数；探测 kernel 描述符错误导致探测失败 → tensor API 整体禁用；另有 legacy 16x16x8 路径（静态 K=8）探测通过但运行时反复编译失败反而回退。
- **实测**：修复后 iPhone 17 Pro Qwen3.5-2B prefill **953 → 1884 tok/s（+97%）**。

## 2.3 Attention Kernel

### 2.3.1 Causal 三角 QK dispatch + 有界 softmax/AV（CAUSAL_TRI / CAUSAL_BOUND，默认开）

`9948c74e1` → `78ae7bc55` → `f28510967`。**M4/M5 上 prefill 最大单项收益**。

**机制**：causal mask 下三角假设下，上三角区域在 mTempQK/mTempSoftMax 中**完全不写不读**，省 QK 写 + softmax 读 + softmax 写各 O(seq²/2) 带宽：

1. `CAUSAL_TRI`（prefill_qk）：host 只 dispatch 因果对角线下的梯形 tile（pp512 tile 数 -48%），kernel 内二次方程反解线性 tile id → (slq, slk)；interior tile 跳过全部 per-element mask 读取/分支（三区域分解）。
2. `CAUSAL_BOUND`（softmax_plane/_sg + prefill_qkv）：softmax 每行只归约/写出 causally-valid 前缀 + 24 元素零 pad（覆盖 prefill_qkv 的 8 对齐 tile 读界）；prefill_qkv 的 av_k_upper 截断同步激活。

**实测**（M4 Pro，交替配对 rep=3，`-fa 0`）：

| 指标 | Baseline | 优化后 | Δ |
|---|---|---|---|
| 0.6B pp512 | 4879.4 | **5088.1** | **+4.3%** |
| 0.6B pp2048 | 3689.0 | **4346.9** | **+17.8%** |
| 4B pp512 | 686.1 | 695.6 | +1.4% |
| 4B pp2048 | 610.3 | **649.2** | **+6.4%** |
| 4B tg128 | 75.1 | 75.6 | ~0%（decode 无回归 ✓）|

M5（tensor-API 路径扩展增量）：0.6B pp2048 **+38.9%**，累计 +51.5% vs master。CAUSAL_BOUND 单项：pp2048 +26%、pp512 +12%、4B pp2048 +8.2%。

**门控现状（已与文档早期版本不同，注意）**：
- `MNN_METAL_QK_CAUSAL_TRI` **已删除**——causal 判定改为**数据驱动**：`MetalAttention.mm:595-606` 的 `mCausalLayout` 由 `inputs[3]` 是否为标量哨兵（`|v|<1e-6`）或无 mask + 有 KV cache 推出。真实 mask 张量 ⇒ 非 causal ⇒ 自动关掉 causal-tri/bound/FA。**非 causal 模型不再需要设任何 env**（原开关是"忘了设就静默乱码"的正确性陷阱，且盖不住 FA-v1）。
- 早期文档写的"仅非 tensor-API 设备"**已过时**：causal-tri 现已扩展到 tensor 路径（32x32 tile，`MetalAttention.mm:1145-1146`）。
- `mQkCausalTri`（:803-805）= `mCausalLayout && !mShortSeq && (mQkSimdMatrix || mQkTensorMatrix) && !FA && !faNax && !mKvInDisk && mKvSeqLen >= mSeqLen`；`mCausalBound`（:809-810）同条件但不要求 matrix 路径。
- decode 侧不存在等价优化——seq_q=1 时 1×kv 分数行 100% 因果有效，无三角可跳。

### 2.3.2 M4 级设备 FlashAttention 降级到三段路径（`472c76bd8`）

优化后的三段路径（+causal-tri/bound）在 M4 Pro 上反超 FA，且差距随 seq 增长——causal-bound 省的是 O(seq²) 带宽而 FA kernel 未享受：

| 指标 | FA on | 三段+causal-tri v2 | 三段反超 |
|---|---|---|---|
| 0.6B pp512 | 4947.6 | 5088.1 | +2.8% |
| 0.6B pp2048 | 4077.6 | **4346.9** | **+6.6%** |
| 0.6B pp3312 | 3362（FA kernel 414ms）| 3623（QK+AV+softmax 合计 362ms）| +7.8% |

**处置**：M4 档默认走三段+causal-tri（pp2048 4088→4319，+5.7%，kv≤8192 生效）；FA 保留给长上下文（kv>8192，省 O(seq²) scratch 内存）/ head_dim ∉ {64,128,256} 兜底；`MNN_ENABLE_FLASH_ATTN_PREFILL=1` 可强制。M5 同样默认 demote。M3 待验证。

### 2.3.3 Fused Prefill Flash-Attention（保留场景：长上下文 / 特殊 head_dim）

**问题**：三段 pipeline `prefill_qk` → `softmax_plane_sg` → `prefill_qkv` 通过 global memory 传递中间结果。Qwen3-0.6B pp2048 单层 mTempQK / mTempSoftMax 各 128 MiB，write+read ~512 MB/前向。

**方案**：融合 Q·K^T + online softmax + P·V 到一个 kernel，中间数据全留在 threadgroup memory 和寄存器：
- **Q_TILE=16, KV_TILE=32, NSG=4**（128 线程）；Grid `(ceil(seq_q/16), num_head*batch, 1)`；
- 每 simdgroup 拥有 2 行 Q 的 `M`（running max）/ `S`（running sum）寄存器；
- 每 KV 块：QK → 在线 softmax → 同一段 P 做 PV → 累加到 `so`（O accumulator）。

**Threadgroup memory 布局**（D=128 时 ~15 KB）：

| 名 | 类型 | 用途 |
|---|---|---|
| `sq[Q_TILE * HEAD_DIM]` | half | Q 分块（cooperative load 一次）|
| `sf[Q_TILE * KV_TILE]` | float | QK 的 fp32 scratch |
| `ss[Q_TILE * KV_TILE]` | half | 归一化后的 P（half 存以便 half×half → float MMA）|
| `so[Q_TILE * HEAD_DIM]` | float | O accumulator，在线 rescale |

**关键文件**：`MetalFlashAttnShader.hpp`（`gPrefillFlashAttn`）、`MetalAttention.mm/hpp`。

**实施要点**：
1. 门控 / pipeline keys / dispatch grid 见 `MetalAttention.mm:690-750`：config `attentionOption/8 >= 1` 或 `MNN_ENABLE_FLASH_ATTN_PREFILL=1`；要求 simd-matrix + fp16 + `mCausalLayout` + head_dim ∈ {64,128,256} + group_size ∈ {1,2,4,8} + `mSeqLen>=128`。
2. **在线 softmax 数值稳定性**：`M_new = simd_max(fmax(M[j], s))`；`ms`/`vs` 对 `-INFINITY` 的双短路是必需的（否则 `exp(-inf - -inf)` = NaN 从初始态或全 masked 行传播）。
3. **KV int8**：D=256 下不能整 tile 反量化到 threadgroup（爆 32KB），每 k_step 分批 8×8 dequant；`k_scales`/`v_scales` 是 `device ftype*`（fp16）**不是** float，错声明必乱码；用 `threadgroup_barrier` 不是 `simdgroup_barrier`（8 lane 写、32 lane 读）。

**避坑要点**：
1. **`ATTENTION_C4` 输出布局 —— 最重要的坑**。c4-head export 时 output 实际布局是 `[num_head*(head_dim/4), batch*seq_q, 4]`（NC4HW4-packed）。不区分则 token 输出**从第一步就乱码**，且代码逻辑看着完全正确、地址全部合法。正确 epilogue：

   ```cpp
   #ifdef ATTENTION_C4
       int o_off = (h * (param.head_dim / 4) + (d / 4)) * 4 * param.batch * seq_q
                 + (b * seq_q + q_abs) * 4 + (d & 3);
   #else
       int o_off = ((b * seq_q + q_abs) * param.head_num + h) * param.head_dim + d;
   #endif
   ```
2. **不要用 threadgroup memory 中转 K/V**：初版怀疑 `simdgroup_load` 5 参 transpose flag 有 bug 而预排布局，正确但 pp2048 掉 45%。真 bug 是 ATTENTION_C4。经验：**先怀疑数据布局，再怀疑 Metal API**。
3. **mixed-dtype MMA 只有 all-half 或 all-float**：QK 输出先写 fp32 `sf`，softmax 读 fp32、算完转 half 写 `ss` 供 PV MMA——两块 scratch 不能合并。
4. **softmax→PV 之间的 `threadgroup_barrier` 不可少**：各 SG 只 rescale 自己 2 行 `so`，但 PV 读全部 8 行。
5. **正确性验证必须 greedy sampling**（temperature=0, top_k=1）对拍前 40-60 token byte-identical。

**性能数据**（M4 Pro，W4-block32 Q4，fp16，vs 未优化三段路径）：

| Model | pp512 | pp2048 | tg128 |
|---|---:|---:|---:|
| Qwen3-0.6B | +2.5% | **+11.7%** | noise |
| Qwen3-4B | +2.1% | **+4.7%** | noise |
| Qwen3-8B | +3.1% | **+9.1%** | — |

**参数调优实验记录**：

| 变体 | 0.6B pp2048 | 0.6B pp512 | 结论 |
|---|---|---|---|
| Q_TILE=8, KV_TILE=32（初版）| +5.2% | +0.8% | 起点 |
| **Q_TILE=16, KV_TILE=32** | **+11.7%** | **+2.5%** | ✅ 采用 |
| Q_TILE=16, KV_TILE=64 | +3.6% | **−6.4%** ⚠ | ❌ 回退 |
| Q_TILE=32 | — | — | 跳过：threadgroup mem ~30KB 逼近 32KB 上限 |

教训：**减少 K read 冗余（Q_TILE↑）是长 prompt 最有效杠杆**（Grid.x 减半换 K 读半减）；KV_TILE↑ 反而差；whole-tile causal early-exit 已达下三角 50% 理论上界，无需显式 block classifier。

**已探明的收益/风险边界**（不建议盲改）：F=2 多头融合（threadgroup mem 15→30KB，occupancy 减半，净收益不确定）；去循环末 barrier（4B pp512 **-14%**，barrier 有带宽调度作用，全保留）；`so` 显式清零必须保留（threadgroup 初值可能 NaN）；NSG 4→8 无收益。

### 2.3.4 tensor-API 版 FlashAttention（`prefill_flash_attn_nax`，M5+ 默认开）

`MetalFlashAttnShader.hpp:412`。用 Metal4 cooperative tensor 重写（§1.10 的布局表就是为它测的）：`matmul2d(16,32,16, tb=true)` + input cooperative tensor + **零 threadgroup memory**，S/O 全寄存器、score 不落全局。

- **门控**（`MetalAttention.mm:759-783`）：`MNN_METAL_PREFILL_FA_TENSORAPI`（unset 时随 `mCausalLayout`）&& `isSupportTensorCoopInput()` && fp16 && causal && 非量化 KV && `mKvSeqLen>=mSeqLen` && head_dim ∈ {64,128} && `mSeqLen>=64`。命中即关 legacy FA。
- **实测（M5 冷机配对 rep5）**：0.6B pp512 **+4.2%** / pp1024 **+5.5%** / pp2048 **+9.0%** / pp4096 **+17%**（随 seq 单调增大）；4B +1.2~3.7%；2B（head_dim=256，被门挡掉）中性。三模型 greedy byte-identical，run_test 388/388。
- **M4 Pro 验证**：M4 在 `noAICoreDevice` 名单，`isSupportTensorCoopInput()=false` → nax 结构性休眠，默认开是**零影响 no-op**。故默认开只对 M5+ 生效，对 M4/M3/iPhone 无风险。
- ⚠️ **测新 kernel 性能必须用 `llm_bench -rep≥5`**：`llm_demo` 单次会把 JIT 编译计入 prefill，同一份 kernel 测出 2172 vs 7488（差 3.4×）。
- **判负存档：simdgroup 版 FA2（`prefill_flash_attn_v2`，8x8 simdgroup_matrix）已删除**。数值全对但 M5 上 p512/p2048/p4096 = 0.90/0.76/0.68×，二次系数 0.067 = 三段的 1.9 倍；根因是 M5 上 8x8 simdgroup MMA 对该 shape 只有 `matmul2d` 一半的 FLOP 效率（四种载入策略都是 760-920us/层 vs 三段 357）。**无重启价值**。

### 2.3.5 单 pass 融合 SDPA decode（`MNN_METAL_DECODE_SDPA`，auto 开）

> ⚠️ **命名遗留**：shader 变量 `gDecodeSplitKV`（`MetalAttentionShader.hpp:2650`）与 kernel 名 `decode_splitkv`（:2683）是历史名字。**真正的 split-KV 路径已于 2026-07-30 删除**（连同 `decode_splitkv_reduce` kernel 与 `MNN_METAL_DECODE_SPLITKV/_NWG` 开关）。现在这个 kernel 是**单 pass 融合 SDPA**：`MetalAttention.mm:1087-1110` nwg=1 硬编码、grid `(1, B*H, 1)`、无 reduce、score 不落全局、kernel 直写输出。

- **机制**：一个 TG 负责一个 q head（对齐 MLX `sdpa_vector`），QK + online softmax + AV 单 kernel 完成。
- **门控**（`MetalAttention.mm:627-653`）：`decodeSdpa > 0`（默认 1=auto）&& `totalKv >= sdpaThresh` && `mKVCache` && `mSeqLen==1` && `!mKvInDisk` && (`mCausalLayout` || trivial mask) && `mHeadDim%32==0` && tg 内存 ≤30KB。
  - 阈值（:614-622）：tensor-API 设备 3072 / 其余 1536，再对 fused kernel 的 kv 容量 clamp（group2:2048 / group4:1024 / group8:512）。
  - NSG（:638-644）**device-tiered**：tensor-API/M5 → 8；非 tensor-API/M4 类 → 32。
- **实测（M5，0.6B）**：p2048 e2e **+5.2~6.4%**（3 对干净配对，base bracket ±0.3%）；p1024 以下阈值外结构性零影响；短中 kv 强开为负（p1024 -0.5~-4.8%）故必须阈值门控。4B p2048 **+2.5~3.4%**；Qwen3.5-2B（仅 6/24 层 full attention）中性无回退。
- **实测（M4 Pro，配对 rep5）**：默认(nsg32) 255.8 vs `=0`(legacy) 240.8 → **+6.3%**；p4096 **+7.5%**；p6144 158 vs 124 = +28%。8B p2048 仅 +4%（decode 被权重 GEMV 稀释）。
- **NSG 标定 M5 与 M4 结论相反**：M5 p2048 nsg8 +6.2% > nsg16 +4.8% > nsg32 +3.3%（e2e 随 TG 变宽单调恶化）；M4 Pro nsg8 **-3.5%** / nsg16 +3.8% / **nsg32 +6.0%**，nsg4 崩到 192（-20%）。⇒ 分档取值。M1/M2/M3/iPhone 未标定，继承 M4 分支。
  - ⚠️ profile build 与 e2e 常给出相反结论，标定只认 e2e 配对。
- **正确性**：0.6B/4B/2B × kvq8 × replay × 跨阈值 200tok 全 byte-identical，run_test 388/388。
- **已删除的变体（勿重试）**：
  - `MNN_METAL_DECODE_SDPA_QSPLIT=0`（每 kv head 一个 TG、GROUP_SIZE 头共享 K 读）判更差（p1024 -4.8% vs -3.6%），分支连同宏删除，`GS_LOCAL` 定死 1。
  - `MNN_METAL_DECODE_SDPA_COALESCED`（QK 合并读，simdgroup↔kv 行、256B 连续 K 读）：e2e p2048 co8 +6.0% ≈ leg8 +6.2%，**合并读不兑现**（同一模式第三次复现）。**勿再重试 K 读合并方向**——典型的"kernel 级快 / e2e 平"伪收益。

**历史 split-KV 数据存档**（路径已删，仅供判断同类方案）：flash-decoding 式 KV 分段到多 workgroup 各算 online-softmax 部分结果再跨 workgroup reduce。M4 kv4K decode 0.6B +19% / 4B +5.5%；KV int8 版 0.6B kv2K +16.6% / 4B kv4K +15.6%（fp16 同点仅 +5.5%，印证 int8 带宽减半假设）。踩坑三条：① 路径 flag 判定必须放在 `handleKVAllocMemory()` **之前**，否则首个 decode step 临时缓冲未分配 → `setTensor(null)` SIGSEGV；② reduce kernel 必须 128 线程（32 线程版占用率不足吃掉收益）；③ nwg 启发式 div256 是甜蜜点（div128/512 均 -2~5%）。**短 kv 反证**：强制在 kv~512 触发 → p512 decode 334→310（-7%），多出的 reduce dispatch + partial buffer 全局��写在短 kv 下开销 > 并行度收益。

### 2.3.6 Q-head-split Fused Decode QK+Softmax（QK_QSPLIT，kv∈[512,1536)，auto 开）

背景：0.6B group_size=2 时 fused `decode_qk_softmax` grid 仅 8 TG（每 kv-head-group 一个），GPU 核心吃不满；SDPA 只覆盖 kv≥1536。

- **机制**：`QK_QSPLIT` 编译变体（仅 GROUP_SIZE==2）——`grid.z = group_size`，每个 q-head 独占一个 TG（8→16 TG），threadgroup 内存减半（单 scores 流）；配合**半宽 threadgroup**（localSize = kv/2 向上取 32 对齐，总线程数与不拆分持平）。代价：K 每 kv-group 读 2 次、失去 s0/s1 双流 ILP。
- **门控**（`MetalAttention.mm:836-837`）：`mDecodeQkSoftmax && group_size==2 && !isSupportTensorApi() && mKvSeqLen>=512`。env 覆盖已删（:835）。决策在 `_computePathFlags`（每 token 重估），变体翻转纳入 `_pathSignature` bit25 → replay 正确失效重录。
- **实测**（M4 Pro 0.6B tg128）：p1024 **+2.7%**（294→300）/ p768 +1.6~2.4% / p512 中性 / p12 ~-1%；p2048 走 SDPA 不受影响。greedy 全 byte-identical。
- **踩坑**：threadgroup 宽度是命门——沿用 kv/6 窄公式（TG 数翻倍触发启发式换挡）时 **-5%**，必须用 kv/2 半宽公式。
- **M5 标定为负**（强开 p1024 -2~3%），auto 排除 tensor-API 设备正确。iPhone 未标定；group_size=4/8 泛化未做（generic kernel 数组索引有 15% 编译器劣化前科）。

### 2.3.7 Fused Decode Attention GQA 扩展（group_size 2-8）

原 `decode_qk_softmax` fused kernel 只支持 group_size=1。扩展为模板化 group_size（`MetalAttentionShader.hpp` 编译宏 + `MetalAttention.mm` 按 `num_heads/num_kv_heads` 选 kernel），避免 Q/K 显式 repeat_kv 拷贝。对 GQA 模型（Qwen3 g=2、Llama3 g=4）**decode attention 提速 10-20%**。

### 2.3.8 decode/prefill attention 路由速查

全部路由在 `MetalAttention.mm:573 _computePathFlags()`，**每 token 重算**；`_pathSignature()`（:840-871）决定 replay 是否失效。

**decode**（优先级从高到低）：
1. 单 pass 融合 SDPA（`mSdpaSinglePass`，§2.3.5）——抢占下面的 fused 路径；
2. 融合 `decode_qk_softmax`（`mDecodeQkSoftmax`，:821-823）：`mKVCache && mShortSeq && mSeqLen<=8 && (mCausalLayout || trivialFloatMask) && !mKvInDisk && group_size>=2 && mHeadDim%8==0 && mKvSeqLen<=maxKvForFusion`；可叠 `QK_QSPLIT`（§2.3.6）；
3. 三段 `decode_qk` → softmax → `decode_qkv`（:1134-1141，else 分支）。

**prefill**（优先级 faNax > legacy FA > 三段）：
1. `prefill_flash_attn_nax`（§2.3.4，M5+）；
2. legacy `gPrefillFlashAttn`（§2.3.3）；
3. 三段 + CAUSAL_TRI/BOUND（§2.3.1，M4/M5 默认）。

---

## 2.4 其他 Kernel

### 2.4.1 RMSNorm 小 Batch 单 TG 路径（`MetalLayerNorm.mm`）

Decode 时 batch=1，默认 kernel 选择倾向大 batch tile，launch overhead 反而盖过计算。
`batch <= 4 && hidden_size <= 4096` 时改用**单 threadgroup 处理整个 norm**。
Decode RMSNorm 本身提速 ~5%，链路 ~1%。

这是"小 shape 走窄路径"的典型：**decode 的 batch=1 与 prefill 的 batch=seq 是两个世界，
同一个 op 常需要两套 dispatch 形态**。新写 kernel 时先问 decode 形态是否退化。

### 2.4.2 LinearAttention（gated delta rule，Qwen3.5 系）kernel 演进

**背景画像**（0.8B p2360 prefill）：LinearAttention 占 GPU **32.6%**（与全部 GEMM 相当），
decode 仅 5.8% ⇒ **prefill kernel 是最大单点**。非 tensor-API 设备长 prefill 原走
`fused_chunk_sg`（chunk 内逐 timestep 标量 delta rule，零 matmul 化）。

**变体全景**（`MetalLinearAttention.mm:721-799` 路由）：

| kernel | 命中条件 | 状态 |
|---|---|---|
| `linear_attn_gated_delta_rule_sg_v4` | 非 tensor-API && `mHeadKDim==128` && `seqLen>=16` prefill，配 `qkv_prep_sg` | **主力**（§2.4.3）|
| `linear_attn_fused_chunk_sg` | `dk != 128` 的长 prefill | 在用 |
| `linear_attn_flash_chunk_sgmm` | 编译门 :353-355、路由门 :771 | **事实休眠**：sgmm 编译条件蕴含 dk==128 ⇒ v4 必然存在 ⇒ 分支不可达。未删，等 M5 结论 |
| `linear_attn_fused_sg_align` | `2 <= seqLen < 16` 短 prefill | 在用 |
| `linear_attn_fused_sg_tg` | `H < 16 && seqLen == 1` | 在用（阈值见下）|
| `linear_attn_fused_sg` | 上面都不命中的兜底 | 在用 |
| chunk64 双 kernel（flash_chunk）| tensor-API 设备（`supportTensorOps()` :298-300）| 在用，M5+ |

**sgmm 批次（2026-08-03，M4 Pro，历史）**：把 tensor-API 版 `flash_chunk` 的分块算法
（K@Kᵀ / Q@Kᵀ / T 前代求逆 / K@S / Q@S / state MAC）用 `simdgroup_float8x8` 8x8 MMA 重写。
SIMDS_PER_TG 扫描 4/8/16/32 → 356/285/**269**/454ms
（TG 内存 ~29KB 限 1 TG/核，**宽 TG 是唯一 occupancy 杠杆，但 32 过宽反噬**）。
e2e：0.8B pp512 **+14.7%** / pp2048 **+15.8%**，2B pp2048 **+8.2%**。

- ⚠️ **dk=64 证伪**：标量基线在 dk<128 用 CHUNK_BT=32，全 L 段反而更快 ⇒ sgmm 门控收紧到 dk==128。
- ⚠️ **测量教训重演**：pp512 首轮固定顺序 A/B 测出 "+5.9%"，反序即翻转；真收益要等 sg16 才显现。

**短 prefill (2≤L<16) 路由 `fused_sg_align`**：省 `qkv_prep` dispatch + mQ/mK/mV 物化往返。
微基准 L8：d64 **-21%** / d128 -4%；token 一致。

**`fused_sg_tg` 阈值重标定证伪**：`preferTG = H < 16` 恰好排除 Qwen3.5（H=16）；
放宽到 `H<=16` 实测 decode e2e 与微基准均中性偏负 ⇒ 阈值维持 `H<16`。
TG 共享 Q/K 读在 H=16 档无收益（冗余读被 cache 吸收）。

**decode encode-replay 接入**（结构性，收益中性）：`canRecordEncode()` 从恒 false 改为
`seqLen==1 && gated_delta_rule`。关键障碍是 `Pipeline.cpp` 对 LinearAttention
**每 token 强制 re-resize**，`onResize` 每次重建 `mConvOut` → 录制绑定悬垂 `Tensor*`
（症状：每 token invalidate→ban）。修法：shape 不变时保留 Tensor 对象 +
resize-generation 守卫在 `onReplayUpdate` 里 bail（**必须先于** `metalReplayEmit` 的解引用）。
详见 `runtime-scheduling.md` 的 Encode Replay 一节。

### 2.4.3 寄存器 state vec4 scan 替代 chunk（commit `3479c962f`，2026-08-07）

当前 LinearAttention prefill 主力路径的由来，也是**"分块 MMA 不一定赢顺序 scan"**的样本。

**定位**：MLX 对照探针（口径对齐 LinearAttention op 范围）测出 prefill 差距全部在 LA：
MNN sgmm chunk **6.1us/tok/层** vs MLX 顺序 scan **2.4**（2B，18 层 LA ≈ 全部 e2e 差距 ~67us/token）；
decode LA 反而 MNN 更快。MLX 的赢法是"不分块直接扫"：每 (b,h,dv) 一个 simdgroup、
**state 驻寄存器**、零 barrier / 零 threadgroup memory、~2048 SG 拉满 occupancy。

**实现三处**：
1. `linear_attn_gated_delta_rule_sg` 的 state **寄存器化**（原来每 timestep 读写 device 两遍）；
2. 新 kernel `linear_attn_gated_delta_rule_sg_v4`（**dk==128 特化**：每 lane 持 4 个连续元素，
   `half4` load + `dot`，对齐 MLX 数据布局）；
3. 路由：非 tensor-API && dk==128 && L≥16 的 prefill 走 `qkv_prep + v4 scan`，
   替代 sgmm / `fused_chunk_sg`；tensor-API（M5+）保持 chunk64 flash 不动。无 env 开关。

**实测**（双向 A/B rep3）：2B pp512 **+11.6%**（1732 vs 1552）、pp2048 **+10.4%**（1791 vs 1622）、
0.8B pp2048 **+23.6%**（3974 vs 3215）；decode 全部中性（L=1 路径未动）。

**两次证伪（勿重试）**：
- ❌ **`fused_sg_align` 直接路由长 prefill：-26%**——它从 `conv_out [B,D,L]` 读，
  lane 间 stride=L 完全非合并。**token-major 输入是 scan 赢的前提**；
  `qkv_prep` 的物化成本（~0.12us/tok/层）远小于收益。
- ❌ **跨步 lane 映射（`lane + ii*32`）的寄存器版只追到 -8.5%**——4 次分散 2B load
  vs vec4 一次 8B load。**向量化才是从 -8.5% 到 +11% 的关键**（与 MLX `n_per_t` 连续切分同构）。

> 两条证伪合起来是同一条原则：**寄存器化只是必要条件，内存访问形态（token-major + vec4）
> 才是决定项**。改 scan 类 kernel 时先画 lane→地址映射，再谈算法。

**遗留**：M5 上 scan vs chunk64 flash 未对比（`!mUseFlashChunk` 门控保守保留旧路径）；
sgmm kernel 在 dk==128 非 tensor-API 设备上已休眠，等 M5 结论后可考虑删除。

### 2.4.4 `linear_attn_gated_norm`（`MetalGatedNormShader.hpp`）

Qwen3.5 linear-attention **输出门控段**的单 kernel 实现，替代原本 7 个 dispatch 的链：

```
LinearAttention [H,dv,1,1] → Raster1(identity) → Cast(half→half 位拷贝)
  → RMSNorm(per-head, layernorm_c4_rms_sg) × SILU(Raster2(z)) → MUL(half)
  → Raster3 → out_proj
```

Raster1 是纯位拷贝；**Raster2/3 是真实 C4 重排且互为逆**——融合掉它们省的是真实搬运。

**kernel 设计**（这是本节的重点，融合链路见 `graph-fusion.md`）：
- **一个 simdgroup 一个 head**。dv=128 ⇒ 每 lane 恰好 1 个 `float4` ⇒
  **无 threadgroup 内存、单次 `simd_sum`** 就完成 per-head 的平方和归约。
- 索引 `la[c*outside+h]` 读、`z, out[h*(inside/4)+c]` 写——**后者天然吃掉 Raster2/3 两次重排**，
  不需要额外搬运指令。
- `z->length(0)==1` 同时充当 decode 门控（prefill 的 z batch=seq 天然不匹配，零影响）。
  ⚠️ 上移导出图后 shape 契约已泛化为 x `[batch*heads, inside]` / z,out `[batch, heads*inside]`，
  decode 是 batch==1 特例；**首版硬编码 batch==1，prefill resize 直接 Compute Shape Error**。

**判负：不折进 out_proj 的 GEMV 前导**（LN_FUSED 模式）——每个 TG 要多读一份 z，
约 **+9MB/token 冗余流量**，decode 带宽敏感，得不偿失；独立 kernel 只读一次（12KB/层）。

**实测**（双向配对 rep3，各 4 对全同向）：0.8B tg128@p512 **+6.4%**（286.7 vs 269.4）、
2B **+2.9%**（151.8 vs 147.6）。18/18 层命中。

**正确性口径 = fp32 bit-identical，而非 fp16 byte-identical**：
- `precision: high`（fp32）下 fold 与不 fold **bit-identical** ⇒ 索引、布局、数学全部等价；
- fp16 下 0.8B 全 prompt、2B 长 prompt、Qwen3-0.6B 与基线 token 完全一致，
  仅 2B 短 prompt 分叉一次（greedy 边界 token）；输出确定（5 连跑同 hash），
  对 `COMMIT_NUM=1` / `DISABLE_REPLAY=1` 均不敏感 ⇒ 不是内存竞争；
- 差异二分到**只有 RMSNorm 半边**，源码与 `layernorm_c4_rms_sg` 逐字相同 ⇒
  判定为编译器 codegen 层面的等价重排。本仓其它已合入融合（LN 融合、QKV 融合）同类。

**kernel 级二分方法（可复用，比 token hash 强得多）**：
1. **fp32 当 oracle，最先做**。fp32 bit-identical 一次排除索引/布局/逻辑错，
   把问题锁死在 fp16 rounding。
2. 分阶段临时 env 探针：关匹配 / 只做 STATIC 提升 / 只装 leader 不 claim /
   leader 退化为纯搬运 / 用链路中间结果替换我的 LN 或我的 SILU。
   本次结论：纯搬运、「链路 norm × 我的 SILU」、「链路 norm × 链路 gate」**全部精确复现基线**
   ⇒ 读写索引、SILU、half 乘法全对；只有「我的 LN × 链路 gate」不同 ⇒ 差异只在 RMSNorm。
3. ⚠️ **诊断本身必须可靠**：替换读源的探针若不给那个中间张量做 STATIC 提升，
   它的生命周期在原消费者处就结束、内存可能已被回收，**探针会读到脏数据给出假结论**。
   本次前一轮就因此得到三个互相矛盾的 hash。
4. 能算清的假设先算清再测：`channelUnit == SIMD_GROUP_WIDTH` 时每 lane 只迭代一次，
   `0 + d*d` 与 `fma(d,d,0)` 恒等，该实验无信息量。
5. ⚠️ **冷/热**：刚删 `mnn_cachefile.bin` 的第一次跑与后续不同（pipeline cache），
   所有对拍前先预热一次。
