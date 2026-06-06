---
name: metal-optimize
description: MNN Metal 后端 op/kernel 优化与扩展。覆盖 Metal shader 字符串嵌入流程、conv1x1 多 pipeline 选路、SIMD group reduce/matrix kernel、weightTransform CPU pack、Apple GPU 实测验证。
---

# MNN Metal 优化 Skill

> **触发**：扩展或优化 Metal 端 kernel（conv/gemm/gemv/attention 等），新增算子，调度选路或 weight pack 调整。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## 核心原则

1. **shader 是嵌入的 C++ 字符串**。Metal kernel 写在 `*.hpp` 里的 `R"metal(...)metal";` 字符串中（如 `ConvSimdGroupShader.hpp`），不是独立 `.metal` 文件。改完直接 `make` 就拿到，不需要 codegen。**字符串拼接顺序决定 `#define` 作用域**——下面的"宏 alias 陷阱"特别依赖这点。

2. **dispatcher 要先摸清**。Metal conv1x1 同一 op 常有多条 kernel（gemv 多种 / gemm 多种 / outer dequant），按 `area`、`oc`、`ic_4` 等 case 切。新加 quant bit 不可能一次扩完所有路径，**必须先决定支持哪几条 + 让其他路径不被 dispatch**。

3. **Apple GPU 是开发 + 主测平台**。MoltenVK / Metal 性能与稳定性数字 Mac M3/M4 真实可用。但**这条路不代表 Android**，Vulkan/OpenCL 上同算法行为可能完全不同。

4. **正确性 oracle 先于性能**。同 OpenCL skill。

---

## 入口定位

```bash
grep -rn "OpType_<MyOp>" source/backend/metal/                    # Execution
grep -rn "kernel void <my_kernel>" source/backend/metal/*.hpp     # shader 字符串入口
```

低 bit 量化 conv 入口：`MetalConvolution1x1::onResize`（`MetalConvolution1x1.mm`）。识别 quant：`mDequantBits ∈ {2,3,4,8}`（在 `MetalConvolutionCommon::loadWeight` 里设置）。

`onResize` 把 `area` × `oc` × `dequantInShader` × simdgroup 能力组合分流到 gemv / gemm / outer-dequant 等多条路径，扩 quant bit 前先把目标 shape 走的那条 case 标出来。

**示例：当前 conv1x1 低 bit 量化路径**
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

新加 quant bit 时需要决定**至少扩哪几条**。典型组合是 decode gemv path + prefill outer-dequant path，其他 gemv/gemm 实例 dispatcher 显式 fallback；一次性扩完所有 path 工作量太大。

---

## 通用陷阱

### 陷阱 A：宏 alias 让 `#ifdef` 多分支同时为真

最严重的 Metal 坑。给"未扩展的 kernel"在新 quant bit 下编译过，常加 `#define W_QUANT_4` alias：

```c
#if defined(W_QUANT_2) && !defined(W_QUANT_4) && !defined(W_QUANT_8)
#define W_QUANT_4    // 让其它 kernel 还能编译
#endif
```

**坑**：alias 让 `#ifdef W_QUANT_4` 在你想扩展的那个 kernel 里**也**被命中。kernel signature 里 `#ifdef W_QUANT_2` 第一个匹配（`uchar4* wt`），但 body 里 `#ifdef W_QUANT_4` 也匹配（`MNN::uchar4x2 w_int4 = xy_wt[z]`），类型混淆 → 编译过 → 数值错。

修法：扩展的 kernel 里**所有相关 `#ifdef` 必须 W_QUANT_2 → W_QUANT_3 → W_QUANT_4 → W_QUANT_8 顺序**，新 bit 优先匹配。signature 和 body 都要这个顺序，少一处都 sneaky 错。

### 陷阱 B：dispatcher 漏路径（lm_head）

LLM 的 lm_head conv (`oc = vocab_size ~150k`) 走 `oc > 16384` 的特殊路径（如 `g16`）。新加 quant bit 没扩 g16 时，dispatch 还会进 g16 → 用错 layout 读 buffer → 数值错或 crash。

通用应对：dispatcher 选路写 `mDequantBits != 2 && mDequantBits != 3 && oc > 16384` 这种白名单，把没扩的路径强制 fallback 到已扩的（如 `g8`）。

### 陷阱 C：weightTransform 的多签名同步

`weightTransform(...)` 在 `MetalConvolutionCommon`、`MetalConvolutionWinograd`、`MetalConvolutionDepthwise` 都有 override。改签名（如加 `subBits` 参数）时这 3 处 + `.hpp` 4 处都要同步，否则 build 报 `'override' but does not override any member function`。

### 陷阱 D：getDequantScale 的 `coef` fp16 范围补偿

Metal `getDequantScale` 用 `coef = 1000/max_data` 做 fp16 范围补偿（host 写 `s*coef`，shader `/coef`）。新加 quant bit 不要碰这个流程；scale/offset 的 originOffset 折叠**完全在 host alpha 写入时**完成，shader 一律按 signed 解出后 `signed_w * scale + bias` 即可。

### 陷阱 E：tile 内 byte index 选择（OC vs K_inner）

W_QUANT_8 的 tile layout 是 `byte = ro * 4 + ri`（OC 外、K_inner 内），即 `xy_wt[z]` 取 16 字节 = 一个 (4 OC, 4 IC) tile，`w[i] = char4` 是 1 OC × 4 IC。

新 bit 的 packing 必须**镜像已有最高 bit kernel 的字节顺序**：byte i = OC i 的多个 IC，不能反过来变成"byte i = IC i 的多个 OC"。

**示例**：把 byte ↔ (oc, ic) 映射写反（OC 内、K 外），shader 编译过、kernel 能跑，但输出乱码——只有 dump 第一个 op 的 weight 前几行和 CPU 对照才能发现。GPU 输出"乱"很容易先怀疑 dequant 数值或 sampler，反到字节顺序的可能性需要主动检查。

---

## Packed weight 设计

新加 quant bit 时**先固定 5 个量**：

| 量 | 解释 |
|---|---|
| tile = (IC_inner × OC_inner) | Metal conv1x1 一次原子访问的最小区块 |
| 字节/tile | 由 bit 决定，新 bit 推 layout 时镜像已有 bit 的 stride |
| byte index 内的语义 | 与已有最高 bit kernel 的 byte ↔ (oc, ic) 映射保持一致 |
| bit 顺序 | 与 host packing / 跨后端约定一致 |
| signed/unsigned 存储 | 存 unsigned，shader 内减 offset 还原 signed |

**bit 不齐 32 位时的 split layout**（如 3bit）：常见做法是低 2 bit 一段 + 高 1 bit 另一段，避免跨 word 边界。host packing 与 shader unpack 双向严格镜像。

**示例（w3 = 6B/tile）**：bytes 0..3 装低 2 bit（与 w2 layout 一致），bytes 4..5 装高 1 bit（byte 4 = OC{0,1} 的 high bit、upper nibble = OC even / lower = OC odd，byte 5 = OC{2,3}）。每个 nibble 内 bit `3-k` 对应 IC k 的 high bit。比"32 weights = 12B 跨 word 边界"方案更友好，shader 用一次 `vload8 + vload4` 就能取到一个 (4 IC × 8 OC) tile。

**signed/unsigned 与 originOffset**：模型导出器把 `b = min_val + offset_signed * scale` 的 originOffset 已折进 bias。shader 解出 signed 权重做 `signed_w * scale + b` 即可，不要再折一次。

---

## Shader 修改流程

```bash
# 直接编辑 .hpp 里的字符串
vi source/backend/metal/ConvSimdGroupShader.hpp

# 编译（无需 codegen）
cd build && cmake .. -DMNN_METAL=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j8 MNN llm_demo
```

**新加 `W_QUANT_N` 时同步检查清单**：

| 位置 | 检查 |
|---|---|
| kernel signature | `#ifdef W_QUANT_N` 分支声明 `wt` 类型 |
| kernel body | `#ifdef W_QUANT_N` unpack 分支，**优先级在 W_QUANT_4 之前** |
| 宏 alias 块 | `#if (defined(W_QUANT_N) || ...) && !defined(W_QUANT_4)` 让未扩展 kernel 编译过 |
| weightTransform | CPU pack 路径（`subBits == N` 分支） |
| `MetalConvolution1x1.mm` `mDequantBits` 设置 | `useIntN ? N : (int4Path ? 4 : 8)` |
| dispatcher 选路 | `mDequantBits == N` 时设 `W_QUANT_N` 宏，避开未扩展的 path |
| prefill（multi-token）| `(mDequantBits == N) && area > 1` 时 force `dequantInShader = false` 走 outer dequant |

**编译错调试**：Metal 编译错运行时打 log（`Warning: pipelineWithSource error`），常见：

| 错误 | 原因 |
|---|---|
| `use of undeclared identifier 'wt'` | 某 `#ifdef` 分支没声明 → alias 没设对，或新加 bit 没补 signature |
| `no viable conversion from uchar4 to uchar4x2` | signature 里某分支声明了 `uchar4`，body 用 `uchar4x2` 读取 → 多个 `#ifdef` 同时为真，body 命中错的分支 |
| 编译过但乱码 | tile byte 顺序反了；或 dispatcher 漏路径 |

---

## 正确性验证

```bash
# build
cd build && make -j8 llm_demo MNN

# 切后端
sed 's/"backend_type": "cpu"/"backend_type": "metal"/' transformers/llm/export/<model>/config.json > <model>/config_mtl.json

# 跑
DYLD_LIBRARY_PATH=build:build/express build/llm_demo transformers/llm/export/<model>/config_mtl.json /tmp/prompt.txt
```

CPU/Metal 同 prompt + `temperature=0.0` 前 N 个 token 应一致（fp16 误差内）。

**模型本身可能就坏**（小模型在低 bit 上量化退化常见），先用更大的模型 baseline CPU 跑通，再验 Metal kernel 正确性。

**数值偏差容忍** 同 OpenCL skill：fp16 路径 abs < 1e-2 / rel < 5e-3，量化 dequant + fp16 abs < 1e-1。

---

## 性能优化方法论

### Apple GPU 经常不是 BW bound

unified memory 带宽很高，但 launch overhead + reduction sync 容易先饱和。LLM decode 的 BW 饱和度往往 < 50%，瓶颈不在 weight BW。

推论：BW 减半（如 w4 → w2）的 decode 提升常被其他开销吃掉，**先量化饱和度再决定要不要做 bit 杠杆**，按饱和度选杠杆同 OpenCL skill。

### Metal 特有杠杆

- **simdgroup matrix (sg_matrix) for prefill**：area > short_seq + simdgroupMatrix supported 时走 `gemm_*_wquant_sg`，比 outer-dequant + fp gemm 快一档。sg_matrix kernel 单 quant bit 单独实例化，新加 bit 想覆盖 prefill 必须扩 sg_matrix；否则只能 prefill 走 outer-dequant + fp gemm，prefill 性能会明显回落。
- **simdgroup reduce (sg_reduce) for decode**：area = 1 走 `gemv_*_wquant_sg`，依赖 `simdgroupReduce`（`simd_sum` intrinsic）。WGS 通常 128（4 simdgroup × 32 lane）。
- **g4mN 模板化**：`conv1x1_gemv_g4mN_wquant_sg` 是 template `<int AREA_THREAD>`，按 area 实例化 N。新加 bit 不扩它就要让 area=1 走 g8（dispatcher 显式 fallback）。

### Mac 不代表 iPhone

iPhone 的 Apple GPU（A 系列）和 Mac 的 M 系列在调度、occupancy、SIMD width 上差异显著，性能数字不能互推。需要 iPhone 性能时单独跑。

---

