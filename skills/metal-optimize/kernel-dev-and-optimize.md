# Metal Kernel 开发与优化

> **配套 SKILL.md 的 sub-doc**：新增 Metal kernel 的**命名、写法**，以及 kernel 层的**优化手段与其收益原理**。
>
> 结构：**第一部分 开发规范**（写任何新 kernel 前必读）；**第二部分 优化知识库**（每个已落地手段的机制、为什么赚、适用边界、陷阱）。
>
> 融合（导出图声明 + 运行时单 dispatch）见 [`graph-fusion.md`](./graph-fusion.md)；管线同步 / replay / H2D 见 [`runtime-scheduling.md`](./runtime-scheduling.md)。

> ## ⛔ 本文件的记录规则
>
> 1. **只写机制与原理**：一个手段为什么能赚、赚在哪一类资源上（带宽 / 并行度 / 指令 / 同步）、
>    什么条件下成立。**不写任何性能数字**（百分比、ms、GB/s、TFLOPS、tok/s、扫点表、配对明细）。
> 2. **不写溯源**：commit hash、日期、设备型号与模型名只在"这条结论只在某类设备/形状成立"时
>    以**类别**出现（如"tensor-coop 设备"、"GQA group 较大的模型"），不写具体机型与日期。
> 3. **不写反例存档 / 实验叙事**：证伪记录、翻案过程、标定过程一律不进本文件。
>    一个被证伪的方向只有在能提炼成**可复用判据**时才留一句（写成"什么条件下这类手段不成立"，
>    放进对应手段的「适用边界」），不留实验档案。
> 4. **陷阱要留**：会让人重复踩的语义/UB/布局陷阱是操作知识，写"触发条件 + 规矩 + 验证方式"，
>    不写事故经过。
> 5. 环境变量只在本文件中作为**路径开关**被引用；名称、默认值、语义的唯一登记处是
>    [`env-registry.md`](./env-registry.md)。

---

# 第一部分：Kernel 开发规范

## 1.1 核心原则

1. **shader 是嵌入的 C++ 字符串**。Metal kernel 写在 `*Shader.hpp` 里的 `R"metal(...)metal";` 字符串中（如 `ConvSimdGroupShader.hpp`），不是独立 `.metal` 文件。改完直接 `make` 就拿到，不需要 codegen。**字符串拼接顺序决定 `#define` 作用域**——「宏 alias 陷阱」特别依赖这点。
   - ⚠️ 旧路径 `source/backend/metal/shader/*.metal` 经 `makeshader.py` 生成 `AllShader.cpp/hpp`，属历史遗留（kernel 名形如 `main0`）。**新 kernel 一律走 `*Shader.hpp` 字符串**，不要往 `shader/` 里加。

2. **dispatcher 要先摸清**。Metal conv1x1 同一 op 常有多条 kernel（gemv 多种 / gemm 多种 / outer dequant），按 `area`、`oc`、`ic_4` 等 case 切。新加 quant bit 不可能一次扩完所有路径，**必须先决定支持哪几条 + 让其他路径不被 dispatch**。

3. **Apple GPU 内部也不能互推**。M 系列不同代、iPhone A 系列与 M 系列在 occupancy / 调度上差异显著；更不能把 Metal 上的结论推给 Vulkan/OpenCL。设备调优分档看 `MTLDevice.architecture.name`，Tensor API 可用性看 `isSupportTensorApi()`，不要混用两者。

4. **正确性 oracle 先于性能**。CPU / `temperature=0` greedy 对拍前 N token 是黄金标准；fp32（`precision: high`）bit-identical 是最强证据。错误 kernel 一样能"变快"。

## 1.2 入口定位与 dispatcher 结构

```bash
grep -rn "OpType_<MyOp>" source/backend/metal/                    # Execution
grep -rn "kernel void <my_kernel>" source/backend/metal/*.hpp     # shader 字符串入口
```

低 bit 量化 conv 入口：`MetalConvolution1x1::onResize`。识别 quant：`mDequantBits ∈ {2,3,4,8}`（在 `MetalConvolutionCommon::loadWeight` 里设置）。`onResize` 把 `area` × `oc` × `dequantInKernel` × simdgroup 能力组合分流到多条路径，扩 quant bit 前先把目标 shape 走的那条 case 标出来。

**当前 conv1x1 低 bit 量化路径**：

```
kernel 内反量化有两条路：legacy wquant kernel（*_wquant_sg）和 tensor-API fusedQuant GEMM。
路由由单个 selectQuantRoute(shape) 决定，返回 gemmPlan.dequantInKernel（= wquantKernel || fusedQuant）+ GEMM plan：
dequantInKernel = dequantScale && gemmPlan.dequantInKernel（kernel 内反量化总闸，一个变量）。
wquantKernel = area==1 || legacy 优先级（先 block quad-fit 检查，不 fit → true；再按家族分）：
  W2/3 → !sgMatrix（无 tile GEMM；有 sgMatrix 走 fused/prepass）
  tensor-API && Q4/Q8 → area<=5 && sgReduce（g4mN 单次权重读，优于 M8 split-K），否则 false
  非 tensor-API Q4/Q8 → area<64 || !sgMatrix → true；否则 权重>4M 参数 && area<512 → true
GEMM plan 先按 shape/gen 定反量化位置（preDequant，不含 API 判据），外层 if(preDequant)，
各分支内按 useTensorApi 选 kernel：M4+（g16+）全 fused；M1–M3（g13–15）area>=64 走
preDequant（kernel 外），其余 fused；非 tensor 两臂都落 simdgroup fp GEMM。
legacy 优先级不可省：纯 preDequant×API 四象限会漏掉 decode/g4mN/大权重等 legacy 抢先（约 38% shape 误判）。
反量化位置 = kernel 内（fusedQuant）还是 kernel 外（Outer*）是独立一维，命名按该维取值：
OuterTcM64 = kernel 外 dequant + tensor cooperative matmul2d；OuterSgM64 = kernel 外 dequant + simdgroup matrix；
FusedQuantM8/M32/M64 = kernel 内 unpack，后缀是 tile 的 M 尺寸。

mDequantScaleBias && dequantInKernel
  ├─ !wquantKernel → conv1x1_fused_q4_gemm_stage*（buildQuantGemm，kernel 内 unpack）
  │    tile M：W4/W8 area<=8 → M8 核（q4：conv1x1_fused_q4_gemm_stage_m8 / _ksplit_m8；
  │    q8：conv1x1_fused_q8_gemm_stage_m8，独立 kernel，不复用 q4 核加宏）；
  │    area>=64 → M64 核（conv1x1_fused_q4_gemm_stage_m64，W8 同核不启用 dual/silu 折叠）；其余 M32 核
  └─ wquantKernel（MNN_LOW_MEMORY）：
     ├─ supportSimdGroupReduce && (area <= short_seq=16 || w23NoMatrix)
     │    ├─ area ∈ [2,16]（或 ≤32 非 heavyMemory，halve 后 piece=2）→ conv1x1_gemv_g4mN_wquant_sg
     │    ├─ lm_head（oc > 16384 且 oc_4 偶，g16LmHeadRoute）→ conv1x1_gemv_g16_wquant_sg
     │    │    （decode 且 oc%8==0、blockCount 偶数除外：落到 area == 1 的 2sg 路由）
     │    ├─ lm_head 其余 shape → conv1x1_gemv_g4m1_2sg_wquant_sg（oc16Sg4K1 / oc8Sg4K2
     │    │    按 §2.1.5 的"网格对 TG 预算"规则选路）
     │    ├─ area == 1 → conv1x1_gemv_g4m1_2sg_wquant_sg   (满足 shape 门控时 GEMV_SPLIT_K=2)
     │    └─ else → conv1x1_gemv_g8_wquant_sg   (仅 w23NoMatrix 大 area 兜底)
     └─ supportSimdGroupMatrix && area > short_seq && oc > 8 && ic_4 偶 → conv1x1_gemm_*_wquant_sg
          (ic_4%8≠0 → 8x16；大 shape → 32x64_wquant_split_k / 32x16 / 16x32；默认 16x16)

mDequantScaleBias && !dequantInKernel
  → conv1x1_w_dequant + fp gemm（buildQuantGemm，kernel 外 dequant → mTempWeight → fp gemm）
     默认 conv1x1_gemm_32x64_split_k_sg；较新 arch-gen → 64x64_split_k；
     tensor-API OuterTcM64 → conv1x1_gemm_device_tensor
```

`w23NoMatrix` = W2/W3 且设备无 simdgroupMatrix：outer-dequant 与 g1z4 fallback 均不可用，g4mN/g8/g16 需覆盖全部 area——g4mN 实例只到 g4m16，超界 area 落 g8。

新加 quant bit 时典型组合是 decode gemv path + prefill outer-dequant path，其他 gemv/gemm 实例 dispatcher 显式 fallback；一次性扩完所有 path 工作量太大。

Conv1x1 的矩阵乘选型不读取 `KVMeta` 或 speculative 标记。同一 M/N/K、量化格式和设备能力必须得到同一路径；小 M 的 tile 行数与 K 分区数统一驱动 kernel、dispatch 网格和 partial buffer，不能为调用场景另设一套布尔开关。

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

- C++ 常量命名 `g` + 大驼峰（`gBasicConvPrefix`、`gPrefillFlashAttnSg`、`gDecodeSplitKV`）。
- 文件头统一 `#if MNN_METAL_ENABLED`；transformer 专用的再套 `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE`。
- **公共头通过 C++ 字符串拼接共享，不是 `#include`**：

  ```cpp
  std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
  ```

  `gBasicConvPrefix`（`ConvSimdGroupShader.hpp`）提供 `#include <metal_stdlib>`、`conv_activation_type`、`activate()`、`conv1x1_constants` 结构体，以及 `namespace MNN { uchar4x2 / char4x4 }` 等打包类型。写新 conv 类 kernel 就拼这个前缀，不要重复声明。
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
| 算法变体 | `split_k` / `rms` / `fused` / `align` / `chunk64` / `inplace` | |
| 后缀 `_sg` | **依赖 simdgroup 原语**（需 `supportSimdGroupReduce()` / `supportSimdGroupMatrix()`）| `prefill_flash_attn_sg` |
| 后缀 `_tc` | **使用 tensor cooperative input**（`matmul2d` + cooperative tensor，由统一的 `isSupportTensorApi()` 检查能力） | `prefill_flash_attn_tc` |
| 版本 | `_v2` / `_v4` = 算法迭代 | `linear_attn_gated_delta_rule_sg_v4` |

前缀谓词放最前（`binary_layernorm_c4_rms_sg` = binary 前导 + layernorm + C4 + RMS + simdgroup）。

**已知不一致（照抄前先确认）**：
- `decode_qk_softmax` 在 `MetalAttentionShader.hpp` 出现多次，靠 `#if` 互斥选择——同名不同体是允许的，但新 kernel 别学。
- 存在名为 `copy`、`main0` 的历史 kernel。
- `linear_attn_gated_norm` 用了 `simd_sum` 却没有 `_sg` 后缀。
- kernel 名与 Execution 类名**不要求**一致。

### 变量 / 字段 / 宏名必须等于真实物理量

**规矩**：变量、成员、宏、env 名指代的必须是代码里真正装的那个量。**修改任何后端算子 / kernel 代码时，先核对沿途已有名字与它实际计算的东西是否一致**——不只是自己新加的名字。名字骗人时，后面所有基于它写的判据、分档、阈值都会跟着错，而且编译、对拍、单测全都发现不了。

高危位置（错了不报错，只是结论错）：

- **分档 / 阈值判据的自变量**。写"按 head 数分档"但 kernel 真正的并行度是 dispatch 的 threadgroup 数（`batch * head_num / qh`）——decode SDPA 的 nsg 分档就踩过：用一个模型标出的默认值到另一个模型上直接反向。**先写清"这个数字物理上是什么"，再写不等式。**
- **"size" / "count" / "step" 这类模糊后缀**。`blockSize` 在 split-K 门控里实际是"每行的 quant block **个数**"，不是"块内元素数"；两种读法都讲得通，用错一个整条门控就失效。这类名字要么改成 `blockCount` / `elemPerBlock`，要么在声明处一行注明单位。
- **随算法迭代变味的宏**。`ROW_2` / `SG_DUAL_STREAM` 早已不是"两行"或"双流"，经 `GEMV_2OCQUAD_PER_SG` 现名 `GEMV_SG_OC_QUADS_2`；`fusedQ4` 只在 tensor API 路径上有效，现名 `tensorApiFusedQuant`。**改算法就顺手改名，别让旧名留着当陷阱。**
- **env 名比宏名活得久**。为不打破外部脚本而保留的旧 env 名必须在 `env-registry.md` 的"功能"列写明它现在真正控制什么，不能让读者从名字反推；没有外部依赖的旧 env 直接删，别留一个名字与语义脱钩的开关。

复核动作：改完一段 kernel，把它读过 / 写过的每个变量对着**实际的索引算术**读一遍（`idx = a * b + c` 里 `b` 是谁的 stride，就决定了 `a` 的物理含义）；名字与算术不符时改名，改名要 `rg` 全仓确认零残留（宏名、pipeline 缓存 key、宏字典三处一起改，见 §1.4）。

## 1.4 编译期变体：宏 + pipeline 缓存 key

代码库**没有任何 `MTLFunctionConstantValues`**，全部变体走 `MTLCompileOptions.preprocessorMacros`：

```objc
MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
option.preprocessorMacros = @{
    @"ftype"      : @(ftype.c_str()),
    @"ftype4"     : @(ftype4.c_str()),
    @"SGS_PER_TG" : @(std::to_string(sgsPerTG).c_str()),
};
```

布尔开关用 `[dic setValue:@"1" forKey:@"XXX"]` 配 shader 里的 `#ifdef`；shader 中可调的取值型宏要写 `#ifndef / #define` 兜底默认值。

**pipeline 缓存 key 是 `std::vector<std::string>`**（`MetalBackend.hpp` / `.mm`）：key[0] = kernel 名，key[1] = ftype，之后每个**生效的宏**依次 append。

```cpp
std::vector<std::string> keys = {"linear_attn_gated_norm", ftype, "sgs" + std::to_string(sgsPerTG)};
```

`MetalAttention.mm` 的 pipeline 装配是最规范的范式（多变体、互斥判断齐全）。

> ⚠️ **新增变体宏必须同步改 4 处，漏一处就是静默错误**：
> 1. shader 里的 `#ifdef` 分支（含 `#ifndef` 默认值）；
> 2. `keys.emplace_back(...)`——**漏了会取到别的变体的缓存 pipeline**；
> 3. `[dic setValue:... forKey:...]`——漏了宏根本没生效；
> 4. `onResize` 里因该宏改变的 grid / threadgroup 尺寸。
>
> 第 4 条例：`MetalConvolution1x1.mm` 里双流宏把 grid.x 改成 `UP_DIV(slice,4)`，`GATE_UP_FUSED` 加 z=2 维并强制 64 线程。
>
> **rebase / 解冲突后必须逐处重核这 4 处**：key、宏字典、grid 三者不同源，冲突解决时最容易只保住其中一两处，症状是"host 按变体 A 派发、kernel 编成变体 B"，输出静默半错。

其他约定：
- 取值型宏的 key 要带前缀区分（`"HEAD_DIM_" + head_dim_str`，避免与别的数字宏撞串）。
- 互斥的融合宏要显式挡住叠加。
- 编译失败用 `pipelineCompileFailed(keys)` 记住，避免每次 resize 重试。

## 1.5 Execution 骨架与注册

继承 `MetalExecution`（`MetalExecution.hpp`），只需实现 `onResize` + `onEncode`。**最小范本：`MetalGatedRMSNorm.mm`**。

`onResize` 标准流程：

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

**注册**：`REGISTER_METAL_OP_CREATOR` 与 `REGISTER_METAL_OP_TRANSFORMER_CREATOR` 宏展开完全相同，区别只在约定与调用点——transformer 版在 `MetalOPRegister.mm` 里被 `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` 包住。

CMake 门控掉的 kernel，`#else` 分支必须提供同名空函数，否则注册表链接失败：

```cpp
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

**坑**：alias 让 `#ifdef W_QUANT_4` 在你想扩展的那个 kernel 里**也**被命中。signature 里 `#ifdef W_QUANT_2` 第一个匹配（`uchar4* wt`），body 里 `#ifdef W_QUANT_4` 也匹配（`uchar4x2`）→ 类型混淆 → 要么编译失败、要么编译过而数值错。

**规矩**：扩展的 kernel 里**所有相关 `#ifdef` 必须按 W_QUANT_2 → W_QUANT_3 → W_QUANT_4 → W_QUANT_8 顺序**，新 bit 优先匹配；signature 和 body 都要这个顺序，少一处都是 sneaky 错。改 alias 字符串内任何 kernel 的 `#ifdef` 阶梯前，先确认该 kernel 是否在 W2/3 下编译——低 bit pipeline 编译失败是静默的，可以潜伏很久。

### 陷阱 B：dispatcher 漏路径（lm_head）

LLM 的 lm_head conv（`oc = vocab_size`，十万量级）走 `oc > 16384` 的特殊路径（如 `g16`）。新加 quant bit 没扩 g16 时，dispatch 还会进 g16 → 用错 layout 读 buffer → 数值错或 crash。

应对：dispatcher 选路写白名单，把没扩的路径强制 fallback 到已扩的（如 `g8`）。

### 陷阱 C：weightTransform 的多签名同步

`weightTransform(...)` 在 `MetalConvolutionCommon`、`MetalConvolutionWinograd`、`MetalConvolutionDepthwise` 都有 override。改签名（如加 `subBits` 参数）时这 3 处 + `.hpp` 共 4 处要同步，否则 build 报 `'override' but does not override any member function`。

### 陷阱 D：getDequantScale 的 `coef` fp16 范围补偿

Metal `getDequantScale` 用 `coef = 1000/max_data` 做 fp16 范围补偿（host 写 `s*coef`，shader `/coef`）。新加 quant bit 不要碰这个流程；scale/offset 的 originOffset 折叠**完全在 host alpha 写入时**完成，shader 一律按 signed 解出后 `signed_w * scale + bias` 即可，不要再折一次。

### 陷阱 E：tile 内 byte index 选择（OC vs K_inner）

W_QUANT_8 的 tile layout 是 `byte = ro * 4 + ri`（OC 外、K_inner 内），即一次 16 字节取一个 (4 OC, 4 IC) tile，`w[i] = char4` 是 1 OC × 4 IC。

新 bit 的 packing 必须**镜像已有最高 bit kernel 的字节顺序**：byte i = OC i 的多个 IC，不能反过来变成"byte i = IC i 的多个 OC"。写反了 shader 照样编译通过、kernel 能跑，但输出乱码——只有 dump 第一个 op 的 weight 前几行和 CPU 对照才能发现。

### 陷阱 F：fp16 后端下 `Tensor::createDevice<float>` 只给一半字节

Metal 后端开 fp16（`useFp16InsteadFp32`）时，float 类型的 device tensor 按 **2 字节/元素**分配；若 shader 把这块 scratch 按 `device float*`（4B）读写，**buffer 实际只有需求的一半**。越界不会立刻炸：先静默踩别的内存（数值损坏），直到撞上未映射页才报 GPU 故障（`kIOGPUCommandBufferCallbackErrorInnocentVictim`，且真凶 buffer 常不在日志里）。

这种越界还常常**与规模相关**：分配公式少一个 `sizeof(float)` 时，小 kv / 小 head_dim 恰好贴边不炸，长上下文或大 head_dim 才引爆——短形状测试全绿不代表安全。

**规矩**：shader 按 `device float` 消费的 scratch，host 一律 `createDevice<uint8_t>({bytes})` 按字节分配，公式写明 `* sizeof(float)`。检查现有代码时 grep `createDevice<float>` 与对应 shader 的指针类型是否一致。

定位手段：`MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1` 直接报出 kernel 名与越界 offset。

### 陷阱 G：barrier 与 per-simdgroup 早退不可共存

给已有 kernel 的前导/尾段加 threadgroup 归约时，真正的代价不是 barrier 指令本身，而是它对**上游控制流**施加的全局约束：TG 内每个线程都必须到达 barrier。而 2sg/4sg GEMV 的惯用写法恰恰是先按 simdgroup 算出 `uz` 再 `if (uz >= output_slice) return;`——尾部 threadgroup 于是出现"一个 SG 已 return、另一个 SG 在等 barrier"。Metal 下这是 UB：可能挂死，也可能静默出错。

**修法**：
1. 把含 barrier 的前导段**整体上提**到所有 per-simdgroup 出界检查之前；
2. 早退判定换成 **TG-uniform** 形式——用"整个 TG 覆盖的最小 index"（`tg_first_uz = tg_x * slicesPerTG`）判断：全 TG 结论一致 ⇒ 要么全退要么全留；
3. 部分有活的 TG 里，越界的那个 SG 压到归约**之后**再各自 return。

**规矩**：加 barrier 前先枚举它上游所有 `return` / `continue`，逐个判定是否 TG-uniform，不是的就下移或改造成 uniform 形式。**barrier 结构必须编译期确定**（宏变体，不要用运行时分支包住 barrier）。配套还要重审"唯一写者"守卫（§2.1.9）。

**验证**：这类 bug 只在尾块出现（`output_slice % slicesPerTG != 0`），整除 shape 跑一万遍都测不出——必须专门用**非整除 output_channel** 的 shape 对拍，不能只跑 2 的幂维度。

### 陷阱 H：`in4 * FLOAT4x4` 改写成标量循环 = 转置积

Metal 的 `v * M`（行向量×矩阵）语义是 `(v*M)[j] = dot(v, M[j])`，其中 `M[j]` 是**列** j。把 `result += in4 * w_dequant` 改写成 `for i: result += in4[i] * (FLOAT4(w[i])*scale[i]+bias[i])` 看起来只是消掉 4x4 临时矩阵，实际是**转置积 + scale/bias lane 错位**：

```
前者 result[j] = Σ_i in4[i]·(w[j][i]·s[j]+b[j])
后者 result[j] = Σ_i in4[i]·(w[i][j]·s[i]+b[i])
```

正确的无临时矩阵写法是 per-output-lane 点积 + 延迟 dequant：`raw_dot[j] += dot(in4, FLOAT4(w[j])); result = raw_dot*scale + input_sum*bias`。

**规矩**：改写任何 `v*M` / `M*v` 形式前，先把两种形式的分量公式各写一行对照；review 时看到"row by row / same math"类注释要求给出该对照。W8 的 `char4x4` 中 `w[j]` = 输出 lane j 的 4 个 ic 权重（transform 按 `ro*4+ri` 打包），不是 ic lane j。

**这类错误的排查配方**：错值对所有 kernel 变体开关都不变 ⇒ 错在共享数据或共享代码体；`got[c]-bias[c]` 出现跨通道成对相等 ⇒ 行/lane 索引错位；先用 CPU 参考验证 weight transform（排除布局），最后用脚本**精确模拟嫌疑 kernel 的数学**，模拟值与实测吻合即定案。

### 陷阱 I：`ftype4*` 指针配标量单位偏移 = 4× 越址

形参声明成 `device ftype4*` 而偏移量仍按**标量**计算时，指针按 ftype4 步进，**偏移放大 4 倍**，整块数据从错误地址取数。典型来源是把 kernel 从 `ftype*` 改成 `ftype4*` 而索引表达式没跟着改；同 kernel 内若有分支先 `(const device char*)` 再加偏移（int8 下标量==字节）反而是对的，**这种"一半分支正确"的形态在 review 里非常像对的**。

表现是输出乱码，不崩不告警；影响面往往是"某个 seq 区间"或"某个回退路径"，正好落在常规测试的缝隙里。

**规矩**：把 kernel 形参在标量指针（`ftype*`）和向量指针（`ftype4*`）之间改动时，**逐个检查所有用它做指针算术的地方，确认偏移量的单位跟着换了**；同一 kernel 里若有分支先 cast 成 `char*` 再偏移，那条分支不能作为"其他分支也对"的证据。

**定位配方**：① 用 seqLen 逐点 sweep 找**精确激活边界**，边界对齐某个 flag 的阈值就说明"不是数值误差而是换了 kernel"；② 一条链路里多级都有 tensor / 非 tensor 两版且由同一个 flag 一起切换时，必须能**分别**把某一级退回非 tensor 版才能二分；③ ⚠️ 做这个隔离时，**shader 名、pipeline 缓存 key 的宏、dispatch grid 三者必须一起改**——只换 shader 名会让宏与 grid 不匹配，制造出与真 bug 无关的假失败。

### 陷阱 J：`Tensor::createDevice` 默认 TENSORFLOW 维度序，`channel()` 读到 area

`Tensor::createDevice<T>({n,c,h,w})` 的 dimType 默认是 **TENSORFLOW (NHWC)**，`channel()` 读 dim[3]。直接构造 Execution 做路由/形状测试时若忘传 `Tensor::CAFFE`，ic/oc 会被读成 w/h，路由决策整体错乱但**不崩不告警**；当 oc==area 时个别 case 还会碰巧"看起来对"，掩盖问题。

**规矩**：测试或 host 代码手工 `createDevice` 喂给按 NCHW 语义读 `channel()/width()` 的 Execution 时，一律显式传 `Tensor::CAFFE`。快照/路由结果与源码逻辑矛盾时，先打印 Execution 实际读到的 ic/oc，再怀疑二进制新鲜度。

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
| prefill（multi-token）| `selectQuantRoute` 里为 N 补分支（legacy 优先级 + GEMM plan），area>1 落到有 `W_QUANT_N` 支持的 kernel，无则走 prepass |

**跨分支移植优化前先核对隐含前置**：性能提交可能基于已 squash 的基础设施提交开发，单看补丁看不出依赖。移植前用 `git log -S'<新增 override/API>' -- <目录>` 定位接口来源，确认目标分支是否已有实现；没有就只补最小前置提交，不要整文件采用来源分支的公共 registry/backend 文件。解决冲突后检查：核心实现与来源分支目标行为一致；被新 kernel 取代的旧 shader、pipeline 成员、dispatch 分支必须一起删除并用 `rg` 确认零引用。最后 clean rebuild + 至少一次真实 Metal 运行——**普通 C++ 编译发现不了运行时 Metal shader 编译错误**。

**编译错调试**：Metal 编译错在运行时打 log（`Warning: pipelineWithSource error`）：

| 错误 | 原因 |
|---|---|
| `use of undeclared identifier 'wt'` | 某 `#ifdef` 分支没声明 → alias 没设对，或新 bit 没补 signature |
| `no viable conversion from uchar4 to uchar4x2` | 多个 `#ifdef` 同时为真，body 命中错的分支（陷阱 A）|
| 编译过但乱码 | tile byte 顺序反了；或 dispatcher 漏路径 |

shader 编译报错原文走 `NSLog`，**stderr 非 tty 时被丢弃**，管道里只剩一行裸的 `pipelineWithSourceOption error.`。用 `script -q /dev/null ./run_test.out ...` 包一层即可拿到原文。

⚠️ **pipeline 编译失败常常是静默"通过"**：attention 一类 op 会 fallback 到旧路径，测试照样绿。测新 kernel 必须同时确认它的 banner 出现，且没有 `pipeline unavailable` 之类的回退提示。

**提交前最后一遍：命名复核**。改动涉及的变量 / 成员 / 宏 / env 名，对着实际索引算术与分档不等式再读一遍，确认名字就是它装的那个量；不符就改名并 `rg` 确认零残留。判据见 §1.3「变量 / 字段 / 宏名必须等于真实物理量」。

## 1.9 正确性验证

```bash
cd build && make -j8 llm_demo MNN
sed 's/"backend_type": "cpu"/"backend_type": "metal"/' <model>/config.json > <model>/config_mtl.json
DYLD_LIBRARY_PATH=build:build/express build/llm_demo <model>/config_mtl.json /tmp/prompt.txt
```

CPU / Metal 同 prompt + `temperature=0.0` 前 N 个 token 应一致（fp16 误差内）。**模型本身可能就坏**（小模型在低 bit 上量化退化常见），先用更大模型 baseline CPU 跑通，再验 Metal kernel。

数值偏差容忍：fp16 路径 abs < 1e-2 / rel < 5e-3；量化 dequant + fp16 abs < 1e-1。

**低 bit 专用 oracle**：低 bit 小模型输出常是乱码，而 CPU oracle 不可用（CPU int2/3 ARM kernel 在 Apple Silicon 上有已知 bug）。可用 oracle = **`transformers/llm/export/mnn_quant_ref.py`**：从导出的 `.mnn.weight` 直接解码权重（header + MSB-first 位解包 + fp16 alpha）注入 HF 模型 greedy 生成，独立于 MNN 运行时。判据：
- 连贯模型（W3/W4）逐 token 一致；
- 乱码模型看共同前缀 + **噪声敏感度标定**：给 oracle logits 加 ε 噪声，若 1e-3 不翻转、1e-2 在与 Metal 分叉点相近位置翻转 ⇒ Metal 偏差是精度级 = 合法；量级再大才是 bug。

⚠️ **改归约顺序的改动不能用 byte-compare 判正确**（nsg、split-K、跨 SG 合并都属此类）：它给出的是另一个合法求和序，判据只能是带容差的 op 单测 + 归约序论证。

⚠️ **对拍口径**：`llm_demo` 输出内嵌 `cost time` 行与末尾统计块，直接 hash 整个 stdout 会让**默认稳定配置也"每次不同"**。对拍必须先剔除计时行。

⚠️ **greedy 基线未必确定**：部分机器 / 模型上同一份 build 连跑两遍文本就会分叉（热节流、非确定归约都可能导致）。**用 byte-compare 之前先自证基线本身可重复**（同臂跑两遍），否则会把基线抖动当成自己的 bug。

⚠️ **冷/热**：刚删 `mnn_cachefile.bin` 的第一次跑与后续不同（pipeline cache），所有对拍前先预热一次。

⚠️ **默认 `config.json` 通常是 `backend_type: cpu` + `sampler_type: mixed`**：直接拿它跑 `llm_demo` 既没上 Metal 也不确定；对拍必须另存 `backend_type: metal` + `sampler_type: greedy` 的副本。从模型 config 派生副本时只改了 sampler 会留下 `cpu`（多模态 config 的 `mllm.backend_type` 是另一项）；跑前用 `MNN_METAL_PIPELINE_INFO=1` 确认目标 kernel 的 pipeline 确实出现，0 条即没上 Metal，对拍结论无效。

⚠️ **“无大段重复”必须对照 HEAD**：0.6B / 0.8B 级小模型在 greedy 和 mixed 下都会退化成循环。改了累加顺序后，循环会在不同 prompt 之间“搬家”。判据是新版本没有 HEAD 上不存在、且更长的重复，而不是重复数为 0。

## 1.10 Metal tensor API：cooperative tensor 的逐元素布局

MNN 现有 tensor 用法（`prefill_qk_tensor`、`conv1x1_fused_q4_gemm_stage`）只用到「destination cooperative tensor + `run(sA,sB,cT)` + 一次性 `cT.store()`」。要写**融合 attention** 这类需要在两次 matmul 之间对中间结果做 online softmax 的 kernel，必须用到三个额外能力：

| 能力 | 说明 |
|---|---|
| `get_left_input_cooperative_tensor<A,B,C>()` / `get_right_...` | 让 A/B 也是寄存器 coop tensor，把上一次 matmul 的 destination 直接当下一次的 A（P 不落内存）|
| coop tensor 的 `operator[]` | 逐元素读写，softmax/mask 全在寄存器完成 |
| `run(ct_a, ct_b, ct_c)` | 三个操作数都是 coop tensor |
| `get_capacity()` / `get_multidimensional_index(i)` | 官方坐标查询 |

⚠️ **硬约束**：input cooperative tensor 只允许单 simdgroup 作用域 ⇒ 必须 `metal::execution_simdgroup`，每个 simdgroup 独立跑一个小 matmul。`execution_simdgroups<4>` + threadgroup source tensor 的写法**无法表达 online softmax**。

⚠️ `mSupportTensorApi` 仅表示构建和设备准入。使用真实 kernel 的 pipeline 创建结果决定是否回退；可选编译失败不能提前标记整个 runtime 失败，编译成功也不能替代数值验证。

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

**N=64 destination**（`matmul2d_descriptor(16, 64, 32, false, true, true, multiply_accumulate)`）：A cap 16（A 是 16×K，与 N 无关）、B cap 64、D cap 32。D 就是**两块 N=32 的 D tile 沿 n 堆叠**，`fm`/`fn` 与 N=32 完全一致：

```
M = fm + ((i>>2)&1)*8
N = fn + (i&3) + ((i>>3)&1)*16 + ((i>>4)&1)*32
```

即 `i & 15` 部分与 N=32 逐位相同 ⇒ 现有 16 元素循环可直接推广成"外层跑 `i>>4` 个 32 宽 block，每 block 结构不变、n 偏移 `nf*32`"。

⚠️ **K 宽度不同、`transpose_b` 不同时，额外那一级 K 落在元素索引的哪一位不能靠类比推导**（`tb=false` 的 B 与 `tb=true` 的 B 结论不同），必须单独 dump 一次。

> ⚠️ Apple 文档明确说 coop tensor 布局是 implementation-defined。上表是实测值，**换设备/OS 需重跑 dump**。

### dump 手法：改在 MNN 进程内做

独立 harness 里一旦调 `get_multidimensional_index` 或 `run()` 就可能被 SIGKILL（无 stdout、无 crash report；只读 `get_capacity()` 的版本正常）。**改在 MNN 自己的进程里 dump**，用独立诊断 shader、`pipelineWithSourceOption` 和自建 command queue dispatch 32 线程；诊断代码不能作为生产初始化步骤。

且不必依赖 `get_multidimensional_index`：把 A/B 都做成 device tensor handle，用数值编码反推坐标——
`A[m][k] = (k==0)?1:(k==1)?m:0`，`B[n][k] = (k==0)?n:(k==1)?64:0` ⇒ `D[m][n] = n + 64*m`，读回来 `m = v/64, n = v%64`，每个元素自报坐标（fp16 精确）。tensor `dextents` 第 0 维 = 内存连续维，`slice(origin0, origin1)` 同序。模板函数里调 `mm.get_destination_cooperative_tensor<...>` 要写 `mm.template ...`。

**两个 API 陷阱**：
1. `matmul2d::run()` 收的是**非 const 左值引用**。`run(ct_a, t_v.slice(...), ct_o)` 这种内联 rvalue 绑不上，slice 必须先赋给命名局部变量。
2. tensor 的元素类型**必须无 cv 限定**。`tensor<const device ftype, ...>` 触发 `static_assert`，并连带 `get_destination_cooperative_tensor` 匹配失败；写 `tensor<device ftype, ...>` + 强转。
3. **A 用 cooperative tensor、B 用 tensor handle 是合法混搭。**

头文件离线可读，操作数形态/布局的合法性可以直接查，不必只靠运行时 pipeline 编译失败反推：
`/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h`、
`/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/*/Libraries/lib/clang/*/include/metal/metal_tensor`。

## 1.11 Apple GPU 杠杆选择

- **simdgroup matrix（sg_matrix）for prefill**：`area > short_seq` + 支持 simdgroupMatrix 时走 `gemm_*_wquant_sg`，比 outer-dequant + fp gemm 快一档。sg_matrix kernel 每个 quant bit 单独实例化，新加 bit 想覆盖 prefill 必须扩它。
- **simdgroup reduce（sg_reduce）for decode**：`area == 1` 走 `gemv_*_wquant_sg`，依赖 `simd_sum`。
- **g4mN 模板化**：`conv1x1_gemv_g4mN_wquant_sg` 是 template `<int AREA_THREAD>`，按 area 实例化 N。
- **Metal4 tensor API**：tensor-coop 设备上的 `matmul2d` + cooperative tensor（§1.10）。
- ⚠️ **`MTLGPUFamily` 区分不了相邻两代**（同 Apple9），设备分档必须用 `MTLDevice.architecture.name`（`applegpu_g<gen><size>`）；旧 OS 无此 API 时保守关（gen 解析为 0，所有 gen 门控 flag 关）。tensor ops 的开关是 **Metal4 family（OS 侧条件）&& `gpuGen >= 17`（芯片侧条件，M5/A19 才有 AI TensorCore）**——"支持 Metal4 family"不等于"有 TensorCore 硬件"（iOS 26 会把 Metal 4 API 下放到 A13）。判断"本机走哪条路"直接读 `MetalRuntime` 构造函数里 `gpuGen`/`gpuSize` 的派生式。
- **archName 世代对照（实测锚点）**：g13 = M1/A14，g14 = M2/A15，g16 = M4，g17 = M5；size 字母 `p` = 手机 A 系列、`g` = base M-chip、`s` = Pro。g15（M3/A16 cohort）未实测；按"M4 = g16"外推，A17 Pro 应为 `g16p`，**这是推断值，从未在真机上直接见证**。两条反直觉的推论：① **gen 分不出 M-chip 与同 cohort 的 A-chip**（M2 与 A15 同为 g14）；② **size 也分不出 M-chip iPad 与 A-chip iPad**（两者都是 `g`）。⇒ 凡是要区分产品线的分档，**只有 `device.name` 前缀（`"Apple M"` vs `"Apple A"`）可用**，别指望 gen/size；反过来，凡是只需要"GPU 世代"的分档就用 gen，别用 name 去数型号。
- ⚠️ **芯片分档要用 `mSimdGroupMatrix` 打底，不能用 `mSimdGroupReduce`**：后者 OR 进了 Metal3 family，而 Metal3 在无 simdgroup_matrix 的最老一档上也读 true，用它打底会把那一档拖进 sg-matrix 标定里（历史上真出过这个门控洞）。

---

# 第二部分：Kernel 优化知识库

> 每节只写：**机制**（做了什么）、**为什么赚**（省下哪类资源）、**适用边界**（什么条件下不成立）、**陷阱**。
> 数字与历史实验记录不进本文件。

## 2.0 优化总纲

LLM decode 每步生成一个 token，核心链路：

```
RMSNorm → Q/K/V Linear(GEMV) → RoPE → Attention(QK+Softmax+AV) → O Linear(GEMV)
       → RMSNorm → Gate/Up Linear(GEMV) → SiLU*mul → Down Linear(GEMV) → Residual
```

- **Decode**：GEMV 占大头（六到八成），是优化主战场；其次 Attention 和 RMSNorm。
- **Prefill**：GEMM 约占一半；Attention 三段中间物化（mTempQK / mTempSoftMax）是长 prompt 显存/带宽瓶颈，causal-bound / flash-attention 是主杠杆。
- **战略**：**Prefill 走 kernel 深化，decode 走管线深化**（decode 侧管线优化见 [`runtime-scheduling.md`](./runtime-scheduling.md)）。

### 定性画像：小模型 decode 是 GPU-bound 且 occupancy 受限

实测口径下 GPU busy ≈ 生产 wall ⇒ decode 基本 GPU-bound，**没有可回收的空泡**。"双实例并发能加速"不等于单实例有空闲，而是**单实例 kernel 填不满 GPU（occupancy 不足）**——是 occupancy 效应，不是 idle-gap 效应。这两者的处方完全不同：前者要加并行度/在途请求，后者要治同步点。

decode GEMV 的带宽兑现率**随单 dispatch 体量单调上升**：几 MB 的小投影只到峰值的四五成，上百 MB 的 lm_head 能到八九成。差别不在布局也不在图结构，而在**单次 dispatch 撑不起 ramp-up 与足够的在途读**。⇒ 小 GEMV 的正确解法是**减少 dispatch 数 / 增大单 dispatch 体量 / 提高在途请求数**，不是继续抠 kernel 指令。

> 优化项要**按模型档分别评估**：大模型 GEMV 占比更高且 GPU busy 逼近 wall，GPU 侧优化直接兑现；小模型受管线与 occupancy 约束，同一个 kernel 收益可能完全不同。

### 铁律：占比小的 kernel 必须在 op 单测上 A/B，不能只看 e2e

> 📖 **完整诊断流程**（怎么造 op 单测、怎么造对手镜像、固定/流式分解、标定尺度分工、
> 收工判据）见 [`op-bench-and-diagnosis.md`](./op-bench-and-diagnosis.md)。本节只讲"为什么"。

**在 e2e 上 A/B 一个只占 x 成时间的 kernel，等于把信号按 x 缩小再和噪声比。** e2e run-to-run 抖动在个位数百分比量级；kernel 侧的 ΔK 落到 e2e 只剩 `ΔK × 占比`。占比低时，真实收益会整段埋进噪声带，**测多少轮 rep 都救不回来**——正负号由当天的热漂移决定。这个失效模式已经真实地把可观的收益雪藏过，还附带了错误的机制解释。

**流程**：

1. 先查 profile 拿到目标 kernel 的 e2e 占比。
2. 占比低（经验线 <30%）时，**先建/找 op 级 speed 单测**（`test/speed/*.cpp`，形状照抄目标模型），在单测上做定向 A/B——这里它占满被测时间，ΔK 满幅可见。
3. 单测确认方向后，**再回 e2e 确认「不回归」**（而不是用 e2e 去发现收益）。e2e 的作用是排除该改动在真实调度下引入的副作用，不是测量精度来源。
4. 只有在单测上**同样为负**的实验才能称为证伪。仅 e2e 为负 ⇒ 只能说"e2e 未观察到收益，待单测复核"。

反向也成立：**同 op 紧循环的 micro 带宽 ≠ 上下文内性能**。背靠背跑同一个 kernel 时 TG 尺寸/占用的权衡与真实链路不同，micro 里"慢"的 kernel 在 e2e 里可能正好填满邻居的空泡。**micro 只用于产生假设，是否落地一律 e2e 交替配对裁决。**

> ⚠️ **末位 precision 必须是 `2`（Low=fp16）**。`1` 是 High=fp32，会让所有 fp16-only
> prefill kernel **静默**退回旧路径——不报错、不打 banner，只是数字掉一档。跑之前先确认
> 目标 kernel 的 active banner 出现。

### 铁律：结论必须写清适用域；一个旋钮动两个量时必须先隔离

上一条管**测量口径**，这一条管**结论的适用域**。

- **写下适用域**。任何"某手段无效"的结论都必须连同标定形状一起记：`head_num`、`group_size`、`head_dim`、ctx、设备档。`「X 无效」`不是合法结论，`「X 在 group=2 / 16 head 上无效」`才是。
- **先问这个旋钮动了几个量**。若同时动两个方向相反的量（典型：共享度↑ vs 并行度↓、tile 变大 vs 寄存器压力↑、batch↑ vs occupancy↓），**必须把其中一个固定住再扫**，否则测到的是合力，符号没有诊断意义。正确做法是换坐标轴——例如按 threadgroup 数分组比较，不同模型的数据立刻自洽。
- **优先在极端形状上验机制**，再回中间形状定门控。冗余度最高的配置信号最强；在冗余度最低的配置上判"机制是否成立"等于自找最差信噪比。
- **门控参数要挑真正的因变量**，不要挑它的代理变量。代理变量（group_size、TG 数）在标定过的两三个形状上和真因变量共线，换个形状就分道扬镳。
- **警惕共线的候选解释**。两个候选解释被现有数据同时满足时，**先去找能把它们分开的第三个形状，再动门控**——否则会把 confounder 写进产品代码。
- **错误的机制解释比错误的结论更贵**：它会连带污染后续的实验设计（把注意力锁在错误的自变量上）。

### 归因方法：kernel 内部消融阶梯（ablation ladder）

适用场景：**已确认某 kernel 是瓶颈，但不知道时间花在它内部哪一类工作上。** GPU counter 只给整个 kernel 的 busy / 带宽 / 占用率，给不出"softmax 占多少、smem fragment load 占多少"。

**核心思路：不要正向猜哪里慢，要反向逐类删，读差值。**

1. **逐类删工作，每类一个临时 env 宏。** 一个探针整体去掉一类工作（标量 softmax / 每条 MMA 前的 fragment load / device 端 K·V 搬运 / mask …）。**探针的输出故意是错的**，它不承担正确性，只回答"少做这一类时快多少"。从完整 kernel 一路删到只剩核心计算（MMA/FMA），相邻两级之差就是该类工作的成本，最后一级是**地板值**（这个算法形态在本设备上的下限）。
2. **每个探针都要防死代码消除。** 删掉一类工作时编译器可能顺带删掉下游一大片，于是该类成本被高估。**做法**：删工作但保留一个编译期不可推导的非零值喂给下游；**自查**：一个占几十条指令的小段删掉后省掉半个 kernel，一定是 DCE 而不是发现。
3. **拿地板值和对手比，不要拿总时间和对手比——这是整套方法杠杆最大的一步。** 前提是先核对四项可比性：**MMA/FLOP 条数、tile 形状、grid 规模、稀疏/提前退出策略**。

   | 观察 | 结论 | 该做什么 |
   |---|---|---|
   | 对手 ≈ 你的地板 | 两边**算的量一样**，差距全在**重叠效率**（访存与依赖链没被计算盖住）| 提高 MMA:load 比、批量预发射、加宽复用；**不要**再去省指令 |
   | 对手 < 你的地板 | 差距在**算法/工作量**（tile 形状、MMA 条数、有没有提前退出）| 改算法形态；改重叠是白费力 |
   | 对手 > 你的总时间 | 已领先，转去守回归 | —— |

   反过来说：**没有地板值，"对手比我快一截"这句话无法转成任何动作**——它既可能是算法问题也可能是重叠问题，两者的处置完全相反。
4. **按机制选手段。** 判定为重叠问题后，候选手段从 §2.5 里按"能否提高计算:访存比"筛，而不是列一张通用清单逐个试。

**四条纪律**：

- **组合扫，不要单旋钮扫。** 知识点之间有顺序依赖：A 的停顿盖住了 B 的收益时，单独扫 B 会把一个正向项判负。已判负的旋钮，在同一 kernel 有新的正向项落地后**要复测**。
- **先证明变体真的被编译进去了。** "改了没效果"和"改的代码根本没进 shader"在数字上完全一样。做法：在 `#ifdef` 里塞一个 `#error`，确认只有开开关时才 pipeline 编译失败，再撤掉。同理**测试全绿 ≠ 该路径被跑过**——加 variant 就要同时加一个真能点亮它的用例。
- **度量卫生**：交替轮转配对（A/B、B/A）、丢弃每臂第一轮 warmup、轮内取 min、多次同向且不重叠才算信号；手工替换 `libMNN.dylib` 后必须 `codesign -f -s -`，否则进程被 SIGKILL 而脚本只记下一个 0；多变量 env 用 `${=var}`（SKILL.md 原则 7b）；测 e2e 前后要降温，无风扇机型连续跑会大幅漂移。
- **停在哪里要诚实。** 收工时要能说出**剩余差距的机制**和**它被什么挡住**（例：想提高 MMA:load 比需要更大的 q tile，但寄存器足迹会溢出）。说不出机制就不是"已到极限"，只是还没查到；归因于"编译器已经优化得很好了"一律视为未完成。

## 2.1 GEMV（decode 主战场）

### 2.1.1 Q4 GEMV Deferred Dequantization

- **机制**：标准 Q4 GEMV 在累积内层同时做 nibble 解包 + 反量化（乘 scale + bias）+ FMA。改成内层只做整数累积（int8 × int8 → int32），循环结束后一次性反量化。

  ```metal
  int32_t isum = 0;
  for (k) { int8_t w = unpack(packed_w[k]); isum += int32_t(input_quant[k]) * int32_t(w); }
  sum = half(isum) * scale + bias;
  ```
- **为什么赚**：把每步的 fp16 乘加从内层彻底移走，内层只剩整数 MAC 与解包；反量化成本从 O(K) 降到 O(1)。
- **实现要点**：① input 也要动态量化为 int8（`mTempInput` + `mInputScales` 双 buffer）；② 结果式 `isum * input_scale * weight_scale + weight_bias * input_sum`；③ weight 非对称量化 ⇒ 需额外累积 `sum(input_quant)` 做 bias 修正。
- **门控**：`area == 1`（decode）+ `supportSimdGroupReduce`。

### 2.1.2 双 Simdgroup GEMV + 宽向量加载（g4m1_2sg）

- **机制**：① 一个 threadgroup 内 2 个 simdgroup 分别处理不同 OC 范围，input 经 threadgroup memory 共享，TG 数减半；② weight 用 `ushort4`（8B）一次读取，load 指令数减半。
- **为什么赚**：共享 input 读 + 减少 load 指令数；TG 数减半降低调度固定开销。
- **陷阱**：宽 load 需 weight buffer 对齐；双 simdgroup 要求 OC 下限，小 OC 层仍走单 simdgroup kernel。

### 2.1.3 Pre-scaling Nibble Extraction

- **机制**：host pack weight 时预乘位权系数，nibble 提取用乘法（MAD）替代 `>>` / `&` / `-8` 三步，同时隐式完成 zero point 减法——mask-only 解包，位权补偿预乘进输入。
- **为什么赚**：省掉每个 nibble 的移位与减法 ALU 指令。**pre-scale 全取 2 的幂 ⇒ fp 位精确**，不引入数值差异。
- **适用边界**：只在 ALU 占比可见的路径上有效；纯带宽瓶颈的形状上指令级微调不兑现（见 §2.1.4）。

### 2.1.4 GEMV 带宽画像：小 dispatch 是 latency-bound

- **观察**：同一个 GEMV kernel 的带宽兑现率随每 dispatch 权重体量单调上升，小投影只到峰值的三四成，超大 lm_head 接近顶。
- **推论（本节最重要的一条）**：**小权重 GEMV 是 latency-bound，不是 kernel-bound。** 正确解法是提高在途读（split-K / 双流）、减少 dispatch 数（融合）、增大单 dispatch 体量；**继续微调 lane 划分、循环展开、常量化几乎不兑现**。
- **同一不变量解释了一批"重分组"手段为什么无效**：总在途字节数 =（oc/4 个 simdgroup）×（每 simdgroup 的 load 数），**在任何重新分组下都不变**。只把同样的读换个分法（改 lane 划分、改行/列归属）不会变快；能变快的是**真正增加并发在途请求**或**减少总字节数**的改动。
- **lm_head 方向的 kernel 级优化已关闭**：它既不缺在途 lane（kernel 内 split-K 无收益，因为加倍在途 lane 不减少搬运字节），e2e 也不受它的 GPU 时间支配（省下的 GPU 时间被尾部 GPU→CPU 同步吃掉）。下一个杠杆是设备端采样，见 `runtime-scheduling.md`。
- **编译期常量化不是收益来源**：把循环边界、块步长做成编译期宏省下的整数除法与尾判，会藏在访存延迟下面。投入前先量化"收益来源是哪类资源"。

### 2.1.5 Split-K Decode GEMV

> **命名**：几何由两个数值宏描述——`GEMV_QUADS_PER_TG`（每 threadgroup 的 output quad 数）×
> `GEMV_SPLIT_K`（每 quad 切成几段 K）。旧名 `SPLIT_K_2` / `SPLIT_K_WIDE` 描述的其实是
> threadgroup 宽度，却读起来像 K 的深度，已废弃。

- **机制**：把每行的 quant block 对半拆给 2 个 simdgroup，各算半段部分和，经 threadgroup memory 合并。保留原有 pre-scaling 内循环不动。
- **为什么赚**：**行内在途读加倍**——直接对着 §2.1.4 的"小 GEMV latency-bound"下药。收益本质是提高访存并发，不是省 barrier（免 barrier 的 shuffle 变体反而更差，已删除）。
- **门控**：只在 `area==1` 的**普通** decode GEMV 分支；要求 `oc%8==0` 且每行 block 数可整除切分数（不整除会让 barrier 前出现不对齐的早退，见陷阱 G）。融合 leader 走自己的管线（§2.1.6）。
- **块数窗口**：M4 世代起 split-K 只在 blockCount < 32 或 ≥ 96 时启用，[32, 96) 默认走 oc16Sg4K1（4 SG × 4 quads/TG）。Pro 档与 base M4 起（gen>=16）档在 Q4 block64 fp16 上改按下面的"网格对 TG 预算"规则选路；只有无预算的档位（A 系、pre-M4 base）仍按窗口。
- **设备分档（pre-gen16 世代，`Conv1x1GpuTier::belowGen16` = archName gen < 16 且窄于 Pro die（M1–M3 base 与同代 A 系），含 gen 0 的元数据地板以下机型）**：`oc%16==0` 且 blockCount ≤ 96 时否决 split-K、走 oc16Sg4K1——该世代 split-K 的 threadgroup 归约开销盖不过翻倍的在途读。**> 96 块保留 split-K**：那里孤立 op 读数仍支持 oc16Sg4K1，e2e 却反号——更多 TG 与前一 kernel 尾部的交叠更好，而孤立 op 循环里同一 kernel 背靠背、测不到这条。**大 block 数的路径开关以 e2e 相邻配对为判据，op 级只用于 kernel 内部旋钮。**
- **网格对 TG 预算（`tgBudget`：Pro 档 160、base M4 起（gen>=16）档 80，其余档 0 即不启用；`highBandwidth` = `tier >= TierPro`：所有代的 Pro / Max / Ultra，与 `MetalAttention` 的 `highBandwidthGpu` 同一谓词）**：Q4 block64 fp16、blockCount ≥ 32、`oc%16==0` 时，两条路都发 128 线程 TG——oc16Sg4K1 发 `oc16Sg4K1Threadgroups = oc/16` 个，split-K 发两倍、每个扫半段 K。按 `oc16Sg4K1Threadgroups` 对 `tgBudget` 分三段：① `2*oc16Sg4K1Threadgroups ≤ tgBudget`（split-K 加倍后的网格仍装得下）→ split-K；[32, 96) 内 split-K 网格达到 TG 预算一半（`4*oc16Sg4K1Threadgroups ≥ tgBudget`）用 lanes 4，不到一半用 `gemvLanesPerBlock` 的默认 lanes（网格小时 pass 少的默认 lanes 赢，网格大时 lanes 4 赢），≥ 96 沿用 `gemvLanesPerBlock`；② `oc16Sg4K1Threadgroups > oc16Sg4K1GridLimit`（oc16Sg4K1 自己的网格已远超预算）→ split-K；`oc16Sg4K1GridLimit` 在 base 档是 `tgBudget`、Pro 档是 `2*tgBudget`，**这个倍数是按档标定的参数**。lanes：Pro 在这段用 `gemvSplitKFillLanes`（按 `blocksOwned = blockCount/2` 在 {2,4} 里选空槽最少的那个），base 档用 `gemvLanesPerBlock` 的默认 lanes。词表级 lm_head 是这段的极端情况（网格几千个 TG），预算规则覆盖全部块数，不按形状特判；③ 其余走 oc16Sg4K1。**`tgBudget` 是按档标定的参数，不是读出的硬件常驻 TG 数**。中段 oc16Sg4K1 的优势一部分来自层间依赖——串联（chain）测法与同形状独立 kernel 测法会给出相反符号，**所以这一段的标定要用 chain 测法**。Pro 侧的原理：split-K 把每行串行的 quant block 链减半，只有在这条链（而不是全部 oc 行的 DRAM 排空）限速时才赚；两者都正比于 blockCount，所以分界只随 oc 移动、不随 blockCount 移动。每 TG 几个 ocQuad（q1/q2/q4/q8，即 TG 数）不是杠杆。pre-M4 的 Pro / Max / Ultra 不受上一条否决，按此条走。**这类路径开关以 e2e 相邻配对为判据**：op 级赢、e2e 不兑现的形状真实存在（lanes 选择与大词表 lm_head 段都出现过），别只据 op 读数改默认。
- **带宽档与档位表（`Conv1x1DieClass` = Base / Pro / Max / Ultra / Unknown，`MetalConv1x1Decode.hpp`）**：每个带宽档在档位表里独占一行（`pro160` / `max160` / `ultra160` / `base16` / `base17` / `nobudget32` / `nobudget4`）。**Max / Ultra 目前照抄 Pro 的数字**——这两档还没有在本路径上测过；行独立出来是为了将来某档被实测后可以单独改，不牵动 Pro 行。`highBandwidth`（Pro / Max / Ultra）为真时走上面那条预算规则。
- **适用边界**：**融合管线上不要扩展 split-K**——融合 kernel 的寄存器/占用余量吃不起翻倍的 simdgroup + barrier，实测为负。融合管线的正确解是双流（§2.1.6）。
- **陷阱**：K 切分改变 weight 读取模式，**量化块边界必须与切分对齐**，否则跳块/错块；若该 kernel 后紧跟 GPU→CPU 同步，kernel 加速不兑现为 e2e。
- **g16 lm_head 的路由与已退役变体（别重试）**：基线 g16 = 2 SG/TG、threadgroup size 64、每 TG 覆盖 16 OC（2 SG × 8 OC/SG）。试过并退役两个：**4SG**（grid 减半，e2e 中性且抖动显著变大）与 **G16_OC4**（每 SG 4 个 oc_4 行，kernel 级读数变好但 e2e 中性）——都是"kernel 读数好看、e2e 不兑现"。现行路由（`g16LmHeadRoute`）**不按设备分档**：`oc > 16384` 且 oc_4 偶即走 g16，唯一例外是 decode 且 `oc%8==0`、blockCount 偶数那一格——落到 2sg 路由（`lmheadSplitKMode`）按 §2.1.5 的预算规则选 oc16Sg4K1 / oc8Sg4K2。g16 自身的 split-K 只在**宽词表 + 短 K**一格成立过，统一规则落地后已移除，别按旧文档复活它。

### 2.1.6 双流融合 GEMV（`oc16Sg2K1`，宏 `GEMV_SG_OC_QUADS_2`）

- **机制**：仅用于 gate_up / QKV / LN 融合 leader。每 simdgroup 同时处理 2 个相邻 output slice，两条 raw_dot 累加流**共享同一次 input 读**（LN 前导也共享），**无 barrier、无额外 simdgroup**，grid.x 减半。第二行越界时别名到第一行（安全读，结果丢弃），投影尾部由 per-row guard 处理。
- **gateup 臂上叠的 K 拆分（`GEMV_SG_OC_QUADS_2_KSPLIT`，base M4 起档取整除块数的 2–16，其余档取 2）**：只作用于 `GATE_UP_SILU` 双流体及其 LN 折叠变体，是融合管线里唯一成立的 K 拆分——gateup 的 oc 体量最小、并行度最饿，加宽 TG 的在途读收益盖过占用损失，且 LN 前导仍被全部 simdgroup 共享。host 值链 `mGateUpSiluKSplit`（常量）→ 宏，在 `setupGateUpFusion` / `setupLNFusion` 两处进 `preprocessorMacros` 与 cache key。base M4 起（gen>=16）档与无 tensor API 的 A 系 M4 起代（`gateUpFactorSkRoute`：fp16、Q4/Q8、`blockCount*64 == ic`、`oc%8==0`）先按下面的选档法则取因子，取不到再落到 pow2 兜底；兜底的生效门槛 `mGateUpSilu && outputDepthQuad%2==0 && blockCount 为 2 的幂且 ≥ 下限`（下限按档取 4，pre-M4 档与 Pro 及更宽档取 32——小块数上 split-K 的归约开销盖不过在途读翻倍）——偶数 quad 数保证 grid 精确、无 SG 在归约 barrier 前早退，2 的幂保证每对的块区间与 lanes-per-block 都精确。shader 侧 5 处消费（全在双流体分支内）：① `tg_simds = 2*factor`（4 → 8 SG / 256 线程）；② `uz = tg_x*2 + (sgitg&1)` 选两个 quad 中的哪一个、`ds_sk = sgitg>>1` 选该对扫 K 的第几段；③ `lanesPerBlock` 加宽补偿——每对只管 `blockCount/factor` 个块，不加宽则多出的 lane 组空转；④ 块区间 `[ds_sk*owned, +owned)`；⑤ 收尾各对 partial 写 `threadgroup ds_partial[2][2*factor]`，一次 barrier 后由 `ds_sk==0` 那对的 lane 0 按流加总并写输出（SiLU epilogue 不变）。**选档法则（`gateUpSiluSplitK`，只在 `gateUpFactorSkRoute` 成立的档位生效）**：分两步。① **可行带**——一对 SG 只管 `owned = blockCount/factor` 个块，shader 把 lanes-per-block 加宽到 `32/owned`；可行要求 (a) **单遍**：`owned ≤ 32/legacyLanes`（`legacyLanes = quadsPerBlock/4`），否则 lane 组要跑多遍、末遍有 lane 组空转；(b) **每 lane 每块至少 2 个 quad**：`owned ≥ 64/quadsPerBlock`，否则每个 lane 只读一个 quad、per-block 的 scale/bias 与归约摊不开。block64（每块 16 quad）即 `owned ∈ [4, 8]`。违反 (b) 在各档上一律更差；违反 (a) 在小块数上一律更差，大块数（≥56 块）例外（见下条的适用边界）。② **带内**取不超过 SG 对上限 `kMaxPairs` 的**最大**因子（每 lane 串行段最短）；带内因子全超上限时取**最小**的；带内无因子返回 0，落到 pow2 兜底。**最优因子不随 oc（TG 数）移动**——填不填得满机器只取决于 TG 数，而最优因子与它无关，"取总在途线程足以填满机器的最小因子"不成立。

**现值**（`mGateUpSiluKSplit`）：由 `gateUpSiluSplitK` 按块数现算，SG 对上限 `kMaxPairs = 6`，base M4 起（gen>=16）档与无 tensor API 的 A 系 M4 起代共用。**具体读数不写进仓库**——重测方法见 op-bench-and-diagnosis.md，重测时只需重标对上限。
**这条规则含一个标定参数**：SG 对上限（相邻因子的交叉点），没有被独立的硬件量解释，换 die 必须重标；当前各档统一取 6，不按 die 分档（分档的维护成本大于收益）。**op 与 iOS e2e 反号时，先做 mmap 开/关对照**：iOS App 默认 `use_mmap=true`，权重走 `MetalRuntime::onCreate` 里的 mmap 静态分配器，它曾只有 32B 对齐（普通分配器是 64B），这种对齐下大因子的 gate/up kernel 在 e2e 里变慢、小因子不受影响，而 op 单测不走 mmap ⇒ 两个尺度反号。该分配器的对齐已改为 64B。**适用边界**：大块数（ic ≥ 3584）上 op 级可能偏好可行带之外更窄的 TG（甚至跑两遍），机制未证实，规则在这些格上有 regret；因子不要求是 2 的幂（`tg_simds = 2*factor`、`ds_sk = sgitg>>1` 对任意 factor 成立），只要求 `blockCount % factor == 0` 且 `64*factor ≤ 1024` 线程；host 在 pipeline 的 `maxTotalThreadsPerThreadgroup` 不够时撤销融合。

**⚠️ 因子按 die 分档，跨档外推会反号（这是 `gateUpFactorSkRoute` 设备门的依据，别为"全机器统一"拆掉它）**：

- **base M4 起（gen>=16）档**：走上面的两步法则。16/32/40 块取因子 4/4/5；非 2 的幂块数与 64 块取带内因子（如 56 块取 7、64 块取 8）。
- **Pro die 及更宽（所有代的 Pro / Max / Ultra 按同一规则）**：因子 ≥ 4 在所有形状上都亏、越大越亏。**32 块以下不拆胜 SK=2**（小块数 gate/up），所以 Pro 档与 pre-M4 档共用下限 32；32 块 SK=2 与不拆持平，保留 pow2 兜底给的 SK=2；40 块非 pow2 不拆。已知边界：16 块且 oc 很小（gate/up 的 TG 数约 256 以下）时 op 级 SK=2 反而小胜，现有模型不落这一格，未加形状特判。**op 级与 e2e 在这一档会反号**，Pro 档改因子以 e2e 为准。
- **A 系**：只有 16 块那一格验过，SK=4 胜 SK=2。**A15 是负例**：gate/up 只编得出 SK=2、编不出 4，`gen >= 16` 地板因此生效。
- **pre-M4 档**：同样偏好不拆（见 §2.1.5 的设备分档条）。
- **"核多 ⇒ 因子要高"不成立**：核更多的 Pro die 反而要更低因子；"带宽余量"也解释不了——pre-M4 档带宽最低、小块数上同样不拆稳赢。自变量未定；现有证据只支持"base M4 起档偏好高因子，pre-M4 与 Pro die 都偏好 ≤2"。
- **在用的工作机上重测的注意事项**：背景负载会造成跨轮漂移 ⇒ 臂间轮转 + 同轮配对 + 空臂定噪声底；大模型连跑两轮就降频，**降频探针用 prefill 而不是 thermalState**（后者可以全程为 0）。**op 级要测依赖链**（FusedLinear → down_proj 串成一条，接近 decode 实况）而不是独立层——独立层会把 Pro 档小块数上不拆的优势压到噪声里。

- **为什么赚**：**加行内 ILP 而不加 simdgroup**——占用不变，同时把前导（input / LN）读量减半。
- **⚠️ 收益来自"共享前导"，不是来自"双行"**：plain（未融合）GEMV 路径里没有可共享的前导，权重读就是全部流量，而总在途字节数在重分组下不变（§2.1.4）⇒ **这个手段不要搬到 plain 路径**，实测为负。
- **为什么它赢而 split-K 在融合管线上输**：融合 kernel 吃不起翻倍的 simdgroup + barrier（占用下降）；双流在同一线程内加 ILP，占用不变。
  - **别过度推广这条**：它否掉的是"为拆分主体计算而翻倍 SG"的 barrier；**不增加 SG、只做一次 TG-uniform 标量归约的 barrier 仍然是赚的**，见 §2.1.9。
- **门控**：三处 setup（`setupGateUpFusion` / `setupQKVFusion` / `setupLNFusion`）解析同一个公式，保证 pipeline 宏与 grid 一致；设备门控（非 tensor-API 设备）即唯一逻辑。
- ⚠️ **名字的不准确处**：在 `GATE_UP_SILU` 臂上，第二条流是同一个 quad 的 up 矩阵，每 SG 只有 1 个 oc quad × 2 个矩阵。shader 几何块里已就地注明。
- ⚠️ **QKV 臂的双流体被 wide 几何一票否决**：`useOc16Sg2K1` 谓词是 `!useOc16Sg8K2 && !isSupportTensorApi()`，`useOc16Sg8K2` 为真时直接短路。`useOc16Sg8K2` 有两个入口：`qkvAllowOc16Sg8K2`（base M4（gen==16）+ fp16 + Q4/Q8，不按形状列举）与 `proAllowOc16Sg8K2`（Pro 档 + fp16 + Q4/Q8，由 `qkvOc16Sg8K2Admissible` 准入）⇒ base M4 上 Q4/Q8 一律 oc16Sg8K2，Pro 档只在准入时 oc16Sg8K2，其余非 tensor-API 形状走双流体。另外 `useOc8Sg4K2` 在非 tensor-API 设备上恒不可达（`!useOc16Sg8K2` 即 `useOc16Sg2K1`，而 `useOc8Sg4K2` 要求 `!useOc16Sg2K1`），所以 base M5 起（gen>=17）档的 plain 拆分只在 tensor-API 设备上成立。**任何"某路径应该出现却没出现"的结论，先确认谓词链没把它短路。**
- **QKV split-K 的 lanes 与 Pro 准入按 lane 组槽占用推，不按形状列举**（`gemvLanesPerBlock(splitK=true)` / `qkvOc16Sg8K2Admissible`）：每个 SG 拥有 `blockCount/2` 个块，32 lane 按 lanes-per-block 切成若干 lane 组、一轮在途若干块。lanes 下限默认 2；若 2-lane 时第二轮不超过半满，就降到 1 让一轮扫完（多出的那轮几乎全是空槽，比单块少一半 lane 更亏）。plain（base M5 起档）与 oc16Sg8K2 共用这一个函数，不再另设单轮版本。Pro 准入看整段 K 扫描的槽利用率 `owned / (inflight × passes)`：利用率不够时空槽的成本盖过 split-K 的收益，双流体（不拆 K、无归约 barrier）更好；短 K 规则另行保留。base M4 上 oc16Sg8K2 在全部标定宽度上都胜双流体，不设利用率门。**有无前置 LN 不是判别量**（无 LN 的首层同样 oc16Sg8K2 胜），`setupQKVFusion` 不再带 `hasLn` 参数。M5 Pro 与 pre-M4 的 Pro / Max / Ultra 按同一规则外推，未实测。
- **QKV plain 拆分（`useOc8Sg4K2`，GEMV_SPLIT_K=2、2 quads/TG × 4 SG）的上限标定在已退役的 uint4 宽加载权重体上**：宽加载已把在途读做够、barrier 变纯开销，所以小块数上拆分为亏。窄加载（ushort4）单次 load 的在途字节只有一半，这条结论**未必平移**——`qkvSplitKMaxBlocks` 现值 16（`MetalEnv.hpp`），要改先在当前体上重测。
- **LN staging 的路由与存储精度**：`MetalConvolution1x1::setupLNFusion`（`MetalConvolution1x1.mm`）只允许 QKV oc16Sg8K2 split-K（`useOc16Sg8K2`）使用 staging，且 auto 仅开在 hidden ≤ 1024。两个预算分开：`lnStageFits` 是 threadgroup 内存上限（staged 向量是 `ftype4`，`*16` 取 float4 最坏情况，≤ 8192 B），`lnStageWorth` 更窄（≤ 4096 B），因为 hidden 一大 staging 就不再是稳定的赢。**hidden=2048 上符号随形状变、驱动因素未定**（同一档设备上有形状赢、有形状输，幅度都不大）⇒ 宁可放弃可能的小赢也不吃可能的亏，所以 staging 留在它原本调好的 hidden ≤ 1024。注意成本/收益比**不是**显然的形状函数：packed grid 下每个 TG 恰好产出 `quadsPerTG` 个 output quad，所以 staging 恒花 `lnStageQuads/quadsPerTG` 次写/输出 quad，与融合成员多不平衡无关。双流体在两个输出流间共享输入读取，不能仅按输出流数推断 staging 收益；QKV plain 拆分不进入 staging。`ConvSimdGroupShader.hpp` 的 `ln_staged` 保存 `(in+residual)*gamma`，类型为 `ftype4`（fp16 存储时为 `half4`）；RMS 平方和归约与 GEMV 累加仍为 fp32，读 staging 后才乘 `inv_rms`，因此不能宣称与未融合 AddRMSNorm 的 fp16 输出逐位相同。
- **改宏名 / 动这类几何时的验证方式**（因为它同时动 6 处 cache key 与宏字典，单测通过不算证据）：① `git diff -U0` 过滤掉纯改名 token 后**没有剩余 `+` 行**，证明是纯令牌替换；② key/`setValue` 成对计数平衡，排除"key 在宏没了"；③ **下毒正向证明宏进了 shader**（故意改坏一个只在该变体里用到的量，e2e 必须乱码，撤掉后恢复）；④ op 单测覆盖不到 gate/up 的双流体，**那条只有 e2e 能覆盖**。

### 2.1.7 ⚠️ 短序列 GEMV 路径（area 2..16）从未优化

单 token 路径的微调候选已基本穷尽，剩下的数量级杠杆是"一次前向多算几个 token"（权重只读一次摊薄到 B 个 token，投机解码正走这条路径）。

- **现状**：`area ∈ [2,16]` 走 `conv1x1_gemv_g4mN`，拿到的 simdgroup 并行度只有 decode 路径的一半（每 TG 1 个 simdgroup vs 4 个），**且完全没有融合**——`mDecodeFusionEligible` 与融合注册只在 `area == 1` 分支设置，所以 B≥2 时每层是若干个独立投影 dispatch 而非几个融合 dispatch。
- **为什么慢得不该**：decode GEMV 是权重带宽 bound，理想情况下 `cost(B) ≈ cost(1)`；实测 B=2 就显著变贵，而且 B=2 反常地比 B=4 更差。与 §2.0 画像自洽：**把 simdgroup 数砍半 ⇒ 时间近乎翻倍**，是并行度问题不是算法问题。
- **可动项**：把 split-K / 多 SG 并行度以及融合移植到 `conv1x1_gemv_g4mN`。
- **前置依赖已就绪**：`transformers/llm/engine/src/speculative_decoding/` 已有 lookahead / ngram / tokentree / eagle / mtp，n-gram lookahead 不需要 draft 模型。实际收益 = 摊薄曲线 × 接受率，立项时需实测接受率。

### 2.1.8 W2/W3 decode 对齐 W4 优化栈

- **背景**：低 bit 支持落地后 decode 实际是坏的（陷阱 A 回归让 W2/3 pipeline 静默编译失败），且 2sg / split-K / 融合 / g16 / SharedGather 全部只支持 4/8bit。
- **实施要点**：
  1. g8 body 阶梯重排 W2/W3→W4→W8（修复 alias 遮蔽）+ g4mN 分支门控（非 sgMatrix 设备防越界）；
  2. g8 的 W2/W3 deferred + pre-scaling：**W2** 每 z 读 uchar4，mask `0xC0/0x30/0x0C/0x03` 配 in×(1/64,1/16,1/4,1)，`adj = dbias - 2*scale`；**W3** lo 面同 W2 + hi 面（tile bytes 4..5，nibble bit3=IC0..bit0=IC3）mask `0x8/0x4/0x2/0x1` 配 in×(1/2,1,2,4)，`adj = dbias - 4*scale`；**pre-scale 全为 2 的幂 ⇒ fp 位精确**；
  3. 2sg kernel 阶梯扩展（signature + 各 body 变体）+ dispatcher 解禁 + **三个 fusion setup 的 keys/dic 补新 bit**（漏改是静默退融合，靠 profile subtag 检测）；
  4. g16 lm_head 双行分支（注意 W3 的 row stride）；
  5. SharedGather（tied lm_head 的 GatherV2 clone）分支——**onClone 门控与宏链必须同时落**，否则低 bit 静默编成 W8。
- **W3 比 W4 慢的机制**：W3 tile 6B 非对齐，内环是多次标量 load + hi 面额外 mask，而 W4 是一次宽 load + mask。改成少量宽 load 的变体在小模型上中性（小 GEMV latency-bound，指令微调不兑现，§2.1.4），结构性差距要在更大模型上才值得再评。
- **遗留**：`conv1x1_w_dequant` 的 W2 分支散字节读未优化 ⇒ W2 prefill 偏慢。

### 2.1.9 LN 前导跨 simdgroup 拆分

- **机制**：融合 leader 的 LN 前导里 `inv_rms` 是 TG-uniform 标量，原来每个 simdgroup 各自把 input+residual **全量**扫一遍重算它。改成各 SG 分扫一段（`z = sgitg*32 + tiisg; z += 32*tg_simds`），各自 `simd_sum` 后经 `threadgroup float ln_sq_partial[tg_simds]` + 一次 barrier 合并。
- **为什么赚**：前导读量除以 SG 数，代价是一次 barrier。**通用判据**：kernel 里凡有可证明 TG-uniform 的标量（rms / mean / softmax 的 max & sum），其前导读量都应除以 SG 数。这类冗余在 timeline 上不显示为独立开销，只让 kernel"整体略慢"，**只有按"哪些量跨 simdgroup 恒等"重扫一遍代码才会暴露**。
- **与 §2.1.6 结论的边界**（重要，否则会被旧结论直接否掉）：ROW_2 那条"融合 kernel 吃不起 barrier"针对的是**为拆分主体计算而翻倍 simdgroup**的 barrier（占用下降）；本项在既有 SG 数下用一次 barrier 换前导读量减半，**SG 数与占用均不变**。判据是"这个 barrier 是否附带 occupancy 代价"。
- **主要工作量与唯一 UB 风险点是早退改造**，见陷阱 G。
- **划分一改，"唯一写者"守卫必须重审**：`ln_residual_out` 原来由 `sgitg == 0` 独写——这个条件本是"每个 SG 都扫全量"那份冗余的副产品；拆分后它的含义会从"避免重复写"悄悄变成"只写一半"。此类错误静默（下游拿到半截 residual）、与 barrier 无关、review 极易滑过。**规矩**：改并行划分时 grep 该 kernel 全部 `sgitg ==` / `tiisg ==` / `gid.* == 0` 守卫，逐个确认语义是否被新划分掀翻。
- **隐式契约**：初版把"恰好 2 个 simdgroup"写死在步长与 partial 数组里，依赖 setup 恒发固定线程数；split-K / wide 几何进来后必须改由 `tg_simds` 驱动。shader 注释只防得住读代码的人，防不住改 dispatch 的人——**隐式契约要么消掉（SG 数当宏传入），要么在契约两侧都留可执行检查**。

### 2.1.10 IndexedLinear route GEMV

- **机制**：IndexedLinear 的单 token/多路由形态要服务动态专家，不能按普通 conv 的固定 weight 指令流走。把每个专家的 W4/W8 在 host 重排成 `[expert][stream][outputQuad][inputQuad][tile]`，shader 侧沿用 2-simdgroup GEMV 的宽 tile 加载和延迟反量化；gate/up 是两条共享 input 的 stream。
- **为什么赚**：索引专家路径同样受惠于"在途读 + 宽 load + 一次性反量化"；保留按路由的单个 dispatch 是为了守住小路由形态的固定开销。**该 packed buffer 就是唯一 resident 权重布局**，grouped prefill 由 128 线程按“固定输出行 × 4 连续输入”直接读 row bytes，不与 raw bank 并存；同一线程的 4 个输入落在同一个 quant block，因此只查一次 scale/offset，不能按元素重复查元数据。
- **门控**：只有 W4/W8 且 IC/result quad/quant block 都对齐 4 才走 packed；其余形状回退 scalar。因此**新增 packed kernel 时必须配对齐且 fused/nonfused 的可点亮正确性 case**，否则默认单测可能整段根本到不了新路径。
- **陷阱**：host 把原始 nibble 作为 `q+8` 写入 packed tile 时，shader 的 dequant offset 必须是 `scale.y - 8*scale`，不能沿用 W8 直接加 offset。

## 2.2 GEMM（prefill）

### 2.2.1 Fused Q4/Q8 GEMM（in-kernel 解包）+ M64 tile

- **机制**：tensor-API 设备上 prefill 量化 conv 在 GEMM kernel 内解包反量化；M64 tile 再增大每 TG 的 M 维。
- **为什么赚**：① 省掉 dequant 预处理 dispatch 与 `mTempWeight` 分配（约 4× 权重体积的带宽往返）；② 更大的 M tile 减少跨 TG 的权重读冗余。
- **适用边界**：**小模型 + 低 bit 组合上可能翻负**，因此对权重规模设了下限（低于阈值的 conv 路由回 outer-dequant）。
- **M tile 加大被双重寄存器预算锁死，缩 N 救不回来**：destination cooperative tensor 摊到每线程 `M*N/threads` 个 float（超过约 32 float/thread 即溢出），但**输入 staging 的活跃行寄存器随 M 同步增长**（每线程行数 = M/16，M64 持 4 行、M128 持 8 行）。所以即便缩 N 把累加器压回 32 float/thread（M128×N32），多出的活跃输入寄存器仍把核推到同一条溢出悬崖——M128×N64、M128×N32、§2.2.5 的 N128×M64 全部翻负。判据：若变慢比值**跨短/中/长序列恒定**，则是每 TG 固有的寄存器/MMA 效率悬崖，**不是 DRAM 冗余流量**（后者随 grid.x∝area 变化、必有交叉点）。据此"缩 N 加 M 减权重冗余"在这族 fused-quant 核上已被证伪，别再试 M tile 增长。附流量模型：总重读 ≈ `(oc/N)·InputBytes + (area/M)·WeightBytes`，输入 fp16(2B/elem) 比 Q4 权重(0.5B/elem) 重 4×，缩 N 会放大更贵的输入重读——即使没有寄存器悬崖也是净亏。
- **短序列差距先归属再选手段**：短 prefill 的 GEMM 常是 latency-bound（达成带宽随 TG 数单调升）。用消融删 MMA 读"加载+staging 地板"：**地板≈对手总时间 ⇒ 差距全在 MMA 重叠效率**（tensor-op `mmOps.run` 同步退休，MMA 串行暴露在访存之上），不在权重冗余。此时加深 B 预取、预取 A、单缓冲提占用都会因寄存器压力/barrier 净亏而翻负——别再去省指令或加预取，也**别靠缩 N 加 M 减冗余流量**（见上条，已证伪）；唯一剩下的结构性手段是改 MMA 与 load 的重叠本身（让 tensor-op 不同步退休、把 MMA 藏到访存之下）。
- **位宽覆盖**：M64 tile 的门控是 `bits ∈ {4, 8} && area >= 64`（tensor-API 设备）。判据与位宽无关的原因是 **staging 存的是反量化后的 ftype**，`SMEM_OPERAND` 只随 ftype 与 tile 常量变化——W8 不会因权重字节翻倍而撑爆 threadgroup memory 预算。W2/W3 不在内：它们的解包是逐 nibble 标量，且 prefill 本来就走 outer-dequant。
- **epilogue 仍按位宽分档**：gate/up 双累加器（`FQ4_GATEUP_DUAL`）与 MUL_SILU 折叠（`FQ4_SILU_MUL`）**只对 Q4 开**，因为它们的面积阈值是按 Q4 形状标定的；W8 只拿 M64 tile 与双缓冲 staging，gate 张量照旧落地并由 owner 单独 dispatch。双缓冲宏因此改名为 `FQ_DOUBLE_BUF`（它不再只服务 Q4），而 dual/silu 两个宏保留 `FQ4_` 前缀以如实反映其位宽范围。
- **回退**：`OUTER_DEQUANT_GEMM_TENSORAPI` 开关回到 outer-dequant + fp GEMM；非 tensor-API 设备该开关是 no-op。

### 2.2.2 M64 sg_matrix GEMM 与设备分档

- **机制**：M64 的计算主体依赖 tensor ops，移植到只有 simdgroup matrix 的设备等于重写一个 kernel（寄存器 ×2、threadgroup mem 翻倍、全新 index math）。
- **结论与处置**：收益远低于预期 ⇒ **prefill GEMM 早已接近算力峰值，权重重复读不是它的瓶颈**，所以"减少权重重读"类手段在这里天花板很低。
- **设备分档（当前）**：gate 就是 `supportsFamily(Apple7)`，**M-chip 与 A 系列一视同仁**，不再按 device.name 的产品线前缀收窄。tile 的真实前置条件只有 simdgroup_matrix + 16 KB threadgroup memory（4 simdgroup / 128 线程），Apple7 全体都满足 ⇒ 按**能力位**分档才是对的自变量。历史：先是 archName `gen >= 16`（把 M1–M3 也误关在外），再是 device.name `"Apple M"` 前缀（把整个 A 系列误关在外），两版都是把"没测过"写成了"不适用"。
- **可复用的判据**：
  - **动手前先估这个 op 距离它自己的硬件上限还有多远**。已接近算力峰值的 kernel，任何"省访存"类手段的上限都被算力锁住。
  - **按产品线名收窄的门控不是实测结论，只是"没测过"**。这种 gate 读起来像"另一条产品线实测变慢"，会长期埋没收益且没人怀疑。正确顺序是：先按能力位（family / simdgroup_matrix / threadgroup memory 预算）放宽到该 kernel 真正能跑的全集，再**只在实测反号的那一档**上用 archName 世代加地板。
  - **放宽必须逐代实测，同一产品线不同代会反号**——见 §2.1.6 的 gateup split-K：同一个 tile 类改动在老世代手机档上大赚、新世代上小赚，而同一条 decode 默认值改动在新世代手机档上赚、老世代上亏。两个方向都存在，所以"某一档测出来是赚的"永远不能推及整条产品线。

### 2.2.3 In-shader dequant 阈值改为面积相关

- **机制**：原来只按权重元素数决定走 in-shader dequant 还是 outer-dequant；长 prompt 下该判据失效，改为同时要求 `area` 小于阈值。
- **为什么赚**：in-shader dequant 省的是权重往返，**但它把解包成本乘进了 M 维**——area 越大，重复解包越贵。所以正确的自变量是 `(权重规模, area)` 两维，不是权重规模一维。
- **代价**：峰值内存略增（仅非 tensor-API 设备的 prefill 路径）。
- ⚠️ 这类启发式阈值**必须在多个设备档上验证**才能改默认（老档有过回退前科）。

### 2.2.4 tensor API 能力探测的坑

- MPP `matmul2d` 要求 M/N 至少一个 16 倍数、静态 K 16 倍数。**探测 kernel 的描述符写错会让 tensor API 整体禁用**，症状是新设备上性能没有任何变化（而不是报错）。
- 另有一类更隐蔽的：探测**通过**但运行时 pipeline 反复编译失败并回退，反而比不探测更慢。⇒ **能力探测必须验证"探测结果 == 运行时真的能编译并跑"**，不能只看探测返回 true。

### 2.2.5 MoE grouped GEMM：staging 是主成本，不是 MMA

- **判据**：每专家行数小（M ≈ routes/experts）的 grouped GEMM，kernel 时间大头在 staging——量化权重解包、dequant scale 取数、A 的间接 gather——MMA 本体只占小头。动手前先用消融阶梯把 B staging / A gather / MMA 逐类删掉读差值，别按"GEMM=算力 bound"的直觉选手段。
- **有效杠杆**（都作用于 staging 指令数与冗余流量，不改变总在途字节，§3.6 不变量仍然成立）：
  ① staging 加载向量化：A 沿 C4 连续轴按 `ftype4` gather（每 lane 一个 quad，替代逐元素标量读）；B 每个 packed tile 一次宽加载覆盖其全部行（如 Q4 的 `ushort4` = 4 行 × 4 nibble），替代逐行 2B 标量读。
  ② scale 按 K 窗口单次暂存 smem 复用：逐元素直读会把同一 (oc, block) 的 scale 冗余取数倍于唯一字节量；smem 暂存 + 一次 barrier 换掉全部冗余 device 读。
  ③ 加宽 K（如 32→64）：barrier 与 matmul2d 调用次数减半，destination capacity 不变（§2.3.4 判据）；smem 预算随之翻倍，先核对 threadgroup memory 上限。
- **陷阱**：编译期 smem 数组按运行时参数的上限分配（如 scale 子块数 = K 窗口/blockSize）时，dispatcher 必须对该参数加门控，否则小参数用例（极小 blockSize）越界写 smem，症状是 NaN 而非编译错误；不满足门控的形状回退旧路径，而不是缩小数组。
- **循环不变的间接 gather 元数据要手动 hoist 出 K 循环**：A 的行映射（`order[row]`→source row）与 K 窗口无关，但 Metal 编译器不对 device load 做 LICM，留在循环内会每窗口重发 `order[]` 依赖加载。提到循环外每线程只取一次，窗口多的短 K 形状小赚、长 K 中性，纯重构逐位等价。反向把 B 的 packed 加载 hoist 到 A barrier 前求延迟重叠则翻负——跨 barrier 持寄存器加剧压力（§2.2.1 寄存器悬崖），与加宽 K 跨占用边界同类，别试。
- **M tile 按 rows/expert 分档路由，不可全局加大**：M 加大把每专家权重读摊到更多 route 行（长序列下权重重复读是主成本），但 ceil-padding 的 MMA 浪费随 M 增大；rows/expert 小时 padding 项反超流量项，净收益为负。正确形态是 M16/M32/M64 编译期变体 + dispatcher 按 rows/expert 阈值路由，阈值用 op 单测在临界 T 上交替配对扫描定。tile 粒度常量同时活在 routing 预处理（tileOffsets 构建）与 grouped kernel（firstRow）里，变体宏必须同步这两处并进 pipeline 缓存 key；resize 改变 M 时作废受影响 kernel 的本地 pipeline 缓存。attention tc 的"加 M 溢出"判据不能外推到 GEMM：GEMM 只有 D 累加器，没有 q_reg/softmax 状态一起翻倍。
- **smem 相位复用**：K 循环结束后 staging tile 已死，epilogue 的 C tile 可别名同一块 threadgroup buffer（dense M64 GEMM 的做法），既省出大 M 的预算，也可能因占用下降直接提速；staging 与 epilogue 之间必须有 barrier。
- **A 直喂 device tensor（staging 地板的结构性手段）**：廉价杠杆（向量化/scale 暂存/加宽 K/hoist）用尽后，把 A 在路由预处理里 permute 成专家排序的连续 scratch（每 route 行一线程，`inputChannels%4==0`），grouped GEMM 用 `tensor<device ftype,…>` handle 的 `slice(firstInput, firstRow)` 直喂 matmul2d（§2.3.9 `FATC_KV_DEV_TENSOR` 机制），消掉 A 的 smem 往返与跨 N-tile 散读，`slice()` 自带尾行 zero-fill。已验证 `execution_simdgroups<4>` + device-A/threadgroup-B 混合地址空间可行；destination CT 类型用 `remove_addrspace_t<decltype(slice)>` 推。**形状门控**：短 K 非融合（down，staging-bound）赚，长 K 融合（gate_up，compute-bound）因 permute 开销 + 长 K device 重读收不回而翻负——按 `!fuseSwiGLU` 门控（down-like 的稳健代理），融合路径保留 smem staging。permute 在 routing 之后、grouped 之前 dispatch（serial encoder 隐式定序），scratch 绑 grouped buffer 0 不改签名。**连带杠杆**：handle 直喂后 A 的 smem 区已死，令 `A_BYTES=0` 缩 smem 可提占用（M32 上叠加收益），cTile 别名仍成立。
- **验证**：staging 改写属于归约顺序不变但加载布局全变的改动，正确性 oracle 必须覆盖 K 尾块、奇数输出通道 partial tile、融合/非融合、全部位宽与两种 inputMode；"全绿"前先确认新路径真的被用例命中（门控可能把用例全部推回旧路径）。M 变体还要覆盖 resize 序列（token 数变化触发 M 切换与缓存失效路径）。
- **N/K tile 先算占用与 MMA 效率再动手**：加宽的判据是 smem 预算跨过 32KB/2 占用边界或累加器寄存器超预算；缩 N 省 smem 提占用同样翻负——tensor-op MMA 效率随 N 走，且 TG 数与 barrier 总数翻倍，占用收益补不回来。N64 是双向局部最优，两个方向均已实测否决。
- **smem 字节数学必须 `sizeof(ftype)`**：staging 偏移若硬编码 `*2`（fp16 假设），fp32 路径会越界写 threadgroup buffer，症状是 fp32 单测大数值用例错而 fp16 全绿——两种精度都要跑才算验证完。
- **路由预处理融合 + 原子 RW epilogue**：Clear/Histogram/Prefix/Scatter 四 dispatch 可合成单 TG(1024) kernel（tg atomic 直方图 + tid0 串行 O(E) prefix + scatter），op 级中性但每层省 3 次 dispatch；down-proj 的 routeWeight 折叠可用 half2 CAS 原子加直接累进 C4 输出（fp16 无原生原子加），省 route-row scratch 与 reduce dispatch（op −5%、e2e pp512 +4%）。**RAW probs（routeWeight=2）的 topK 归一化 sum 必须按行 hoist 进 smem wTile**：逐 channel-pair 重算会把 op 收益吃光并造成 e2e 回退——op 测速用例必须镜像模型的 routeWeight 模式，只测 mode=1 会漏掉 mode=2 的尾段成本。原子路径的宏（RW_ATOMIC）改变 kernel 签名与 smem 布局，必须在 grouped pipeline 编译**之前**定夺，且编译失败回退时同步关掉原子标志重建。

## 2.3 Attention Kernel

### 2.3.1 Causal 三角 QK dispatch + 有界 softmax/AV（CAUSAL_TRI / CAUSAL_BOUND）

**prefill 上最大的单项收益来源。**

- **机制**：causal mask 下三角假设下，上三角区域在 mTempQK / mTempSoftMax 中**完全不写不读**。
  1. `CAUSAL_TRI`（prefill_qk）：host 只 dispatch 因果对角线下的梯形 tile，kernel 内二次方程反解线性 tile id → (slq, slk)；interior tile 跳过全部 per-element mask 读取/分支（三区域分解）。
  2. `CAUSAL_BOUND`（softmax_plane/_sg + prefill_qkv）：softmax 每行只归约/写出 causally-valid 前缀 + 少量零 pad（覆盖 prefill_qkv 的对齐 tile 读界）；prefill_qkv 的 av 上界截断同步激活。
- **为什么赚**：省掉 QK 写 + softmax 读 + softmax 写各 O(seq²/2) 的带宽，**收益随 seq 增长**。whole-tile early-exit 已达下三角理论上界，不需要额外的 block classifier。
- **门控是数据驱动的**：causal 判定（`AttnFacts::causal`）由"mask 输入是否为标量哨兵 / 无 mask + 有 KV cache"推出。真实 mask 张量 ⇒ 非 causal ⇒ 自动关掉 causal-tri/bound 与两条融合 prefill。**非 causal 模型不需要设任何 env**——原先的 env 开关是"忘了设就静默乱码"的正确性陷阱，已删除。这是个可复用的设计选择：**正确性相关的路径开关应当从数据推导，不该交给用户设 env。**
- decode 侧不存在等价优化——seq_q=1 时分数行全部因果有效，无三角可跳。

### 2.3.2 为什么 Apple Silicon 全档把 FlashAttention 降级到三段路径

- 优化后的三段路径（+causal-tri/bound）在无 tensor-coop 的设备上**反超 legacy FA，且差距随 seq 增长**——因为 causal-bound 省的是 O(seq²) 带宽，而 legacy FA kernel 没享受到这一层。
- **处置**：所有 simdgroup-matrix Apple GPU（M 全档 + A 系列）默认走三段 + causal-tri。
- **结局**：legacy FA 已**整体删除**（kernel 源码 + 路由 flag + `attention_mode/8` 门控）。它最后仅剩的两条理由都不成立了：① "超长上下文省 O(seq²) scratch" 已被三段路径的 q 序列分片（`MNN_METAL_ATTN_QSPLIT_MB`，见 §2.3.1）覆盖；② "head_dim 兜底" 由 FA-TC / FA-SG 与三段路径分担。删除前它在运行时已**不可达**：`eligible ⇒ causal ⇒ kvCache ⇒ kvLen ≥ seqLen ⇒ causal-bound 可用 ⇒ 立刻被否决`，只有 `preparePipelines` 的代表性 kv 扫描（kv < seqLen）会点亮它、白编译一个 pipeline。
- **可复用的判据**：
  1. **"融合 kernel 一定比多段快"是错的**。多段路径若能把 O(n²) 中间量的读写整段砍掉，可以赢过没做这层优化的融合 kernel。比较对象必须是"两边都做了同等算法优化"的版本。
  2. **被更优路径无条件否决的开关就是死代码**，而它的存在会让"路径判定"这套布尔互相取否、读不出真实优先级。发现某条路径的实际准入恒为假时，要么修门控让它真能生效，要么连 kernel 一起删——留着只会让后来人照着文档去排查一条不存在的路。

### 2.3.3 Fused Prefill Flash-Attention（legacy，**kernel 已删除**，机制与避坑留存）

> ⚠️ `gPrefillFlashAttn` / `prefill_flash_attn` 已随 §2.3.2 的结论整体删除，运行时无此路径。
> **本节只作机制留存**：它是 MNN 上第一个 online-softmax 融合 attention，下面这些坑
> （`ATTENTION_C4` 输出布局、在线 softmax 的 NaN 双短路、mixed-dtype MMA、PV 前的 barrier）
> 在 FA-TC / FA-SG 上同样成立，排查那两条路径时仍然适用。

- **机制**：融合 Q·K^T + online softmax + P·V 到一个 kernel，中间数据全留在 threadgroup memory 和寄存器；每 simdgroup 拥有若干行 Q 的 running max / sum 寄存器；每 KV 块做 QK → 在线 softmax → 同段 P 做 PV → 累加进 O。
- **为什么赚**：三段路径要把 O(seq²) 的 QK / softmax 结果写进 global 再读回；融合后这些流量归零。
- **threadgroup 布局**：`sq`（Q 分块，half）/ `sf`（QK 的 fp32 scratch）/ `ss`（归一化后的 P，half）/ `so`（O accumulator，float，在线 rescale）。
- **关键文件**：原 `MetalFlashAttnShader.hpp`（`gPrefillFlashAttn`，已删除）、`MetalAttention.mm/hpp`。
- **实施要点**：
  1. 门控要求 simd-matrix + fp16 + causal + head_dim / group_size 在支持集合内 + seq 下限。
  2. **在线 softmax 数值稳定性**：`M_new = simd_max(fmax(M[j], s))`；`ms`/`vs` 对 `-INFINITY` 的双短路是必需的，否则 `exp(-inf - -inf)` = NaN 会从初始态或全 masked 行传播。
  3. **KV int8**：大 head_dim 下不能整 tile 反量化到 threadgroup（爆 32KB），要按 k_step 分批 dequant；`k_scales`/`v_scales` 是 `device ftype*`（fp16）**不是** float，错声明必乱码；用 `threadgroup_barrier` 不是 `simdgroup_barrier`（写者与读者 lane 数不同）。
- **避坑要点**：
  1. **`ATTENTION_C4` 输出布局 —— 最重要的坑**。c4-head export 时 output 实际布局是 `[num_head*(head_dim/4), batch*seq_q, 4]`（NC4HW4-packed）。不区分则输出**从第一步就乱码**，且代码逻辑看着完全正确、地址全部合法：

     ```cpp
     #ifdef ATTENTION_C4
         int o_off = (h * (param.head_dim / 4) + (d / 4)) * 4 * param.batch * seq_q
                   + (b * seq_q + q_abs) * 4 + (d & 3);
     #else
         int o_off = ((b * seq_q + q_abs) * param.head_num + h) * param.head_dim + d;
     #endif
     ```
  2. **先怀疑数据布局，再怀疑 Metal API**。曾因怀疑 `simdgroup_load` 的 transpose flag 有 bug 而改用 threadgroup 中转 K/V——结果是正确但大幅变慢，真 bug 是上面那条布局。
  3. **mixed-dtype MMA 只有 all-half 或 all-float**：QK 输出先写 fp32 scratch，softmax 读 fp32、算完转 half 写另一块供 PV MMA——两块 scratch 不能合并。
  4. **softmax→PV 之间的 `threadgroup_barrier` 不可少**：各 SG 只 rescale 自己那几行 O，但 PV 读全部行。
  5. **正确性验证必须 greedy sampling** 对拍前几十个 token。
- **参数取向**：**减少 K read 冗余（增大 Q_TILE）是长 prompt 最有效杠杆**（grid.x 减半换 K 读减半）；增大 KV_TILE 反而更差；再往上加 Q_TILE 会逼近 threadgroup memory 上限。
- **已探明的边界（不建议盲改）**：多头融合会让 threadgroup mem 翻倍、occupancy 减半，净收益不确定；**去掉循环末 barrier 会显著变慢**（barrier 有带宽调度作用，全保留）；`so` 显式清零必须保留（threadgroup 初值可能是 NaN）；加倍 NSG 无收益。

### 2.3.4 tensor-API 版 FlashAttention（`prefill_flash_attn_tc`）

- **机制**：用 Metal4 cooperative tensor 重写融合 attention（§1.10 的布局表就是为它测的）：`matmul2d` + input cooperative tensor + **零 threadgroup memory**，S/O 全寄存器、score 不落全局。
- **门控**：`fusedPrefillOk`（causal && fp16 && 非量化 KV && KV 在内存 && `kvLen >= seqLen`）&& `isSupportTensorApi()` && head_dim 在支持集合 && seq 下限。输入形式不对应单独的设备能力标志。
- **宽 head 选路**：`gpu.gen <= 16 && head_dim == 256` 使用分阶段 Tensor Attention；gen>=17 与窄头（64/128）保持 Tensor FA。宽头在 gen<=16 上分阶段明显更快、且差距随 seq 拉大；窄头方向相反。**不要改成无条件按 gen 选路**，窄头会大幅回退。
- **宽头分歧的机制**：g16 上常驻 Q/O cooperative tensor 触发寄存器驻留悬崖，把 Q 移入 threadgroup memory 反而赚；g17 上常驻形态高效。是设备共性差异，不是 kernel 写法问题。每 tile 重载 Q 的形态重载流量 ∝ seq²、省寄存器是固定收益，两者在长 seq 交叉，印证同一机制。

#### 根因：tc 是 coop-tensor 打包/解包 bound，不是 MMA throughput bound

在 attention 占满被测时间的 op 单测上归因（方法见 §2.0）后定位：**瓶颈是每次 `matmul2d` 前后围绕 cooperative tensor 的标量寄存器搬运**，不是张量核算力。每个 kv tile 每 simdgroup 要发十几次 `matmul2d`，每次都要往 destination CT 填十几个元素再读回 ⇒ 几百次寄存器 move，对着同量级 cycle 的真实 MMA 工作，等于一倍量级的税。

⇒ **有效杠杆 = 减少每次 matmul2d 的标量流量 / 调用次数。** 已转默认开的四项，按机制排列：

| 变体 | 机制 |
|---|---|
| `FATC_Q_REG` | q tile 载入并预缩放一次后常驻寄存器，不再在每个 (kv tile, head_dim frag) 上重新 gather |
| `FATC_O_CT` | O 累加器常驻 destination cooperative tensor，省掉每次迭代与标量数组之间的往返 |
| `FATC_QK_K32` | QK 的 K 16→32，head_dim 累加调用数减半 ⇒ 打包次数减半 |
| `FATC_PV_K32` | PV 的 K 16→32，一次吃整个 kv tile，PV 调用数减半 |
| `FATC_KV_DEV_TENSOR` | K/V 以 device `tensor` handle 就地喂给 matmul2d，**彻底干掉每 lane 的 B-fill** |

**`FATC_KV_DEV_TENSOR` 是这一串里机制最干净的一项**，值得单独记：手填 `ct_b` 曾被当成不可约成本（每 lane 每 kv tile 的 B-fill = `BK * HEAD_DIM / 32` 次 half move，对 K 与 V 各一份，且对 `BK` 与 K 宽度都不变），实际上它是这个 kernel 最大的单项标量开销。`matmul2d::run()` 的操作数**既可以是 cooperative tensor，也可以是 `tensor` handle 的 slice**，改用 handle 后硬件自己发操作数 load，这些 move 直接归零：

```cpp
const array<int, 2> kv_strides = {1, param.batch * kv_heads * param.head_dim};
auto t_k = tensor<device ftype, dextents<int32_t, 2>, tensor_inline>(
    (device ftype*)(K + kv_head_off), dextents<int32_t, 2>(param.head_dim, seq_k), kv_strides);
// QK: auto ct_b = t_k.slice(dd * FATC_QK_K, kv0);
// PV: auto ct_b = t_v.slice(T_IDX * 32, kv0 + ik * FATC_PV_K);
```

就地读顺带省掉三件事：staging、barrier、以及为 barrier 而**放宽到 threadgroup 最大 q 行**的 kv 循环上界（每个 SG 得以保留自己更紧的 `kv_lim`）。`slice()` 自带 edge checking，把手写的尾行守卫也替掉了。

⇒ **tc 上任何"先搬进 smem 再喂"的变体都不要做**：付了 stage + barrier + 放宽的 kv 循环，抵不过省下的 B-fill（两种形态都实测为负）。

> **coop-tensor 加宽的通用判据（本节最有复用价值的一条）**：加宽 **K** 赚
> （destination capacity 不变，纯省调用次数），加宽 **N** 亏
> （destination capacity 翻倍 ⇒ 活跃寄存器翻倍，掉过 occupancy 悬崖）。
> **先看 destination capacity 会不会涨**，涨就别做——除非能在算下一半之前把上一半的
> softmax+PV 消费掉。同理，**加大 M（每 SG 多吃 q 行）会让 O 累加器 + q_reg + s_acc
> 一起翻倍而溢出**，这个方向不要重试。

**实施注记**：
- PV 的 B 操作数是 `transpose_b=false`（`[k][n]`），QK 是 `true`——**K=32 多出来的那一级落在元素索引哪一位，不能从 QK 类比推导**，必须单独 dump。
- 只有常驻-CT 的 PV 分支实现了宽 K 填充，故 `PV_K32` 依赖 `O_CT`，shader 里加 `#error` 守卫。
- 让 QK 累加也常驻单个 destination CT 是**中性**的——K32 之后编译器已自行合并了那处往返。
- **大 head_dim 的 O 累加器寄存器悬崖必须按 D 切，但切法决定 QK 是否冗余**：online softmax 要求 O 跨 kv tile 常驻，full head_dim 的 O（每 lane `HEAD_DIM/32×16` float，d256=128）单独就超预算（destination CT >~32 float/thread 即溢出）。**不切 O 的两条路都实测翻负**：Q_CT 常驻（O+Q 双份寄存器）数倍慢；流式 Q（腾 Q 寄存器、Q 每 kv tile 重读）长 seq 仍明显慢——O 单独撞悬崖，腾 Q 救不回。切 O 有两种：① grid.z 切到不同 TG 各持半 O，但每 slice 重算整段 QK（QK 跨 full head_dim），冗余随 S² 增长；② **simdgroup 级切（`FATC_WN`）**——同一 row group 的 WN 个 simdgroup 各算一段 D 的 partial QK，经 threadgroup memory 交换求和得完整 score，QK 只算一次，O 仍按 D 切片常驻。② 实测优于 ①（消冗余 QK，长 seq 尤甚），代价是每 kv tile 两次 TG-wide barrier + smem 交换 + 各 d_group 冗余 softmax；且 WN>1 必须用 **block 级 kv_lim**（整 64 行块的顶行）保证全 TG 走相同 kv tile 数，否则 barrier 死锁，低 row group 的多余 tile 由逐行 causal mask 清零。P 落 smem/global 共享不可行（P 行=16×seq_k，smem 放不下；global 的 S² 带宽比重算 QK 更贵）。
- **短 seq attention op 单测的时钟爬坡陷阱**：Apple GPU 轻负载不升频，短 seq（如 T512）kernel 太轻喂不 boost clock，**先跑的臂冷低频、后跑的臂被前任暖升频**，A/B 顺序直接决定符号（同核可差 3×）。短 seq 的 TFLOPS 随 seq 单调升很大一部分是升频假象，不是核效率。可靠 A/B 只在重负载长 seq（T2048，rounds≥25 收敛到稳态）做；短 seq 要么 interleaved 配对、要么先重负载暖机再测，别信冷却后第一跑。

### 2.3.5 单 pass 融合 SDPA decode

> ⚠️ **两种形态，按 kv 长度自动切**：短/中 kv 走单 pass 融合 SDPA——grid `(1, B*H/qh, 1)`，
> 一个 threadgroup 一个 q-head 组（`qh` = `SDPA_QH_PER_TG`，见 §2.3.5b），无 reduce、score
> 不落全局、kernel 直写输出。kv ≥ 2048 额外编出 2-pass 形态（`SPLITKV_PARTIAL` 宏变体 +
> `decode_splitkv_reduce` + partial buffer），见 §2.3.5d。env 切换开关已删，形态由路由决定。

- **机制**：一个 TG 负责一个 q head 组，QK + online softmax + AV 单 kernel 完成。
- **为什么赚**：score 不落任何内存、无第二段 AV dispatch、无跨 workgroup reduce。
- **门控**：`totalKv >= 阈值` && `mKVCache` && `mSeqLen==1` && `!mKvInDisk` && (causal || trivial mask) && `mHeadDim%32==0` && tg 内存在上限内。阈值取得极低（kv≥2），⇒ **实践上 decode 全部走这条路**；它在 `_resolvePlan` 的阶梯上排第一，命中即返回，下面的 fused flag 根本不会被算。
  - ⚠️ 因此**排查 decode attention 问题时不要按旧文档去看 QK_QSPLIT 那条路**——kv≥2 时活跃路径只有单趟 SDPA（无开关可对照）。
- **NSG 是 device-tiered 的**：tensor-API 档取较大常量；非 tensor-API 档按 `clamp(常数 / (batch*head_num/qh), lo, hi)`，短 kv 再 cap。
  - ⚠️ **分档必须排在 `qh` 解析之后**：它的输入量是 grid.y（整个 dispatch 的并行度），不是 head 数。**两个 auto 旋钮读同一个量时，谁先算谁就看不见对方——顺序即正确性**（这条错过一次，代价是一个模型档拿到窄了 qh 倍的 nsg）。
  - nsg 与 qh 之间**无交互**，可以各自标定。
  - ⚠️ nsg 改变跨 simdgroup 归约的求和顺序 ⇒ 输出**不与旧版逐字节一致**（另一个合法求和序），判据用带容差的 op test，不要用 byte-compare。
- **适用边界**：短/中 kv 强开为负，必须阈值门控。
- **已删除的变体（勿重试）**：
  - **QK 合并读**（simdgroup↔kv 行、长连续 K 读）：kernel 级快、e2e 平，**合并读不兑现**，同一模式已复现多次。
  - **kv token 批量取**（一次预取 KVB 个 kv token 的 K/V 行）：**作为单纯的 load 预发射在这个 kernel 里无效**——它的 V 行读**本来就已经 hoist 到 `simd_sum(score)` 之上**，单 token 的 load 延迟早已被覆盖，批量只多付寄存器。**注意这与 FA-SG 上同类手段有效并不矛盾**（那里 smem fragment load 与 MMA 是 1:1 且阻塞 MMA，批量才拉开 ILP）。⇒ **"批量预发射"只在被阻塞的依赖链上有效，先确认那条 load 是否已经被 hoist。** 但同一个"多 token"外壳换一个依赖目标就成立了——见 §2.3.5c（共享 online-softmax 更新）。
  - **2-pass split-KV 加 TG 数**（kv 轴切到 grid.x、partial + reduce 第二趟）：split 不改变总 KV 字节，却把**每个 TG 的固定尾款**（跨 simdgroup 转置归约的 barrier 串、q 寄存器装载、reduce dispatch 本身）乘上 TG 数——固定开销随 TG 数线性涨，而流式带宽不动。该 kernel 的低利用率是 token 循环内依赖链（QK 点积 → simd_sum → online-softmax 更新）的**延迟属性**，TG 数足够时加 TG 只会更差。所以现行形态只在 `kv >= 2048`（TG 数真正不够填满 GPU）时启用（§2.3.5d），**不要把加 TG 的逻辑外推到短 kv**。
- **split-KV 形态的坑**（对当前 2-pass 形态依然适用）：路径 flag 判定必须放在 `handleKVAllocMemory()` **之前**，否则首个 decode step 临时缓冲未分配 → `setTensor(null)` SIGSEGV（单 pass SDPA 的 early return 跳过 mTempQK/mTempSoftMax，fallback 时靠 re-run 补救）。partial buffer 若要按 kv 增长扩容，必须按**进程级固定值**（如 env 上限）一次分够：按当前 ntg 扩容会在 ntg 翻档时 free 掉 recorded replay 仍绑定的 Tensor*，是 `metalReplayValidate` 看不见的 UAF。

### 2.3.5b KV 跨 GQA 组共享（`SDPA_QH_PER_TG`）

- **机制**：`decode_splitkv` 原本一个 TG 只算一个 q head，于是 GQA group 内同一份 K/V 行被 group 个 TG 各读一遍。`SDPA_QH_PER_TG > 1` 让一个 TG 拥有若干个连续 q head（须整除 `group_size`），K 行进寄存器 `k_row[DPT]` 后对每个 head 各做一次点积，V 行同样只取一次。
- **为什么赚**：**削掉请求侧的冗余 KV 读**。GQA group 大的模型上，unique KV 流量远小于请求侧流量，达成带宽被压在 DRAM 峰值之下；分组后能把达成带宽推到接近峰值。
- **代价与真正的因变量**：`qh↑` ⇒ KV 读请求↓（收益）**且** threadgroup 数 `= batch*head_num/qh` ↓（代价）。所以净效果是**两个反向效应的乘积**，`group_size=2` 的模型上分组一次就把冗余削光、同时把 TG 数压到下限，两者刚好抵消 ⇒ 看起来"机制不成立"。**真正的因变量是"请求侧带宽还是不是瓶颈"**，TG 数和 group_size 都只是它的代理。
- **自动门控**（`MetalAttention.mm` `_resolveSdpa`）：从 qh=2 起逐次翻倍，取满足下面两条的最大 2 的幂（须整除 `group_size` 与 `head_num`）：
  1. `batch*head_num/qh >= 8` —— **硬下限**，低于此纯粹并行度饿死，没有任何带宽收益能救；
  2. 若落在 **8~15 TG** 这一档，还要求**剩余冗余 `group_size/qh >= 2`** —— 这一档要付并行度代价，只有"分完之后还有冗余可削"时才划得来。剩余冗余 ≥2 说明请求侧带宽仍是瓶颈；剩余 =1 说明 unique KV 的 DRAM 流量已经是瓶颈，再分组换不到东西。
- **已知的门控局限**：`head_dim` 会调制 TG 下限（head_dim 大 ⇒ 每 TG 单 token 工作量大 ⇒ 更少的 TG 也能填满）。真正的并行度单位是 `TG 数 × 每 TG 工作量`，当前用"TG 数 ≥8"当代理是**偏保守**的近似；想再榨这一档得把 head_dim 放进门控并重新标定。
- **短 ctx 与长 ctx 的最优 qh 不同**（短 ctx 偏好更大的 qh），按 kv 切 qh 是可行的（`_pathSignature` 已支持每 token 翻转变体），但收益不值这个复杂度，**未做**。
- **正确性**：分组不改变任何单个 q head 的归约顺序（每个 g 是独立累加器、kv 迭代序不变），只改"哪个 TG 算它" ⇒ 可以与不分组**逐位一致**。kernel 主体本就完整泛化于 `SDPA_QH_PER_TG`（`S/M/O[qh]`、`s_vs[NSG][qh][C]`、输出 `q_head_base+g`）。变体要进 `_pathSignature`（它同时改 kernel 与 grid）。

### 2.3.5c KV token 交叠（`SDPA_KVB`）

- **机制**：单 token 形态里每个 kv token 都要走完整依赖链：QK 点积 → `simd_sum` shuffle 串 → online-softmax 更新（`M/S/O` 逐 token 串行）。`SDPA_KVB = 2` 让一个迭代处理同一 stripe 步里的 2 个 token：两条点积/shuffle 链互相独立、在时间里交叠，且 2 个 score **共享一次 online-softmax 更新**（max/exp2 因子可代数合并，求和序仍是合法序）⇒ 每 token 的串行更新次数减半。收益来自**改依赖结构**，不是 load 预发射（那条路 §2.3.5 已证伪）。
- **为什么赚**：该 kernel 的低利用率是 token 循环依赖链的**延迟属性**（§2.3.5 split-KV 条目已定位）；TG 数少、occupancy 遮不住这条链时，交叠直接把链长减半。
- **自动门控**：`DecodeSdpa 路由 && totalKv >= kvbMinKv && tgCount < kvbMaxTg`（`tgCount = batch*head_num/qh`，与 nsg 分档读同一个量，**必须排在 qh 解析之后**）。TG 上限的依据：occupancy 足够遮住依赖链时，交叠只剩寄存器代价；kv < 2048 时循环太短、正负都在噪声内，不开。env `MNN_METAL_DECODE_SDPA_KVB`：0=auto（默认），1/2/4 强压。
- **KVB=4 稳定输给 KVB=2**：`k_row/v_row` 各多一份 DPT 寄存器，register pressure 吃掉交叠收益。别再扫更宽。
- **正确性**：合并更新对每个 q head 是独立累加器、kv 迭代序不变，与串行形在 fp16 输出上逐位一致或差 ≤1 ulp（合并 vs 顺序累加的结合序差）。变体随 kv 跨档每 token 翻转 ⇒ **必须进 `_pathSignature`**（bit 28），否则 recorded replay 拿到旧 kernel。

### 2.3.5d 2-pass split-KV decode（`SPLITKV_PARTIAL` + `decode_splitkv_reduce`）

- **机制**：把 kv 轴按 `SPLITKV_BLK = 512` 切块，grid `(blocks, B*H/qh, 1)`，一个 TG 一块，写出**未归一化**的 partial（每 head `head_dim + 2` 个 float：O 累加器 + running max + sum）；第二趟 `decode_splitkv_reduce` 用 log-sum-exp 合并各块。目的是在长 kv 上把 TG 数从 8 提到 blocks×8，填满 GPU。
- **门控**：`kv >= 2048` 且 partial/reduce 两个 pipeline 都编得出来（partial 还要过 `32*nsg` 线程上限与 threadgroup 内存上限）。再短的话单趟 grid 已经够了。
- **实现要点**：
  - partial buffer **按字节分配**（`Tensor::createDevice<uint8_t>`）：声明成 float 会被 fp16 后端砍半，而 shader 每元素写 4 字节 ⇒ 越界。
  - block 数跟**执行期的真实 kv 长度**走（`UP_DIV(kvLen, 512)`），不是 prepare 期的代表长度。
  - ⚠️ **它是 plan 属性，不是路由键**：形态由"pipeline 编不编得出来"决定，而 `pipelineSignature()` 必须在编译**之前**算完才能查表 ⇒ two-pass 位若进 signature，插入侧与查表侧会算出不同的键，运行时报 `pipeline was not prepared by resize`。当前实现里它只存在于 `PipelinePlan`，`updateExecutionState` 从 plan 恢复。
  - 也因此 **2-pass 不走 replay**：它有 3 个 dispatch，而 `onReplayUpdate` 对 SDPA 路由只接受 2 个事件，事件数不符自然退回普通 encode。

### 2.3.6 Q-head-split Fused Decode QK+Softmax（QK_QSPLIT，**当前不可达**）

> ⚠️ §2.3.5 的 SDPA 阈值极低且在阶梯上排在 `DecodeFusedQkSoftmax` 之前 ⇒ QK_QSPLIT 连同它的宿主
> `decode_qk_softmax` 已不可达。**本节只作机制留存**，不要拿它给
> decode 问题定位。

- **机制**：`grid.z = group_size`，每个 q-head 独占一个 TG（TG 数翻倍），threadgroup 内存减半（单 scores 流）；配合**半宽 threadgroup**（总线程数与不拆分持平）。代价：K 每 kv-group 读 2 次、失去双流 ILP。
- **为什么在当时赚**：小模型 group=2 时 fused kernel 的 TG 数太少、GPU 吃不满，**并行度换冗余读是划得来的**（与 §2.3.5b 是同一根轴的相反方向）。
- **陷阱**：**threadgroup 宽度是命门**——沿用窄公式会让 TG 数翻倍触发启发式换挡而变慢，必须同步改成半宽公式。
- **适用边界**：tensor-API 档标定为负，auto 排除它是对的；group_size>2 的泛化未做（generic kernel 的数组动态索引有编译器劣化前科）。

### 2.3.7 Fused Decode Attention GQA 扩展

原 `decode_qk_softmax` 只支持 group_size=1，扩展为模板化 group_size（编译宏 + 按 `num_heads/num_kv_heads` 选 kernel）。**赚在避免 Q/K 的显式 repeat_kv 拷贝**（一次真实搬运 + 一次 dispatch）。

### 2.3.8 decode/prefill attention 路由速查

**代码结构**：路由是**一个枚举 + 一个 resolver**，不再是一堆互相取否的布尔。

| 概念 | 类型 | 作用 |
|---|---|---|
| `AttnFacts` | struct | 决策输入的快照，按三轴分组：seqLen 档 / 设备能力 / mask-KV 形态。`_collectFacts()` 每 token 重建 |
| `AttnRoute` | enum(6) | 唯一的"走哪条路"：`DecodeSdpa` / `DecodeFusedQkSoftmax` / `DecodeStaged` / `PrefillFaTc` / `PrefillFaSg` / `PrefillStaged` |
| `AttnPlan` | struct | route + 该 route 专属旋钮（nsg/kvb/qhPerTg/twoPass、qkQsplit、faTcWN、causalTri/causalBound、qseqSplit、各 simd 变体）。`_resolvePlan()` 是 `(facts, veto) -> plan` 的纯函数 |
| `AttnVeto` | bitmask | 编译期否决（tensor API / FA-TC / FA-SG / SDPA 的 pipeline 编不出来或超 threadgroup 上限），只被 resolver 消费 |
| `PipelinePlan` | struct | signature -> 编译产物 + `veto` + 两个编译期才确定的旋钮（`sdpaNsg` 降档结果、`sdpaTwoPass`） |

入口链：`onResize -> preparePipelines`（按代表性 kv 长度扫一遍边界，逐条编译并缓存）；`onExecute -> updateExecutionState -> _resolveRoute -> 查表 -> handleKVAllocMemory -> onEncode`。`_pathSignature(true)` 决定 replay 是否失效，`pipelineSignature()` = `_pathSignature(false)` + SDPA 旋钮 + kv 区间。

⚠️ **查表键取"未否决"的解析结果**：`preparePipelines` 与 `updateExecutionState` 都先把 `mVeto` 清零再算 signature，命中后才把 `plan.veto` 叠回去重解析。两侧规则必须一致，否则运行时报 `pipeline was not prepared by resize`。

**decode 阶梯**（`f.decode` = seqLen < 16）：
1. `DecodeSdpa`（§2.3.5）——`kvLen >= 2` 且 kvCache、非 disk、causal 或单元素 add-mask、`headDim%32==0`、tg 内存在 30KB 内 ⇒ **实践中 decode 全部走这一条**；kv ≥ 2048 自动切 2-pass（§2.3.5d）；旋钮 nsg（设备分档）、`SDPA_QH_PER_TG`（§2.3.5b）、`SDPA_KVB`（§2.3.5c）；
2. `DecodeFusedQkSoftmax`——`seqLen<=8 && kvLen <= FusedQkSoftmaxMaxKv(group)`（group 2/4/8 → 2048/1024/512）；可叠 `QK_QSPLIT`（§2.3.6）；
3. `DecodeStaged`——`decode_qk` → softmax → `decode_qkv`。

**prefill 阶梯**：
1. `PrefillFaTc`（§2.3.4）——tensor API 设备，`headDim ∈ {64,128,256}`，`seqLen >= 64`；**例外**：`headDim==256 && gen<=16` 让位给三段 tensor（寄存器驻留悬崖）；
2. `PrefillFaSg`（§2.3.9）——非 tensor API 设备，`headDim ∈ {64,128}`，`seqLen >= 512`（Pro 档）/ `1024`（其余）；
3. `PrefillStaged`——三段 + `CAUSAL_TRI`/`CAUSAL_BOUND`（§2.3.1），并按 `MNN_METAL_ATTN_QSPLIT_MB` 切 q 序列约束 scratch。

两条融合 prefill 的公共前提（`fusedPrefillOk`）：非 sinks、fp16、causal、KV 在内存、**非量化 KV**、`kvLen >= seqLen`。⇒ 开了 KV int8 就退回三段。

> legacy `gPrefillFlashAttn`（原第 2 优先级）已删除，见 §2.3.2/§2.3.3；`attention_mode / 8` 在 Metal 上不再选路。

### 2.3.9 STEEL-like fused prefill（`prefill_flash_attn_sg`，无 tensor-coop 设备）

- **机制**（`gPrefillFlashAttnSg`）：Q 进 smem（scale 折成 `scale*log2e`）；K/V 沿 head_dim 用 half4 合并读进**两块** smem（每 KV tile 一道 barrier）；QK 用 `simdgroup_load(K, transpose)` + 8x8 MMA；softmax / O 走 MMAFrag 布局（`thread_elements()` reinterpret `float2`，行归约 `shuffle_xor` 1 和 8，`fast::exp2`）；**P 存回 `simdgroup_half8x8`**，PV 用 half×half MMA 累加进 float O；causal 整 tile early-exit；ATTENTION_C4 epilogue，S 不落全局。
- **为什么赚**：与 §2.3.3 同源（打掉 `mTempQK` / `mTempSoftMax` 的 O(n²) 带宽），但用 STEEL 式 tile + 双块 smem 把 barrier 数压到每 tile 一道。
- **前几版为什么不赚**：S 落 threadgroup、单块 smem 分三次 barrier、float PV——**是"dual smem + half PV"两刀才转正的**。

> 🚨 **PV 必须走 half MMA**。把 V 从 `half8x8` 提升成 `float8x8` 再和 float 的 P 相乘，
> PV（占 attention 一半 FLOP）会掉到 float MMA 速率，整个 fused attention 慢一大截。
> **新写任何 MMA kernel，先确认两个乘数都是 half，只有累加器是 float。**

- **默认**：非 tensor-coop 且 seq 达到下限时 auto-on；短 prefill 与三段打平，故留三段。
- **正确性**：`op/attention*`（fp32 naive 参考）全过；e2e greedy 长 prompt 与三段可 byte-identical。个别 prompt 会在几十个 token 后因 fp16 累加顺序漂移在近平局 argmax 分叉，属 fp16 数值属性而非计算错误。

#### 2.3.9.1 内部归因与三档提速

**归因（消融阶梯，方法见 §2.0）**：从完整 kernel 逐类删到只剩 MMA，得到分项成本表与地板值。结论：**对手只比 MMA 地板高一点点，而 MMA 条数 / tile 形状 / grid / causal 提前退出四项两边一致 ⇒ 瓶颈是重叠效率，不是指令数。** 这直接决定了后续只做"提高 MMA:load 比"类手段。

**三档转正**（`FASG_LOADBATCH` 开关见 [`env-registry.md`](./env-registry.md)；另两档已固化为 kernel 唯一形态，回退开关已删）：

| 变体 | 机制 |
|---|---|
| `FASG_LOADBATCH=N`（env 保留） | 先发射一个 head_dim slab 的全部 fragment 再让 MMA 消费 ⇒ load 延迟被前面的 MMA 盖住，而不是停在需要它的那条 MMA 上 |
| 8 simdgroup / q tile 64 行（已固化） | simdgroup 4→8，q tile 行数翻倍 ⇒ 每层对 device K/V 流的重读次数减半 |
| Q 常驻寄存器（已固化） | Q fragment 常驻寄存器，不再每个 kv tile 从 smem 重发；单独中性，但腾出 LOADBATCH 要填的槽位 |

⚠️ **8 simdgroup 那一档在 LOADBATCH 之前单独测是判负的**——fragment load 的停顿本来就盖住了 K/V 重读的收益。**知识点之间存在顺序依赖，单旋钮 sweep 会把正向项误判掉**（§2.0 四条纪律第一条的来源）。

**这个 kernel 上已确定不是杠杆的方向**：寄存器 prefetch（WAR 串行化）、K/V smem union、smem 省用（32KB 上限下两种占用都只驻 1 个 threadgroup，**占用率没变 ⇒ smem 确定性地不是杠杆**）、K/V double buffering（一个 tile 才几 KB，而每 tile 每 simdgroup 有几十条 MMA，延迟本来就被盖住）、标量化 fragment load（编译器本来 lower 成等价代码）、mask 快路径（softmax 总占比本来就小）、给常量边界小循环加 `unroll(full)`。

**剩余差距的机制**：fragment load 与 MMA 是 1:1，因为每个 simdgroup 只做 8 行 q（`TQ=1`），每个 K fragment 只喂 1 条 MMA。改善只能 `TQ≥2`，但 head_dim=128 下 `mO[TQ][ND]` 加 `mQreg` 必溢出 ⇒ **再试 TQ≥2 前必须先证明寄存器足迹能压下来。**

**本例特有的坑**：**动态下标的 simdgroup-matrix 数组会溢出寄存器，代价数倍**，每个下标都要靠外层 `#pragma clang loop unroll(full)` 折成编译期常量。另外这个 kernel 曾**零覆盖**——`op/attention` 的 seq 太短、head_dim 不匹配，进不了它的门；后补了 `op/attention_prefill`（head_dim 64/128 × 长 seq）。**加了带门槛的新 kernel 就要同时加一个能真正点亮它的用例。**

## 2.4 其他 Kernel

### 2.4.1 RMSNorm 小 Batch 单 TG 路径（`MetalLayerNorm.mm`）

Decode 时 batch=1，默认 kernel 选择倾向大 batch tile，launch overhead 反而盖过计算。`batch <= 4 && hidden_size <= 4096` 时改用**单 threadgroup 处理整个 norm**。

这是"小 shape 走窄路径"的典型：**decode 的 batch=1 与 prefill 的 batch=seq 是两个世界，同一个 op 常需要两套 dispatch 形态**。新写 kernel 时先问 decode 形态是否退化。

### 2.4.2 LinearAttention（gated delta rule）变体全景

**画像**：长 prefill 上 LinearAttention 可以占到 GPU 时间三分之一（与全部 GEMM 相当），decode 侧占比很小 ⇒ **prefill kernel 是最大单点**。

**路由**（`MetalLinearAttention.mm`）：

| kernel | 命中条件 | 状态 |
|---|---|---|
| `linear_attn_gated_delta_rule_sg_v4` | 非 tensor-API && `mHeadKDim==128` && 长 prefill，配 `qkv_prep_sg` | **主力**（§2.4.3）|
| `linear_attn_fused_chunk_sg` | `dk != 128` 的长 prefill | 在用 |
| `linear_attn_flash_chunk_sgmm` | 编译门蕴含 dk==128 ⇒ v4 必然存在 ⇒ 分支不可达 | **事实休眠** |
| `linear_attn_fused_sg_align` | 短 prefill（`2 <= seqLen < 16`）| 在用 |
| `linear_attn_fused_sg_tg` | `H < 16 && seqLen == 1` | 在用 |
| `linear_attn_fused_sg` | 兜底 | 在用 |
| chunk64 双 kernel（flash_chunk）| tensor-API 设备 | 在用 |

几条已定型的机制结论：

- **短 prefill 走 `fused_sg_align`**：省 `qkv_prep` dispatch + mQ/mK/mV 物化往返；窄 head_dim 上收益更大。
- **`fused_sg_tg` 的阈值维持 `H < 16`**：TG 共享 Q/K 读在 H=16 档没有收益（冗余读被 cache 吸收）。**"共享读"类手段在冗余已被 cache 吸收的规模上不成立**，判断前先估 working set。
- **sgmm（8x8 MMA 重写 chunk 算法）的唯一 occupancy 杠杆是 TG 宽度**（threadgroup 内存接近上限 ⇒ 每核只驻 1 个 TG），但**过宽会反噬**；且它在窄 head_dim 上输给标量基线（那里用更小的 chunk），故门控收紧到 dk==128。
- **decode 接入 encode-replay** 是结构性改动（收益中性）：`canRecordEncode()` 从恒 false 改为 `seqLen==1 && gated_delta_rule`。关键障碍是 `Pipeline.cpp` 对 LinearAttention **每 token 强制 re-resize**，`onResize` 每次重建 `mConvOut` → 录制绑定悬垂 `Tensor*`。修法：shape 不变时保留 Tensor 对象 + resize-generation 守卫在 `onReplayUpdate` 里 bail（**必须先于** `metalReplayEmit` 的解引用）。详见 `runtime-scheduling.md`。

### 2.4.3 寄存器 state vec4 scan 替代 chunk（非 tensor-API 设备的 prefill 主力）

**"分块 MMA 不一定赢顺序 scan"的样本。**

- **定位方式**：与对手在同一 op 范围上做镜像对比，发现 prefill 差距全部落在 LinearAttention；对手的赢法是"不分块直接扫"：每 (b,h,dv) 一个 simdgroup、**state 驻寄存器**、零 barrier / 零 threadgroup memory、simdgroup 数拉满 occupancy。
- **实现三处**：① `linear_attn_gated_delta_rule_sg` 的 state **寄存器化**（原来每 timestep 读写 device 两遍）；② 新 kernel `linear_attn_gated_delta_rule_sg_v4`（**dk==128 特化**：每 lane 持 4 个连续元素，`half4` load + `dot`）；③ 路由：非 tensor-API && dk==128 && 长 prefill 走 `qkv_prep + v4 scan`。
- **两条决定性机制（合起来是同一条原则）**：
  1. **token-major 输入是 scan 赢的前提**。直接从 `conv_out [B,D,L]` 读会让 lane 间 stride=L、完全非合并；为此付出的 `qkv_prep` 物化成本远小于收益。
  2. **向量化才是决定项，寄存器化只是必要条件**。跨步 lane 映射（`lane + ii*32`）的寄存器版仍然输——4 次分散小 load vs 一次宽 load。
  > ⇒ 改 scan 类 kernel 时**先画 lane→地址映射，再谈算法**。
- **适用边界（重要）**：**tensor-API 设备上 chunk64 完胜 scan**（同一 op 单测上差出数倍）。`!mUseFlashChunk` 的保守门控是对的——在一个设备档上赢很多的路径在另一档上可能是灾难。

### 2.4.3.1 chunk64 内部归因 + conv row4

用临时 env 消融（只 dispatch 其中一段，结果故意错，仅做计时归因）拆长 seq 的 chunk64 路径，得到三段占比：`conv_silu`（depthwise conv1d + SiLU 头）/ `chunk64_prep` / `chunk64_recurrent`。

**意外**：大家一直在优化 scan/prep，而三分之一的成本在那个从没被碰过的 conv 头上。它是 one-thread-per-element 的朴素标量 kernel：每个输出 K 次 device 权重 load + K 次边界分支 + K 次 offset 重算。

**修法 `linear_attn_conv_silu_row4`**：一个线程负责同一 `(b,d)` 行的 **4 个连续 l**。4 个输出的 K-tap 窗口重叠，用 **4 寄存器滚动窗口** ⇒ 每 4 个输出只需 **K+3 次输入 load + K 次权重 load**（原来各 4K）；`conv_state` 左 padding 只影响一行的前若干个输出，所以内部走**无分支快路径**，只有行首/行尾走 clamp 慢路径。**k 累加顺序不变 ⇒ 数值逐位相同。** 门控 `seqLen>=32`：更短的 block 里 grid 缩窄 4x 的损失大于省 load 的收益。decode 形态（L=1）未改动。

> 教训：**先做消融归因再选目标。** 这个 op 被优化过多轮，没人量过 conv 头占多少；一个从未被看过的朴素 kernel 藏着三分之一的成本，而且是最容易修的那种（纯访存冗余）。

### 2.4.4 `linear_attn_gated_norm`（`MetalGatedNormShader.hpp`）

linear-attention **输出门控段**的单 kernel 实现，替代原本 7 个 dispatch 的链：

```
LinearAttention [H,dv,1,1] → Raster1(identity) → Cast(位拷贝)
  → RMSNorm(per-head) × SILU(Raster2(z)) → MUL → Raster3 → out_proj
```

Raster1 是纯位拷贝；**Raster2/3 是真实 C4 重排且互为逆**——融合掉它们省的是真实搬运。

**kernel 设计要点**（融合链路见 `graph-fusion.md`）：
- **一个 simdgroup 一个 head**。dv=128 ⇒ 每 lane 恰好 1 个 `float4` ⇒ **无 threadgroup 内存、单次 `simd_sum`** 完成 per-head 平方和归约。
- 索引 `la[c*outside+h]` 读、`z, out[(head*(inside/4)+c)*z_batch + b]` 写——**后者天然吃掉 Raster2/3 两次重排**，不需要额外搬运指令。
- shape 契约是 x `[batch*heads, inside]` / z,out `[batch, heads*inside]`，decode 是 `batch==1` 特例。**硬编码 batch==1 会让 prefill resize 直接 Compute Shape Error**；泛化做法是多带一个 `z_batch` 常量并把 `h` 按 `h = b*heads + head` 拆开，`z_batch==1` 时整式塌回原式 ⇒ **decode 逐位不变，不需要 pipeline 变体**。
- 归约本身必须与 `layernorm_c4_rms_sg` **逐字相同**——这是 fp32 bit-identical 的唯一依据。
- **适用边界**：**不要把它折进 out_proj 的 GEMV 前导**（LN_FUSED 形态）——每个 TG 要多读一份 z，decode 带宽敏感，冗余流量得不偿失；独立 kernel 只读一次。

**正确性口径 = fp32 bit-identical，而非 fp16 byte-identical**：fp32 下 fold 与不 fold bit-identical ⇒ 索引、布局、数学全部等价；fp16 下的个别 greedy 边界 token 分叉，若能二分到"只有 RMSNorm 半边不同"且源码与参考 kernel 逐字相同，判定为编译器 codegen 层面的等价重排。

**kernel 级二分方法（可复用，比 token hash 强得多）**：
1. **fp32 当 oracle，最先做**。fp32 bit-identical 一次排除索引/布局/逻辑错，把问题锁死在 fp16 rounding。
2. 分阶段临时 env 探针：关匹配 / 只做 STATIC 提升 / 只装 leader 不 claim / leader 退化为纯搬运 / 用链路中间结果替换我的 LN 或我的 SILU。逐个替换能把差异锁到某一半。
3. ⚠️ **诊断本身必须可靠**：替换读源的探针若不给那个中间张量做 STATIC 提升，它的生命周期在原消费者处就结束、内存可能已被回收，**探针会读到脏数据给出假结论**（曾因此拿到三个互相矛盾的 hash）。
4. 能算清的假设先算清再测：`channelUnit == SIMD_GROUP_WIDTH` 时每 lane 只迭代一次，`0 + d*d` 与 `fma(d,d,0)` 恒等，该实验无信息量。

### 2.4.5 融合 RoPE：寄存器复用 + token 分块（`MetalRope.mm`）

`MNN_SUPPORT_TRANSFORMER_FUSE` 下 q/k RMSNorm 与 RoPE 合成一个 kernel，两处已转正的手段：

**一、把 RMSNorm 趟的 load 留在寄存器里复用（`MetalRope.mm:522-529`，shader `X_CACHE_N` 分支）**。kernel 天然要两趟：先求 `square_sum`，再做旋转。原本两趟各从 device 读一遍整条 head。`step=32` 下 lane `t` 覆盖 `{t, t+32, ...}`，而旋转要配对的 `i` 与 `i+ropeHalfD`——**`ropeHalfD % 32 == 0` 时这两个下标落在同一 lane**，于是第一趟的值存进 `ftype xv[X_CACHE_N]` 就够，既不用重读也不用 shuffle。

- shape 门槛（不满足则走原重读路径，该 shader 臂**不是死代码**）：`USE_SG` 路径（`q_norm`/`k_norm` 存在 + `supportSimdGroupReduce()`）、`headDim % 32 == 0`、`ropeHalfD % 32 == 0`、`headDim/32 <= 8`。最后一条是寄存器预算：8 个 entry 在 fp16 下是每 lane 16 B。
- ⚠️ **`ROPE_D` / `ROPE_HALF_D` 必须一起烘成 shader 常量并进 pipeline cache key**。否则循环上界是 runtime 值、循环不展开，`xv[]` 变成动态下标数组直接溢到 thread-local 内存，省下的那趟 DRAM 读还不够抵消。这是 §2.5.7「编译期常量」的一个实例：**寄存器数组的前提是所有下标都能折成编译期常量**。

**二、token 分块的 `rope_kernel_tile`（`MetalRope.mm:295-380`，开关见 [`env-registry.md`](./env-registry.md)）**。C4 输入里同一 plane 内 token 相邻，而标量 kernel 一 lane 一个 channel slice ⇒ 32 lane 的地址在 token 维上是散的，一条 simd load 被拆成 32 次请求。分块 kernel 第一趟让 lane 铺 `ROPE_TILE_T` 个连续 token、每个 plane group 只发一次连续读并暂存进 threadgroup memory，第二趟切回"一 lane 一个输出 quad"做 `simd_sum` 与旋转（输出是稠密 `[token][head][dim]`，该映射下写本来就连续）。两条 pipeline 都在构造函数里编译，`onResize` 按 `outerSize` 选（`MetalRope.mm:622-633`，auto 门限 `kRopeTileMinSeq=64`）——**decode 的 `outerSize==1` 恒走标量，一个 token 没有连续 run 可铺**。

## 2.5 Kernel 优化手段方法论（原理 / 适用条件 / 陷阱 / 验证）

> 选题、方案评审、优化不见效时对照排查。

### 2.5.1 GEMV 融合 epilogue（尾段折叠）→ §2.1.6

- **原理**：decode 主体是带宽瓶颈型 GEMV。把紧随其后的逐元素算子（SwiGLU/bias/激活）折进 GEMV 尾段就地计算，省一次 dispatch 的固定开销，更省中间结果写回再读回的显存往返。
- **适用**：尾段只依赖本 kernel 已算出的元素；尾段算子逐元素或短邻域。
- **陷阱**：需要多路结果对齐的尾段（如 SwiGLU 需 gate/up 同 TG）要先解决数据汇聚问题（见 `graph-fusion.md`）。
- **验证**：对拍融合前后输出 bit 级一致；确认 dispatch 数真的减少。

### 2.5.2 LN 前序拆分到多 simdgroup → §2.1.9

- **原理**：多 SG kernel 中若每个 SG 各自加载同一份输入做前处理，读取量按 SG 数翻倍；改为按 SG 分工 + threadgroup 内存交换，公共输入只读一次。
- **适用**：2sg 及以上、前处理输入相同的 GEMV/GEMM kernel。**判据是"这个 barrier 是否附带 occupancy 代价"**：不增加 SG 数的 barrier 通常是赚的。
- **陷阱**：**部分线程提前退出 + barrier 是 UB**（陷阱 G）——必须全部线程到达 barrier 后再分工。改并行划分时还要重审所有"唯一写者"守卫。
- **验证**：输入读取量（profile 字节数）减半；输出对拍；**用非整除 shape 测尾块**。

### 2.5.3 Split-K GEMV → §2.1.5

- **原理**：小 batch GEMV 并行度不足时沿 K 切分给更多 lane/SG。收益本质是**翻倍在途 lane、提高访存并发**，不是省掉 barrier（免 barrier 的 shuffle 变体反而更差，已删除）。
- **适用**：K 很大而并行单元不饱和的 GEMV（decode 短上下文）；**不适用于已有共享前导的融合管线**（那里翻倍 SG 的占用代价更贵，用双流）。
- **陷阱**：K 切分使 weight 读取模式改变，量化块边界要与切分对齐，否则跳块/错块；若该 kernel 后紧跟 GPU→CPU 同步，kernel 加速不兑现为 e2e——**优化前先确认瓶颈段是不是你在优化的那段**。
- **验证**：conv/wquant 单测全变体通过；e2e 配对 A/B 而非只看 kernel 计时。

### 2.5.4 向量宽 load 与访存合并 → §2.1.4

- **原理**：带宽瓶颈 kernel 的快慢取决于访存模式能否吃满 DRAM 带宽。lane 持连续若干元素一条向量 load（ftype4/char4），simdgroup 32 lane 恰好覆盖整行 → 完全合并 burst；反之逐 token 跨步标量读每 2KB 只碰几字节，load 指令数翻数倍、burst 利用率骤降。
- **适用**：一切流式读取 KV cache / 权重的 kernel。
- **要点**：**数据布局决定 kernel 可达的访存形态**。想让某 kernel 跑成合并读，先改布局；布局是全局不变量，翻转必须所有读写方原子落地，中间状态不可运行。
- **反面**：**单纯"重新分组"同样多的读不会变快**（总在途字节数不变，§2.1.4）；"合并读"若没减少字节数也没增加并发，常常 kernel 级快而 e2e 平。
- **验证**：profile 带宽兑现率；同字节数下 load 指令数对比。

### 2.5.5 寄存器驻留 + 单遍流式（sdpa_vector 形态）→ §2.3.5

- **原理**：decode attention（seq_q=1）最优形态：Q 进寄存器不再碰显存；逐 token 交错流式（`i = sgitg; i < kv; i += NSG`），每 token K 行点积 → simd_sum → 在线 softmax → 立即用同 token V 行更新 O；score 不落任何内存，无第二段 AV dispatch；跨 SG 归并用转置写法让归并读合并。
- **适用**：decode 单 query 的融合 attention。
- **陷阱**：**lane↔输出维度映射**。流式循环 lane 持 `d = lane*DPT + dd`，归并写回若沿用旧映射（`d = dd*32 + lane`）会短 KV 正常、长 KV 后乱码。对拍必须覆盖长 KV + prefill 后首个 decode token。
- **验证**：与禁用路径（env 开关）对拍，长 prompt 必测；改了归约序就不能用 byte-compare。

### 2.5.6 量化解包向量化与公共加载去重 → 陷阱 H、§2.1.2

- **原理**：int4/int8 解包用向量指令一次 4/16 元素；多 SG 共用输入只加载一次经 threadgroup 共享。
- **陷阱**：把向量积（`in4 × FLOAT4x4`）重构成标量循环极易写成**转置乘积**且 scale/bias lane 错位——此类重构必须 bit 级对拍，肉眼/greedy 短 prompt 都可能漏掉。
- **验证**：fp32 bit-identical 或量化单测全模式通过。

### 2.5.7 编译期常量与 host 预算

- **原理**：host 可确定的量（split-K 中段步长等）以宏注入编译期，省 kernel 内除法/分支。
- **陷阱**：**甄别伪优化**——纯带宽瓶颈的 kernel 里，省下的整数除法与尾判会藏在访存延迟下面，收益为零。先定位"这个 kernel 现在被哪类资源锁住"再投入。
- **验证**：汇编/寄存器占用对比 + 配对 A/B。

### 2.5.8 线程组规模（NSG / TG 数）校准 → §2.3.5 / §2.3.5b

- **原理**：单 workgroup kernel 的 simdgroup 数是占用率/调度开销/归并成本的三方折中，无跨设备通用最优；dispatch 的 threadgroup 数则决定能不能填满 GPU，存在明显的并行度悬崖。
- **要点**：按设备档分别 sweep；**候选值全集都要测**（只比两端会漏掉中间的最优点）；KV/prompt 维度也要扫（最优点会随 KV 长度移动）；**多个 auto 旋钮读同一个量时注意解析顺序**（§2.3.5）。
- **验证**：多轮配对、逐对同向性检查，而非单次均值；归约序变了就换带容差的判据。

### 2.5.9 递推状态驻留与并行扫描（LinearAttention）→ §2.4.2 / §2.4.3

- **原理**：递推/scan kernel 把状态从 device 往返改寄存器驻留（每步一次 load/write）；chunk 内前缀和用 Hillis-Steele 并行 scan，前代求解摊到全部 simdgroup；窄 head_dim 写专用特化避免通用路径分支。
- **要点**：**寄存器驻留是必要条件，访存形态（token-major + 向量化）才是决定项**。
- **适用**：gated delta rule 等线性注意力的 chunk 递推；**chunk 与 scan 的优劣随设备档翻转**，必须分档门控。
- **验证**：与参考实现长序列对拍（递推误差会随序列累积）。
