---
name: vulkan-optimize
description: MNN Vulkan 后端 op/kernel 优化与扩展。覆盖 GLSL .comp + makeshader 双轨、conv1x1 dispatcher 多路径、coopMat/subgroup 路径、weight prepare/pack 流程、Adreno 真机稳定性。
---

# MNN Vulkan 优化 Skill

> **触发**：扩展或优化 Vulkan 端 kernel（compute shader / conv / gemv / gemm / attention 等），新增算子，调度选路或 weight pack 调整。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## 核心原则

1. **改 `.comp` 必跑 makeshader**。GLSL 源不会被构建系统直接编译。`buildKernelFromSource` 读的是 `AllShader.h/cpp` 中的 SPIR-V 二进制（由 `makeshader.py` 把 `.comp` → `glslangValidator -V -O` → 嵌入字符串）。**最常见的"改了不生效"根因**就是 makeshader 没跑。

2. **dispatcher 选路要先摸清**。Vulkan 同 op 多条路径：CoopMat (Adreno only) / subgroup (Mali/Adreno) / nosubgroup (兜底) / FP gemm。新加 quant bit 不可能一次扩完所有，必须显式让未扩展的 path 退到已扩展的（或 fallback 到 fp/CPU）。

3. **packed weight 必须 packing/unpack 双向镜像**。host weight prepare shader 写出的字节布局要和 decode shader 读取布局逐 bit 匹配，任何一边改动都要双边同步改。

4. **正确性 oracle 先于性能**。同 OpenCL/Metal skill。

5. **真机才算数，且面向多个 vendor**。Vulkan 同时跑 Adreno (Android)、Mali (Android)、Apple (iOS / MoltenVK) 等，**driver 稳定性差异极大**。Mac MoltenVK 的稳定性 ≠ Android Adreno。**进入性能/稳定性测试前先 `adb devices` 检查目标 Android 设备**，没设备时直接停下来给用户提需求。

6. **Buffer / Image 是编译期二选一**。MNN Vulkan 有两种"基础数据形态"，由 `MNN_VULKAN_IMAGE` 决定：`-DMNN_VULKAN_IMAGE=OFF` → `source/backend/vulkan/buffer/*`；`-DMNN_VULKAN_IMAGE=ON` → `source/backend/vulkan/image/*`（独立的 GLSL/Execution 树，**不会**自动从 buffer 同步）。**两个后端是完全独立的代码树**，buffer 那边的改动 image 那边没有。

   **重要陷阱**：`MNN_VULKAN_IMAGE` 默认值在不同 build 配置下不同。Android `project/android/build_64.sh` 默认 `=ON`（image 后端），常见 desktop / `MNN_VULKAN=ON` 默认 `=OFF`（buffer）。**意味着你在 Mac 上跑通的 buffer 后端 shader 改动，Android 默认 build 完全用不到**。

   修代码或测试前必做的两步：
   - `grep MNN_VULKAN_IMAGE project/android/build/CMakeCache.txt` 看当前 build 走哪边
   - 确认你修改的代码路径（`source/backend/vulkan/buffer/*` vs `image/*`）和 build 选的后端一致
   - 不一致时显式指定：`../build_64.sh -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF` 或 `=ON`

   **写 shader 时最稳的做法**：在两个后端同步改，或在功能描述里显式说"仅支持 buffer 后端"。否则用户切到默认 image 后端会以为"功能 broken"。

---

## 入口定位

```bash
grep -rn "OpType_<MyOp>" source/backend/vulkan/buffer/execution/   # buffer mode
grep -rn "OpType_<MyOp>" source/backend/vulkan/image/execution/    # image mode
```

低 bit 量化 conv 的 dispatcher 在 `VulkanConvolution.cpp::onCreate`，conv1x1 的多路径选择从 `useInt8Conv && is1x1 && singleInput` 开始。识别 quant：`quanWeight->canUseInt2/3/4`。

conv1x1 选路（粗略）：

```
useInt8Conv && is1x1
  ├─ coopMat supported && Adreno
  │    ├─ perChannelAsym + S8S8S32 → VulkanConv1x1CoopA8
  │    └─ else → VulkanConv1x1Coop          (CoopMat 只支持 int4/int8)
  └─ → VulkanConv1x1General                  (有 native int8/int4/int2/int3 path)

else (非 1x1 / 非 single input)
  → VulkanConvolutionSlideWindowsInt8
```

新加 quant bit 时**关键决策**：CoopMat 路径要不要扩？工作量大但 Adreno 性能高；不扩就在 dispatcher 让 w2/w3 跳过 CoopMat 走 `VulkanConv1x1General`（已写有 native 低 bit gemv）。

---

## 通用陷阱

### 陷阱 A：makeshader 流程是预编译，不是运行时编译

GLSL `.comp` → SPIR-V 二进制需要 `glslangValidator`，由 `source/backend/vulkan/buffer/compiler/makeshader.py` 一次性把所有 shader 编译成 `AllShader.h/cpp` 嵌入二进制。改 `.comp` 后必跑：

```bash
cd source/backend/vulkan/buffer/compiler && python3 makeshader.py
```

verify: `grep -c '<my_kernel>' AllShader.h` > 0。

新加 shader 还要在 `glsl/macro.json` 里登记 `useFP16: true/false` 决定是否生成 `_FP16` 变体，否则 host 端 `getPipeline("glsl_<name>_FP16_comp", ...)` 找不到。

### 陷阱 B：CoopMat 路径只覆盖 int4/int8

CoopMat 用 hardware S8S8S32 / S4S8S32 cooperative matrix 指令，**只对 int8 / int4 weight 有定义**。新加 w2/w3 时 host 必须显式跳过 CoopMat：

```cpp
const bool isLowBit23 = quanWeight && (quanWeight->canUseInt2 || quanWeight->canUseInt3);
if (!isLowBit23 && coopMatInfo.supportCoopMat && ...) {
    return new VulkanConv1x1Coop(...);
}
return new VulkanConv1x1General(...);    // 走 native low-bit gemv
```

否则跑到 CoopMat 用错 layout 解 buffer，要么数值乱，要么直接 driver crash。

### 陷阱 C：未扩展路径吃错 buffer 大小

Vulkan 一个 op 的 buffer 大小（`decodeWeightBytes`）通常按 `mIsInt4 ? padK/8 : padK/4` 算。新加 w2 (`padK/16`) 时如果 dispatcher 漏一个分支让某 kernel 还按 `padK/4` 计算 buffer size，会 OOB 读到下个 op 的数据 → 数值错或 crash。

应对：dispatcher 改完后在 host 处搜 `mIsInt4`，**所有以它分流 buffer size / stride 的地方**都加上对 `mIsInt2`/`mIsInt3` 的处理，或者 fallback。

### 陷阱 D：multi-buffer pack（如 w3 lo2 + hi1）的 stride

3-bit weight 不齐 32 位常用 split：`[N, padK/16, 2]` uint pairs，pair[0] = 16 个 weight 的低 2 bit，pair[1] = 高 1 bit（低 16 bit 用）。这种 layout 的 host buffer size 计算要乘 `wordsPerGroup = 2`，**不是单纯 stride/16**：

```cpp
const uint32_t wordsPerGroup = isInt3 ? 2u : 1u;
const size_t decodeWeightBytes = padN * decodeWeightStrideWords * wordsPerGroup * sizeof(uint32_t);
```

shader 内 `xy_wt + uz * weightStride * 2` 也要相应 ×2，否则跨行读错位。

### 陷阱 E：driver OOM / 设备重启

Vulkan 后端在某些 driver 上每个 op 留一组 GPU buffer（raw weight + decoded weight + temp pack）常驻，**累积超 GPU 单进程 VRAM limit 直接重启手机**。这不是 quant bit 问题，是后端架构问题。

应对：
- 跑大模型前先用小模型（< 1B）验证 kernel 通路
- 大模型崩 → **不要循环 retry**，让设备恢复，单独 issue（buffer 复用 / lazy alloc / 不留 raw buffer）
- 0.6B 通 → 4B 崩 = 是后端 buffer 总量问题；0.6B 也错 = kernel 自己有 bug

### 陷阱 F：subgroup vs nosubgroup 双 shader 必须同步

每个 gemv/gemm 路径通常有 `_comp` 和 `_nosubgroup_comp` 两个变体（subgroup intrinsic 不可用时走后者）。新加 quant bit **必须两个都加**，否则不支持 subgroup 的设备 (老 Mali) 直接挂。

---

## Packed weight 设计

新加 quant bit 时**先固定 5 个量**：

| 量 | 解释 |
|---|---|
| tile（最小访问区块） | Vulkan conv1x1 typical 是 row-major `[N, padK/W]` 每 word 装 W 个 weight |
| word/group 内 weight 数 | w2: 16 weights / uint32 (2 bit each); w4: 8 weights / uint32; w8: 4 weights / int32 |
| 多 word split（如有） | w3 用 lo2 + hi1 双 word，每 group 16 weights × 2 uint = 8 bytes |
| bit 顺序 | 低 bit 先（`(packed >> (i*N)) & mask`） |
| signed/unsigned 存储 | unsigned (`signed + offset`)，shader 内 `(unpacked - offset) * scale + bias` |

**signed/unsigned 与 originOffset**：模型导出器写出的 alpha 是 `b = min_val + offset_signed * scale`，**originOffset 已折进 bias**。但 Vulkan int8 path 用 `bitfieldExtract(packedW, 0, 8)` **会做 sign-extension**，所以 signed bytes 直接当 signed 解出来，无需再减 offset。这点和 OpenCL/Metal 不同——后两者从 unsigned char 解，需要自己减。

**w3 split 设计**：`[N, padK/16, 2]` uint pairs。pair[0] 低 16 bit 装 16 个 2bit，pair[1] 低 16 bit 装 16 个 1bit。decode shader 单次读一对，按 `q = (low2 >> (i*2)) & 3 | ((hi1 >> i) & 1) << 2` 重组成 3bit unsigned [0,7]，减 4 得 signed [-4, 3]。这样比"32 weights = 12 bytes 跨 word 边界"对齐方案简单很多。

---

## Shader 修改流程

```bash
# 1) 编辑 .comp（注意 buffer 还是 image 后端）
vi source/backend/vulkan/buffer/execution/glsl/<my_kernel>.comp

# 2) 在 macro.json 登记（新文件必须）
vi source/backend/vulkan/buffer/execution/glsl/macro.json
# 加 "<my_kernel>.comp": { "useFP16": true } 等

# 3) 重新生成 SPIR-V 嵌入（默认用 .cache 缓存）
cd source/backend/vulkan/buffer/compiler && python3 makeshader.py
# 强制重编译（绕过缓存）：python3 makeshader.py -f
# image 后端：cd source/backend/vulkan/image/compiler && python3 makeshader.py [-f]

# 4) 验证嵌入
grep -c '<my_kernel>' ../shaders/AllShader.h  # > 0 才算进
ls .cache/shader/generated/<my_kernel>*       # 看到 .comp 和 _FP16.comp 才算

# 5) build
cd build && make -j8 MNN_Vulkan MNN llm_demo
# or Android
cd project/android/build && ../build_64.sh -DMNN_VULKAN=ON ...
# 想要 GPU per-op 时间统计：加 -DMNN_GPU_TIME_PROFILE=ON（仅 MNN_VULKAN=ON 生效，有额外开销，release 不开）
```

makeshader 重生成的文件：`AllShader.cpp`、`VulkanShaderMap.cpp`、`shaders/AllShader.h`。**Buffer 与 Image 后端各有自己一份**，不会互相同步。需要 host 工具链：`glslangValidator`（GLSL→SPIR-V）、`xxd`（.spv → C 数组）。缺工具时让用户安装 Vulkan SDK 或系统包。

**新加 kernel 的同步检查清单**：

| 位置 | 检查 |
|---|---|
| `.comp` 主路径 + `_nosubgroup` 变体 | 都加 |
| `macro.json` | 新加 entry，`useFP16` 决定 `_FP16` 变体 |
| host pipeline 选择 | `mIsInt2/3` 分支选 `glsl_<name>_FP16_comp` 或 `glsl_<name>_comp` |
| `decodeWeightStrideWords` | 按 packed bit 重新算 |
| `decodeWeightBytes` (含 `wordsPerGroup`) | split layout 时 ×N |
| 入口 `_init` 标志位 (`mIsInt2/3`) + clone 复制 | 都加 |
| dispatcher (`VulkanConvolution.cpp`) | 跳过 CoopMat / 选 General |

**编译错调试**：`makeshader.py` 调用 `glslangValidator`，错误直接打到 stderr。运行时 driver 编译错则在 `Warning: Create function failed: ...` 后看 SPIR-V validation 报错。常见：

| 错误 | 原因 |
|---|---|
| `'undeclared identifier'` | bind 漏写或 `layout(binding=N)` 序号冲突 |
| `Compilation failed` from glslangValidator | GLSL 语法错（`bitfieldExtract` 参数顺序、`uvec` vs `ivec` 等） |
| 编译过但运行时崩 | descriptor layout / push constant size mismatch host vs shader |

---

## 正确性验证

```bash
# 切后端
sed 's/"backend_type": "cpu"/"backend_type": "vulkan"/' transformers/llm/export/<model>/config.json > <model>/config_vk.json

# Mac 测试
DYLD_LIBRARY_PATH=build:build/express build/llm_demo transformers/llm/export/<model>/config_vk.json /tmp/prompt.txt

# Android 测试（必须 adb 真机）
adb push project/android/build/libMNN.so /data/local/tmp/MNN/
adb shell "cd /data/local/tmp/MNN && rm -rf tmp/mnn_cachefile.bin; LD_LIBRARY_PATH=. timeout 180 ./llm_demo <model>/config_vk.json prompt.txt 2>&1 | tail -20"
```

CPU/Vulkan 同 prompt + `temperature=0.0` 前 N 个 token 应一致（fp16 误差内）。

**模型本身可能就坏**（小模型在极低 bit 上量化退化常见，CPU 跑也乱）。Vulkan 验证前先 baseline CPU。**示例**：Qwen3-0.6B 的 w2/w3 量化 CPU 跑就是乱码，4B / 8B 才是有效验证样本。

**Mac MoltenVK 行为不代表 Android**。Mac 上跑通的 shader 在 Adreno/Mali 可能因 driver 实现细节崩。最终验收必须 Android 真机。

**数值偏差容忍**同 OpenCL/Metal skill。

---

## 实现新分支前的调研

在动 Vulkan 代码前先做一轮跨后端调研，能省掉 80% 设计弯路：

1. **CPU 是 ground truth**：算子的语义、shape 约束、KV cache 协议、数值范围都先在 CPU 实现里读一遍
2. **Metal / OpenCL 是 GPU 同类参考**：如果 Metal 或 OpenCL 已经有同语义分支（如低 bit 量化、GQA、新 attention 变体），优先复用其 layout / dequant / dispatch 思路。三个 GPU 后端的 GLSL/MSL/CL kernel 结构通常类似，跨后端迁移比从零设计省时
3. **Vulkan baseline 是不是已经 cover**：grep `onCreate / onResize / onEncode / VulkanXxx::create`，看是否已有 early-return / fallback / TODO guard。如果有 explicit unsupported-path，要决定是补这个分支还是新增独立 path

**只有 CPU/其他 GPU 后端都没有这个分支** 才需要从零给方案。新方案至少要明确：分支触发条件、layout 前提、shader/pipeline 选择规则、不支持时的 fallback、验证路径。

**在 dispatcher 上加白名单 / 黑名单的方式接入**：Vulkan 的多 path 调度让"未实现的分支"非常容易 silently 走到错误 kernel。新增 quant bit / 形状分支时，dispatcher 上要么显式 route 到新 path，要么显式 fall back 到已实现 path / fp / CPU；不要让"碰巧编译过"的 kernel 跑你的新 buffer。

## 性能优化方法论

### Vulkan 路径选择影响巨大

Vulkan 同 op 的 CoopMat / subgroup / nosubgroup 三种路径性能差距很大（量级差异常见）。先用 `vkBn->getDevice().getCoopMatInfo().supportCoopMat` 等查询设备能力，再选合适路径。**不要把 fallback (`nosubgroup`) 路径性能当 baseline**——它本身就慢。

### Decode 通常不是 BW bound

Vulkan 在 mobile GPU 上 decode 实测 BW 饱和度往往不高，瓶颈在 driver / launch / sync 而不是 BW。

按饱和度选杠杆同 OpenCL skill。

### Vulkan 特有杠杆

- **CoopMat int4/int8 path 已优化**：扩展新 quant bit 想达到同档性能必须扩 CoopMat（涉及 `int4_weight_to_coop.comp` 等），工作量大于 General path
- **prefill outer dequant + fp gemm 是次优 fallback**：用 `int<N>_weight_to_pack.comp` 一次 dequant 到 fp16 临时 buffer 跑 fp gemm。简单但比 native CoopMat 慢一档
- **subgroup size 影响**：`gl_SubgroupSize` 可能 32 (Adreno) / 16 (Apple) / 4 (老 Mali)。WGS 设计时按 `getSubgroupSize()` 动态算，不要 hardcode

### Apple GPU 经常不是 BW bound

unified memory 带宽很高，但 LLM decode 实测 BW 饱和度往往不高。BW 减半（如 w4 → w2）的 decode 提升常远小于理论值，先量化饱和度再决定要不要做 bit 杠杆。

---

## 调试套路（症状到先看哪里）

| 症状 | 第一怀疑 |
|---|---|
| 改了 `.comp` 跑出来还是旧行为 | (a) makeshader 没跑 / 没在 macro.json 登记；(b) **改的是 buffer/ 但 build 默认 image/，或反之**（先看 `MNN_VULKAN_IMAGE`）|
| 在 host 端代码里加的 log 完全不输出 | 你的代码所在后端（buffer 还是 image）没被 build 选中。Android `build_64.sh` 默认 `MNN_VULKAN_IMAGE=ON` |
| 输出全 FFFF / 同一个 token | weight buffer layout 不对，或 dispatch 调到了未扩展的 path（CoopMat 拿到 w2 buffer 等） |
| 输出合法字但没逻辑 | dequant 数值错；先看 sign-extension（`bitfieldExtract` int vs uint）和 originOffset fold |
| Mac 通 Android 崩 | (a) driver 差异 → 换 nosubgroup 路径试；(b) buffer 总量超 limit → 用小模型验证 |
| 4B 模型 Android Vulkan 重启 | 90% 是 image vs buffer 后端不匹配（默认 image 后端对某些 op 路径未实现）。先 `-DMNN_VULKAN_IMAGE=OFF` 切 buffer 后端验证。如果切 buffer 仍崩，才是 buffer 总量管理问题 |
| 性能不达预期 | 先确认走的哪条路径 (CoopMat / subgroup / nosubgroup)，再算饱和度 |
