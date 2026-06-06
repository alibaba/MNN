---
name: opencl-optimize
description: MNN OpenCL 后端 op/kernel 优化与扩展。覆盖 .cl + codegen 双轨、kernel 选路、packed weight 设计、tune 机制、Android 真机验证。
---

# MNN OpenCL 优化 Skill

> **触发**：扩展或优化 OpenCL 端 kernel（conv/gemm/gemv/attention/elementwise 等），新增算子，调度选路或 pack layout 调整。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## 核心原则

1. **改 `.cl` 必跑 codegen**。`.cl` 源不会被构建系统直接编译；运行时读的是 `*_mnn_cl.cpp` 嵌入字符串。每次改 `.cl` 后跑 `python3 source/backend/opencl/execution/cl/opencl_codegen.py . .`，并用 `strings libMNN.so | grep <你新加的宏>` 确认进二进制。**最常见的"改了不生效"根因**。

2. **dispatcher 选路要先摸清**。OpenCL 同一个 op 常有多条 kernel（subgroup/no-subgroup/Image/Buffer/不同 WGS）。改 kernel 前先读 host 选路代码，确认目标 shape/dtype 落到哪一条。盲改经常发现根本没被调度。

3. **packed weight 必须 packing/unpack 双向镜像**。host 写出的字节布局要和 shader 读的字节布局逐 bit 匹配，任何一边改动都要双边同步改。

4. **正确性 oracle 先于性能**。GPU 输出错和算法错难以分辨。优化前要有"已知正确"的 baseline（CPU 或上一版 GPU kernel），每一步改完都比对它。

5. **真机才算数（Android）**。OpenCL 主要面向 Android 手机 GPU（Adreno / Mali）。Mac/desktop OpenCL 行为差异大，不能代表手机性能或稳定性。**进入性能/稳定性测试前先 `adb devices` 检查**，没设备时直接停下来给用户提需求："请连一台 Android 真机，目标 GPU [Adreno 740 / ...]"，不要拿 Mac 数字当结论。

---

## 入口定位

改 OpenCL kernel 之前先回答：**目标 op 走哪个 Execution？哪条 dispatch 分支？编译进二进制的 kernel 字符串在哪个文件？**

```bash
grep -rn "OpType_<MyOp>" source/backend/opencl/execution/buffer/  # 入口
```

低 bit 量化 conv 入口在 `ConvBufLowMemoryExecution`（**不是** `ConvBufExecution`），识别：op 有 `quanParameter` 且 `aMaxOrBits ∈ {2,3,4,8}`。

每个 Execution 的 `onResize` 决定本次走哪个 kernel。同一 op 的常见 dispatch 维度：

| 维度 | 分支 |
|---|---|
| batch | `==1`（gemv, decode）vs `>1`（gemm, prefill） |
| weight | image (RGBA texture) vs buffer (linear) |
| local size | WGS 8/16/32/64/128/256（runtime tune 选） |
| 后处理 | `OUTPUT_CHANNEL_LEAVES` / `INPUT_CHANNEL_LEAVES_NUM` 是否非零 |
| 量化 bit | `QUANT_BIT=2/3/4/8` 决定 unpack 路径 |

读 `onResize`、`tuneXxxLowMemory`、`useFPWeightGemmLowMemory`，把目标 shape 代入，确定它落到哪个 `buildKernel(...)`，再去改对应 `.cl`。

---

## 通用陷阱

### 陷阱 A：host weight prepare 与上游 quant 路径约定不一致

新增量化 bit / 新 op 接入 host 时，常踩到"`ConvolutionCommon::load` 这次没填 outputPtr"或"alpha buffer 的 originOffset 已折叠/未折叠"这类**约定不一致**。

**示例**：4bit `forceQuant` 路径会把权重写进 host 提供的 outputPtr，但 2/3bit `forceQuant` 路径不写（留在 `quanCommon->weight` 里单独 alloc）。host 端如果假设两条路径同样写 outputPtr，staging buffer 就空着，packing kernel 读垃圾 → 输出全 `!!!!!`。

应对：
- 进入新路径前确认 `quanCommon->weight.get()` 是 packed 还是 unpacked、signed 还是 unsigned、offset 是否已折进 alpha
- 不要假设新 bit 会走和已有 bit 同样的预处理，每条新路径单独验证 staging/scale buffer 是否真的被写入
- 验证：dump 第一个 op 的 alpha + 前 64 byte weight 到日志，CPU 和 GPU 对照

### 陷阱 B：kernel/host pack-size 不匹配

每加一种 quant bit 或 layout，下面**所有**位置必须同步：

1. host 端 buffer 分配（`output_size` 公式）
2. packing kernel 写入字节数
3. dispatch global size 计算
4. unpack kernel 的 stride 和 offset 公式
5. Image2D 的 image_format 和宽高（是否能整除 RGBA pixel）

少改任何一处都是数值错或 OOB。改动时把这 5 处列成 checklist。

### 陷阱 C：误进 Image2D 路径

OpenCL 在 mobile GPU 上对小尺寸 weight 倾向 Image2D（专用 texture cache 比 buffer 友好）。但 Image RGBA pixel = 16 字节，packing tile 不是 16 整数倍时会强制 round-up，**浪费带宽抵消甚至超过 cache 收益**。新加 layout 时 default `mUseImage = false`，确认 host 端 image 路径分支已 explicitly handle + tile size 对齐 16B 后再开。低 bit packing（如 8B/12B per tile）通常应禁用 Image，老老实实 buffer。

### 陷阱 D：tune level 重复 tune

默认 `mTuneLevel = Wide` 对常见 GEMV/GEMM **已经**在搜索 WGS / shape variant（如 `tuneGemvLowMemory` 自动搜 `WGS ∈ {8,16,32,64,128,256}`）。再在外层手动 tune 或 hardcode WGS 等于跟内置 tune 抢资源，反而错过最优。先 grep `getCLTuneLevel()` 确认 tune 在哪一层做的，再决定要不要新加。Heavy 不会让 GEMV 进一步加速——Heavy 的额外搜索是给其他 kernel 用的。

### 陷阱 E：直觉的 BW 浪费可能已被 GPU cache 吸收

mobile GPU 有 L2 + texture cache。"表面看每个 wavefront 都重读同一段数据"未必意味着 DRAM 真的多读 N 次。改 kernel 让 wavefront 共享读之前，先验证是否真 DRAM bound（饱和度 + 实测对比），否则代码复杂度上来了性能没动。**典型例子**：GQA 注意力的 K/V 在 query head 间 N× 重复读，看似浪费 75% BW，但 Adreno L2 自动吸收 + 占总 BW < 10%（attention 不是热点），group-shared 改造常常零收益甚至略降（寄存器压力 offset）。

### 陷阱 F：宏 alias 让多个 #ifdef 同时为真

为了让"未扩展的 kernel"在新 quant bit 下编译过，常加 `#define W_QUANT_4` alias。**坑**：alias 让 `#ifdef W_QUANT_4` 在你想扩展的 kernel 里也被命中，body 用 W_QUANT_4 layout 跑你的 W_QUANT_2 buffer → 数值错但能跑。

修法：扩展的 kernel 里所有相关 `#ifdef` 必须 `W_QUANT_2 → W_QUANT_3 → W_QUANT_4 → W_QUANT_8` 顺序，**新 bit 放最前**优先匹配。

---

## Packed weight 设计

新加 quant bit 或调整 tile 排布时，**先固定 5 个量**：

| 量 | 解释 |
|---|---|
| tile = (IC_inner × OC_inner) | 一次原子访问的最小区块（OpenCL 常见 4×8 = 32 weights） |
| 字节/tile | 由 bit 决定：w2 = 8B, w3 = 12B, w4 = 16B, w8 = 32B |
| byte index 内的语义 | 哪个 byte 对应哪个 (oc_inner, ic_inner) 子集 |
| bit 顺序 | 单 byte 内 OC0/OC1/... 在哪几个 bit |
| signed/unsigned 存储 | shader 解出后是否还要减 originOffset |

这 5 个量先在 PR 描述里写死，packing 和 unpack 各自照表实现。先跑通正确性，再优化。

**signed/unsigned 与 originOffset**：模型导出器（`torch_utils.py`）写出的 alpha 是 `b = min_val + offset_signed * scale`，**originOffset 已折进 bias**。shader 解出 signed 权重（如 w2 `[-2,1]`）后做 `signed_w * scale + b` 即可。**不要**再做 `(unsigned - offset) * scale + raw_b`，会重复折一次。CPU `ConvInt8TiledExecutor` 是反方向（存 unsigned，host 折 bias），别照搬。

**block-quant alpha 索引**：内存布局通常 `[OC/4, blockNum, 2 (s,o), 4 (oc_inner)]`。kernel 读 `dequantScale[(oc/8 * 2 + bi/blockSize) * dstChannelC4 * 8 + ...]` 时 4 个维度的顺序要和 host 写法严格一致。

---

## .cl 修改流程

```bash
# 1) 编辑 .cl
vi source/backend/opencl/execution/cl/<my_kernel>.cl

# 2) 重新生成 _mnn_cl.cpp（必跑）
cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . .

# 3) 验证嵌入
grep -c '<新宏名>' <my_kernel>_mnn_cl.cpp     # > 0 才算进
strings build/.../libMNN.so | grep '<新宏名>' # build 后再确认

# 4) build
cd project/android/build && ../build_64.sh -DMNN_BUILD_LLM=ON ...
```

**新加 `QUANT_BIT==N` 时通常每个 shader 有 4 处都要加分支**（不是 1 处）：

| 位置 | 含义 |
|---|---|
| WGS≥8 主循环 | `useLocalMem=true`，IC ≥ 32 |
| WGS≥8 leaves | `INPUT_CHANNEL_LEAVES_NUM != 0` 时尾部 |
| WGS<8 主循环 | 单线程 reduce，IC < 32 |
| WGS<8 leaves | 同上的尾部 |

改完用 `grep -n "QUANT_BIT == 4" <file>.cl` 数 N 个，确认 `== 2` 也有 N 个。

**host buildOptions 与 .cl #ifdef 对应**：宏写错（`QUANT_BIT_2` vs `QUANT_BIT == 2`）shader 编译不报错，悄悄走 `#else` → 数值错。新加分支后扫一遍宏名拼写。

**重复展开的 unpack 用 `#define` macro，不用 inline function**——Adreno 老编译器对 inline 稳定性差。

**codegen 会重新生成所有 `*_mnn_cl.cpp`**，git diff 看到无关文件也变（unary_buf_mnn_cl.cpp 等）属正常，正常提交即可。

---

## 正确性验证

GPU 输出"乱"容易被错怪成模型问题或 sampler 问题，性能调优前必须独立完成。

**三层 oracle，从近到远**：

| 层级 | oracle | 检验点 |
|---|---|---|
| 数值层 | CPU 跑同一 op，dump tensor | 单 op fp16 误差 < 1e-2 |
| op 层 | `MNNV2Basic.out` 单层 conv | 输出与 CPU 对齐 |
| 端到端 | 跑模型 | 文本/输出语义合理 |

发现端到端乱码时**先回到数值层**，不要直接调端到端 sampler。

**端到端必须关 sampler 随机性**：`temperature: 0.0`, `sampler_type: greedy`，CPU/GPU 同 prompt + 同 config 前 N 个 token 应完全相同。温度 > 0 的乱码无法判断对错。

**模型本身可能就坏**：小模型在极低 bit 上量化退化常见，CPU 跑也乱。GPU 验证前先 baseline CPU。CPU 都乱 → 换更大模型测。

**示例**：Qwen3-0.6B 的 w2/w3 量化 CPU 跑出来就是乱码（PPL 已崩），不是 GPU kernel 的锅；4B / 8B 才是 GPU 低 bit 验证的有效样本。

**真机测试入口**：

```bash
adb devices       # 必须有设备
# 推 binary
adb push project/android/build/{libMNN.so,libMNN_Express.so,libllm.so,llm_demo} /data/local/tmp/MNN/
# 切后端
adb shell "cd /data/local/tmp/MNN && sed 's/\"backend_type\": \"cpu\"/\"backend_type\": \"opencl\"/' <model>/config.json > <model>/config_cl.json"
# 跑
adb shell "cd /data/local/tmp/MNN && rm -rf tmp/mnn_cachefile.bin; LD_LIBRARY_PATH=. timeout 180 ./llm_demo <model>/config_cl.json prompt.txt 2>&1 | tail -20"
```

`timeout 180` 重要：模型 load 慢或 hang 时不阻塞 shell。

**设备掉线 / 重启**：跑较大模型后手机可能 hang。`adb devices` 列表为空就**等设备回来再继续**，不要循环 retry 让用户手机一直挂。如果是后端 buffer 总量超 GPU 单进程 limit（Adreno 典型），是后端架构问题，不是当前 kernel 的问题，先换小模型继续。

**数值偏差容忍**：

| 路径 | 容忍误差 |
|---|---|
| fp32 vs fp32 | abs < 1e-5, rel < 1e-4 |
| fp16 vs fp16 | abs < 1e-2, rel < 5e-3 |
| 量化 dequant + fp16 | abs < 1e-1, rel < 1e-2 |

LLM 端到端跑 fp16 一般 token 完全一致到 ~50-100 个后开始分叉，**对比靠前 N 个 token 是否一致**而不是整段输出。

---

## 性能优化方法论

**不要先猜瓶颈再优化**——先量化，再选杠杆。

### 1. 算 BW 饱和度

```
BW saturation = weight_bytes_read_per_token / decode_time_per_token / theoretical_BW
```

设备的 LPDDR/unified memory 理论带宽查 spec，单进程实测可达通常打 6-8 折，具体数字按当前测的设备实测后再用。

**示例（数字会过时，仅作 order-of-magnitude 参考）**：

| 设备 | LPDDR 理论 | GPU 单进程实测可达 |
|---|---|---|
| Snapdragon 8 Gen3 | ~67 GB/s | ~50 GB/s |
| Snapdragon 8 Elite | ~76 GB/s | ~55 GB/s |
| Mali-G715 | ~50 GB/s | ~35 GB/s |

### 2. 按饱和度选杠杆

| 饱和度 | 瓶颈 | 主要杠杆 | 改动量 |
|---|---|---|---|
| < 50% | 不在 BW（ALU/launch/sync） | 简化 dequant ALU；减少 dispatch；persistent threads | 中-大 |
| 50-70% | launch overhead + L1/L2 miss | kernel fusion；调整 WG 大小 | 大 |
| > 70% | 真 BW bound | 减小 weight bit；packed 存储；无中间 fp inflate | 中 |

**饱和度 < 50% 改 weight bit 收益微小**：bit 减半带来的 BW 节省被其他开销吃掉，decode 提升远小于理论的 2×。

**饱和度 > 70% 才适合做 bit 杠杆**：把 weight bit 减半才有接近线性的 decode 提升。

**launch overhead 估算**：dispatch 数 × 每次约 5-10 μs。LLM decode 1 token ≈ `n_layer × ops_per_layer` 次 dispatch，先估算 launch 上限再判断瓶颈在 BW 还是在 dispatch；当 BW 优化把 decode 拉到 launch 上限附近时再做 kernel fusion 等大改。

**示例**：`n_layer = 32, ops_per_layer ≈ 7`，每 dispatch 5-10 μs → 1 token 约 1.7 ms，decode 上限 ~600 tok/s。BW 优化把 decode 拉到 100+ tok/s 时 launch 占比已经 > 17%，再优化 BW 收益递减。

### 3. 性能改动的标准流程

1. 量化当前瓶颈（数字，不是猜）
2. 选 1 个杠杆（不要一次改两个，结果归因不清）
3. 用 oracle 验正确性
4. **目标设备**测速（Android 真机）
5. 没收益就 revert，记录"为什么不 work"

GPU 跑分波动 ~5-10%，单次数字不可靠。同设备跑 10 次取中位数对比。

---

