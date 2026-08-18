# Vulkan 优化技巧手册

> 本文件从 MNN Vulkan 后端（buffer 后端，主要面向 Adreno / Mali / Apple）的实际优化经历中提炼。§2 技巧都有真实提交佐证和量化收益；§6 是有依据但尚未实测的候选方向。覆盖 **kernel/访存级** 与 Vulkan 特有的 **CPU 侧调度级** 优化。
>
> 用法：先用 §1 定位瓶颈（**Vulkan 要先分清 CPU 调度 vs GPU kernel**）→ 按 §5 速查表筛技巧 → 读 §2 对应条目。
>
> **维护约定：本文件是 Vulkan 优化技巧/陷阱的唯一来源。** 每次沉淀新经验都更新这里——新技巧进 §2 + 登记 §5；新陷阱进 §3；方法论进 §1。只写方法论（含最小示例和实测收益），不写单次任务流水账；新增时顺手精简过时条目。

---

## 1. 性能分析方法论

### 1.1 算计算强度（roofline）

`AI = FLOPs / Bytes`，与 `ridge_point = peak_FLOPS / peak_BW` 比：`AI < ridge` → memory-bound；`> ridge` → compute-bound。典型：GEMV(decode) memory-bound；GEMM(prefill) 随 batch 增长；elementwise/raster ≈ memory-bound。移动 GPU LPDDR 理论带宽打 6–8 折为实测可达。

### 1.2 memory-bound 时算 BW 利用率

`BW_util = actual_bytes / kernel_time / peak_BW`。<50% 访存模式/launch 有问题；50–70% cache miss/WG 大小可调；>70% 只能减数据量（packed 存储）。

### 1.3 occupancy / 寄存器 & 共享内存墙

roofline 说 memory-bound 但减流量/减 ALU 都不涨，甚至更慢 → 大概率是 occupancy 墙。Vulkan compute shader 的 occupancy 受**寄存器**和**共享内存（`shared` 数组）**双重限制：

- **共享内存直接吃 occupancy**：一个 workgroup 用 8KB `shared` 会显著减少常驻 workgroup 数。本仓库 conv1x1 融合 epilogue 引入 `shared[64*64]`（8KB）使大 N conv occupancy 下降、+148ms（见技巧 5 / 陷阱 I）。
- **寄存器 tile 有甜点、过宽断崖**：加宽 register-tile 摊薄加载，但过宽爆寄存器 occupancy 骤降。逐档实测。

### 1.4 ⚠️ Vulkan 第一诊断：CPU 调度 vs GPU kernel

**这是 Vulkan 区别于 OpenCL/Metal 的最关键一步。** MNN Vulkan 的 prefill 端到端常常**受 CPU 侧命令录制/调度瓶颈，而非 GPU kernel**。实测数据（Adreno，Qwen3-0.6B，514-token prefill）：

| 阶段 | 耗时 | 占比 |
|------|------|------|
| GPU 计算 | ~263ms | 60% |
| **命令录制（op onResize/onEncode ×1129）** | **~63ms** | 14% ← CPU 侧最大 |
| submit / 等待 | ~52ms | 12% |
| allocMemory 其余 + geometry + shape + 真显存分配 | ~47ms | 11% |

结论：**CPU 调度开销（~187ms）> GPU kernel 中可优化的部分**。所以 **attention GPU kernel 微优化对 prefill e2e 常测不出收益**（softmax kernel −38% 但 e2e 0%；见陷阱 F）。

**怎么判断**：用 `MNN_GPU_TIME_PROFILE=ON` 拿 GPU kernel 累计（op 级 `[Execution Profiling]` + shader 级 `[Shader Profileing]`），和干净 build 的端到端 wall 时间比：
- GPU 累计 ≈ wall（差一点点）→ **GPU-bound**，优化 kernel 有效。
- GPU 累计 ≪ wall（只占几分之一）→ **CPU 调度 bound**，优化 kernel 无效，改去攻调度（技巧 6/7 + §6）。

**profiler 读法坑**：两块 profiling 都**累计到程序退出**（含 decode + load 期 auto-tune 的 forward）。要稳态 prefill 数据，只解析 `Prepare for tuning opt End` 之后的 `[Execution Profiling]` 块并按 op 求和；op 级 "Attention/Convolution time" 与 shader 级同名 shader 求和大致吻合，可交叉校验。模型规模不同瓶颈会翻转：小模型（0.6B）CPU-bound，大 int4 模型（4B）conv GPU 占比大反而 GPU-bound。

### 1.5 标准流程

1. 先判 CPU 调度 vs GPU kernel（§1.4）。
2. GPU kernel 内再算 AI / BW 利用率 / occupancy。
3. 选 1 个杠杆（不要一次改两个）。
4. oracle 验正确性。
5. **交替 A/B** 测速（Android 真机，见 SKILL.md 性能测量；换 shader 每轮清 `mnn_cachefile.bin`）。
6. 没收益就 revert，记录"为什么不 work"。

---

## 2. 优化技巧

### 技巧 1：Cooperative matrix 重写 matmul（Adreno）

**针对瓶颈**：GEMM / QKV 的 compute + 访存。**适用**：Adreno 支持 `VK_KHR_cooperative_matrix`（`getCoopMatInfo().supportCoopMat`），K 维够大（headDim/hidden ≥ COOP_K 且能整除）。

**做法**：用 `coopmat<...>` + `coopMatLoad/coopMatMulAdd/coopMatStore` 替代 scalar 累加；COOP_M/N/K 用设备 `selectedFP16CoopMatShape`（Adreno 常 16×16×16 或 64×64×16）经 spec constant 传入；local_size_x = subgroup size，一个 workgroup 一个 subgroup。

**注意**：coop matrix **只对 int8/int4 有硬件定义**（陷阱 H）；shader 需 `--target-env vulkan1.1`（陷阱 D）；K 维太小时 shared staging 开销盖过收益（headDim=128 做 coop-QK 实测 e2e −3%，负收益，已弃）。

**收益参考**：attention QK·V 用 coop（`885be352de`），scalar `qkv_acc_full` 102ms → `coop_qkv` 54ms（−47%，Adreno prefill）。

---

### 技巧 2：Subgroup 归约替代 shared-memory tree reduction

**针对瓶颈**：归约类 kernel（softmax / reduce）里的 barrier 开销。**适用**：设备支持 `VK_SUBGROUP_FEATURE_BASIC|ARITHMETIC`（`getSubgroupInfo().ops`）。

**做法**：一个 workgroup = 一个 subgroup（`local_size_x_id=0` 设为 subgroup size），用 `subgroupMax/subgroupAdd` 替代"shared 数组 + barrier + stride 折半"的树形归约，去掉所有 reduction barrier。保留 tree-reduction pipeline 作 Mali/老 driver fallback。

**注意**：需 `#extension GL_KHR_shader_subgroup_arithmetic : require` + `--target-env vulkan1.1`；host 按 `getSubgroupSize()` 动态设 local size，别 hardcode。

**收益参考**：prefill online-softmax（`6ac7caa5f5`）28.79→17.87ms（−38% kernel）。⚠️ 但 prefill e2e 0%（CPU-bound，见 §1.4 / 陷阱 F）——收益体现在散热/长文脉/大模型，不在短 prefill e2e。

---

### 技巧 3：Epilogue 合并写（coalesced store）

**针对瓶颈**：memory（非合并写 = scatter → DRAM 流量爆炸）。**适用**：coop matmul / 任意把 tile 写成非行连续布局（如 NC4HW4 转置）的 epilogue。

**做法**：让**输出的连续维在相邻线程间最快变化**。NC4HW4 输出 `[N/4, M, 4]`（`Cvec4[gn4*M + gm]`，token 维 gm 连续），epilogue 循环里让 `m = idx % TILE_M`（gm 最快）而非 `m = idx / n4PerTile`（n4 最快）——后者相邻线程写地址跨步 `M`（=token 数），完全非合并。纯线程→元素重映射，输出逐字节不变。

**注意**：改后 shared 读可能变 strided（bank conflict），但 global 合并写的收益 ≫ smem bank conflict，通常不用补 padding（本仓库实测加 smem padding 无额外收益）。

**收益参考**：conv1x1 coop epilogue（`80e26673f`），大 N int4 conv 该 shader 从 scatter 恢复合并写，4B-c4 prefill −13%→回收约一半（Convolution op 1304→1144ms）。

---

### 技巧 4：融合 epilogue（把 unpack/scale 折进 matmul）

**针对瓶颈**：多余 dispatch + 中间 buffer 往返（Vulkan CPU 调度重，省 dispatch 收益大）。**适用**：matmul 后紧跟一个只读结果、逐元素变换再写出的独立 pass（unpack / 转置 / bias+relu / scale）。

**做法**：matmul 把 tile 存进 `shared`，`barrier()`，再直接按目标布局写出（含 bias/激活），省掉"写 temp → 独立 pass 读 temp"的一次 dispatch + 整个 temp buffer 往返。

**注意**：⚠️ 引入的 `shared` staging **会吃 occupancy**——小 N 净赚（省 dispatch/temp），**大 N 反伤**（occupancy 下降 > 省的调度，见技巧 5 + §1.3）。必须配合技巧 5 按规模门控。

**收益参考**：conv1x1 融合 COOP_to_C4（`45249b9c3`）+ scale_oacc 融进 coop QKV（`fba8885fb4`）：小模型（0.6B-c4）prefill +8%。

---

### 技巧 5：按输出规模门控融合 vs 分离路径

**针对瓶颈**：occupancy——同一融合优化对小/大问题规模收益相反。**适用**：技巧 4 的融合 epilogue 用了 `shared`，在大 N 上 occupancy 损失盖过省下的 dispatch。

**做法**：保留两条路径——小 N 走**融合**（零 temp，省 dispatch），大 N 走**分离**（matmul 写 row-major temp，**不用 shared → 高 occupancy** + 一个独立 unpack pass）。用 `padN` 阈值门控（env 可调）。阈值靠**逐 conv dump 出真实 N 分布**确定：本仓库 dump 出 2B 最大 conv N=6144 偏好融合、4B N=9728 偏好分离，故阈值取 8192 干净切分。

**注意**：阈值是设备/模型相关，做成 env 可调（`MNN_VK_CONV_FUSE_MAXN`）便于调优；两条路径都要过正确性。用 spec-constant 或两个 shader 变体实现分离路径的"零 shared matmul"。

**收益参考**：conv1x1 coop 按 N 门控（`80e26673f`），4B-c4 prefill −6%→**+2.6%**（超 baseline），同时 0.6B +9% / 2B +7% 全保住。

---

### 技巧 6：push_constant 替代小 uniform buffer

**针对瓶颈**：CPU 侧命令录制（per-op 的 `allocUniform`+map+memcpy+writeBuffer）。**适用**：uniform ≤128B（Vulkan push constant 常见 limit）的 op（raster 的 SamplerInfo/NCHWInfo、conv 的 M/N/K 等）。

**做法**：`layout(binding=N) uniform` → `layout(push_constant) uniform`，host 侧 `allocUniform+writeBuffer` → `vkCmdPushConstants`。pipeline layout 已内置 128B push constant range，无需改 layout；pipeline 的 `types` 去掉 `UNIFORM_BUFFER`。

**注意**：收益比预期小——实测 raster onResize 只降 15–25%，因为 uniform buffer 池化已较成熟、`vkAllocateDescriptorSets` 才是大头（见陷阱 J）。push_constant 本身近乎零开销，可作基础优化保留，但别指望单独它能显著提 e2e。

**收益参考**：raster/conv shader push_constant 化（worklog §6.13），raster onResize avg −15~25%，但端到端仅 +2%（噪声内）。

---

### 技巧 7：indirect batch 模式（多 op 共享 command buffer）

**针对瓶颈**：`vkQueueSubmit` 次数 + 每 op 独立 command buffer 的录制开销。**适用**：任何多 op 的 Vulkan 推理（LLM prefill/decode）。

**做法**：开 `MNN_GPU_RECORD_BATCH`（`ScheduleConfig::mode` 位 `0x200`），让 `mDirect=false`——1000+ 个 op 打包到少数 command buffer segment，submit 从 ~1129 次降到少数次。LLM 里通过 config `"vulkan_record_batch": true`（`llm.cpp initRuntime` 已默认开）。

**注意**：这是 host/调度级优化，不改 GPU kernel。是攻 §1.4 CPU 瓶颈的最大单一杠杆。

**收益参考**：Adreno Qwen3-0.6B prefill **+56%**、decode +8.7%（worklog §6.4）。

---

## 3. 常见陷阱

### 陷阱 A：全量 makeshader 污染 AllShader.cpp

本机 `glslangValidator`/`spirv-opt` 版本与仓库不同，跑全量 `makeshader.py` 会重编/重排**所有** shader 数组，`AllShader.cpp` 出现几万行无关 diff。**只外科式重生成改动的那几个数组**：单独用相同管线（header + body → glslangValidator `-V` [`--target-env vulkan1.1`] → spirv-opt `-O` → xxd）编译，正则替换 `AllShader.cpp` 里对应数组段（数组头到 `_len` 行），新增 shader 再补 `AllShader.h` + `VulkanShaderMap.cpp`。`git diff --stat` 确认只有目标数组变。

### 陷阱 B：持久化 pipeline cache 换 shader 后 stale → segfault

MNN 把编译好的 `VkPipelineCache` 序列化进 `tmp/mnn_cachefile.bin`。shader 变了之后，旧 cache 里的 pipeline 与新 SPIR-V 不匹配，加载时**直接 segfault**（不是报错）。改 shader 后测试/发布/换库做 A/B 前必须 `rm tmp/mnn_cachefile.bin`。升级发布时也要让用户清一次（或做 shader version 检测，TODO）。

### 陷阱 C：buffer/image 编译期二选一，改错树白改

`MNN_VULKAN_IMAGE` 决定走 `buffer/*` 还是 `image/*`，两棵独立代码树。Android `build_64.sh` 默认可能 `=ON`（image），desktop `MNN_VULKAN=ON` 默认 `=OFF`（buffer）。**在 Mac 上跑通的 buffer shader，Android 默认 image build 完全用不到**。改前 `grep MNN_VULKAN_IMAGE .../CMakeCache.txt`，确认改的路径与 build 一致。症状：改了 `.comp` 或加了 host log 完全不生效 → 先怀疑改错后端。

### 陷阱 D：coop / subgroup / memory_scope shader 需 target-env vulkan1.1

含 `GL_KHR_cooperative_matrix` / `GL_KHR_shader_subgroup*` / `GL_KHR_memory_scope_semantics` 的 shader 会 emit `SPV_KHR_vulkan_memory_model`（需 SPIR-V ≥ 1.3）。不加 `--target-env vulkan1.1`，glslang 默认 emit SPIR-V 1.0，spirv-opt 拒绝 → 静默回退到未优化 SPIR-V（或编译失败）。makeshader 已按扩展关键字判断，手动外科式重生成时务必带上。

### 陷阱 E：descriptor set 池化在 Adreno 上反变慢

CPU 侧 `vkAllocateDescriptorSets`（raster onResize 里占 82%、~44us/op）看似该池化复用。但实测在 Adreno driver 上，**descriptor set 被反复重绑到不同 buffer 会引入 GPU 侧 hazard 检测/stall**，decode −12%（净负），已回滚。结论：Adreno driver 对 descriptor set 复用不友好，池化不是免费午餐；要动这里需在目标 driver 实测。

### 陷阱 F：attention GPU kernel 微优化对 prefill e2e 常无效

prefill 受 CPU 调度瓶颈（§1.4），attention GPU kernel 只占 GPU 的一部分、GPU 又只占 e2e 一部分。多次实测：softmax kernel −38% / coop QKV kernel −47% 都 **e2e 0%**。**别为 prefill e2e 做 attention kernel 微优化**——要动 e2e 攻 CPU 调度（技巧 7 + §6）。kernel 收益体现在：散热、长文脉（softmax 占比放大）、大模型（GPU 占比大时才 GPU-bound）。动 kernel 前先按 §1.4 确认真的 GPU-bound。

### 陷阱 G：跨 session 顺序测 A/B 被热漂移欺骗

设备温度随时间漂 ~±8–10%，"先测完 base 再测 opt"（哪怕各取中位数）不可比——同一二进制隔一段时间重测能差 9%。**必须交替 A/B**（base→opt→base→opt 背靠背配对，看每轮胜负）。本仓库曾因顺序测把 4B 的真实 −15% regression 差点当噪声、把 0.6B 收益高估。

### 陷阱 H：cooperative matrix 只支持 int8/int4

coop matrix 用硬件 S8S8S32 / S4S8S32 指令，只对 int8/int4 weight 有定义。新加 w2/w3 时 host 必须显式跳过 coop 路径走 `VulkanConv1x1General`（native 低 bit gemv），否则用错 layout 解 buffer → 数值乱或 driver crash。

### 陷阱 I：shared 数组吃 occupancy

Vulkan compute 的 `shared` 数组直接限制常驻 workgroup 数。融合 epilogue 引入 8KB `shared` 在大 N conv 上使 occupancy 下降、净 +148ms（技巧 5）。加 `shared` 前评估 occupancy 影响，大问题规模考虑无-shared 的分离路径。

### 陷阱 J：CPU 侧真正大头是 vkAllocateDescriptorSets 不是 uniform

raster onResize 分步计时：`createSet`（`vkAllocateDescriptorSets`）占 82%（~44us/op），dispatch 11%，bind 5.5%，writeBuffer 0.6%，pushConstants ≈0。所以 push_constant 化（技巧 6）只能省一小部分；要大幅降 CPU 得攻 descriptor set（但池化在 Adreno 反变慢，陷阱 E）或跨 forward 复用命令（§6）。

### 陷阱 K：subgroup vs nosubgroup 双 shader 必须同步

每个 gemv/gemm/softmax 路径通常有 `_comp` 和 `_nosubgroup_comp` 两个变体（subgroup intrinsic 不可用时走后者）。新加 quant bit / 改逻辑**必须两个都改**，否则不支持 subgroup 的设备（老 Mali）挂或走错逻辑。

### 陷阱 L：coop 端数 tile 越界读靠 robustBufferAccess 兜底（UB）

coop `coopMatLoad` 按固定 COOP_M/N/K 块读，M（如 qLen/token）非块整数倍时末尾 tile 会读到 buffer 之外（此前靠 robustBufferAccess 返回 0 + store guard 兜底，输出对但是 UB）。修法：host 侧把对应 workspace 的 M 维 padding 到 COOP_M 倍数，让 coop load 恒在界内（本仓库 coop_qkv 的 W buffer 修复，`80e26673f`）。

### 陷阱 M：未扩展路径吃错 buffer 大小

buffer 大小按 `mIsInt4 ? padK/8 : padK/4` 算。新加 w2（`padK/16`）时若某分支漏改仍按 `padK/4`，会 OOB 读到下个 op 数据。dispatcher 改完搜所有以 `mIsInt4` 分流 buffer size/stride 的地方，同步加 `mIsInt2/3` 处理或 fallback。

### 陷阱 N：driver OOM / 设备重启

某些 driver 上每 op 常驻一组 GPU buffer（raw + decoded weight + temp），累积超 GPU 单进程 VRAM limit → 重启手机。跑大模型前先用小模型（<1B）验通路；大模型崩不要循环 retry，让设备恢复。0.6B 通 4B 崩 = buffer 总量问题；0.6B 也错 = kernel bug。

---

## 4. Packed weight 设计

新加 quant bit / 调 tile 排布时**先固定 5 个量**：tile（最小访问区块，Vulkan conv1x1 常见行主 `[N, padK/W]` 每 word 装 W 个 weight）/ word 内 weight 数（w2:16, w4:8, w8:4 per uint32）/ 多 word split（w3 用 lo2+hi1 双 word，`[N, padK/16, 2]`）/ bit 顺序（低 bit 先）/ signed 存储。

**signed/unsigned 与 originOffset**：导出器写出的 alpha `b = min + offset_signed*scale`，**originOffset 已折进 bias**。Vulkan int8 path 用 `bitfieldExtract(packedW,0,8)` **会 sign-extend**，signed bytes 直接当 signed 解，**无需再减 offset**（这点和 OpenCL/Metal 从 unsigned 解不同）。

**w3 split**：`[N, padK/16, 2]` uint pairs，pair[0] 低 16bit 装 16 个 2bit，pair[1] 低 16bit 装 16 个 1bit；decode `q = (low2>>(i*2))&3 | ((hi1>>i)&1)<<2`，减 4 得 signed[-4,3]。host buffer size 乘 `wordsPerGroup=2`，shader 内 stride 也 ×2。

---

## 5. 优化技巧速查表

| # | 技巧 | 针对瓶颈 | 适用场景 | 难度 | 收益参考 |
|---|------|---------|---------|------|---------|
| 1 | Cooperative matrix 重写 matmul | compute/访存（GEMM）| Adreno + coop，K 维够大 | 高 | QKV −47% kernel |
| 2 | Subgroup 归约替代 tree reduction | 归约 barrier | softmax/reduce，支持 subgroup | 中 | softmax −38% kernel |
| 3 | Epilogue 合并写（coalesced store）| memory（scatter 写）| NC4HW4/转置 epilogue 非合并 | 中 | 大 N conv 回收约一半 regression |
| 4 | 融合 epilogue（unpack/scale 折进 matmul）| dispatch + temp 往返 | matmul 后紧跟独立逐元素 pass | 中 | 小模型 prefill +8% |
| 5 | 按输出规模门控融合 vs 分离 | occupancy | 融合用 shared、大 N 反伤 | 中 | 4B −6%→+2.6% |
| 6 | push_constant 替代小 uniform | CPU 命令录制 | uniform ≤128B 的 op | 低 | raster onResize −15~25%（e2e 小）|
| 7 | indirect batch（RECORD_BATCH）| submit + 命令录制 | 多 op 推理 | 低 | prefill +56% |

> **先按 §1.4 分清 CPU 调度 vs GPU kernel**：CPU-bound 选 6/7 + §6；GPU-bound 选 1–5。

---

## 6. 候选优化手段（尚未在 MNN 验证 / 部分已排除）

> 有依据但未在 MNN 落地实测，或已实测排除。想用先做 spike 验证。

### 候选 A：fixResizeCache 跨 forward 复用命令（攻 CPU 调度的结构性大招）

**针对**：CPU 命令录制（~63ms/prefill）。同 shape 的第二次 forward 整段跳过 resize/encode。落地：(1) 让 `OpCommonUtils::supportDynamicInputMemory(MNN_FORWARD_VULKAN)` 在某 hint 下返回 true；(2) LLM 调 `setSessionMode(Session_Resize_Fix)`；(3) prefill 长度分桶 clone。预期省整个 `_allocForTensor`（~59ms）。风险中、收益结构性。**未落地。**

### 候选 B：算子融合减 op 数

**针对**：命令录制条目（1129 op 里 raster 占比大）。在 GeometryComputer 层合并 raster，或 Vulkan 后端把连续 raster 打包成一个 dispatch。工作量大。

### 候选 C：region 合并（CanCombine，OpenCL 已有）

**已排除（LLM 场景）**：实测 LLM raster 只有 1% 可合并（`max_run=2`），省 ~0.4ms，噪声内。OpenCL raster 比 Vulkan 快 10× 不是靠合并，而是每 kernel 更轻（setArg vs vkAllocateDescriptorSets）。CV 模型可能不同，实施前先 dump `shrink` 比例。

### 候选 D：descriptor set 池化

**已排除（Adreno）**：见陷阱 E，decode −12% 净负。其它 driver 可重试。

### 候选 E：coop matrix 用于 QK（不只 QKV）

**已排除（headDim=128）**：K 维太小，shared staging 开销盖过收益，e2e −3%。更大 headDim 或不同 shape 可重试。
