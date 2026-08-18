---
name: vulkan-optimize
description: MNN Vulkan 后端 kernel/算子性能优化与新特性集成。覆盖 benchmark 基线、kernel 优化迭代、集成验证全流程，以及 GLSL .comp + makeshader 双轨、conv1x1/attention dispatcher 多路径（coopMat/subgroup/nosubgroup）、cooperative matrix、packed weight 设计、pipeline cache、CPU 侧调度瓶颈、Adreno/Mali/Apple 多 vendor 真机验证等参考知识。
---

# MNN Vulkan 优化 Skill

> **触发**：优化 Vulkan 端 kernel 性能（conv/gemm/gemv/attention/elementwise 等），新增算子，调度选路或 pack layout 调整，或集成 Vulkan 新特性（cooperative matrix、subgroup、扩展等）。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## 核心原则

1. **改 `.comp` 必跑 makeshader**。GLSL 源不会被构建系统直接编译；运行时读的是 `AllShader.h/cpp` 里由 `makeshader.py`（`.comp` → `glslangValidator -V` → `spirv-opt -O` → `xxd` 嵌入）生成的 SPIR-V 字节数组。每次改 `.comp` 后必须重生成并确认进二进制。**最常见的"改了不生效"根因**。⚠️ 但**本机 glslang/spirv-opt 与仓库版本常不同，跑全量 makeshader 会重排/重编所有 shader 数组、污染 `AllShader.cpp`**——必须只外科式重生成改动的那几个数组（见「Shader 修改流程」+ 手册陷阱 A）。

2. **dispatcher 选路要先摸清**。Vulkan 同一个 op 常有多条 kernel：CoopMat（Adreno only）/ subgroup（Mali/Adreno）/ nosubgroup（兜底）/ FP gemm / 融合 vs 分离路径。改 kernel 前先读 host 选路代码，把目标 shape/dtype 代入确认落到哪一条。盲改经常发现根本没被调度。**不要把 fallback（nosubgroup）路径性能当 baseline**——它本身就慢。

3. **packed weight 必须 packing/unpack 双向镜像**。host weight prepare shader 写出的字节布局要和 decode shader 读取逐 bit 匹配，任何一边改动都要双边同步改（详见 `optimization-handbook.md` §4）。

4. **正确性 oracle 先于性能**。GPU 输出错和算法错难分辨。优化前要有"已知正确"的 baseline，每步改完都比对它。数学等价的 kernel 改动应与 baseline **逐 token 一致**。

5. **真机才算数，且面向多个 vendor**。Vulkan 同时跑 Adreno（Android）、Mali（Android）、Apple（iOS/MoltenVK）等，**driver 稳定性和性能差异极大**。Mac MoltenVK ≠ Android Adreno。进入性能/稳定性测试前先 `adb devices` 检查目标 Android 设备；没设备直接停下来给用户提需求。

6. **Buffer / Image 是编译期二选一**。由 `MNN_VULKAN_IMAGE` 决定：`=OFF` → `source/backend/vulkan/buffer/*`（LLM 全链路走 buffer）；`=ON` → `source/backend/vulkan/image/*`。**两个后端是完全独立的代码树**，一边改动另一边没有。改前先 `grep MNN_VULKAN_IMAGE project/android/build_64/CMakeCache.txt` 确认，并保证改的路径和 build 选的后端一致（详见手册陷阱 C）。

7. **先量化瓶颈，再选优化杠杆**。不要先猜再优化——用数字说话。**Vulkan 特有的第一诊断**：prefill 端到端常常**受 CPU 侧命令录制/调度瓶颈，而非 GPU kernel**（实测 attention GPU kernel 优化 −38% 但 e2e 0%）。动手前先分清瓶颈在 CPU 还是 GPU（方法论见 `optimization-handbook.md` §1）。

8. **单点突破，迭代验证**。每次只改一个优化点，验证正确性和性能后再下一个。

9. **换 shader 后必清 pipeline cache**。持久化的 `VkPipelineCache`（存在 `tmp/mnn_cachefile.bin`）在 shader 变了之后会 **stale → 直接 segfault**。改 shader 后测试/发布前先 `rm tmp/mnn_cachefile.bin`（详见手册陷阱 B）。

10. **新特性必须有示例代码 + fallback**。集成 coop matrix / subgroup / 扩展时：(1) 要求用户提供可运行示例——不同 vendor 对同一特性行为可能不同；(2) 用 runtime 特性检测分发，保留原路径做 fallback，不支持的设备不受影响。

---

## 执行流程

### 通用规则

1. **顺序执行**：工作流内的步骤不能跳过。
2. **允许回退**：对某阶段结果不满意可回到该阶段重做。
3. **每步验证**：每个步骤有明确通过标准，通过后才进入下一步。

### 选择优化方向

| 场景 | 方向 | 说明 |
|------|------|------|
| 用户要求提升整个模型的推理性能 | **A：模型级优化** | Profile 全模型 → 定位瓶颈（先分 CPU/GPU）→ kernel 级或算子级优化 → 重新 profile → 迭代 |
| 用户指定优化某个算子 | **B：指定算子优化** | 直接进入指定算子，判断 kernel 级或算子级 |
| 集成 Vulkan 新特性 / coop matrix / subgroup / 扩展 | **C：新特性集成** | 理解示例代码 → 适配 MNN → 特性检测 + fallback → 验证 |

---

### 方向 A：模型级优化

```
Profile 全模型（op 级 + shader 级耗时）
    ↓
先判断瓶颈在 CPU 调度 还是 GPU kernel（handbook §1.4）
    ├─ CPU 调度瓶颈 → 算子融合减 op / indirect batch / fixResizeCache（handbook §2/§6）
    └─ GPU kernel 瓶颈 → 定位耗时占比最高的 kernel/op
            ├─ Kernel 级 → Kernel 优化流程
            └─ 算子级 → 算子级优化流程
    ↓
重新 Profile 全模型，验证整体提升
    ↓
定位下一个瓶颈，重复直到满足需求
```

#### Step A.1：Profile 全模型

用 `-DMNN_GPU_TIME_PROFILE=ON` 编译（编译期宏，非运行时开关；命令见「编译与真机运行」）。Vulkan profiler 有**两块输出**：
- **`[Execution Profiling]`（op 级）**：按 op 类型（Convolution/Attention/Raster/…）聚合 GPU 时间。
- **`[Shader Profileing]`（shader 级）**：按具体 shader 名（`glsl_..._comp`）的 GPU timestamp。

⚠️ 两块都**跨 execute 累计到程序退出**，且含 load 期的 auto-tune forward——分析稳态时只取 `Prepare for tuning opt End` 之后的块（见 handbook §1）。

提取：每个 op/shader 的总耗时 + 占比排序、Top-N 瓶颈、端到端 prefill/decode tok/s 基线。

**通过标准**：有 op 级 + shader 级耗时排序，已确定第一个优化目标，并已判断瓶颈在 CPU 还是 GPU。

#### Step A.2：判断优化级别

| 信号 | 级别 | 进入流程 |
|------|------|---------|
| 端到端 ≫ GPU kernel 累计（GPU 只占几分之一）| **CPU 调度瓶颈** | 算子融合 / indirect batch / fixResizeCache（handbook §2, §6）|
| 单个 shader 占比高、本身有优化空间 | **Kernel 级** | → Kernel 优化流程 |
| 同一算子多个 shader 合计耗时高 / 有大量格式转换 | **算子级** | → 算子级优化流程 |
| 算子计算模式不适合 GPU（单 work-item 串行）| **算子级** | → 算子级优化流程 |

#### Step A.3：验证并迭代

重新 profile，确认瓶颈耗时下降 + 端到端提升；未满足则回 A.1 定位下一个瓶颈。

---

### 方向 B：用户指定算子优化

先 profile 算子内各 shader 耗时：
- **Kernel 级**：定位最慢 shader → Kernel 优化流程。
- **算子级**：跨 kernel 边界全局优化（合并/拆分 dispatch、改中间排布、融合 epilogue）→ 算子级优化流程。

---

### 算子级优化流程（方向 A/B 共用）

```
定位算子内所有 shader（读 Execution::onEncode/onResize）
    ↓
Profile 各 shader 耗时占比
    ↓
整体分析策略：
  - 合并 dispatch（减少命令录制 + 中间 buffer；Vulkan CPU 调度重，收益常更大）
  - 融合 epilogue（把 unpack/scale 折进 matmul，省一次 dispatch + temp）
  - 拆分（提高并行度）
  - 改中间数据排布（消除格式转换 / raster）
  - 用 cooperative matrix 重写 matmul
    ↓
迭代验证，直到算子整体性能满足需求
```

| 手段 | 适用场景 | 本仓库案例 |
|------|---------|-----------|
| **融合 epilogue** | matmul 后有独立 unpack/转置 pass | conv1x1 coop 把 COOP_to_C4 折进 matmul epilogue（小 N 收益，大 N 反伤 occupancy → 按 N 门控，见 handbook 技巧 3/5）|
| **coop matrix 重写** | GEMM/QKV，Adreno 支持 coop | attention QK·V 用 coop（scalar qkv_acc 102ms → coop 54ms）|
| **subgroup 归约** | 有 tree reduction + barrier | prefill softmax 用 subgroupMax/Add 替代树形（37→23ms）|
| **合并 dispatch / indirect batch** | 命令录制占大头 | `MNN_GPU_RECORD_BATCH` 让多 op 共享 command buffer（prefill +56%）|

> **Vulkan 关键提醒**：算子级优化前先确认瓶颈是 GPU 还是 CPU 调度。若是 CPU 调度，GPU kernel 再快端到端也不动（见 handbook §1.4 + 陷阱 F）。

---

### Kernel 优化流程（方向 A/B 共用）

```
分析 shader 的计算强度 / BW 利用率（handbook §1）
    ↓
判断瓶颈类型（compute / memory / occupancy）
    ↓
从 §2 技巧 + §5 速查表选匹配方法
    ↓
实施 → 验证正确性 → 测量性能（交替 A/B）
    ↓
未达预期？换一种技巧重试
```

| 阶段 | 文档 | 目标 |
|------|------|------|
| 基准 | `benchmark.md` | 建立性能基准，分析瓶颈类型 |
| 优化 | `kernel-opt.md` | 至少尝试 3 种技术，迭代提升 |
| 集成 | `integrate.md` | 全量回归、代码审查、性能报告 |

---

### 方向 C：新特性集成

| 文档 | 目标 |
|------|------|
| `new-feature.md` | 理解示例 → 评估兼容性 → 适配 MNN + 特性检测 → fallback → 验证 |

**触发**：用户说"用这个 Vulkan 特性/coop matrix/subgroup 扩展"并提供示例代码。没提供示例时**主动要求**。

---

### 收尾：沉淀经验到手册

**完成一个较复杂的优化任务后**（重写算子、新技巧奏效、踩了非显而易见的坑、验证了/排除了某方向），把**可复用的方法论**回写到 `optimization-handbook.md`——它是 Vulkan kernel/访存/调度级技巧与陷阱的**唯一来源**。

| 信号 | 回写到 |
|------|--------|
| 新的可迁移优化手法 | §2 新增技巧 + 登记 §5 速查表 |
| 别人也会踩、非显而易见的坑 | §3 新增陷阱 |
| 新的瓶颈定位/测量方法论 | §1 |
| 验证通过某 §6 候选方向 | 从 §6 移入 §2 + §5，补实测收益 |
| 实测排除了某方向 | 在 §6 标注「已排除 + 原因」，避免重复尝试 |

**回写要求**：只写方法论（适用场景/做法/注意/最小示例/实测收益），不写单次任务流水账；技巧编号是稳定 ID（只追加不复用）；新增时顺手精简过时条目；**某具体 shape/kernel 的一次性发现属于 agent memory，不进手册**。若本次无可复用经验，**跳过回写**。这一步与 `retrospective` skill 互补。

---

## 参考知识

### 编译与真机运行（Android，唯一来源）

**编译**（`project/android/build_64/`，Vulkan buffer 后端 + LLM，`SEP_BUILD=OFF` 时 Vulkan 编进 `libMNN.so`）：

```bash
cd project/android/build_64
# 首次配置（确认 buffer 后端 + LLM）
cmake .. -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF -DMNN_BUILD_LLM=ON \
         -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_LOW_MEMORY=ON -DMNN_ARM82=ON
make llm_demo -j8            # SEP_BUILD=OFF → 会重编 libMNN.so
# 需要 GPU per-op/shader 耗时：cmake -DMNN_GPU_TIME_PROFILE=ON .（测干净 tok/s 时务必 =OFF）
```

> `make MNN` 不会连带重编 Vulkan 静态库；用 `make llm_demo`。改 shader 后 `AllShader.cpp` 变了也会重编。

**推送 + 运行**：

```bash
adb push libMNN.so llm_demo /data/local/tmp/MNN/
adb -s <serial> shell "cd /data/local/tmp/MNN && rm -f tmp/mnn_cachefile.bin && \
  LD_LIBRARY_PATH=. ./llm_demo <model>/config_vk.json <prompt.txt> <ndecode>"
```

> **换 shader 后必清 `tmp/mnn_cachefile.bin`**（原则 9）；否则 stale pipeline cache 会 segfault。

### 入口定位

```bash
grep -rn "OpType_<MyOp>" source/backend/vulkan/buffer/execution/   # buffer 后端
grep -rn "OpType_<MyOp>" source/backend/vulkan/image/execution/    # image 后端
```

conv1x1 低 bit 选路（`VulkanConvolution.cpp::onCreate`）：

```
useInt8Conv && is1x1
  ├─ coopMat supported && Adreno
  │    ├─ perChannelAsym + S8S8S32 → VulkanConv1x1CoopA8
  │    └─ else → VulkanConv1x1Coop      (CoopMat 只支持 int4/int8)
  └─ → VulkanConv1x1General             (native int8/int4/int2/int3)
else → VulkanConvolutionSlideWindowsInt8
```

attention prefill 走 `VulkanAttention.cpp`：rearrange_q → init_state → (per k-block: qk → softmax → qkv) → finalize，coop QKV / subgroup softmax 由设备能力选路。每个 `onEncode` 决定本次 dispatch 哪些 shader，把目标 shape 代入确认，再改对应 `.comp`。

### Shader 修改流程（含防污染）

```bash
# 1) 编辑 .comp（确认 buffer 还是 image 后端）
vi source/backend/vulkan/buffer/execution/glsl/<my_kernel>.comp
# 2) 新文件在 glsl/macro.json 登记 useFP16（决定是否生成 _FP16 变体）
# 3) 重新生成 SPIR-V 嵌入数组
```

⚠️ **不要直接跑全量 `makeshader.py`**——本机 glslang/spirv-opt 与仓库版本不同，会把所有 shader 数组重编/重排，`AllShader.cpp` 出现几万行无关 diff（污染）。**正确做法：只重生成改动的那几个数组，外科式替换进 `AllShader.cpp`**：

```bash
# 对每个改动的 shader（fp32 + fp16 变体），用与 makeshader 相同的管线单独编译：
#   header(FP32/FP16) + body → glslangValidator -V [--target-env vulkan1.1(若含 coop/subgroup/memory_scope)] → spirv-opt -O → xxd -i
# 得到 const unsigned char glsl_<name>_comp[]={...}; unsigned int glsl_<name>_comp_len=N;
# 再用脚本精确替换 AllShader.cpp 里对应的那一段（正则匹配数组头到 _len 行）。
# 新增 shader 还要在 AllShader.h（extern 声明）+ VulkanShaderMap.cpp（name→数组 map）追加。
```

改完确认：`grep -c 'glsl_<name>_comp_len' AllShader.cpp` 且用 `git diff --stat` 确认**只有目标数组变了**（无关 `_len` 行不应出现在 diff）。SPIR-V 合法性可 `spirv-val` 验证。

**新加 kernel 同步检查清单**：`.comp` 主路径 + `_nosubgroup` 变体都加 / `macro.json` 登记 / host pipeline 选择分支 / `AllShader.{cpp,h}` + `VulkanShaderMap.cpp` 三处注册 / weight stride & buffer size 重算 / dispatcher 显式选路或 fallback。

### 正确性验证

三层 oracle（数值层 dump tensor → op 层 `MNNV2Basic.out` 单层 → 端到端跑模型）。端到端**关 sampler 随机性**（`temperature:0.0`, greedy），CPU/Vulkan 同 prompt 前 N token 应一致（fp16 误差内）。

**CPU oracle 不可用时的兜底**（低 bit 路径 CPU 本身就乱）：用**冻结 baseline 二进制**做 GPU-baseline vs GPU-opt 的逐 token 贪心对比（数学等价改动应完全一致）。改 kernel 前先 `git stash` 存一份 baseline `libMNN.so`。

**模型本身可能就坏**：小模型极低 bit 量化 CPU 跑也乱（如 Qwen3-0.6B 的 w2/w3），用 4B/8B 才是有效验证样本。**Mac MoltenVK 行为不代表 Android**，最终验收必须 Android 真机。数值容忍：fp16 vs fp16 abs<1e-2；量化 dequant+fp16 abs<1e-1。

### 性能测量（Android 真机）

小收益最容易被噪声骗。手机 GPU 两个陷阱：

1. **冷启动首跑不算数**：首跑含 auto-tune + pipeline 编译。清了 `mnn_cachefile.bin` 后第一次也是冷的。**先 warm 再测**。
2. **跨时段热漂移 ~±8–10%**：设备温度随时间变，"先测完 base 再测 opt"不可比。**必须交替 A/B**——base 和 opt 两份二进制（或同一二进制不同 env）背靠背配对：`base→opt→base→opt`，看**每轮配对**里 opt 是否稳定胜出，而非比两组绝对值。本仓库多次踩过跨 session 顺序测把噪声当收益的坑（见 handbook §1）。

> 换 shader 做 A/B 时，每次切库 pipeline cache 会 stale（原则 9）；**每轮跑前 `rm tmp/mnn_cachefile.bin`**（tok/s 不含 load，清 cache 只影响 load 时间，A/B 仍公平）。

带 `MNN_GPU_TIME_PROFILE=ON` 的 build 只看相对占比（放大绝对耗时且改变调度）；**最终收益以不带 profile 的干净 build 的端到端 tok/s 为准**。

### 优化技巧与常见陷阱

详见 `optimization-handbook.md`：
- **§1 性能分析方法论**：roofline + **CPU 调度 vs GPU kernel 的先判断**（Vulkan prefill 关键）；两块 profiler 的读法。
- **§2 优化技巧**：coop matrix、subgroup 归约、epilogue 合并写（coalescing）、按 N 门控融合、融合 epilogue/scale、push_constant、indirect batch 等，每条含适用场景/做法/注意/收益。
- **§3 常见陷阱**：makeshader 污染、pipeline cache stale、buffer/image、coop target-env、descriptor pool 在 Adreno 反变慢、attention 微优化对 prefill e2e 无效、热漂移、coop 只支持 int4/8、subgroup/nosubgroup 双 shader、weight pack 5 处同步、driver OOM 等。
- **§4 Packed weight 设计**、**§5 速查表**、**§6 候选方向**（fixResizeCache 跨 forward 复用命令等）。

### 新特性集成参考

Vulkan 设备能力查询在 `VulkanDevice`：`getCoopMatInfo().supportCoopMat` / `getSubgroupInfo()` / `getSubgroupSize()` / `getCoopMatInfo().selectedFP16CoopMatShape`。新特性用 spec constant（`layout(constant_id=N)`，host 经 `getPipeline(name, types, localSize, spec)` 传入）或 build 宏控制路径 + runtime 检测 fallback。coop/subgroup shader 需 `--target-env vulkan1.1`（makeshader 已按扩展关键字判断）。详见 `new-feature.md`。
