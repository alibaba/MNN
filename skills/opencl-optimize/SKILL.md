---
name: opencl-optimize
description: MNN OpenCL 后端 kernel 性能优化与新特性集成。覆盖 benchmark 基线、kernel 优化迭代、集成验证全流程，以及新 OpenCL 特性适配、.cl + codegen 双轨、kernel 选路、packed weight 设计、tune 机制、Android 真机验证等参考知识。
---

# MNN OpenCL 优化 Skill

> **触发**：优化 OpenCL 端 kernel 性能（conv/gemm/gemv/attention/elementwise 等），新增算子，调度选路或 pack layout 调整，或集成 OpenCL 新特性（扩展、新指令、vendor extension 等）。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

## 核心原则

1. **改 `.cl` 必跑 codegen**。`.cl` 源不会被构建系统直接编译；运行时读的是 `*_mnn_cl.cpp` 嵌入字符串。每次改 `.cl` 后跑 codegen，并确认进二进制。**最常见的"改了不生效"根因**。
2. **dispatcher 选路要先摸清**。OpenCL 同一个 op 常有多条 kernel。改 kernel 前先读 host 选路代码，确认目标 shape/dtype 落到哪一条。盲改经常发现根本没被调度。
3. **packed weight 必须 packing/unpack 双向镜像**。host 写出的字节布局要和 shader 读的逐 bit 匹配，任何一边改动都要双边同步改。
4. **正确性 oracle 先于性能**。GPU 输出错和算法错难以分辨。优化前要有"已知正确"的 baseline，每一步改完都比对它。
5. **真机才算数（Android）**。OpenCL 主要面向 Android 手机 GPU（Adreno / Mali）。Mac/desktop OpenCL 行为差异大，不能代表手机性能或稳定性。没设备时直接停下来给用户提需求。
6. **先量化瓶颈，再选优化杠杆**。不要先猜瓶颈再优化——用数字说话（方法论见 `optimization-handbook.md` §1）。
7. **单点突破，迭代验证**。每次只改一个优化点，验证正确性和性能后再进行下一个。
8. **新特性必须有示例代码 + fallback**。集成 OpenCL 新特性（扩展、vendor extension、新指令等）时：(1) **必须要求用户提供可运行的示例代码**——不同 GPU 厂商对同一扩展行为可能不同，别凭文档猜；(2) **必须保留原有路径作为 fallback**，通过 runtime 特性检测分发，确保不支持的设备功能不受影响。
9. **单测必须确认实际执行了目标内存模式**。只有 Buffer Execution 的算子，仅指定 `backend=3` 仍可能走 Image 模式并回退 CPU，形成假通过。测试时显式设置 Buffer mode（例如 `run_test.out <case> 3 2 68`），并用 profiler、后端日志或明显的 GPU 同步耗时确认没有 fallback。

---

## 执行流程

### 通用规则

1. **顺序执行**：工作流内的步骤不能跳过
2. **允许回退**：对某阶段结果不满意可回到该阶段重做
3. **每步验证**：每个步骤有明确的通过标准，通过后才进入下一步

（"单点突破——每次只改一个优化点"见核心原则 7,此处不再重复。）

### 选择优化方向

根据任务类型，选择对应的方向进入：

| 场景 | 方向 | 说明 |
|------|------|------|
| 用户要求提升整个模型的推理性能 | **A：模型级优化** | Profile 全模型 → 定位瓶颈 → 判断 kernel 级或算子级优化 → 重新 profile → 迭代 |
| 用户指定优化某个算子 | **B：指定算子优化** | 直接进入指定算子，判断 kernel 级或算子级优化 |
| 集成 OpenCL 新特性 / 扩展 / vendor extension | **C：新特性集成** | 理解示例代码 → 适配 MNN → 实现 fallback → 验证 |

> 方向 C 完成后，如需进一步优化该特性路径的性能，可衔接方向 A 或 B。

---

### 方向 A：模型级优化

用户目标是提升整个模型的端到端性能。通过 profile 定位瓶颈，逐个击破。

#### 整体流程

```
Profile 全模型（获取所有 kernel 耗时）
    ↓
定位耗时占比最高的瓶颈
    ↓
判断优化级别：
  ├─ Kernel 级：瓶颈在单个 kernel → Kernel 优化流程
  └─ 算子级：瓶颈在算子整体设计 → 算子级优化流程
    ↓
重新 Profile 全模型，验证整体性能提升
    ↓
定位下一个瓶颈
    ↓
重复，直到满足用户的性能需求
```

#### Step A.1：Profile 全模型

打开 MNN 的 profile 功能，统计模型所有 kernel 的耗时。需要用 `MNN_GPU_TIME_PROFILE=ON` 编译（编译期宏，不是运行时开关）——编译/推送命令见「编译与真机运行」，端到端跑法见「正确性验证 · 真机测试入口」，都要带 `-DMNN_GPU_TIME_PROFILE=ON`。

从 profile 输出中提取：
- 每个 kernel 的名称、调用次数、总耗时、平均耗时
- 耗时占比排序，定位 Top-N kernel
- 端到端的 prefill tok/s 和 decode tok/s 基线

**通过标准**：有完整的 kernel 耗时排序表，已确定第一个优化目标。

#### Step A.2：判断优化级别

分析瓶颈所在，选择优化粒度：

| 信号 | 优化级别 | 进入流程 |
|------|---------|---------|
| 单个 kernel 耗时占比高，kernel 本身有优化空间 | **Kernel 级** | → Kernel 优化流程 |
| 同一算子的多个 kernel 合计耗时高 | **算子级** | → 算子级优化流程 |
| 算子内 kernel 间有大量格式转换 / raster 开销 | **算子级** | → 算子级优化流程 |
| 算子的计算模式不适合 GPU（如单 work-item 串行） | **算子级** | → 算子级优化流程 |

#### Step A.3：验证并迭代

1. 优化完成后，重新 Profile 全模型
2. 确认瓶颈耗时下降，端到端性能提升
3. 如果用户性能需求未满足，回到 Step A.1 定位下一个瓶颈
4. 重复直到达到用户目标或主要瓶颈已全部优化

---

### 方向 B：用户指定算子优化

用户直接指定优化某个算子（如 LinearAttention、Conv、MatMul 等）。同样需要先判断优化级别：

- **Kernel 级**：Profile 算子内各 kernel 耗时，定位最慢的 kernel → Kernel 优化流程
- **算子级**：需要跨 kernel 边界做全局优化 → 算子级优化流程

---

### 算子级优化流程（方向 A/B 共用）

将整个算子作为优化单元，不局限于单个 kernel 的改动。

#### 整体流程

```
定位算子内所有 kernel（读 Execution::onResize）
    ↓
Profile 算子内各 kernel 耗时占比
    ↓
整体分析优化策略：
  - 合并 kernel（减少 dispatch + 中间 buffer）
  - 拆分 kernel（提高并行度）
  - 改变中间数据排布（消除格式转换开销）
  - 重写算子整体实现
    ↓
迭代验证，直到算子整体性能满足需求
```

#### 算子级特有的优化手段

跨 kernel 边界做全局决策：

| 手段 | 适用场景 | 示例 |
|------|---------|------|
| **合并 kernel** | 多个小 kernel 串行执行，launch overhead 占比大 | 将 Q/K/V 三次 GEMV 合并为一次 |
| **拆分 kernel** | 单 kernel 内部串行依赖过重，并行度不足 | 将 RNN 式时间步循环拆为并行的维度 |
| **改中间数据排布** | kernel 间的数据传递格式不一致导致额外 raster | 统一使用 NC4HW4 避免中间转换 |
| **算子整体重写** | 现有实现的计算模式根本不适合 GPU | LinearAttention：从单 work-item 改为 workgroup 并行（112x）|

> **参考案例**：LinearAttention 优化（`6a361d3e4`）——原实现用单个 work-item 处理一整个 (batch, head)，导致大量私有数组和寄存器溢出。重写为 2D workgroup 并行 + local memory + 并行 reduce，decode 263x、prefill 60x。详见 `optimization-handbook.md` 技巧 4。

#### 验证标准

- 算子整体输出与 CPU baseline 一致（参考"正确性验证"章节）
- 算子整体耗时优于优化前

---

### Kernel 优化流程（方向 A/B 共用）

不管是模型级还是算子级优化，最终都会落到单个 kernel 的优化。单个 kernel 的优化流程：

```
分析 kernel 的计算强度和 BW 利用率（optimization-handbook.md §1）
    ↓
判断瓶颈类型（compute-bound / memory-bound）
    ↓
从优化技巧中选择匹配的方法（optimization-handbook.md §2 + §5 速查表）
    ↓
实施优化 → 验证正确性 → 测量性能
    ↓
未达预期？换一种技巧重试
```

#### 选择优化方式

| 方式 | 适用场景 | 流程 |
|------|---------|------|
| **框架内优化** | 改动较小，或需要端到端验证 | benchmark → kernel-opt → integrate |
| **Demo 驱动优化** | 需要大量迭代，避免反复编译整个 MNN | demo → kernel-opt → integrate |

#### 方式一：框架内优化

直接在 MNN 框架内修改 kernel 并测试。

| 阶段 | 文档 | 目标 | 复杂度 |
|------|------|------|--------|
| 基准 | `benchmark.md` | 建立性能基准：获取基线数据，分析计算强度和瓶颈类型 | 低 |
| 优化 | `kernel-opt.md` | 优化 kernel：至少尝试 3 种不同优化技术，迭代提升性能 | 高 |
| 集成 | `integrate.md` | 验证集成：全量回归测试、代码质量审查、性能报告 | 中 |

每个阶段的详细操作、代码模板和通过标准见对应文档。

#### 方式二：Demo 驱动优化

**适用场景**：需要频繁改 kernel 测性能，不想每次编译整个 MNN。用**独立 demo** 代替方式一的 `benchmark.md` 阶段建立基线，之后的迭代 / 集成复用同样的文档。

| 阶段 | 文档 | 差异 |
|------|------|------|
| 基准 | 本节（建立 demo） | 用独立 demo 记录 baseline 输出+性能，替代 `benchmark.md` |
| 优化 | `kernel-opt.md` | 在 demo 内迭代，每次与 baseline **逐元素对比**；至少 3 种技术 |
| 集成 | `integrate.md` | 优化 kernel 换回 MNN 原 `.cl` → codegen → 全量回归 |

**建立独立 Kernel Demo**（放 `tools/opencl_bench/` 或用户指定目录）：

1. 定位目标 `.cl` 与 kernel 函数名，记录其 args、数据类型、GWS/LWS
2. 写独立程序：初始化 context/queue → 备输入（模型 dump 或固定 seed 随机）→ 直接 include `.cl` 源编译 → 设与 MNN 相同的 args/GWS/LWS → 跑并存 **baseline 输出与性能**（多次取中位数）
3. 验证 demo 输出与 MNN 中同一 kernel 一致（允许 fp16 误差）

**通过标准**：demo 可独立编译运行，baseline 已保存。之后按 `kernel-opt.md` 迭代（正确性硬约束：与 baseline 不一致的方案直接废弃），按 `integrate.md` 合入验证。

---

### 方向 C：新特性集成

将用户提供的 OpenCL 新特性示例代码适配并集成到 MNN 中。

| 文档 | 目标 | 复杂度 |
|------|------|--------|
| `new-feature.md` | 理解示例代码 → 评估兼容性 → 适配 MNN 架构 → 实现 fallback → 验证 | 中-高 |

**触发条件**：用户说"用这个 OpenCL 特性"、"集成这个扩展"、"把这个示例代码融入 MNN"等，并提供了示例代码或参考实现。如果用户没有提供示例代码，**主动要求用户给出示例代码**。

详细操作和通过标准见 `new-feature.md`。

---

### 收尾：沉淀经验到手册

**完成一个较复杂的优化任务后**（重写算子、新技巧奏效、踩了非显而易见的坑、验证了/排除了某个候选方向），把**可复用的方法论**回写到 `optimization-handbook.md`——它是 OpenCL kernel/访存级技巧与陷阱的**唯一来源**。这样下一个任务能直接站在本次经验上。

**触发判断**（满足任一即回写）：

| 信号 | 回写到 |
|------|--------|
| 发现一种新的、可迁移到其他 kernel 的优化手法 | §2 新增技巧 + 登记 §5 速查表（含「针对瓶颈」列）|
| 踩了一个别人也会踩、非显而易见的坑 | §3 新增陷阱 |
| 总结出新的瓶颈定位/测量方法论 | §1 |
| 验证通过某个 §6 候选方向 | 从 §6「毕业」移入 §2 + §5，补实测收益 |
| 实测排除了某个方向（证明无效） | 在 §6 对应条目标注「已排除 + 原因」，避免重复尝试 |

**回写要求**：

- 只写**方法论**（适用场景、做法、注意事项、最小示例、实测收益），**不写单次任务的流水账**；单次任务的具体数字/文件清单留在性能报告里。
- 技巧编号是稳定 ID（连续 1-N），只追加不复用旧号；其他文件（`kernel-opt.md` 等）只**引用不复制**。
- 新增时**顺手精简**过时或显而易见的条目，避免手册膨胀。
- 只收 **kernel/访存级**经验；host/init 级（零拷贝、mmap、kernel 预编译、tune 启发式等）不在本手册范围。
- 若沉淀出的是**某个具体 shape/kernel 的一次性发现**（而非可迁移方法论），它属于 agent memory，不进手册。

> 若本次任务无可复用经验（纯粹套用已有技巧、无新坑），**跳过回写**，不要为凑数往手册塞内容。这一步与 `retrospective` skill 互补：retrospective 沉淀跨技能的通用教训，本步只沉淀 OpenCL kernel 级技巧。

---

## 参考知识

以下内容是执行流程中各步骤会用到的参考资料，按主题组织。

### 编译与真机运行（通用，唯一来源）

各步骤反复用到的「编译 → 推送 → 运行」统一在此；其它文件只写与本步不同的部分（测哪个 case、是否开 profile）。

**编译**（Android 交叉编译，`project/android/build_64/`）：

```bash
cd project/android/build_64
../build_64.sh \
  -DMNN_BUILD_TEST=ON -DMNN_ARM82=ON -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_OPENCL=ON \
  -DMNN_GPU_TIME_PROFILE=ON   # 仅在需要 kernel 级耗时时加；测干净的端到端 tok/s 时务必去掉
```

**推送**：

```bash
../updateTest.sh   # 推 run_test.out + 库（单测 / speed 用）
# 端到端用 llm_demo 时手动推：
adb push build_64/{libMNN.so,libMNN_Express.so,libllm.so,llm_demo} /data/local/tmp/MNN/
```

**运行单测 / speed**：

```bash
adb shell "cd /data/local/tmp/MNN && LD_LIBRARY_PATH=. ./run_test.out <case> 3 <precision> <numThread>"
# 3 = OpenCL；precision 1=fp32 / 2=fp16；numThread 见下
```

> **numThread 含义**：`+64` 强制 buffer 模式，`+0` 为 auto（Adreno 默认 image，Mali 默认 buffer）。**强制 buffer 只适用于 LLM**（全链路 buffer，`68 = 64+4`）；优化非 LLM 算子时按其生产模式测（Adreno 上常为 image，用 `4`），否则会落到没被调度的 kernel 路径（buffer/image 是两套 Execution，见原则 2）。

端到端（`llm_demo`）的完整命令见下方「正确性验证 · 真机测试入口」。

---

### 入口定位

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

**低 bit prefill GEMM 有两条竞争路径**：native-int4 kernel（`gemm_b*_c8_int4_buf`，边算边反量化）vs 反量化+FP-GEMM（`useFPWeightGemmLowMemory`，先解成 fp weight 再走通用 GEMM）。`onResize` 按 shape **tune 择优并缓存决策**（大 batch 常选 FP 路径）。改 native kernel 前先确认目标 shape 真的走它——最省事是临时加一行 `MNN_PRINT` 打印 kernelName，别盲改发现根本没被调度。

> 抽 offset helper 时当心「布局 helper 和调用点重复累加偏移」——见 `optimization-handbook.md` 陷阱 N。

---

### .cl 修改流程

```bash
# 1) 编辑 .cl
vi source/backend/opencl/execution/cl/<my_kernel>.cl

# 2) 重新生成 _mnn_cl.cpp（必跑）
cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . .

# 3) 验证嵌入
grep -c '<新宏名>' <my_kernel>_mnn_cl.cpp     # > 0 才算进
strings build/.../libMNN.so | grep '<新宏名>' # build 后再确认

# 4) build（命令见「编译与真机运行」）
```

**新加 `QUANT_BIT==N` 时通常每个 shader 有 4 处都要加分支**（不是 1 处）：

| 位置 | 含义 |
|---|---|
| WGS>=8 主循环 | `useLocalMem=true`，IC >= 32 |
| WGS>=8 leaves | `INPUT_CHANNEL_LEAVES_NUM != 0` 时尾部 |
| WGS<8 主循环 | 单线程 reduce，IC < 32 |
| WGS<8 leaves | 同上的尾部 |

改完用 `grep -n "QUANT_BIT == 4" <file>.cl` 数 N 个，确认 `== 2` 也有 N 个（这 5 处 pack-size 同步见 `optimization-handbook.md` 陷阱 B）。

> 两个相关的宏/编译器坑见 `optimization-handbook.md`：**陷阱 O**（build option 宏名拼错静默走 `#else` → 数值错）、**陷阱 P**（unpack 用 `#define` 不用 inline function，Adreno 老编译器不稳）。

**codegen 会重新生成所有 `*_mnn_cl.cpp`**，git diff 看到无关文件也变属正常，正常提交即可。

---

### 正确性验证

GPU 输出"乱"容易被错怪成模型问题或 sampler 问题，性能调优前必须独立完成。

**三层 oracle，从近到远**：

| 层级 | oracle | 检验点 |
|---|---|---|
| 数值层 | CPU 跑同一 op，dump tensor | 单 op fp16 误差 < 1e-2 |
| op 层 | `MNNV2Basic.out` 单层 conv | 输出与 CPU 对齐 |
| 端到端 | 跑模型 | 文本/输出语义合理 |

发现端到端乱码时**先回到数值层**，不要直接调端到端 sampler。

**端到端必须关 sampler 随机性**：`temperature: 0.0`, `sampler_type: greedy`，CPU/GPU 同 prompt + 同 config 前 N 个 token 应完全相同。注意 LLM config 的 `thread_num` 对 OpenCL 是 buffer/image 模式编码（如 516=512+4），直接拿去跑 CPU 会当成线程数而输出乱码——切 CPU 时改回正常值（如 4）。

**模型本身可能就坏**：小模型在极低 bit 上量化退化常见，CPU 跑也乱。GPU 验证前先 baseline CPU。

**CPU oracle 不可用时的兜底**：若 CPU 后端对目标模型直接输出乱码（低比特路径 bug / 配置问题），改用**冻结的 baseline 二进制**做参考——`git stash` 掉本次改动、重建一份 baseline `libMNN.so` 存到 `/tmp`，再和优化版做 **GPU-baseline vs GPU-opt 的逐 token 贪心对比**（数学等价的 kernel 改动应完全一致）。这份 baseline 二进制同时用于下面的性能 A/B。改 kernel 前就先存一份 baseline 二进制。

**数值偏差容忍**：

| 路径 | 容忍误差 |
|---|---|
| fp32 vs fp32 | abs < 1e-5, rel < 1e-4 |
| fp16 vs fp16 | abs < 1e-2, rel < 5e-3 |
| 量化 dequant + fp16 | abs < 1e-1, rel < 1e-2 |

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

**设备掉线 / 重启**：跑较大模型后手机可能 hang。`adb devices` 列表为空就**等设备回来再继续**，不要循环 retry。如果是后端 buffer 总量超 GPU 单进程 limit（Adreno 典型），是后端架构问题，先换小模型继续。

---

### 性能测量（Android 真机）

小收益（几 %）最容易被测量噪声骗。手机 GPU 有两个陷阱，"跑 N 次取中位数"并不足以规避：

1. **冷启动首跑不算数**：首跑把 auto-tune + kernel 编译都算进耗时（LLM prefill 实测 31 vs warm 1280 tok/s，40×）。**必须先 warm（跑 1 次丢弃），再测**。清了 `mnn_cachefile.bin` 后第一次也是冷的。
2. **跨时段热漂移 ~8%**：设备温度随时间变，"先测完 base 再测 opt"不可比。而同一次连续跑的方差只有 ±0.3%，**紧凑的数字极具迷惑性**，会让你把噪声级差异当成真实收益。

**正确姿势——交替 A/B**：保留 base 和 opt 两份二进制，`base→opt→base→opt` 背靠背配对推送+运行，丢弃冷启动首轮，看**每轮配对**里 opt 是否稳定胜出（例如 5/5 轮都赢才算数），而不是比两组的绝对值。

**注意换库会抖乱 tune cache**：base 和 opt 若是不同 kernel（如 gemm_b4 vs gemm_b8），每次切库都可能触发重调优，测出的是重调优开销而非稳态（症状：数字异常低且每轮都低）。规避：要么 A/B 前让 cache 把两套 kernel 都 warm 稳定，要么改用"每个库连续多跑取稳态、两块背靠背"的方式（牺牲一点热隔离换 cache 稳定）。

用带 `MNN_GPU_TIME_PROFILE=ON` 的 build 只能看 kernel 相对占比（它把绝对耗时放大 ~30×，且按 op 名而非 cl kernel 名聚合）；**最终收益以不带 profile 的干净 build 的端到端 tok/s 为准**。

---

### 优化技巧与常见陷阱

详见 `optimization-handbook.md`，包含：

- **性能分析方法论**：计算强度（roofline）判断瓶颈类型、BW 利用率诊断 memory-bound 问题
- **13 条优化技巧**（Kernel 级 / Host 级 / 系统级），每条有适用场景、做法、注意事项和收益参考
- **16 个常见陷阱**（A-P），每条有应对方法
- **Packed weight 设计**：5 个必须固定的量、signed/unsigned 与 originOffset 约定
- **优化技巧速查表**：按难度和收益排序

---

### 新特性集成参考

#### 特性检测机制

MNN 的 `OpenCLRuntime` 封装了设备能力查询，新特性集成时复用已有机制：

```cpp
// source/backend/opencl/core/runtime/OpenCLRuntime.hpp 中已有的检测接口示例
bool isWeightCpuTransHalf();          // 是否支持 CPU 端 FP16 weight 转换
bool isSupportedFP16() const;         // 是否支持 FP16
bool isSupportedIntelSubgroup() const; // 是否支持 Intel subgroup 扩展
uint32_t getMaxSubGroupSize() const;  // subgroup 大小
```

新增特性检测的标准方式：

```cpp
// 1. 在 OpenCLRuntime.hpp 中新增成员和接口
private:
    bool mIsSupportedFeatureXxx = false;
public:
    bool isSupportedFeatureXxx() const { return mIsSupportedFeatureXxx; }

// 2. 在 OpenCLRuntime.cpp 构造函数中检测
// 扩展检测
if (mOpenCLVersion >= 200 && isExtensionSupported("cl_xxx_extension_name")) {
    mIsSupportedFeatureXxx = true;
}
// 或设备能力检测
cl_uint value;
clGetDeviceInfo(device, CL_DEVICE_XXX, sizeof(value), &value, nullptr);
mIsSupportedFeatureXxx = (value > 0);
```

#### 示例代码适配 MNN 的关键映射

用户提供的示例代码通常是独立的 OpenCL 程序，集成到 MNN 时需要做以下映射：

| 示例代码中的写法 | MNN 中的对应写法 |
|---|---|
| `float` / `half` | `FLOAT`（由 precision mode 宏控制） |
| `float4` / `half4` | `FLOAT4` / `FLOAT16` |
| `(float4)(a,b,c,d)` | `(FLOAT4)(a,b,c,d)` |
| `convert_float4(x)` | `CONVERT_FLOAT4(x)` |
| 直接分配 `cl_mem` | `openCLBackend->onAcquireBuffer(...)` |
| `clCreateBuffer(...)` | `cl::Buffer` 封装 |
| `clSetKernelArg(...)` | `kernel->setArg(idx, value)` / `cl_int ret = CL_SUCCESS; ret |= kernel->get().setArg(...)` |
| `clEnqueueNDRangeKernel(...)` | `runKernel2D(...)` / `runKernel3D(...)` |
| `clCreateProgramWithSource(...)` | `runtime->buildKernel("kernel_file", "kernel_name", buildOptions)` |
| 自定义 GWS/LWS | `localWS2DDefault(gws, maxWorkGroupSize, runtime)` |

#### Kernel 中新扩展的声明方式

```c
// 在 .cl 文件头部声明扩展（#ifdef 保护）
#ifdef FEATURE_XXX_SUPPORTED
#pragma OPENCL EXTENSION cl_xxx_extension_name : enable
#endif

// host 端通过 buildOptions 传入宏
if (runtime->isSupportedFeatureXxx()) {
    buildOptions.emplace("-DFEATURE_XXX_SUPPORTED");
}
```

#### 新特性相关陷阱

集成新扩展时的两个典型陷阱见 `optimization-handbook.md`：

- **陷阱 K**：扩展函数在部分设备上行为不同——必须 Adreno / Mali 都测。
- **陷阱 L**：新特性引入的首次编译耗时——注意 kernel 编译缓存与卡顿。

