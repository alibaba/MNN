# OpenCL 优化技巧手册

> 本文件从 MNN OpenCL 后端的实际优化经历中提炼而成。§2 的技巧都有真实提交佐证和量化收益；§6 是有依据但尚未实测的候选方向。本文件聚焦 **kernel / 访存级**优化；host/init 级（零拷贝加载、mmap、kernel 预编译、启发式 tune、pipeline 守卫等）不在此范围。
>
> 用法：先用 §1 定位瓶颈 → 按 §5 速查表「针对瓶颈」列筛技巧 → 读 §2 对应条目。在执行 `kernel-opt.md` 选择优化策略时参考本文件。
>
> **维护约定：本文件是 OpenCL kernel/访存级优化技巧/陷阱的唯一来源。** 每次优化任务沉淀出新经验都更新到这里——新技巧进 §2 并登记到 §5 速查表；新陷阱进 §3；分析方法论进 §1。`kernel-opt.md` 等只引用不复制。只写方法论（含最小示例和实测收益），不写单次任务细节；新增时顺手精简过时/显而易见的条目。

---

## 1. 性能分析方法论

优化前先定量分析瓶颈，不要凭直觉猜。

### 1.1 算计算强度（Arithmetic Intensity）

计算强度是 roofline 模型的核心指标，用于判断 kernel 是 **compute-bound** 还是 **memory-bound**：

```
AI = FLOPs / Bytes_transferred
```

- `FLOPs`：kernel 执行的浮点运算总数
- `Bytes_transferred`：kernel 从 global memory 读写的总字节数

将 AI 与设备的 **ridge point** 比较：

```
ridge_point = peak_FLOPS / peak_BW
```

- `AI < ridge_point` → **memory-bound**：性能受内存带宽限制
- `AI > ridge_point` → **compute-bound**：性能受计算能力限制

**典型 kernel 的计算强度**：

| Kernel 类型 | 典型 AI (FLOPs/Byte) | 通常瓶颈 |
|---|---|---|
| GEMV (decode, batch=1) | 0.5-2 (int4) | memory-bound |
| GEMM (prefill, batch>1) | 随 batch 增长 | 小 batch memory，大 batch compute |
| Elementwise / Raster | ≈ 0 | memory-bound |
| Attention (长序列) | 中等 | 取决于 seq_len |

### 1.2 设备参考数据

| 设备 | peak BW (实测) | peak FLOPS (FP16) | ridge point |
|---|---|---|---|
| Snapdragon 8 Gen3 | ~50 GB/s | ~4.5 TFLOPS | ~90 |
| Snapdragon 8 Elite | ~55 GB/s | ~5.7 TFLOPS | ~104 |
| Mali-G715 | ~35 GB/s | ~3.0 TFLOPS | ~86 |

> LPDDR 理论带宽需打 6-8 折作为 GPU 单进程实测可达值。

### 1.3 按瓶颈类型选优化杠杆

**第一步：用计算强度判断瓶颈类型**

| 判断 | 瓶颈 | 优先方向 | 对应技巧 |
|---|---|---|---|
| AI ≪ ridge_point | **memory-bound** | 减少数据搬运、提高缓存命中 | → 第二步看 BW 利用率 |
| AI ≈ ridge_point | **均衡** | 两边都有优化空间 | 技巧 1, 6, 7 |
| AI ≫ ridge_point | **compute-bound** | 简化计算、增加并行度 | 技巧 3, 4, 5 |

**第二步（memory-bound 时）：算 BW 利用率**

```
BW_utilization = actual_bytes / kernel_time / peak_BW
```

| BW 利用率 | 含义 | 优先方向 | 对应技巧 |
|---|---|---|---|
| < 50% | 带宽远未跑满，访存模式或 launch 开销有问题 | 优化访存模式、减少 dispatch | 技巧 1, 2, 6, 8, 9 |
| 50-70% | cache miss 或 launch overhead 有改善空间 | 调整 WG 大小、加宽 tile | 技巧 1, 6, 8 |
| > 70% | 带宽已接近上限 | 减小数据量、packed 存储 | 技巧 2, 6 |

**第三步（前两步的杠杆都无效时）：怀疑 register/occupancy 墙**

若 AI 说 memory-bound 但减流量/减 ALU 都不涨，甚至减 ALU 反降——大概率是 §1.5 的 register/occupancy 墙（DRAM 流量已被缓存吸收）。此时走**技巧 8（tile 甜点扫描）/ 技巧 4（workgroup 重写）**，并**避免技巧 3 的 block 外提**（会加活跃累加器）。

> **诊断 → 技巧 的总入口**：先按上面三步定位瓶颈类型，再到 §5 速查表用「针对瓶颈」列筛出候选技巧，读 §2 对应条目的做法/注意/收益。`kernel-opt.md` 的决策树是按"症状"的另一条导航路径，二者互补。

### 1.4 性能改动标准流程

1. 算计算强度，判断 compute-bound 还是 memory-bound
2. memory-bound 时再算 BW 利用率，定位具体问题
3. 选 1 个杠杆（不要一次改两个，结果归因不清）
4. 用 oracle 验正确性
5. **目标设备**测速（Android 真机）
6. 没收益就 revert，记录"为什么不 work"

GPU 跑分波动 ~5-10%，单次数字不可靠。同设备跑 10 次取中位数对比。

**launch overhead 估算**：dispatch 数 x 每次约 5-10 us。LLM decode 1 token ≈ `n_layer x ops_per_layer` 次 dispatch。当 BW 优化把 decode 拉到 launch 上限附近时再做 kernel fusion 等大改。

### 1.5 register/occupancy 墙（移动 GPU int4 GEMM 常见真实瓶颈）

roofline 说 memory-bound 时，别急着认定"减少 DRAM 流量"就能提速——mobile GPU 的 L2/纹理缓存常把权重流量吸收掉（陷阱 E），此时真正的墙是**每个 work-item 的寄存器数决定的 occupancy**，而 grid 通常已经足够大（几千 work-item），不缺并行度。判断信号：

- **减 ALU 反而变慢** → 不是 compute-bound，且你的改动多半引入了活跃寄存器（见技巧 3 陷阱）。
- **加宽 register-tile 有甜点、且过宽会断崖**：实测 batch-4→8 提速、8→16 因累加器爆寄存器 −30%。别假设"更大 tile 摊薄流量一定更好"或"更小 tile 提 occupancy 一定更好"——**必须实测每一档**。
- **精度模式决定寄存器账**：LLM `precision:"low"` = fp16 compute（`COMPUTE_FLOAT=half`），累加器是 half（2 个/寄存器），算寄存器压力要按 half 折算；normal/high 是 fp32 累加，同样 tile 翻倍占用。改 tile 宽度前先确认当前精度模式。

结论：这类 kernel 的主要杠杆是**在寄存器预算内找 tile 甜点**，而不是继续压 ALU 或 DRAM 流量。用交替 A/B（见 SKILL.md 性能测量）逐档实测，别用直觉。

---

## 2. Kernel / 访存级优化技巧

### 技巧 1：Local memory 缓存 + workgroup 并行归约

**适用场景**：数据被多个 work-item 重复读取；需要做求和/最大值等归约操作。

**做法**：
- 将频繁访问的 global memory 数据加载到 `__local` 数组中，用 barrier 同步后从 local memory 读取
- 归约操作用并行 tree reduction：每个 work-item 先做局部累加，再用 `barrier + stride 折半` 归并

**注意事项**：
- 移动设备 local memory 通常限制 32KB，超出会段错误或性能反降（→ 需添加 fallback 到 global memory 的路径）
- `inputChannels >= 32` 时启用 local memory 路径，低于此值 workgroup 协作开销大于收益

**收益参考**：GEMV local memory 归约（`146dd0958`）2-10x；LinearAttention 组合 local memory + 并行 reduce + 向量化 + 2D workgroup 达 112x（`6a361d3e4`）

---

### 技巧 2：image1d_buffer_t 利用纹理缓存

**适用场景**：只读、连续访问的输入数据（如 GEMV/GEMM 的 input tensor）。

**做法**：
- 将 `__global const FLOAT* input` 改为 `__read_only image1d_buffer_t input`，用 `read_imageh` / `read_imagef` 读取
- 不需要改数据排布——`image1d_buffer_t` 直接 wrap 已有 buffer

**注意事项**：
- GPU 纹理单元有专用缓存，对只读数据的空间局部性访问比通用 L1/L2 更高效
- 只适用于只读数据，写入仍用 `__global`
- `read_imageh` 的坐标计算需要对齐 NC4HW4 排布

**收益参考**：int4 GEMV input 读取（`99f8632d3`），Qwen3-0.6B prefill +34%

---

### 技巧 3：常量预计算 / offset 折叠到 host 端

**适用场景**：kernel 内层循环中有重复执行的常量运算。

**做法**：
- 识别内层循环中每次迭代都重复计算的常量表达式
- 将计算移到 host 端，结果作为 kernel 参数传入

**典型案例**：int4 反量化公式 `wei = (nibble - 8) * scale + offset`，将 `-8` 预折叠进 `offset`（host 端做 `offset_new = offset - 8 * scale`），kernel 中简化为 `wei = nibble * scale + offset_new`

**注意事项**：
- 注意 `originOffset` 是否已折进 bias。模型导出器写出的 alpha 中 **originOffset 已折进 bias**，shader 解出 signed 权重后做 `signed_w * scale + b` 即可。**不要**再做 `(unsigned - offset) * scale + raw_b`，会重复折一次
- 每次消除的运算看似微小，但乘以百万级 weight 就很可观
- **外提到 block 粒度时当心引入活跃累加器**：把 `wei*scale+offset` 从内层每元素提到"每 block 反量化一次"（如 block-quant 下用 `out += scale·Σ(in·nibble) + offset·Σ(in)`）确实减 ALU，但需要额外的 per-(batch,oc) 累加器活跃整个内层循环。在 register/占用受限的 kernel 上这会降 occupancy 而**净变慢**（实测 int4 prefill GEMM −7%）。减 ALU 前先确认不是 register/occupancy 墙（§1.5）。

**收益参考**：配合技巧 2 组合使用（`99f8632d3`），贡献了 +34% 中的一部分

---

### 技巧 4：单 work-item → workgroup 并行化重写

**适用场景**：一个 work-item 独占太多工作，导致私有数组过大、寄存器溢出。

**诊断信号**：kernel 中存在大量私有数组（如 `float q_local[256]`）；单个 work-item 处理一整个 (batch, head) 或类似粒度。

**做法**：
1. 识别可并行的维度——即使算法有串行依赖（如 RNN 时间步），也可以在**其他维度**（d_k、d_v、IC）做并行化
2. 2D work decomposition：`gid(0)` = workgroup 内并行，`gid(1)` = 任务分配
3. 私有数组 → `__local` 数组，跨时间步复用
4. 归约操作用并行 reduce（参考技巧 1）

**注意事项**：
- 寄存器压力大时，`__local` 是更好的选择——GPU local memory 带宽远高于 global
- 过多私有数组导致寄存器不足时，性能反降而非报错

**收益参考**：LinearAttention 重写（`6a361d3e4`），平均 112x 加速（decode 263x，prefill 60x）

---

### 技巧 5：Decode / Prefill 分支特化

**适用场景**：同一个算子在 decode（seq_len=1）和 prefill（seq_len>1）下的瓶颈不同。

**做法**：
- Decode 路径用 `#ifdef DECODE_PHASE` 特化：seq_len=1 时简化地址计算、减少循环层数
- Prefill 路径保留完整循环
- Host 端根据 batch 维度选择不同的 kernel 或宏

**原理**：
- GEMV（decode）是 BW bound，每个 weight 只用一次，必须减少读取量（原生 packed kernel 更优）
- GEMM（prefill）有数据复用，weight 被多行复用，反量化开销可摊薄（可先反量化再用通用 GEMM）

**收益参考**：LinearAttention decode 特化（`c2b64723f`）；w2/w3 GEMV 原生 packed kernel vs GEMM outer-dequant（`0feefdef9`）

---

### 技巧 6：权重重排 + 并行粒度协同设计

**适用场景**：当前权重布局和 kernel 并行策略不匹配，导致访存不连续或并行粒度不足。

**做法**：
- 调整权重 pack 布局（如从 `pack=16/32` 改为 `packCin=2/4, packCout=8`），使每个 work-item 处理更多 output channel
- 统一多个独立 kernel（如 `gemv_conv_c1/c2/c4` → `gemv_conv_c8`）
- WGS 从固定值改为可 tune（8/16/32/64/128/256）

**注意事项**：
- 权重布局和 kernel 并行策略必须**协同设计**，单独改 kernel 不改布局效果有限
- 改布局涉及 host packing 和 shader unpack 双边同步改（→ 参考 Packed weight 设计）

**收益参考**：GEMM/GEMV 权重重排（`146dd0958`），1.3-2x

---

### 技巧 7：向量化（float4 / float8）

**适用场景**：连续内存访问，可以一次处理多个元素。

**做法**：
```c
int N4 = N / 4;
for (int i = 0; i < N4; i++) {
    FLOAT4 val = vload4(i, input + gid * N);
    vstore4(val * (FLOAT4)2.0f, i, output + gid * N);
}
for (int i = N4 * 4; i < N; i++) { ... }  // 处理余数
```

**注意事项**：
- 使用 `FLOAT4` / `CONVERT_FLOAT4()` 宏而非裸 `float4`，确保 FP16/FP32 兼容
- vload/vstore 时类型必须匹配，类型不匹配导致数值错（不报错）
- FP16 场景考虑 float8 进一步提升吞吐

**收益参考**：1.2-2x，通常与其他技巧组合使用

---

### 技巧 8：register/output tiling（加宽每 work-item 输出块）

**针对瓶颈**：occupancy / 均衡——grid 已足够大（不缺 work-item）、权重/输入被缓存吸收，想提高单 work-item 的 compute:load 比、摊薄权重加载与调度开销。

**做法**：让每个 work-item 多算几行/列输出（如 batch tile 4→8，或 oc tile 8→16），一次加载的权重（或输入）复用到更多输出，从而减少总加载次数和 dispatch 数。host 端 GWS 对应维度 `/tile`，边界用技巧 10 处理。

**注意事项**：
- **累加器随 tile 线性增长，存在寄存器断崖**（§1.5 / 陷阱 H）：过窄 compute:load 比低，过宽爆寄存器 occupancy 骤降。甜点**必须逐档实测**（batch-4/8/16 各测），别外推。
- fp16 compute 时累加器是 half（2 个/寄存器），按 half 折算寄存器预算。
- 和技巧 6（权重重排）互补但不同：本技巧**不改权重布局**，只改并行粒度，风险低；技巧 6 改布局收益上限更高但要双边同步。

**收益参考**：int4 prefill GEMM batch-4→8（`gemm_b8_c8_int4_buf`）+0.6~3.6%（Adreno 840，fp16）；batch-16 因寄存器断崖 −30%（反例，见 §1.5）。

---

### 技巧 9：遍历顺序转置消除 Cache Set Thrashing

**适用场景**：NC4HW4 格式 tensor 的 raster / 格式转换，当 batch 维度 N 是 2 的幂次时性能异常下降。

**诊断方法**：
1. 对比 N 为 2 的幂次（如 512）和非 2 的幂次（如 514）的性能差异
2. 如果 N=512 明显慢于 N=514，大概率是 cache set thrashing

**根因**：NC4HW4 布局 `[N*C/4][H][W][4]`，当 H=W=1 时 channel 组间地址差 = `N * 4` 元素。N 是 2 的幂次 → 地址差是 2 的幂次 → 不同 channel 组映射到同一 L2 cache set → thrashing。

**做法**：
- 在 **host 端**修改 gws 和 stride（不改 kernel 代码）
- 将 1D 遍历 `gws={N*C, 1, 1}` 改为 2D `gws={N, C, 1}`
- 连续 work items 改为沿 batch 维度遍历，NC4HW4 物理地址间距仅 4 元素（连续）

**关键洞察**：
- 问题不在 kernel 代码，而在 **work item 到内存地址的映射关系**
- 纯内存拷贝算子（如 raster，计算密度 ≈ 0）受 cache 效率影响最大；计算密集型算子的 cache miss 被计算延迟掩盖

---

### 技巧 10：边界钳位 + guard 存储替代 leaves 枚举

**适用场景**：把一个更宽 tile 的 kernel（如 batch-8/16）推广到任意尺寸时，边界 work-item 的有效元素不足一个 tile。

**做法**：不要为 1..N-1 每种残余数枚举分支。而是
- **读**：用 `min(idx, lastValid)` 钳住每个 sub-element 的加载下标，越界的 lane 读到最后一个合法元素（值无所谓，反正不写回），对 image/buffer 输入都安全，无需依赖纹理 clamp；
- **写**：逐 sub-element 带 `if (idx <= lastValid)` guard 存储（vstore4/单元素），只写有效部分。

这样一个 `if(tail)` 分支就覆盖所有残余数：主体（满 tile）走原向量化快路径，边界走钳位+guard 慢路径。分支对 tile 内是 uniform 的，快路径几乎零开销。

**注意事项**：
- 别用运行时变量索引私有数组来做 guard 存储（`out[b]`）——会把整组累加器打进 private memory，**连快路径的 occupancy 一起拖累**。用宏展开成 `out0/out1/...` 的枚举 guard。
- host 侧对应放开 dispatch 条件（如从 `%8==0` 放开到全部），GWS 用 `UP_DIV(n, tile)`。

**收益参考**：让 batch-8 int4 GEMM 覆盖任意 prefill 长度，非整除 tile 的场景反而收益更大（+3.6% vs 整除的 +1.5%），且快路径不回退。

---

### 技巧 11：spike 先验证假设，再产品化

**适用场景**：一个优化点实现成本高（要动 host dispatch、加 leaves/边界处理、双边镜像），但收益未知。

**做法**：先写一个**一次性 spike**——哪怕只对当前目标 shape 正确（如硬编码 tile、跳过边界处理、直接改 GWS），只要能在真机上测出性能假设成立与否。赢了再投入干净实现（加边界/回退/双边同步）；输了直接 revert，省下产品化的功夫。

**原理**：性能假设（"加宽 tile 能提速吗"）和工程正确性（"处理所有边界"）是两件事，先用最小代价验证前者。配合交替 A/B（SKILL.md 性能测量）判定。

**注意事项**：spike 常常对非目标 shape 不正确，务必记得它只是测量工具，产品化时重写而非直接合入。

---

## 3. 常见陷阱

### 陷阱 A：host weight prepare 与上游 quant 路径约定不一致

新增量化 bit / 新 op 接入 host 时，常踩到 `ConvolutionCommon::load` 没填 outputPtr 或 alpha buffer 的 originOffset 已折叠/未折叠这类约定不一致。

**应对**：
- 进入新路径前确认 `quanCommon->weight.get()` 是 packed 还是 unpacked、signed 还是 unsigned、offset 是否已折进 alpha
- 不要假设新 bit 会走和已有 bit 同样的预处理，每条新路径单独验证
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

OpenCL 在 mobile GPU 上对小尺寸 weight 倾向 Image2D。但 Image RGBA pixel = 16 字节，packing tile 不是 16 整数倍时会强制 round-up，浪费带宽。新加 layout 时 default `mUseImage = false`，确认 host 端 image 路径分支已 explicitly handle + tile size 对齐 16B 后再开。低 bit packing 通常应禁用 Image。

### 陷阱 D：tune level 重复 tune

默认 `mTuneLevel = Wide` 对常见 GEMV/GEMM **已经**在搜索 WGS / shape variant。再在外层手动 tune 或 hardcode WGS 等于跟内置 tune 抢资源。先 grep `getCLTuneLevel()` 确认 tune 在哪一层做的，再决定要不要新加。

### 陷阱 E：直觉的 BW 浪费可能已被 GPU cache 吸收

mobile GPU 有 L2 + texture cache。"表面看每个 wavefront 都重读同一段数据"未必意味着 DRAM 真的多读 N 次。改 kernel 前先验证是否真 DRAM bound，否则代码复杂度上来了性能没动。

### 陷阱 F：宏 alias 让多个 #ifdef 同时为真

为了让"未扩展的 kernel"在新 quant bit 下编译过，常加 `#define W_QUANT_4` alias。扩展的 kernel 里所有相关 `#ifdef` 必须 `W_QUANT_2 → W_QUANT_3 → W_QUANT_4 → W_QUANT_8` 顺序，**新 bit 放最前**优先匹配。

### 陷阱 G：Local memory 超限

使用过多 local memory 导致段错误或性能下降。移动设备通常限制在 32KB。超出时需添加 fallback 到 global memory 的路径。

### 陷阱 H：寄存器溢出

过多私有数组导致寄存器不足，性能反降。消除不必要的中间数组，或改用 local memory。不同平台寄存器数量差异大——Adreno 寄存器多，Mali 寄存器少，同样的 kernel 在 Adreno 上正常但在 Mali 上溢出。

不止私有数组：**加宽 register-tile（更多 batch/oc 累加器）也会撞寄存器断崖**——超过某档 occupancy 骤降甚至 spill，性能断崖式下跌（实测 batch-16 int4 GEMM −30%，且首跑编译更慢）。tile 宽度存在甜点，逐档实测（见 §1.5）。fp16 compute 时累加器按 half 折算寄存器。

### 陷阱 I：向量化类型不匹配

使用 vload4/vstore4 时类型不匹配导致数值错。使用 `CONVERT_FLOAT4()` 处理 FP16/FP32 类型转换。

### 陷阱 J：忘记更新 kernel 映射

修改了 .cl 但忘记运行 codegen，编译成功但运行时使用旧 kernel。**每次改 .cl 后必须**：
```bash
cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . .
```

### 陷阱 K：扩展函数在部分设备上行为不同

同一个扩展在不同厂商的实现可能有细微差异（参数语义、精度、edge case 行为）。**必须在 Adreno 和 Mali 上都测试**，不能只在一个平台验证通过就认为没问题。如果只有一个平台的设备，在代码中用 `runtime->getGpuType()` 限制只在已验证的平台启用。

### 陷阱 L：新特性引入的编译耗时

部分 OpenCL 扩展（如 subgroup、inline asm）会显著增加 kernel 首次编译时间。MNN 有 kernel 编译缓存（`mnn_cachefile.bin`），但首次运行或 cache miss 时用户会感知到卡顿。评估新特性时要关注编译耗时，必要时在 `buildKernel` 外加条件判断避免不必要的编译。

### 陷阱 M：NC4HW4 格式的 L2 Cache Set Thrashing

详见技巧 9。

### 陷阱 N：布局 helper 和调用点重复累加偏移

抽出 NCHW/NC4HW4 offset helper 时，先明确返回的是 tensor base、vector base，还是完整 element offset。若 helper 已包含 `channel4 * 4`，调用点再加一次 `x` 会造成错位写甚至越界；`C=4` 的测试因为 `x=0` 还会假通过。回归至少覆盖两个 vector block，并优先使用真实模型的 `head_dim` / channel 规模。

### 陷阱 O：build option 宏名拼写错，shader 静默走 #else

host `buildOptions` 传入的宏名与 .cl 里的 `#ifdef` 不一致（如 `QUANT_BIT_2` vs `QUANT_BIT == 2`）时，shader **编译不报错**，悄悄走 `#else` 分支 → 数值错而非编译失败,极难定位。新加 `#ifdef` 分支后扫一遍宏名拼写，确认 host 侧 `emplace` 的字符串和 kernel 里的宏严格对应。

### 陷阱 P：Adreno 老编译器 inline function 不稳定

重复展开的逻辑（如 int4 unpack）用 `#define` macro 展开，**不要用 inline function**——Adreno 老编译器对 inline 的稳定性差，可能生成错误代码或编译失败。

---

## 4. Packed Weight 设计

新加 quant bit 或调整 tile 排布时，**先固定 5 个量**：

| 量 | 解释 |
|---|---|
| tile = (IC_inner x OC_inner) | 一次原子访问的最小区块（OpenCL 常见 4x8 = 32 weights） |
| 字节/tile | 由 bit 决定：w2 = 8B, w3 = 12B, w4 = 16B, w8 = 32B |
| byte index 内的语义 | 哪个 byte 对应哪个 (oc_inner, ic_inner) 子集 |
| bit 顺序 | 单 byte 内 OC0/OC1/... 在哪几个 bit |
| signed/unsigned 存储 | shader 解出后是否还要减 originOffset |

这 5 个量先在 PR 描述里写死，packing 和 unpack 各自照表实现。先跑通正确性，再优化。

**signed/unsigned 与 originOffset**：模型导出器写出的 alpha 是 `b = min_val + offset_signed * scale`，**originOffset 已折进 bias**。shader 解出 signed 权重后做 `signed_w * scale + b` 即可。**不要**再做 `(unsigned - offset) * scale + raw_b`，会重复折一次。

**block-quant alpha 索引**：内存布局通常 `[OC/4, blockNum, 2 (s,o), 4 (oc_inner)]`。kernel 读取时 4 个维度的顺序要和 host 写法严格一致。

---

## 5. 优化技巧速查表

按「针对瓶颈」列筛选：先用 §1 定位瓶颈，再挑对应技巧。

| # | 技巧 | 针对瓶颈 | 适用场景 | 难度 | 收益参考 |
|---|------|---------|---------|------|---------|
| 1 | Local memory + 并行归约 | memory（重复读）| 数据被多次读取、归约操作 | 中 | 2-10x |
| 2 | image1d_buffer_t 纹理缓存 | memory（只读 BW）| 只读连续输入数据 | 中 | +34% |
| 3 | 常量预计算 / offset 折叠 | compute（轻）| 内层循环有常量运算 | 低 | +5-15% |
| 4 | 单 work-item → workgroup 重写 | 寄存器溢出 / 串行 | 一个 work-item 独占太多工作 | 高 | 10-260x |
| 5 | Decode / Prefill 分支特化 | 两者（分场景）| 不同 seq_len 瓶颈不同 | 中 | 1.2-2x |
| 6 | 权重重排 + 并行粒度协同 | memory / 并行粒度 | 访存不连续、并行粒度不匹配 | 中-高 | 1.3-2x |
| 7 | 向量化（float4/float8） | memory / 吞吐 | 连续内存访问 | 低-中 | 1.2-2x |
| 8 | register/output tiling | occupancy / 均衡 | grid 已够大、想摊薄加载与调度 | 中 | +0.6~3.6%（有断崖）|
| 9 | 遍历顺序转置消除 cache thrashing | memory（cache thrash）| NC4HW4 raster，N 是 2 的幂次 | 中 | 消除异常 |
| 10 | 边界钳位 + guard 存储替代 leaves 枚举 | 工程（覆盖尺寸）| 更宽 tile 推广到任意尺寸 | 中 | 覆盖全尺寸不回退 |
| 11 | spike 先验证假设再产品化 | 方法论 | 实现成本高、收益未知 | 低 | 省无效工程 |

---

## 6. 候选优化手段（尚未在 MNN 验证）

> 与上面 §2 不同，本节是**有理论依据但还没在 MNN 落地实测**的方向，无提交佐证、收益是预估。想用先按技巧 11（spike）验证，验证通过并有实测数据后再"毕业"为正式技巧（移入 §2 + §5 表）。

### 候选 A：Split-K（K 维拆分 + 部分和归约）

**针对瓶颈**：并行度不足（grid 太小填不满 GPU）。把 K 维的 reduction 拆给多个 work-item 各算一段部分和，再归约（原子加或第二个 kernel）。
**用/不用**：只在 **grid 太小**（输出 tile 数 ≪ GPU 可并发 wavefront 数）时值得。若 grid 已上千 work-item（常见的大 N / 大 batch prefill），occupancy 是寄存器限的，加 work-item 填不满更多 wavefront → **无效**（本次 int4 prefill 已据此排除）。fp16 原子加支持差，通常要 temp buffer + 归约 kernel，有额外读写。

### 候选 B：Double-buffering / 软件预取

**针对瓶颈**：访存延迟（latency-bound，尤其配 local memory staging）。计算当前 tile 时预取下一 tile 的权重/输入，用计算掩盖加载延迟。
**风险**：需要双份 staging 寄存器/local，会加重寄存器压力——在已是 register/occupancy 墙的 kernel 上可能反降（先看 §1.5）。适合计算较重、寄存器有余量的场景。

### 候选 C：Subgroup / wave 原语

**针对瓶颈**：memory（跨 lane 共享）/ 归约。用 subgroup shuffle/broadcast 让同 wave 内 work-item 共享权重（免 local memory + barrier），或用 subgroup reduce 替代 tree reduction。
**前置**：属于"新特性"，走方向 C 流程——**必须有可运行示例代码 + runtime 特性检测 fallback**（`isSupportedIntelSubgroup` / `getMaxSubGroupSize`），Adreno/Mali 都要测（陷阱 K），注意首次编译耗时（陷阱 L）。

### 候选 D：循环展开（`#pragma unroll` / 手工 K-unroll）

**针对瓶颈**：compute（ILP 不足、循环开销）。展开 K 循环增加独立 mad 链、减少分支/地址计算开销。现有 kernel 多为手工展开 4 IC。
**风险**：展开过多增加寄存器与代码体积（首跑编译变慢），过犹不及；与技巧 8 一样存在甜点，实测定档。
