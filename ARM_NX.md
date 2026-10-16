# MNN ARM NX 接入说明

## 1. 目标与接入方式

本功能在 MNN 现有 Vulkan Buffer 后端中增加面向 ARM NX 的实验路径。MNN 直接生成包含 TOSA 算子的 SPIR-V Graph，由 Vulkan 实现编译和执行。模型仍使用 MNN 格式，调度仍由 MNN 负责，无需接入 ARM Model Converter、VGF Library、Workload Library 或 Scenario Runner。

MNN 自有类、宏、shader 和测试工具统一使用 ARM NX 命名；`VK_ARM_data_graph`、`VkDataGraph*ARM`、`vk*DataGraph*ARM` 等官方扩展与 API 名称保持不变。当前路径通过图接口提交矩阵乘法，没有在普通 compute shader 中直接发射 NX 指令。

## 2. 计算流程

量化 Linear 对应 MNN 中的单输入、group=1 的 1×1 Convolution，prefill 和 decode 共用以下流程：

```text
MNN FP32/FP16 输入（NC4HW4）
  -> compute shader：逐行、逐量化组动态 INT8 量化
  -> SPIR-V Graph：TOSA INT8 x INT8 MATMUL -> INT32
  -> compute shader：反量化 + 跨组累加 + bias + 可选 ReLU/ReLU6
  -> MNN FP32/FP16 输出（NC4HW4）
```

W4 权重在资源准备阶段展开为有符号 INT8；W8 权重也整理为 Graph 使用的布局。保留原始逐输出通道、逐量化组的 scale 和 offset。因此当前计算是 W4/W8 权重来源的 A8×INT8 矩阵乘法，不是原生 INT4 或 W4A16 计算。

设输入为 `[M,K]`，输出通道数为 `N`，量化组数为 `G`，每组大小为 `S=K/G`。这里的 `G` 是权重量化组数，与 Convolution 的 group 参数不同。

```text
Aq: [G,M,S], INT8
Wq: [G,S,N], INT8
C:  [G,M,N], INT32

C[g,m,n] = sum_j Aq[g,m,j] * Wq[g,j,n]
sumA[g,m] = sum_j Aq[g,m,j]

Y[m,n] = bias[n]
       + sum_g sA[g,m] * (float(C[g,m,n]) * sW[n,g]
                         + sumA[g,m] * offsetW[n,g])
```

输入量化使用 `sA=max(abs(A_group))/127`，全零组使用 `sA=1`；舍入为 round-to-nearest-even，范围为 `[-127,127]`。各组先按自己的 scale 恢复浮点值，再跨组累加；`sumA*offsetW` 用于非对称权重量化修正。反量化和跨组累加目前保留在 compute shader 中，Graph 内只有一个 batched MATMUL。

## 3. 支持范围与开关

| 项目 | 当前行为 |
| --- | --- |
| 后端 | Vulkan Buffer，要求 `MNN_VULKAN_IMAGE=OFF` |
| 编译开关 | `MNN_VULKAN_ARM_NX=ON`，默认关闭 |
| 运行开关 | `MNN_VULKAN_ARM_NX=1`，在 MNN 创建设备时读取 |
| 日志开关 | `MNN_VULKAN_ARM_NX_TRACE=1`，打印 Linear 路径 |
| 量化 Linear | W4/W8、对称与非对称量化 |
| 分组约束 | 组数整除 K，组大小为 4 的倍数 |
| 其他 MatMul | 支持部分二维连续 FP32 MatMul，要求高精度、无转置、无 bias；用于接口集成验证 |
| 回退 | 不支持的类型、形状、设备或 Graph 创建失败时走原 Vulkan 实现 |

INT2/INT3 不走此路径。应用自行创建并传入的 Vulkan device 暂不启用 ARM NX 路径，因为 MNN 无法确认其已启用的扩展和 feature。

当前实现检查 Vulkan 1.3+、同时支持 compute 和 Data Graph 的所选队列、`VK_ARM_tensors`、`VK_ARM_data_graph`、`VK_ARM_data_graph_instruction_set_tosa`、`VK_KHR_maintenance5`、`VK_KHR_deferred_host_operations`，以及 tensors、tensorNonPacked、shaderTensorAccess、dataGraph、dataGraphShaderModule、synchronization2 和 MNN shader-int8 支持。

## 4. 构建与小规模测试

需要包含 ARM Tensor/Data Graph 扩展的新版本 Vulkan-Headers，以及包含 `SPV_ARM_graph` 和 TOSA 指令定义的 SPIRV-Headers。以下命令从仓库根目录运行，使用系统 Vulkan，并生成静态链接的测试程序。

```sh
cmake -S . -B build-arm-nx \
  -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF -DMNN_VULKAN_ARM_NX=ON \
  -DMNN_USE_SYSTEM_LIB=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF \
  -DMNN_METAL=OFF -DMNN_BUILD_TEST=ON -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DLLM_SUPPORT_HTTP_RESOURCE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm-nx \
  --target vulkan_arm_nx_test.out vulkan_arm_nx_llm_test.out -j2
```

没有 NX 硬件时，可以在 Apple Silicon Mac 上使用 MoltenVK 和 ARM Graph/Tensor Emulation Layers 验证接口与数值正确性，不需要 Android 设备。先按[详细操作说明](tools/vulkan/README_ARM_NX.md#mac-setup)配置模拟层；必须先加载 Graph 层，再加载 Tensor 层。

```sh
# 先应用 emulation_layer 工具输出的环境变量。
MNN_VULKAN_ARM_NX=1 ./build-arm-nx/vulkan_arm_nx_test.out --require-arm-nx

# 显式关闭 ARM NX 路径，检查 compute 回退。
MNN_VULKAN_ARM_NX=0 ./build-arm-nx/vulkan_arm_nx_test.out

# 移除显式模拟层，检查缺少扩展时的回退。
env -u VK_INSTANCE_LAYERS MNN_VULKAN_ARM_NX=1 \
  ./build-arm-nx/vulkan_arm_nx_test.out
```

`--require-arm-nx` 要求实际记录本路径的 Graph dispatch，防止静默回退掩盖集成失败。它不检测物理 NX 计算单元，模拟器也能满足此条件。

## 5. Qwen3-0.6B 使用示例

使用已经转换好的 Qwen3-0.6B MNN 模型。将模型的 `config.json` 复制为同目录下的 `vulkan.json`，保留模型、权重、tokenizer 等路径，设置 `backend_type=vulkan`、`precision=high`、`thread_num=1`、`sampler_type=greedy`，并将 `tmp_path` 设置为可写的绝对路径。详细模型版本和配置要求见[LLM 操作说明](tools/vulkan/README_ARM_NX.md#qwen3-06b-end-to-end-test)。

```sh
# 配置好原生 Vulkan 或模拟层后，执行短输入正确性测试。
MNN_VULKAN_ARM_NX=1 MNN_VULKAN_ARM_NX_TRACE=1 \
  ./build-arm-nx/vulkan_arm_nx_llm_test.out \
  /absolute/path/Qwen3-0.6B-MNN/vulkan.json \
  '中国的首都是哪里？' 3 --require-arm-nx
```

## 6. 测试与验证

当前测试环境为 Apple M3 Pro、macOS 26.2、MoltenVK 1.4.2、Vulkan Loader/Headers 1.4.363.0、ARM Emulation Layer 0.11.0。

本次重命名后已重新验证：启用功能时两个测试程序构建通过；40 次底层 FP32/INT8 MatMul、32 次 MatMul Module forward、64 次 INT8 Linear forward 通过；显式关闭和缺少扩展时的 MatMul 回退检查通过；FP32/INT8 Graph 的 `spirv-val` 检查通过；关闭编译开关后，使用 MNN 自带 Vulkan 头文件的 Buffer 后端构建通过。shader 源码和内嵌 SPIR-V 字节保持不变。

前期 Qwen3-0.6B 模拟器端到端验证中，prefill 和 decode 每步的 197 个 Linear 均使用本路径，生成过“中国的首都是北京。”；15 步相同历史输入的比较中，与原 Vulkan 路径 argmax 一致率为 15/15，平均 KL 为 0.003306。这是小规模正确性结果，重命名后没有重新跑 LLM 性能测试。

尚未在真实 NX 硬件上验证。启用本路径或记录 Graph dispatch 不能证明实际使用了 NX；原生驱动的算子支持、执行单元选择和性能，需要真机及相应诊断、计数器确认。模拟器耗时不能作为 NX 硬件性能结论。

## 7. 内存与当前限制

每个 Linear execution 保留完整 `[G,M,N]` INT32 中间结果，另有 Graph 输出张量、输入/权重副本及内部工作内存。当前通过 GPU buffer copy 连接 MNN 与 Graph，尚未实现跨层共享中间缓冲、零拷贝输出或分块执行。Module clone 在同一设备上共享准备好的静态权重，但各 Graph 仍有独立的权重分配和拷贝。

此前在这台 Mac 上，128-token prefill 加 64 步 decode 的 Qwen3-0.6B 模拟进程物理内存占用约 9.6 GiB；512-token 尝试造成严重内存压力并被终止。当前实现没有总内存预算限制。模拟测试应从短输入开始；扩大长度前需要监控 CPU/GPU 总物理占用并设置自动终止阈值，不能仅依据进程 RSS。

## 8. 代码入口

| 文件 | 职责 |
| --- | --- |
| [VulkanArmNx.cpp](source/backend/vulkan/component/VulkanArmNx.cpp) | 构造 SPIR-V/TOSA 单算子 Graph，管理 Tensor、pipeline、session 和 dispatch |
| [VulkanConv1x1ArmNx.cpp](source/backend/vulkan/buffer/execution/VulkanConv1x1ArmNx.cpp) | 量化 Linear 资源与三阶段执行 |
| [VulkanDevice.cpp](source/backend/vulkan/component/VulkanDevice.cpp) | 扩展、feature 检测与运行开关 |
| [arm_nx_quantize.comp](source/backend/vulkan/buffer/execution/glsl/arm_nx_quantize.comp) | 输入动态量化 |
| [arm_nx_dequantize.comp](source/backend/vulkan/buffer/execution/glsl/arm_nx_dequantize.comp) | 反量化、跨组累加、bias 和激活 |
| [VulkanArmNxTest.cpp](tools/vulkan/VulkanArmNxTest.cpp) | 小矩阵、Module、量化 Linear 和回退检查 |
| [VulkanArmNxLlmTest.cpp](tools/vulkan/VulkanArmNxLlmTest.cpp) | LLM logits、逐步输出与 Graph 覆盖检查 |

详细安装、Vulkan Validation Layers、SPIR-V 校验和模型版本信息见 [README_ARM_NX.md](tools/vulkan/README_ARM_NX.md)。

---

# MNN ARM NX Integration

## 1. Purpose and integration

This feature adds an experimental ARM NX path to MNN's existing Vulkan buffer backend. MNN directly generates a SPIR-V graph containing TOSA operators, which the Vulkan implementation compiles and executes. Models retain the MNN format and MNN continues to schedule execution. ARM Model Converter, VGF Library, Workload Library and Scenario Runner are not linked.

MNN classes, switches, shaders and test tools use ARM NX names. Official extension and API names such as `VK_ARM_data_graph`, `VkDataGraph*ARM` and `vk*DataGraph*ARM` are unchanged. The current path submits matrix multiplication through the graph interface; it does not emit NX instructions inside an ordinary compute shader.

## 2. Execution flow

Quantized Linear is represented as a single-input 1×1 Convolution with convolution group=1 in MNN. Prefill and decode use the same flow:

```text
MNN FP32/FP16 activation (NC4HW4)
  -> compute shader: per-row, per-quantization-group dynamic INT8 quantization
  -> SPIR-V Graph: TOSA INT8 x INT8 MATMUL -> INT32
  -> compute shader: dequantization + group sum + bias + optional ReLU/ReLU6
  -> MNN FP32/FP16 output (NC4HW4)
```

W4 weights are expanded to signed INT8 during resource preparation. W8 weights are also arranged in the graph layout. Original scales and offsets per output channel and quantization group are preserved. Computation therefore uses A8×INT8 MatMul with weights originating from W4 or W8 storage; native INT4 and W4A16 execution are not implemented.

For an input `[M,K]`, let `N` be the output channel count, `G` the quantization group count and `S=K/G` the group size. `G` denotes weight quantization groups, independently of the Convolution group parameter.

```text
Aq: [G,M,S], INT8
Wq: [G,S,N], INT8
C:  [G,M,N], INT32

C[g,m,n] = sum_j Aq[g,m,j] * Wq[g,j,n]
sumA[g,m] = sum_j Aq[g,m,j]

Y[m,n] = bias[n]
       + sum_g sA[g,m] * (float(C[g,m,n]) * sW[n,g]
                         + sumA[g,m] * offsetW[n,g])
```

Activation quantization uses `sA=max(abs(A_group))/127`, with `sA=1` for an all-zero group, round-to-nearest-even and a range of `[-127,127]`. Each group is restored to its floating-point scale before summation. `sumA*offsetW` corrects asymmetric weight quantization. Dequantization and group summation remain in the compute shader; the graph contains one batched MATMUL operator.

## 3. Coverage and switches

| Item | Current behavior |
| --- | --- |
| Backend | Vulkan buffer; requires `MNN_VULKAN_IMAGE=OFF` |
| Build option | `MNN_VULKAN_ARM_NX=ON`; off by default |
| Runtime switch | `MNN_VULKAN_ARM_NX=1`; read when MNN creates the device |
| Trace switch | `MNN_VULKAN_ARM_NX_TRACE=1`; logs Linear routing |
| Quantized Linear | W4/W8; symmetric and asymmetric quantization |
| Group constraints | Group count divides K; group size is a multiple of four |
| Other MatMul | Selected contiguous 2-D FP32 MatMul with high precision, no transpose and no bias, for integration testing |
| Fallback | Original Vulkan implementation for unsupported types, shapes, devices or graph creation failures |

INT2/INT3 do not use this path. Application-owned Vulkan devices passed into MNN leave this path disabled because their enabled extensions and features cannot be established by MNN.

The implementation checks Vulkan 1.3+, compute and Data Graph support on the selected queue, `VK_ARM_tensors`, `VK_ARM_data_graph`, `VK_ARM_data_graph_instruction_set_tosa`, `VK_KHR_maintenance5`, `VK_KHR_deferred_host_operations`, and the tensors, tensorNonPacked, shaderTensorAccess, dataGraph, dataGraphShaderModule, synchronization2 and MNN shader-int8 capabilities.

## 4. Build and focused tests

Recent Vulkan-Headers with the ARM Tensor/Data Graph extensions and SPIRV-Headers with `SPV_ARM_graph` and TOSA instruction definitions are required. Run these commands from the repository root to use system Vulkan and build statically linked test programs.

```sh
cmake -S . -B build-arm-nx \
  -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF -DMNN_VULKAN_ARM_NX=ON \
  -DMNN_USE_SYSTEM_LIB=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF \
  -DMNN_METAL=OFF -DMNN_BUILD_TEST=ON -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DLLM_SUPPORT_HTTP_RESOURCE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm-nx \
  --target vulkan_arm_nx_test.out vulkan_arm_nx_llm_test.out -j2
```

Without NX hardware, an Apple Silicon Mac can validate API integration and numerical behavior using MoltenVK and ARM Graph/Tensor Emulation Layers. An Android device is not required. Configure emulation using the [detailed setup guide](tools/vulkan/README_ARM_NX.md#mac-setup), with the Graph layer loaded before the Tensor layer.

```sh
# Apply the environment exports printed by your emulation_layer helper first.
MNN_VULKAN_ARM_NX=1 ./build-arm-nx/vulkan_arm_nx_test.out --require-arm-nx

# Explicitly disable the path and check compute fallback.
MNN_VULKAN_ARM_NX=0 ./build-arm-nx/vulkan_arm_nx_test.out

# Remove explicit emulation layers and check missing-extension fallback.
env -u VK_INSTANCE_LAYERS MNN_VULKAN_ARM_NX=1 \
  ./build-arm-nx/vulkan_arm_nx_test.out
```

`--require-arm-nx` requires graph dispatch recordings from this path, preventing silent fallback from hiding integration failures. It does not identify a physical NX engine; emulation can satisfy this requirement.

## 5. Qwen3-0.6B example

Use a converted Qwen3-0.6B MNN model. Copy its `config.json` to `vulkan.json` in the same directory, preserving paths to the model, weights and tokenizer. Set `backend_type=vulkan`, `precision=high`, `thread_num=1`, `sampler_type=greedy` and a writable absolute `tmp_path`. See the [LLM instructions](tools/vulkan/README_ARM_NX.md#qwen3-06b-end-to-end-test) for the model revision and configuration details.

```sh
# A short correctness smoke test after configuring native Vulkan or emulation.
MNN_VULKAN_ARM_NX=1 MNN_VULKAN_ARM_NX_TRACE=1 \
  ./build-arm-nx/vulkan_arm_nx_llm_test.out \
  /absolute/path/Qwen3-0.6B-MNN/vulkan.json \
  '中国的首都是哪里？' 3 --require-arm-nx
```

## 6. Testing and validation

The recorded test environment is Apple M3 Pro, macOS 26.2, MoltenVK 1.4.2, Vulkan Loader/Headers 1.4.363.0 and ARM Emulation Layer 0.11.0.

After the rename, both test programs built with the feature enabled. All 40 low-level FP32/INT8 MatMul forwards, 32 MatMul Module forwards and 64 INT8 Linear forwards passed. MatMul fallback checks passed with explicit disable and missing extensions. FP32/INT8 graphs passed `spirv-val`. The buffer backend built with the feature compiled out and MNN's bundled Vulkan headers. Shader source and embedded SPIR-V bytes are unchanged.

Earlier Qwen3-0.6B end-to-end emulation used this path for all 197 Linear operators in both prefill and decode and generated “中国的首都是北京。”. A 15-step comparison with identical token histories matched the original Vulkan argmax on 15/15 steps, with mean KL 0.003306. These are small correctness results. LLM performance tests were not rerun after the rename.

Physical NX hardware has not been validated. Enabling the path or recording graph dispatches does not prove NX execution. Native operator support, engine selection and performance require hardware validation and suitable diagnostics or counters. Emulation timings do not establish NX hardware performance.

## 7. Memory and current limitations

Each Linear execution retains a full `[G,M,N]` INT32 intermediate, alongside graph output tensors, input/weight copies and internal working memory. GPU buffer copies bridge MNN and the graph. Intermediate buffers are not shared across layers, output is not zero-copy and execution is not tiled. Module clones share prepared static weights on the same device, while each graph still owns a separate weight allocation and copy.

On this Mac, an earlier Qwen3-0.6B emulation run with a 128-token prefill and 64 decode steps reached approximately 9.6 GiB physical footprint. A 512-token attempt caused severe memory pressure and was stopped. The implementation has no aggregate memory budget. Start emulation with short inputs; larger runs require total CPU/GPU physical-memory monitoring and an automatic stop threshold. Process RSS alone is insufficient.

## 8. Code entry points

| File | Role |
| --- | --- |
| [VulkanArmNx.cpp](source/backend/vulkan/component/VulkanArmNx.cpp) | builds the single-op graph and manages tensors, pipelines, sessions and dispatch |
| [VulkanConv1x1ArmNx.cpp](source/backend/vulkan/buffer/execution/VulkanConv1x1ArmNx.cpp) | quantized Linear resources and three-stage execution |
| [VulkanDevice.cpp](source/backend/vulkan/component/VulkanDevice.cpp) | extension and feature checks, runtime opt-in |
| [arm_nx_quantize.comp](source/backend/vulkan/buffer/execution/glsl/arm_nx_quantize.comp) | dynamic activation quantization |
| [arm_nx_dequantize.comp](source/backend/vulkan/buffer/execution/glsl/arm_nx_dequantize.comp) | dequantization, group summation, bias and activation |
| [VulkanArmNxTest.cpp](tools/vulkan/VulkanArmNxTest.cpp) | focused MatMul, Module, quantized Linear and fallback checks |
| [VulkanArmNxLlmTest.cpp](tools/vulkan/VulkanArmNxLlmTest.cpp) | LLM logits, token output and graph coverage checks |

See [README_ARM_NX.md](tools/vulkan/README_ARM_NX.md) for installation, Vulkan Validation Layers, SPIR-V validation and model revision details.
