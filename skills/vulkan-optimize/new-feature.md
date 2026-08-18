# 方向 C：集成 Vulkan 新特性

> **说明**：本文件是 SKILL.md「方向 C」的展开，与优化轨（benchmark/kernel-opt/integrate）是**并行的独立轨道**。
>
> **目标**：把用户提供的 Vulkan 新特性示例代码（cooperative matrix、subgroup、扩展等）适配集成进 MNN。
>
> **前置**：用户提供了示例代码或参考实现。
>
> **复杂度**：中-高。
>
> **参考**：shader 修改 / 正确性验证见 `SKILL.md`。

---

## 3.0 收集输入

开始前必须确认（缺任何一项主动要求补）：

```
□ 示例代码（.comp + host，或完整可运行 demo）
□ 特性说明：解决什么问题？（如 coop matrix 加速 GEMM、subgroup shuffle 免 barrier）
□ 目标算子：用在 MNN 哪个算子？
□ 目标平台：哪些 GPU 支持？（Adreno / Mali / Apple；coop 目前主要 Adreno）
□ 预期收益：性能 / 精度 / 功能？
```

---

## 3.1 理解示例代码

```markdown
## 示例分析
**特性**: ____（如 VK_KHR_cooperative_matrix、GL_KHR_shader_subgroup_arithmetic）
**扩展/SPIR-V 版本**: ____（coop/subgroup 需 SPIR-V ≥1.3 → --target-env vulkan1.1）
### Host 端
- 用了哪些 Vulkan API / 设备能力查询？pipeline / descriptor / push constant 怎么建？
- local_size / dispatch grid 怎么配？spec constant 传了什么？
### Device 端（.comp）
- 用了哪些新内置（coopMatMulAdd / subgroupAdd / …）、新 layout（constant_id）、扩展 require？
- 数据类型：FLOAT 宏还是裸类型？内存作用域/同步（memory_scope）？
### 关键逻辑
- 核心计算流程；新特性在哪个环节起作用；有无 fallback？
```

如能独立运行，先在目标设备验证示例本身正确（`--target-env vulkan1.1` 编译 + spirv-val）。

---

## 3.2 评估兼容性

| 适配维度 | MNN 做法 | 示例做法 | 改动 |
|---|---|---|---|
| 数据排布 | NC4HW4 | ? | ? |
| 数据类型 | `FLOAT`/`FLOAT4`（fp16/fp32 宏）| ? | ? |
| Buffer/Image | 编译期 `MNN_VULKAN_IMAGE` | ? | ? |
| 内存管理 | `vkBn->onAcquireBuffer` / `getMemoryPool` | ? | ? |
| pipeline 构建 | `vkBn->getPipeline(name, types, localSize, spec)` | ? | ? |
| 参数传递 | descriptor set writeBuffer + push constant / uniform | ? | ? |
| local_size | `getSubgroupSize()` 动态 | ? | ? |

**特性检测**（`VulkanDevice`）：
```cpp
auto coop = vkBn->getDevice().getCoopMatInfo();  // supportCoopMat, selectedFP16CoopMatShape
const auto& sg = vkBn->getDevice().getSubgroupInfo();  // size, stages, ops
uint32_t sgSize = vkBn->getDevice().getSubgroupSize();
```

**Fallback 必须设计**：支持特性 → 新路径；不支持 → 原路径（功能正确）。coop 目前一般 gate 在 `gpuType()==ADRENO && supportCoopMat`。

---

## 3.3 实施集成

按序逐步，每步编译验证：

1. **特性检测**：复用 `VulkanDevice` 已有查询，必要时新增；host 里 `if (supported) 建新 pipeline`。
2. **shader**：适配为 MNN `.comp`——用 `FLOAT` 宏、NC4HW4、spec constant（COOP_M/N/K、activation 等）、扩展 require + 头部注释。coop/subgroup/memory_scope 记得进 `--target-env vulkan1.1`（陷阱 D）。
3. **外科式重生成** shader 数组进 AllShader.cpp + 注册 AllShader.h / VulkanShaderMap.cpp / macro.json（勿跑全量 makeshader，陷阱 A）。
4. **host 选路**：`onCreate`/`onEncode` 里 `if (feature supported) 用新 pipeline else 原路径`，两条都 createSet + 绑定。
5. **编译验证**：`make llm_demo` → 推 → `rm mnn_cachefile.bin` → 单测。

---

## 3.4 验证

- **正确性**：三层 oracle（SKILL.md）；数学等价改动与 baseline 逐 token 一致。**支持和不支持特性的设备都要测**（fallback 路径也要对）。
- **性能**：交替 A/B（新路径 vs 原路径），warm 后测，每轮清 pipeline cache。
- **兼容性**：全量 op/ 无回归；不支持设备走 fallback 无退步。

---

## 3.5 文档

shader 头注释（特性 / 适用设备 / 原理 / fallback）。提交：
```
[Vulkan:Feature] Add <特性> for <算子>

- Runtime feature detection via VulkanDevice
- New shader using <特性>, --target-env vulkan1.1
- Fallback path for unsupported devices (Mali/old driver)
- Tested on <设备>, <加速比>
```

---

## 通过标准

- [ ] 示例充分理解（能解释原理/核心逻辑）
- [ ] 特性检测正确（runtime 能判设备是否支持）
- [ ] shader 适配 MNN（FLOAT 宏 / NC4HW4 / spec constant / target-env）
- [ ] fallback 存在且正确（不支持设备走原路径）
- [ ] shader 外科式重生成 + 三处注册，AllShader.cpp 无污染
- [ ] 支持 + 不支持设备都验证通过
- [ ] 有交替 A/B 性能数据

### 常见问题

| 问题 | 原因 | 修复 |
|------|------|------|
| shader 编译失败 / spirv-opt 拒绝 | 缺 `--target-env vulkan1.1` | coop/subgroup/memory_scope 必带 |
| 运行 segfault | pipeline cache stale / descriptor layout mismatch | `rm mnn_cachefile.bin`；核对 binding/push constant size host↔shader |
| 支持设备上性能反降 | 特性 overhead > 收益（如 coop K 维太小）| 限制特定 shape 才走新路径（陷阱：coop-QK headDim=128 负收益）|
| fallback 被破坏 | 改动影响原路径 | 用独立 pipeline/分支隔离，别改原 shader |
| 数值乱 | 用错 layout（coop 只 int4/8）| dispatcher 显式 gate（陷阱 H）|
