# 方向 C：集成 OpenCL 新特性

> **说明**：本文件是 SKILL.md「方向 C」的展开,与优化轨（benchmark/kernel-opt/integrate 三个"步骤"）是**并行的独立轨道**,不是"步骤 2 之后的步骤 3"。
>
> **目标**：将用户提供的 OpenCL 新特性示例代码适配并集成到 MNN 框架中。
>
> **前置条件**：用户提供了 OpenCL 新特性的示例代码或参考实现。
>
> **复杂度**：中-高（需要理解特性语义 + 适配 MNN 架构）
>
> **参考**：codegen 流程和正确性验证标准，见 `SKILL.md` 对应章节。

---

## 3.0 收集输入

### 必须向用户获取的信息

在开始工作前，**必须确认用户提供了以下内容**，缺少任何一项都要主动要求补充：

```
□ 示例代码（.cl kernel 或完整的 OpenCL host+device 代码）
□ 特性说明：这个特性解决什么问题？（例如 subgroup shuffle、inline assembly、特定扩展）
□ 目标算子：要把这个特性用在 MNN 的哪个算子上？（例如 MatMul、Attention、Conv）
□ 目标平台：哪些 GPU 支持这个特性？（例如 Adreno 730+、Mali-G715+）
□ 预期收益：引入这个特性预期能带来什么提升？（性能 / 精度 / 功能）
```

如果用户只提供了示例代码，没有说明其余信息，主动询问：

> "请补充以下信息：
> 1. 这个特性的作用是什么？
> 2. 要集成到 MNN 的哪个算子中？
> 3. 目标 GPU 平台是什么？
> 4. 预期收益是什么？"

---

## 3.1 理解示例代码

### 3.1.1 分析示例代码结构

阅读用户提供的示例代码，理清以下要素：

```markdown
## 示例代码分析

**特性名称**: ____（例如 cl_qcom_subgroup_shuffle、cl_arm_integer_dot_product）
**所用扩展/版本**: ____（例如 OpenCL 2.0、vendor extension）

### Host 端
- 使用了哪些 OpenCL API？（标准 API / 扩展 API）
- 如何创建 buffer / image？
- 如何设置 kernel 参数？
- GWS / LWS 如何配置？
- 是否有特殊的 context / queue 属性？

### Device 端（.cl kernel）
- 使用了哪些新的内置函数？（例如 sub_group_shuffle、dot 等）
- 使用了哪些新的限定符或属性？（例如 __attribute__、reqd_work_group_size）
- 数据类型：标准类型还是扩展类型？（例如 half、uchar4）
- 内存模型：是否依赖特定的内存序或同步原语？

### 关键算法逻辑
- 核心计算流程是什么？
- 与常规实现相比，新特性在哪个环节发挥作用？
- 是否有 fallback 路径（不支持特性时的替代实现）？
```

### 3.1.2 验证示例代码正确性

如果有条件，先在目标设备上独立运行示例代码，确认其本身是正确的：

```bash
# 如果示例是独立可编译的程序
adb push example_binary /data/local/tmp/
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./example_binary"
```

如果无法独立运行，至少确认：
- 代码能通过 OpenCL 编译器编译（无语法错误）
- 逻辑上可以理解其正确性

---

## 3.2 评估兼容性

### 3.2.1 MNN 架构适配评估

对照 MNN OpenCL 后端的架构，评估示例代码需要做哪些适配：

| 适配维度 | MNN 的做法 | 示例代码的做法 | 需要的改动 |
|---------|-----------|-------------|-----------|
| 数据排布 | NC4HW4（channels packed by 4） | ？ | ？ |
| 数据类型 | FLOAT（可能是 fp16 或 fp32，宏控制） | ？ | ？ |
| Buffer vs Image | 由 runtime 决定 | ？ | ？ |
| 内存管理 | `OpenCLBackend::onAcquireBuffer` | ？ | ？ |
| Kernel 构建 | `runtime->buildKernel(...)` + buildOptions 宏 | ？ | ？ |
| GWS/LWS | `localWS2DDefault` / `localWS3DDefault` / tune | ？ | ？ |
| 精度控制 | `FLOAT` / `FLOAT4` 宏，precision mode | ？ | ？ |

### 3.2.2 特性可用性检测

确认 MNN runtime 如何检测目标特性是否可用：

```cpp
// 检查 OpenCL 扩展
bool supported = runtime->isExtensionSupported("cl_qcom_subgroup_shuffle");

// 检查 OpenCL 版本
bool hasOpenCL20 = runtime->getCLVersion() >= 2.0f;

// 检查设备能力
bool hasSubgroups = runtime->getMaxSubGroupSize() > 1;
```

在 `source/backend/opencl/core/runtime/OpenCLRuntime.cpp` 中查找现有的特性检测方式，确定是否需要新增检测逻辑。

### 3.2.3 Fallback 策略

**必须设计 fallback 路径**。新特性不是所有设备都支持，必须保证：

```
支持特性的设备 → 走新特性路径（性能更好）
不支持特性的设备 → 走原有路径（功能正确）
```

---

## 3.3 实施集成

### 3.3.1 修改清单

根据适配评估，列出需要修改的文件：

```markdown
## 修改清单

### 新增文件
- [ ] source/backend/opencl/execution/cl/xxx_feature.cl（如需新 kernel）

### 修改文件
- [ ] source/backend/opencl/core/runtime/OpenCLRuntime.cpp（特性检测）
- [ ] source/backend/opencl/core/runtime/OpenCLRuntime.hpp（新增检测接口）
- [ ] source/backend/opencl/execution/buffer/XxxExecution.cpp（kernel 调用和选路）
- [ ] source/backend/opencl/execution/buffer/XxxExecution.hpp（新增成员变量）
- [ ] source/backend/opencl/execution/cl/xxx.cl（kernel 实现）
```

### 3.3.2 实施步骤

按以下顺序逐步集成，**每步完成后都编译验证**：

#### 第一步：特性检测

在 `OpenCLRuntime` 中添加特性可用性检测：

```cpp
// OpenCLRuntime.hpp - 新增接口
bool isSupportedFeatureXxx() const;

// OpenCLRuntime.cpp - 实现检测
bool OpenCLRuntime::isSupportedFeatureXxx() const {
    // 检查扩展 / 版本 / 设备能力
    return mIsDeviceSupportedExtension_xxx;
}
```

#### 第二步：Kernel 编写

将示例代码的核心逻辑适配为 MNN 的 .cl kernel：

```c
// 关键适配点：
// 1. 使用 MNN 的数据类型宏：FLOAT, FLOAT4, FLOAT16 等
// 2. 适配 NC4HW4 数据排布
// 3. 使用 MNN 的精度转换宏：CONVERT_FLOAT4 等
// 4. 通过 #ifdef 控制特性路径

#ifdef FEATURE_XXX_SUPPORTED
// 新特性路径
__kernel void xxx_kernel_v2(...) {
    // 使用新特性的实现
}
#else
// Fallback 路径（保持原有实现不变）
__kernel void xxx_kernel(...) {
    // 原有实现
}
#endif
```

**注意**：改完 .cl 必须跑 codegen（参考 `SKILL.md` ".cl 修改流程"）：
```bash
cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . .
```

#### 第三步：Host 端选路

在 Execution 的 `onResize` 中添加特性分发逻辑：

```cpp
// XxxExecution.cpp
if (runtime->isSupportedFeatureXxx()) {
    // 构建使用新特性的 kernel
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DFEATURE_XXX_SUPPORTED");
    mKernel = runtime->buildKernel("xxx", "xxx_kernel_v2", buildOptions);
    // 可能需要不同的 GWS/LWS 配置
} else {
    // 原有路径不变
    mKernel = runtime->buildKernel("xxx", "xxx_kernel", buildOptions);
}
```

#### 第四步：编译验证

编译 + 推送见 SKILL.md「编译与真机运行」，然后跑正确性测试：

```bash
adb shell "cd /data/local/tmp/MNN && ./run_test.out op/XxxTest 3 1 68"
```

---

## 3.4 验证

### 3.4.1 正确性验证

分三层验证（详见 `SKILL.md` "正确性验证"）：

1. **数值层**：新特性路径的输出 vs CPU 输出，检查误差在容忍范围内
2. **op 层**：`run_test.out op/XxxTest` 通过
3. **端到端**（如有条件）：模型推理结果正确

**特别注意**：必须在**支持和不支持**目标特性的设备上分别测试，确认两条路径都能正确工作。

### 3.4.2 性能验证

```bash
# 在支持特性的设备上
adb shell "cd /data/local/tmp/MNN && ./run_test.out speed/XxxSpeed 3 1 68"
```

记录对比数据：

| 场景 | 原路径(us) | 新特性路径(us) | 加速比 |
|------|-----------|-------------|--------|
| 场景1 | xx | xx | x.xx |
| 场景2 | xx | xx | x.xx |

### 3.4.3 兼容性验证

```
□ 支持特性的设备：新路径正确 + 有性能提升
□ 不支持特性的设备：fallback 路径正确 + 性能无回退
□ 全量 op/ 测试无回归
```

---

## 3.5 文档记录

完成集成后，在代码和提交中记录关键信息：

### Kernel 注释

```c
// 使用 <特性名称> 优化 <算法描述>
// 适用设备: <Adreno 730+ / Mali-G715+ / ...>
// 原理: <简要说明新特性如何提升性能>
// Fallback: 不支持时走 xxx_kernel 原有路径
```

### 提交信息

```
[OpenCL:Feature] Add <特性名称> support for <算子名>

- Add runtime feature detection in OpenCLRuntime
- Implement optimized kernel using <特性>
- Add fallback path for unsupported devices
- Tested on <设备> with <加速比> speedup
```

---

## 通过标准

- [ ] **示例代码已充分理解**：能解释特性原理和核心逻辑
- [ ] **特性检测已实现**：runtime 能正确检测设备是否支持
- [ ] **Kernel 已适配 MNN**：使用 MNN 数据类型宏、适配 NC4HW4、精度控制
- [ ] **Fallback 路径存在**：不支持特性的设备能走原有路径
- [ ] **Codegen 已运行**：`python3 opencl_codegen.py . .`
- [ ] **正确性验证通过**：支持和不支持特性的设备上都测试通过
- [ ] **有性能数据**：新特性路径 vs 原路径的实测对比

### 常见问题

| 问题 | 原因 | 修复 |
|------|------|------|
| Kernel 编译失败 | 设备不支持该扩展 | 确认 buildOptions 中的 #ifdef 控制 |
| 编译通过但结果错 | 数据排布不匹配 | 检查 NC4HW4 适配 |
| 支持设备上性能反降 | 新特性 overhead 大于收益 | 检查 GWS/LWS 配置，或限制特定 shape 才走新路径 |
| Fallback 路径被破坏 | 修改影响了原有逻辑 | 用 #ifdef 隔离，不要改动原有 kernel 代码 |
| 扩展函数 undefined | 缺少 `#pragma OPENCL EXTENSION` | 在 kernel 头部添加扩展声明 |
