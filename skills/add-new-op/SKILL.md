---
name: add-new-op
description: 为 MNN 框架新增算子。包含 Schema 定义、形状计算、几何计算、后端实现、单元测试的完整 TDD 流程。分 5 步执行，每步有独立测试标准。
---

# MNN 新增算子 SKILL

> **触发条件**：当用户请求为 MNN 添加/实现一个新的算子时触发。常见表述包括："添加xxx算子"、"实现xxx op"、"支持xxx操作"等。

## 概述

本 SKILL 指导 AI Agent 为 MNN 框架新增算子支持。采用 **TDD（测试驱动）模式**，每一步都有明确的测试标准。

### 核心原则

**优先级顺序**：几何计算 > 后端实现

- **几何计算**：如果新算子可以通过已有算子的拆解、组合实现，优先用几何计算，不需要为每个后端单独实现
- **后端实现**：如果算子无法拆解，或性能敏感需要原生实现，则添加后端实现

### 注意事项

> **严禁访问以下目录**：`schema/private/` 和 `source/internal/`，包含内部私有代码，**不得读取、修改或引用**。

---

## 核心文件清单

| 目录/文件 | 作用 | 何时修改 |
|----------|------|---------|
| `schema/default/MNN.fbs` | 算子类型和参数定义 | **每个新算子都需要** |
| `schema/default/CaffeOps.fbs` | Caffe 框架算子参数 | 有参数时 |
| `schema/default/TensorflowOp.fbs` | TF 框架算子参数 | 有参数时 |
| `source/shape/Shape*.cpp` | 形状（维度）计算 | 输出形状与输入不同时 |
| `source/geometry/Geometry*.cpp` | 几何计算（算子拆解） | 优先实现 |
| `source/backend/cpu/CPU*.cpp/.hpp` | CPU 后端实现 | 无法几何拆解时 |
| `source/backend/metal/Metal*.cpp/.hpp/.metal` | Metal 后端实现 | 扩展到 Metal |
| `source/backend/opencl/execution/*.cpp/.h/.cl` | OpenCL 后端实现 | 扩展到 OpenCL |
| `source/backend/vulkan/image/execution/Vulkan*.cpp/.hpp` | Vulkan 后端实现 | 扩展到 Vulkan |
| `source/backend/cuda/execution/*.cu/.cuh` | CUDA 后端实现 | 扩展到 CUDA |
| `include/MNN/expr/ExprCreator.hpp` | 表达式 API 函数 | 需要用户可调用时 |
| `test/op/*Test.cpp` | 单元测试 | **每个新算子都需要** |
| `tools/script/register.py` | 注册脚本 | 添加 shape/geometry/backend 后运行 |

---

## 分步流程总览

```
┌──────────────────────────────────────────────────────────┐
│  步骤 1: Schema 定义 (step1-schema.md)                    │
│  输入: 算子名称、语义、输入输出定义                          │
│  输出: MNN.fbs 中的 OpType + 可选的参数 table              │
│  测试: flatc 编译通过                                      │
├──────────────────────────────────────────────────────────┤
│  步骤 2: 形状计算 (step2-shape.md)                        │
│  输入: 算子的输入输出维度关系                                │
│  输出: source/shape/Shape*.cpp                             │
│  测试: register.py 运行成功 + cmake 编译通过               │
├──────────────────────────────────────────────────────────┤
│  步骤 3: 几何计算或 CPU 实现 (step3-compute.md)            │
│  输入: 算子的计算逻辑                                      │
│  输出: Geometry*.cpp 或 CPU*.cpp                           │
│  测试: register.py + 编译通过                              │
├──────────────────────────────────────────────────────────┤
│  步骤 4: 单元测试 (step4-test.md)                         │
│  输入: 算子的预期行为                                      │
│  输出: test/op/*Test.cpp                                  │
│  测试: ./run_test.out op/MyOp 全部通过                     │
├──────────────────────────────────────────────────────────┤
│  步骤 5: 扩展后端与优化 (step5-optimize.md)                │
│  输入: 步骤 4 通过                                        │
│  输出: Metal/OpenCL/Vulkan/CUDA 实现                      │
│  测试: 各后端单元测试通过                                   │
└──────────────────────────────────────────────────────────┘
```

### 步骤选择指南

| 情况 | 需要执行的步骤 |
|------|--------------|
| 输出形状与输入完全一致，可几何拆解 | 1 → 3(几何) → 4 |
| 输出形状与输入不同，可几何拆解 | 1 → 2 → 3(几何) → 4 |
| 输出形状与输入不同，需要后端实现 | 1 → 2 → 3(CPU) → 4 → 5 |
| 输出形状与输入完全一致，需要后端实现 | 1 → 3(CPU) → 4 → 5 |

---

## 开始执行

**现在请打开 `skills/add-new-op/step1-schema.md`，开始步骤 1。**
