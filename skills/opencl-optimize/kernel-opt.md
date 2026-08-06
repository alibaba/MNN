# 步骤 1：优化 Kernel 性能

> **目标**：基于基线数据，尝试多种优化手段提升 kernel 性能。
>
> **前置条件**：步骤 0 已通过（基线数据已获取）。
>
> **复杂度**：高（需要编译运行）
>
> **注意**：优先保证正确性，性能精调可后续迭代。
>
> **参考**：codegen 流程和正确性验证方法见 `SKILL.md`；优化技巧和常见陷阱见 `optimization-handbook.md`。

---

## 1.0 分析性能瓶颈

### 1.0.1 读取 kernel 代码

在 `source/backend/opencl/execution/cl/` 目录中找到对应的 kernel 文件，读取瓶颈 kernel 的代码。

### 1.0.2 瓶颈定量分析

先用 `optimization-handbook.md` §1 中的计算强度公式判断瓶颈类型（compute-bound / memory-bound），再针对性分析 kernel 代码：

```markdown
## Kernel 性能分析

**Kernel 名称**: xxx_kernel
**代码位置**: source/backend/opencl/execution/cl/xxx.cl
**性能数据**: 占总时间 xx%，绝对耗时 xx us

**计算强度**: xx FLOPs/Byte（参考 `optimization-handbook.md` §1.1-1.2 中的 ridge point 数据）
**瓶颈类型**: memory-bound / compute-bound / 均衡
**BW 利用率**（memory-bound 时）: xx%

**代码级分析**:

1. **内存访问模式**
   - [ ] Global memory 读写次数
   - [ ] 是否有重复读取同一数据？
   - [ ] 访问是否连续（合并访问）？
   - [ ] 是否可以用 local memory 缓存？

2. **计算特征**
   - [ ] 寄存器使用: 是否有大量私有数组？
   - [ ] 循环嵌套: 是否有多层循环？
   - [ ] 数据依赖: 循环间是否有依赖？
   - [ ] 并行度: 当前并行粒度是否合理？

3. **Work-Group 组织**
   - [ ] 当前 GWS/LWS 配置
   - [ ] 每个 work-item 的工作量
   - [ ] 是否充分利用 GPU 并行能力？
```

---

## 1.1 优化策略选择

**先按瓶颈类型选杠杆**：用计算强度 AI 与 ridge point 比较,判断 memory-bound / compute-bound / 均衡,再选方向——判断表以 `optimization-handbook.md` §1.3 为准(唯一来源),不在此重复。

下面的决策树是按**症状**导航的另一条互补路径:

### 优化决策树

```
分析 kernel 瓶颈 →
  ├─ Global memory 访问频繁？
  │   ├─ 数据被多次读取？ → 用 Local Memory 缓存
  │   ├─ 访问不连续？ → 调整数据排布或访问模式
  │   └─ 写入频繁？ → 用私有变量累积，最后写入
  │
  ├─ 寄存器压力大（私有数组多）？
  │   ├─ 中间结果可以不存储？ → 消除中间数组
  │   ├─ 可以用 local memory 替代？ → 改用 local 数组
  │   └─ 可以减少工作量？ → 调整 work-group 组织
  │
  ├─ 循环有数据依赖无法并行？
  │   ├─ 可以拆解为独立部分？ → 拆分为多个 kernel
  │   ├─ 可以用并行 reduce？ → 实现并行归约算法
  │   └─ 依赖无法消除？ → 保持串行，优化其他部分
  │
  ├─ 并行度不足？
  │   ├─ 可以增加并行维度？ → 使用 2D/3D work-group
  │   ├─ 每个 work-item 工作量太小？ → 增加每个 work-item 的工作量
  │   └─ 每个 work-item 工作量太大？ → 减少工作量，增加 work-item 数
  │
  └─ 计算密集但访问连续？
      ├─ 可以向量化？ → 使用 float4/float8
      └─ 已经很优化？ → 考虑算法级优化
```

### 技巧目录 → 见 handbook（唯一来源）

上面的决策树按**症状**导航；具体技巧的完整定义（适用场景 / 做法 / 注意事项 / 收益 / 实战案例）不在这里重复，一律以 `optimization-handbook.md` 为准：§2 是 kernel/访存级技巧目录，§5 是按难度和收益排序的速查表（含「针对瓶颈」列），§6 是待验证的候选手段。改任何技巧前读对应条目。

> **跨切经验**（详见 handbook）：组合多种技术效果远超单一（LinearAttention 8x→112x）；GEMV（BW bound，原生 packed kernel 更优）与 GEMM（有数据复用，可先反量化再走通用 GEMM）最优策略不同。（handbook 只收 kernel/访存级技巧；host/init 级优化不在其范围。）

---

## 1.2 实施优化

### 每次优化的标准流程

```
1. 记录当前性能数据
2. 实施单个优化（只改一个优化点）
3. 更新 cpp 调用代码（xxxExecution.cpp，参数传递和 GWS/LWS）
4. 更新 kernel 映射（参考 SKILL.md ".cl 修改流程"）：
   cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . .
5. 编译 → 推到真机 → 正确性验证 → 性能测试
6. 评估结果：提升→保留 / 下降→回退 / 正确性失败→修复
```

> 各技巧的代码写法见 `optimization-handbook.md` 对应条目；注意 kernel 内一律用 `FLOAT4/COMPUTE_FLOAT8/CONVERT_FLOAT4` 等宏而非裸 `float4`（精度模式兼容，见 handbook 陷阱 I）。

---

## 1.3 性能验证

每次优化后都要验证（参考 `SKILL.md` "正确性验证" 中的三层 oracle 和误差容忍标准）：

```bash
# 正确性验证
adb shell "cd /data/local/tmp/MNN && ./run_test.out op/XxxTest 3 1 68"

# 性能对比
adb shell "cd /data/local/tmp/MNN && ./run_test.out speed/XxxSpeed 3 1 68"
```

记录优化结果：

```markdown
## 优化尝试X: [优化名称]

**优化方案**: 简要描述
**修改文件**: xxx.cl, xxxExecution.cpp

| 场景 | 优化前(us) | 优化后(us) | 加速比 | 状态 |
|------|-----------|-----------|--------|------|
| decode_H4_d64 | 2,028 | 247 | **8.2x** | OK |

**关键发现**: ...
**决策**: 保留 / 回退 / 继续改进
```

---

## 1.4 迭代优化

### 核心要求：必须多次尝试不同优化手段

> **重要**：不能只尝试一种优化手段就停止！即使第一次优化已经取得不错的加速比，仍然必须继续尝试其他优化技术。原因：
> 1. 单一优化可能不是最优解
> 2. 组合优化效果往往远超单一技术
> 3. 不同场景可能需要不同优化
> 4. 每次尝试都能加深对瓶颈的理解

### 推荐尝试顺序（至少 3 种）

从 `optimization-handbook.md` §5 速查表（按难度排序）里**从低难度往高试**，用上面的决策树按症状挑，至少 3 种：

- **先手（低难度）**：消除中间数组、常量 / offset 折叠（技巧 3）、向量化（技巧 7）
- **再上（中难度）**：image1d 纹理缓存（技巧 2）、local memory + 并行 reduce（技巧 1）、权重重排 + 并行粒度（技巧 6）、register/output tiling（技巧 8，逐档实测甜点）、Decode/Prefill 特化（技巧 5）、2D work-group
- **最后（高难度）**：单 work-item → workgroup 重写（技巧 4）、kernel 拆分、算法级重写、组合多种技术

各条的做法 / 注意 / 收益见 handbook 对应技巧，不在此重复。

### 停止条件（必须同时满足所有条件）

- [ ] 已尝试至少 3 种不同的优化技术
- [ ] 性能已达到目标，或连续 3 次优化尝试都无明显提升（<5%）
- [ ] 每次优化尝试都有完整的性能数据记录

---

## 通过标准

- [ ] 至少尝试了 3 种不同的优化技术
- [ ] 每次优化都有正确性验证
- [ ] 每次优化都有性能数据对比
- [ ] 最终性能相对基线有显著提升（建议 >2x）
- [ ] 所有优化记录已文档化
