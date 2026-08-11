# 步骤 2：验证和集成

> **目标**：将优化后的实现集成到 MNN 主体并运行全量回归测试，确保不影响其他算子。
>
> **前置条件**：步骤 1 已通过。
>
> **复杂度**：中（需要编译运行全量测试）
>
> **参考**：正确性验证标准和真机测试方法，见 `SKILL.md` "正确性验证" 章节。

---

## 2.1 全量回归测试

### 2.1.1 算子正确性测试

重新编译 + 推送见 SKILL.md「编译与真机运行」，然后跑正确性测试：

```bash
# 优化算子单测
adb shell "cd /data/local/tmp/MNN && ./run_test.out op/XxxTest 3 1 68"
# 全量 op 测试
adb shell "cd /data/local/tmp/MNN && ./run_test.out op/ 3 1 68"
```

**通过标准**: 所有测试显示 `all tests passed`

### 2.1.2 性能回归测试

```bash
adb shell "cd /data/local/tmp/MNN && ./run_test.out speed/ 3 1 68"
```

**检查要点**:
- 优化的算子性能是否符合预期
- 其他算子性能是否有异常下降

### 2.1.3 模型级端到端测试（如有条件）

使用 `SKILL.md` "正确性验证" 中的真机测试入口命令。必须关闭 sampler 随机性（`temperature: 0.0`, `sampler_type: greedy`）。

```bash
adb shell "cd /data/local/tmp/MNN && \
  LD_LIBRARY_PATH=. timeout 180 ./llm_demo <model>/config_cl.json prompt.txt 2>&1 | tail -20"
```

---

## 2.2 代码质量审查

### 2.2.1 通用检查

```
□ 所有 .hpp 注释与 .cpp 实际实现一致
□ 没有声明但未使用的变量
□ 没有残留的调试代码（printf、临时变量、注释掉的旧代码）
□ Kernel 文件中的注释与实际代码一致
```

### 2.2.2 OpenCL 特定检查

```
□ 已运行 python3 opencl_codegen.py . . 更新 kernel 映射（参考 SKILL.md ".cl 修改流程"）
□ GWS/LWS 配置合理，没有超出设备限制
□ Local memory 使用量在限制范围内（通常 32KB）
□ 所有 kernel 参数类型正确（__global, __local, __private）
□ barrier 的使用都是必要的
□ 没有不必要的数据拷贝
□ 新加 quant bit 时，shader 中所有 4 处分支都已覆盖（参考 SKILL.md ".cl 修改流程"）
```

### 2.2.3 设计合理性检查

```
□ 多个小 kernel 是否可合并？（减少启动开销）
□ 是否有不必要的 kernel 调用或数据传输（CPU<->GPU）？
□ Fallback 机制是否完善？（处理超大数据，如 local memory 超限场景）
□ 是否考虑了不同设备的兼容性？（Adreno / Mali）
```

---

## 2.3 性能报告

**报告文件**：`<算子名>_opencl_optimization.md`

**报告必须包含以下章节**，缺少任何一项则视为不通过。**性能数据必须是实测数据**，不能写"预期"或"估计"值。

### 报告模板

```markdown
# Xxx OpenCL 性能优化报告

## 1. 优化概述
- 算子名称、目标平台
- 优化前/后性能、总体加速比
- 采用的主要优化技术列表

## 2. 性能数据（实测）
- 基线 vs 优化后对比（按场景分组，包含每个 kernel 的耗时）
- 不同参数组合性能表
- 各优化手段的贡献分解
- 优化历程（每次尝试的方案、结果、分析、决策）

## 3. 正确性验证
- op 测试、speed 测试、全量回归、端到端测试的结果

## 4. 代码质量
- 编译 warning、注释一致性、codegen、GWS/LWS、local memory、fallback 等检查项

## 5. 修改文件清单

## 6. 技术细节
- 关键优化技术说明（实现、收益、限制）
- 性能瓶颈分析（优化前后对比）
- 不同场景性能差异的原因

## 7. 未优化 / 后续方向

## 8. 经验总结
- 成功经验、踩过的坑、优化建议
```

---

## 2.4 提交前最终检查

```bash
# 检查所有修改的文件
git status

# 确认包含：
# - source/backend/opencl/execution/cl/xxx.cl (kernel 实现)
# - source/backend/opencl/execution/buffer/XxxExecution.cpp (调用代码)
# - source/backend/opencl/execution/buffer/XxxExecution.hpp (头文件，如有修改)
# - xxx_opencl_optimization.md (性能报告)
```

### 提交信息格式

```
[OpenCL:Perf] Optimize Xxx kernel performance

- 优化技术1
- 优化技术2

Performance: Decode xx× / Prefill xx×
Platform: Android SM8350 (Adreno 660)
All op/ tests passed
```

---

## 2.5 沉淀经验到手册

任务较复杂、且本次产生了**可复用的方法论**（新技巧 / 非显而易见的坑 / 新分析方法 / 验证或排除了某候选方向）时，把它回写到 `optimization-handbook.md`——OpenCL kernel/访存级技巧与陷阱的唯一来源。

- 触发判断表、回写位置（§1/§2/§3/§5/§6）和「只写方法论不写流水账」的要求见 `SKILL.md` "收尾：沉淀经验到手册"。
- 单次任务的具体数字/文件清单留在 2.3 的性能报告里，**不进手册**。
- 纯套用已有技巧、无新经验时**跳过本步**，不要为凑数塞内容。

---

## 通过标准

- [ ] **全量 op/ 测试通过**：`./run_test.out op/` 无失败
- [ ] **性能报告完整**：包含全部 8 个章节
- [ ] **有实测数据**：每个用例都有基线和优化后的实测对比数据
- [ ] **代码质量审查通过**：所有检查项都已确认
- [ ] **已运行 opencl_codegen.py**：kernel 映射已更新
- [ ] **可复用经验已回写手册**（或已确认本次无可沉淀经验）

### 失败处理

- **单个 op 测试失败** → 回到步骤 0/1 检查正确性
- **报告缺少实测数据** → 必须运行 benchmark 获取数据
- **编译有 warning** → 修复 warning
- **忘记运行 codegen** → 运行 `python3 opencl_codegen.py . .`
