# 步骤 5：集成与回归测试

> **目标**：将优化后的实现集成到 MNN 主体并运行全量回归测试，确保不影响其他算子。
>
> **前置条件**：步骤 3 或步骤 4 已通过（C++ 优化或汇编优化完成且正确性验证通过）。
>
> **复杂度**：中（需要编译运行全量测试）

---

## 5.1 注册到 CoreFunctions

将新的函数指针注册到 MNN 运行时选择的路径中。

注册位置根据指令集不同：

| 指令集 | 注册文件 |
|--------|---------|
| FP32 NEON (arm64) | `source/backend/cpu/arm/arm64/MNNCoreFunctions.cpp` 或类似 |
| FP16 (arm82) | `source/backend/cpu/arm/arm64/MNNCoreFunctions_ARM82.cpp` 或类似 |
| I8MM (arm86) | ARM86 相关初始化文件 |
| SME2 | SME2 相关初始化文件 |
| FP32 NEON (arm32) | `source/backend/cpu/arm/arm32/` 对应文件 |

```cpp
// 典型注册方式
void MNNCoreFunctionInit() {
    auto gcore = MNNGetCoreFunctions();
    // ...
    // 注册优化后的函数
    gcore->MNNPackedMatMul = MNNPackedMatMul_optimized;
    gcore->MNNPackedMatMulRemain = MNNPackedMatMulRemain_optimized;
    // ...
}
```

---

## 5.2 CMakeLists 更新

如果添加了新的 `.S` 汇编文件，需要在 CMakeLists.txt 中添加：

```bash
# 查找对应的 CMakeLists.txt
grep -r "MNNPackedMatMul" source/backend/cpu/CMakeLists.txt
```

确认新文件在编译列表中。

---

## 5.3 全量回归测试

### 5.3.1 算子正确性测试

```bash
cd build

# 运行所有算子测试
./run_test.out op/

# 如果优化的是 MatMul 相关，特别关注：
./run_test.out op/MatMul
./run_test.out op/BatchMatMul
./run_test.out op/Convolution
./run_test.out op/InnerProduct
./run_test.out op/Attention

# 量化相关
./run_test.out op/ConvInt8
```

### 5.3.2 性能回归测试

```bash
# 运行所有 speed 测试
./run_test.out speed/

# 关注关键测试
./run_test.out speed/MatMulTest
./run_test.out speed/MatMulBConstTest
./run_test.out speed/ConvSpeedInt8Test
```

### 5.3.3 模型级端到端测试（如有条件）

```bash
# 用实际模型验证
./llm_demo model_dir prompt
# 检查输出是否正常
```

---

## 5.4 代码质量审查

优化代码在功能正确后，提交前必须通过以下审查：

### 编译质量

```bash
# 确认无编译 warning
make -j$(nproc) 2>&1 | grep -i warning
```

### 代码一致性检查

```
□ 所有在 .hpp 中的注释与 .cpp 中的实际实现一致
  （特别是 buffer 大小、数据排布等数值/格式描述）
□ 没有声明但未使用的变量（优化迭代中常见遗留）
□ onResize 中的每个 onAcquireBuffer 都有对应的 onReleaseBuffer
□ 没有残留的调试代码（printf、临时变量、注释掉的旧代码）
□ 保持手动循环的地方都有注释说明不替换的原因
□ 使用 MNN 函数 in-place (dst==src) 的地方都已确认安全
```

### 设计合理性检查

```
□ 多个小 buffer 是否可合并为一块连续分配？（减少分配次数）
□ 线程数获取是否有不必要的重复调用？（可在 onResize 缓存）
□ 是否有不必要的 memcpy？（检查是否可以 in-place 或调整指针）
□ 朴素循环内是否有可提取到循环外的不变量？
```

---

## 5.5 性能报告

**报告文件位置**：`<算子名>_optimization.md`

**报告必须包含以下所有章节**，缺少任何一项则视为不通过。

**⚠️ 性能数据必须是实测数据**，不能写"预期"或"估计"值。如果无法在当前环境实测（如需要 ARM 设备），需明确标注"待实测"，并给出将要使用的测试命令。

### 报告模板

```markdown
# Xxx ARM CPU 性能优化报告

## 1. 优化方案

### 1.1 计算拆解与函数替换

逐项列出原始代码中的每个计算逻辑，以及替换为了什么 MNN 函数：

| # | 计算描述 | 原实现 | 优化后 | 说明 |
|---|---------|--------|--------|------|
| 1 | ... | ... | `MNN 函数` | 替换理由 |
| 2 | ... | ... | 保持手动循环 | ⚠️ 不替换原因：... |

### 1.2 多线程优化

- 并行维度: ____（例如按 Head 划分）
- 线程数: ____
- 是否存在写冲突: ____

### 1.3 数据排布变更

如有排布变更，说明原因和影响。

### 1.4 新增汇编 kernel（如有）

如果编写了新的汇编 kernel，列出文件路径、针对的指令集、实现的计算。

## 2. 性能数据（必须是实测数据）

### 2.1 测试环境

- 平台: ____（芯片型号）
- CPU 特性: fp16=__, sdot=__, i8mm=__, sme2=__
- 编译选项: ____
- 线程数: ____

### 2.2 基线 vs 优化后对比

**所有用例都必须同时有基线和优化后的实测数据**。

| 用例 | 参数 | 基线(ms) | 优化后(ms) | 加速比 | GFLOPS(前) | GFLOPS(后) |
|------|------|---------|-----------|--------|-----------|-----------|
| ... | ... | xx.xx | xx.xx | **x.xx×** | xx.xx | xx.xx |

### 2.3 各优化手段的贡献分解（如可测量）

| 优化手段 | 单独效果 | 说明 |
|---------|---------|------|
| 函数替换 | ~x.xx× | |
| 多线程 | ~x.xx× | |
| 内存池替换 | ~x.xx× | |

## 3. 正确性验证

- [ ] `./run_test.out op/Xxx`: ✅ PASSED
- [ ] `./run_test.out speed/XxxSpeed`: ✅ PASSED
- [ ] 全量 op/ 测试无回归: ✅
- [ ] 端到端模型测试（如有条件）: ✅ / ⏳

## 4. 代码质量

- [ ] 编译无 warning: ✅
- [ ] .hpp 注释与 .cpp 实现一致: ✅
- [ ] 无未使用的变量/include: ✅
- [ ] 保持手动循环的地方均有注释: ✅
- [ ] in-place 调用均已验证安全: ✅

## 5. 修改文件清单

| 文件 | 修改类型 | 修改内容 |
|------|---------|---------|
| ... | ... | ... |

## 6. 未优化 / 后续方向

明确列出哪些**没有**优化，以及原因：

| 未优化项 | 原因 | 后续建议 |
|---------|------|---------|
| ... | ... | ... |
```

---

## 步骤 5 测试标准

### 通过标准

- [ ] **全量 op/ 测试通过**：`./run_test.out op/` 无失败
- [ ] **性能无回归**：`./run_test.out speed/` 无性能下降
- [ ] **优化目标达成**：性能数据满足步骤 2 的预期
- [ ] **代码质量审查通过**：5.4 清单全部勾选
- [ ] **性能报告完整**：包含全部 6 个章节（优化方案、性能数据、正确性验证、代码质量、文件清单、未优化项）
- [ ] **有实测数据**：每个用例都有基线和优化后的实测对比数据（ms + GFLOPS + 加速比）

### 失败处理

- **单个 op 测试失败** → 回到步骤 3/4 检查正确性
- **性能回归** → 检查是否影响了其他算子的函数指针注册
- **编译 warning** → 回到 5.4 清理代码
- **报告缺少实测数据** → 必须运行 benchmark 获取数据

---

## 完成

**恭喜！当全量测试通过、代码质量审查通过且性能报告完成后，ARM CPU 优化工作完成。**

### 部分完成也是可以接受的

```
✅ 已完成：
- C++ 多线程优化
- FP32 NEON 汇编
- 正确性测试通过
- 代码质量审查通过

⏳ 待完成：
- FP16 版本
- SDOT/I8MM 版本
- SME2 版本（需要硬件支持）
```
