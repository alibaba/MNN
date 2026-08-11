# 步骤 4：单元测试

> **目标**：编写并运行单元测试，验证算子的计算正确性。
>
> **前置条件**：步骤 3 已通过（计算实现编译成功）。
>
> **这是最关键的步骤**：算子是否正确完全由单元测试决定。

---

## 4.1 编写单元测试

在 `test/op/` 下创建 `MyCustomOpTest.cpp`：

```cpp
//
//  MyCustomOpTest.cpp
//  MNNTests
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class MyCustomOpTest : public MNNTestCase {
public:
    virtual ~MyCustomOpTest() = default;

    virtual bool run(int precision) {
        // ===== 测试用例 1：基本功能 =====
        {
            // 1. 创建输入
            auto input = _Input({2, 3}, NCHW, halide_type_of<float>());
            std::vector<float> inputData = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
            };
            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

            // 2. 执行算子
            // 方式 A：如果有表达式 API
            // auto output = _MyCustomOp(input);

            // 方式 B：手动构造 Op
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type = MNN::OpType_MyCustomOp;
            // 如有参数：
            // op->main.type = MNN::OpParameter_MyCustomOpParam;
            // op->main.value = new MNN::MyCustomOpParamT;
            // auto param = op->main.AsMyCustomOpParam();
            // param->axis = 0;
            auto expr = MNN::Express::Expr::create(op.get(), {input});
            auto output = MNN::Express::Variable::create(expr);

            // 3. 预期结果
            std::vector<float> expectedOutput = {
                // 填入预期的计算结果
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
            };

            // 4. 对比
            auto outputPtr = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(outputPtr, expectedOutput.data(),
                                                    expectedOutput.size(), 0.005)) {
                MNN_ERROR("MyCustomOpTest test1 failed!\n");
                return false;
            }
        }

        // ===== 测试用例 2：边界条件 =====
        {
            // 测试空输入、单元素、大 Tensor 等边界情况
            // ...
        }

        // ===== 测试用例 3：不同数据类型 =====
        {
            // 测试 int32, float16 等
            // ...
        }

        return true;
    }
};

// 注册测试
MNNTestSuiteRegister(MyCustomOpTest, "op/MyCustomOp");
```

### 关键测试 API

| API | 说明 |
|-----|------|
| `_Input(shape, format, type)` | 创建输入变量 |
| `input->writeMap<float>()` | 获取可写数据指针 |
| `output->readMap<float>()` | 触发计算并获取只读数据指针 |
| `checkVectorByRelativeError<T>(got, expect, size, tol)` | 对比结果（相对误差） |
| `checkVector<T>(got, expect, size, tol)` | 对比结果（绝对误差） |
| `MNNTestSuiteRegister(Class, "op/Name")` | 注册测试用例 |

### 构造 Op 的方式

如果算子已有表达式 API（在 `include/MNN/expr/ExprCreator.hpp` 中），直接调用：

```cpp
auto output = _CosineSimilarity(input_a, input_b, input_dim);
```

如果没有表达式 API，手动构造：

```cpp
std::unique_ptr<MNN::OpT> op(new MNN::OpT);
op->type = MNN::OpType_MyCustomOp;
auto expr = MNN::Express::Expr::create(op.get(), {input1, input2});
auto output = MNN::Express::Variable::create(expr);
```

---

## 4.2 编译测试

```bash
cd build
cmake .. -DMNN_BUILD_TEST=ON
make -j$(nproc)
```

---

## 4.3 运行测试

```bash
cd build
# 运行特定算子测试
./run_test.out op/MyCustomOp

# 如果想确认不影响其他算子
./run_test.out op/
```

### 正常输出示例

```
Test op/MyCustomOp passed!
```

### 失败输出示例

```
MyCustomOpTest test1 failed!
Test op/MyCustomOp failed!
```

---

## 4.4 测试用例设计指南

一个好的单元测试应覆盖：

| 测试维度 | 说明 | 示例 |
|----------|------|------|
| **基本功能** | 标准输入，验证计算正确 | 小矩阵验证 |
| **不同形状** | 1D / 2D / 3D / 4D | `{4}`, `{2,3}`, `{2,3,4}`, `{1,2,3,4}` |
| **边界大小** | 单元素、大 Tensor | `{1}`, `{1,1,1,1}`, `{16,32,64,64}` |
| **不同数据类型** | float, int32 | `halide_type_of<float>()`, `halide_type_of<int>()` |
| **参数变化** | 不同 axis、不同 keepDims | axis=0, axis=1, axis=-1 |
| **数值边界** | 0, 负数, NaN, Inf | `0.0f`, `-1.0f`, `NAN` |

### 预期结果的计算

预期结果应该用**独立的方法**计算（不要用自己的实现计算预期值）：

```
✅ 好的方式：
- 手工计算小规模案例
- 用 Python/NumPy 计算参考值
- 用其他框架（PyTorch/TF）计算对照

❌ 不好的方式：
- 用自己实现的算子来生成"预期"结果
```

---

## 步骤 4 测试标准

### 通过标准

- [ ] 单元测试编译通过
- [ ] `./run_test.out op/MyCustomOp` 输出 `passed`
- [ ] 至少覆盖 3 种测试场景（基本功能 + 不同形状 + 边界条件）
- [ ] 预期结果由独立方法计算（不循环论证）

### 常见错误与排查

| 错误 | 原因 | 排查方式 |
|------|------|---------|
| 结果全是 0 | 算子没有执行或数据指针为空 | 检查 `readMap`/`writeMap` 的调用顺序 |
| 结果与预期偏差大 | 计算逻辑错误 | 打印中间结果，逐步对比 |
| Crash / Segfault | 内存越界或空指针 | 检查 `onComputeSize` 的输出形状是否正确 |
| 结果正确但偶尔失败 | 浮点精度问题 | 增大 tolerance（如 0.005 → 0.01） |
| `op/MyCustomOp not found` | 测试未注册 | 检查 `MNNTestSuiteRegister` 的第二个参数 |

### 失败处理

- **结果全错** → 检查步骤 3 的计算逻辑
- **部分用例失败** → 检查边界条件处理
- **Crash** → 检查步骤 2 的形状计算和内存访问
- **在所有测试通过之前，不要进入步骤 5**

---

## 4.5 可选：添加表达式 API

如果算子需要被用户方便调用，可以在 `include/MNN/expr/ExprCreator.hpp` 中添加表达式函数：

```cpp
// 在 ExprCreator.hpp 中
MNN_PUBLIC VARP _MyCustomOp(VARP input, int axis = 0);
```

对应的实现在 `express/` 下。这一步是可选的，根据实际需求决定。

---

## 下一步

- **如果只需要 CPU 实现（或已用几何计算）** → 🎉 算子支持完成！
- **如果需要扩展到其他后端（Metal/OpenCL/Vulkan/CUDA）** → 进入 `step5-optimize.md`
