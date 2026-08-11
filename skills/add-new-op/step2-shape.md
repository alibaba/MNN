# 步骤 2：形状计算

> **目标**：实现算子的输出形状推理逻辑。给定输入 Tensor 的形状和算子参数，计算输出 Tensor 的形状和数据类型。
>
> **前置条件**：步骤 1 已通过（Schema 定义完成）。
>
> **跳过条件**：如果算子的输出形状与第 1 个输入 Tensor 完全一致（维度数量、每个维度的大小都相同），可以跳过此步。

---

## 2.1 理解形状计算的含义

形状计算 **不做实际计算**，只推理输出 Tensor 的：
- `dimensions`：维度数量
- `dim[i].extent`：第 i 维的大小
- `buffer().type`：数据类型（float、int 等）

**示例**：
- MatMul `[M, K] × [K, N]` → 输出 `[M, N]`
- Reduction sum `[B, C, H, W]` axis=1 → 输出 `[B, H, W]`（或 `[B, 1, H, W]` keepDims）
- Reshape `[B, C*H*W]` → 输出 `[B, C, H, W]`

---

## 2.2 创建形状计算文件

在 `source/shape/` 下创建 `ShapeMyCustomOp.cpp`：

```cpp
//
//  ShapeMyCustomOp.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
class MyCustomOpSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // 1. 输入检查
        MNN_ASSERT(inputs.size() >= 1);
        auto input = inputs[0];
        auto output = outputs[0];

        // 2. 从 op 获取参数（如有）
        // auto param = op->main_as_MyCustomOpParam();
        // int axis = param->axis();

        // 3. 计算输出形状
        // 示例：输出与输入相同
        output->buffer().dimensions = input->dimensions();
        for (int i = 0; i < input->dimensions(); ++i) {
            output->setLength(i, input->length(i));
        }

        // 4. 设置输出数据类型
        output->buffer().type = input->getType();

        // 5. 设置输出的数据格式（通常 NCHW）
        TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;

        return true;
    }

    // 可选：计算 FLOPS
    virtual float onComputeFlops(const MNN::Op* op,
                                 const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        // 返回计算量估算（可以简单返回输出元素数量）
        float flops = 1.0f;
        auto output = outputs[0];
        for (int i = 0; i < output->dimensions(); ++i) {
            flops *= output->length(i);
        }
        return flops;
    }
};

// 注册形状计算
// 参数 1: 类名
// 参数 2: OpType 枚举值
REGISTER_SHAPE(MyCustomOpSizeComputer, OpType_MyCustomOp);

// 如果算子有 const 输入（如 axis 通过 Tensor 传入），使用：
// REGISTER_SHAPE_INPUTS(MyCustomOpSizeComputer, OpType_MyCustomOp, (std::vector<int>{2}));
// 其中 {2} 表示第 3 个输入（index=2）是 const 输入，在形状计算时就需要读取其值

} // namespace MNN
```

### 关键 API 说明

| API | 说明 |
|-----|------|
| `input->dimensions()` | 获取输入维度数 |
| `input->length(i)` | 获取第 i 维大小 |
| `output->buffer().dimensions = N` | 设置输出维度数 |
| `output->setLength(i, size)` | 设置输出第 i 维大小 |
| `output->buffer().type = input->getType()` | 设置输出数据类型 |
| `op->main_as_XXXParam()` | 获取算子参数 |
| `REGISTER_SHAPE(Class, OpType)` | 注册形状计算 |
| `REGISTER_SHAPE_INPUTS(Class, OpType, constInputs)` | 注册形状计算（指定 const 输入） |

---

## 2.3 运行注册脚本

```bash
# 在项目根目录下运行
python3 tools/script/register.py
```

这会自动更新 `source/shape/ShapeRegister.cpp`。

---

## 步骤 2 测试标准

### 测试方法

```bash
# 1. register.py 运行成功
python3 tools/script/register.py

# 2. 确认注册文件已更新
grep "MyCustomOp" source/shape/ShapeRegister.cpp
# 应该找到对应的 extern 声明和函数调用

# 3. cmake + 编译通过
cd build
cmake .. -DMNN_BUILD_TEST=ON
make -j$(nproc)
# 应该编译无错误
```

### 通过标准

- [ ] `register.py` 运行无错误
- [ ] `ShapeRegister.cpp` 中包含新算子的注册
- [ ] 编译通过，无链接错误

### 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| `REGISTER_SHAPE` 编译错误 | OpType 名称拼写与 Schema 不一致 | 检查 `OpType_MyCustomOp` 的准确拼写 |
| 链接错误 `undefined symbol` | register.py 未运行 | 重新运行 `python3 tools/script/register.py` |
| `ShapeRegister.cpp` 未更新 | 文件放错目录 | 确保在 `source/shape/` 目录下 |

---

## 下一步

**步骤 2 通过后，进入 `step3-compute.md`（步骤 3：计算实现）。**
