# 步骤 3：计算实现

> **目标**：实现算子的实际计算逻辑。优先尝试几何计算（拆解为已有算子组合），如果不可行再实现 CPU 后端。
>
> **前置条件**：步骤 1 已通过（Schema 定义完成），如果输出形状与输入不同则步骤 2 也已通过。

---

## 3.0 选择实现方式

**优先级**：几何计算 > CPU 后端实现

### 判断是否可以用几何计算

几何计算的本质是**将新算子拆解为已有 MNN 算子的组合**。适合的情况：

```
✅ 可以几何拆解的算子：
- CosineSimilarity = Mul + Reduce(Sum) + Sqrt + Div
- LayerNorm = Mean + Sub + Variance + Rsqrt + Mul + Add
- Softmax = Exp + ReduceSum + Div
- 各种 element-wise 组合算子

❌ 不适合几何拆解的算子：
- 需要复杂循环/条件分支的（如 NMS, Sort）
- 需要特殊内存访问模式的（如 Im2Col, Gather 的特殊变体）
- 性能敏感且几何拆解效率远低于原生实现的
```

**判断结果**：
- **可以拆解** → 走 3.A（几何计算）
- **不可拆解** → 走 3.B（CPU 后端实现）

---

## 3.A 几何计算实现

### 3.A.1 创建几何计算文件

在 `source/geometry/` 下创建 `GeometryMyCustomOp.cpp`：

```cpp
//
//  GeometryMyCustomOp.cpp
//  MNN
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
class GeometryMyCustomOp : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs, Context& context,
                           CommandBuffer& res) const override {
        auto input  = inputs[0];
        auto output = outputs[0];

        // ===== 拆解为已有算子组合 =====

        // 示例 1：创建中间 Tensor
        // std::shared_ptr<Tensor> tmpTensor;
        // tmpTensor.reset(Tensor::createDevice<float>({batch, channel, height, width}));
        // auto des = TensorUtils::getDescribe(tmpTensor.get());
        // des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        // res.extras.emplace_back(tmpTensor);

        // 示例 2：调用 Binary 算子
        // auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL,
        //     inputA, inputB, outputTensor);
        // res.command.emplace_back(std::move(cmd));

        // 示例 3：调用 Unary 算子
        // auto cmd = GeometryComputerUtils::makeUnary(UnaryOpOperation_SQRT,
        //     inputTensor, outputTensor);
        // res.command.emplace_back(std::move(cmd));

        // 示例 4：调用 Reduce 算子
        // auto cmd = GeometryComputerUtils::makeReduce(ReductionType_SUM,
        //     inputTensor, outputTensor);
        // res.command.emplace_back(std::move(cmd));

        // 示例 5：虚拟 Tensor（内存重映射，零拷贝 reshape/transpose）
        // auto outputDes = TensorUtils::getDescribe(output);
        // outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        // Tensor::InsideDescribe::Region desReg;
        // desReg.size[0] = dim0;
        // desReg.size[1] = dim1;
        // desReg.size[2] = dim2;
        // desReg.dst.offset = 0;
        // desReg.dst.stride[0] = dim1 * dim2;
        // desReg.dst.stride[1] = dim2;
        // desReg.dst.stride[2] = 1;
        // desReg.src.offset = 0;
        // desReg.src.stride[0] = src_stride0;
        // desReg.src.stride[1] = src_stride1;
        // desReg.src.stride[2] = 1;
        // desReg.origin = input;
        // outputDes->regions.emplace_back(std::move(desReg));

        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryMyCustomOp);
    GeometryComputer::registerGeometryComputer(comp, {OpType_MyCustomOp});
}

REGISTER_GEOMETRY(GeometryMyCustomOp, _create);

} // namespace MNN
```

### 常用几何计算工具函数

| 函数 | 作用 |
|------|------|
| `GeometryComputerUtils::makeBinary(op, a, b, out)` | 二元运算（加减乘除等） |
| `GeometryComputerUtils::makeUnary(op, in, out)` | 一元运算（sqrt, exp, log 等） |
| `GeometryComputerUtils::makeReduce(type, in, out)` | 规约运算（sum, mean, max 等） |
| `GeometryComputerUtils::makeMatMul(a, b, out, ...)` | 矩阵乘法 |
| `context.allocConst(op, shape, type)` | 创建常量 Tensor |
| `Tensor::createDevice<float>(shape)` | 创建设备中间 Tensor |
| `MEMORY_VIRTUAL` + `regions` | 零拷贝内存重映射 |

### 3.A.2 运行注册脚本并编译

```bash
python3 tools/script/register.py
cd build
cmake .. -DMNN_BUILD_TEST=ON
make -j$(nproc)
```

**几何计算完成后，跳到步骤 4（单元测试）。**

---

## 3.B CPU 后端实现

### 3.B.1 创建头文件

在 `source/backend/cpu/` 下创建 `CPUMyCustomOp.hpp`：

```cpp
//
//  CPUMyCustomOp.hpp
//  MNN
//

#ifndef CPUMyCustomOp_hpp
#define CPUMyCustomOp_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUMyCustomOp : public Execution {
public:
    CPUMyCustomOp(Backend* backend, const Op* op);
    virtual ~CPUMyCustomOp() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) override;

private:
    // 算子参数
    // int mAxis = 0;
    // 临时缓存
    // Tensor mCache;
};
} // namespace MNN

#endif // CPUMyCustomOp_hpp
```

### 3.B.2 创建实现文件

在 `source/backend/cpu/` 下创建 `CPUMyCustomOp.cpp`：

```cpp
//
//  CPUMyCustomOp.cpp
//  MNN
//

#include "CPUMyCustomOp.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

CPUMyCustomOp::CPUMyCustomOp(Backend* backend, const Op* op) : Execution(backend) {
    // 从 op 中读取参数
    // auto param = op->main_as_MyCustomOpParam();
    // mAxis = param->axis();
}

ErrorCode CPUMyCustomOp::onResize(const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs) {
    // 在此申请临时缓存（仅在输入形状变化时调用）
    // backend()->onAcquireBuffer(&mCache, Backend::DYNAMIC);
    // backend()->onReleaseBuffer(&mCache, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUMyCustomOp::onExecute(const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs) {
    // 1. 获取输入输出数据指针
    auto input  = inputs[0];
    auto output = outputs[0];
    auto inputPtr  = input->host<float>();
    auto outputPtr = output->host<float>();

    MNN_ASSERT(inputPtr != nullptr);
    MNN_ASSERT(outputPtr != nullptr);

    // 2. 获取形状信息
    int totalSize = 1;
    for (int i = 0; i < input->dimensions(); ++i) {
        totalSize *= input->length(i);
    }

    // 3. 实际计算（简单单线程版本，先确保正确性）
    for (int i = 0; i < totalSize; ++i) {
        outputPtr[i] = inputPtr[i]; // ← 替换为实际计算逻辑
    }

    return NO_ERROR;
}

// 注册 Creator
class CPUMyCustomOpCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op,
                                Backend* backend) const override {
        return new CPUMyCustomOp(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUMyCustomOpCreator, OpType_MyCustomOp);

} // namespace MNN
```

### 3.B.3 关键 API 说明

| API | 说明 |
|-----|------|
| `input->host<float>()` | 获取 float 数据指针 |
| `input->host<int32_t>()` | 获取 int32 数据指针 |
| `input->dimensions()` | 维度数 |
| `input->length(i)` | 第 i 维大小 |
| `input->elementSize()` | 总元素数 |
| `backend()->onAcquireBuffer(&buf, Backend::DYNAMIC)` | 申请临时缓存 |
| `backend()->onReleaseBuffer(&buf, Backend::DYNAMIC)` | 释放临时缓存（可被复用） |
| `NO_ERROR` | 返回成功 |
| `INPUT_DATA_ERROR` | 返回输入错误 |
| `REGISTER_CPU_OP_CREATOR(Creator, OpType)` | 注册 CPU 算子 |

### 3.B.4 运行注册脚本并编译

```bash
python3 tools/script/register.py
cd build
cmake .. -DMNN_BUILD_TEST=ON
make -j$(nproc)
```

---

## 步骤 3 测试标准

### 测试方法

```bash
# 1. register.py 运行成功
python3 tools/script/register.py

# 2. 编译通过
cd build
cmake .. -DMNN_BUILD_TEST=ON
make -j$(nproc)
# 应该无编译和链接错误
```

### 通过标准

- [ ] **register.py 运行无错误**
- [ ] **编译无错误、无链接错误**
- [ ] 如果是几何计算：`GeometryOPRegister.cpp` 中包含新算子
- [ ] 如果是 CPU 实现：注册文件中包含新算子

### 编译通过 ≠ 逻辑正确

步骤 3 只保证代码能编译。**逻辑正确性在步骤 4（单元测试）中验证**。

### 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| 编译错误 `OpType_MyCustomOp undeclared` | Schema 头文件未更新 | 重新执行步骤 1 的 `generate.sh` |
| 链接错误 `undefined reference` | register.py 未运行 | 重新运行 `python3 tools/script/register.py` |
| 编译错误头文件未找到 | include 路径错误 | 检查 `#include` 的路径 |

---

## 下一步

**步骤 3 通过后，进入 `step4-test.md`（步骤 4：单元测试）。**
