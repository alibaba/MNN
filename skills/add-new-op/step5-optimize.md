# 步骤 5：扩展后端与性能优化

> **目标**：将算子扩展到其他硬件后端（Metal/OpenCL/Vulkan/CUDA），并进行性能优化。
>
> **前置条件**：步骤 4 已通过（CPU 单元测试全部通过）。
>
> **注意**：这一步通常逐个后端实施，不需要一次全部完成。

---

## 5.0 优化路线

```
CPU 基础实现（已完成）
│
├─ CPU 多线程优化
│   └─ SIMD 优化（ARM NEON / x86 SSE/AVX）
│
├─ Metal 后端（Apple GPU）
│
├─ OpenCL 后端
│
├─ Vulkan 后端
│
└─ CUDA 后端
```

每完成一个后端，都可以用步骤 4 的单元测试来验证正确性。

---

## 5.1 CPU 多线程优化

在 `onExecute` 中使用 MNN 的线程池：

```cpp
ErrorCode CPUMyCustomOp::onExecute(const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    int threadCount = static_cast<CPUBackend*>(backend())->threadNumber();
    int totalSize = input->elementSize();

    MNN_CONCURRENCY_BEGIN(tId, threadCount) {
        int start = tId * totalSize / threadCount;
        int end = (tId + 1) * totalSize / threadCount;
        for (int i = start; i < end; ++i) {
            output->host<float>()[i] = /* 计算 */;
        }
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}
```

---

## 5.2 Metal 后端

### 5.2.1 创建实现文件

在 `source/backend/metal/` 下创建 `MetalMyCustomOp.hpp` 和 `MetalMyCustomOp.cpp`：

**MetalMyCustomOp.hpp**：
```cpp
#ifndef MetalMyCustomOp_hpp
#define MetalMyCustomOp_hpp

#include "MetalExecution.hpp"

namespace MNN {
class MetalMyCustomOp : public MetalExecution {
public:
    MetalMyCustomOp(Backend* backend, const Op* op);
    virtual ~MetalMyCustomOp() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) override;
    virtual void onEncode(const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
private:
    id<MTLComputePipelineState> mPipeline;
    MTLSize mThreads;
    MTLSize mThreadgroupSize;
};
} // namespace MNN
#endif
```

**MetalMyCustomOp.cpp**（关键部分）：
```cpp
#include "MetalMyCustomOp.hpp"
#include "backend/metal/MetalBackend.hpp"

namespace MNN {

MetalMyCustomOp::MetalMyCustomOp(Backend* backend, const Op* op)
    : MetalExecution(backend) {
    auto mtbn = static_cast<MetalBackend*>(backend);
    mPipeline = [mtbn->context() pipelineWithName:@"my_custom_op"]; // Metal shader 名
}

ErrorCode MetalMyCustomOp::onResize(const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    // 计算 thread group 大小
    int totalSize = outputs[0]->elementSize();
    mThreads = {(NSUInteger)totalSize, 1, 1};
    mThreadgroupSize = {256, 1, 1};
    return NO_ERROR;
}

void MetalMyCustomOp::onEncode(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                id<MTLComputeCommandEncoder> encoder) {
    auto input  = inputs[0];
    auto output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder dispatchThreads:mThreads threadsPerThreadgroup:mThreadgroupSize];
}

// 注册
class MetalMyCustomOpCreator : public MetalBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const MNN::Op* op, Backend* backend) const {
        return new MetalMyCustomOp(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalMyCustomOpCreator, OpType_MyCustomOp);

} // namespace MNN
```

### 5.2.2 编写 Metal Shader

在 `source/backend/metal/shader/` 下添加 `my_custom_op.metal` 或将 kernel 写入已有的 shader 文件。

### 5.2.3 更新工程

```bash
cd source/backend/metal
python3 MetalCodeGen.py .
```

---

## 5.3 OpenCL 后端

### 5.3.1 编写 Kernel

在 `source/backend/opencl/execution/cl/` 下创建 `my_custom_op.cl`：

```opencl
__kernel void my_custom_op(
    __read_only image2d_t input,
    __write_only image2d_t output,
    __private const int width,
    __private const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float4 in = read_imagef(input, SAMPLER, (int2)(x, y));
    float4 out = in; // ← 替换为实际计算
    write_imagef(output, (int2)(x, y), out);
}
```

### 5.3.2 生成 Kernel 映射

```bash
cd source/backend/opencl/execution/cl
python3 opencl_codegen.py
```

### 5.3.3 创建实现文件

在 `source/backend/opencl/execution/` 下创建 `MyCustomOp.h` 和 `MyCustomOp.cpp`，注册：

```cpp
OpenCLCreatorRegister<TypedCreator<MyCustomOp<cl_data_t>>> __my_custom_op(OpType_MyCustomOp);
```

---

## 5.4 Vulkan 后端

### 5.4.1 生成模板代码

```bash
cd source/backend/vulkan/image/compiler
python3 VulkanCodeGen.py
```

### 5.4.2 编写 Compute Shader

在 `source/backend/vulkan/image/execution/glsl/` 下创建 `myCustomOp.comp`。

### 5.4.3 编译 Shader

```bash
cd source/backend/vulkan/image/compiler
python3 makeshader.py
```

### 5.4.4 实现并注册

```cpp
class VulkanMyCustomOpCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new VulkanMyCustomOp(op, backend);
    }
};
static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_MyCustomOp, new VulkanMyCustomOpCreator);
    return true;
}();
```

---

## 5.5 CUDA 后端

在 `source/backend/cuda/execution/` 下添加 `.cu` 和 `.cuh` 文件，编写 CUDA kernel 并注册。

---

## 步骤 5 测试标准

### 测试方法

每完成一个后端，都用同一套单元测试验证：

```bash
cd build

# CPU（已通过）
./run_test.out op/MyCustomOp

# Metal（需要 Mac + Metal 支持）
./run_test.out op/MyCustomOp 0 0 3    # 3 = Metal

# OpenCL
./run_test.out op/MyCustomOp 0 0 6    # 6 = OpenCL

# Vulkan
./run_test.out op/MyCustomOp 0 0 7    # 7 = Vulkan

# CUDA
./run_test.out op/MyCustomOp 0 0 1    # 1 = CUDA
```

> **后端 ID 参考**：0=CPU, 1=CUDA, 3=Metal, 6=OpenCL, 7=Vulkan

### 通过标准

- [ ] 目标后端的单元测试全部 `passed`
- [ ] 计算结果与 CPU 版本一致（在浮点精度范围内）

### 部分完成也是可以接受的

```
✅ 已完成：
- Schema 定义
- 形状计算
- CPU 实现 + 单元测试通过
- Metal 实现 + 单元测试通过

⏳ 待完成：
- OpenCL 后端
- Vulkan 后端
- CUDA 后端
- SIMD 性能优化
```

---

## 完成

**恭喜！当所需的后端测试全部通过后，算子支持工作完成。**
