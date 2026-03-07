# MNN MUSA 后端编译问题修复计划

## 问题分析

MUSA后端代码基于旧版MNN API编写，与MNN 3.0+不兼容。

### 主要API变更

| 旧API | 新API |
|-------|-------|
| `Storage_Internal` | `STATIC` |
| `Storage_External` | `DYNAMIC` / `DYNAMIC_SEPERATE` |
| `MemObj.storage` | 移除，使用MemChunk |
| `MemObj.size` | 移除 |
| `MemObj.base` | 移除，使用MemChunk.ptr() |
| `BufferAllocator::clear()` | `release(true)` |
| `BufferAllocator::onResizeBegin/End` | 移除 |
| `TensorUtils::getDescribe()->memory` | 移除，使用buffer().device |
| `TensorUtils::getDescribe()->elements` | 直接计算 |
| `TensorUtils::getDescribe()->type.bytes()` | `tensor->getType().bytes()` |
| `DataType_FLOAT32` | `DataType_DT_FLOAT` |

## 修复步骤

### 1. 更新 MusaBackend.cpp
参考 CUDA 后端 `CUDABackend.cpp` 更新API调用

### 2. 更新 MusaRuntime.cpp
确保与最新内存管理API兼容

### 3. 更新 execution 文件
确保算子实现与新API兼容

## 兼容层已完成

✅ `3rd_party/musa_compat/include/musa_runtime.h` - MUSA API兼容头文件
✅ `source/backend/musa/CMakeLists.txt` - 更新构建配置

## 下一步

需要逐文件更新MUSA后端代码以匹配MNN 3.0+ API