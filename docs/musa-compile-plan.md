# MNN MUSA 编译方案

## 问题
不改MNN主代码，如何让MUSA后端代码编译通过？

## 方案思路

### 方案1: 头文件兼容层
创建 `musa_compat.h` 头文件，将MUSA API映射到CUDA或空实现：

```c
#ifndef MUSA_COMPAT_H
#define MUSA_COMPAT_H

#ifdef MNN_MUSA
// 如果有MUSA SDK
#include <musa_runtime.h>
#else
// 没有MUSA SDK时，提供兼容定义
#define musaMalloc cudaMalloc
#define musaFree cudaFree
#define musaMemcpy cudaMemcpy
// ... 或提供空实现
#endif

#endif
```

### 方案2: 条件编译
在现有CUDA代码中添加MUSA条件编译分支

### 方案3: 独立后端 + 桥接层
MUSA后端完全独立，通过桥接头文件连接MNN核心

## 待验证
- [ ] 检查MNN现有后端架构
- [ ] 分析CUDA后端如何处理无CUDA环境
- [ ] 设计最小侵入方案