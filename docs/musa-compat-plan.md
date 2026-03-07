# MNN MUSA 编译兼容层方案

## 问题分析

当前MUSA后端编译问题：
1. `find_package(MUSA)`找不到MUSA SDK时会直接`return()`
2. 代码中`#include <musa_runtime.h>`无法找到头文件
3. 编译会失败

## 解决方案：MUSA兼容层

### 方案设计

创建`musa_compat`目录，提供MUSA API的兼容定义：

```
3rd_party/musa_compat/
├── CMakeLists.txt
├── include/
│   └── musa_runtime.h      # MUSA API兼容头文件
└── stub/
    └── musa_stub.c         # Stub实现（可选）
```

### 核心思路

1. **兼容头文件**：定义MUSA类型和函数声明（映射到CUDA或stub）
2. **条件编译**：
   - 有MUSA SDK → 使用原生MUSA
   - 无MUSA SDK，有CUDA → 映射到CUDA
   - 都没有 → 编译通过但运行时报错（或空实现）
3. **最小侵入**：不修改MNN主代码，只添加兼容层

### 实现步骤

1. 创建`3rd_party/musa_compat/`目录
2. 编写`musa_runtime.h`兼容头文件
3. 修改MUSA后端CMakeLists.txt查找兼容层
4. 测试编译