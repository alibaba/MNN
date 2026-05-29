# 步骤 3：C++ 层优化与 oracle

> **目标**：在进入 asm 前，把语义、layout、内存和调度问题先在 C++ 层解决。

---

## 3.1 CoreFunctions 复用

优先替换能安全复用的循环：

| 代码模式 | 可考虑函数 |
|----------|------------|
| 大规模 GEMM | `MNNPackedMatMul` |
| E=1 GEMV | `MNNComputeMatMulForE_1` |
| scale+bias | `MNNScaleAndAddBias` / `MNNScaleAndAddBiasScalar` |
| softmax/norm/activation | `MNNSoftmax` / `MNNNorm` / `MNNExp` / `MNNSiLu` |
| pack/unpack | 对应 `gcore` pack helpers |

每处替换都要检查：

- 数学语义、layout、转置和 tail 是否一致。
- `dst == src` 是否安全；不确定就读实现或加小测试。
- 小 shape 下函数调用和 pack 开销是否值得。
- 保留手写循环时，原因是否清楚。

---

## 3.2 C++ 标量 oracle

复杂 kernel 必须先写标量 oracle。oracle 可以是临时 debug 代码，也可以是测试中的 reference。

低 bit oracle 要显式表达：

- bit unpack。
- zero point / scale / bias。
- block metadata。
- fp32 min/max。
- add-dst。
- tail 和 OC split。

建议分层比较：

| 输出 | 目的 |
|------|------|
| unpack int weight | 查 bit layout |
| int32 accumulator | 查 dot/smmla 分组 |
| dequant FP32 | 查 scale/zp |
| final dst | 查 postprocess |

---

## 3.3 C++ SIMD/寄存器模拟

当目标是 w2/w3、sdot、smmla、SME2 或复杂 pack/tail 时，写 C++ 模拟版再迁移 asm。

要求：

- 用局部数组或 `std::array` 表达 lane，不依赖编译器自动向量化。
- helper 名称对应目标指令，例如 `simulateSdot4`、`simulateSmmla`、`unpackW3Block64`。
- 变量名尽量对应未来 asm 寄存器角色，例如 `vAcc0`、`vInput0`、`vMin`。
- 每次改变 unroll、pack layout 或 block64 分支，先让模拟版和标量 oracle 对齐。

简单 elementwise NEON 不强制模拟；小 reference + asm register plan 通常足够。

---

## 3.4 内存和线程

检查项：

- onExecute 中不要反复 `new`、`malloc` 或大 `std::vector` 临时缓存。
- 需要 scratch 时，在 onResize 中用 Backend 内存池申请。
- 多线程按独立输出区域划分，避免写冲突和 false sharing。
- OC split 的 per-thread pointer 要按真实 packed cell stride 计算。
- 线程数、tile 数、block 数不要在内层重复计算。

不要过早做大小核特化；先让 runtime 现有线程策略和目标 kernel 路径正确。

---

## 3.5 C++ 阶段验证

每个小改动后优先跑 focused test：

```bash
cd build
cmake --build . -j 8
./run_test.out op/lowMemory/blockConv 0 1 4
./run_test.out op/lowMemory/HybridConv 0 1 4
./run_test.out op/lowMemory/blockConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑
./run_test.out op/lowMemory/HybridConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑
```

如果不是低 bit LLM kernel，替换为对应算子的最小 op test。

---

## 通过标准

- 可复用的 CoreFunctions 已复用，或已说明不复用原因。
- C++ 标量 oracle 可运行，复杂 SIMD 语义已有 C++ 模拟。
- 没有引入 onExecute 大临时分配。
- 线程拆分和 pointer offset 已覆盖非 0 线程。
- focused correctness test 通过后，再进入 asm。
