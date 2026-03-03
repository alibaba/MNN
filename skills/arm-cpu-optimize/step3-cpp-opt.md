# 步骤 3：C++ 优化

> **目标**：用 MNN 已有函数替换循环 + 多线程 + 内存池 + 数据排布优化。
>
> **前置条件**：步骤 0 的映射表 + 步骤 2 的方案已制定。
>
> **复杂度**：高（需要编译运行验证）
>
> **执行顺序**：3.0(函数替换) → 3.1(多线程) → 3.2(数据排布) → 3.3(Cache/内存) → 3.4(验证)

---

## 3.0 用 MNN 函数替换循环（最高优先级）

**根据步骤 0 的映射表，逐一将循环替换为 MNN 已有函数。这是提升最大的一步。**

### 常见替换模式

```cpp
// ========== 1. 标量乘/加 → MNNScaleAndAddBiasScalar ==========
// 替换前：
for (int i = 0; i < size; ++i) data[i] *= scale;
// 替换后：
MNNScaleAndAddBiasScalar(data, data, 0.0f, scale, size);

// ========== 2. 循环 expf → MNNExp ==========
// 替换前：
for (int i = 0; i < size; ++i) dst[i] = expf(src[i]);
// 替换后：
float offset[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // exp(src * 1.0 + 0.0) + 0.0
MNNExp(dst, src, offset, size);

// ========== 3. SiLU → MNNSiLu ==========
// 替换前：
for (int i = 0; i < size; ++i) {
    float s = 1.0f / (1.0f + expf(-src[i]));
    dst[i] = src[i] * s;
}
// 替换后（注意：不支持 in-place，需要 dst != src）
MNNSiLu(dst, src, size);

// ========== 4. MatVec y = A @ x → MNNComputeMatMulForE_1 ==========
// 替换前：
for (int j = 0; j < N; ++j) {
    float sum = 0;
    for (int i = 0; i < K; ++i)
        sum += A[i * N + j] * x[i];
    y[j] = sum;
}
// 替换后：
MatMulParam param;
param.e = 1; param.l = K; param.h = N;
param.numberThread = 1;
param.ATranspose = false; param.BTranspose = false;
gcore->MNNComputeMatMulForE_1(x, A, y, nullptr, &param, 0);

// ========== 5. Depthwise Conv1D → MNNConvRunForLineDepthwise ==========
// 替换前：
for (int l = 0; l < L; ++l) {
    float sum = 0;
    for (int k = 0; k < K; ++k)
        sum += input[l + k] * weight[k];
    output[l] = sum;
}
// 替换后：使用 gcore->MNNConvRunForLineDepthwise

// ========== 6. std::vector → Tensor + 内存池 ==========
// 替换前：
std::vector<float> temp(size);  // 每次 onExecute 都 malloc
// 替换后（在 onResize 中）：
mTemp.reset(Tensor::createDevice<float>({size}));
backend()->onAcquireBuffer(mTemp.get(), Backend::DYNAMIC);
backend()->onReleaseBuffer(mTemp.get(), Backend::DYNAMIC);
// 在 onExecute 中直接使用：
float* temp = mTemp->host<float>();
```

### 检查清单

完成替换后，用以下清单逐一确认：

**函数替换检查：**
- [ ] 代码中不再有 `expf()` 的循环调用 → 已替换为 `MNNExp`
- [ ] 代码中不再有循环乘标量 → 已替换为 `MNNScaleAndAddBiasScalar`
- [ ] 代码中不再有 `x * sigmoid(x)` → 已替换为 `MNNSiLu`
- [ ] 代码中不再有双重循环的矩阵向量乘 → 已替换为 `MNNComputeMatMulForE_1`
- [ ] 代码中不再有 `std::vector` 临时缓存 → 已替换为 `Tensor + Backend`
- [ ] 代码中没有用 Vec4 替代已有 MNN 函数的情况（无已有函数时允许 Vec4）

**替换质量检查（对每处替换逐一确认）：**
- [ ] 函数的数学语义与原始循环完全一致（不是"大概相似"）
- [ ] 如果 dst==src（in-place），确认该函数支持 in-place
- [ ] 如果数据规模很小，确认函数调用开销不大于朴素循环
- [ ] 保持手动循环的地方，已注释说明不替换的原因

---

## 3.1 多线程优化

### MNN 多线程 API

```cpp
// 获取线程数
int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();

// 多线程并行（按 tId 划分工作）
MNN_CONCURRENCY_BEGIN(tId, threadNum) {
    int start = tId * totalWork / threadNum;
    int end = (tId + 1) * totalWork / threadNum;
    for (int i = start; i < end; ++i) {
        // 执行计算
    }
} MNN_CONCURRENCY_END();
```

### 大小核调度

MNN 自动管理线程池，但需要注意：

```cpp
// 获取 CPU 核心信息
auto runtime = static_cast<CPUBackend*>(backend())->getRuntime();
int bigCoreNum  = runtime->hint().cpuDecreaseRate;  // 大核数量
int threadNum   = static_cast<CPUBackend*>(backend())->threadNumber();

// 如果支持 FP16，大核用 FP16 通路，小核用 FP32
auto gcore = static_cast<CPUBackend*>(backend())->functions();
if (gcore->supportFp16arith) {
    // 使用 FP16 函数指针
}
```

### 多线程划分策略

| 场景 | 划分维度 | 原因 |
|------|---------|------|
| 矩阵乘 C = A × B | 按 B 的列（N/hP）划分 | 避免写冲突 |
| Attention Q×K^T | 按 Head 划分 | Head 间独立 |
| Softmax | 按行（batch）划分 | 行间独立 |
| LayerNorm | 按行划分 | 行间独立 |

---

## 3.2 数据排布优化

### Pack 数据用于 SIMD

```cpp
// 将 NCHW 数据 Pack 为 NC4HW4
auto gcore = static_cast<CPUBackend*>(backend())->functions();
int pack = gcore->pack;  // FP32=4, FP16=8

// Pack: [C, H, W] → [C/pack, H, W, pack]
gcore->MNNPackCUnit(dst, src, area, depth, areaOffset);

// Unpack: [C/pack, H, W, pack] → [C, H, W]
gcore->MNNUnpackCUnit(dst, src, area, depth, areaOffset);
```

### MatMul 数据重排

```cpp
// A 矩阵 Pack: [M, K] → [M/eP, K/lP, eP, lP]
int eP, lP, hP;
gcore->MNNGetMatMulPackMode(&eP, &lP, &hP);
gcore->MNNPackC4ForMatMul_A(packedA, srcPtrs, info, el);

// B 矩阵 Pack: [K, N] → [N/hP, K/lP, lP, hP]
gcore->MNNPackForMatMul_B(packedB, src, h, kernelSize, ic, transpose);
```

---

## 3.3 Cache 优化

### Tiling 策略

```cpp
// 参考 DenseConvolutionTiledExecutor 的 Tiling
// 将大矩阵分块，让每个 Tile 适配 L1 Cache
int L1_CACHE = 32 * 1024;  // 32KB L1（典型值）
int L2_CACHE = 256 * 1024; // 256KB L2

// A 的 Tile 大小适配 L1
int tileE = eP;
int tileL = L1_CACHE / (eP * sizeof(float) * 2);  // A 和 B 各占一半
tileL = (tileL / lP) * lP;  // 对齐到 lP

// 计算循环：外层遍历 K 分块，内层遍历 M 和 N
for (int kb = 0; kb < K; kb += tileL) {
    // Pack A[:, kb:kb+tileL]
    // 对每个 N 分块计算 MatMul
}
```

### 内存分配

```cpp
// onResize 中申请临时缓存（会被内存池复用）
ErrorCode onResize(...) {
    mTempBuffer.reset(Tensor::createDevice<float>({threadNum, bufferSize}));
    backend()->onAcquireBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}
```

---

## 3.4 正确性验证

每次 C++ 优化后，立即运行测试：

```bash
# 功能测试（正确性）
./run_test.out op/Xxx

# 性能测试
./run_test.out speed/XxxSpeed
```

### 对比结果

```markdown
## 优化结果对比

| 用例 | 基线(ms) | 优化后(ms) | 提升 |
|------|---------|-----------|------|
| decode_1token | xx.xx | xx.xx | x.xx |
| prefill_128 | xx.xx | xx.xx | x.xx |
```

---

## 3.5 代码质量检查

优化完成后提交前，必须检查代码质量：

```
□ 编译无 warning（特别关注 -Wunused-variable、-Wunused-parameter）
□ 没有声明但未使用的变量（优化过程中容易遗留）
□ .hpp 头文件的注释与 .cpp 中的实际实现一致
  （如 buffer 分配大小的注释与代码计算是否匹配）
□ 所有保持手动循环的地方都有注释说明原因
□ 所有使用 MNN 函数 in-place 的地方都已确认安全
□ onResize 中分配的 buffer 都有 onReleaseBuffer 配对
□ 没有残留的调试代码（printf、临时变量等）
```

---

## 步骤 3 测试标准

### 通过标准

- [ ] **功能测试通过**：`./run_test.out op/Xxx` 全部 passed
- [ ] **性能有提升**：至少一个用例有明显提升
- [ ] **无内存泄漏**：多次运行内存稳定
- [ ] **代码质量检查通过**：3.5 清单全部勾选
- [ ] **有实测性能数据**：优化前后的对比数据已记录（不能只写"预期"）

### 常见问题

| 问题 | 原因 | 修复 |
|------|------|------|
| 多线程结果错误 | 写冲突 / 未正确划分 | 检查是否有共享写入 |
| 性能反而下降 | 线程开销大于计算量 | 小 Tensor 用单线程 |
| 偶尔结果不一致 | 浮点累加顺序不同 | 放宽 tolerance |
| 编译 warning | 优化过程遗留的未使用变量 | 清理所有 unused 变量和 include |
| 头文件注释不一致 | 优化中改了逻辑但没更新注释 | 每次改逻辑后同步更新注释 |

---

## 下一步

**步骤 3 通过后，进入 `step4-asm.md`（步骤 4：ARM 汇编优化）。**
