# 步骤 0：建立性能基准

> **目标**：获取优化前的基线数据，判断瓶颈在 CPU 调度还是 GPU kernel，再识别 GPU 内的瓶颈 kernel。
>
> **前置**：明确待优化的算子/模型和目标参数。
>
> **复杂度**：低（需编译运行真机）。
>
> **参考**：编译/推送/运行、正确性验证见 `SKILL.md`；瓶颈类型判断见 `optimization-handbook.md` §1。

---

## 0.1 确定优化目标

```
待优化对象：____（整模型 / LinearAttention / conv1x1 / MatMul …）
目标数据类型：____（FP16 / FP32；LLM 通常 fp16）
后端：____（Vulkan buffer，MNN_VULKAN_IMAGE=OFF）
典型参数：____（prefill L=512/1024/4096；decode L=1；model 0.6B/2B/4B）
目标平台：____（Adreno / Mali / Apple；serial）
当前 / 目标性能：____
```

---

## 0.2 定位目标 kernel / 路径

改前先确认 dispatch 路径（详见 `SKILL.md` 入口定位）：

```bash
grep -rn "OpType_<MyOp>" source/backend/vulkan/buffer/execution/
```

读 `onEncode`/`onResize`，把目标 shape 代入，确认落到哪条路径（CoopMat / subgroup / nosubgroup / 融合 vs 分离）和哪些 `.comp`。**不要把 fallback 路径当 baseline**。

---

## 0.3 Profile 全模型 / 算子

用 `-DMNN_GPU_TIME_PROFILE=ON` 编译（命令见 `SKILL.md` 编译与真机运行），跑端到端：

```bash
adb -s <serial> shell "cd /data/local/tmp/MNN && rm -f tmp/mnn_cachefile.bin && \
  LD_LIBRARY_PATH=. ./llm_demo <model>/config_vk.json 512.txt 2 2>&1" > prof.txt
```

Vulkan profiler 两块输出（都累计到退出，含 load 期 tuning）：
- `[Execution Profiling]`：op 级（Convolution/Attention/Raster/…）。
- `[Shader Profileing]`：shader 级（`glsl_..._comp`）。

**取稳态**：只解析 `Prepare for tuning opt End` 之后的块并按 op/shader 求和（见 handbook §1.4 profiler 读法）。

---

## 0.4 先判 CPU 调度 vs GPU kernel（Vulkan 必做）

对比「GPU kernel 累计」和「干净 build（`MNN_GPU_TIME_PROFILE=OFF`）的端到端 wall」：

| 观察 | 结论 | 去哪 |
|------|------|------|
| GPU 累计 ≈ wall | GPU-bound | 0.5 定位瓶颈 kernel → Kernel/算子级优化 |
| GPU 累计 ≪ wall（只占几分之一）| **CPU 调度 bound** | handbook 技巧 6/7 + §6（indirect batch / fixResizeCache / 算子融合）|

> 小模型（0.6B）常 CPU-bound；大 int4 模型（4B）conv GPU 占比大常 GPU-bound。别默认 GPU-bound（见 handbook 陷阱 F）。

---

## 0.5 记录基线 + 定位 GPU 瓶颈

```markdown
## 基线数据
**平台**: Adreno <型号> (serial <xxx>)  **日期**: <填写>
**编译**: MNN_VULKAN=ON, MNN_VULKAN_IMAGE=OFF, MNN_BUILD_LLM=ON, GPU_TIME_PROFILE=ON
**端到端（干净 build）**: prefill xx tok/s, decode xx tok/s

### GPU op/shader 耗时（稳态，512-token prefill）
| shader/op | 时间(ms) | 占比 |
|---|---|---|
| glsl_xxx | xx | xx% |
| **总计** | xx | 100% |
```

瓶颈识别清单：
```
□ 端到端 wall vs GPU 累计 → CPU 调度 or GPU？
□ 哪个 shader/op 占 GPU 最多？
□ 该 kernel 计算强度？memory-bound / compute-bound / occupancy 墙？（handbook §1）
□ 有无明显异常（某 shape 特别慢、N 是 2 的幂次时退化等）？
```

---

## 0.6 （可选）独立 kernel demo

需频繁改 kernel 又不想每次重编 libMNN 时，仿 OpenCL 的 demo 驱动：在 `tools/vulkan_bench/`（或用户指定目录）写独立 Vulkan 程序，初始化 device/queue → 备输入（模型 dump 或固定 seed）→ 编译目标 `.comp`（同 makeshader 管线）→ 设与 MNN 相同 spec constant/push constant/dispatch → 跑并存 baseline 输出+性能。之后按 `kernel-opt.md` 在 demo 内迭代（与 baseline 逐元素比），赢了再换回 MNN。

---

## 通过标准

- [ ] 已判断瓶颈在 CPU 调度还是 GPU kernel。
- [ ] 有稳态 op/shader 耗时排序 + 端到端 tok/s 基线。
- [ ] 已确定第一个优化目标。

### 常见问题

| 问题 | 原因 | 修复 |
|------|------|------|
| 改了 `.comp` 不生效 | makeshader 没跑 / 改错 buffer↔image 后端 | 见 SKILL.md，`grep MNN_VULKAN_IMAGE CMakeCache.txt` |
| 跑起来 segfault | pipeline cache stale | `rm tmp/mnn_cachefile.bin` |
| GPU 累计远小于 wall | prefill 是 CPU 调度 bound | 别优化 kernel，攻调度（handbook 技巧 7 / §6）|
| 性能波动大 | 冷启动 / 热漂移 | warm 后测 + 交替 A/B（SKILL.md 性能测量）|
