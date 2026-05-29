# 步骤 1：建立基线与验证矩阵

> **目标**：在优化前拿到可复现的正确性和性能基线，避免靠模型输出或单次耗时猜测。

---

## 1.1 优先使用已有测试

不要默认新建 speed test。先查当前仓库已有测试是否覆盖目标路径：

```bash
rg -n "LinearRoofline|MatMul|lowMemory|HybridConv|blockConv" test source/backend/cpu
```

常用低 bit LLM kernel 基线：

```bash
cd build
./run_test.out op/lowMemory/blockConv 0 1 4
./run_test.out op/lowMemory/HybridConv 0 1 4
./run_test.out op/lowMemory/blockConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑
./run_test.out op/lowMemory/HybridConv 0 1 4  # 在 SDOT 目标设备/构建配置上复跑

./run_test.out speed/GemvBW 0 2
```

如果已有测试不能覆盖目标 shape，再新增 focused test。

---

## 1.2 正确性基线

普通算子至少记录：

- 输入 shape、precision、线程数。
- 参考实现或已有后端输出。
- 误差阈值和失败样例。

低 bit LLM kernel 额外记录：

| 维度 | 必测组合 |
|------|----------|
| bit | w2、w3，如任务只涉及一个 bit 则只测该 bit |
| ISA | I8MM、SDOT、SME2 等目标路径 |
| block | block32、block64、per-channel 中任务涉及的组合 |
| 后处理 | fp32 min/max、bias/scale/zp、add-dst 如存在 |
| 模型 | 一个短 prompt 的 no-thinking/greedy sanity |

模型输出异常时，先固定采样变量，再判断 kernel 是否错误。

---

## 1.3 性能基线

记录可解释的指标，不只记录耗时：

| 指标 | 说明 |
|------|------|
| `us/iter` | 端到端 kernel 或 test 平均耗时 |
| bytes/elem | 包括权重和 metadata 的真实字节 |
| eff GB/s | 用真实读取字节估算 |
| `%peak` | 对比 streaming bandwidth |
| GFLOPS / AI | 判断 compute/issue/memory 倾向 |
| threads | 线程数必须固定 |

w2/w3 的 eff GB/s 低时，不要直接判定内存带宽不足。先看 unpack 指令数、寄存器压力、metadata load 和 postprocess。

---

## 1.4 需要新增 speed test 时

只有已有测试不能回答问题时才新增。新增测试要小而准：

- 覆盖目标 shape，而不是泛化一堆无关 case。
- 能选择目标 ISA 路径，或在对应能力设备上分别验证。
- 输出真实 bytes/elem、eff GB/s、GFLOPS。
- 对 cold-cache / warm-cache 做出明确选择。
- 首次运行要有 correctness check。

低 bit GEMV 可优先写 roofline 风格测试，而不是完整模型 benchmark。

---

## 1.5 输出基线

```markdown
## Baseline

commit:
platform:
build options:
threads:
ISA path:
shape/block:

| test | bit | ISA | us/iter | bytes/elem | eff GB/s | GFLOPS | note |
|------|-----|-----|---------|------------|----------|--------|------|

Correctness:
- op test:
- model sanity:
```

---

## 通过标准

- 已有正确性基线，且覆盖目标 ISA/path。
- 已有性能基线，且线程数、shape、precision 可复现。
- 如果无法在当前设备实测，已给出准确命令并标注待实测。
- 没有把 mixed sampling 下的模型复读直接当成 kernel 错误。
