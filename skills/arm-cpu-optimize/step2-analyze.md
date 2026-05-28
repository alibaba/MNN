# 步骤 2：瓶颈分析与方案选择

> **目标**：根据基线判断该优化该改哪里，避免盲目写汇编或盲目加 unroll。

---

## 2.1 先分类瓶颈

| 类型 | 常见信号 | 优先方向 |
|------|----------|----------|
| memory bound | eff GB/s 接近 memcpy roofline | 减少字节、改善连续访问、减少 metadata |
| unpack/issue bound | eff GB/s 低但指令很多 | 减少 bit unpack、常量复用、调整 layout |
| compute bound | GFLOPS 接近指令峰值 | 更大 tile、更多 accumulator、ISA 指令 |
| postprocess bound | dot 快但 scale/minmax/store 慢 | 合并后处理、缩短 min/max live range |
| dispatch/pack 错配 | 质量错或某 ISA 独坏 | 检查 packer、unit、kernel pointer |

w2/w3 常见是 unpack/issue bound，不是单纯 bandwidth bound。

---

## 2.2 低 bit kernel 专项分析

分析以下问题：

- packed cell 的真实字节数是多少？是否有 padding？
- block32、block64、per-channel 的 metadata stride 是否不同？
- OC split 时 `weightPtr` 是否按真实 cell stride 前进？
- i8mm 和 sdot 是否共享 packer？共享是否安全？
- `sdot`/`smmla` 的 lane 分组和 pack layout 是否完全匹配？
- fp32 min/max、scale、bias、zp 是否被 unpack 临时寄存器覆盖？
- E=1 decode 是否走了和 E>1 不同的 kernel？

如果质量问题只出现在一个 ISA，优先查 dispatch/pack/register lifetime。
如果 no-thinking/greedy 正常但 mixed sampling 复读，先不要判定 kernel 错。

---

## 2.3 方案优先级

按风险从低到高选择：

1. 修正确认的 correctness bug：stride、tail、register clobber、dispatch 错配。
2. 复用已有 CoreFunctions 或已有 kernel 分支。
3. C++ 层减少拷贝、修正线程拆分、避免 onExecute 分配。
4. block64/default case 专用分支。
5. 常量 hoist、`ld1r` replicate load、减少重复 unpack。
6. 调整 pack layout，但不增加 packed bytes。
7. 扩大 unroll 或新增 asm tile。
8. 增加 packed bytes 或改变量化格式，只有用户明确接受时才做。

---

## 2.4 写方案时必须包含

```markdown
## 方案

目标路径：
当前瓶颈：
不优化的路径：

Correctness guard:
- C++ oracle:
- op tests:
- model sanity:

实现计划:
1.
2.
3.

性能目标:
- metric:
- expected direction:

风险:
- register lifetime:
- pack/dispatch:
- tail/block:
```

性能目标可以是方向性的，例如减少 unpack 指令或提升 eff GB/s，不要编造未实测数字。

---

## 通过标准

- 已用数据解释瓶颈，而不是凭直觉。
- 已明确哪些路径不在本次范围。
- 已有 correctness guard，能在每个小改动后快速验证。
- 方案没有依赖用户未接受的格式变化，例如扩大 w2/w3 字节。
