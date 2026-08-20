---
name: metal-perf-methods
description: MNN Metal 后端 LLM 优化手段方法论库（feature/metal-speed 战役沉淀）。按 kernel 类 / 算子级别 / 调度类三层组织，只记录优化原理、适用条件与陷阱，不含具体性能数字。做 Metal 性能优化选题、评审优化方案、复盘优化方向时阅读。
---

# Metal 优化手段方法论（索引）

> **触发**：为 Metal LLM 推理选择/设计优化方案；评审某个优化想法是否成立；
> 优化不见效时对照排查方向。
>
> **边界**：只讲原理与适用条件，**不含任何具体性能数字**（数字属于性能报告，
> 会随设备/模型过时；方法论不会）。MNN Metal 的开发规范、陷阱与构建测试
> 流程见姊妹 skill `metal-optimize`。

## 三层分类

| 层次 | 文件 | 覆盖的手段 |
|---|---|---|
| Kernel 类 | [`kernel-methods.md`](./kernel-methods.md) | GEMV 融合 epilogue、LN 前序拆分、split-K 及变体、向量宽 load 与访存合并、寄存器驻留单遍流式（sdpa_vector 形态）、量化解包向量化、编译期常量、NSG 校准、递推状态驻留与并行扫描 |
| 算子级别 | [`op-fusion-methods.md`](./op-fusion-methods.md) | gate/up+SwiGLU 折叠、QKV 打包 grid、导出期声明融合链路、融合正确性门槛 |
| 调度类 | [`scheduling-methods.md`](./scheduling-methods.md) | 采样去 expr 化、GPU→CPU 同步点治理、Encode Replay 资源生命周期、多路径自动阈值与降级链、实验开关收敛纪律 |

## 方法学总纲（所有层次通用）

1. **正确性先于性能**：bit/greedy 对拍通过之前测出的任何收益都不可信——
   错误 kernel 一样能"变快"。
2. **A/B 必须交替配对**：热漂移可制造 10% 量级的虚假差异，顺序单向对比无效。
3. **先定位瓶颈段**：GPU busy vs wall、kernel 计时 vs e2e、带宽兑现率是三种
   不同口径；优化必须对准真正的那一段，否则 kernel 变快 e2e 不动。
4. **布局是全局不变量**：数据布局改动牵动所有读写方，必须原子落地，
   中间状态不可运行。
5. **设备分档显式化**：Apple GPU 代际间结论不互推，每个结论标注适用设备档。
6. **证伪也是成果**：中性/负收益实验连方法一起归档，防止重复投入。

## 选题决策路径

```
瓶颈在哪？
├─ 每 token 固定开销大（dispatch 多 / CPU 段长） → 算子融合 / 调度类
├─ 权重带宽未吃满（decode GEMV）               → kernel 类 §1.3/§1.4
├─ attention 随 KV 变慢                        → kernel 类 §1.4/§1.5/§1.8
├─ GPU→CPU 同步密集                            → 调度类 §2/§1
└─ 不确定                                      → 先 profile，回到方法学总纲 3
```
