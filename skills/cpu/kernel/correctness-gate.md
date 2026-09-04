# 跨 ISA 正确性闸门

> **何时读**：kernel 写完、单测过了、准备说「done」之前。以及 review 别人的 kernel 改动时。
>
> **本文不给命令**，命令与报告格式在 [`cpu/shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md)。
> 本文只回答两件事：**必须覆盖哪些维度**，以及**什么才算过**。

CPU kernel 的失败模式有个共同特征：**默认参数下跑一遍全绿，换个参数就错，而且不崩。**

原因是 CPU 后端有大量「参数决定走哪条代码路径」的分支。你测的那一组参数，可能根本没进你改的那段代码。所以正确性闸门的本质不是「多跑几次」，而是**逐个点名那些会切换代码路径的轴**。

---

## 一、会切换代码路径的六条轴（不是六种输入）

这六条每一条都在源码里对应一个真实的 `if`。少覆盖一条，就是有一整段代码没被执行过。

> **这六条轴全树唯一一份**。[`cpu/optimize/bugfix.md`](../optimize/bugfix.md) §2.6 讲的是
> 「为什么这些轴互不可推断、前五类 bug 因此逃过门禁」，它引用本节，不重列必测取值。

### 1.1 线程数 —— 切换 OC 拆分 vs plane 拆分

```cpp
// ConvInt8TiledExecutor.cpp
mSplitByOc = true;
if ((preferLinearPlaneSplit || threads < planeSize || mOnlineReorderWeightSme) && !mMixedKernel) {
    ...
    if (preferLinearPlaneSplit || mTileCount > threads || (mOnlineReorderWeightSme && planeSize > 1)) {
        mSplitByOc = false;
    }
}
```

`threads` 直接参与判断。**单线程跑过 ≠ 多线程正确**，而且不是「多线程可能有竞争」这种笼统风险，是**两条不同的指针计算代码**：`mSplitByOc == true` 走 OC 分片（每线程负责一段输出通道，`weightPtr` 要按真实 packed cell stride 偏移），`false` 走 plane 分片。

必测：`1` 和 `4`（或目标机核数）各一遍。低 bit kernel 还要专门确认 `tId > 0` 的那一片结果正确——`tId == 0` 的偏移量是 0，能掩盖所有 stride 算错。

### 1.2 planeSize / batch —— 切换量化粒度

```cpp
// ConvInt8TiledExecutor.cpp
if (inputPlane > 1)  { mUseBatchQuan = true; }
if (!fastway) {
    mIm2ColBasedInt8 = false;
    if (planeSize > 1) { mUseBatchQuan = true; }
    if (inputBlockQuantOption == 1) { mIm2ColBasedInt8 = true; mUseBatchQuan = false; }
}
```

`mUseBatchQuan` 后续控制 scale 指针步进方式（`ConvInt8TiledExecutor.cpp`：`mUseBatchQuan ? inputScale + step * blockNum * QUANT_INFO_BYTES : inputScale`）和 `sumParams.oneScale`。

翻译成 LLM 语言：**decode（`planeSize == 1`）和 prefill（`planeSize > 1`）是两条量化路径。** 这是「op test 全过、模型输出乱码」的最高频来源——op test 默认形状往往落在 prefill 侧。

必测：`E == 1` 与 `E > 1` 各一遍。

### 1.3 精度 —— fp16 与 fp32 是两套独立 kernel

`MatmulRelatedFunctions` 里 `Int8GemmKernel*`（fp32 出口）和 `MNNGemmInt8AddBiasScale_*_Unit_FP16`（fp16 出口）是**互不相关的字段**（`CommonOptFunction.h`）。fp16 还整体走另一张二级表（`Arm82Functions`）。

**fp16 正确推不出 fp32 正确，反之亦然。** 后处理的 min/max 裁剪、累加中间精度都不同。

必测：`precision = 0`（Normal/fp32）与 `2`（Low/fp16）各一遍。注意 x86_64 上 `Precision_Low` 不是 fp16（`bytes` 仍为 4，被 `AVX2Backend` 写死），所以这一轴在 x86_64 上等价于只有一档——详见 [`cpu/SKILL.md`](../SKILL.md)「三侧不同构对照表」与 [`arch/x86_64.md`](arch/x86_64.md) §四。

### 1.4 blockNum / block 粒度 —— 切换 metadata 步进

`blockNum` 参与 weight shape 的每一维（`ConvInt8TiledExecutor.cpp`），并决定 dequant scale/bias 的布局（同文件注释：`dequant bias: [blocknum, ocUp4]`）。还有一条快慢路判断直接依赖它：

```cpp
// ConvInt8TiledExecutor.cpp
bool fast = (kernelCount == 1 && ROUND_UP(oc, UNIT) == oc && (ic % (blockNum * SRC_UNIT)) == 0);
```

必测：per-channel（`blockNum == 1`）、block32、block64 各一遍。**`ic % (blockNum * SRC_UNIT) != 0` 的形状要专门造一个**，它走的是慢路，且是唯一能暴露 padding 处理错误的形状。

### 1.5 tail —— 非对齐形状

`oc` 不是 `UNIT` 整数倍、`ic` 不是 `SRC_UNIT` 整数倍、`E` 不是 `DST_XUNIT` 整数倍，三者独立。

对齐形状是 kernel 的主循环；tail 是另一段代码（常常是 `..._Remain` 系列或主循环内的边界分支）。**只测对齐形状等于没测 tail。**

必测：三个维度各造一个非对齐形状。最省事的做法是取一个对齐形状然后 `oc-1` / `ic-1` / `E-1`。

### 1.6 ISA 档位 —— 目标档 + 每一个回退档

你只改了 SME2 档，仍然要跑 sdot 档和基线档。理由不是「怕改错」，是 §二 要求的**逐位一致**只能靠实跑证明。

用 `MNN_CPU_TARGET` 逐档降级（需 `-DMNN_PIPELINE_PROFILE=ON` 构建），档位含义见 [`cpu/shared/env-registry.md`](../shared/env-registry.md)。

---

## 二、两条判定标准

### 2.1 目标档：与 C++ oracle 对齐

不是「和改动前一致」——改动前可能就是慢实现。

**oracle 多数时候不用自己写，仓库自带**：`compute/Int8FunctionsOpt.cpp` 在 `#ifndef MNN_USE_NEON`
下有一份纯标量的 `MNNGemmInt8AddBiasScale_16x4_Unit`（及 `_FAST`），它就是 int8 gemm 的参考语义。
要自己写时两条硬要求：**按真实 pack layout 与真实 cell stride 读权重**（不要用理想化公式，
见 [`pack-and-abi.md`](pack-and-abi.md) §3.1），以及**覆盖 `tId > 0` 的 OC chunk**（分线程的指针偏移是高频错点）。

对齐必须**分层**比，不要只比最终 logits——越靠近 kernel 输出，定位越快：

| 比较点 | 对上了能确认什么 |
|---|---|
| unpack 后的整数权重 | bit layout / 低 bit 解包对不对 |
| int32 accumulator | dot / 矩阵指令的**数学分组**对不对（错位症状见 [`arch/arm.md`](arch/arm.md) §4.2） |
| dequant 后 fp32 | scale / zero point 对不对 |
| postprocess 后 dst | bias / min-max / add-dst 对不对 |

oracle 可以是临时 debug 代码，不必进最终提交，但**必须在写 SIMD 之前跑通**——否则「数学错」与
「寄存器 / ABI 错」会混在一起，只能靠猜。

### 2.2 非目标档：与改动前**逐位一致**

这一条经常被跳过，后果最严重。

「我只是加了一条 SME2 路径，不会影响别人」是错的。三层表结构里，你在 init 里动的任何一行都可能被别的档位看到（快照时序、基表 vs 二级表、嵌套子表——见 [`dispatch-and-register.md`](dispatch-and-register.md) §四）。

**逐位一致，不是「误差在容差内」。** 非目标档的计算图完全没变，任何位差都说明你改到了不该改的东西。

---

## 三、静默失败清单：症状 → 优先怀疑

这些症状的共同点是**不崩、不报错**。

本表是**交付前**这一面：症状 → 哪条轴没覆盖 → 怎么把它跑出来。
如果改动已经合进去、要从「哪一档错」倒推到哪一层，去
[`cpu/optimize/bugfix.md`](../optimize/bugfix.md) §一（那边给机制、真实提交与预检）。

| 症状 | 优先怀疑 | 定位手段 |
|------|----------|----------|
| op test 全过，LLM 输出乱码/复读 | §1.2 decode 路径（`E == 1`、`mUseBatchQuan == false`）未覆盖 | 固定采样（greedy / no-thinking）后复现；再单独构造 `E == 1` 的 op test |
| 单线程对，4 线程错 | §1.1 OC 拆分的 `weightPtr` stride 算错（用了 payload 比例而非真实 packed 字节） | 只跑 `tId > 0` 的那一片，与 oracle 比 |
| fp16 对，fp32 错（或反之） | §1.3 两套 kernel，postprocess 未分别验证 | 两档分别对齐 oracle，不要交叉推断 |
| 结果对但 benchmark 没提升 | kernel 根本没被调用 | kernel 入口打日志；或对比 tile 值。见 [`dispatch-and-register.md`](dispatch-and-register.md) §四、§五 步 6 |
| 精度整体略差，但没有明显错值 | tile 换了、packer 没换（或反过来） | 对照 [`pack-and-abi.md`](pack-and-abi.md) §一 的五个同源量 |
| 换编译器 / 换 `-O` 级别表现不同 | 读到了未初始化字段（`new CoreFunctions` 默认初始化 + 二级表漏赋值） | 见 [`dispatch-and-register.md`](dispatch-and-register.md) §3.2；检查新字段是否在声明处给了 `= nullptr` |
| 目标机好，另一台机崩 | 回退档未验证；或能力位判断写反 | §1.6 逐档跑；打印能力位 |
| block32 对，block64 错 | §1.4 metadata 按 block 粒度步进错 | 单 block 单 OC group 最小 case 对齐 |
| 加了 unroll 后质量变差 | 寄存器 live range 冲突（accumulator 被 unpack tmp 覆盖） | 回退到上一个正确 unroll，重做 live range 表（[`arch/arm.md`](arch/arm.md) §3.5） |
| C++ SIMD 与标量 oracle 就对不上 | pack / unpack 数学本身错，与寄存器无关 | 停在 C++ 层修，**不要**进 asm——asm 只会把同一个错藏得更深 |
| intrinsic 档对、同逻辑的 asm 档错 | 寄存器 / ABI 问题（clobber、callee-saved 未恢复、结构体偏移） | 重做五个 live range 问题（[`arch/arm.md`](arch/arm.md) §3.5）与字节偏移表（同文 §3.4） |

---

## 四、交付前自查

按顺序过，每条都要有**实跑记录**，不接受「应该没问题」。

1. 六条轴（§一）每条至少两个取值，各跑一遍。
2. 目标档与 oracle 逐层对齐（§2.1）。
3. 每个非目标档与改动前逐位一致（§2.2）。
4. 至少一个真实模型的 sanity 输出（不只是 op test）。乱码判定前先固定采样变量。
5. 五个同源量逐项确认（[`pack-and-abi.md`](pack-and-abi.md) §一）。
6. 新增 `CoreFunctions` 字段都在声明处给了默认值（[`dispatch-and-register.md`](dispatch-and-register.md) §3.2）。
7. 若改了 init 时序，确认赋值位置在正确的快照点之前（[`dispatch-and-register.md`](dispatch-and-register.md) §四）。
8. 性能数字带完整环境标签（ISA / 线程数 / precision / block size / shape），否则数字不可复现。

第 4 条不能用第 1 条替代。op test 的形状分布和真实模型不一样，§1.2 那条轴就是被 op test 系统性漏掉的。

---

## 五、相关文档

- 四级实现阶梯与每级退出条件：[`SKILL.md`](SKILL.md) 铁律 1（症状 → 退回哪一层已并入本文 §三）
- 五个同源量与后处理 ABI：[`pack-and-abi.md`](pack-and-abi.md)
- 注册与派发：[`dispatch-and-register.md`](dispatch-and-register.md)
- 命令、测试用例选择、性能报告格式：[`cpu/shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md)
- 开关与降级：[`cpu/shared/env-registry.md`](../shared/env-registry.md)
