# §5 数值精度 / fp16 表示能力不足

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
>
> **不在本文**：强制 fp32（`precision: high`）仍错 ⇒ 不是表示能力问题，回
> [`export-and-quant.md`](export-and-quant.md)（导出权重）或 [`kernel-assumptions.md`](kernel-assumptions.md)（kernel 假设）。
> §5.6 收的是「实时计算 → 预计算查表」这类重构自带的三个陷阱，做该类重构时**必读**。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

**触发**（满足以下之一强烈怀疑本类）：
- 长 prompt / 长上下文输出**重复、漂移、退化**，短 prompt 完全正常；
- fp16 后端错、强制 fp32（config `"precision": "high"`）对；
- **所有 fp16 后端一致地错**（Metal 和 CPU arm82 同样错）—— 与 [`memory-aliasing.md`](memory-aliasing.md) 的"一个后端错一个对"相反；
- 出错阈值恰好是**2 的幂**（2048 / 4096 / 8192）；
- torch 侧 `--test` 正常（torch 用 fp32 跑）。

## 5.1 核心心法

**"fp32 对 / fp16 错 + 只有长序列错 + 阈值是 2 的幂" ≈ 图里存在动态范围过大的中间张量。**

fp16 只有 10 bit 尾数，**整数的精确表示上限是 2048**（2^11）：

| 数值区间 | fp16 可表示的最小间隔 |
|---|---|
| < 2048 | 1（精确） |
| 2048 ~ 4096 | 2 |
| 4096 ~ 8192 | 4 |
| 8192 ~ 16384 | 8 |

也就是说 fp16 里 `2048.0` 和 `2049.0` 是**同一个数**。任何把"大整数"或"大整数 × 系数"作为中间张量烘进计算图的做法，在 fp16 后端都会让相邻取值塌缩成 bit 完全相同的结果。

而 LLM 的 `position_ids` 可以到 128k，远超这个上限。

**方法论一句话**：**沿数据流找"绝对值最大的中间张量"，把它改写成值域小的等价形式**——不要指望后端"精度高一点"能解决，这是表示能力的硬上限。

## 5.2 相关背景

- MNN 的 fp16 由 backend precision 决定：Metal 默认 fp16、CPU 走 arm82 时也是 fp16。config 里 `"precision": "high"` 可强制 fp32。
- 导出侧任何写成 `x.float() * const` 的表达式都会变成图里一个真实的中间张量，**它的值域就是后端要承受的动态范围**。导出期用 torch fp32 验证是发现不了的。
- LLM 里典型的大值域中间量：`position_ids`（0~128k）、`position * inv_freq`（RoPE 角度，可达上万弧度）、未归一化的 logits、累加型 reduce 的中间和。

## 5.3 排查流程

### Step 1: 用 precision 开关分流（第一步，代价最小）

```bash
# 在 config.json 里加 "precision": "high" 强制 fp32
./llm_demo config_fp32.json prompt.txt
```

fp32 对、fp16 错 → 基本坐实本类，不用再去查 kernel 逻辑或内存。

### Step 2: 长度扫描找阈值

```bash
for L in 512 1024 2048 3000 4096 8192; do
  echo "=== len=$L ==="; ./llm_demo config.json /tmp/prompt_${L}.txt 32
done
```

**阈值落在 2 的幂上是最强的信号**。若阈值 = 2048/4096，直接对照 5.1 的间隔表反推是哪个量越过了精确表示区间。

### Step 3: 逐 step logits 对拍，定位第一个发散点

用 `llm_logits_diff` 工具（`transformers/llm/engine/demo/llm_logits_diff.cpp`）做**teacher-forced** 对比——A 后端的 argmax 同时喂给两个模型，保证每一步比较的是同一 KV/history 状态下的 logits：

```bash
./llm_logits_diff config_fp32.json config_fp16.json prompt.txt 64
```

输出每步的 argmax 是否一致、margin、maxAbsDiff、KL(A||B) 和 teacher-forced NLL。关键读法：
- **前 N 步 byte-identical、第 N+1 步突然发散** → 找出第 N+1 步对应的绝对位置，往往正好是阈值；
- KL 逐步单调放大 → 累积型误差；KL 在某步跳变 → 表示能力塌缩（本类）。

### Step 4: 定位是哪个中间张量

在导出侧 dump 候选中间量的绝对值上界，找出超过 2048 的那个。RoPE 场景直接看 `position * theta` 的量级：position 到 128k、theta 最大接近 1 → 角度可达 1e5 弧度，远超 fp16 的整数精确区。

### Step 5: 改写成小值域等价形式

不是降精度要求，而是**做数学等价变换让所有中间量都落进 fp16 的舒适区**。常用手法：

- **整数-余数拆分 + 查表**：`p = S*q + r`（整数运算精确），再用角度和公式合成，所有中间量落在 `[-1, 1]`；
- **提前折叠周期**：三角函数按 `mod 2π` 折叠（在 float64 下算完再降精度）；
- **减去最大值再 exp**：softmax 的标准做法，同类思路；
- **保持整数就是整数**：不要过早 `.float()`，整数索引类的量一路用 int 传到 Gather。

## 5.4 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| fp32 对 / fp16 错，长序列才错 | 中间张量动态范围超 fp16（本册） |
| 阈值恰为 2048 / 4096 | 整数在 fp16 里塌缩（5.1 间隔表） |
| 所有 fp16 后端一致地错 | 图结构问题（导出侧），不是某后端 kernel |
| 输出"大段重复" | RoPE / 位置编码相关（相邻位置塌缩成同一个） |
| 误差随步数单调放大 | 累积误差，非表示能力（可能是正常的 fp16 噪声） |
| 换 fp32 仍错 | 不是本类，回查 [`export-and-quant.md`](export-and-quant.md)（导出权重）或 [`kernel-assumptions.md`](kernel-assumptions.md)（kernel 假设） |

## 5.5 参考案例：RoPE position 在 fp16 下塌缩（长中文 prompt 大段重复）

**症状**：长中文 prompt 回答出现大段重复；短 prompt 正常；fp32 正常；Metal 与 CPU arm82 一致地错。

**根因**：导出侧 RoPE 把 `position_ids.float() * theta` 直接烘进图。position ≥ 2048 后 fp16 无法精确表示，相邻位置产生 **bit 完全相同**的 cos/sin —— 模型对"第 3000 个 token"和"第 3001 个 token"的位置感知完全一致，4096 以后（间隔 4）彻底失去位置区分能力 → attention 退化 → 复读。

**修复**：把位置保持为整数，拆成 `p = 2048*q + r`（整数运算精确），用两张预计算表 + 角度和公式合成：

```
angle(p) = q*(2048*theta) + r*theta   (mod 2pi)
cos(p)   = cosH[q]*cosL[r] - sinH[q]*sinL[r]
sin(p)   = sinH[q]*cosL[r] + cosH[q]*sinL[r]
```

表的角度在 **float64** 下折叠进 `[0, 2π)` 后再降 fp32。这样后端接触到的每个张量都在 `[-1, 1]`，fp16 精度 ~6e-4，只剩无害的相位抖动。

**避坑要点**：这个 bug 无法通过 review 逻辑代码发现——`position * theta` 数学上完全正确，torch fp32 下验证也完全正确。**必须靠"fp32/fp16 A/B + 阈值是否 2 的幂"来揭穿。**

## 5.6 「实时计算 → 预计算查表」重构的三类陷阱（重要）

5.5 的修复把"每次实时算"改成了"构造期预计算 + 运行期查表"。这是一类通用优化手法（也见于各种 LUT 化、常量折叠、预烘焙），**它本身会引入三个新的 bug 类别**，全部在 code review 阶段才被发现。改这类代码时逐条自查：

### 陷阱 A：构造期固化的数据与后续参数修改脱钩（最严重）

**机制**：改造前 `forward()` 读 `self.theta`，外部改 `theta` 立刻生效；改造成查表后，表在 `__init__` 里烘焙，`forward()` 不再读 `self.theta`——**任何在构造之后修改输入参数的代码路径都会静默失效**。

**实例**：`utils/model.py` 的 Gemma3/Gemma4 dual-RoPE 路径先 `Rotary(full_config)` 建表（`head_dim=256` → 表宽 128），再按 `partial_rotary_factor=0.5` 改 `rotary_dim=128` 和 `theta`。表还是旧的 → 生成全宽 RoPE，把本该 pass-through 的维度也旋转了，且 theta 频率分布也错 → 导出模型输出胡言乱语，**不报错、不崩溃**。

**自查清单**：
- `grep` 所有对该对象属性的外部赋值（`\.theta\s*=`、`\.rotary_dim\s*=`），确认没有发生在构造之后；
- 若必须允许后置修改，把建表提成**公开方法**（如 `build_rope_tables()`）并要求调用方改完显式重建；注释里写明"Must be re-called by anyone mutating X"；
- 检查**子类**：子类 `__init__` 里 `super().__init__()` 之后改参数，是同一个陷阱。（本案例中 `DitRotary`/`OmniRotary`/`VisionRotary` 恰好都完整覆写了 `forward()` 不走查表路径，才躲过一劫——这是运气不是设计。）

### 陷阱 B：导出图丢失了框架的边界检查

**机制**：查表 = `embedding` / `Gather`。**PyTorch 的 `embedding` 有边界检查会抛 `IndexError`，但导出成 MNN 图后的 Gather 算子不做边界检查**——越界索引直接读表外内存，把垃圾当数据用。

**后果特征**：导出期一切正常（torch 会拦），线上真机长上下文才爆，且不是崩溃而是静默读脏数据。这是最难排查的组合。

**自查清单**：
- 表的行数是否覆盖索引的**理论最大值**，而不只是"典型值"？（`max_position_embeddings` 不是硬上限，NTK / 用户改 config 外推都会超）；
- 高表很小的时候直接**放大预留**：本案例高表预留 `4 * max_pos`，增量成本仅 ~96KB（d=128/128k 模型高表 33KB → 129KB），预留范围内的角度是数学精确的；
- 注意算清**哪张表是大头**：低表行数固定 2048、与 `max_pos` 无关，`2048 × (rotary_dim/2) × 4B` 才是主要开销（d=128 时两张低表共 1MB，d=256 时 2MB）；4 张表合计 d=128/128k 约 1.1MB、d=256/128k 约 2.25MB。放大高表预留几乎免费，放大低表则不然；
- 再在 `forward()` 里加 `clamp` 作为**最后兜底**——但注意陷阱 C。

### 陷阱 C：加 clamp 会破坏与之配对的运算

**机制**：为修陷阱 B 给索引加 `clamp`，会让原本互相配对的两个量**解耦**。

**实例**：原代码 `q = floor(pos/2048)`、`r = pos - q*2048`。对负 `pos` 是安全的（floor 除法和减法天然配对：`pos=-1` → `q=-1` → `r=2047` ✓）。但一旦给 `q` 加 `clamp(0, ...)`，clamp 把 `q` 从 -1 抬到 0，`r = -1 - 0*2048 = -1` → **负索引越界**。修 B 的动作直接制造了新的越界。

**修法**：让配对量各自独立成立，不要依赖对方。本案例 `r` 改用 `torch.remainder(pos, 2048)`——数学模运算，结果恒在 `[0, 2048)`，与 `q` 怎么 clamp 完全无关。

**自查清单**：给某个量加钳制/饱和后，`grep` 所有用到它的表达式，逐个确认不变量是否仍成立。

### 验证方式

这三类陷阱都不产生异常，必须主动验证：

```python
# 1. 表宽/表长与最终参数一致（陷阱 A）
print(r.rotary_dim, r.rope_cos_low.shape[1])   # 应为 rotary_dim//2

# 2. 极端索引不越界（陷阱 B/C）
for p in [0, 2047, 2048, max_pos+5000, 999999, -1]:
    q, r_ = ...; assert 0 <= q < high_entries and 0 <= r_ < split

# 3. 范围内与实时计算逐点对齐（保证等价变换没写错）
ref = torch.cos(pos.double().reshape(-1,1) * theta.double()).float()
assert (table_out - ref).abs().max() < 1e-6
```

## 5.7 相关文件索引

| 文件 | 作用 |
|------|------|
| `transformers/llm/export/utils/transformers.py` | `Rotary.build_rope_tables()` / `forward()`——本案例修复处，注释里写了 fp16 拆分原理 |
| `transformers/llm/export/utils/model.py` | Gemma3/Gemma4 dual-RoPE 后置修改 theta 的位置（陷阱 A 实例） |
| `transformers/llm/engine/demo/llm_logits_diff.cpp` | 逐 step teacher-forced logits 对拍工具（Step 3） |
| `transformers/llm/engine/src/llm.cpp` | `gen_position_ids`——超过 `max_position_embeddings` 时打 warning |
| `transformers/llm/engine/src/llmconfig.hpp` | `max_position_embeddings()` 等 config 字段解析 |
