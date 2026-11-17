# §12 非有限值（inf/NaN）注入与掩盖

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
>
> **不在本文**：fp16 动态范围不足（值太大表示不下）见 [`fp16-range.md`](fp16-range.md)；
> 读脏内存见 [`memory-aliasing.md`](memory-aliasing.md)；shader 越界见 [`gpu-oob.md`](gpu-oob.md)。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

**触发**（满足以下之一强烈怀疑本类）：
- **只有带激活（relu / relu6 / clamp）的用例才错**，无激活版"恰好"通过；
- 输出**恒为 0 或恒为 clamp 边界值**，不是随机垃圾，也不是偏移一点；
- 错误值是"整齐的"有限数（如恰好 0），但无激活路径的原始输出打印出来是 `nan`；
- 只有**退化输入**（全零权重块、零 scale、空/单元素张量）才触发；
- 测试/对拍工具明明该报错却放行（比较式对 NaN 恒 false）。

## 12.1 核心心法

**"只有 relu 错"不是 relu 的 bug，是上游有 NaN。**

两条掩盖链会把同一个 NaN 藏到几乎看不见：

1. **激活函数把 NaN "定影"成有限错误值**。Metal 的 `max(nan, 0) = 0`（fmax 语义，丢弃 NaN 操作数），
   `clamp` 同理。于是 relu 版输出一个干净的 0 —— 看上去像"bias 丢了"、"写错了位置"，
   实际只是 NaN 被压成了有限值。无激活版把 NaN 原样透传。
2. **比较器把 NaN 当 pass**。`if (fabs(a-b) > tol) fail` 对 NaN 恒为 false —— 无激活版的
   NaN 输出在测试里**静默通过**。于是唯一可见的失败是被激活定影的那条，症状极具误导性
   （"relu 丢 bias"），而把 no-relu 版的 NaN 直接打印出来只需一行 readMap。

注入点通常在 **host 侧预处理**而非 kernel：除零产生 inf（如预缩放系数 `coef = 1000 / max_data`
在 `max_data = 0` 时得 inf），然后 `0 * inf = NaN` 被烘进常量表（fp16 存储的量化 scale 表、
LUT、预烘焙权重），整层输出全部被污染。kernel 数学完全正确也救不回来。

**方法论一句话**：看到"只有激活版错 / 输出恒 0"，**先直接观察无激活路径的原始输出是不是 NaN**，
再沿数据流向上找第一个非有限值 —— 不要先怀疑激活或 epilogue 的实现。

**配套陷阱：`-ffast-math` 会让 NaN/inf 守卫整体失效。** iOS toolchain
（`cmake/ios.toolchain.cmake`，所有 Release 变体）给 C++ 编译加 `-ffast-math`，其中蕴含的
`-ffinite-math-only` 让编译器假定程序里不存在 NaN/inf，于是 `std::isfinite` / `isnan` 被
**常量折叠成恒 true/false —— 守卫存在但永远不触发**。host（macOS/Linux）构建通常没有
fast-math，同一个守卫在 host 上工作正常 → "host 修好了、真机 still broken"的割裂现场。
**凡是依赖 isnan/isfinite 的守卫，必须改成不依赖 NaN 检测的形式**（如对分母加阈值判断、
钳制输入而不是事后检查结果），并在真机上验证一次。

## 12.2 排查流程

### Step 1: 直接观察原始输出（不要先猜）

在 layout/dtype convert **之前** readMap 输出张量（Express 里 `output->readMap<float>()` 先于
`_Convert(output, NCHW)` 调用），或在 kernel 出口把未激活的值写进 padding lane 读出。
看到 `nan` 即坐实本类；看到正确值则回到入口分流表重新分类。

### Step 2: 沿数据流向上找第一个非有限值

候选注入点（按概率排序）：
- **除法**：grep 预处理路径上的 `/`，逐个检查分母能否为 0（`max_data`、`sum`、`norm`、`count`）；
- **量化 scale / 预缩放系数生成**：全零权重块 → alpha 全 0 → 任何 `const / alpha_max` 都是 inf；
- **归一化**：0 方差 → `rsqrt(0)` = inf；0 均值归一 → `0/0` = NaN；
- **fp16 转换**：`(half)(0 * inf)`、`(half)inf` 直接进表。

### Step 3: 跨后端对照同款逻辑

同一预处理在别的后端往往有守卫。本案例中 OpenCL 的同款系数计算有
`if (abs(max_data) >= 0.000001f)` 守卫而 Metal 没有 —— **跨后端 diff 是找缺失守卫最快的办法**。

### Step 4: 修复 + 加固比较器

- 注入点加守卫（除零 / 非有限结果回退到安全值），而不是在 kernel 里兜底 —— kernel 不该为
  非法常量表负责；
- **守卫形式必须扛得住 `-ffast-math`**（iOS 全量，见 12.1 配套陷阱）：用
  `if (max_data > 1e-35f) coef = 1000 / max_data;` 这种对**分母**的阈值判断，而不是
  `if (!isfinite(coef))` 这种对结果的 NaN 检测 —— 后者会被常量折叠掉；
- 测试比较器必须让 NaN **失败**：把 `if (err > tol) fail` 改成 `if (!(err <= tol)) fail`
  （NaN 使 `<=` 为 false，这个写法本身不依赖 isnan，fast-math 下依然有效），如需容忍
  "参考也是 NaN"的用例再显式放行 `isnan(a) && isnan(b)`（注意 fast-math 平台上该放行失效，
  NaN==NaN 会被判 fail，通常可接受）。

## 12.3 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| 只有 relu/relu6 版错，输出恒 0 | 上游 NaN 被 `max(nan,0)` 定影（本册） |
| 输出恒 clamp 边界（如全 6） | NaN 被 clamp 定影（具体落 0 还是上界依实现） |
| no-relu 版"通过"但原始输出是 nan | 比较器 `> tol` 写法放行 NaN |
| fp32 对、fp16 错，且是**退化输入**触发 | 只在 fp16 路径跑的预处理（预缩放 coef）除零 |
| 所有 shape 都错、值是随机垃圾 | 不是本类，查 [`memory-aliasing.md`](memory-aliasing.md) / [`gpu-oob.md`](gpu-oob.md) |

## 12.4 参考案例：weighti8i4conv2d Metal relu-only 1/255 偏差（全零权重块除零）

**症状**：单测 `op/convolution/weighti8i4conv2d`（1×1×1×1 int8 卷积，weight=0，bias=1/255）在
Metal fp16 下**只有 relu=1 的迭代失败**，输出 0；no-relu 迭代通过；CPU 全过。看上去像
"Metal kernel 在 relu 下丢 bias"。

**排查路径**：
1. 静态读 kernel（`conv1x1_gemv_g4m1_2sg_wquant_sg` epilogue）——bias 先加后 activate，
   数学上不可能把 0.0039 变成 0，排除 epilogue；
2. 怀疑输入 NC4HW4 padding 带 inf/NaN 垃圾（`result = raw_dot*scale + input_sum*db`，
   scale=0 时 `0*inf=NaN`）——但 converter（`gNC4HW4Convert`）对 padding 零填充，排除；
3. **直接观察**：在 `_Convert` 前 readMap —— **no-relu 版原始输出就是 nan**，relu 版是 0。
   一次打印同时解释两个版本，并暴露比较器放行 NaN 的问题；
4. 向上找注入点：`getDequantScale`（MetalConvolutionCommon.mm）的 fp16 预缩放
   `coef = 1000.0 / max_data`，全零 alpha 块使 `max_data = 0` → `coef = inf` →
   存储的 `(half)(0 * inf) = NaN` 进 dequant scale 表 → kernel 输出全 NaN；
5. 跨后端对照：OpenCL 同款代码有 `>= 1e-6` 守卫，Metal 漏了；
6. **host 修复后真机仍错**：加打印发现真机上 `coef` 仍为 inf —— iOS toolchain 的
   `-ffast-math` 把 `!std::isfinite(coef)` 常量折叠为 false，守卫从未触发
   （host 构建无 fast-math，所以 host 上"修复"是真的）。改为分母阈值守卫后真机通过。

**根因**：host 预处理的除零，不是 kernel；relu-only 失败是 NaN 被 `max(nan,0)=0` 定影后的假象。
第二层的 host/真机差异是 `-ffast-math` 使 isfinite 守卫失效，不是第二处 NaN 源。

**修复**：
- `MetalConvolutionCommon.mm::getDequantScale`：`if (max_data > 1e-35f) coef = 1000.0 / max_data;`
  —— 分母阈值判断，不依赖 isnan/isfinite，fast-math 下有效；
- `test/TestUtils.h` 三个 checkVector 比较器：`> tol` 改 `!(<= tol)`，NaN==NaN 显式放行 ——
  否则同类 bug 的 no-relu 路径继续静默通过。

**避坑要点**：
- 排除项与误导：**kernel epilogue**（bias/activate 顺序完全正确，review 浪费时间）和
  **输入 padding 垃圾**（converter 有零填充，且若真脏两版都会错，不是只有 relu 错）；
- "只有激活版错"应直接触发 NaN 假设，一行 readMap 即可验证，不要先读几百行 shader；
- 测试报错文案（"expect 0.003922, real 0"）描述的是定影后的值，**不是** kernel 的真实输出。

## 12.5 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/backend/metal/MetalConvolutionCommon.mm` | `getDequantScale`：fp16 预缩放 coef 生成（本案例注入点与修复处） |
| `source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp` | 同款系数计算的参照实现（带 `>= 1e-6` 守卫） |
| `source/backend/metal/MetalBackend.mm` | `gNC4HW4Convert`：NCHW→NC4HW4 converter，padding 零填充（排除项证据） |
| `test/TestUtils.h` | checkVector / checkVectorByRelativeError：NaN 放行的比较器（已加固） |
| `test/op/ConvolutionTest.cpp` | weighti8i4conv2d 用例：activations 循环（relu 版先暴露） |
