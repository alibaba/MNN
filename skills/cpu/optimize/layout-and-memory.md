# L3 Layout / 内存

> **何时读**：[`diagnose-and-route.md`](diagnose-and-route.md) 把问题定位到 L3——pack 格式、weight reorder、
> 分块粒度、访存次数、peak RSS。
>
> **不在本文**：
> 线程与划分 → [`runtime-and-scheduling.md`](runtime-and-scheduling.md)；
> 命令与测试 → [`shared/build-test-and-benchmark.md`](../shared/build-test-and-benchmark.md)；
> 开关 → [`shared/env-registry.md`](../shared/env-registry.md)；
> ISA 路径 → [`arch/arm.md`](arch/arm.md) / [`arch/x86_64.md`](arch/x86_64.md)；
> kernel 内部与 packer 实现 → [`cpu/kernel/pack-and-abi.md`](../kernel/pack-and-abi.md)；
> 正确性 bug → [`bugfix.md`](bugfix.md)。

---

## 一、`pack` / `bytes` / `matmulBytes`：三个运行期变量，不是常量

L3 的一切都建立在这三个数字上，而它们**由运行期选中的函数表决定**。把它们当常量是这一层最常见的错误。

| 字段 | 声明 | 语义 | 实测取值 |
|------|------|------|---------|
| `pack` | `CommonOptFunction.h`（**无 NSDMI**） | NC4HW4 里 C 方向打包宽度 | 基表 4（`CommonOptFunction.cpp`）；AVX2 8（`AVX2Functions.cpp`）；AVX512 16；arm82 8（`Arm82Functions.cpp`） |
| `bytes` | `CommonOptFunction.h`（**无 NSDMI**） | 一个"float"占几字节 | 基表 4（`CommonOptFunction.cpp`）；arm82 2（`Arm82Functions.cpp`） |
| `matmulBytes` | `CommonOptFunction.h`（有 `= 0`） | dense matmul 的 A/B 字节数，**C 仍用 `bytes`** | 只有 BF16 设为 2（`BF16Functions.cpp`），其余为 0 = 与 `bytes` 相同 |

三个推论：

1. **`pack` 和 `bytes` 在声明处没有默认值**。基表和二级表都赋了值所以现在没事，但这正是
   `new CoreFunctions` 是**默认初始化而非零初始化**的活证据——新加字段务必在声明处写 `= nullptr`/`= 0`，
   机制见 [`cpu/kernel/dispatch-and-register.md`](../kernel/dispatch-and-register.md) §3.2。
2. **`matmulBytes != 0` 是不对称的**：A/B 用 `matmulBytes`，C 用 `bytes`。消费点必须逐个分支，
   现有三处：`CPUMatMul.cpp`、`DenseConvolutionTiledExecutor.cpp`、
   `kleidiai/KleidiAIDenseConvolution.cpp`。新写 matmul 路径时漏掉这个分支，BF16 会静默算错。
3. **BF16 不是"把一切变成 2 字节"**：`BF16Functions.cpp` 是 `new` + 整体拷贝基表，
   之后只改了 5 个 matmul 相关指针加 `matmulBytes`，**`pack` 仍是 4、`bytes` 仍是 4**。
   另外它 `#if !defined(MNN_USE_NEON) return false`，x86_64 上根本不存在。

`x86_64` 侧 pack 由能力位决定这一点还有个反直觉后果：**同一份 NC4HW4 代码在 AVX512 机器上是 NC16HW16**。
任何写死 4 的 stride 计算都会在 AVX512 上错。

---

## 二、分块粒度：逻辑块 ≠ 物理 chunk

这是 L3 最贵的一类坑，因为它**不崩、不报错，只在越过某个长度后开始出错**。

### 2.1 机制

一个缓存通常有两个独立的粒度：

- **逻辑块**：一次计算处理多少行（如 flash attention 的 kv block）。
- **物理 chunk**：内存里按多少行为一段组织（决定地址公式）。

两者可以不同（逻辑块是物理 chunk 的整数倍时合法），但**取址公式必须按物理 chunk 写**：
`chunk 索引 + chunk 内偏移 + bExtraStride`。只放宽逻辑块而地址仍按旧 chunk 算，
短序列（不跨 chunk）全对，越过第一个 chunk 边界后开始读错行。

真实事故：`0fd8efff1e` 把逻辑块放宽到 256 却仍按 64 行 chunk 取址，kv > 64 后读错行——
**靠长 prompt 金丝雀才捕获，op 单测全绿**。

反向教训同样重要：把物理 chunk 也放大到 256、逻辑块保持 64，在 t4 掉约 4%
（物理行距 4x，L2/TLB 压力）。**两个粒度要分别调，不能绑在一起改。**

### 2.2 仓库里的真实例子

`CommonOptFunction.h`（都在 `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` 内）：

```
MNN_FLASH_ATTENTION_BLOCK_SIZE   = 64      // 通用
MNN_FLASH_ATTENTION_BLOCK_DECODE = 2048    // 单线程 decode 专用
```

`compute/CommonOptFunction.h` 里 `MNN_FLASH_ATTENTION_BLOCK_DECODE` 的声明注释给出了 2048 的来历：单线程 decode 没有 causal-mask 浪费，大 kv block 摊薄每块固定成本
并把 K/V 连续流从 64KB 拉到 2MB，实测 DRAM 有效带宽 35 → 49 GB/s；
kv≈4160 上 256→512→1024→2048→4096 = 14.7→12.9→11.8→11.1→11.2 ms，**2048 已经撞到单核流墙**。

选择逻辑在 `CPUKVCacheManager.hpp flashAttentionChunkKv()`，只在
**单线程 + V 不量化 + K 是 None/Int8** 时才用宽 chunk。该函数自身的注释写清了为什么其余情况留在 64：
多线程 decode 会退化，且 V-int8 的 PV 调用点**写死了** `MNN_FLASH_ATTENTION_BLOCK_SIZE`。

这就是"写死旧值"的典型：放宽粒度时必须 grep 所有直接引用该常量的地方。

### 2.3 改分块粒度的预检清单

- [ ] 物理 chunk 尺寸有没有跟着变？地址公式是按哪个算的？
- [ ] 整除门限：`block % hP`、`chunk % pack`、`block % DST_XUNIT` 是否仍成立？
- [ ] 量化路径里有没有写死旧值（grep 常量名，不要只看 kernel）？
- [ ] 是否与线程数联动（宽块常只对单线程有利）？
- [ ] 正确性用例是否覆盖了**跨过 chunk 边界**的长度？只测短序列等于没测。

---

## 三、低 bit 权重布局：stride 必须是真实字节数

低 bit 的 packed cell 常有 padding，于是出现两个不同的数：

- **useful payload**：`bits / 8` 的理论比例（w4 = 0.5 字节/元素）。
- **真实 packed cell 字节数**：含 padding 与 metadata 的实际步进。

**所有指针推进必须用后者。** 用前者的典型症状是：单线程对、多线程（`tId > 0` 的 OC chunk）错，
或者能跑不崩、只是模型输出质量变差。

相关坐标：`ConvInt8TiledExecutor.cpp` 的 `reorderWeight()`、`packWeightAndQuantInfo()`。
packer 与 kernel 必须配套（SME2 的 packer 喂给 i8mm/sdot kernel时形状仍然"合法"）——
这属于 L3↔L4 交界，契约与实例都在
[`cpu/kernel/pack-and-abi.md`](../kernel/pack-and-abi.md) §四。

---

## 四、内存：两条完全不同的路径

**把"占用高"和"慢"分开**，也把"静态缓存"和"per-forward scratch"分开。这四者互不相干，
混着看必然误判。

### 4.1 STATIC 与 DYNAMIC 走不同的分配器

`CPUBackend::allocBuffer`（`CPUBackend.cpp`，由同文件的 `onAcquire` 转发过来）：

| storageType | 分配器 | 实例 | 用途 |
|-------------|--------|------|------|
| `STATIC` | `mRuntime->mStaticAllocator` | **`EagerBufferAllocator`**（`CPUBackend.cpp`，恒定） | 权重、KV cache 等跨 forward 存活的东西 |
| `DYNAMIC` / `DYNAMIC_SEPERATE` | `mDmaInfo->mCurrentDynamicAllocator` | 由 hint 选，**默认 Defer** | per-forward scratch |

默认值容易记反：`RuntimeHint::memoryAllocatorType = 0`（`Backend.hpp`）而
`Allocator_Defer = 0`、`Allocator_Eager = 1`（`Backend.hpp`），
所以**动态分配器默认是 Defer，静态分配器恒为 Eager**。

 还有一条容易忽略的规则：如果 tensor 已有的 mem `>= size`，**原地复用、不重新分配**。
即 buffer 只涨不缩。

### 4.2 两个分配器的复用语义不同 —— 这决定 peak RSS

| 分配器 | 找不到足够大的空闲块时 | 后果 |
|--------|--------------------|------|
| `EagerBufferAllocator` | `lower_bound` 到 `end()` → **返回 nullptr**（`BufferAllocator.cpp`），上层去向 OS 新申请 | 复用落空 = 一次 fresh OS alloc，**peak RSS ≈ 累计 fresh alloc** |
| `DeferBufferAllocator` | `lower_bound` 到 `end()` → **`--iter` 取最大的那块**，再 `selectChunk->size = size` **扩容**它 | 不新申请，把已有块撑大 |

`free()` 都**不还内存给 OS**，只挂回 free-list；只有 session 销毁（`release(allRelease)`）才真正归还。

于是"增长型缓存的 peak RSS"有了明确机制。KV cache 走 STATIC → Eager：

- **平坦布局**：字节数 = 长度的一次函数，每次扩容都比刚释放的旧块**大一点** →
  `lower_bound` 恒落空 → 每次扩容都 fresh alloc → RSS 单调上涨。
- **量化到 chunk**：字节数是阶梯函数，一个 chunk 窗口内多次扩容**尺寸恒定** →
  旧块被复用 → RSS 基本持平。

**推论：把扩容粒度对齐到物理 chunk 边界能降 peak RSS，而这与"单次 buffer 多大"无关**——
per-forward scratch 甚至可能因为 block 变大而变大（短上下文 peak 略升）。两者方向相反、量级不同
（KB 级 vs GB 级）。真实数据：`bb6bdcf827` 直接复用宽 V chunk 使 peak RSS **+190MB**。

### 4.3 诊断手法：埋点实测，不要推断

**不要凭 realloc 的触发条件推断内存机制。** 案例里就因此误判过一次：以为"粗粒度 → 全程零扩容"，
实际触发条件 `maxLength = kv + mExpandChunk` 与粒度无关，两种粒度扩容次数**完全相同**。

四步（用完即删，不提交）：

1. **fresh alloc 计数**：在 `EagerBufferAllocator` 复用落空、真正 `onAlloc` 的分支上，
   对 ≥ 阈值（如 2MB）的块打印 `size` 与累计 `mTotalSize`。
2. **per-tensor churn 归因**：在扩容函数（`CPUKVCacheManager::expandKVCacheInMem`）里，
   alloc K 前/后、alloc V 后各取一次 fresh 字节快照，差值即各自 churn——避免把多类缓存混算。
3. **env 覆盖粒度做同 build A/B**：给粒度选择逻辑加临时 env override（chunk=64/256/2048），
   一次构建对比完，不要反复切 git 重建。
4. **外部 RSS 时间线印证**：
   `while kill -0 $PID; do ps -o rss= -p $PID; sleep 0.2; done`。
   与 fresh-alloc 日志**互相对得上才算数**。

### 4.4 两套动态分配器与 resize

`onSelectDynamicAllocator(index, maxIndex)`（`CPUBackend.cpp`）支持第二套动态分配器
（`mDynamicAllocatorBackup`，惰性创建），配合 `mCacheGroup` 切换。
排查"内存翻倍"时先确认是不是同时活着两套。

---

## 五、删 memset 之前，先找出谁依赖这块内存已被清零

`bb6bdcf827` 在**分配时**去掉了一句"看起来多余的" V memset（单线程 flash float 路径，
2048 行窗口下整块清零会多摸掉近一个空闲 chunk 的页，占该次回归 ~160MB）。
两件事必须分清：

**1. 失效机制不是"读到脏数据"，而是 `0 × Inf/NaN = NaN`。**
PV matmul 会算到 `ROUND_UP(length, lP)` 行，超出真实长度的 lP-padding 行的 softmax
权重**恰好是 0**——所以第一直觉是"乘 0 就无所谓"。但 arena 垃圾可能本身就是 Inf/NaN，
`0 × Inf = NaN`，NaN 会污染整行输出。判断"谁依赖清零"时要问的是**下游会不会与这块内存
做乘加**，不是"下游会不会用它的值"。

**2. 分配时跳过清零 ≠ 不用清零；它把清零挪到了写入时，两者作用在不同时刻。**

| | `bb6bdcf827`：分配时 | `7efbf0a98b`：写入时 |
|---|---|---|
| 时机 | `onAlloc`，一次 | append 前沿进入一个新 lP-tile 时 |
| 范围 | 整个 V buffer（含远超当前长度的空闲 chunk） | 只有当前前沿那一个 tile |
| 现状 | 单线程 flash float **跳过**；多线程保留 | 补上前沿 tile 的清零 |

多线程为什么保留分配时 memset：它的 64 行 chunk 让 buffer 本来就小，省不下 RSS，
而且实测**依赖**清零的 padding 行（去掉就乱码）。锚点 `CPUKVCacheManager.cpp`。

**3. "写时补零"本身可行，错的是粒度。** 最初的尝试是补**连续的尾行**，但 V 行在 tile 内
按 `lP` 交错存放，连续 memset 会溢出到下一个 tile。正确做法是**整块清 tile**：
前沿跨进新 tile 时按 `UP_DIV(mHeadDim, hP)` 个 hP 组各清一段，
既保证 `ROUND_UP(length, lP)` 以下全部初始化，又不碰已用区之外的页。
锚点 `CPUKVCacheManager.cpp`。

规则：

- 删任何 memset 前，明确写出**谁**依赖这块内存已清零（哪个 kernel、哪个尾部分支），
  并检查它是否与这块内存做乘加（`0 × Inf/NaN` 陷阱）。
- 分配时的清零被删掉后，要写清"改由谁在什么时刻补"，不要只记"这里不用清零了"。
- 补零的粒度必须匹配物理布局单元（tile），不能按逻辑行数连续 memset。
- 交叉验证维度至少：多线程 × 多 chunk 尺寸 × 量化/非量化。单一配置绿了不能推断。

ARM 侧的事故索引见 [`arch/arm.md`](arch/arm.md) §四。

---

## 六、访存次数：融合语义优先于 kernel 复用

带宽敏感场景下，把一个融合算子拆成几次 `CoreFunctions` 调用**会净变慢**：省的是开发量，
付的是多遍访存。优先扩展现有入口的签名以保留融合语义。

判断方法（替换法）：算出理论访存字节数（融合 vs 拆开），与实测 eff GB/s 对照。
如果拆开后的实测时间接近"访存字节数比例 × 原时间"，就确认是访存次数问题而不是 kernel 效率问题。

扩签名要同步改所有架构实现与调用点，清单见
[`cpu/kernel/dispatch-and-register.md`](../kernel/dispatch-and-register.md) §3.3。

---

## 七、改动前自查

- [ ] 用到的 `pack` / `bytes` 是从 `core->` 读的，还是写死的常量？
- [ ] 有没有新增 `CoreFunctions` 字段而忘了在声明处给默认值？
- [ ] 改了分块粒度：物理 chunk、地址公式、整除门限、写死的常量、线程数联动，五项都过了吗？
- [ ] 低 bit 指针推进用的是真实 packed cell 字节数吗？测了 `tId > 0` 吗？
- [ ] 动了内存路径：分得清 STATIC/DYNAMIC、Eager/Defer 吗？peak RSS 有 fresh-alloc 埋点实测吗？
- [ ] 删了 memset：谁依赖清零、会不会与它做乘加（`0 × Inf/NaN`）、改由谁在什么时刻补，都写出来了吗？多线程 × 多 chunk 交叉验证了吗？
- [ ] 正确性用例覆盖了跨 chunk 边界的长度、以及 AVX512（`pack == 16`）这类非 4 的 pack 吗？

---

## 八、代码坐标速查

| 内容 | 位置 |
|------|------|
| `pack` / `bytes` / `matmulBytes` 声明 | `compute/CommonOptFunction.h` |
| 基表赋值 | `compute/CommonOptFunction.cpp` |
| AVX2 / AVX512 的 pack | `x86_x64/AVX2Functions.cpp` |
| arm82 的 pack / bytes | `arm82/Arm82Functions.cpp` |
| BF16 表（整体拷贝 + 只改 matmul） | `cpu/bf16/BF16Functions.cpp` |
| `matmulBytes` 消费点 | `CPUMatMul.cpp`、`DenseConvolutionTiledExecutor.cpp`、`kleidiai/KleidiAIDenseConvolution.cpp` |
| flash attention 两个块尺寸 | `compute/CommonOptFunction.h`（紧邻的注释里有实测数据） |
| 宽 chunk 的启用条件 | `CPUKVCacheManager.hpp` |
| 分配时跳过 V memset（单线程 flash float） | `CPUKVCacheManager.cpp` |
| 写入时清前沿 lP-tile | `CPUKVCacheManager.cpp` |
| 权重 reorder / pack | `compute/ConvInt8TiledExecutor.cpp` |
| 分配入口（含 `>= size` 原地复用） | `CPUBackend.cpp` 的 `allocBuffer` |
| 静态分配器（恒 Eager） | `CPUBackend.cpp` |
| 动态分配器选择（默认 Defer） | `CPUBackend.cpp`；枚举在 `Backend.hpp` |
| Eager 复用落空 → nullptr | `core/BufferAllocator.cpp` |
| Defer 复用落空 → 扩容最大块 | `core/BufferAllocator.cpp` |
| 第二套动态分配器 | `CPUBackend.cpp` |
