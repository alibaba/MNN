# §8 持久化缓存误信（weight-mmap cache / 陈旧缓存）

> **归属**：[`general-debug`](SKILL.md) 的分类分册之一，先在入口的分流表确认类别再读本文。
>
> **不在本文**：权重从导出那一刻就是坏的（与缓存无关）见 [`export-and-quant.md`](export-and-quant.md)；
> 拿到的不是缓存而是同进程内的野内存见 [`nondeterminism.md`](nondeterminism.md) §9。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

**触发**（满足以下之一强烈怀疑本类）：
- 开启 `use_mmap`（权重 mmap 落盘）后输出乱码/单字符刷屏（`!!!`、连续换行），关掉 `use_mmap` 或 `use_cached_mmap` 就好；
- **同一二进制：App 内错、`llm_demo` 对**（或反之）；iOS/Android 真机错、Mac/host 对；
- 清空 tmp/缓存目录后第一次跑就好，之后又坏；换一个 `tmp_path` 就好；
- 前几个 token 正常、随后整段崩坏（部分权重是真的、部分是垃圾的典型混合特征）。

## 8.1 核心心法

**"缓存是否有效"只能在运行起点判定一次。** `use_cached_mmap` 的契约是"上一个进程写完整套权重并留下 sync 标记 → 本次按相同分配顺序直接复用磁盘内容"。这个契约有两个隐含前提，破坏任何一个都是静默乱码：

1. **标记不能被本次运行自己写的 sync 污染**（判定时机必须在 mmap 分配器创建时刻，之后不可变）；
2. **缓存文件必须属于同一个模型**（缓存文件名前缀 `0_0_0_0_` 只含 precision/memory/power，**不含模型标识**——换模型不换目录必然拿到错误权重）。

另外牢记：跳过权重读取的 execution 拿到的 STATIC buffer **必须真的来自 mmap 池**。首次 `onClearBuffer` 后静态分配器切回 RAW malloc（封池），此后任何被重建的带权重 execution 若仍处于"信任缓存"模式，就是在拿未初始化内存当权重。

## 8.2 排查流程

1. **配置对齐分流**：App 与 demo 的默认配置差异先列全（`use_mmap` / `use_cached_mmap` / `tmp_path` / 加载次数）。"App 错 demo 对"大概率不是平台问题，是配置或加载模式差异。
2. **缓存卫生三连**：换全新 `tmp_path` → 跑一次；同目录再跑一次（warm）；换另一个模型同目录跑（污染探测）。三个结果就能区分"自我污染 / warm 复用坏 / 跨模型污染"。
3. **复刻加载模式**：iOS App 是"启动预载 + 使用时重载"的**同进程双加载**；`llm_demo` 是单加载。双加载可疑时写 20 行的 double-load 复现器（load → destroy → 清目录 → load → generate），在 host 上复现比真机埋点便宜一个量级。
4. **埋点看 hint 演化**：在 `CPURuntime::onCreate` 的 weightMemoryPath 分支打印 `useCachedMmap`/`syncValid`，在各 `useCachedMmap > 1` 跳读点（`ConvInt8TiledExecutor` 等）打印命中。判据：hint 在**运行中途**从 1 变 2 = 自我污染坐实；跳读发生在首次真实推理 resize（而非装载期）= 重建的 execution 在拿野内存。
5. **修复方向**：判定移入分配器首次创建分支（`MetalBackend.mm:onCreate` 是正确参考实现）；更彻底的加固是跳读前校验 buffer 确实来自 mmap 池。

## 8.3 常见对照表：症状 → 优先怀疑

| 症状 | 最可能的原因 |
|------|-------------|
| mmap 开着才乱码，冷启动+干净目录也乱 | 本次运行自我污染（sync 标记中途被自己看见） |
| 冷启动好、同目录第二次坏 | warm 复用路径的分配顺序/布局不匹配 |
| 换模型不换目录后乱码 | 缓存文件名无模型标识，跨模型污染 |
| App 错、demo 对 | App 双加载模式触发 + demo 单加载不触发 |
| 只有某类 op（如 geometry 分解的 fuse op）坏 | 该 op 的 execution 在封池后重建，跳读拿到 RAW 内存 |

## 8.4 参考案例：fuse 模型 iOS CPU 全乱码（useCachedMmap 自我污染，2026-08-06）

**症状**：4 个 FusedLinear 导出模型在 iPad/iPhone 上 CPU 后端（`use_mmap=true`）全部输出单字符刷屏（`!!!`/`\n`），偶发 SIGSEGV；同设备 Metal 正常；Mac `llm_demo` 单跑正常；非 fuse 模型任何配置都正常。

**排查路径**（两条红鲱鱼 + 一次真命中）：
1. Mac 上 `use_mmap=true` "复现"乱码 → 实为 `llm_demo` 强制 `tmp_path:"tmp"` + 缓存无模型标识，吃了之前另一个模型的缓存（**红鲱鱼一：跨模型污染**）。教训：对拍前 `rm -rf tmp`。
2. "非 fuse 模型也坏" → bisect 全 GOOD 才发现主 build 目录增量编译产物陈旧（**红鲱鱼二**）。教训：怀疑"分支回归"先开全新 build 目录验证，别信老增量目录。
3. 干净构建 + 干净缓存后锁定复现矩阵：**fuse × use_mmap × CPU × iOS**；给 App 加 `nommap` 判别开关 → mmap=false 立好。
4. 关键洞察：iOS App 是**同进程双加载**（启动预载 + benchfiles 重载并清 tmp 目录）。按此写 double-load 最小复现器 → **Mac 上完整复现**，真机问题降维成 host 调试。
5. 埋点两处：`useCachedMmap` 每次 resize `+= syncValid` 自增（单载 trace 1→2→3→4→5）；140 个 FusedLinear 分解出的成员 conv 在**首次真实推理 resize** 时重建并全部跳读权重。
6. 根因链闭合：首次 `onClearBuffer` 写出 sync.static 并封池 → 下一次 resize 重查看见**自己刚写的标记** → hint 1→2 → resize 重建的 conv 跳读 + STATIC buffer 来自 RAW malloc → 权重=未初始化内存。非 fuse conv 装载期创建一次且跨 resize 复用，永远踩不到；Metal 不分解 FusedLinear 且判定本来就只做一次，双重幸免。

**修复**（`4c50f4b12`）：CPU 的 sync 检查移入 `mStaticAllocatorMMap == nullptr` 创建分支，对齐 Metal。验证：Mac double-load 修复、warm 二进程正常、iPhone 13 Pro 真机 0.6b/0.8b/2b CPU 全部恢复。

**避坑要点**：
- "真机错 host 对"先对齐**配置与加载模式**（mmap 开关、双加载），不要先怀疑硬件/SIMD/平台；
- 复现器要**复刻加载模式**而不只是配置——单载复现不出双载 bug；
- `MNN_ASSERT` 在 release 构建是空操作，mmap 分配器里的断言不会救你；
- 直跑被 SIGKILL（rc=137）而 lldb 下正常时，先在 lldb 里拿结果，别死磕信号来源；
- 已知遗留：同进程重载且**不清缓存目录**（陈旧 sync + 旧权重文件）仍会误信；`llm_demo` 共享 `tmp/` 无模型标识。见 8.1 前提 2。

## 8.5 相关文件索引

| 文件 | 作用 |
|------|------|
| `source/backend/cpu/CPUBackend.cpp` | CPURuntime::onCreate 的 weightMemoryPath/sync 判定（本案例修复处） |
| `source/backend/metal/MetalBackend.mm` | Metal 的一次性判定正确参考（`onCreate` :1984 附近） |
| `source/core/BufferAllocator.cpp` | `MmapAllocator`：缓存文件命名（`prefix + allocTimes`）、sync() 写标记、autoRemove 语义 |
| `source/backend/cpu/compute/ConvInt8TiledExecutor.cpp` | Q4 conv 的 `useCachedMmap > 1` 跳读点 |
| `source/core/ConvolutionCommon.cpp` / `source/backend/cpu/CPULayerNorm.cpp` | 其余跳读点 |
| `transformers/llm/engine/demo/llm_demo.cpp` | 强制 `tmp_path:"tmp"` 的共享缓存陷阱（:274） |
