# Metal Shader 体积优化（iOS arm64）

## 概述

Metal 后端的 shader 源码以纯文本形式打进二进制，占用 `__TEXT,__cstring` 达 657KB。本次优化在构建期对 shader 文本做压缩、运行时首次使用时解压，在**不关闭任何线上能力、不改动任何调用点、不新增外部依赖**的前提下，arm64 产物 `__TEXT` 段减少 **409,600 字节**。

优化由 `MNN_METAL_PACK_SHADER` 控制，`MNN_REDUCE_SIZE=ON` 时默认开启。关闭时产物与优化前**逐字节相同**。

## 根因

MNN 的 Metal 后端**没有构建期 shader 编译**：所有 shader 都在运行时通过 `newLibraryWithSource:` 由驱动 JIT 编译（`MNNMetalContext.mm:57`、`:147`），因此 shader 文本必须完整打进二进制。以 C 字符串字面量形式存放时，这些文本全部落入 `__TEXT,__cstring`。

baseline 实测（arm64、Release、`-Oz`、`MNN_REDUCE_SIZE=ON`）：

| Section | 字节数 |
|---|---:|
| `__TEXT,__text` | 3,332,164 |
| **`__TEXT,__cstring`** | **657,597** |
| `__TEXT,__const` | 120,731 |

用 `size -m` 按 object 归因 `__cstring`，**605,432 字节（92%）来自 Metal shader 文本**：

| Object | `__cstring` |
|---|---:|
| `MetalConvolution1x1.mm.o` | 176,390 |
| `MetalAttention.mm.o` | 140,000 |
| `MetalLinearAttention.mm.o` | 89,509 |
| `AllShader.cpp.o` | 83,288 |
| `MetalLayerNorm.mm.o` | 24,852 |
| 其余 Metal object | ~91,000 |

关键结论：**按文件大小归因会得出错误判断**。MetalAttention、MetalLinearAttention、MetalLayerNorm 等文件的 object 体积大，主因是内嵌的 shader 字符串而非其代码。`MetalConvolution1x1.mm.o` 共 224,955 字节，其中 176,390 字节是字符串数据。

另有两个加剧因素：

- `ConvSimdGroupShader.hpp` 中的 shader 声明为 `static const char*`，而该头文件被 `MetalConvolution1x1.mm` 和 `MetalSharedGather.mm` 同时包含，每个 TU 各生成一份副本（174KB）。
- shader 文本高度冗余（大量重复的 kernel 模板、宏展开、参数结构体），压缩后仅约 17%。

## 方案

构建期把 shader 文本压缩成二进制 blob，运行时首次访问时解压并缓存。

### 1. 构建期打包（`packshader.py`）

- **Minify**：剥离注释与缩进。需要正确处理三类陷阱：
  - 字符串字面量中的 `//` 和 `/*` 不是注释（Metal 的 `[[host_name("...")]]` 属性内含字符串）；
  - 以 `\` 结尾的续行属于预处理宏体，物理换行必须保留，否则宏语义会被静默改变；
  - 预处理指令必须独占一行。
- **压缩**：面向字节的 LZ77，带 repeat-offset 槽位，最短路径 parse。不做熵编码，使解码器足够小且无需任何外部依赖。
- **生成**：输出压缩 blob 的 `.cpp`，以及把每个 shader 符号定义为访问器宏的 `.hpp`，因此**所有调用点无需改动**。
- **确定性**：相同输入必然产生逐字节相同的输出；内容未变时不改写文件，避免触发全量重编译。

覆盖 43 个 shader，508,383 字节压缩为 87,905 字节（17.3%）。

### 2. 运行时解码（`MetalShaderCodec.cpp`）

- 解码器约 150 行，`__text` 仅增加 4,216 字节。
- 拒绝所有畸形流（越界 offset、截断字面量、尾部冗余字节），不会越界读写。
- 匹配可与当前输出位置重叠，因此按字节正向复制以实现 run-length 语义。
- 惰性解码 + 进程级缓存，`std::mutex` 保护；缓存键为 blob 地址。使用函数内 static，**不引入全局动态初始化**（符合项目对 `__GLOBAL__sub_I_*` 的禁令）。

### 3. 兼容路径

五个 raw-literal shader 头文件与生成的 `AllShader.hpp/.cpp` 都把原始字面量放在 `#else` / `#ifndef` 分支中，因此**同一份内容绝不会以新旧两种形式同时进入二进制**。`makeshader.py` 重新生成时会复现同样的宏门控，避免后续重新生成丢失优化。

## 体积数据

baseline 与 candidate 使用两个全新独立构建目录、相同 AppleClang toolchain、相同 CMake 参数、相同 Release `-Oz` 配置，仅比较 arm64。

**完整链接后的 Mach-O**（`-force_load` 链入全部 MNN，最接近 App 集成口径）：

| 指标 | 优化前 | 优化后 | 差值 |
|---|---:|---:|---:|
| `__TEXT` 段 | 4,374,528 | 3,964,928 | **−409,600** |
| 文件字节数 | 5,772,456 | 5,363,624 | **−408,832** |
| `__cstring` | 615,349 | 106,917 | −508,432 |
| `__const` | 85,168 | 173,008 | +87,840 |
| `__text` | 3,623,320 | 3,624,820 | +1,500 |

**Framework 静态库**（未压缩，非 zip 包）：

| 指标 | 优化前 | 优化后 | 差值 |
|---|---:|---:|---:|
| 归档字节数 | 14,119,864 | 13,719,216 | **−400,648** |
| 各 section 合计 | 4,334,634 | 3,913,035 | **−421,599** |
| `__TEXT,__cstring` | 657,597 | 143,229 | −514,368 |
| `__TEXT,__const` | 120,731 | 208,636 | +87,905 |
| `__TEXT,__text` | 3,332,164 | 3,336,380 | +4,216 |

`llm_demo` 可执行文件同步从 5,656,200 减至 5,248,936 字节（−407,264）。

**单项收益拆解**：

| 项 | 收益 |
|---|---:|
| shader 文本移出 `__cstring` | −514,368 |
| 压缩 blob 计入 `__const` | +87,905 |
| 解码器代码计入 `__text` | +4,216 |
| **净收益** | **−421,599** |

关闭优化（`-DMNN_METAL_PACK_SHADER=OFF`）时，产物与优化前**逐字节相同（+0）**。

## 正确性验证

| 验证项 | 结果 |
|---|---|
| 生成脚本可独立运行 | 通过 |
| 可重复：两次运行 + 构建产物逐字节一致 | 通过 |
| 43 个 shader 解码结果与未压缩源码**逐字节相同** | 通过 |
| 编解码单测：畸形流、重叠匹配、repeat-offset | 通过 |
| 线程安全：16 线程 × 40 轮，TSan | 通过，无数据竞争 |
| 开关 ON / OFF 均可编译 | 通过 |
| `MNN_METAL=OFF` 构建不受影响 | 通过 |
| `makeshader.py` 重新生成保留宏门控 | 通过 |
| Metal 算子测试（真实 Apple GPU） | 250 通过 / 7 失败，**与 baseline 完全一致** |
| CPU 算子测试 | 257 / 257 通过 |
| expr 测试 | 50 通过 / 1 失败，与 baseline 一致 |
| model 测试 | 7 失败，与 baseline 一致 |
| 真实 LLM 在 Metal 上文本生成 | 正常，**greedy 输出逐字节相同** |
| Metal 错误标记（`Init Metal Error`、`Can't create executor`、pipeline 编译错误、`MTLLibraryErrorDomain`） | **0 次** |
| 确认测试确实走 Metal backend | 已确认：executor 创建成功，GPU 结果位级一致 |

15 个失败项全部为**既有问题**，已用未修改源码的 baseline 构建跑同一批测试对照确认。

验证过程中发现并修复了一个真实缺陷：minify 最初会丢掉源码末尾换行，导致被直接拼接的 shader（`std::string(gBasicConvPrefix) + gConv1x1WqSgReduce`）首尾行粘连，把后续的 `#if` 指令破坏，Metal 编译报错。已修正为始终以换行结尾，并复验所有拼接边界。

## 性能与内存影响

必须区分一次性开销与每次推理开销：

- **一次性**：解压全部 43 个 shader 共约 **0.8ms**；实际运行只会触达其中少数几个。相比 Metal 的 `newLibraryWithSource:` JIT 编译（量级大得多），该开销可忽略。
- **每次推理**：**零开销**。每个 shader 只解压一次，后续访问为一次哈希查找（43 次查找约 0.015ms）。解压发生在 pipeline 创建阶段，不在执行循环中。
- **实测 LLM 吞吐**：decode 181.6 / 180.9 / 185.6 vs 181.4 / 177.5 / 185.2 tok/s（ON vs OFF），持平。
- **常驻内存**：真实 3B LLM 推理峰值 RSS +65,536 字节（473.17MB vs 473.10MB，+0.014%）。解压文本刻意保留，因为调用方会长期持有该指针。
- **锁竞争**：仅一把互斥锁，只在每个 shader 首次解压时持有，稳定运行阶段无竞争路径。
- **未新增全局初始化器**（`__GLOBAL__sub_I` 数量不变）。
- **不影响 Metal pipeline cache**；`otool -L` 依赖列表**完全一致**，未新增系统库、第三方库或运行时链接依赖。

## 兼容性

线上必须保留的能力全部保留：Metal、ARM82 FP16、OpenCV、文本 LLM、`MNN_LOW_MEMORY`、`MNN_SUPPORT_TRANSFORMER_FUSE`。

- **未删除任何算子**。
- **未改动 Tokenizer**：`tokenizer.txt` 旧格式及全部 5 种 tokenizer 类型均未触碰。
- **未引入低 bit 量化或投机解码相关宏**，因此不存在 CPU / ARM82 / Metal / OpenCL / Vulkan 各后端行为不一致的风险。
- 业务集成构建命令保持不变；开关随既有的 `MNN_REDUCE_SIZE` 自动开启。
- 关闭优化后原有实现仍可正常编译运行，且产物逐字节相同。

## 风险与待验证项

- **平台代理**：验证机器仅有 Command Line Tools，**无 Xcode / iOS SDK**（`cmake/ios.toolchain.cmake` 无法 configure，缺少 `iPhoneOS.sdk`），因此以 **arm64 macOS** 配合相同 AppleClang、`-Oz`、Release 及完全相同的选项集进行测量。收益来自数据段删除，与平台无关，但 iOS 绝对数值建议在真实打包流水线上复核。
- 集成归因受 dead strip 与链接顺序影响；`__cstring` 原先部分可跨模块共享，最终 App 实际收益可能略有差异。
- 既有的 7 个算子 / 1 个 expr / 7 个 model 失败项未修复（不在本次范围，已用 baseline 对照确认）。
- 修改 shader 的同学需知晓文本会在构建期被 minify。minifier 已正确处理宏续行与字符串字面量，但引入非常特殊的构造时建议重新构建确认。

## 相关文件

| 文件 | 职责 |
|---|---|
| `source/backend/metal/packshader.py` | minifier + LZ77 编码器 + 代码生成 |
| `source/backend/metal/MetalShaderCodec.hpp/.cpp` | 解码器 + 线程安全惰性缓存 |
| `source/backend/metal/CMakeLists.txt` | 构建期代码生成接线；关闭时不编译解码器 |
| `CMakeLists.txt` | `MNN_METAL_PACK_SHADER` 选项，随 `MNN_REDUCE_SIZE` 默认开启 |
| `ConvSimdGroupShader.hpp`、`MetalAttentionShader.hpp`、`MetalLinearAttentionShader.hpp`、`LayerNormSimdGroupShader.hpp`、`MetalFlashAttnShader.hpp` | raw literal 置于宏门控之下 |
| `AllShader.hpp/.cpp` | 18 个生成 shader 的同类门控 |
| `makeshader.py` | 重新生成时复现宏门控 |
