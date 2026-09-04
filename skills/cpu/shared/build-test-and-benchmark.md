# 构建、测试与跑分纪律（ARM + x86_64 通用）

> **何时读**：准备跑正确性门禁、做 A/B 跑分、或要把一个性能结论写进 commit 之前。
> 本文只管**怎么跑、怎么判断跑成了**；每条 ISA 的降档命令与能力位自证在
> [`optimize/arch/arm.md`](../optimize/arch/arm.md) §三/§五 与 [`optimize/arch/x86_64.md`](../optimize/arch/x86_64.md) §三/§五，不在这里重复。
> 环境变量完整清单见 [`env-registry.md`](env-registry.md)。
> **归属**：`skills/cpu/shared/` 下的**共享工具**文档，[`optimize/`](../optimize/SKILL.md)（为什么慢）与 [`kernel/`](../kernel/SKILL.md)（怎么写）两个分支共用，改动前请照顾两侧读者。

## 一、构建命令与常用开关

```bash
mkdir -p build && cd build
cmake .. -DMNN_BUILD_TEST=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
         -DMNN_BUILD_LLM=ON -DMNN_PIPELINE_PROFILE=ON
make -j8 > make.log 2>&1; echo make_exit=$?     # 退出码必须显式打印，见 §六.1
```

| 开关 | 声明处 | 默认 | 关掉之后会怎样（都是静默的） |
|---|---|---|---|
| `MNN_BUILD_TEST` | `CMakeLists.txt` | OFF | 没有 `run_test.out`（`test/CMakeLists.txt` 全 glob 编译，单个用例文件不可单独开关） |
| `MNN_LOW_MEMORY` | `CMakeLists.txt` | OFF | 低 bit 权重 kernel 整族不编译；且 `ConvolutionFloatFactory.cpp` 走 `DenseConvolutionTiledExecutor`（反量化 float）而非 `DenseConvInt8TiledExecutor` → `op/lowMemory/*` 照样跑、照样过，**测的不是你改的 kernel** |
| `MNN_SUPPORT_TRANSFORMER_FUSE` | `CMakeLists.txt` | OFF | `test/op/AttentionTest.cpp`、`test/op/LinearAttentionTest.cpp`、`test/speed/LinearAttentionSpeed.cpp` 整文件不编译 → attention 全家变成"匹配 0 用例" |
| `MNN_BUILD_LLM` | `CMakeLists.txt` | OFF | 没有 `llm_demo` / `llm_bench`（`transformers/llm/engine/CMakeLists.txt` + `tools/CMakeLists.txt`，受 `MNN_LLM_BUILD_DEMO` 默认 ON 控制）。★ 开它会 **FORCE** `MNN_LOW_MEMORY=ON` + `MNN_SUPPORT_TRANSFORMER_FUSE=ON`（`CMakeLists.txt`）——所以"带 LLM 的构建"和"纯 op 构建"编出来的测试集合不同 |
| `MNN_ARM82` | `CMakeLists.txt` | ON | aarch64/armv7 上加 `-DENABLE_ARMV82`（`cpu/CMakeLists.txt`）→ `core/Macro.h` 定义 `MNN_USE_ARMV82`；关掉就没有 fp16 第二张表，`precision=2` 退化为 fp32 |
| `MNN_SME2` | `CMakeLists.txt` | ON | aarch64 分支加 `-DMNN_SME2` 并把 `arm64/sme2_asm/*.S` 纳入编译（`cpu/arm/CMakeLists.txt`） |
| `MNN_AVX512` | `CMakeLists.txt` | **OFF** | 默认构建在 AVX512 机器上测到的是 AVX2；关联连锁见 [`optimize/arch/x86_64.md`](../optimize/arch/x86_64.md) §4.3 |
| `MNN_PIPELINE_PROFILE` | **全仓无 `option()` 声明** | 不存在 | 唯一消费点 `source/backend/cpu/CMakeLists.txt`（`target_compile_options(MNNCPU PRIVATE ...)`）。命令行 `-DMNN_PIPELINE_PROFILE=ON` 仍然生效（`if()` 直接读 cache 变量），但你在 `cmake -LH` 里找不到它 |

★ **`MNN_PIPELINE_PROFILE` 不只 gate 打印，它 gate 整个 `MNN_CPU_TARGET` 机制。**
ARM 侧 `compute/CommonOptFunction.cpp`：`#ifdef` 包在 `if (getenv("MNN_CPU_TARGET"))` **之外**，
连能力位屏蔽一起包掉。x86_64 侧 `x86_x64/FunctionDispatcher.cpp`：宏未定义时
`_MNNApplyCpuTarget()` 直接 `return cpuFlags`，一个位都不屏蔽。
所以默认构建里 `MNN_CPU_TARGET` 是**彻底的空操作**，不是"降了档只是没打印"。
判据：**看不到 `effective ARM features:` / `effective x86 features:` 那一行，就说明这次没降档**，
不要据此得出"降档对性能没影响"。（档位上限也不同：ARM clamp 到 3，x86_64 clamp 到 4。）

## 二、`run_test.out` 使用规则（规则先于名单）

argv 全是**位置**参数，无名字、错位不报错（`test/main.cpp`）：

| 位 | 含义 | 默认 | 备注 |
|---|---|---|---|
| `argv[1]` | 测试名前缀，或 `all` | 缺省 → `runAll` | — |
| `argv[2]` | backend（`MNNForwardType`） | 0 = CPU | — |
| `argv[3]` | precision | 1 = High | 0 Normal / 1 High / 2 Low / 3 Low_BF16（`MNNForwardType.h`）；>3 被强制成 0 |
| `argv[4]` | thread | 1 | ★ 同时写入 `pStaus.thread`，但表达式是 `argc > 4 ? thread : 0` |
| `argv[5]` | flag / tag | `""` | 只拼进测试报告字符串（`MNNTestSuite.cpp`），**但它是占位符**：要设 memory 就必须补上 |
| `argv[6]` | memory | 0 = Normal | `Memory_Low = 2`（`MNNForwardType.h`）。★ 低 bit int8 executor 的真正开关 |
| `argv[7]` | dynamicOption | 0 | → `hint.dynamicQuantOption` |
| `argv[8]` | enableKleidiAI | false | — |
| `argv[9]` | divisionRatio | 1 | — |

1. **名字是前缀匹配**：`test->name.find(prefix) == 0`（`MNNTestSuite.cpp`）。`op/lowMemory` 一次跑全 5 个。
2. **argv[1] 不支持逗号分隔多 key**。逗号只用于 `MNN_TEST_SKIP` 环境变量里的**精确名**跳过表
   （`MNNTestSuite.cpp` 里解析并生效）。要跑多个前缀就跑多次。
3. **`runAll` 会跳过名字含 `speed` 或 `model` 的用例**，并在每个用例后 `gc(FULL)`。
   所以 `speed/*` 永远必须显式点名。
4. ★ **判有效性看 `passed` 数，不看退出码。** `run()` 返回 `wrongs.size()`。名字不匹配时
   `runUnit = 0`、`wrongs` 为空 → 打印 `√√√ all <key> tests passed.` 加
   `{"blocked":0,"failed":0,"passed":0,"skipped":0}`，**退出码 0**。`"passed":0` 的含义是"你什么都没测"。
5. `run(precision)` 只拿到 `argv[3]`；线程数不经参数传递，用例要自己读 `MNNTestSuite::get()->pStaus`
   （当前只有 `speed/GemvBW` 这么做，见 §三）。

## 三、真实测试名注册表

逐字核对自 `MNNTestSuiteRegister(...)`，标注源文件便于复核。「构建门」列空 = **无任何 `#ifdef`**。

| 测试名 | 源文件 | 构建门 |
|---|---|---|
| `op/lowMemory/DenseConv` | `test/speed/HybridConvSpeedTest.cpp` | — |
| `op/lowMemory/HybridConv` | `test/speed/HybridConvSpeedTest.cpp` | — |
| `op/lowMemory/blockConv` | `test/speed/HybridConvSpeedTest.cpp` | — |
| `op/lowMemory/mixedKernel` | `test/speed/HybridConvSpeedTest.cpp` | — （⚠ Vulkan 后端上会挂死，见 `test_stages.json`） |
| `op/lowMemory/lowBitScale` | `test/speed/HybridConvSpeedTest.cpp` | — （w2/w3 + block64，贴 LLM 量化尺寸） |
| `speed/HybridConv` | `test/speed/HybridConvSpeedTest.cpp` | — |
| `op/int4Ptq` | `test/speed/HybridConvSpeedTest.cpp` | **`MNN_LOW_MEMORY`** |
| `op/attention` / `op/attention_nocache_mask` / `speed/attention_threads` / `op/attention_kvblock` / `op/attention_c4` / `op/attention_c4_tail` / `speed/attention` | `test/op/AttentionTest.cpp` | **`MNN_SUPPORT_TRANSFORMER_FUSE`** |
| `op/linear_attention` `_c4_tail` `_decode` `_rollback` `_chunked_layer_index` `_pending_write_unsynced` `_gate_fold` | `test/op/LinearAttentionTest.cpp` | **`MNN_SUPPORT_TRANSFORMER_FUSE`** |
| `speed/LinearAttentionSpeed` | `test/speed/LinearAttentionSpeed.cpp` | **`MNN_SUPPORT_TRANSFORMER_FUSE`** |
| `speed/GemvBW` | `test/speed/GemvBWTest.cpp` | — （低 bit GEMV roofline，★ 见下） |
| `speed/GemmSpeedFloat` / `Int8` / `Int4` / `All` | `test/speed/GemmSpeed.cpp` | — |
| `speed/MatMulTest` / `MatMulBatchTest` / `MatMulBConstTest` | `test/speed/MatMulSpeed.cpp` | — |
| `op/matmul` / `op/matmulBConst` | `test/op/MatMulTest.cpp` | — |
| `op/ConvInt8/im2col_gemm` / `winograd` / `depthwise` | `test/op/ConvInt8Test.cpp` | — |
| `op/ConvInt8/im2col_spmm` | `test/op/ConvInt8Test.cpp` | `__arm__ \|\| __aarch64__` |
| `speed/ConvInt8/winograd` / `depthwise` | `test/op/ConvInt8Test.cpp` | — |
| `speed/ConvInt8/im2col_gemm` / `multi_instance` | `test/speed/ConvSpeedInt8Test.cpp` | — |
| `op/convolution/conv2d` / `weighti8i4conv2d` / `sparse_conv2d` / `depthwise_conv` / `conv_group` | `test/op/ConvolutionTest.cpp` | — |
| `speed/convolution/conv2d` | `test/op/ConvolutionTest.cpp` | — |
| `kleidiai/int4_conv_e2e` | `test/kleidiai/conv_int4.cpp` | `MNN_KLEIDIAI_ENABLED` + `MNN_LOW_MEMORY` |
| `imatmul/lhs` | `test/kleidiai/imatmul.cpp` | `MNN_KLEIDIAI_ENABLED` |

★ **纠正旧结论**：`op/lowMemory/blockConv` 与 `op/lowMemory/HybridConv` **确实已注册**，且
`HybridConvSpeedTest.cpp` 里唯一的条件编译块只包住 `op/int4Ptq`，两者**没有任何 `#ifdef` 保护**。
旧 skill 与旧 case 文档中"这些名字在注册表里不存在 / 匹配 0 个用例"的警告是**错的**，不要再据此改名。
真正会让人白跑一轮的是另两件事：①没开 `MNN_SUPPORT_TRANSFORMER_FUSE` 时 attention 全家才是真的 0 用例；
②没给 `memory=2` 时 `op/lowMemory/*` 根本不进低 bit int8 executor（§一）。

★ **`speed/GemvBW` 的两个坑**（`test/speed/GemvBWTest.cpp`）：
- 线程数取 `pStaus.thread > 0 ? pStaus.thread : 4`，而 `pStaus.thread` 只在 `argc > 4` 时才被赋值
  （`main.cpp`）。所以 `./run_test.out speed/GemvBW 0 2` 实测的是 **4 线程**，不是 1。必须显式写第 4 位。
- 它自建 executor 并写死 `Memory_Low`，忽略 `argv[6]`；M/K 默认 4096×14336，
  用 `MNN_GEMVBW_M` / `MNN_GEMVBW_K` 覆盖。输出 `us/iter`、`W MiB`、`bytes/elem`、
  `eff GB/s`、`%peak`、`GFLOPS`、`AI`，w2/w3 的 `eff GB/s` 偏低先查 unpack 指令/字节比。

## 四、`llm_demo` / `llm_bench`

**`llm_demo` 没有任何 flag**，全位置参数（`transformers/llm/engine/demo/llm_demo.cpp`）：

```
./llm_demo <config.json> [prompt.txt] [max_token_number] [<任意第 5 个参数>]
```

- 只给 config → 进交互 chat；`argv[3]` = max_token_number。
- `argv[4]` **存在即生效**（值被忽略）→ 关 thinking，仅 Qwen3 有效。
  旧文档里的 `... prompt.txt 64 1` 就是"最多 64 token + 关 thinking"。
- 按 config 绝对路径 hash 建 `tmp_<hash>` 权重 mmap 缓存目录，换模型不会串旧权重缓存。
- 定位：**内容 sanity 手段，不是跑分手段。**

`llm_bench` 的权威 usage 与 flag 解析都在 `transformers/llm/engine/tools/llm_bench.cpp`：

| flag | 长名 | 默认 | 语义 |
|---|---|---|---|
| `-m` | `--model` | `./Qwen2.5-1.5B-Instruct` | 逗号可给多个 |
| `-a` | `--backends` | cpu | — |
| `-c` | `--precision` | 2 | 0/1/2；CPU 上 Normal 等价 High |
| `-t` | `--threads` | 4 | — |
| `-p` | `--n-prompt` | 512 | **prefill-only**，标签 `ppN`，不复用 KV |
| `-n` | `--n-gen` | 128 | **decode-only**，从 1 token 上下文起，标签 `tgN` |
| `-pg` | — | `0,0` | prefill pp 个 token 后**复用同一 KV cache** 生成 tg 个，prefill/decode 分开报，标签 `ppA+tgB` |
| `-rep` | `--n-repeat` | 5 | 实际跑 `nRepeat + 1` 轮，**第 1 轮当 warmup 丢弃** |
| `-kv` | `--kv-cache` | false | 已废弃：`-p A -n B -kv true` == `-pg A,B` |
| `-qa` | `--quant-attention`（亦 `-qatten`） | 0 | KV 量化模式 0 none / 1 QK-int8 / 2 QKV-int8 / 3 QK-TQ3 / 4 QKV-TQ3 / 5 QK-TQ4 / 6 QKV-TQ4 |
| `-fa` | `--flash-attention` | 1 | — |
| `-mr` | `--mixedSme2NeonRatio` | 41 | SME2/NEON 混合划分比，可试 41/49/33 |
| `-dyo` | `--dynamicOption` | 0 | 8 = 以内存换 decode 速度 |
| `-mmp` `-fp` `-j` `-load` `--profile` | `--mmap` / `--file-print` / `--json` / `--loading-time` / — | 0 / stdout / `llm_bench.json` / true / off | — |
| `--memory` `--power` | — | 2 / 0 | usage 里**没列**但会解析 |

★ **`-scn` / `--sme-core-num` 已不存在**：随 `CPU_SME_CORES` hint 链在 `096230039b` 一并删除
（该 hint 写入后从未被读，真实 SME 核数来自芯片名查表的硬件探测）。旧 case 文档里的 `-scn` 扫参行
是**无效测量**的记录，不是调参结论。要整条关 SME2 只有构建期 `-DMNN_SME2=OFF` 或运行期
`MNN_CPU_TARGET≤2`，见 [`env-registry.md`](env-registry.md) §四。

★ **`llm_bench` 只验速不验内容**：prompt 是合成的重复 token id 16
（`std::vector<int> tokens(prompt_tokens, 16)`），生成文本无意义。
任何基于 `llm_bench` 的结论必须配一次**同线程档、同 precision** 的正确性门禁（§五）。

## 五、验证矩阵模板

不要再写"同一条命令重复两遍、第二遍加注释说在别的设备上复跑"。填下表，每格记**`passed` 数**（不是"过了"）：

| # | ISA / 降档 | precision | thread | memory | 命令 | passed | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | 基线（不设 `MNN_CPU_TARGET`） | 1 High | 1 | 2 | `./run_test.out op/lowMemory 0 1 1 '' 2 2` | | |
| 2 | 同上 | 1 High | 4 | 2 | `./run_test.out op/lowMemory 0 1 4 '' 2 2` | | |
| 3 | 同上 | 1 High | > P 核数 | 2 | `./run_test.out op/lowMemory 0 1 8 '' 2 2` | | 线程 cap / barrier 类 bug 只在这档暴露（`arm.md` §4.3/§4.6） |
| 4 | 同上 | 2 Low | 4 | 2 | `./run_test.out op/lowMemory 0 2 4 '' 2 2` | | ARM = fp16 第二张表；x86_64 上 Low 被写死、无区分（`x86_64.md` §4.1） |
| 5 | weight-dequant 对照 | 1 / 2 | 1 | 省略 | `./run_test.out op/lowMemory 0 1 1` | | 不给 memory 走的是反量化 float 路径，也要跑（对齐 `test_stages.json`） |
| 6 | 降档档位 | 按 §一 | 4 | 2 | 见 [`optimize/arch/arm.md`](../optimize/arch/arm.md) §五 / [`optimize/arch/x86_64.md`](../optimize/arch/x86_64.md) §五 | | 需 `-DMNN_PIPELINE_PROFILE=ON`，且必须先确认打印出 effective features |
| 7 | attention | 1 / 2 | 1 / 4 / >P | — | `./run_test.out op/attention 0 1 4` | | 需 `MNN_SUPPORT_TRANSFORMER_FUSE` |
| 8 | LLM 内容 sanity | 同 #9 | 同 #9 | — | `./llm_demo <config.json> prompt.txt 64 1` | — | prompt 要含长文本，覆盖 kv 跨 chunk 边界 |
| 9 | 跑分 | 同 #8 | 同 #8 | — | `./llm_bench -m <config.json> -t 4 -pg 2048,32 -rep 3` | — | 与 #8 配对才算成立（§四） |

- 每一维至少两格，不要用一格代表全部；线程档固定取 **1 / 4 / 超过 P 核数**三档。
- `memory=2` 与 `dynamicOption` 的既有分档参照 CI：`test_stages.json`（dyn：`memory=2`+`dynamicOption=2`）
  与 （wdeq：省略 memory）。两类语义不同，都要覆盖。
- ISA 降档的档位含义与自证方法不在本文，见各 ISA 文档 §三。

## 六、实验纪律（硬性六条）

1. **make 退出码必须自己看**。绝不 `make ... | tail -1 && bench`——pipe 的退出码是 `tail` 的，
   编译失败会拿旧二进制跑完整轮并据此分析。写 `make > make.log 2>&1; echo make_exit=$?`，非零立即停。
2. **热漂移用 inert 对照组分离**。持续 make+bench 会让 SoC 降频，同一配置在不同热态可差 30%+。
   每个 sweep 里带一个"本改动不可能影响"的配置（例如不触发新路径的线程档）；它若同比漂移就是环境不是回归。
   结论只用同热态或冷却后的数据。`llm_bench` 每组之间只 `sleep 5ms`（`llm_bench.cpp`），不足以散热。
3. **回归判定用同二进制交错 A/B**。给可疑路径加**临时** env 开关（实验后移除），在同一个二进制里交替跑开/关。
   跨会话、跨二进制的绝对值对比会被构建差异与热态污染。
4. **就地替换 `build/libMNN.dylib` 并每轮打印 md5**。`llm_bench` 的 `LC_RPATH` 是**绝对构建目录**：
   把 `llm_bench` + `libMNN.dylib` 拷到 `/tmp/xxx` 运行，dyld 仍从 `build/` 加载 libMNN
   → 两边其实都在跑 baseline，会得到"优化后反而更慢"的假结论。md5 是唯一的自证。
5. **跑分必须配对同线程档的正确性门禁**。`llm_bench` 只验速不验内容（§四）。
   只报速度、没有同档 `run_test.out` / `llm_demo` 门禁的结论，不予采信。
6. **zsh 不对未加引号的参数展开分词**（本机默认 shell 是 zsh，这条真实产出过一个完全错误的结论）。
   `for cfg in "0 1 1 tag 2"; do ./run_test.out op/lowMemory/x $cfg; done` 会把整串当成**一个** argv 传进去：
   `argv[2]` = `"0 1 1 tag 2"` → `atoi` 得 0，其后全部取默认值 → **`memory` 退回 0**，
   于是 `op/lowMemory/*` 走反量化 float 路径、**测不到低比特 kernel**，还会 `"passed":1` + 退出码 0 地"通过"。
   bash 会分词、zsh 不会，同一条命令在两个 shell 下语义不同。
   规则：循环里用**字面参数**或数组展开（`"${(z)cfg}"` / `set -- $=cfg`），并且每轮用输出里的
   `memory=` / `thread=` / `precision=` 行**自证参数真的传进去了**——这三行是 `test/main.cpp` 直接打印的。

## 七、结果记录规范

- **commit body 必须同时写「原理」和「性能提升」**：原理是"消除了什么开销"（访存次数、syscall、
  同步等待、冗余 pass……），一两句话让 review 者不读 diff 也能判断合理性；性能提升是 before→after 实测值，
  注明平台、模型/shape、线程数、precision。**持平也要写**（"decode 中性"是有价值的结论）。
  commit body 是唯一不会丢的载体。
- **数据不进本仓，方法进 skill**：分支级性能数据留在 commit body 与外部台账，
 skill 文档只写措施与方法论。
- 每条数据自带可复现坐标：commit、构建开关、ISA 降档档位、thread、precision、memory/dynamicOption、`-rep`。
- 全量回归与设备侧（Android / iOS）跑分走 [`skills/test-ci/SKILL.md`](../../test-ci/SKILL.md)
  （`test.sh` + `test_stages.json`）；本文只覆盖优化循环里的手动门禁。
