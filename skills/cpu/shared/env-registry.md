# CPU 环境变量与运行时开关注册表

> **何时读**：要用某个开关做 A/B、路径覆盖或性能归因之前；在 case 文档里读到一个开关名、
> 准备去 shell 里设它之前；以及给 CPU 侧**新增**任何调试开关之前。
> **本表的核心价值是区分四种机制**：运行时 env / 编译宏 / backend flag / `constexpr`。
> 它们名字都长得像 `MNN_XXX`，但设置方式完全不同，混淆会直接产出错误的技术结论（见 §二、§四）。
> 构建选项（`option()` 声明的 CMake 变量）不在本表，见 [`build-test-and-benchmark.md`](build-test-and-benchmark.md)，
> 唯一例外是 `MNN_PIPELINE_PROFILE`（它**没有** `option()` 声明，见下表）。
> **归属**：`skills/cpu/shared/` 下的**共享工具**文档，[`optimize/`](../optimize/SKILL.md)（为什么慢）与 [`kernel/`](../kernel/SKILL.md)（怎么写）两个分支共用，改动前请照顾两侧读者。

## 一、总表

| 名称 | 类型 | 位置 | 默认 | 语义 | 状态 |
|---|---|---|---|---|---|
| `MNN_CPU_TARGET` | env | ARM：`cpu/compute/CommonOptFunction.cpp`；x86_64：`cpu/x86_x64/FunctionDispatcher.cpp` | unset = 不降档 | ISA 能力位降档。ARM clamp 到 0..3（`≥1` 放行 fp16+sdot、`≥2` i8mm、`≥3` sme2，关 sme2 时同时把 `smeCoreNumber` 置 0）；x86_64 clamp 到 0..4（`≥1` AVX2、`≥2` FMA3、`≥3` AVX512、`≥4` AVX512VNNI）。**只屏蔽运行时能力位，屏蔽不掉"没编进来"**。用法见 [`optimize/arch/arm.md`](../optimize/arch/arm.md) §三 / [`optimize/arch/x86_64.md`](../optimize/arch/x86_64.md) §三 | 在用；**仅在 `-DMNN_PIPELINE_PROFILE=ON` 构建下存在** |
| `MNN_PIPELINE_PROFILE` | 编译宏（**无 `option()` 声明**） | 消费方 `cpu/CMakeLists.txt`（只 `target_compile_options` 给 `MNNCPU`）；使用点 `CommonOptFunction.cpp`、`FunctionDispatcher.cpp` | 默认构建里**不存在** | CPU 侧唯一作用是放行 `MNN_CPU_TARGET` 的读取与打印，不引入任何计时开销，可放心常开跑路径覆盖 | 在用；必须手动 `cmake -DMNN_PIPELINE_PROFILE=ON`（根 `CMakeLists.txt` 里搜不到它，因为没人声明过） |
| `MNN_GEMVBW_M` / `MNN_GEMVBW_K` | env | `test/speed/GemvBWTest.cpp` | M=4096 / K=14336（Llama-3-8B FFN，对齐 llama.cpp `gemv_roofline.cpp`） | 覆盖 `speed/GemvBW` 的 GEMV 形状。GEMV 效率强 shape 相关，测目标模型必须改形状。**`atoi(e) > 0` 才生效**，写 `0`/非数字会被静默忽略而不是报错 | 在用（`142f294b0c`） |
| `MNN_TEST_SKIP` | env | `test/MNNTestSuite.cpp` | unset = 不跳过 | 逗号分隔的**精确**（非前缀）测试名黑名单，供 `test.sh` 绕过设备特有的上游 bug 而不牺牲其余覆盖。解析一次后进 function-static | 在用（`96e98be64d`） |
| `MNN_LLM_CONTENT_RESIZE_ALWAYS` | env | `express/module/StaticModule.cpp` | unset = 走 content 缓存 | `=1`（**严格判 `e[0] == '1'`**）恢复"每次 forward 全量 re-resize"的旧行为。做 decode 每-token resize 开销归因时的对照组 | 在用（`1792d0782a`） |
| `MNN_LLM_BENCH_PROFILE_NAME` | env | `transformers/llm/engine/tools/llm_bench.cpp` | unset = 只按 op type 汇总 | **存在即生效**（不看值）：额外调 `printTimeByName(1)`，按 op 名而非 op 类型汇总。仅当 `llm_bench` 已开 profile 时才会读到 | 在用 |
| `MNN_ASSEMBLER` | 构建期 env（CMake `$ENV{}` 读） | `cpu/x86_x64/CMakeLists.txt` | 未设 | MSVC + 64 位下指向外部汇编器，置上才开 `WIN_USE_ASM`；未设则 `.S` 被跳过（文件头注释：*may cause low performance*）且整个 `avx512/` 不编译 | 在用；**只在 configure 时读**，运行时 export 它毫无作用 |
| `MNN_CPU_USE_DEFAULT_BACKEND` | backend flag（`#define ... 4`） | 定义与消费都在 `cpu/CPUBackend.cpp`（`flags = config->flags` 及其分支） | 未设 | `BackendConfig::flags == 4` 时直接 `new CPUBackend(MNN_FORWARD_CPU)` 并 `break`，位置在 `AVX2Backend::isValid()` **之前** → x86_64 上静默绕过整条 AVX2/AVX512；ARM 上不影响 fp16（fp16 分支更早）。后果见 [`optimize/arch/x86_64.md`](../optimize/arch/x86_64.md) §4.2 | 在用；**不是 env**，只能由调用方在 `BackendConfig` 里置，shell 改不了 |
| `kThreadPoolSpinBudget` | constexpr | `cpu/ThreadPool.cpp` | `512`（`uint32_t`） | ARM64 barrier 等待中 `isb sy` 的有界退避次数，耗尽后**锁存**（该次等待余下全部 `yield`，不清零）。`yield` 在 Darwin 是 `swtch_pri` 系统调用，逐 spin 调用会主导每-op barrier 开销 | 已定型（`142f294b0c`）；扫参用的临时 env `MNN_SPIN` 已随实验丢弃，今天设它无效 |
| `kWorkerIdleTimeout` | constexpr | `cpu/ThreadPool.cpp` | `std::chrono::milliseconds(8)` | worker 在池 active 时允许空转的**时间**预算（不是次数），超时后挂 condvar 睡眠（`mSleepMask` + 持锁 notify 防漏唤）。取 8ms 是因为要高于 decode 的亚毫秒 token 间隔，让 decode worker 永不睡 | 已定型（`502dc4511b`）；扫参用的临时 env `MNN_TP_IDLE_MS` 已随实验丢弃，今天设它无效 |
| `MNN_THREAD_POOL_MAX_TASKS` | 编译宏（文件内 `#define`） | `cpu/ThreadPool.cpp` | `2` | 线程池并发 task slot 数，决定 `mTasks` / `mTaskAvailable` 尺寸 | 在用；纯常量，既无 env 也无 CMake 入口，改它必须改代码 |
| `MNN_OPENCL_FUSED_PROJ_DISABLE` | env | `source/core/FusedProjCommon.hpp` | unset = OpenCL 接管 fused proj | 置**任意值**（判 `!= nullptr`）→ OpenCL 全面拒绝该 op，回落 geometry 分解。**位于 `source/core/` 但与 CPU 无关** —— 登记在此仅为消除"core 里的 env 一定影响 CPU"这一误判 | 在用（非 CPU 路径） |

## 二、四种机制不要混为一谈

CPU 侧全库只有 **两处** 运行时 `getenv`（`CommonOptFunction.cpp`、`FunctionDispatcher.cpp`），且读的是同一个变量。
其余名字像开关的东西都不是 env：

| 机制 | 怎么改 | 何时生效 | 典型 |
|---|---|---|---|
| 运行时 env | `export` / 命令行前缀 | 进程启动后首次读取（多为 function-static，只读一次） | `MNN_CPU_TARGET` |
| 编译宏 | `cmake -D...` 传下去的 `target_compile_options` | 构建期，改了要重编 | `MNN_PIPELINE_PROFILE`、`MNN_THREAD_POOL_MAX_TASKS` |
| backend flag | 调用方填 `BackendConfig::flags` | 每次 `onCreate` | `MNN_CPU_USE_DEFAULT_BACKEND` |
| `constexpr` | 改源码 | 构建期，无外部入口 | `kThreadPoolSpinBudget`、`kWorkerIdleTimeout` |

还有一类是**运行时探测值**，看着像可调参数但没有覆盖入口：`computeThreadNumber()`
（`cpu/CPUBackend.cpp`）在 `workItems > 1` 时把并发度 cap 到
`mCoreFunctions->perfCoreNumber`（来自 `sysctl hw.perflevel0.physicalcpu`），
`workItems == 1`（decode，带宽 bound）保留全部线程。想在异构核上做"限核 / 不限核"A/B，
只能改代码重编，**不存在对应 env**（`502dc4511b`）。

## 三、命名与生命周期规范

**命名（in-tree 观察到的约定）**

- 前缀带后端名：`MNN_CPU_*`（CPU 派发）、`MNN_METAL_*`、`MNN_VK_*`、`MNN_QNN_*`、`MNN_HEXAGON_*`；
  跨后端的运行时行为用 `MNN_LLM_*`；测试工具用 `MNN_TEST_*` / `MNN_<TESTNAME>_*`。
  **不要新增无后端前缀的 `MNN_XXX`**（历史上 `MNN_DISABLE_GATE_UP_FUSION` 就因此被改名补前缀）。
- 极性：默认关的写 `*_ENABLE_*`，默认开的写 `*_DISABLE_*`；多值模式用具名字符串或整数档位（如 `MNN_CPU_TARGET` 的 0..4）。
- 值判定要写清并在本表登记：`!= nullptr`（存在即生效）/ `e[0] == '1'` / `atoi > 0` 三种在树里都有，
 不一致本身就是坑（见 §四）。
- 集中声明优于散落 `getenv`：`source/backend/metal/MetalEnv.hpp` 是树里的参考实现
  （单一 struct 声明 + 解析一次 + 明文要求同步更新 registry）。CPU 侧目前只有两处 `getenv`，
  尚不值得建同样的注册中心；**一旦超过三处就应照 Metal 的做法收拢**。

**生命周期（硬规则）**

1. **临时调试开关用完即删。** 定型结论应落成 `constexpr` / 探测值 / 代码分支，不要把扫参用的 env 留在主干
   —— 留着就会有人以为它是受支持的调参入口。
2. **删除时同步把本表里的行删掉或标注"已移除"**，并写清它服务的实验、定型产物与产物提交。
3. **case 文档引用历史开关时必须在正文注明"已移除"。** case 里的表头保留原始 env 名（那是实验记录的一部分，改了就失真），
 这样 case 可读但**不可执行**，不会诱导 Agent 去设一个死开关。
4. 新增仍在用的开关：同时改代码、改本表、并在对应 ISA 文档（`arm.md` / `x86_64.md`）的自证章节说明用法。

## 四、常见误判

| 现象 | 真实原因 |
|---|---|
| 设了 `MNN_CPU_TARGET` 但**没有任何打印** | `MNN_PIPELINE_PROFILE` 没开。注意此时 **`getenv` 整段都被 `#ifdef` 编译掉了，降档也没发生** —— 你跑的是满档。要区分"没打印但降档了"和"什么都没发生"：本仓是后者 |
| `MNN_CPU_TARGET` 降了档，性能却没变 | 该 ISA 本来就没编进来（x86_64 上 `MNN_AVX512` 默认 OFF），或这个 op 根本不走那条 kernel。降档只屏蔽运行时能力位，见 [`optimize/arch/x86_64.md`](../optimize/arch/x86_64.md) §3.1 |
| 在 case 文档里看到 `MNN_DBG_CAP_THREADS` / `MNN_TP_IDLE_MS` / `MNN_SPIN` 并去设置 | 三者**均已移除且从未进入过提交的源码**，设了不会有任何效果。别把"没反应"当成技术结论；结论已落成 `ThreadPool.cpp` 的 `constexpr` 与 `computeThreadNumber`，要复现实验得自己改代码 |
| 在旧 case 文档里看到 `llm_bench -scn` / `--sme-core-num` 并照着用 | 该 flag 已随 `CPU_SME_CORES` hint 链在 `096230039b` 删除，现在不存在。旧 case 里的 `-scn` 行是**无效测量**的记录（该参数当时也没有消费者），不是调参结论 |
| 想用 hint 覆盖 SME 核数 / 关掉 SME 派发 | 无此入口。核数来自芯片名查表的硬件探测（`CPURuntime.cpp`）；要整条关 SME2 只有两条路：构建期 `-DMNN_SME2=OFF`，或运行期 `MNN_CPU_TARGET` 降到 `≤2`（需 `MNN_PIPELINE_PROFILE=ON`，它会同时把 `smeCoreNumber` 置 0） |
| 想调线程池自旋 / 空闲超时，找不到 env | 已定型为 `constexpr`（`ThreadPool.cpp`）。改它要动源码重编，且四条正确性要点缺一不可，见 [`optimize/runtime-and-scheduling.md`](../optimize/runtime-and-scheduling.md) §1.4 |
| 想在异构核上 A/B "限核 / 不限核" | 无 env。cap 逻辑在 `CPUBackend.cpp`，只能改代码 |
| x86_64 上 pack 掉回 4、一半算子还在 AVX | `MNN_CPU_USE_DEFAULT_BACKEND` 被置进 `BackendConfig::flags` 了。这**不是** env，查调用方而不是查环境 |
| `export MNN_ASSEMBLER=...` 后 AVX512 还是没编 | 它是 configure 时读的，必须重新 `cmake`（且仅 MSVC + 64 位有意义） |
| `MNN_GEMVBW_M=0` 想"用默认" | `atoi(e) > 0` 才赋值，写 0 与不写等价 —— 恰好等价，但**别依赖**这种巧合；不想覆盖就 `unset` |
| 在 `source/core/` 里搜到 `MNN_OPENCL_FUSED_PROJ_DISABLE`，以为影响 CPU | 名字里的 `OPENCL` 是准确的，只让 OpenCL 拒收该 op。CPU 无关 |
