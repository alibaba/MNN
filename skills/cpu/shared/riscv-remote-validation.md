# RISC-V 开发板远端验证（本地改、板上验）

> **何时读**：RISC-V 侧的改动要编译、跑正确性或跑性能之前。
> **本文只写 RISC-V 与本机 ARM/x86_64 不同的那部分**：交叉/远端构建矩阵、板端正确性分层、
> 板端性能实验的额外约束、对外报告的脱敏。
> **通用纪律不在本文重复**：make 退出码、热漂移 inert 对照、同二进制交错 A/B、md5 自证、
> `run_test.out` 位置参数与 zsh 不分词、结果记录规范 → [`build-test-and-benchmark.md`](build-test-and-benchmark.md) §二/§六/§七，那六条对 RISC-V 一字不改地成立。
> 路径矩阵与构建门自证 → [`../optimize/arch/riscv.md`](../optimize/arch/riscv.md)；
> kernel 写法 → [`../kernel/arch/riscv.md`](../kernel/arch/riscv.md)。
>
> **归属**：`skills/cpu/shared/` 下的共享工具文档，[`optimize/`](../optimize/SKILL.md) 与
> [`kernel/`](../kernel/SKILL.md) 两个分支共用。

## 一、分工与不可协商的前提

**本地负责阅读、修改、提交；目标板负责 ISA 编译、运行与性能验证。**

主机名、代理、用户名、远端仓库目录、build 目录、模型目录都是**会话参数**，
只在对话里传，**不写进 skill、不写进仓库文档、不进 commit**。本文一律用占位符。

**本机 ARM/x86_64 编译只能发现通用接口污染**（头文件、签名、`#ifdef` 漏洞、
`MNN_USE_RVV=OFF` 是否还编得过），**不能替代目标板结论**：ISA 可用性、运行时 VLEN、
核拓扑、TCM 可用性、持续带宽全部只在板上成立。

执行远端操作前四步（顺序不能换）：

1. 用用户已提供/已配置的 SSH 入口，不自己造连接方式；
2. 确认远端仓库与 build 目录**确实是**这次要用的那个；
3. **只读**检查远端分支、HEAD 与工作区：
   ```bash
   ssh <RISCV_HOST> 'uname -m'                                        # 必须是 riscv64
   ssh <RISCV_HOST> 'git -C <REMOTE_REPO> status --short --branch'
   ssh <RISCV_HOST> 'git -C <REMOTE_REPO> rev-parse HEAD'
   ```
4. 发现远端已有未提交修改就**保留**，改用隔离 worktree/clone 或明确列举的文件同步范围。

**禁止**：`git reset --hard`、`git clean`、覆盖式全目录同步（`rsync --delete` 之类）。
远端那份未提交改动可能是别人的在制品。同步代码优先可审计方式——已授权 push 时在远端拉明确提交；
未提交时只同步本次修改的**显式文件列表**；同步后用 hash 或 `git diff --check` 确认远端源码与候选版本一致。

## 二、构建矩阵：vendor ON / OFF 必须是两个目录

RISC-V 侧的选项都声明在 `source/backend/cpu/riscv/CMakeLists.txt`（除 `MNN_USE_RVV` 在根 `CMakeLists.txt`）：

| 选项 | 默认 | 关掉之后会怎样 |
|---|---|---|
| `MNN_USE_RVV` | OFF | 整个 `riscv/` 不参与；基表全部留标量实现 |
| `MNN_RVV_SPACEMIT_IME2` | OFF | 无 vendor lib；fast-path 注册回到 `rvv/MNNRvvFastPathRegistration.cpp` |
| `MNN_RVV_MARCH` | `rv64gcv` | 基线 ISA 串；`_xsmtvdotii` 会被 `string(REPLACE ...)` 剥掉只留给 vendor lib |
| `MNN_RVV_MCPU` / `MNN_RVV_MTUNE` | 空 | 仅调优，不改可用指令集 |
| `MNN_RVV_FAST_MATH` | OFF | 开启会给 `MNNRVV` 加 `-ffast-math`，**改变浮点语义**；数值对照时两侧必须一致 |
| `MNN_LOW_MEMORY` | OFF | vendor 侧 `MNNSpacemitIme2ConvInt8Executor.cpp` 不编译 → **低比特 conv 的 vendor 路径根本不存在**，别拿这种构建谈 W4 收益 |

**两个 build 目录，不要在同一目录翻选项**：

```text
<IME2_BUILD>      cmake ... -DMNN_USE_RVV=ON -DMNN_RVV_SPACEMIT_IME2=ON
<PURE_RVV_BUILD>  cmake ... -DMNN_USE_RVV=ON -DMNN_RVV_SPACEMIT_IME2=OFF
```

原因是硬的：`MNN_RVV_SPACEMIT_IME2=ON` 会把 `rvv/MNNRvvFastPathRegistration.cpp` 从 `MNNRVV`
源列表里 `REMOVE_ITEM`，两个注册 TU 是**构建期互斥**的同名符号
（见 [`../kernel/arch/riscv.md`](../kernel/arch/riscv.md) §一）。同目录翻选项后的增量构建
不保证把这个文件集合变化正确传导，拿到的对照可能根本不是你以为的两条路径。

复用已配置目录只做增量构建：`cmake --build <BUILD> -j <JOBS>`。
**退出码自己看**（`make > make.log 2>&1; echo make_exit=$?`），理由同
[`build-test-and-benchmark.md`](build-test-and-benchmark.md) §六.1。

构建后核对四条：

- vendor kernel object 带厂商 ISA 参数，`MNNRVV` 与 runtime lib **不带**；
- `libMNN`、LLM 库、`llm_demo`、`llm_bench` 都真的链接成功；
- OFF 变体里没有 vendor symbol、也没有未解析引用；
- **benchmark 加载的是本次产物**：用 `realpath` / `stat` / `readelf -d` 核对二进制与 `.so` 的真实路径、
  时间戳和 loader 解析结果。这条在 RISC-V 上尤其容易翻车，因为板上常同时存在系统安装的 MNN
  和多个 build 目录；机制与 macOS 侧的 `LC_RPATH` 陷阱同源，见
  [`build-test-and-benchmark.md`](build-test-and-benchmark.md) §六.4。

## 三、正确性矩阵：按成本从低到高

| 层级 | 必测内容 |
|---|---|
| Kernel | 主 tile、K/N tail、signedness、scale/offset、direct output |
| Op | `run_test.out` 的 DenseConv / MatMul / Attention，C4、tail、线程分片 |
| Vendor smoke | 短 prompt、固定采样、若干 decode token |
| Pure RVV smoke | **OFF 变体跑同一 prompt** |
| Long prompt | 跨过 prefill / Attention 的分支阈值（IME2 fused Attention 的门禁有 `seqLen` 上下界与整除条件） |
| Cross-model | 至少一个小模型 + 一个较大或结构不同的模型 |

`run_test.out` 的位置参数语义、`memory=` / `thread=` / `precision=` 自证行、
以及 zsh 不分词导致的空测，全部与本机一致，见
[`build-test-and-benchmark.md`](build-test-and-benchmark.md) §二/§六.6。

量化 kernel 用逐层比较，别只比最终输出：

```text
packed input -> integer accumulator -> dequantized FP32 -> zero-point correction -> epilogue/output layout
```

标量 oracle 必须从**实际 packed layout** 读并复现 kernel 的运算顺序，理由见
[`../kernel/arch/riscv.md`](../kernel/arch/riscv.md) §六。
TCM 路径开发期可让同一 tile 同时跑 TCM 与 DRAM kernel 逐位比较；**验证完删掉重复计算与诊断输出**。

模型输出异常时先固定采样方式，再依次比较 fallback / 纯 RVV / FP16 / FP32，
区分 kernel 错、量化误差与采样波动——三者的修法完全不同。

## 四、性能实验：板端的额外约束

命令形态与本机一致（`llm_bench` 的 `-p` / `-n` / `-rep` / `-t` / `-load` 见
[`build-test-and-benchmark.md`](build-test-and-benchmark.md) §四），
**用 benchmark 原生参数选 prefill/decode，不要为跑分往生产代码里加 `getenv`。**

RISC-V 板上额外要做的：

1. **每个候选与基线至少跑多个新进程**，尽量 A/B、B/A、A/B 交替；
   进程内低方差**不等于**跨进程稳定。
2. **prefill 与 decode 分别报**，不要合成一个"提升 x%"。
3. 做线程数或门槛 sweep 时**一次只改一个变量**；
   **不要默认用满全部核心**——增加 worker 可能只增加共享矩阵单元争抢、DRAM 竞争与 barrier 成本。
4. 每轮记录频率、温度、系统负载与 CPU governor。热漂移用 inert 对照组分离，同
   [`build-test-and-benchmark.md`](build-test-and-benchmark.md) §六.2。
5. **decode 用带宽反推上限**，别拿接口峰值或稀疏 TOPS 当可达吞吐：
   ```text
   effective_bandwidth = bytes_per_token * tokens_per_second
   ```
6. **prefill 同时记 pack / 动态量化时间与 GEMM 时间**，防止 kernel 变快但前处理吃掉收益。
7. TCM / DMA 流水分别记 copy 时间、compute 时间、barrier/dispatch 时间、流水总时间、DRAM fallback 时间。
   **只有总时间稳定下降且结果一致才保留。**
8. **首次 cold prefill 与连续多个不同 shape 逐请求计时**，不要只报汇总速度：
   一次性的权重重排会被平均值吃掉（机制与 `onRecompute` 修法见
   [`../optimize/runtime-and-scheduling.md`](../optimize/runtime-and-scheduling.md) §2.6）。
9. 性能提升后**重新跑一遍正确性 smoke**（改快了顺手改错的概率不低）。

板端一次性核对清单（第一次上板或换板时）：编译器版本与 vendor ISA 支持、
`lscpu` / runtime 报告的核心数与 VLEN（IME2 汇编要 VLENB=128）、TCM runtime/device 是否可用、
当前 governor / 频率 / 负载、目标模型与量化配置。

## 五、交付时必须交代的五件事

板端结论离开会话就只剩 commit body 和 CR 描述。这五条缺一条，review 者无法判断改动的适用范围：

1. **改了哪三层里的哪几层，以及为什么不能只复用通用路径**；
2. **哪些 shape / 量化格式真的命中新路径，哪些自动回退**——vendor fast path 的门禁很窄，
   不写清就会被当成「全场景提升」；
3. **正确性和性能各跑了什么**（§三 / §四 的哪几层，多少个进程，什么线程档）；
4. **纯 RVV OFF 变体是否验证过**——只证明 vendor 构建可用不算交付完成；
5. **改完之后仍受计算、带宽还是同步限制**，以及是否存在目标板 / runtime / 模型条件。

## 六、日志与脱敏

内部开发记录可以引用远端路径和日志。**对外报告必须移除**：

- SSH 主机、代理、用户名；
- 内部仓库 / build / model 路径；
- 内部提交号或评审编号；
- 未公开的硬件信息；
- 无法由公开口径支持的竞品结论。

提交前跑一遍敏感信息扫描，并**人工检查命令块、表格脚注和图片说明**——
这三处是自动扫描最容易漏的地方。

每条数据自带可复现坐标（commit、构建选项、量化格式、benchmark 完整参数、thread、
多进程原始数据与汇总、正确性结果、频率/温度限制、回退条件与未覆盖场景），
写法同 [`build-test-and-benchmark.md`](build-test-and-benchmark.md) §七：
**数据不进本仓，方法进 skill。**
