---
name: hexagon-optimization
description: MNN Hexagon/HVX/HMX DSP 后端（`source/backend/hexagon`）的优化、重构、构建与回归验证。覆盖设备实测的相位分解、测量纪律、v79/v81 双架构差异、常见瓶颈模式、已否证方向与 cDSP crash 诊断。
---

# MNN Hexagon 优化 Skill

> **触发条件**：优化/重构 MNN Hexagon DSP 代码路径，分析 `DSPOpType` 性能数据，诊断 cDSP crash，或做 Hexagon 侧内存占用测量。

在这个后端里，**正确性、设备稳定性和实测数据的价值远高于凭直觉的微优化**。本文所有经验规则都由实测得出，多数是某次错误结论换来的。

## 硬约束

- 不允许用 CPU fallback 当优化。不要拒绝 Hexagon 支持、把算子改路由到 CPU，也不要保留任何"把工作搬离 Hexagon 换来的耗时下降"。
- 不允许用算子融合当优化，除非用户明确解除该限制。包括图级融合、host 命令融合、DSP 命令组序列融合，以及用融合自定义算子替代多个算子。
- 优化范围限定在 Hexagon/HTP/DSP 的调度、kernel、分块、数据搬运、资源加锁、profiling 清理等**不改变后端归属**的实现细节内。
- **代码分支必须由设备决定**（`info().hvxArch`、`info().maxThreads`、运行时查询的 VTCM 大小），**不能由环境变量决定**。同一颗芯片必须永远走同一条路径。env 开关只允许作为改动开发期的临时 A/B，上线前必须删除 —— `MNN_HEXAGON_Q4_GEMV_I8` 和 `MNN_HEXAGON_PATHA` 就是这类，已按此惯例移除。
- 不要在热 kernel 里留"默认关闭的半成品实现"来"保留未来可能性"。git 历史比散布在热函数里的八处 `#if` 更适合保管它。

## 工作流

1. 改之前先读现有实现：
   - host/backend 侧：`source/backend/hexagon`
   - DSP kernel 与派发：`source/backend/hexagon/htp-ops-lib/src/dsp`
   - 复用已有的 HVX/HMX/DMA 辅助函数，不要另起一套平行抽象。
2. 改动先求小而保行为：
   - 优先做小的机械重构，再改算法。
   - 不要随意加大 worker 栈，除非明确要求。
   - 最终代码里不留临时 profiling、调试日志、实验性开关。
3. **先测量再优化。** 打开对应的 gated profiler（见「相位 Profiler」），让相位分解决定优化目标。这个后端里大部分收益都落在没人会猜到的相位上：那个"矩阵乘"算子曾经只有 **4%** 的时间在做矩阵乘，**49%** 花在把结果写回内存。
4. 每个有意义的 DSP 改动都要验证：
   - 重编 DSP `.so`，**同时重编 host**（见测量纪律第 1 条，无条件）。
   - 把生成的 `.so` 拷进 `project/android/build_64`。
   - 跑目标模型测试，检查 `DSPOpType` 输出。
   - cDSP/qurt crash、设备掉线、profile RPC 失败，都算测试失败。
5. 直接比较目标算子：
   - 用 `DSPOpType <OP>` 的数字，不要只看 RTF 或总墙钟。
   - 多跑几次，报告离散度，尊重该设备的噪声带。
   - 只保留"目标算子变快且不损害正确性与稳定性"的改动。
6. **声明收益前必须两个架构都测。** 一个改动在某个 arch 上有效、在另一个上无效甚至变差，原因往往与 kernel 本身无关 —— 见「双架构」。
7. **记录被否证的尝试，连同否证它的那个数字。** 负面结果通常能定位瓶颈（那个"毫无收益"的 `:nt` store 恰恰证明了该相位是 issue-bound），并且能避免下一个人重跑一遍。

## 构建与同步

按项目脚本预期的目录执行命令。

- `Executor::RuntimeManager::createRuntimeManager` 会把 backend runtime 加到**创建时当前的** Executor 上。要在 benchmark 期间保持 DSP 电源，必须在 runtime manager 存在之后、从同一个 Executor 创建 activation guard，并让它在被测区间内一直存活；不要为了供电另建一个 Hexagon Executor。
- Android 构建目录：`cd project/android/build_64`
- 重编 host 产物并更新设备：
  - `cd project/android/build_64 && ../build_64.sh -DMNN_HEXAGON=ON -DMNN_GPU_TIME_PROFILE=ON && ../updateTest.sh`
  - `updateTest.sh` 对未构建的可选二进制会打印 `adb: error: cannot stat`，属正常。确认 `libMNN.so`、`ModuleBasic.out` 和目标 demo 已推送即可。
- 改了 `htp-ops-lib/src/dsp/*` 之后重编 DSP：
  - `cd source/backend/hexagon/htp-ops-lib && source ~/.bash_profile && sh sync_remote_build.sh`
  - 也可以显式传构建机与 SDK，比依赖 profile 更可靠：
    `REMOTE_SSH=<user@buildhost> HTP_OPS_SDK_ENV=<path/to/setup_sdk_env.source> bash sync_remote_build.sh v81`
  - 指定架构就显式传，例如 `sh sync_remote_build.sh v79`。
  - 确认脚本报告"无未预期的 undefined symbol"并成功推送 DSP 库。
  - **给非当前设备的架构构建是个坑。** 脚本会自动推送到唯一连接的设备，而 FastRPC stub `libMNN_htpops.so` 是按 arch 不同的，所以一次 v79 构建会覆盖 v81 设备的 stub。跨架构构建时屏蔽 adb（`PATH=/usr/bin:/bin bash sync_remote_build.sh v79`，脚本会打印 `skip adb push`），再自己把产物拷到另一台设备。
  - **DSP 构建不可复现**：远程构建目录的唯一 id 会进入产物，同一份源码每次编出的 md5 都不同。`.so` 的 md5 只能证明"编过一次"，**永远不能证明"设备上是哪份源码"**。设备状态要用行为验证 —— 见测量纪律。
- 本地测试时把 DSP 库拷进 Android 构建目录：
  - `cp source/backend/hexagon/htp-ops-lib/outputs/libMNN_htpops.so project/android/build_64/libMNN_htpops.so`
  - `cp source/backend/hexagon/htp-ops-lib/outputs/libMNN_htpops_skel.so project/android/build_64/libMNN_htpops_skel.so`

## 测量纪律

**下面每一条都是因为违反过一次、得出了一个自信但错误的结论。**

1. **每个测量点都无条件重编 host。** 绝不能以"这个提交只改了 DSP 代码"为理由跳过。host 与 DSP 库是配对的：拿提交 N 的 DSP 库配提交 N-1 的 `libMNN.so`，会静默选中不同的 kernel。曾因此把某个点测成 `tg128 57.4`（权重路径实际被关闭）而不是 `89.2`。host 增量构建只要几秒，这个错误却要花掉几小时。
2. **相信一个数字之前，先确认设备上到底跑的是什么。** md5 做不到这件事（构建不可复现）。用行为来确认：
   - 身份行 `[MNN::Hexagon] vectorSize=.. vtcmSize=.. maxThreads=.. hvxArch=..` 确认是哪台设备、哪个 skel 架构；
   - 一次 greedy 解码与**该设备自己的**参考文本逐字节比对，确认跑的是哪份代码。
3. **只在同一份代码版本内做比较。** 不要拿旧提交的数字和当前树的数字比，更不要跨两条不同算法路径比。两者同时犯就会得出"少两个线程反而快 16%"这种结论 —— 同代码扫一遍线程数即被完全否证（8/7/6 线程下平坦）。
4. **先确定每台设备的噪声带，带内的差值一律不读。** 裸开发板 prefill 可以做到 ±0.3%；跑着完整 Android 系统的真机是 prefill ±4%、decode ±7%，短 prompt 更差。任何 `std/mean` 超过约 2% 的点都要重测。
5. **测量周期的首尾都要复测基线，不只是开头测一次。** 设备最凉最闲时的第一次读数可能是离群值：曾有这样一个读数让"持平"看起来像 7% 的性能回退，只有复测同一个提交才暴露出来。
6. **真机上的噪声是单边的。** 后台系统服务会制造单次灾难性掉速 —— 曾观测到某次 prefill 均值的标准差达到均值的 **69%**。既然只会变慢，这类设备上就该用 best-of-N 或中位数而非均值，并说明用了哪个估计量。
7. **基线取"改动前能达到的最好配置"，不是"出厂默认配置"。** 曾有一个默认路径比同一个 build 里另一条已存在的路径慢 19%，若按默认配置做基线，就会把"修好别人的劣化"算成自己的优化成果。
8. **分清每个数字覆盖什么。** `llm_bench -p N` 是纯 prefill、`-n N` 是纯 decode，但 `llm_demo` 带 prompt 文件跑的时候会**混入尾部的 decode 步**，所以那次运行里的单算子时间是两个阶段之和。只有确知是哪个阶段贡献的，才能做算子级归因。
9. **相位数字与吞吐数字是两种测量，不能混用。** 打开 gated profiler 会扰动被测对象本身，所以永远不要把带 profiler 的相位数字塞进吞吐对比里；要说明每个数字出自哪个 build。

## 相位 Profiler

编译期开关，默认关闭，关闭时零开销。**这些是仪器，动手推测之前先用它们。**

| 开关 | 位置 | 报告内容 |
|---|---|---|
| `HTP_MM_PHASE_PROFILE` | `ops/matmul_q4fp16.c` + `execute_command.cc` | prefill GEMM 各相位 → `profile[200..212]`：HMX 计算、等权重反量化、DMA 等待、输出写回、激活 shuffle、setup、worker 侧写回/反量化累计时间 |
| `HTP_QATTN_PHASE_PROFILE` | `attention_hmx.cc` + `execute_command.cc` | attention 队列线程相位 → `profile[244..250]`：DMA 等待、HMX 计算、score 写回，外加 K/V 字节数、激活字节数和 **dmstart 次数** |
| `HTP_WATTN_PHASE_PROFILE` | `attention_sync_process.cc` + `execute_command.cc` | attention worker 相位 → `profile[251..255]`，按 worker 累加：Q 收集、QK 提交+等待、softmax、SV 提交+等待、O 散出 |

**一对文件里的 define 都要改**，重编后这些值会作为伪算子出现在 `DSPOpType` 表里。报告吞吐前记得改回关闭。

怎么读：

- **dmstart 计数器的存在意义**是回答"这个 DMA 等待是按字节收费还是按次数收费"。用等待时间除以次数：若得到一个稳定的每次耗时、且**字节数变化时它不变**，就是**按次数收费**，解法是"更少但更大的传输"，而不是"更少的字节"。
- **worker 侧数字是跨 worker 累加的。** 要除以 worker 数才得到墙钟贡献，再与队列线程总和比较，才知道哪一侧是临界路径。
- **DSP 侧的 FARF 不一定能到 `adb logcat`**（在 v79 上就到不了，`-b all` 也不行）。不要把诊断建立在 FARF 探针上，优先用 profile slot —— 它们通过 profile buffer 回传。

## 双架构（v79 / v81）

`info().hvxArch` 和 `info().maxThreads` 在 host 侧可用（`HexagonRuntime::info()`）；arch 值源自 DSP（`commu.cc` 把 `__HVX_ARCH__` 写进 info 结构体），因此它报告的是 **FastRPC 为这颗芯片加载的那份 skel**。查询失败会读到 0 —— 按"更旧/更安全的路径"处理。

- **逐算子的 v79/v81 耗时比集中在 1.1–1.3×**，即**优于** 6 对 8 线程本身预测的 1.33×。某个算子明显偏离这个区间，才是"该 arch 有专属问题"的信号，而不是普遍偏慢。
- **绝不能让工作划分数量与线程数撞上。** 某条路径恰好切成 `n_kv_heads` = 8 个任务：在 8 线程设备上正好一波跑完，在 6 线程设备上变成两波，尾波只有 2/6 的占用率 —— 足以把一个 30% 的算法收益全部吃掉。对每个划分都要问一句：**这个数是不是偶然等于某个硬件参数？** 任务数要远多于线程数；当任务开销不均时（因果掩码使靠后的块贵得多）**必须先派发最贵的**。
  反过来也要注意：本来就能填满整波的划分，再细拆只会增加开销 —— 无条件细拆曾让某算子变慢 3.5%。
- **一台设备就能验证 arch 假设。** 临时把 `worker_pool.cc` 里的 `g_max_num_workers` 钉死成更窄设备的线程数。若 N 与 N-1 之间出现悬崖、且 N-1 与 N-2 几乎相同，就是**波次量化**；若是平滑缩放，才是真正的单线程性能差异。正是这一招把"v79 硬件不同"（错）和"任务数恰好等于 v81 的线程数"（对）区分开。
- **有些 arch 分支是真实硬件约束，不可统一**：v79 的 HMX 只支持共享锁，v81 是独占（`hmx_mgr.cc`）。另一些则是历史遗留、值得重测 —— 一条被 `__HEXAGON_ARCH__ >= 81` 门控的 prefill 路径，曾把五项优化整个从 v79 的构建里编译掉，而 host 侧仍在为它们分配 workspace。
- **权重布局按设计就是 arch 相关的**：M=1 整数 GEMV 需要的 vrmpy 布局只在 v81 及以上划算，所以新架构只保留该布局、旧架构只保留 HMX 布局。**要在重排之前就决定**，让用不到的布局根本不被生成 —— "先生成再释放"曾占掉模型加载时间的 5.4%。

## 本后端已实测的瓶颈模式

想新理论之前先对照这些，每一条都在这里至少测到过一次。

- **DMA 的成本由每次传输的固定延迟主导，而非字节数。** 把大量小传输合并成一次 2D 传输。
- **写回/搬运相位可能是 issue-bound 而非带宽受限。** 症状：换成非临时存储毫无变化。解法是提高并行发射或减少每字节的指令数，而不是加 cache 提示。
- **不满一个槽位的写入会静默退化成 read-modify-write。** 若一次写只覆盖目标 packing 槽位的一半，它就变成"读—改—写"，触碰 2 倍内存。攒到能配对再写。
- **搬运修好之后瓶颈会转到计算，反之亦然。** 每接受一个改动都要重跑相位分解；**相位级大幅收益但墙钟收益很小，说明临界路径已经不在那里了**，而这恰好指出下一步该看哪。
- **多趟 DDR 遍历会藏在看起来很朴素的辅助函数里。** 某个 softmax 把 score 读两遍、输出也读两遍，合计每元素 10 字节 DDR；把整行暂存进 L1 后降到 4 字节，整个算子 −9.4%，且结果逐位一致。
- **把工作交给 worker 不是免费的。** 当这份工作本来就顶在 DDR 带宽上限时，握手就是纯增开销：inline 写回打败了提交给 worker 的同一份写回。

## 已否证方向（不要重做）

以下全部实现并测量过，别再重新推导一遍。

- prefill attention 的 `dmlink` 连续链 DMA 解耦：实现正确，**收益 0 ms**。
- 提高因果分组的 Q 行数（64 → 128）：**两次否证**。
- prefill GEMM 用非临时（`:nt`）存储：无收益（但由此得出上面的 issue-bound 结论）。
- 激活 shuffle 里加提前一块的显式 `l2fetch`：**慢 1.7 ms**。
- 让派发线程自己反量化第一块而不是等 worker：**持平**。
- prefill 权重 staging 双缓冲：**导致设备重启**。host 侧已经按"恰好一份权重填满 VTCM"来确定 `np`，而 `vtcm_seq_alloc` 是无边界检查的 bump 分配器，第二份直接越界。**任何增加 VTCM 占用的改动，都必须在同一个提交里改 host 侧的 sizing。**
- 为队列线程预留一个 HVX 槽位（怀疑超额订阅）：**否证**，耗时在各 worker 数下平坦。

## 设备风险

- **4096-token 的 prefill 会硬重启设备**，在干净的历史版本上同样如此。除非专门排查那个 crash，长 prompt 一律封顶到 2048。
- `vtcm_seq_alloc` **不做边界检查**。加宽任何 staging buffer 之前，手算一遍 VTCM 占用。
- host 上传会落进被回收的设备内存，而 DSP 可能仍持有该内存的脏 cache 行，这些行可能在之后任意时刻被驱逐、盖在刚上传的数据上。**必须在 host 写入之前**把目标区间刷回 —— 写入之后再刷会让情况严重得多。
- 设备掉线后：`adb kill-server && adb start-server`，确认序列号回来之后，才可以对之前那次失败的运行下任何结论。

## Profile 与 Crash 检查

- 查失败前先清日志：`adb logcat -c`
- 运行后取日志：`adb logcat -d`
- 搜索失败特征：
  - `rg -i "execute_command_group_profile failed|qurt|sysfatal|fatal|crash|tlb|cdsp.*crash|adsp.*crash|segv|signal 11"`
- 若运行失败、超时、返回 profile RPC 错误，或设备短暂掉线，继续之前先查 cDSP tombstone/ramdump：
  - `adb shell "ls -lt /data/tombstones /data/vendor/tombstones /data/vendor/ramdump /data/vendor/ssrdump 2>/dev/null | head -80"`
  - `adb shell "find /data/tombstones /data/vendor/tombstones /data/vendor/ramdump /data/vendor/ssrdump -maxdepth 2 -type f 2>/dev/null | tail -40"`
  - 只把最新的相关文件或小目录拉到本地临时路径检查。
  - 在拉取的文件里搜 DSP 失败特征：
    - `rg -i "cdsp|adsp|qurt|sysfatal|fatal|crash|tlb|page fault|protection|signal|MNN|htp|fastrpc" <pulled_path>`
  - 总结出有用信号后删除临时拉取的 tombstone/ramdump 文件。
- 以下一律算失败：
  - `[Hexagon] execute_command_group_profile failed with code -2147482610`
  - qurt/cDSP fatal 日志。
  - 测试窗口内新增或更新的 cDSP tombstone/ramdump。
  - 测试期间设备 offline/掉线。
  - 改了 DSP 代码后 `.so` 缺失或过期。
- 有用的 profile 字段：
  - `Hexagon DSP Profile`
  - `Command groups`
  - `Command dirty`
  - `DSPOpType <name> (<id>): <time> ms`
  - `Hexagon onCopyBuffer Profile`

## HVX/HMX 指引

- HVX 指令细节参考 `~/Download/hvx.pdf`。
- 引入新的 intrinsic 写法之前，先找项目里已有的例子。
- 只在对齐有保证时用 `vmem`；不对齐访问用 `vmemu`。
- DMA/HVX/HMX 的改动要落在"内存流量"和"实测算子时间"上。
- HMX 路径在加宽 tile 或 block 之前，先核对 VTCM 分配大小、tile 数和 descriptor 数。
- 不要假设一个微优化能在 v79/v81 之间通用；针对目标架构构建并测试。不通用的原因、以及如何用单台设备验证 arch 假设，见「双架构」。

## DSP DMA-BUF 内存测量

用于验证 Hexagon/DSP 内存占用，或对比 CPU 与 `forwardtype=10`。

1. 把模型配置成走 DSP：
   - 模型 `config.json` 设 `forwardtype=10`。
   - 除非任务明确要求，保持模型原本的 precision。
2. 用足够多的重复输入让进程活着，便于采样。
3. 找到进程 PID：`adb shell ps | grep <process_name>`
4. 进程还活着时，把三个内存视图都采一遍：
   - 进程 RSS/HWM：`adb shell "awk '/VmRSS:|VmHWM:/{print}' /proc/<PID>/status"`
   - 进程 DMA-BUF：`adb shell dmabuf_dump <PID>`
   - 完整内存：`adb shell dumpsys meminfo <PID>`
5. 尝试系统级 DMA-BUF 总览，但要明确记录权限/路径失败：
   - `adb shell cat /sys/kernel/debug/dma_buf/bufinfo`
   - 有些设备暴露 `/sys/kernel/dmabuf/buffers`，可能需要 root。
6. 内存数字分开报告：
   - `VmHWM`、`TOTAL RSS`、`TOTAL PSS`、`dmabuf total` 或 `PROCESS TOTAL`
   - 需要时用 `VmHWM + dmabuf total` 近似"进程可见的 DSP 压力"。

## 安全检查清单

- 目标模型通过正确性检查，例如 `cos_sim` 可接受。
- 目标 demo 输出符合预期且不 crash。
- greedy 解码输出与**该设备的**参考文本逐字节比对过。如果变了，必须已经弄清原因并明确说明 —— 这个后端里大部分 kernel 工作都应当是数值保持的，**输出有差异是一个发现，不是一个细节**。
- 报告了 `DSPOpType` 目标耗时，并与明确的基线对比，且**基线是改动前能达到的最好配置**。
- v79 和 v81 都构建并测量过；只覆盖其中一个时，说明理由。
- 报告的差值超出测量设备的噪声带，且均值旁边给出了离散度。
- 每个测量点都重编过 host。
- `git diff --check` 干净。
- 没有残留的临时日志、计数器、逐算子 tracing、写死的 worker 数或 env 开关，除非明确要求。gated profiler 已恢复默认关闭。
- 报告性能之前，重编的 DSP `.so` 已拷贝/推送到位。
