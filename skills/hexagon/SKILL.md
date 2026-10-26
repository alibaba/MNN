---
name: hexagon-optimization
description: MNN Hexagon/HVX/HMX DSP 后端（`source/backend/hexagon`）的优化、重构、构建与回归验证。覆盖设备实测的相位分解、测量纪律、v79/v81 双架构差异、常见瓶颈模式、已否证方向与 cDSP crash 诊断。
---

# MNN Hexagon 优化 Skill

> **触发条件**：优化/重构 MNN Hexagon DSP 代码路径，分析 `DSPOpType` 性能数据，诊断 cDSP crash，或做 Hexagon 侧内存占用测量。

在这个后端里，**正确性、设备稳定性和实测数据的价值远高于凭直觉的微优化**。本文所有规则都由实测得出，多数是某次错误结论换来的。

**本文件只放可复用的判据，不放逐轮的对标大表** —— 那些随每次测量过期，放进来只会让下一个人拿旧数字当基线（违反测量纪律第 3 条）。要历史数据就去翻 git log 里对应优化提交的 message。

## 硬约束

- 不允许用 CPU fallback 当优化。不要拒绝 Hexagon 支持、把算子改路由到 CPU，也不要保留任何"把工作搬离 Hexagon 换来的耗时下降"。
- 不允许用算子融合当优化，除非用户明确解除该限制。包括图级融合、host 命令融合、DSP 命令组序列融合，以及用融合自定义算子替代多个算子。
- 优化范围限定在 Hexagon/HTP/DSP 的调度、kernel、分块、数据搬运、资源加锁、profiling 清理等**不改变后端归属**的实现细节内。
- **代码分支必须由设备决定**（`info().hvxArch`、`info().maxThreads`、运行时查询的 VTCM 大小），**不能由环境变量决定**。同一颗芯片必须永远走同一条路径。env 开关只允许作为开发期的临时 A/B，上线前必须删除。
- 不要在热 kernel 里留"默认关闭的半成品实现"来"保留未来可能性"。git 历史比散布在热函数里的八处 `#if` 更适合保管它。

## 工作流

1. 改之前先读现有实现（host 侧 `source/backend/hexagon`，DSP kernel 在 `htp-ops-lib/src/dsp`）。复用已有的 HVX/HMX/DMA 辅助函数，不要另起一套平行抽象。
2. 改动先求小而保行为：优先做小的机械重构再改算法；不要随意加大 worker 栈；最终代码里不留临时 profiling、调试日志、实验性开关。
3. **先测量再优化。** 打开对应的 gated profiler，让相位分解决定目标。这个后端里大部分收益都落在没人会猜到的相位上：那个"矩阵乘"算子曾经只有 **4%** 的时间在做矩阵乘，**49%** 花在把结果写回内存。
4. 每个有意义的 DSP 改动都要验证：重编 DSP `.so` **并同时重编 host**（测量纪律第 1 条，无条件）→ 拷进 `project/android/build_64` → 跑目标模型 → 检查 `DSPOpType`。cDSP/qurt crash、设备掉线、profile RPC 失败，都算测试失败。
5. 比较时用 `DSPOpType <OP>` 的数字，不要只看 RTF 或总墙钟；多跑几次并报告离散度。
6. **声明收益前必须两个架构都测。**
7. **记录被否证的尝试，连同否证它的那个数字。** 负面结果通常能定位瓶颈，并避免下一个人重跑一遍。

## 构建与同步

- Android 构建目录：`cd project/android/build_64`
- 重编 host 并更新设备：
  `cd project/android/build_64 && ../build_64.sh -DMNN_HEXAGON=ON -DMNN_GPU_TIME_PROFILE=ON && ../updateTest.sh`
  （`updateTest.sh` 对未构建的可选二进制打印 `adb: error: cannot stat`，属正常。）
- 改了 `htp-ops-lib/src/dsp/*` 之后重编 DSP：
  `cd source/backend/hexagon/htp-ops-lib && source ~/.bash_profile && sh sync_remote_build.sh v81`
  显式传构建机与 SDK 比依赖 profile 可靠：`REMOTE_SSH=<user@buildhost> HTP_OPS_SDK_ENV=<path/to/setup_sdk_env.source> bash sync_remote_build.sh v81`
- ⚠️ **给非当前设备的架构构建是个坑。** 脚本会自动推送到唯一连接的设备，而 FastRPC stub `libMNN_htpops.so` 按 arch 不同，一次 v79 构建会覆盖 v81 设备的 stub。跨架构构建时屏蔽 adb（`PATH=/usr/bin:/bin bash sync_remote_build.sh v79`，脚本会打印 `skip adb push`），再自己把产物拷到另一台设备。
- ⚠️ **DSP 构建不可复现**：远程构建目录的唯一 id 会进入产物，同一份源码每次编出的 md5 都不同。`.so` 的 md5 只能证明"编过一次"，**永远不能证明"设备上是哪份源码"**。
- `Executor::RuntimeManager::createRuntimeManager` 会把 backend runtime 加到**创建时当前的** Executor 上。要在 benchmark 期间保持 DSP 电源，必须在 runtime manager 存在之后、从同一个 Executor 创建 activation guard 并让它在被测区间内存活；不要为了供电另建一个 Hexagon Executor。

## 测量纪律

**下面每一条都是因为违反过一次、得出了一个自信但错误的结论。**

1. **每个测量点都无条件重编 host。** 绝不能以"这个提交只改了 DSP 代码"为理由跳过。host 与 DSP 库是配对的：拿提交 N 的 DSP 库配提交 N-1 的 `libMNN.so`，会静默选中不同的 kernel。曾因此把某个点测成 `tg128 57.4` 而不是 `89.2`。
2. **相信一个数字之前，先确认设备上到底跑的是什么。** md5 做不到（构建不可复现）。用行为确认：身份行 `[MNN::Hexagon] vectorSize=.. vtcmSize=.. maxThreads=.. hvxArch=..` 确认设备与 skel 架构；一次 greedy 解码与**该设备自己的**参考文本逐字节比对，确认跑的是哪份代码。
3. **只在同一份代码版本内做比较。** 不要拿旧提交的数字和当前树的比，更不要跨两条不同算法路径比。两者同时犯就会得出"少两个线程反而快 16%"这种结论。
4. **先确定每台设备的噪声带，带内的差值一律不读。** 裸开发板 prefill 可做到 ±0.3%；跑完整 Android 的真机是 prefill ±4%、decode ±7%，短 prompt 更差。`std/mean` 超过约 2% 的点要重测。
5. **测量周期的首尾都要复测基线，不只开头测一次。** 设备最凉最闲时的第一次读数可能是离群值：曾有这样一个读数让"持平"看起来像 7% 的回退。
6. **真机上的噪声是单边的。** 后台服务会制造单次灾难性掉速（曾观测到标准差达均值的 **69%**）。既然只会变慢，就用 best-of-N 或中位数而非均值，并说明用了哪个估计量。
7. **测量前必须等设备降温。** 连续测量会让 NSP/CPU 升温降频，**同一份库的读数能差 10% 以上**，而且是单调下行 —— 按顺序测 A/B 会系统性地让后测的那个更差。做法：读 `/sys/class/thermal/thermal_zone*/type` 里匹配 `nsp*` / `cpu-*` / `cpuss-*` 的区，等最高温降到 38 °C 以下再跑，并在每次运行之间留冷却间隔。**A/B 必须交替顺序多轮，不能 AAA 之后 BBB。**
8. **基线取"改动前能达到的最好配置"，不是"出厂默认配置"。** 曾有一个默认路径比同一个 build 里另一条已存在的路径慢 19%，按默认配置做基线就会把"修好别人的劣化"算成自己的成果。
9. **分清每个数字覆盖什么。** `llm_bench -p N` 是纯 prefill、`-n N` 是纯 decode，而 `llm_demo` 带 prompt 文件跑时会**混入尾部的 decode 步**。
   **而且 `llm_demo` 自己报的 `prefill speed` 不能用来判断性能**：它含首次 cache 构建、单次运行无重复。同一份库 `llm_demo` 报 1498 tok/s 而 `llm_bench` 报 4671 —— 差 3 倍。**`llm_demo` 只用来看输出对不对，速度一律看 `llm_bench`。**
10. **相位数字与吞吐数字是两种测量，不能混用。** 打开 gated profiler 会扰动被测对象，永远不要把带 profiler 的相位数字塞进吞吐对比里。
11. ⚠️ **`sync_remote_build.sh` 的 `adb push` 目标写死在 `/data/local/tmp/MNN/`，而测试常在别的目录里跑。** 两个目录各有一份 `libMNN_htpops_skelV81.so`，`ADSP_LIBRARY_PATH=.` 加载的是 cwd 那份 —— **脚本报告 "2 files pushed" 之后，被测目录里的库可能仍是旧的**。曾因此连续十几次测量全跑在旧库上：改动"没有效果"、相位 profiler"打不开"（槽位全 0）、A/B 两组分毫不差，都是同一个原因。
    **每次构建后核对 md5**：`md5 outputs_x/libMNN_htpops_skelV81.so` 对 `adb shell md5sum <测试目录>/libMNN_htpops_skelV81.so`。
    判据：**一个本该有效果的改动若给出和基线逐位相同的数字，先怀疑库没换上，再怀疑结论。**
12. ⚠️ **设备硬重启会丢掉 `/data/local/tmp` 里刚 push 的文件**（push 不 fsync）。于是 A/B 脚本里的 `cp libMNN.$v.so libMNN.so` 静默失败，两组读数其实跑的是同一个库 —— **数字差不到 0.5%，看着像"改动无影响"的漂亮结论**。在换库的脚本里直接断言 md5：
    ```bash
    adb -s $D shell "cd $DIR && cp libMNN.$v.so libMNN.so"
    got=$(adb -s $D shell "md5sum $DIR/libMNN.so" | awk '{print $1}' | tr -d '\r')
    [ "$got" = "$want" ] || { echo "MD5 MISMATCH"; exit 1; }
    ```
    崩溃过的测量周期里，**重启之后要重新 push 一遍全部产物**。
13. ⚠️ **`DSPOpType` 表里有父桶和子桶，直接相加会重复计数。** `execute_command.cc` 先按 `opType` 累加父桶、再按细分类型累加子桶，而 host 把 256 个槽无差别逐个打印。已知两对：`TENSOR_CONVERT` = `TYPE0 + TYPE1`、`UNARY` = `SILU + SIGMOID`。曾因此把布局转换记成 16.8 ms（实际 11.8）并据此排了优先级。
14. ⚠️ **本仓库有多个 worktree，`cd` 一律用绝对路径。** Bash 工具的 cwd 会在某些时刻被重置回主工作树，之后所有相对 `cd` 都落到**另一个分支**上。每次动手前 `pwd` + `git rev-parse --abbrev-ref HEAD` 自检。

## 相位 Profiler

编译期开关，默认关闭，关闭时零开销。**这些是仪器，动手推测之前先用它们。**

| 开关 | 位置 | 报告内容 |
|---|---|---|
| `HTP_MM_PHASE_PROFILE` | `ops/matmul_q4fp16.c` + `execute_command.cc` | prefill GEMM 各相位 → `profile[200..231]`：HMX 计算、等权重反量化、DMA 等待、输出写回、激活 shuffle、setup、worker 侧累计 |
| `HTP_MM_MICROBENCH` | 同上 | 在第一次大调用里单独测每种资源的单价 → `profile[104..118]`（host 打印值/1000） |
| `HTP_QATTN_PHASE_PROFILE` | `attention_hmx.cc` + `execute_command.cc` | attention 队列线程相位 → `profile[244..250]`：DMA 等待、HMX 计算、score 写回，外加字节数和 **dmstart 次数** |
| `HTP_WATTN_PHASE_PROFILE` | `attention_sync_process.cc` + `execute_command.cc` | attention worker 相位 → `profile[251..255]`，按 worker 累加 |

**一对文件里的 define 都要改**，重编后这些值会作为伪算子出现在 `DSPOpType` 表里。报告吞吐前记得改回关闭。

怎么读：

- **dmstart 计数器的存在意义**是回答"这个 DMA 等待是按字节收费还是按次数收费"。用等待时间除以次数：若得到一个稳定的每次耗时、且**字节数变化时它不变**，就是按次数收费，解法是"更少但更大的传输"，而不是"更少的字节"。
- **worker 侧数字是跨 worker 累加的。** 要除以 worker 数才得到墙钟贡献，再与队列线程总和比较，才知道哪一侧是临界路径。
- ⚠️ **有些 slot 存的是计数和 MB，不是毫秒**，而打印器一律当 ms 显示。读之前先看 `execute_command.cc` 的注释。
- ⚠️ **相位 profiler 自身的开销要算进去。** `MM_T0()` 走 `HAP_perf_get_time_us`，约 100 ns 一次；一次 pp512 累计约 234K 次 ≈ **23 ms**，正好就是"各相位加起来对不上总时间"的那一块。**profile 构建的绝对值不能当性能看**，只有相位之间的比例有意义。
- **DSP 侧的 FARF 不一定能到 `adb logcat`**（v79 上就到不了，`-b all` 也不行）。优先用 profile slot。

## 双架构（v79 / v81）

`info().hvxArch` 和 `info().maxThreads` 在 host 侧可用；arch 值源自 DSP（`commu.cc` 把 `__HVX_ARCH__` 写进 info），因此它报告的是 **FastRPC 为这颗芯片加载的那份 skel**。查询失败读到 0 —— 按"更旧/更安全的路径"处理。

- **逐算子的 v79/v81 耗时比集中在 1.1–1.3×**，即**优于** 6 对 8 线程本身预测的 1.33×。明显偏离这个区间才是"该 arch 有专属问题"的信号。
- **绝不能让工作划分数量与线程数撞上。** 某条路径恰好切成 8 个任务：在 8 线程设备上一波跑完，6 线程设备上变成两波、尾波只有 2/6 占用率 —— 足以吃掉一个 30% 的算法收益。对每个划分都要问：**这个数是不是偶然等于某个硬件参数？** 任务数要远多于线程数；开销不均时（因果掩码）**必须先派发最贵的**。
  反过来，本来就能填满整波的划分再细拆只会增加开销 —— 无条件细拆曾让某算子慢 3.5%。
- **一台设备就能验证 arch 假设。** 临时把 `worker_pool.cc` 的 `g_max_num_workers` 钉死成更窄设备的线程数。若 N 与 N-1 之间出现悬崖、而 N-1 与 N-2 几乎相同，是**波次量化**；平滑缩放才是单线程性能差异。
- **有些 arch 分支是真实硬件约束**（v79 的 HMX 只支持共享锁、v81 是独占，见 `hmx_mgr.cc`）；**另一些是历史遗留、值得重测** —— 一条被 `__HEXAGON_ARCH__ >= 81` 门控的 prefill 路径曾把五项优化整个从 v79 的构建里编译掉，而 host 仍在为它们分配 workspace。**旧结论被后续优化改变时要重测，不要沿用。**
- **权重布局只生成一种，要在重排之前就决定**，让用不到的布局根本不被生成 —— "先生成再释放"曾占掉模型加载时间的 5.4%。

## 本后端已实测的瓶颈模式

想新理论之前先对照这些，每一条都在这里至少测到过一次。

### 归因方法

- **动手优化某个算子之前，先用 `MNN_GPU_TIME_PROFILE` 看它在整张图里占多少。** 相位 profiler 回答"算子内谁最慢"，但**算子间的占比得先看一眼**，否则会把力气花在次要项上。注意 `DSPOpType` 计时是跨 run 累加的：跑 `-rep 1` 和 `-rep 2` 相减才得到单次的值（也顺便证明哪些算子属于 decode —— 它们在两次之间不变）。
- ⚠️ **任何「占比」都不是时间占比的代理量 —— MAC 数、指令条数、字节数都不是。** 三次栽在这上面：线性注意力的 GEMM 占 MAC 64%，消融只降 26.7%；权重 DMA"只用了 1 个引擎"，搬进 worker 后 4B 只 +1.19%；Path A 的置换占每 tile 指令数 27.6%，但只占该相位时间 14.6%（它们是便宜的单周期 lane 操作，vlut16/vmpy/store 贵得多）。
- **量天花板的办法是消融：把目标整段旁路掉**（保留写入，写零或写 1.0f 互证，确保循环次数不变），只比墙钟。输出必然是错的 —— 这同时是"库确实换上了"的正向对照。几分钟换掉几轮构建。
  - **要从算子入口 return，不要删它内部的某一层。** 曾只删队列线程那两次 HMX 调用就外推出"attention 天花板只有 12%"，而从入口 return 量到的真实成本是 46.5%。
  - **同一个优化在不同模型上的天花板能差一个数量级**（2B +7.06% 对 4B +1.19%），要在**缺口最大的那个模型**上量。
  - **看到一个相位占比很高时，先问它是不是已经打在物理上限上了。** 长文本 decode 的 attention 占 61.5%，但它是 270 MB/token ÷ 12.71 ms = 21.2 GB/s 的 DDR 饱和 —— 换 FA、改 softmax、做流水都动不了它，只能减字节。
- **按形状拆开计时再动手。** 把 GEMM 的时间按 (N, K) 拆开（临时插桩）比看总相位表有用：4B 的 26% 缺口里七成来自 down_proj 一个形状，其余形状已经到 76–79%。
- **判断一个 worker 相位是"单线程指令效率差"还是"在抢 HVX 单元"：把并行度压到 1，看累计忙时。** 反量化 8 chunk 累计 258 ms、压到 1 chunk 后单线程 296.5 ms —— 只差 15%，说明 8 路并行接近线性，慢在单线程指令效率。反之若单线程耗时接近累计忙时的 1/8，才是争抢。
- **搬运修好之后瓶颈会转到计算，反之亦然。** 每接受一个改动都要重跑相位分解；**相位级大幅收益但墙钟收益很小，说明临界路径已经不在那里了**。
- **性能常量的最优值依赖于别处的相位耗时，改完一处要回头重扫相关常量。** q-block 32→64 曾被否证，softmax 变快之后同一个权衡反号、变成 +2.7%。

### 流水与并发

- **先比两侧的忙时再决定把工作挪给谁。** 把工作从忙的一方挪给另一个忙的一方不会有收益：异步输出存储让主线程省下 9.65 ms，原封不动变成 worker 的工作，而 worker 本来就在满负荷反量化 → 净零。
- **在异步 DMA 两端之间的 HVX 工作不要去优化，先确认它没被传输时间盖住。** attention 的 score 写回占队列线程 24%，整个跳过它零收益 —— 它发生在 `dmstart(wait_after=0)` 和等待之间，本来就藏在传输时间后面。同理主线程在等 worker 时顺手做的 store / shuffle 是免费的。
- **多个 worker 漏斗式共享一个队列线程时，它们等的是该线程的占用率** —— 所以减少队列忙时的收益会按 worker 数放大，即便队列不是最忙的那一侧。
- **"等整块 DMA 完再 fork 计算"会让 DMA 和计算完全串行。** 应把 DMA 拆成按 tile 分块的有序链，worker 轮转取 tile、用 `dstate` 等待自己那块。
- **把工作交给 worker 不是免费的。** 当这份工作本来就顶在 DDR 带宽上限时，握手就是纯增开销：inline 写回打败了提交给 worker 的同一份写回。
- **worker pool 要有足够的槽位和 FIFO 语义。** 原实现只有 workers+1 个槽、按最低位取：排 ~20 个 job 时提交会阻塞主线程，而且回填到低位槽的新 job 会插到旧 job 前面 —— **kernel 无法依赖提交顺序表达依赖**。现在是容量 64 的环形队列。
- **主线程给 worker 派活的时间就是 HMX 时间**（每 job 约 0.5 µs，4B 一次 prefill 派约 39K 个）。交给一个 driver job 能还给 HMX，但**要有形状门限**：mp=4 时一个 panel 的 HMX 只有约 11 µs，盖不住交接延迟（pp128 −1.3%）；mp ≥ 8 才开（+8%）。
- **流水类改动要在小 M 上测，大 M 会把它藏起来。** 同一个双缓冲闸门漏洞在 pp512 值 +1.8%、在 pp2048 完全没有收益（M=2048 时主线程每个 panel 的 HMX 是 4 倍，反量化本来就被掩盖）。

### 内存与搬运

- **DMA 的成本由每次传输的固定延迟主导，而非字节数**（约 0.75 µs/笔）。把大量小传输合并成一次 2D 传输。但**先用 dmstart 计数器确认是按次数还是按字节收费**：attention 的 V 权重一笔 64 KB、1.13 µs，几乎全是传输时间，链成一条只有 +0.33%。
- **user DMA 是 DDR↔VTCM 的搬运器，源在 VTCM 时它不工作。** 把 attention 的整个 per-worker workspace 搬进 VTCM 后，activation load 读到垃圾 —— 模型**全速输出 `!!!!`，pp512 还"涨"了 9.4%**。算错了所以快，不看输出只看读数会直接当成大胜。
  **判据：这块 buffer 是被谁碰的。** 只被 HMX store / HVX load 碰的可以进 VTCM；被 DMA 读的必须留在 DDR。
- **按 token 顺序流式读 DDR 时，先用一次 `l2fetch` 覆盖整个区间。** LinearAttention 的 conv 加一条 `l2fetch` 后该阶段 1180 → 658 ms（worker 累计），pp512 +10%。反量化 job 开头预取本 chunk 的 scale 区间（在等 DMA 之前发）：`MMW_DEQUANT_BUSY` 315 → 155 ms，2B +16.2%，两行代码。**流式读 DDR 的小块数据（scale、bias、表）一定要预取，否则 in-order 核在依赖链上等 DRAM。**
  ⚠️ 但**预取不是无条件的**：激活 shuffle 加 l2fetch 反而慢 1.7 ms，因为它读 392 MiB / 18.4 ms = 21 GB/s 聚合、已经是 DDR 读的合理水位。**先把 per-worker 忙时换算成聚合带宽再决定** —— 单看"3.4 GB/s per worker"会误判成延迟受限。另注意**同一线程发起新的 l2fetch 会取消上一条**。
- **多趟 DDR 遍历会藏在看起来很朴素的辅助函数里。** 某个 softmax 把 score 读两遍、输出也读两遍（每元素 10 字节 DDR），整行暂存进 L1 后降到 4 字节，算子 −9.4%，结果逐位一致。
- **中间往返压在 L1、只让首尾碰 VTCM，是对的设计。** 反过来做（消掉 L1 scratch、直接用 VTCM 当中间缓冲）全线变差 0.7–3.1%：访问次数从"2 次 VTCM + 4 次对齐 L1"变成"5 次非对齐 VTCM"。
- **不满一个槽位的写入会静默退化成 read-modify-write**，触碰 2 倍内存。攒到能配对再写。
- **分两遍写同一块 VTCM 的后处理，代价是双倍流量。** 反量化的非对称分支先存 `q*scale` 再读回来加 offset，有效输出 2048 B 背后是 6144 B 实际流量。融合成一遍。
- **两阶段算子如果中间结果要过一趟 DDR，看能否按归属融合。** LinearAttention 在 Hk == Hv 时每个 conv 通道块只属于一个 head，于是直接把 conv+SiLU 写进 VTCM 的 Q/K/V。
- **互逆的两步相邻时要融合。** 布局转换的最后一步打包 nibble、反量化的第一步又拆开，中间还过一趟 512 字节的栈 —— 融合后该相位 238 → 215 ms，结果逐位一致。
- **写回/搬运相位可能是 issue-bound 而非带宽受限。** 症状：换成非临时存储（`:nt`）毫无变化。解法是提高并行发射或减少每字节的指令数，而不是加 cache 提示。
- **在一个算子里复用缓冲之前，先把每块的读写区间都画出来。** 让 softmax 的 P 原地盖在 S 上值 pp2048 +12.5%（activation 区首次装进 VTCM），但代价是两个原本 alias 到 `scores` 的缓冲必须搬走。

### HVX / HMX kernel

- **活跃的 `HVX_VectorPair` 太多会压爆寄存器文件，编译器就放弃软件流水。** 反量化原来一次展开 4 个输入向量 = 16 个向量寄存器，依赖链整条暴露；改成一次展开 1 个后该相位 258 → 239 ms。**展开度不是越大越好，先数一遍活跃的 VectorPair。**
- **地址本来就对齐就不要用 `vmemu`。** 按 `(((uintptr_t) a) & 127) == 0` 在外层分一次支，该相位 120 → 108.5 ms。
- **HVX 刚写进 VTCM 的数据，再用标量逐个读回会严重 stall。** decode GEMV 构建 `a_splat` 时 224 次"标量 load + vsplat"每次约 36 ns；改用 `vdelta`（控制字 `(k & 0x7c) ^ 4j`）在向量域广播后降到 0.65 µs，逐位一致。水平归约同理：先 vror 树归约，最后只读回 1 个标量。
  ⚠️ 但**混用不同来源的 lane 编号前先单测**：`la_widen` 产生的 fp32 lane 顺序不是内存顺序，和按内存序做的标量转置混用会得到乱码。
- **32×32 fp32 转置**：5 轮 `Q6_W_vshuff_VVR(row[r+16], row[r], -4)`，结果写到 (2r, 2r+1)，每轮把 10 位 (行,列) 地址循环左移一位。每块 80 次 vshuff，替换 4096 次标量读写。
- **HMX 是异步的：`MM_HMX_US` 量的是"发射 + 等硬件算完"。** 而且**真实的 HMX 时间有一部分落在 `MM_STORE_INLINE_US` 里**（store 读输出 tile 时等 HMX 算完），所以看 HMX 占比要看 `HMX + STORE`，或用微基准的单价 × 乘积次数。
- **降精度换速度之前，先确认被省掉的指令真的在关键路径上。** softmax 的分母改 qf16 累加：溢出分析全部成立，但实测**输出乱码而且还更慢** —— qf16 是给单步中间结果用的格式，32 项累加就不行；而省掉的 4 条是便宜的 convert，关键路径是 exp2 那条 ~20 条的依赖链。
- **一趟便宜的循环比把它的工作塞进一趟贵的循环更快。** 把 softmax 的归一化折进 O 的散出（列数窄 16 倍）反而 −3.5%：pass2 的写目标从 L1 变成 VTCM，卡住了它那条很长的 exp2 流水。
- **别在每个 tile 里做除法。** `(k * nblk) / kp` 是 64 位除法、在 Hexagon 上是库调用；改成按 scale 块外层遍历，worker 反量化忙时 140 → 95 ms。
- **反量化不要先重排成 HMX 布局再查表。** vrmpy 布局里一个 oc 的两组 k-pair 同在一个 halfword，把源和它的 `>> 4` 按 halfword 交织（`Q6_W_vshuff_VVR(v >> 4, v, -2)`）后，`vlut16` 的两路输出直接就是 HMX 要的 k-pair 向量：4 条指令对原来的 12 条，62 → 38 ns/tile。
- **HMX accumulator 只有 fp16 读出路径**（`hmx_utils.h`，全库唯一）。要把某段 fp32 HVX 计算搬上 HMX 时，先确认它的下游本来就要降 fp16，否则误差会沿迭代累积。
- **`hmx_queue` 的 mutex 是正确性前提**（HMX accumulator 和 scale 区都是隐式单例）。绝不让 worker 直接 `hmx_unit_acquire()`；用 `hmx_queue_begin()` / `hmx_queue_end()` 包住并行段，**必须覆盖所有提前返回路径**，不配对会让下一个算子在 v81 的独占锁上死锁。每个 callback 开头要重设 column/output scales —— scale 区是隐式状态，上个算子的残留会污染结果。队列线程栈只有 32 KB，tile 缓冲要从 VTCM 指针传入。

### VTCM

- **`vtcm_seq_alloc` 现在会检查边界**（越界夹回区内并让命令组返回 `AEE_ENOMEMORY`，错误码带上越界的 `DSPOpType`），但它只覆盖**分配**、不覆盖**使用** —— 线性注意力和 matmul 内部都是从 `vtcm_base` 手工算指针，那些越界它看不见。**加宽任何 staging buffer 之前仍要手算一遍 VTCM 占用。**
- **任何增加 VTCM 占用的改动，都必须在同一个提交里改 host 侧的 sizing。** prefill 权重 staging 双缓冲曾**导致设备重启**：host 已按"恰好一份权重填满 VTCM"确定 `np`。**最好的写法是让 host 估算器和 DSP 分配器调同一个函数**，两边就不可能漂移。
- **给 kernel 留的 VTCM 要按形状算，不能用一个常数。** 原先留固定 2 MB，理由是"SV kernel 峰值约 0.8 MB"—— 那个数按 K=512 算，而 SV 的 K 是 KV 长度：seq 512 要 0.8 MB、2048 要 2.5 MB、4096 要 5.0 MB。seq 2048 其实早就超了，没炸只是因为 regions 恰好没用满 —— **侥幸安全，不是设计安全**。
- **per-worker 区要按 `g_max_num_workers` 而不是提交的任务数来算**（`worker_index` 的取值范围是前者）。
- **按形状选参数，而不是用一个常量 —— 因为最优值可以随序列长度反向。** q-group 在 pp512/pp1024 下 g=64 更好、pp2048 下 g=32 更好；`htp_attn_causal_group_q_rows()` 从 64 往下试，取第一个能让两个 VTCM 区都放下的。**两边的关系从"必须相等"放松成"DSP ≤ host 上界"比"共享同一个值"更难写错。**
- **不同用途的区要分开判断能否放下。** 一整块搬的写法在 seq 2048 下超预算就整个回退 DDR，长序列一点收益都没有；拆成"score 区优先、act 区放不下就留 DDR"之后 pp2048 +13.2%。
- **图算子按序执行，所以算子期间 VTCM 归它独占**，不需要 `vtcm_manager_reserve_area`。**但"VTCM 是 attention 和矩阵乘的零和资源"只在三段式结构下成立** —— 前提是 scores 要跨三个阶段、两个线程存活。llama.cpp 把三段合成一个 op，于是每个 op 都敢用一大半 VTCM（`FLASH_ATTN_EXT` 2.71 MB、`MUL_MAT` 6.30 MB），互不重叠。

### host 侧

- ⚠️ **host 往 DSP 内存写数据时，排队中的 DSP 命令可能还要读那块内存（写后读冲突）。** 命令在 host 侧排队、到 flush 才执行，而 `onCopyBuffer` 的 host→DSP 拷贝是立即写的。DYNAMIC buffer 按生命周期复用，两个生命周期不重叠的张量会分到同一段 —— x 的消费者还在队列里，z 的上传就把那段覆盖了。症状像"分解路径数值错"（CPU 全对），排查时在 `onExecute` 打印每个输入输出的 `(fd, offset)` 一眼可见。修法：上传前若有排队命令**读或写**目标区间（`HexagonRuntime::pendingCommandTouches`）就先 flush。
- ⚠️ **后端缺某个 op 的 execution 时，Geometry 分解留下的 region 在 pack=64 下可能是真实置换，而通用 C4 快路识别不出来。** Hexagon 没有 `GatedRMSNorm`，`GeometryGatedRMSNorm.cpp` 留了一个"NCHW 视图下是恒等拷贝"的 region，但两侧 channel 不同（128 对 4096）→ pack=64 下内存序不同，18 个层全部掉到 NCHW staging（unpack → blit → pack，一次置换拷贝三趟）。
  **修法是在后端的 Raster 里加专用匹配器，直接从形状构造 region，不要去做 stride 模式识别。** packed 布局是 `[C/pack][B][pack]`，当 `C1 | C2` 时把 `p` 拆成 `ph=p/nb1`、`pl=p%nb1`（`nb1=C1/pack`），正好三轴：
  ```
  size = (C2/C1, nb1, B2)    src stride = (1, B1, C2/C1)    dst stride = (nb1*B2, B2, 1)
  ```
  收益 2B +1.5% / 0.8B +2.5%。**纯数据搬运所以验收门槛是 greedy 输出逐字节一致**，不是 cos_sim。
  **两条可复用的做法**：① 匹配器的条件写死（area==1、两侧 channel 都是 pack 整数倍、`C1|C2`、`B1·C1==B2·C2`、offset=0 全覆盖），宁可漏不可宽；② 在落到 fallback 的分支里打印 region 做诊断 —— 2B 每方向恰好 18 个、0.6B 零个，这同时证明匹配器没写宽。
  ⚠️ **推导布局映射前先从代码里确认维序，再逐点验证**：我第一次推导把它当成恒等映射，因为假设 packed 是 `[N][C/pack][pack]`，实际是 `[C/pack][N][pack]`。
- **权重的分配对齐会影响吞吐。** 权重整块 DMA 进 VTCM，它在 chunk 内的起始偏移决定跨几个 DSP MMU 页。`EagerBufferAllocator` 的 align 从 128 B 提到 64 KB，Qwen3.5-0.8B tg128 **71.9 → 75.6**，单调改善。**改变分配模式的改动要连带检查对齐**，否则会把"对齐退化"误读成"改动本身有问题"。
- **StaticModule 会用空的 inputs/outputs 预创建 Attention 和 LinearAttention**（`backend->onCreate({}, {}, op)`），好让 prefill 和 decode 共享状态。creator 必须只根据参数决定是否接管，张量形状检查放到 resize 里；否则该算子会显示 `inputs=0` 并整体回退 CPU。
- **权重重排结果可以缓存到磁盘。** 首次加载把重排后的权重写进 `tmp_path`，之后直接 pread 进 rpcmem。header 要校验 magic/version/ic/oc/blocks/asymmetric/kind/**hvxArch**/文件长度（hvxArch 不校验会让 v79 的缓存喂给 v81）。v81 加载时间：0.6B 278→201 ms、2B 868→595、4B 1817→1494。
- **`Llm::tuning` 对 Hexagon 是纯开销**（那个 hint 只被 Metal/OpenCL 读），跳过后 4B 的 `llm_demo` 总墙钟 4.83 → 3.83 s。

## 已否证方向（不要重做）

以下全部实现并测量过，别再重新推导一遍。

- **prefill GEMM 对调循环顺序（N 外 ↔ M 外）**：净 **+241 ms**（4B）。`MM_ACT_SHUFFLE` 确实 294.3 → 40.4，但 `MM_DEQUANT_WAIT` 35.5 → 450.1。失败的原因不止"权重 ×mblocks"，更大的是**掩盖能力下降**：N 外层时一个 panel 的反量化被 `mp=16` 个 ox 的 HMX 盖住，M 外层只剩约 5 个。要动这里必须先有 (mp_chunk, np_chunk) 的联合求解 + 更深的权重流水，**单独对调循环没有意义**。
  （按行窗口是另一回事，已成功：host 按行切窗口、每个窗口的激活一次放进 VTCM + 权重双缓冲，4B pp512 +12.5%、pp2048 +37%。早先"M 外层窗口更差"的失败是因为窗口只有 1 个 row tile → 权重反量化 16 遍。）
- **权重 DMA 串成一条链交给主线程引擎**：小 panel 上快 6~7%，但**它会挂死，已两次否证**。引擎忙时 `dmstart` 静默失败，然后所有 worker 死等描述符的完成位。加 `dma_wait_for_idle()` 也只是把挂死推后：v81 全好、v79 的 pp128/pp512 也好，但 2048 长 prompt 仍必挂。**这条路线不要再试。**
- **宿主侧预存 HMX 布局的权重以消除 Path A 的置换**（权重多一份，0.6B 约 +320 MB）：天花板只有算子 −3.9%，不值得。
- **压缩 Path A 的字节置换串**：8 个候选全部不等价（用"当前实现作 ground truth + 4 组伪随机探针在 DSP 上比对"一次性枚举），而且收益上限本来就只有约 1.5%。根因是目标置换**跨向量**，单条 shuffle/deal 给不出来，`vdelta` 只能在单向量内置换。
- **用算术代替 `Q6_Wh_vlut16_VbVhR_nomatch` 做 int4 → fp16**（映射是线性的 `n - 8`）：两种指令组合都更慢（+18% / +6%），**vlut16 是对的**。⚠️ 剥离实验里把 vlut 去掉后 4 个 store 写同一个值被编译器合并，由此得出的"vlut16 约 17 周期"是错的。
- **给 block 量化路径打开 `async_output_store`**：算子 101.3 → 105.3 ms。`MM_STORE_INLINE` 确实从主线程消失，但"未被相位覆盖"的部分同步涨了 10 ms。**两条流水（权重双缓冲 / 异步输出）互斥是设计选择，不是遗漏** —— 放宽条件实测净零。
- **权重反量化改成余数均摊让全部 worker 都有活**：算子 101.4 → 104.4 ms。chunk 数同时切分了权重 DMA 链（`MMW_DMA_SPIN` 22 → 35 ms），而且多出来的自旋 worker 抢内存端口。**宁可少几个大 chunk。**
- **批量提交 worker job**（一次锁 + 一次 `qurt_sem_add`）：提交时间降 1.5 ms，但 8 个 worker 同时被唤醒抢 DMA、等待等量增加，净零。
- **让派发线程自己反量化第一块而不是等 worker**：持平。
- **prefill GEMM 用非临时（`:nt`）存储**：无收益（但由此得出 issue-bound 结论）。
- **prefill attention 的 `dmlink` 连续链 DMA 解耦**：实现正确，收益 0 ms。
- **提高因果分组的 Q 行数（64 → 128）**：**三次否证**（但往下降在长序列是对的）。丢掉的并行度比省下的 DMA 更贵。
- **删掉 attention 里 `valid_end <= 256` 的特例路径让它走通用路径**：无收益。host 的 `HEXAGON_KV_PAGE_SIZE` 正好也是 256，两条路径本就等价。**读到"这条特例看起来更差"时，先把相关的 size 常量代进去算一遍实际走到的分支。**
- **消掉 fp16 softmax 的第三趟**：一点没变。按指令数 pass1(max)≈7、pass2(exp)≈67、pass3≈4，**第三趟只占 5%**。
- **长序列 softmax 的 bound 没定位**：删掉整个 exp 多项式（62% 的指令）只有 +0.3%（pp512 上有 +3.6%）；消掉 `__fp16 stage[2048]` 栈 scratch 全线变差。下次别再猜，用 PMU 计数器或把并行度压到 1。
- **decode attention 的"提交开销"是假目标。** `-n N` 从空上下文开始，那 1.27 ms/token 基本都是与 seq 无关的固定开销，**拿它和 llama.cpp 的 0.71 ms 比，比的是框架开销**。实测长上下文 decode 两边差 0.3%（都被 KV 带宽卡住）。三个假设（排队延迟、worker_pool dispatch、队列往返）全部否证，45 µs/层的去向仍未定位。**decode attention 不要再碰了。**
- **"多 worker 在 decode 时是负担"也是错的**：t=1 对 t=8 的 +5% 是离群值，交替复测后稳定差异 0.3%。
- **为队列线程预留一个 HVX 槽位**（怀疑超额订阅）：否证，耗时在各 worker 数下平坦。
- **调 `weightMinAllocSize` 解决 DSP 内存上限**：只是在改碎片，碰不到约束（见下面"低 4 GB"）。按需精确分配只省 252 MB。**"16 MB 撞条目数上限"这个旧结论也是错的**，DSP 侧 `MAX_MMAP_CACHE_SIZE` 是 512。
- **`hvx_my_inv_vqf32_vsf`（hvx_math.h）对 (1,2] 范围的输入会产生 NaN**，全库没有别的调用者。改用 minimax 线性初值 `(24-8d)/17` 加 2 步牛顿迭代（误差 1.2e-5）后正确。**用这个头文件里没人用过的 helper 之前，先单测它。**
- **worker 内把累加器放在数组里、在内层循环里按下标更新（`acc[j] += ...`）**：结果是 inf 或乱码；改成外层循环 j、累加器放寄存器变量就正确。decode GEMV 把单个 `vrmpyacc` 累加链拆成 4 个独立累加器同样变慢且乱码（疑似寄存器压力溢出到 32 KB 的 worker 栈）。
- **用运行时 K 作下标的 HVX 数组**（conv 的 `w_lo[LA_MAX_K]` 和 history）会放在内存里，每个 token 重新 load/store 全部历史；把 K 做成模板参数完全展开后能进寄存器。另外 **`_Alignas(128)` 的栈上数组被 inline 进很大的函数帧后不可靠**，这类 staging 应由调用方传入 scratch 或 VTCM 指针，并把函数标成 noinline。

### 与 llama.cpp 对标时已排除的三条路

`ggml/src/ggml-hexagon` 是目前最成熟的同类实现。**这三条看起来最像差距来源的路都已排除，不要再重复调查**：

- **HMX int8 路径**：它的 HMX 也只有 fp16 一条路（全仓库 `mxmem` 只 6 处，全是 `.hf`，`cvt_ub_acc` 零命中）。
- **反量化**：它也必须做 int4 → fp16，用纯算术，**实测我们的 vlut16 更快**。
- **功耗 / 频率**：本工程 `src/dsp/power.cc` 与它的四连调用逐项等价且更激进（corner 全 MAX、`dcvs_enable=FALSE`）。不是频率问题。

还有一条**估算与实测吻合但机制不对**的教训：它的 `MUL_MAT_NX`（多个 matmul 共享同一激活）看起来正好对得上 7.8 ms 的矩阵乘差距，用它自己的 env 开关 A/B 实测只值 **2.35 ms**；而本工程的 `g_act_stage`（跨命令缓存激活 staging）已经拿到 **+2.9%**，比它更多。**对照别人的实现时，先确认自己是不是已经用别的形式做到了**；对方有 env 开关时，花一次 A/B 比做任何推算都值。

值得照搬而尚未做的（已读过源码）：**VTCM 布局做成 host/DSP 共享的一个函数**（它的注释写明踩过和本工程一样的坑：*"previously they were duplicated formulas in two files and drifted"*，header 刻意不提 `HVX_Vector` 以保持 host-includable）；**分块参数由 host 按标定代价模型搜、通过 kparams 下发**（除法预转 fastdiv，DSP 侧只读不算）；**op 内 KV 块级流水用队列 FIFO 语义本身做同步**（push 顺序即依赖顺序，无额外 fence）。

## DSP 可用内存的真实上限：低 4 GB 物理地址

**`fastrpc_mmap` 能映射的物理内存只有低 4 GB，和虚拟地址空间、映射条目数都没关系。** 这条曾被误诊三次（"条目数超限"、"连续地址难找"、"VA 耗尽"）。

症状是**别人**失败：2.2 GiB 的 int4 模型加载全部成功，然后 prefill 期申请 48 MB 激活池时 `err=-2147482112`。权重把低 4 GB 吃光，之后 ION 从高地址分配，而高地址映射不进 DSP。

**判据（不要靠猜，adsprpc 的 verbose 日志直接给答案）**：

```bash
adb logcat -c && <跑一次> && adb logcat -d | grep -iE "static_map_fd_alloc|mmap_validate"
```

成功的映射 `pa` 全在 4 GB 以下，失败的那次 `pa 0x3f6e00000` = **17 GB**。对照 llama.cpp 的 `HTP_OP_MAX_VMEM_DEFAULT = 3355443200`（3.125 GiB），它把预算定在 4 GB 以内正是为此。

**推论，都验证过**：`unmap` 不解决问题（只还虚拟地址，不还 `rpcmem` 的物理页）；调 `weightMinAllocSize` 只是改碎片；**唯一有效的方向是减少 `rpcmem` 的物理占用总量**（量它的办法：在 `HexagonAllocator::onAlloc` 里累加 `mappedSize` 打印）。

## ⚠️ DMA 2D 描述符的 roi/stride 是 uint16_t，超 64 KB 静默截断

`dma_desc_2d`（`include/dsp/dma_utils.h`）的 `roi_width` / `roi_height` / `src_stride` / `dst_stride` **全是 `uint16_t`**，而赋值侧通常是 `size_t` 算出来的字节数。`roi_width` 截断成 0 会让描述符非法，**DMA 引擎 fault，`dmwait` 处崩溃 —— v79 上是 cDSP 挂掉，v81 上整机重启**。

**要搬超过 64 KB 的行，用 1D 描述符链**：`dma_desc_1d` 的 `length` 是 24 位（16 MB）。按"每个 pack 一个 1D 描述符、用 `next` 链起来、一次 `dmstart`"即可。

**它制造过一个"算错了所以快"的假收益**：`src_stride` 截断成 0 后 DMA 把同一行重复读了 `K/64` 次、只搬了应搬数据的 1/144，4B pp512 因此报出 664.4 t/s 并被当成 +32% 收录，**真实值是 550.75**。
**判据**：一个"只改搬运方式"的改动若带来远超带宽改善比例的加速，先验证它搬的字节数对不对，再记收益。

## DSP 崩溃要先抓崩溃报告，不要先读代码猜

`adb logcat` 里有 cDSP 的完整崩溃报告，**这应该是第一步**。曾先猜 VTCM 越界、为此白做三轮 DSP 构建。

```bash
adb logcat -d | grep -B45 'End of Crash Report' | sed 's/.*CDSP0:\[R\]: //'
```

报告里有 `Fault PC`、`load address`、`Bad VA` 和调用栈。把 `Fault PC − load address` 交给工具链（在构建机上）：

```bash
T=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/*/Tools/bin
$T/hexagon-addr2line -e skel.so -f -C -i 0x6266C          # Release 构建无 DWARF，只给函数名
$T/hexagon-llvm-objdump -d --start-address=0x62640 --stop-address=0x62690 skel.so
```

**反汇编比行号更有用**：某次 `Fault PC` 处是 `r6 = dmwait`、紧邻 `call copy_shuffle_...`，一眼看出是 DMA 而不是访存越界，而 `Bad VA ≈ SP+0x300` 正是栈上那个描述符。

## 设备风险

- **4096-token 的 prefill 会硬重启设备**，在干净的历史版本上同样如此。除非专门排查那个 crash，长 prompt 一律封顶到 2048。
- ⚠️ **反复撞同一个 DSP 崩溃会把 cDSP 会话搞坏，之后所有测量都无效。** v79 上连跑 3 次崩溃测试后连 2B pp512 都报 `code 18`，换回已验证的 skel 仍然报 —— 是设备不是库。**每次 DSP 崩溃后都要 `adb reboot`**，再用一个已知正常的小模型确认设备健康，才能读任何数字。
- ⚠️ **中断测试会留下僵尸进程**（`llm_demo` / `llm_bench` 卡在 `fastrpc_wait_for_completion`，仍持有 DSP/VTCM），后续运行会报 `execute_command_group failed with code 18`，很容易误诊成代码 bug。测量前先确认没有残留进程，必要时 `kill -9`（有时要连父进程一起）。
- **崩溃的归因要和"上一次运行做了什么"分开。** **换一台设备复跑 + 用旧库复跑**，是把两者分开的最快办法。
- **发起 user DMA 的线程必须先 `dma_wait_for_idle()`（dmwait），再去等 worker 的信号量。** 若主线程 dmstart 后直接阻塞在 semaphore 上、而 worker 正在轮询 `dstate`，DMA 不会完成，cDSP 挂死后被 watchdog 重启。正确顺序见 `conv1x1_w8a16_sym_per_channel.cc` 和 `matmul_q4block_gemv_i8.c`。
- **worker 在 dstate 上自旋时，必须给发起 DMA 的线程留一个硬件线程。** 若派满 `nworkers` 个自旋 worker，发起 DMA 的主线程会被间歇性挤下 CPU，DMA 链停住，单次 decode GEMV 偶发卡几十毫秒（llm_bench 在 18–130 tok/s 之间随机跳）。派 `nworkers - 1` 后 v79 稳定在 146–150。**测速结果双峰、波动大时，先怀疑自旋等待与线程数的关系。**
- **host 上传会落进被回收的设备内存，而 DSP 可能仍持有该内存的脏 cache 行。** 必须在 host 写入**之前**把目标区间刷回 —— 写入之后再刷会让情况严重得多。
- **`thread_num` / `-t` 改不了 DSP 的 worker 数**（身份行的 `maxThreads` 来自 DSP info）。要试 worker 数只能改 `worker_pool.cc` 的 `g_max_num_workers` 重编 DSP 库。
- 设备掉线后：`adb kill-server && adb start-server`，确认序列号回来之后，才可以对之前那次失败的运行下任何结论。

## Profile 与 Crash 检查

- 查失败前先清日志：`adb logcat -c`；运行后取：`adb logcat -d`
- 搜索失败特征：`rg -i "execute_command_group_profile failed|qurt|sysfatal|fatal|crash|tlb|cdsp.*crash|adsp.*crash|segv|signal 11"`
- 若运行失败、超时、返回 profile RPC 错误，或设备短暂掉线，继续之前先查 cDSP tombstone/ramdump：
  - `adb shell "ls -lt /data/tombstones /data/vendor/tombstones /data/vendor/ramdump /data/vendor/ssrdump 2>/dev/null | head -80"`
  - 只把最新的相关文件拉到本地临时路径检查，搜 `cdsp|adsp|qurt|sysfatal|fatal|crash|tlb|page fault|protection|signal|MNN|htp|fastrpc`，总结后删除。
- 以下一律算失败：`execute_command_group_profile failed with code -2147482610`；qurt/cDSP fatal 日志；测试窗口内新增的 cDSP tombstone/ramdump；测试期间设备 offline；改了 DSP 代码后 `.so` 缺失或过期。
- 有用的 profile 字段：`Hexagon DSP Profile`、`Command groups`、`Command dirty`、`DSPOpType <name> (<id>): <time> ms`、`Hexagon onCopyBuffer Profile`

## HVX/HMX 指引

- HVX 指令细节参考 `~/Download/hvx.pdf`。
- 引入新的 intrinsic 写法之前，先找项目里已有的例子。
- 只在对齐有保证时用 `vmem`；不对齐访问用 `vmemu`。
- DMA/HVX/HMX 的改动要落在"内存流量"和"实测算子时间"上。
- HMX 路径在加宽 tile 或 block 之前，先核对 VTCM 分配大小、tile 数和 descriptor 数。
- 不要假设一个微优化能在 v79/v81 之间通用；针对目标架构构建并测试。
- **改置换/布局类的 kernel 时，先做 DSP 侧探针单测**：以当前实现作 ground truth，用 4 组伪随机探针在 DSP 上比对。置换只错一半的症状是"看着对、cos_sim 0.97"，极难归因。

## DSP DMA-BUF 内存测量

用于验证 Hexagon/DSP 内存占用，或对比 CPU 与 `forwardtype=10`。

1. 模型 `config.json` 设 `forwardtype=10`；除非任务明确要求，保持模型原本的 precision。
2. 用足够多的重复输入让进程活着，便于采样。找到 PID：`adb shell ps | grep <process_name>`
3. 进程还活着时，把三个内存视图都采一遍：
   - `adb shell "awk '/VmRSS:|VmHWM:/{print}' /proc/<PID>/status"`
   - `adb shell dmabuf_dump <PID>`
   - `adb shell dumpsys meminfo <PID>`
4. 尝试系统级总览（`/sys/kernel/debug/dma_buf/bufinfo`），但要明确记录权限/路径失败。
5. 分开报告 `VmHWM`、`TOTAL RSS`、`TOTAL PSS`、`dmabuf total`；需要时用 `VmHWM + dmabuf total` 近似"进程可见的 DSP 压力"。

## 安全检查清单

- 目标模型通过正确性检查，例如 `cos_sim` 可接受。
- 目标 demo 输出符合预期且不 crash。
- greedy 解码输出与**该设备的**参考文本逐字节比对过。如果变了，必须已经弄清原因并明确说明 —— 这个后端里大部分 kernel 工作都应当是数值保持的，**输出有差异是一个发现，不是一个细节**。
- 报告了 `DSPOpType` 目标耗时，并与明确的基线对比，且**基线是改动前能达到的最好配置**。
- v79 和 v81 都构建并测量过；只覆盖其中一个时，说明理由。
- 报告的差值超出测量设备的噪声带，且均值旁边给出了离散度；测量前设备已降温。
- 每个测量点都重编过 host；重编的 DSP `.so` 已推送到**被测目录**且 md5 核对过。
- `git diff --check` 干净。
- 没有残留的临时日志、计数器、逐算子 tracing、写死的 worker 数或 env 开关，除非明确要求。gated profiler 已恢复默认关闭。
