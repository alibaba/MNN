---
name: qnn-debug
description: MNN QNN（高通 HTP/NPU）后端的问题定位、修复与算子适配。覆盖运行时报错（1002/6000/1003/6004）、模型转换/算子校验失败、推理结果异常、精度不达标、以及性能问题;并指导为 QNN 新增/适配算子。核心方法:用 MNN2QNNModel --dump_intermediate_outputs 一次性导出 QNN 全部中间张量,与 CPU(fwd=0) 基线逐张量对比,直接定位首个出错算子(旧的 testMNNFromOnnx.py 截断二分已降为回退手段);区分“真 bug/量化/HTP fp16 精度”;新增算子先查 SDK 算子文档(MasterOpDef/HtpOpDefSupplement)再实现。
---

# MNN QNN 后端 定位 / 修复 / 适配 SKILL

> **触发条件**：用户报告 QNN / NPU（高通 HTP）后端任何问题,或要求适配算子。常见表述:"QNN 结果不对/NPU 精度差"、"QNN 报错 1002/6000/1003/6004"、"graphFinalize/graphExecute 失败"、"模型转换失败 / validateOpConfig failed"、"LLM 在 NPU 上输出乱码"、"QNN 性能差"、"给 QNN 加/支持 XXX 算子"、"某算子 QNN 不支持回落 CPU"。

## 概述

QNN 后端有**两条执行路径**,排查前先分清在跑哪条（详见 [reference.md · 两条执行路径](./reference.md#后端内部机制定位时必须知道的)）：

- **在线 finalize 路径**：加载普通 `.mnn`(视觉/CNN),运行时逐算子构图 → `graphFinalize` → 整图执行。入口多为 `ModuleBasic`。报错码常见 **1002(finalize)/6000(execute)**。
- **离线预编译路径**：LLM 经 `llmexport.py`→`generate_llm_qnn.py` 生成预编译 QNN 二进制,`llm_demo` 加载运行。报错码常见 **1003/6004(运行时 IO 与图定义不一致)**、转换期 **validateOpConfig failed**。

本 SKILL 覆盖三类工作,可组合：

- **A. 数值/精度 & 报错定位**(主线)：**建立可信基线 → dump 全部中间张量比对定位首个出错点（回退:截断二分）→ 读误差模式并数值验证 → 区分真 bug / 量化 / HTP fp16 精度 → 打桩定位到代码**。
- **B. 新增/适配算子**：**先查 SDK 算子文档(MasterOpDef/HtpOpDefSupplement/SupportedOps) → 照现有算子模板实现并注册 → 二分探测验证**。见 [新增 / 适配 QNN 算子](#新增--适配-qnn-算子)。
- **C. 性能定位**：开 QNN Profile → 找过多 quant/dequant、数据搬运、CPU fallback → 优化。见 [性能定位与优化](#性能定位与优化)。

### 问题分诊（先按症状选路径）

| 症状 / 错误 | 路径 | 去哪一节 |
|---|---|---|
| `error 1002` graphFinalize 失败 / `could not create op` | 在线 | [步骤 3.5](#步骤-35--graphfinalize--graphexecute-失败时启用-qnn-错误日志) + [新增/适配算子](#新增--适配-qnn-算子) |
| `error 6000` graphExecute 失败 / 输入全零、结果恒为 bias | 在线 | 先试 `shapeMutable=false`(见关键约束) + [步骤 3.5](#步骤-35--graphfinalize--graphexecute-失败时启用-qnn-错误日志) |
| 结果不对 / 精度差 / 与 CPU 不一致 | 在线 | [步骤 0→4](#定位流程)：dump 中间张量比对（回退:截断二分）|
| `error 1003/6004` 运行时报错(离线/LLM) | 离线 | [运行时报错 1003/6004](./reference.md#案例-7--离线llm-路径-10036004输入输出与离线图定义不一致) |
| `validateOpConfig failed` / 转换失败(离线/LLM) | 离线 | [新增/适配算子](#新增--适配-qnn-算子)(同查 OpDef) + [reference 案例 8](./reference.md#案例-8--转换期算子校验失败-validateopconfig) |
| LLM 输出乱码 / 数值异常(能跑不报错) | 离线 | [reference 案例 9 · 量化精度](./reference.md#案例-9--llm-离线推理结果乱码量化参数问题) |
| 性能不达预期 | 两者 | [性能定位与优化](#性能定位与优化) |
| 要新增/适配某算子 | 两者 | [新增/适配算子](#新增--适配-qnn-算子) |

> A/B 常交织:定位到"某 OpType 没实现/实现有 bug"就转 B,补完再用 A 验证。

### 核心原则

1. **先建立可信基线**：任何对比都先确认 `CPU fp32` 与 ONNX 一致（`TEST_SUCCESS` / diff ≈ 1e-4）。参考 txt 与 input.txt 必须是**同一次**生成的，否则会误判（见坑 1）。
2. **一次性 dump 全部中间张量**（首选,替代反复截断）：用 `MNN2QNNModel <sdk> <soc> <arch> <model.mnn> <out> --dump_intermediate_outputs` 生成 debug 版模型,它把**每个** QNN native 激活提升为 `APP_READ` 图输出;在真机上跑一遍即把全部中间张量连同 `manifest_*.tsv` 落盘,再与 CPU 基线**逐张量**比对,一次定位首个出错算子——不必每探一个点就重转一次模型。截断二分（`testMNNFromOnnx.py <model> <tensor>`）保留为**回退手段**（dump 跑不起来、或想快速二分少数点时用）。见 [步骤 1](#步骤-1--一次性-dump-全部中间张量并比对首选)。
3. **看误差模式，别只看 diff 数值**：每通道常数、全常数、转置、NaN 各有明确含义（见 [reference.md](./reference.md#误差模式速查)）。用 Python 对照 bias/权重等**数值验证**你的假设，而不是猜。
4. **三方对比区分“bug”还是“精度”**：QNN-fp16、**CPU-fp16**(fp16 地板,fp32 累加)、CPU-fp32(可信基线)一起对比(`qnn_probe.sh` 一次给全)。真 bug 会在某算子处 **QNN 突跳而 CPU-fp16 不跳**；fp16 精度问题是**平滑累积**、经 Pool/GAP 会**下降**。需要时再拿 OpenCL-fp16 交叉验证。
5. **能定位就停**：定位到首个出错算子 + 数值证据即可下结论；改代码前先 `git blame` 看该处是否近期改动。
6. **闭环沉淀**：每次非平凡定位/修复/适配完成后,若有可复用经验,**主动**按 [复盘:回写 reference](#复盘自动总结并回写-reference) 追加案例——让本 skill 越用越强。

### 关键约束

> **严禁访问 `schema/private/` 和 `source/internal/`。**

> **结果异常先试一招**：若模型输入形状固定，先用 `shapeMutable=false`（`Session_Input_Inside`）跑一遍。QNN 在线路径在 `Session_Input_User`（`shapeMutable=true`，多数入口默认）下有**输入不被拷入、首个算子吃全零**的已知问题；`shapeMutable=false` 零代码规避。详见 [reference.md 案例 1](./reference.md#真实案例)。

> **两套构建目录别搞混**：`build/`（macOS 主机构建，`MNN_QNN=OFF`）只提供 `MNNConvert` 等主机工具；真机 `libMNN.so` 来自 `project/android/build_64`（`MNN_QNN=ON`，NDK arm64）。**改后端代码后要在 `build_64` 里 `make MNN` 并 push 它产出的 `libMNN.so`**，push 错目录的库是最常见的“修了没效果”原因。

---

## 前置依赖（环境准备）

| 依赖 | 用途 | 检测 |
|------|------|------|
| **adb + 真机** | 设备已连接，`/data/local/tmp/MNN/` 下已就绪 `ModuleBasic.out`、`libMNN.so`、`libQnnHtp*.so`、`libc++_shared.so` 等 | `adb devices`；`adb shell ls /data/local/tmp/MNN/` |
| **主机 MNNConvert** | ONNX→MNN 转换（截断后重转） | `ls build/MNNConvert`（或 `which mnnconvert`） |
| **python 环境** | `testMNNFromOnnx.py` 依赖 `onnx onnxruntime numpy` | `python3 -c "import onnx,onnxruntime,numpy"` |
| **Android QNN 构建** | 改后端代码后重编 `libMNN.so` | `grep MNN_QNN: project/android/build_64/CMakeCache.txt` 应为 `ON` |

> 设备上运行都要 `export LD_LIBRARY_PATH=.`（在 `/data/local/tmp/MNN/` 下）。

### 必背：ModuleBasic 命令与后端编号

```
./ModuleBasic.out <model.mnn> <dir> <runMask> <forwardType> <loops> <threads> <precision>
```
- **forwardType**：CPU=`0`，OpenCL=`3`，**QNN=`5`**（QNN 注册为 `MNN_FORWARD_NN`，见 `QNNBackend.cpp` 的 `QNN_FORWARD_TYPE`）
- **precision**：Normal=`0`，High=`1`，Low=`2`；QNN 里 `mUseFP16 = (precision != High)`
- `dir` 内需有 `input.txt`、`input.json` 和 `<outputName>.txt` 参考；比对阈值为 `1%`（`absMaxV*0.01 < diffmax` 判失败）

> 用户给的复现命令示例：`./ModuleBasic.out onnx/test.mnn onnx 0 5 1 4 2`（QNN, fp16）。

### 必背：MNN2QNNModel + 中间张量 dump（定位主力,见步骤 1）

```
MNN2QNNModel <qnnSDKPath> <socId> <hexagonArch> <src.mnn> <outDir> [totalShapeNum] [shape...] [--dump_intermediate_outputs]
```
- 常见 SoC：8Gen2→`socId 43 / arch 73`,8Gen3→`57 / 75`,8Elite→`69 / 79`。
- 加 `--dump_intermediate_outputs` 生成 **debug 版**离线模型(`outDir/<name>.mnn` + `.bin`):它把 QNN 图里每个 native 张量提升为 `APP_READ` 输出;**该标志已烘焙进模型**,运行时无需再传任何 flag,跑一遍即自动 dump。
- **对普通 CNN 同样适用**：MNN2QNNModel 接受任意 `.mnn`,所以在线路径的模型也能用这条离线 dump 通道拿到全部中间张量(在线 `ModuleBasic` 本身不透传 dump flag)。
- 输出落盘：默认写到模型旁的 `qnn_intermediate_outputs/`;设环境变量 `MNN_QNN_DUMP_DIR` 改目录。每次执行产出一个 `manifest_NNNNNN.tsv` + 每张量一个 raw 文件。详见 [reference.md · QNN 中间张量 dump](./reference.md#qnn-中间张量-dump定位主力)。

> 在线路径若要在自己的 runner 里开 dump:`backendConfig.flags = MNN_QNN_DUMP_INTERMEDIATE_OUTPUTS`(`1<<16`,见 `MNNForwardType.h`)。`ModuleBasic.out` 未透传该 flag,故在线模型走上面的 MNN2QNNModel 通道最省事。

### 离线 / LLM 路径的准备（仅 LLM/NPU 预编译模型需要）
```bash
source $QAIRT/bin/envsetup.sh                 # 设 QNN_SDK_ROOT 等
# 1) 导出 NPU 版 MNN(量化):关键 --generate_for_npu --quant_bit 4 --act_bit=16 --sym --smooth --hqq
python3 transformers/llm/export/llmexport.py --path <model> --export mnn --dst_path <out> --generate_for_npu ...
# 2) MNN → QNN 预编译离线图(报错→转"转换期校验失败")
python3 transformers/llm/export/npu/generate_llm_qnn.py --model <mnn> --soc_id=57 --dsp_arch=v75
# 3) 设备运行
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./llm_demo <model>/config_qnn.json prompt.txt"
```
> 除 `libMNN.so` 外,离线路径还需 push QNN SDK 运行库:`libQnnHtp.so`/`libQnnSystem.so`/`libQnnHtpV<arch>Stub.so`/`libQnnHtpV<arch>Skel.so`（来自 `$QNN_SDK_ROOT/lib/{aarch64-android,hexagon-v<arch>/unsigned}`）。

---

## 定位流程

### 步骤 0 · 建立可信基线
```bash
cd build   # 主机构建目录，有 MNNConvert
python3 ../tools/script/testMNNFromOnnx.py ../models/<model>.onnx   # 全模型：生成 onnx/ 参考 + convert_cache.mnn，并做 CPU 自检
```
末尾应打印 `TEST_SUCCESS`。把 `convert_cache.mnn`→`onnx/test.mnn`，连同 `input*.txt/input.json/<out>.txt` push 到设备，跑 `CPU fwd=0 prec=1` 确认 diff≈1e-4。**至此确认模型/转换无误，问题在 QNN。**

> ⚠️ **源 onnx 必须放在 `onnx/` 目录之外**(如 `cp build/onnx/test.onnx build/src_model.onnx`)——否则 `testMNNFromOnnx.py` 把它拷成 `onnx/test.onnx` 时报 SameFileError。多输入模型会生成 `input0.txt/input1.txt/…`,push 时用 `onnx/input*.txt` 覆盖。

### 步骤 1 · 一次性 dump 全部中间张量并比对（首选）
一趟 dump 拿到 QNN 全部中间激活,再与 CPU 基线逐张量比,直接找出**第一个**出错算子——取代"每探一个点就重转一次"的截断循环。

1. **生成 debug 模型并跑一遍**（CNN / LLM 通用）：
   ```bash
   MNN2QNNModel $QNN_SDK_ROOT 57 75 onnx/test.mnn dbg --dump_intermediate_outputs   # soc/arch 按真机改
   adb push dbg/test.mnn dbg/test.bin /data/local/tmp/MNN/                            # 连同 input.txt/input.json/.bin
   adb shell "cd /data/local/tmp/MNN && LD_LIBRARY_PATH=. MNN_QNN_DUMP_DIR=dump ./ModuleBasic.out test.mnn . 0 5 1 4 2"
   adb pull /data/local/tmp/MNN/dump ./dump                                           # manifest_*.tsv + 每张量一个 raw
   ```
2. **读 manifest 逐张量对比 CPU**：`manifest_NNNNNN.tsv` 每行给出 `name / file / data_type / dimensions / quant_encoding / scale / offset`。raw 文件是 **QNN 布局(NHWC)+ QNN dtype**,比对前要按 manifest 的 quant(反量化 `f = (q - offset) * scale`)与布局还原,再和 CPU-fp32 基线(`testMNNFromOnnx.py` 全模型跑出的中间张量,或 `MNNDump2Json` 的张量表)对齐。张量名形如 `t42`,可回查 MNN 张量表定位到具体 op。
3. **找突跳点**：按图执行顺序扫每个张量的 diff,第一个"**输入好、输出坏**"且 QNN 远大于 CPU-fp16 地板的即首个出错算子;随后转 [步骤 2/3](#步骤-2--读误差模式--数值验证) 读误差模式、区分 bug/精度。

> 细节(manifest 字段、布局/量化还原、局限)见 [reference.md · QNN 中间张量 dump](./reference.md#qnn-中间张量-dump定位主力)。

#### 回退：截断 + 二分（dump 跑不起来 / 只想快速二分少数点时）
- 先看图节点顺序：`python3 -c "import onnx;m=onnx.load('src_model.onnx');[print(i,n.op_type,list(n.output)) for i,n in enumerate(m.graph.node)]"`;或 `MNNDump2Json` 看 MNN 侧执行顺序(MNN 名与 ONNX 名可能不同)。
- 用 `scripts/qnn_probe.sh <tensor> [<tensor> ...]` 对中间张量截断→转换→push→**一次并排跑 QNN-fp16 / CPU-fp32 / CPU-fp16**,对节点序号二分找"输入好、输出坏"的第一个算子。
- 脚本已**自动注入 `shapeMutable=false`**(否则 QNN 输入不进去,见关键约束)。
- **陷阱**见 [坑 2/坑 3](./reference.md#常见坑)：换模型前 `rm -f .tempcache`（QNN 图缓存,脚本已带）；单独取 QNN 输出要单独跑 fwd=5 再 `cat output/0_0.txt`（同一条命令里跑 CPU 会覆盖它）。

### 步骤 2 · 读误差模式 + 数值验证
把 QNN 输出与参考 reshape 后用 numpy 比对，对照 [误差模式速查](./reference.md#误差模式速查)。例如“每通道 std=0 的常数”几乎一定是 **conv 收到全零输入 → 输出==bias**，可与 ONNX 里该 conv 的 bias 逐通道核对确认。

### 步骤 3 · 区分“真 bug”还是“HTP fp16 精度”
最简单:直接看 `qnn_probe.sh` 已经并排给出的 **CPU-fp16** 列——它是"行为良好的 fp16 地板"(CPU/OpenCL 的 fp16 都用 fp32 累加器)。某点 **QNN-fp16 ≫ CPU-fp16 且突跳** = 真 bug;**同步平滑增长** = fp16 累积。
需要 OpenCL 做交叉验证时,再用 `scripts/cmp_probe.sh`(设备需 `libMNN_CL.so`)一次输出 QNN 与 OpenCL 的 diff。判据：

| 现象 | 结论 |
|------|------|
| 某算子处 QNN 误差**突跳**、OpenCL 不跳 | 该算子**真 bug**，深挖它 |
| QNN 与 OpenCL **平滑同步**增长，经 GlobalAveragePool/Pool 后误差**下降** | 随机精度噪声，非离散 bug |
| QNN 比 OpenCL 同精度大 **~2.5×/层**并随深度放大 | **HTP fp16 累加**（OpenCL fp16 用 fp32 累加器）|
| QNN `High` 与 `Low` 结果几乎相同 | **HTP 忽略 fp32 请求**，底层纯 fp16（见 reference）|

### 步骤 3.5 · graphFinalize / graphExecute 失败时：启用 QNN 错误日志
如果 QNN 报 error code 1002（graphFinalize 失败）或 6000（graphExecute 失败），需要启用 QNN 内部日志来获取详细错误信息：

1. **启用 log callback**：在 `QNNBackend.cpp` 中找到 `QnnLog_create` 或 log level 设置处，将级别改为 `QNN_LOG_LEVEL_ERROR`（1）或更详细的级别（2=WARN, 3=INFO, 4=DEBUG）。
2. **重编并测试**：`cd project/android/build_64 && make MNN -j8 && adb push libMNN.so /data/local/tmp/MNN/`
3. **查看日志**：运行测试时 grep `QNN_LOG`，关注：
   - `could not create op` → 某算子约束不满足，查 MasterOpDef.html
   - `Wrong number of Inputs` → 输入数量不对
   - `Op creation failure, total_inputs=N` → 检查各 Input 的类型（F16Crouton=fp16, PlainFloat=fp32）
4. **查 SDK 算子文档确认约束**：SDK 根取自编译配置——`SDK=$(grep -i QNN_SDK_ROOT project/android/build_64/CMakeCache.txt | head -1 | cut -d= -f2)`(或 `$QNN_SDK_ROOT`),再进 `$SDK/docs/QNN/`(老版)或 `$SDK/docs/QAIRT-Docs/QNN/`(2.48) 下的 `OpDef/`。同目录 **`HtpOpDefSupplement.html`** 是 HTP 专属约束权威来源,**`SupportedOps.html`** 是各后端支持列表。搜算子名确认输入数量、类型、维度约束。

> 详见 [reference.md · QNN 错误日志](./reference.md#qnn-错误日志qnn-log-callback) 和 [QNN 算子约束查询](./reference.md#qnn-算子约束查询)。

### 步骤 4 · 打桩定位到代码 / 给结论
在 QNN 后端加临时 `MNN_PRINT`（`build_64` make 后 push）常用打点：
- `QNNConvolution::onEncode`：打印 MNN 形状 + `getNativeTensor(inputs[0])->v1.dimensions` 确认喂给 QNN 的张量维度/格式对不对；
- `QnnBackend::onCopyBuffer` / `inputIO`：打印 `usage`、`elementSize` 和拷入的数据，确认输入/输出拷贝是否发生、数据是否正确。

定位后：真 bug 就改代码并**在真机重验**（首个出错点 diff 应回落到 fp16/fp32 级）；属硬件精度则给结论 + 缓解建议（量化、换 SDK/HTP、处理高动态范围层）。

---

## 新增 / 适配 QNN 算子

当某算子在 QNN 后端不支持（`could not create op`、或该 OpType 无 QNN 实现回落 CPU），或用户要求新增算子时,按此流程——**核心是先读 SDK 算子文档,再照现有算子模板实现,最后用二分探测验证**。

### 步骤 A · 先查 SDK 算子定义文档（不看文档就写=大概率违反 HTP 约束）
1. **从编译配置拿 `QNN_SDK_ROOT`**(QNN 后端 CMake 用 `-DQNN_SDK_ROOT=` / 环境变量定义,固化在 CMakeCache),再进它的 `docs/` 找 OpDef 目录(子路径随版本不同)：
   ```bash
   SDK=$(grep -i QNN_SDK_ROOT project/android/build_64/CMakeCache.txt | head -1 | cut -d= -f2)   # 或用 $QNN_SDK_ROOT
   ls "$SDK/docs/QNN/OpDef/" 2>/dev/null || find "$SDK/docs" -iname MasterOpDef.html              # 2.48 在 docs/QAIRT-Docs/QNN/OpDef/
   ```
2. 按优先级读（`grep`/`WebFetch` 搜算子名）：
   - **`SupportedOps.html`** → 目标算子 HTP 到底支不支持、叫什么 QNN 名（MNN 的 OpType 名常与 QNN 不同,如 Interp→`ResizeBilinear`、Deconvolution→`TransposeConv2d`）。
   - **`MasterOpDef.html`** → 该 QNN 算子的**输入个数与顺序、每个 param 的名字/类型(scalar/tensor)、支持的 dtype、rank**。
   - **`HtpOpDefSupplement.html`** → **HTP 专属约束**(fp16-only、axes/rank 限制、量化要求、某些参数必须显式设)。graphFinalize 失败几乎都出在这里。
3. 若 HTP 不支持该算子 → 考虑用已支持算子**组合分解**(如某 Norm 拆成 reduce/sub/mul/rsqrt),或该算子回落 CPU。

### 步骤 B · 照现有算子模板实现
- 在 `source/backend/qnn/execution/` 找一个**最相近**的算子(Conv/Interp/Reduce/Flatten…)照抄骨架:新建 `QNNXxx.cpp/.hpp`,继承 `QNNCommonExecution`,实现 `onEncode`。
- `onEncode` 里三件事:①`createParamScalar/createParamTensor` 建参数(名字**严格**按 MasterOpDef);②`createStaticFloatTensor` 建权重/常量,注意**布局重排**(见下);③`mBackend->addNodeToGraph(...)` 或 `addNodeCommon(inputs, outputs, N)` 加节点。
- 在 `QNNUtils.cpp` 的 `registerQNNOps()` 里加 `___QNNXxxCreator__OpType_Xxx__();`,并在 `.cpp` 末尾 `REGISTER_QNN_OP_CREATOR(QNNXxxCreator, OpType_Xxx)`。
- **务必对照** reference.md 的 **“一类高频 bug：QNN 算子读错 MNN op 字段 / 忽略存储布局”** 一节——新增算子最容易踩:
  - **读错 MNN op 字段**:同一语义可能有"新枚举 + 旧 bool"两份(如 Interp 的 `ctm` vs `halfPixelCenters`),以 schema/converter 实际写入的为准。
  - **NHWC↔NCHW 顺序**:NC4HW4 在 QNN 恒为 NHWC;任何折叠/重排空间维要显式 transpose。
  - **权重布局**:Conv OIHW→HWIO、Deconv IOHW→HWIO,在建常量张量前手工重排。
  - **输入个数**:QNN 与 MNN 不一定一致(如 Resize 只收 1 个)。

### 步骤 C · 编译并验证
- `cd project/android/build_64 && make MNN -j8 && adb push libMNN.so /data/local/tmp/MNN/`（**别 push 错目录**,见坑 4）。
- 用 `scripts/qnn_probe.sh <该算子输出张量>` 截断验证:该算子**输入**应已 OK、**输出** diff 回落到 fp16 级(与 CPU-fp16 同量级)即成功;仍突跳则回步骤 A 复查约束/字段/布局。
- 若 finalize 仍失败,启用 QNN error log(步骤 3.5)看 `could not create op` 的具体算子与各 Input 类型,回 `HtpOpDefSupplement.html` 对约束。

> 真实例子:本仓库已按此法新增 **Deconvolution→TransposeConv2d**(权重 IOHW→HWIO)、修 **Interp** 按 `ctm` 设坐标模式、修 **Flatten** 的 NC4HW4 2D 展平,见 reference 案例 5/6 与约束陷阱表。

---

## 性能定位与优化

QNN 结果正确但速度不达预期时：

1. **开 Profile**：在 `QNNBackend.cpp` 定义 `QNN_PROFILE_OP`(每 op 耗时) 与 `QNN_PROFILE_SUMMARIZE`(汇总),`build_64` 重编 push,跑一遍收集日志（数据仅供参考,用来找明显瓶颈）。
2. **看瓶颈**（按常见度）：
   | 现象 | 含义 | 方向 |
   |---|---|---|
   | Quantize/Dequantize 占比高 | **最常见**:某算子输入/输出缺量化参数,运行时临时量化/反量化 | 补量化参数(见下) |
   | Convert/Transpose 耗时高 | 数据格式频繁转换 | 减少 NHWC/NC4HW4 来回转 |
   | 某单算子异常慢 | 参数配置不优 | 查该算子实现 |
   | 部分算子在 CPU 跑 | CPU fallback → CPU↔NPU 搬运 | 补该 OpType 的 QNN 实现(转 B) |
3. **补量化参数**（最常见修法）：根因常是 `llmexport.py` 校准阶段漏统计某些 tensor(如 Binary 的输入输出)的 scale/zero,导致运行时插入多余 quant/dequant。检查 `llmexport.py` 的校准/observer 逻辑,确保相关 tensor 都收集到量化参数。
4. **验证**：优化后 quant/dequant 数量下降、整体延迟降低,且结果正确性不变。

---

## 复盘:自动总结并回写 reference

> **这是每次任务的收尾步骤,主动执行,不用等用户要求。** 目的:把本次定位/修复/适配中**可复用**的经验固化进 [reference.md](./reference.md),让知识库自增长(QNN 专属版的 retrospective;落点是本 skill 的 reference.md,不是通用 memory)。

### 1. 判断值不值得写（先自评一句）
- **写**:新的误差模式、新的算子约束/字段坑、新错误码根因、一条新的定位捷径、某类 bug 的通用规律、一次多 bug 叠加的完整链路。
- **不写**:一次性环境问题、已被现有案例覆盖的、纯代码实现细节(git/PR 里已有)、仅本模型独有无推广价值的。
- 不满足就跳过,别为凑数写案例。

### 2. 写什么（案例模板,沿用现有风格）
追加到 reference.md「真实案例(续)」区,编号接续现有最大号(`grep '^### 案例' reference.md` 查):
```
### 案例 N · <一句话标题:现象 → 根因>
- **现象**：症状 + 关键数字(哪个张量 diff 从 X→Y / 错误码 / 日志特征)
- **根因**：一句话本质 + 涉及文件/字段/布局
- **定位**：怎么二分/对比到它的(可复现的关键步骤)
- **修复**：改了什么(文件 + 一句话做法)
- **教训/可推广**：这类问题的通用规律(最重要,决定这条经验的价值)
```
并**按内容顺带更新**对应速查表(命中新类别时):误差模式速查、常见约束陷阱、错误码速查、相关代码位置。

### 3. 怎么写（机械步骤,避免重复/膨胀）
1. `grep '^### 案例' reference.md` 看现有编号与主题——**同类问题只更新旧案例**(补一条 bullet),**不新增**重复案例。
2. 追加新案例 N;若引入新的速查表行,同步加表格行。
3. 更新 reference.md **开头 intro 行的案例区间描述**(如 "案例 1–10"→"1–11")。
4. 保持精炼:一条案例控制在 ~6 行内,长推导留在正文、结论进速查表。

> 经验:本 skill 现有的案例 5/6/10 就是这样从一次真实 debug 沉淀下来的——先修 bug,再把"误差模式 + 根因 + 可推广规律"回写。

---

## 已知坑与真实案例

深入的后端内部机制、逐条“坑”、以及本仓库已定位过的真实案例（**QNN 在线路径不拷贝输入**、**HTP fp16 累加导致精度差于 OpenCL**、两条 QNN 执行路径的区别等），见 **[reference.md](./reference.md)**。遇到新问题时，先查那里是否已有同类结论。

## 辅助脚本
- **首选** `MNN2QNNModel ... --dump_intermediate_outputs`（内置工具,非脚本）：一趟导出全部中间张量 + `manifest_*.tsv`,见 [步骤 1](#步骤-1--一次性-dump-全部中间张量并比对首选)。
- [`scripts/qnn_probe.sh`](./scripts/qnn_probe.sh) `<tensor> [<tensor> ...]`（回退）：截断→转换→(自动注入 shapeMutable=false)→push→**并排跑 QNN-fp16 / CPU-fp32 / CPU-fp16**,一条命令二分定位出错点并区分 bug/精度。
- [`scripts/cmp_probe.sh`](./scripts/cmp_probe.sh) `<tensor>`：截断→push→跑 QNN(fp16) 与 OpenCL(fp16) 对比，步骤 3 需要 OpenCL 交叉验证时用。

> 脚本里的路径（工程根、模型名、venv）按实际环境改；`MNN_ONNX` 默认指向 `build/src_model.onnx`（**须在 `onnx/` 之外**）；默认以 `build/` 为主机构建目录、`project/android/build_64` 为设备库来源。