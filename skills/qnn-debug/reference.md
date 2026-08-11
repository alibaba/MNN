# QNN 后端调试参考

QNN 后端内部机制、中间张量 dump（定位主力）、常见坑、误差模式速查、错误码、SDK 算子文档、以及已定位过的真实案例(在线路径案例 1–6、离线/LLM 路径案例 7–9、多 bug 叠加的工作流经验案例 10)。配合 [SKILL.md](./SKILL.md) 使用。

---

## 后端内部机制（定位时必须知道的）

### 两条执行路径（先分清在跑哪条）
QNN 后端有**两套完全不同**的执行路径，输入/输出的喂法不一样：

1. **在线 finalize 路径（`QnnBackend`，逐算子）**：加载普通 `.mnn` 模型，运行时用 `QNNConvolution` 等逐算子构图，`onResizeEnd` 里 `graphFinalize`，`onExecuteEnd` 里 `executeGraph()` 整图执行。普通 CNN/视觉模型走这条。
2. **预编译二进制路径（`PluginExecuteRaw` + `RawExecutorWrapper`）**：模型里含 Plugin 算子引用预编译 QNN 二进制图，`compute()` 里自己 `onCopyBuffer(inputTensor, mRealInputs)` 拷输入。QNN 上的 LLM 走这条。

> 关键区别：路径 2 在 `compute()` 里**主动拷贝输入**；路径 1 历史上依赖 Pipeline 帮它拷——而 Pipeline 对 QNN 是**跳过**的（见下方案例 1）。定位“输入没进去”类问题时先确认在跑哪条路径。

### 张量格式与维度
- QNN Conv2d 等算子期望 **NHWC**。MNN 内部是 NC4HW4/NCHW，`onAcquire` 里对 NC4HW4 输入用 `getNHWCShape` 转成 NHWC 维度登记；`onCopyBuffer`/`inputIO` 用 `CPUTensorConverter::convert` 做实际数据的格式转换。
- QNN 张量的 `clientBuf` 指向 `QNNTensorWrapper::mDataContainer` 的 host（`alloc()` 里分配），**与 MNN tensor 自己的 host 是两块内存**，必须靠 `inputIO`/`outputIO` 搬运。

### 精度
- `mUseFP16 = (precision != Precision_High)`（`QNNBackend` 构造函数）。
- **HTP 是 fp16 硬件**：`QNNBackend` 构造里 HTP 图精度配置 `QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION` 目前**硬编码为 `QNN_PRECISION_FLOAT16`**。实测在 V81 上，即使 `precision=High`（张量声明为 FLOAT_32、图精度改 FLOAT32），结果与 fp16 **几乎一致** → 该 HTP 忽略 fp32 请求，底层仍 fp16 计算 + **fp16 累加**。
- 对比：OpenCL 的 fp16 模式是 **fp16 存储 + fp32 累加**，所以同为 fp16，OpenCL 深网络精度明显好于 QNN。

---

## QNN 中间张量 dump（定位主力）

把 QNN 图里**每个** native 激活提升为 `QNN_TENSOR_TYPE_APP_READ` 图输出,`graphExecute` 后一次性落盘。等价于 ExecuTorch 的 QNN 中间调试器。一趟拿到全部中间张量,取代"截断→重转→再跑"的循环。**仅用于精度调试**——会显著增加图输出、显存与执行时间,发布产物不要带。

### 怎么开
- **离线/序列化 & 通用(推荐)**：`MNN2QNNModel <sdk> <soc> <arch> <src.mnn> <out> --dump_intermediate_outputs`。该标志**烘焙进**生成的 debug `.mnn`(通过模型属性 `dump_intermediate_outputs`,见 `QNNBackend.cpp` 的 `RawExecutorWrapper::compileModel`),运行时无需再传 flag,跑一遍自动 dump。`--dump_intermediate_outputs` 可放在可选动态 shape 参数中间任意位置。CNN 与 LLM 的 `.mnn` 都能走这条。
- **在线路径(自写 runner)**：`backendConfig.flags |= MNN_QNN_DUMP_INTERMEDIATE_OUTPUTS`(`1<<16`,`MNNForwardType.h`)。`QnnBackend` 构造读 `info.user->flags` 决定是否建 `QNNTensorDumper`。**`ModuleBasic.out` 不透传该 flag**,所以在线模型也建议走 MNN2QNNModel 通道,或临时给 runner 加一行 `backendConfig.flags = MNN_QNN_DUMP_INTERMEDIATE_OUTPUTS`。
- **输出目录**：默认 `qnn_intermediate_outputs/`(离线是模型旁);创建 QNN runtime **之前**设环境变量 `MNN_QNN_DUMP_DIR` 可改。

### 输出格式
每次执行产出一个 `manifest_NNNNNN.tsv` + 每个可读张量一个 raw 文件。manifest 列：
`index / name / file / data_type(QNN dtype 枚举) / dimensions(QNN 布局) / quant_encoding / scale / offset`。
- **名字**：MNN 图张量保留 `t42` 之类名字 → 可回查 MNN 张量表(`MNNDump2Json`)定位到具体 op;后端内部产生的 stage 用算子派生名。
- **raw 内容仍是 QNN 布局(NHWC)+ QNN dtype**：比对前必须按 manifest 元数据还原——量化张量先反量化 `f = (q - offset) * scale`,再把 NHWC 转回 CPU 基线的布局,才能和 CPU-fp32/`testMNNFromOnnx.py` 的中间张量对齐。

### 局限
- dump 的是 `graphExecute` 的**完整输出集**(模型输出 + 被提升的中间张量);已 finalize 的旧离线图无法事后提升张量,必须用 `--dump_intermediate_outputs` **重新生成** debug 产物。
- 增加输出/显存/耗时,别用它测性能;定位完用不带该标志的正常产物复测。

> 相关代码：`QNNTensorDumper`(`QNNBackend.cpp`)、`registerDebugTensor`/`mDebugTensorWrappers`、`RawExecutorWrapper::setTensorDump`、工具 `tools/cpp/MNN2QNNModel.cpp`、后端 `source/backend/qnn/README.md`。

---

## 误差模式速查

| QNN 输出相对参考的模式 | 高概率含义 | 验证方法 |
|------|------|------|
| **每通道一个常数（通道内 std=0）** | 该 conv 收到**全零输入** → 输出 = bias | 与 ONNX 该 conv 的 bias 逐通道对比（应等于 bias） |
| 全张量同一个常数 | 输入/权重全零或被广播 | 打印输入 data container |
| 与参考**高度相关但整体偏移/缩放** | 量化 scale/zero、bias 处理错 | 看 quant 参数 |
| **转置/通道错位** | NHWC↔NCHW 维度登记错 | 打印 `getNativeTensor()->v1.dimensions` 与 dimensionFormat |
| **误差≈1.0、且输出是参考的一个"重排"（元素齐全但顺序乱）** | 在 **Reshape/Flatten** 处把 NC4HW4(QNN 存为 NHWC) 按 (h,w,c) 展平，而参考要 (c,h,w) | 该算子**输入** diff 很小、**输出**突跳到 ~1.0；见案例 5 |
| **某算子（尤其大比例 Resize/Interp）处 QNN 突跳，而 CPU-fp16 几乎完美(≈0)** | 该算子**读错了 MNN op 字段或布局**（如 Interp 忽略 `ctm`、conv/deconv 权重布局错）| 对比 CPU-fp16 vs QNN-fp16 在该点的 diff；查该 op 读了 op 的哪些字段（见案例 6）|
| NaN/Inf | 除零、未初始化 buffer、shape 错 | 打点最近算子 |
| 首算子就错、且逐层**平滑放大** | fp16 累加/精度（非单点 bug） | 三方对比（步骤 3）|

> **判 bug 还是精度的快捷判据**：`CPU-fp16` 是"行为良好的 fp16 地板"（CPU/OpenCL 的 fp16 都用 **fp32 累加器**）。在某截断点若 **QNN-fp16 ≫ CPU-fp16**（差一两个数量级、且是**突跳**而非平滑增长）→ 几乎一定是该算子的 QNN 实现 bug；若两者同步平滑增长 → fp16 精度累积（见案例 2）。本判据只需 CPU，不依赖 OpenCL。

> 通用技巧：conv 是线性的，`conv(常数输入) ≈ 输出的空间均值`（边缘受 padding 影响）。“每通道常数 ≈ 该通道均值”提示输入被空间坍缩；“每通道常数 == bias”提示输入为零。

---

## 常见坑

1. **参考 txt 与 input.txt 不匹配**：设备上遗留的旧 `<out>.txt` 可能是用别的输入生成的，导致连 CPU 都“对不上”。**每次用 `testMNNFromOnnx.py` 重新生成 input+参考并一起 push**。
2. **QNN 图缓存 `.tempcache`**：换模型/换精度前 `rm -f .tempcache`，否则可能复用上一个图。
3. **CPU 覆盖 QNN 输出**：`ModuleBasic` 一次 run 会把结果写到 `output/0_0.txt`。若你在同一条命令里既跑 QNN 又跑 CPU，后者会覆盖前者。**要单独拿 QNN 输出，就只跑 `fwd=5` 再 `cat output/0_0.txt`**。
4. **push 错 libMNN.so**：主机 `build/`（QNN=OFF）和设备 `build_64/`（QNN=ON）是两套；改后端后必须 `cd project/android/build_64 && make MNN` 再 push **它的** `libMNN.so`。
5. **`precision=High` 不等于高精度**：HTP 会忽略 fp32（见上）。想验证“是不是 fp16 精度问题”，靠对比 OpenCL，而不是指望 QNN 切 fp32。
6. **随机输入使 argmax 不可靠**：`testMNNFromOnnx.py` 用随机输入，softmax/simcc 类近乎平坦的分布上 argmax 对微扰极敏感，别用它当可用性判据；看相关系数或换真实输入。

---

## 真实案例

### 案例 1 · QNN 在线路径在 Session_Input_User 模式下不拷贝模型输入（结果全错的根因）
- **现象**：某 mmpose 模型（`end2end.onnx`）QNN 结果整体错（diff 0.4~0.8），CPU 正确。二分发现**第一个 Conv** 就错。
- **根因**：QNN 输出**恒等于该 conv 的 bias**（逐通道核对，误差仅 fp16 舍入）→ conv 在**全零输入**上计算。且该问题**只在 `Session_Input_User` 模式（`shapeMutable=true`）下出现**：
  - `Session_Input_User`（`shapeMutable=true`，ModuleBasic 默认）：输入张量靠 `refTensorContent` 共享用户 host，指望 `Pipeline::_copyInputs()` 搬进 QNN；但 `WrapExecution::needWrap()` 对 `MNN_FORWARD_NN` 直接 `return false` → Pipeline 不建 wrap 张量 → `_copyInputs()` 跳过 → QNN 输入 data container 恒为零。
  - `Session_Input_Inside`（`shapeMutable=false`）：`StaticModule::_resize` 里显式 `mInputTensors[i]->copyFromHostTensor(inputTensor)` → `QnnBackend::onCopyBuffer` → `inputIO`，输入被正确送入，**无需任何改动即可跑对**。
- **采用的解决方案（方案3·纯配置,零代码改动）**：让 QNN 走 `Session_Input_Inside` 模式即可正确喂输入——即**用 `shapeMutable=false`**（ModuleBasic 里在 `onnx/input.json` 加 `"shapeMutable": false`;代码里 `Module::Config::shapeMutable=false`)。**验证**（干净 lib、无任何代码改动）：首 conv 692 在 `shapeMutable=true` 下 diff 0.84、`false` 下 0.0011。
  - **代价**：`Session_Input_Inside` 不支持可变输入 shape。对固定输入尺寸的模型（如本例 256×192 mmpose）无影响;若模型确需动态 shape,再考虑下面的代码方案。
  - **注意**：全模型端到端 diff 仍会失败(~0.7),那是**另一个**问题（HTP fp16 累加,见案例 2),与本输入 bug 无关;判断本 bug 是否解决要看**截断到浅层**(如 node0)的 diff,别看全模型。
- **备选（代码修复,未采用）**：在线 `QnnBackend` 自己补输入拷贝——`onAcquire` 记录 `INPUT` 张量,`onExecuteBegin` 里 `inputIO(t,t)`,`clean()` 清空。对两种模式都生效,与预编译 `RawExecutor` 路径 `compute()` 主动拷输入一致;但在 Inside 模式会与 `copyFromHostTensor` 形成一次幂等冗余拷贝。仅当必须支持动态 shape 又要走 QNN 时才考虑。
- **同根的另一种表现（`shapeMutable=true` + error 6000 / 0 输入）**：另一个多输入模型上,`shapeMutable=true` 时不是"吃全零",而是 **graphExecute 报 6000 且 `mInputTensorIndexes` 为空(绑定 0 个输入)**。因为模型输入被 QNN 算子**直接消费**,而这些输入从没被 QNN `onAcquire` 注册(首个消费者是 CPU 上的 Shape/Rank/ConvertTensor 等),`getTensorIdx` 未命中 → 走 fallback 把它们**当常量烘焙**(日志里出现 `Tensor usage is 1.`,即 INPUT 被当 const)。`shapeMutable=false` 同样零代码规避。**记忆点**:QNN 报 6000 且 debug 打印显示 input 个数为 0 / 出现 "Tensor usage is 1" → 先试 `shapeMutable=false`。

### 案例 1b · 为什么不能把 `needWrap` 对 QNN 改成返回 true
- 直接去掉 `needWrap` 里对 `MNN_FORWARD_NN` 的跳过，在 **`Session_Input_User`** 模式下**必 segfault**；`Session_Input_Inside` 模式下反而正常。
- **崩溃定位**：`SIGSEGV @ 0x0`，栈顶 `__memmove_aarch64_nt`，`x1(src)=0x0`、`x2(size)=0x90000=589824=1×3×256×192×4`（正是输入张量大小），栈帧 `MNN::Session::resize()` 内。即在 resize 阶段,通用 `WrapCopyExecution` 对 QNN 输入执行 `memmove(dst, src=NULL, 输入字节数)` —— User 模式下该输入 host 未被物化成通用拷贝所需的普通 buffer,源指针为空。
- **结论**：QNN 刻意不走通用 `WrapExecution`，输入拷贝必须由 QNN 后端自理（见案例 1 修复），不能靠翻 `needWrap`。

### 案例 2 · HTP fp16 累加导致精度差于 OpenCL（非离散 bug）
- **现象**：修好案例 1 后，全模型 QNN 仍不过 1% 阈值（simcc corr≈0.6）；用户指出“同样 fp16，OpenCL 误差没这么大”。
- **数据**（全模型 diff-rate vs fp32 ONNX）：CPU fp32≈3e-4；OpenCL fp32≈2e-5；**OpenCL fp16≈0.26**；**QNN(High/Low)≈0.72**。
- **定位**：逐算子 QNN vs OpenCL 对比——ic=3 的首 conv QNN 反而更好，随通道/深度增加 QNN 以 **~2.5×/层**落后并放大到 8×；无单点突跳；GlobalAveragePool 处误差**下降**（平均抵消随机噪声）。且 QNN `High==Low`。
- **结论**：这是 **HTP fp16 累加**（OpenCL fp16 用 fp32 累加器）的硬件特性，不是某算子的 bug；`QNN_PRECISION_FLOAT16` 硬编码 + V81 HTP 忽略 fp32，MNN 侧改精度配置**实测无效**（故未提交该改动）。
- **缓解建议**：走 int8/int16 量化路径（HTP 原生、精度好）；关注支持 fp32 的 HTP/SDK；该模型激活动态范围大（可达~107）对 fp16 不友好，可对高动态范围层特殊处理。
- **潜在改进（未验证）**：`QnnBackend` 构造里 `mQnnHtpGraphCustomConfig.precision` 硬编码 FLOAT16，静默把 `Precision_High` 降级；在支持 fp32 的 HTP 上应按 `mUseFP16` 条件设 `FLOAT32/FLOAT16`。

---

## QNN 算子约束查询

### SDK 算子文档（新增/修算子、定位 op 报错前**必查**）
QNN/QAIRT SDK 自带一整套算子定义 HTML,是"某算子能不能上 HTP、要几个输入、参数叫什么、dtype/rank 约束"的**权威来源**;`could not create op` / validate 失败也靠它定位。

**先拿到 `QNN_SDK_ROOT`,再进它的 `docs/` 找**——编译 QNN 后端时该路径由 CMake 定义（`source/backend/qnn/CMakeLists.txt`:优先 `-DQNN_SDK_ROOT=...`,回退环境变量 `$QNN_SDK_ROOT`），已固化在构建目录的 CMakeCache 里：
```bash
# 1) 从构建配置拿 SDK 根(最可靠)
SDK=$(grep -i QNN_SDK_ROOT project/android/build_64/CMakeCache.txt | head -1 | cut -d= -f2)
#    或直接用环境变量 $QNN_SDK_ROOT / 你 cmake 时传的 -DQNN_SDK_ROOT
# 2) 进 docs/QNN 找 OpDef(子路径随版本略有不同,用 find 兜底)
ls "$SDK/docs/QNN/OpDef/" 2>/dev/null || find "$SDK/docs" -iname MasterOpDef.html
#    2.40 及更早:  $SDK/docs/QNN/OpDef/
#    2.46/2.48+:   $SDK/docs/QAIRT-Docs/QNN/OpDef/
```

该目录下按**优先级**查这几个文件：
| 文件 | 作用 | 什么时候看 |
|------|------|-----------|
| **`SupportedOps.html`** | 各后端(CPU/GPU/**HTP**/DSP…)**支持哪些算子**的总表 | 先确认目标算子 HTP 到底支不支持 |
| **`MasterOpDef.html`** | 每个算子的**通用定义**:输入/输出个数、各 input 名字与含义、param(scalar/tensor)、dtype、rank | 写实现时对照参数名与输入顺序 |
| **`HtpOpDefSupplement.html`** | **HTP 专属的额外约束/覆盖**(fp16-only、axes 限制、rank≤4、量化要求等) | 查 graphFinalize 失败、`could not create op` 的根因 |
| `CpuOpDefSupplement.html` 等 | 其它后端的补充约束 | 对比/交叉验证时 |

> HTML 用 `WebFetch`(`file://` 不行时先 `cat`/转文本)或直接在文件里 `grep` 算子名。搜算子名（如 `ResizeBilinear`、`TransposeConv2d`、`LayerNorm`）即可定位其定义段。

> **关键原则**：编写/修改 QNN 算子前,**先 `SupportedOps` 确认支持 → `MasterOpDef` 对参数 → `HtpOpDefSupplement` 对 HTP 约束**。绝大多数 graphFinalize 失败(`could not create op`)都是违反了 HTP supplement 里的约束。

### 常见约束陷阱
| 算子 | 陷阱 | 正确做法 |
|------|------|----------|
| ResizeBilinear / ResizeNearestNeighbor | 只接受 **1 个输入**（image），输出 shape 由 output tensor dimensions 决定 | `addNodeCommon(inputs, outputs, 1)` 只传第一个输入 |
| LayerNorm (FP16) | 所有输入（data, gamma, beta）必须都是 FLOAT_16 | 确保 `createGammaBeta` 传入 `QNN_DATATYPE_FLOAT_16` |
| LayerNorm | axes 只支持最后一维或 4D 的最后三维；max rank = 4 | 超过 4D 需先 reshape |
| Conv2d | 权重必须是 HWIO 格式 | `convertWeight` OIHW→HWIO |
| **TransposeConv2d (Deconv)** | 权重要 **HWIO**；MNN deconv 权重存为 `[ic, oc/group, kH, kW]` | 手工重排为 `[kH, kW, ic, oc/group]`，核对 stride/pad/output-padding（案例见 `QNNDeconvolution.cpp`）|
| **Reshape/Flatten (NC4HW4)** | plain Reshape 会按 NHWC 顺序展平，折叠空间维时打乱数据 | channel 维变化则先 NHWC→NCHW 转置再 Reshape（案例 5）|
| **Interp/Resize** | 坐标模式在 `ctm` 字段，不在 `halfPixelCenters` bool | 按 `ctm` 设 align_corners/half_pixel_centers（案例 6）|

---

## QNN 错误日志（QNN Log Callback）

### 启用方法
在 `QNNBackend.cpp` 中，QNN 初始化时设置 log callback 的日志级别：

```cpp
// 在 QnnLog_create 时设置级别
// QNN_LOG_LEVEL_ERROR = 1  （只打印错误）
// QNN_LOG_LEVEL_WARN  = 2  （打印警告+错误）
// QNN_LOG_LEVEL_INFO  = 3  （打印信息+警告+错误）
// QNN_LOG_LEVEL_DEBUG = 4  （全部）
// QNN_LOG_LEVEL_VERBOSE = 5（最详细）
```

当前代码中搜索 `QNN_LOG_LEVEL_ERROR` 或 `logLevel` 相关位置，将级别改为更详细的级别即可获取更多信息。

### 错误码速查
| 错误码 | 含义 | 路径 | 常见原因 |
|--------|------|------|----------|
| 1002 | `QNN_GRAPH_ERROR_MEM_ALLOC` / finalize 失败 | 在线 | 图太大、或某算子 validate 失败（看 QNN_LOG） |
| 6000 | `QNN_GRAPH_ERROR_GENERAL` / execute 失败 | 在线 | graphFinalize 实际失败被忽略、clientBuf 大小不匹配、或**输入未拷入**(shapeMutable) |
| 1003 | `QNN_COMMON_ERROR_SYSTEM` / 系统级 | 离线/LLM | 运行时 IO 尺寸/顺序与离线图定义不一致（案例 7）|
| 6004 | `QNN_GRAPH_ERROR_INVALID_TENSOR` / 无效 tensor | 离线/LLM | IO 形状/dtype 不匹配（案例 7）|
| `validateOpConfig failed` (如 0xc26/3110) | 转换期算子校验失败 | 离线/LLM | 算子参数/维度/dtype 不满足 QNN 约束（案例 8）|

### 日志解读示例
```
QNN_LOG[1]: graph_prepare.cc:219::ERROR:could not create op: q::layernorm_2d_fp16_oneshot_moments_sf
QNN_LOG[1]: graph_prepare.cc:221::ERROR:Op creation failure, op id=... total_inputs=4
QNN_LOG[1]: graph_prepare.cc:207:  Input 0: ... output0=[...F16Crouton_TCMEE]     ← fp16
QNN_LOG[1]: graph_prepare.cc:207:  Input 1: ... output0=[...PlainFloat_TCMEE]      ← fp32 !!
```
- `F16Crouton_TCM` = fp16 格式
- `PlainFloat_TCM` / `PlainFloat` = fp32 格式
- `total_inputs` 包含 HTP 内部优化后的所有输入（可能比用户传入的多）
- `could not create op` = HTP 找不到匹配约束的实现 → 检查输入类型/维度是否符合 MasterOpDef

---

## 真实案例（续）

### 案例 3 · QNN Interp (ResizeBilinear) 输入数量错误导致 validate 失败
- **现象**：graphFinalize 失败，QNN 报 `Wrong number of Inputs 2`（ResizeBilinear 只接受 1 个输入）。
- **根因**：MNN 的 Interp op 有 2 个输入（image + size tensor），但 QNN 的 ResizeBilinear/ResizeNearestNeighbor 只接受 1 个输入（image），输出尺寸由 output tensor 的 dimensions 决定。`QNNInterp.cpp` 中 `addNodeCommon(inputs, outputs)` 默认传了所有输入。
- **修复**：`addNodeCommon(inputs, outputs, 1)` — 第三个参数指定只传第一个输入给 QNN。
- **教训**：QNN 算子的输入数量与 MNN 不一定一致，**必须查 MasterOpDef.html** 确认。`addNodeCommon` 的第三个参数 `inputSize` 为 0 时使用 `inputs.size()`，否则使用指定值。

### 案例 4 · QNN LayerNorm FP16 配置下 gamma/beta 类型不匹配
- **现象**：graphFinalize 失败（error 1002），QNN_LOG 报 `could not create op: q::layernorm_2d_fp16_oneshot_moments_sf`，Input 0 是 F16Crouton（fp16）但 Input 1/2（gamma/beta）是 PlainFloat（fp32）。
- **根因**：`QNNLayerNorm::onEncode` 中通过 `mBackend->getNativeTensor(inputs[0])->v1.dataType` 获取 dataType 传给 `createGammaBeta`。如果该值不是 `QNN_DATATYPE_FLOAT_16`，gamma/beta 会被创建为 fp32。QNN HTP 的 LayerNorm FP16 配置要求 data、gamma、beta **全部**为 FLOAT_16。
- **定位方法**：启用 QNN error log callback（`QNN_LOG_LEVEL_ERROR`），从日志中看到各 Input 的实际类型。
- **状态**：调查中 — 需确认 `getNativeTensor` 返回的 dataType 是否正确反映了 fp16 设置。

### 案例 5 · Reshape/Flatten 把 NC4HW4(NHWC) 按错误顺序展平 → 数据被"重排"
- **现象**（talking-head 模型，QNN fp16）：全模型 diff≈2.0。二分定位到 `Reshape [1,64,4,4] → [1,1024]`（FC 前的 flatten）：该 Reshape **输入** diff=0.0005（好），**输出**突跳到 **1.15**。误差≈1.0 且输出是参考的一个排列（元素齐全、顺序乱）。
- **根因**：NC4HW4 张量在 QNN 里按 **NHWC** 存储（`[1,4,4,64]`）。`QNNFlatten` 有个 `outputDim<=2` 捷径直接 plain `Reshape` → 按 (h,w,c) 展平；而 ONNX/参考要 NCHW 的 (c,h,w) 顺序 → 整段 1024 元素被打乱。
- **修复**（`QNNFlatten.cpp`）：去掉 2D 捷径，统一判据——只要输入是 `MNN_DATA_FORMAT_NC4HW4` 且首尾（channel）维在 reshape 前后变化，就走 `ReshapeTranspose`（先 NHWC→NCHW 转置，再 Reshape）。**顺带修崩溃**：`ReshapeTranspose` 的输出转置要加 `if (permuteOutput)` 保护——输出为 2D 时 `permuteOutput=false`，否则会访问未初始化的 `outputTempIndex`（这正是当初加 2D 捷径想规避、但方式错了的崩溃）。
- **教训**：**任何折叠/拆分空间维的 Reshape/Flatten/Squeeze，在 NC4HW4 下都必须考虑 NHWC↔NCHW 的元素顺序**，不能因为"输出是 2D/低秩"就走 plain reshape。

### 案例 6 · Interp/Resize 忽略 `ctm` 坐标变换模式 → pytorch_half_pixel 退化成 asymmetric
- **现象**（同上模型）：修完案例 5 后，误差在多尺度 U-Net 段逐块增长（0.02→0.09→0.35→1.08）。单独截断到 `Interp(input0, 256→16)`：**QNN-fp16=0.82，而 CPU-fp16=0.0006**（近乎完美）→ 典型"QNN 突跳、CPU-fp16 完美"= 该算子 QNN 实现 bug。
- **根因**：ONNX Resize 是 `pytorch_half_pixel`；MNN 转换器（`tools/converter/source/onnx/ResizeOnnx.cpp`）**只对精确字符串 `"half_pixel"` 置 `halfPixelCenters=true`**，其余坐标模式一律写进 `Interp.ctm` 字段（`halfPixelCenters` 保持 false）。`QNNInterp` 只读 `alignCorners`/`halfPixelCenters` 两个 bool（都为 false）→ ResizeBilinear 配成了 **ASYMMETRIC** 坐标。对 256→16 这种大比例采样，半像素/asymmetric 的坐标偏移差异被放大成 ~0.8 的误差。
- **修复**（`QNNInterp.cpp`）：`onEncode` 开头按 `interpParam->ctm()` 推导有效标志——`AlignCorners`→align；`HalfPixels`/`PytorchHalfPixels`/`TensorflowHalfPixels`→half_pixel；`Asymmetric`→都 false；`NotSet` 时回退到原 bool。
- **教训**（可推广）：**MNN 的 op 参数常有"新字段 + 旧 bool 冗余"的历史包袱，QNN 实现容易只读旧 bool 而漏掉权威字段**。Interp 的 `ctm` 就是典型。写/改 QNN 算子时，先看该 op 在 schema(`schema/default/*.fbs`) 里有哪些字段、CPU/converter 实际以哪个为准。

### 案例 7 · 离线/LLM 路径 1003/6004:输入输出与离线图定义不一致
- **现象**（`llm_demo` 跑预编译 QNN 模型）：运行时报 `1003`(`QNN_COMMON_ERROR_SYSTEM`) 或 `6004`(`QNN_GRAPH_ERROR_INVALID_TENSOR`)。
- **根因**：实际运行时喂入的 IO tensor 的**尺寸/顺序/数量**与离线编译进 QNN 图里的定义不一致。三个环节任一处不一致都会触发：
  ```
  generate_llm_qnn.py 定义的 IO(名字/尺寸/顺序)
      ↓ compilefornpu.cpp 构 QNN 图时的 IO 顺序
      ↓ QNNBackend.cpp 运行时绑定的 clientBuf.data / dataSize / 顺序
      ↓ HTP 执行
  ```
- **排查**：
  1. 读 `generate_llm_qnn.py` 里图的输入/输出定义(名字、shape、顺序、个数)。
  2. 与运行日志里 `GetMNNInfo` 打印的 IO 对比,逐项核对名字/形状/顺序/数量。
  3. 若定义对但仍报错,查 `QNNBackend.cpp` 里绑定处 `tensor.v1.clientBuf.data/dataSize` 是否 = 元素数×每元素字节、顺序是否与图定义一致。
  4. 再查 `tools/cpp/compilefornpu.cpp` 构图时的 IO 顺序。
- **修复**：改到不一致的那一环(定义/绑定/顺序),改 `generate_llm_qnn.py` 要重转模型,改 C++ 要重编。

### 案例 8 · 转换期算子校验失败 (validateOpConfig)
- **现象**（`generate_llm_qnn.py` 转换阶段）：`QnnBackend_validateOpConfig failed`,如 `has incorrect Value 6144, expected equal to 6144` / `Failed to validate op _layers_0_..._Linear with error 0xc26`。
- **排查**：
  1. 从日志提取:出错**算子名**、**算子类型**(从 addNode 参数)、**错误值**、**错误码**。
  2. 定位对应 `source/backend/qnn/execution/QNN<算子>.cpp`(Conv2d→QNNConvolution、MatMul→QNNMatmul、LayerNorm→QNNLayerNorm…)。
  3. 对照 `QnnOpDef.h` / **MasterOpDef.html + HtpOpDefSupplement.html**(见下方"SDK 算子文档")核对参数名、维度、dtype。常见:维度不匹配、dtype 不支持、参数越界、漏必填参数。
  4. 若该算子 QNN 根本没实现 → 转 [SKILL 新增/适配算子] 补;SDK 也不支持 → 组合分解或回落 CPU。
- **注**：这与"新增/适配算子"是同一套查文档 → 改实现的方法,只是触发点在离线转换期。

### 案例 9 · LLM 离线推理结果乱码:量化参数问题
- **现象**：模型能跑不报错,但 `llm_demo` 输出乱码/无意义/数值偏差大。**能跑但结果错,几乎一定是量化精度**,不是结构问题。
- **排查**：
  1. 查 `llmexport.py` 量化逻辑:`scale` 是否 `NaN/Inf/0`、`zero_point` 是否异常、量化范围是否溢出、校准数据是否具代表性。
  2. **缩规模**:只导出单个 transformer block,用 `ModuleBasic`(**`shapeMutable=false`**) 逐算子对比 CPU vs QNN 输出,找第一个误差大的算子(判据:余弦相似度 <0.95 / 相对误差 >10%)。
  3. 检查该算子量化参数。
- **修复**：修 NaN scale(加检测+回退计算)、调量化策略(如增大 `--quant_block`、确保 `--smooth`)、或对高误差层不量化/提精度。

### 案例 10 · 一个模型上多个 QNN bug 叠加（迭代二分的工作流经验）
- **背景**：某 talking-head 模型(双输入,含 Deconv/Interp/LayerNorm),QNN fp16 全模型 diff=2.0。一次会话里连续暴出**三个独立问题**,按二分依次定位/修复：
  1. 先撞 `shapeMutable=true` 导致的 **error 6000 / 0 输入**(案例 1 的另一种表现)→ 配 `shapeMutable=false`;
  2. 首个数值突跳在 `Reshape [1,64,4,4]→[1,1024]`(**案例 5**,Flatten NHWC 展平)→ 修 `QNNFlatten`,该点 1.15→0.0005;
  3. 修完后下一个突跳移到 `Interp(256→16)`(**案例 6**,忽略 `ctm`)→ 修 `QNNInterp`,该点 0.82→0.0005;
  4. 剩下的尾部误差平滑增长、CPU-fp16 也同步 → 判为 fp16 累积(非 bug)。全模型最终 2.0→**0.116**。
- **可复用经验**：
  1. **bug 会叠加,修完一个要重新二分**:第一个出错点修好后,误差会前移到下一个真 bug;别看到"输出还是错"就以为没修好——**对比修复前后该点 diff 是否回落**才是判据。持续迭代到"输出 diff 回落到 fp16 级/与 CPU-fp16 同量级"为止。
  2. **`qnn_probe.sh` 一次给 QNN-fp16/CPU-fp32/CPU-fp16 三列**极大加速:CPU-fp32 确认基线可信、CPU-fp16 当 fp16 地板、QNN 与地板的**突跳点**就是下一个 bug。
  3. **QNN fp16 不总是比 CPU/OpenCL fp16 差**:本例修完后 QNN 尾部(0.06)反而**优于** CPU-fp16(0.27);而案例 2 里 QNN 却差于 OpenCL。所以 CPU-fp16 判据看的是**单点相对突跳**,不是"QNN 端到端一定更差"这种先验。
  4. **`ModuleBasic` 默认 `shapeMutable=true`**:探测脚本务必注入 `false`,否则一开始就被 6000/0 输入卡住,误判成"整个模型都错"。

案例 3/5/6 和 conv 权重布局本质是**同一类**问题——QNN 实现对 MNN op 的"语义映射"不完整：

| 子类 | 例子 | 排查要点 |
|------|------|----------|
| **读错/漏读 op 字段** | Interp 漏读 `ctm`（案例 6）；只认旧 bool 不认新枚举 | 对照 `schema/default/*.fbs` 的字段 + 看 converter/CPU 以哪个字段为准 |
| **NHWC↔NCHW 元素顺序** | Reshape/Flatten 折叠空间维（案例 5）；维度登记转置 | NC4HW4 在 QNN 恒为 NHWC；任何跨 C/H/W 的重排都要显式 transpose |
| **权重/常量布局** | Conv OIHW→HWIO；Deconv **IOHW→HWIO**（TransposeConv2d）| 在 `createStaticFloatTensor` 前手工重排，并核对 in/out channel 与 group |
| **输入个数不一致** | Resize 只收 1 个输入（案例 3）| 查 MasterOpDef.html 的输入数；`addNodeCommon(inputs, outputs, N)` |

> 定位这类 bug 的最快路径：**截断到"输入好、输出坏"的那一个算子**，然后只读该算子 `onEncode` 里"从 MNN op 取了什么、喂给 QNN 什么"，几乎总能一眼看出漏掉的字段或没做的转置。

---