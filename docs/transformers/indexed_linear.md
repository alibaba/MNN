# IndexedLinear 与 MoE 导出

`IndexedLinear` 根据动态专家编号，对静态权重集合中的线性层执行计算。它替代逐专家调用
`MoEModule` 的执行方式；Router、共享专家以及路由分数加权仍保留在计算图中，标准 SwiGLU 可在投影内融合。
旧版 MoE 模型需要使用新版导出工具重新转换，新模型也需要支持该算子的 MNNConvert 和运行库。

## 算子语义

设 `E` 为专家数、`T` 为 token 数、`K` 为每个 token 选择的专家数，`I/O` 为输入/输出通道数。
每个专家包含 `W[e]: [O, I]` 和可选的 `bias[e]: [O]`。无 bias 时视为零。
以下形状表描述普通线性模式（`fusedActivation=-1`）。

| 参数 | 输入 X | 输入 ids | 输出 Y |
| --- | --- | --- | --- |
| `inputMode = 0` | `[T, I]`，同一 token 的专家共享输入 | `[T, K]`，int32 | `[T, K, O]` |
| `inputMode = 1` | `[T, K, I]`，每条路由有独立输入 | `[T, K]`，int32 | `[T, K, O]` |

```text
e = ids[t, k]                  # 必须满足 0 <= e < E；允许重复编号
x = X[t]                      # inputMode = 0
x = X[t, k]                   # inputMode = 1
Y[t, k] = W[e] @ x + bias[e]
```

输入和输出使用普通连续布局，shape 推导将输出标记为 NCHW；X/Y 的逻辑类型为 float32，
实际存储可由后端精度配置选择 FP16。
普通模式不执行激活；该算子不执行 TopK、路由概率乘法或对 K 的归约。

参数包含 `linears`、`inputMode` 和默认值为 `-1` 的 `fusedActivation`。`ids` 索引 `linears` 中的投影。
`fusedActivation` 使用已有 `UnaryOpOperation` 的编号；`-1` 表示普通线性投影，其他值表示
投影保存 `[gate; up]`，输出为 `activation(gate) * up`，通道数减半。当前只支持 `SILU`（34），
其他激活保留在计算图中；运行时不接受尚未实现的融合类型。
旧版本独立存放 gate/up 的参数格式不再支持，已有 MoE 模型请重新转换。
所有专家的 `common.inputCount/outputCount` 必须一致，使用 1×1、group=1、无融合激活的参数。
复用 `Convolution2D` 的 `quanParameter`、bias 和 external 元数据，并不创建专家卷积子图或子 Execution。
权重仍可存于 `.mnn.weight`，后端加载时再组织为自身需要的静态权重布局。

### SwiGLU 融合

设置 `fusedActivation=UnaryOpOperation_SILU` 时，IndexedLinear 直接输出 `SiLU(gate) * up`：

- `linears[id]` 统一保存 `[2D, I]` 权重，前 D 行是 gate，后 D 行是 up。
- 原模型使用合并投影时保留原始量化边界；使用独立投影时分别量化 gate/up，再在转换阶段拼接量化整数、scale、offset 和 bias，重建合并后的编码头与外部权重描述，不重新量化。
- 输出为 `[T, K, D]`；down 仍使用独立的 IndexedLinear，输入为该激活结果。

导出侧沿用 `--disable_fuse_gate_up_proj` 和 `--disable_transformer_c4` 开关。
仅识别标准 SiLU 激活，GPT-OSS 的特殊激活与其他激活保留原路径。
独立 gate/up 的位宽、block 或对称性配置不同时，转换器保留两个普通 IndexedLinear 与激活算子。

加载时只需将合并投影的 gate/up 输出行配对，量化整数、scale、offset 和 bias 做相同的无损重排，不重新量化。
GPU 的 Prefill GEMM 在局部 tile 中完成激活，Decode GEMV 在寄存器中完成激活；两者都只写出 D 个通道。
CPU 复用原有 GEMM 与 SiLU 函数，在每个线程的局部输出缓存中完成激活，再写入最终输出。

## 导出与量化

`IndexedExperts` 同时用于普通 Mlp MoE 和 Gemma4 Decoder 级 MoE。第一层投影使用
`inputMode=0`，标准 SwiGLU 默认在该投影内融合激活；down 投影使用 `inputMode=1`，最终执行
`sum(Y * routing_scores[..., None], axis=1)` 并恢复原输入的 batch/sequence 形状。

每个原始 projection 独立量化，每个专家也独立量化，然后将这些参数收集到相应的 bank：

- 源模型 gate/up 为两个独立 Linear 时，分别量化；支持融合时将结果沿输出通道拼接到同一个 `linears[id]`。
- 源模型本来使用合并的 gate/up Linear 时，保留该合并投影的量化边界；down 始终为单独的 bank。
- HQQ 逐专家、逐原始投影调用，不先拼接浮点权重再量化，避免改变误差统计和停止条件。
- 保留原投影的量化配置名称。一个 bank 内要求位宽、实际 block 大小和对称性一致；量化
  block 会沿用 Linear 导出规则，调整到能整除输入通道的大小。

Router 仍保留模型自身的 sigmoid/softmax、TopK 前后归一化顺序、selection bias、缩放及共享专家逻辑。
GPT-OSS 的交错 gate/up 排列、裁剪和激活，Gemma4 的 router 归一化与 per-expert scale 也保留在图中。

运行时支持 W4、W8、FP16/FP32 权重；当前带权重 bank 的 Python 导出支持
`--quant_bit 4`、`8`、`16`，不接受 `--quant_bit 32`；FP32 权重可由算子参数或转换器提供。
W1/W2/W3 不属于 IndexedLinear 的支持范围，导出会报错。量化 scale 可以使用 FP16 或 FP32
序列化，加载时通过公共权重读取接口获取数值。

CPU 量化执行需要 `MNN_LOW_MEMORY=ON`，量化 block 还必须对齐当前 CPU 内核的输入 pack 单位；
仅满足“整除输入通道”不足以保证任意 block 大小可执行。FP16/FP32 激活的可用性取决于后端与设备；
CPU 的浮点与量化路径目前均不支持 BF16 核心，CUDA 路径不接收 BF16 激活。浮点权重在 Metal/CUDA 上会按后端的
FP16/FP32 执行精度转存，源权重为 FP32 不表示 Low 精度执行仍保留 FP32 权重。

原生 MNN 导出生成 IndexedLinear。标准 ONNX 导出使用 Gather + MatMul + bias 的参考展开，
该展开可用于数值检查，不代表 MNN 后端会 Gather 整个专家矩阵。

## 后端执行策略

| 后端 | Decode / 少量 token | Prefill |
| --- | --- | --- |
| CPU | 单 token 直接遍历 K 条路由；共享输入只 pack/动态量化一次；专家与输出通道共同分配线程任务 | 计数、前缀和与重排形成按专家分组的输入块，复用现有 float/W4/W8 pack 和矩阵乘内核，再写回原路由位置 |
| Metal | 单 token 使用直接读取 ids 的 GEMV，simdgroup 协作归约 | GPU 上完成计数、前缀和、scatter；按专家分块乘法，FP16 且设备支持时使用 simdgroup matrix，否则使用通用 tile 内核 |
| OpenCL Buffer | T≤4 使用直接读取 ids 的 GEMV，workgroup 归约 | GPU 路由分组和固定上界的 tile dispatch；按路由数选择 8×32 或 32×64 输出 tile，K 块为 16；局部权重使用转置并填充的 `[K, N+1]` 布局 |
| Vulkan Buffer | T≤4 使用直接读取 ids 的 GEMV，workgroup 归约 | GPU 路由分组与带有效性判断的固定上界 dispatch；量化 tile 乘法，不要求 subgroup 扩展 |
| CUDA | 单 token 使用 warp 协作的 GEMV | GPU 路由分组与 tile 乘法；适用的 FP16 路径使用 WMMA，其他路径使用通用内核 |

GPU 路由编号及路由分组全程保留在设备上，输出直接写回 `[T, K, O]` 的原路由位置。
W4/W8 权重以压缩形式常驻；执行时只解包当前计算块，不为整个 bank 分配浮点反量化副本。

OpenCL 在 `T*K >= 8*E` 且设备 workgroup/local memory 资源足够时选 32×64 tile（16×8 线程），
其他 Prefill 使用 8×32 tile（8×8 线程）。bank 形状、量化组、输入模式、Top-K 和 tile 参数进入
编译缓存键；局部存储与累加仍使用 float32，地址计算保留 64 位。路由工作区按最小 tile 预留容量，
避免动态输入由大 tile 切到小 tile 时容量不足。
各后端加载一次权重资源并可由 clone 共享，不创建每个专家的 Module。

IndexedLinear 需要 `MNN_SUPPORT_TRANSFORMER_FUSE=ON`；`MNN_BUILD_LLM=ON` 会自动开启该宏。GPU 实现需要开启对应后端，OpenCL/Vulkan 使用 Buffer 模式。
OpenCL 使用 Buffer 路径，需要 `MNN_OPENCL_SIZE_CUT=OFF`；Vulkan 使用
`MNN_VULKAN_IMAGE=OFF` 的 Buffer 构建。Vulkan 还要求设备支持每阶段至少 7 个 storage-buffer descriptor。
这些路径描述实现结构；目标设备上的性能及精度需要分别验证。

当前资源大小限制如下，尚未实现把过大的 bank 拆为多个 buffer 或按需换入专家：

- 公共加载接口要求单个专家的 `I * O <= INT_MAX`；OpenCL/Vulkan 当前进一步要求
  `I * O <= INT_MAX / 4`。GPU 路由数 `T * K` 也使用 int32 范围。
- Metal/CUDA 将整个 bank 的权重、scale 和 bias 分别放入一个 buffer，每个 buffer
  都必须不超过 `INT_MAX` 字节（约 2 GiB），并满足设备分配限制。
- OpenCL 的整个 bank buffer 受设备单次分配和可用内存限制；分配失败会拒绝该执行路径。
- Vulkan 的每个权重、scale、bias buffer 以及 Prefill 路由工作区，分别必须不超过设备的
  `maxStorageBufferRange`。路由工作区还受 int32 索引范围限制，dispatch 受设备工作组数量限制。

这些限制以单个 IndexedLinear bank 为单位，不是模型总大小限制。若某后端不接受模型的
bank 或 shape，应检查实际调度结果，不能把 CPU fallback 的吞吐量记为该 GPU 的性能。

## 回归与导出命令

以下命令从仓库根目录运行。Python 测试使用导出工具的依赖，不需要下载模型或编译 MNN：

```bash
python -B transformers/llm/export/tests/test_indexed_linear.py -v
```

7 项测试覆盖普通和融合 projection/激活一致性、四种 Router、Gemma4 Decoder MoE、标准 ONNX
数值执行，以及逐专家 HQQ 调用、权重字节一致性和不同量化配置的回退。临时模型仅创建在
`build/moe_refactor/export-tests/`，完成后自动删除。

编译时打开 `MNN_BUILD_TEST=ON`、`MNN_LOW_MEMORY=ON`、`MNN_SUPPORT_TRANSFORMER_FUSE=ON` 后，可运行算子测试：

```bash
./build/run_test.out op/IndexedLinear 0 1 4
./build/run_test.out op/IndexedLinear 0 2 4
```

参数依次为测试名、后端、精度、线程/模式。GPU 后端编号为 Metal=1、CUDA=2、OpenCL=3、Vulkan=7；
精度 1/2 分别为 High/Low。验证 GPU 时还需确认实际执行后端，CPU fallback 不构成 GPU 验证。
OpenCL 的模式需要包含 Buffer 标志 64，例如 `op/IndexedLinear 3 2 64`；
开启 command-buffer 记录时使用 `op/IndexedLinear 3 2 321`。

使用当前编译的转换器重新导出 MoE，例如：

```bash
python transformers/llm/export/llmexport.py \
    --path /path/to/MobileMoE-S-SFT \
    --export mnn --hqq --quant_bit 4 \
    --mnnconvert "$PWD/build/MNNConvert" \
    --dst_path "$PWD/build/moe_refactor/mobilemoe"
```

显式指定 `--mnnconvert`，可以避免当前目录变化导致意外使用未包含 IndexedLinear 的已安装转换器。
