# macOS 上的 LLM prefill 对照实验

本实验比较 MNN 原生 Metal、整网提前转换 W8、CPU 逐层准备 W8 和 GPU 逐层准备 W8 四条推理路径。输入为同一套 MNN 非对称 INT4/G64 权重。三个 Core ML 对照都逐层运行完整的 Transformer block，随后将 KV cache 交给 MNN 原生 Metal decode。Embedding、RoPE 输入和词表头沿用 MNN 路径。

| 测试参数 | Prefill | 权重何时转换、如何复用 |
| --- | --- | --- |
| `metal` | MNN Metal | 直接使用源模型，作为时延和 logits 参照 |
| `fixed` | Core ML，每层一个固定权重模型 | 测试前逐层转换并编译整网，七个投影以每输出通道一个 scale 的 W8 常量绑定到各层。热请求复用加载的模型 |
| `cpu-stream` | Core ML，同形状的层共用一个程序 | 每次请求用 6 路 CPU 读取和转换当前投影。在上一层运行期间准备下一层，交替使用两组输出缓冲 |
| `gpu-stream` | 同上 | 每层用一个 Metal command buffer 转换七个投影，直接写入下一组 IOSurface，再提交给 Core ML |

W8 指投影权重使用有符号 8 位编码，scale 为 FP16。流式图将编码作为 8 位图像输入，在图内恢复权重。该实验未量化激活。固定组使用 `constexpr_blockwise_shift_scale + linear`，与流式组使用动态权重的 `matmul` 有不同的设备执行成本。

## 构建和运行

在 Apple Silicon Mac 上安装 Xcode，并将 `xcode-select` 指向该版本。从仓库根目录运行以下命令。

```sh
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=OFF -DMNN_BUILD_TOOLS=OFF \
  -DMNN_METAL=ON -DMNN_ARM82=ON -DMNN_COREML=ON -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_SEP_BUILD=ON -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_COREML_PREFILL_BENCH=ON
cmake --build build/bench --target coreml_prefill_bench -j8

python3 -m venv build/prefill-venv
source build/prefill-venv/bin/activate
python -m pip install 'coremltools==9.0' numpy
bench=transformers/llm/engine/tools/coreml_prefill
bin="$PWD/build/bench/coreml_prefill_bench"

python "$bench/run.py" --binary "$bin" --config /models/Pocket-0.3B/config.json \
  --output "$PWD/build/pocket-prefill" --rounds 7
python "$bench/run.py" --binary "$bin" --config /models/Qwen3-4B/config.json \
  --output "$PWD/build/qwen-prefill" --rounds 7 --qwen-range-fix
```

将示例路径换为可由 MNN 加载的模型配置，包含图、外部 `.mnn.weight`、模型配置和 tokenizer。实测模型为完整因果注意力、Q/K RMSNorm 的 MNN-Pocket-0.3B（28 层）和 Qwen3-4B（36 层）。要复现全部四组，七个投影需使用连续存储的非对称 Q4/G64 和 FP16 分组系数。导出参数为 `--quant_bit 4 --quant_block 64 --scale_bit 16`。GPU 和固定组会检查权重布局。Qwen 固定组的 `--qwen-range-fix` 在第 7 层 SwiGLU 将 up 分支除以 32，再将 down 输出乘回 32，避免该模型的 FP16 中间结果溢出。

每次运行指定一个新的输出目录。若只测流式组和基线，可以直接使用系统 Python 3.8+，并添加 `--modes metal cpu-stream gpu-stream`。只有固定组需要 coremltools 和 `xcrun coremlcompiler`。运行过程中它会从 MNN 图和权重重新导出与编译每一层。固定组会在输出目录保留编译模型，Qwen3-4B 约需 3.39 GiB。

## 如何读结果

- 固定输入经各模型自己的 chat template 和 tokenizer 校验为 128 token。每组独立进程、4 线程、greedy、同步执行，每次清空 KV 并生成 16 token。默认第 1 次单列，其余 6 次取中位数。
- `summary.json` 记录模型加载时间、进程首次 TTFT、热 prefill、热 TTFT、16 token 端到端时间和进程峰值物理内存。TTFT 从调用开始到首次文本回调，模型加载和离线构建单独计时。系统的 Core ML 特化缓存会影响首次请求。峰值采用 Mach 的 `ledger_phys_footprint_peak`，覆盖加载和推理，不含独立的编译与系统服务进程。
- 每组的 `bench.log` 保留逐请求时间和 token ID。脚本另跑一次 prefill 保存首位置全词表 logits 到 `first.f32`，报告相对 Metal 的 Top-1、相对 L2 和 16 个自由生成 token 是否全部相同。CPU/GPU 流式组的首位置 logits 和 token 序列须逐字节一致。该输入仅覆盖一个提示。
- 每个混合请求都须完成全部 Core ML Transformer 层，纯 Metal 组须完成零个 Core ML 层。硬件落点需用下面的 Instruments 步骤另行采样。
- 固定组的 `build.json` 保留逐层权重转换、打包与 Core ML 编译用时。`models/` 下含可重新加载的各层模型。重新测加载和特化可直接用 `MNN_PREFILL_BENCH_FIXED="$PWD/build/qwen-prefill/fixed/models" "$bin" "$PWD/build/qwen-prefill/fixed/config.json" @128 7 "$PWD/build/qwen-prefill/replay"`。

## 实测

M3 Pro、36 GiB、macOS 27.2.1、Xcode 27.1，2026-11-25。下表记录首次对照实验的结果，各组均在 Instruments 外顺序运行。Metal 基线和 Core ML 特化缓存会随运行环境变化。同一行的端到端时间包含 prefill 和 Metal decode。

| 模型 | 方案 | 热 prefill ms | 热 TTFT ms | 16 token 端到端 ms | 峰值 MiB | 首位置 logits 相对 L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pocket-0.3B | metal | 26.1 | 26.3 | 62.5 | 301 | 0.00% |
| Pocket-0.3B | fixed | 18.3 | 18.5 | 53.6 | 354 | 2.71% |
| Pocket-0.3B | cpu-stream | 86.4 | 86.6 | 126.5 | 371 | 2.81% |
| Pocket-0.3B | gpu-stream | 84.1 | 84.3 | 119.8 | 369 | 2.81% |
| Qwen3-4B | metal | 276.9 | 277.2 | 611.5 | 2700 | 0.00% |
| Qwen3-4B | fixed | 128.7 | 129.0 | 425.7 | 3003 | 7.30% |
| Qwen3-4B | cpu-stream | 1064.7 | 1065.0 | 1384.0 | 3333 | 7.64% |
| Qwen3-4B | gpu-stream | 1012.7 | 1013.0 | 1316.9 | 3315 | 7.64% |

Qwen 固定 W8 的热 prefill 从纯 Metal 的 276.9 ms 降到 128.7 ms，加速 2.15 倍。本轮生成额外的 3.39 GiB 编译模型，转换与打包耗时 11.36 s，编译耗时 3.68 s。模型加载后首次请求的 TTFT 为 35.69 s，纯 Metal 为 0.45 s。Pocket 固定组生成约 239 MiB，转换与编译共 3.37 s，进程首次 TTFT 为 15.59 s。首次请求包含加载各层模型和系统设备特化，受机器缓存状态影响。

两个模型上四条路径均生成了相同的 16 个 token。Qwen 固定组与流式组的首位置 logits 相对 L2 分别为 7.30% 和 7.64%，均超过 2% 的实验精度门槛。CPU 和 GPU 逐层准备的首位置 logits 在两个模型上均逐字节一致。

## 用 Instruments 核对逐层落点和重叠

以下以 Qwen GPU 备料为例。采样时关闭其他使用 ANE 的应用。硬件 Prediction 轨道由整台设备共享，采样器同时要求本进程同一层调用窗口内出现成功的 ANE evaluation。系统版本会影响 Instruments 的表名。

```sh
out="$PWD/build/qwen-prefill"
xcrun xctrace record --template 'Core AI' --instrument 'Neural Engine' --instrument os_signpost \
  --output "$out/gpu.trace" --env MNN_PREFILL_BENCH_GPU=1 \
  --launch -- "$bin" "$out/gpu-stream/config.json" @128 3 "$out/trace-first"
xcrun xctrace export --input "$out/gpu.trace" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="os-signpost"]' --output "$out/signpost.xml"
xcrun xctrace export --input "$out/gpu.trace" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="ane-hw-intervals"]' --output "$out/ane.xml"
python "$bench/trace.py" "$out/signpost.xml" "$out/ane.xml" --layers 36 --skip 1 --requests 2
```

测 CPU 备料时选择 `cpu-stream/config.json` 并移除 GPU 环境变量。固定组选择 `fixed/config.json` 并将环境变量换为 `MNN_PREFILL_BENCH_FIXED=$out/fixed/models`。每组使用独立 trace 和导出文件。

`trace.py` 将本进程的 `CoreMLLayer`、成功的 ANE evaluation 与硬件 Prediction 按时间对齐，再计算本层准备与**上一层** ANE 执行的交集。覆盖率按准备时长加权，排除每个请求同步准备的首层。

同次实验对 Qwen3-4B 的两次热请求采样，每组 36 层 × 2 次均匹配到硬件 Prediction。每组可比较的相邻层有 70 对。

| 备料方案 | 准备 P50 ms | ANE 层 P50 ms | Core ML 主机调用 P50 ms | 准备时长被上一层 ANE 覆盖 | 准备耗时短于上一层 ANE | 主机返回后等待下一层权重 ms/请求 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU 六路 | 7.39 | 12.89 | 27.69 | 88.2% | 70/70 | 7.39 |
| GPU | 8.21 | 12.13 | 27.33 | 64.4% | 38/70 | 12.73 |

本轮 GPU 设备执行时间有 80.6% 与 ANE 重叠。较早的同机采样中，CPU/GPU 准备 P50 为 6.96/5.37 ms，对应覆盖率 89.8%/74.0%。主机调度会影响重叠指标。即使 CPU 的 70 次准备均短于上一层 ANE 执行，流式组的实测热 prefill 仍慢于纯 Metal。
