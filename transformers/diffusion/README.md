# MNN Diffusion 模型使用指南

本目录包含MNN实现的Diffusion模型，支持Stable Diffusion和Sana Diffusion两种文生图模型。

## 目录

- [编译](#编译)
- [使用说明](#使用说明)
  - [Stable Diffusion Demo](#stable-diffusion-demo)
  - [Sana Diffusion Demo](#sana-diffusion-demo)
  - [Wan Diffusion Demo](#wan-diffusion-demo)
- [模型转换](#模型转换)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 编译

```bash
cd mnn_path
mkdir build
cd build
# 安卓开启-DMNN_OPENCL=ON，iOS开启-DMNN_METAL=ON
cmake .. -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_LLM_OMNI=ON -DMNN_IMGCODECS=ON -DMNN_BUILD_LLM=ON
make -j32
# 安卓backend_type可设为MNN_FORWARD_OPENCL，iOS backend_type可设为MNN_FORWARD_METAL
```

---

## 使用说明

### Stable Diffusion Demo

#### 命令格式

```bash
./diffusion_demo <resource_path> <model_type> <memory_mode> <backend_type> <iteration_num> <random_seed> <output_image_name> <prompt_text>
```

#### 参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `resource_path` | 模型资源路径 | 模型文件所在目录 |
| `model_type` | 模型类型 | `0`=SD1.5, `1`=SD太乙中文版 |
| `memory_mode` | 内存模式 | `0`=省内存, `1`=速度优先, `2`=平衡 |
| `backend_type` | 后端类型 | `0`=CPU, `3`=OpenCL, `6`=Metal |
| `iteration_num` | 推理步数 | 固定10步 |
| `random_seed` | 随机种子 | 任意整数 |
| `output_image` | 输出图像路径 | 如：output.jpg |
| `prompt_text` | 文本描述 | 英文或中文 |

#### 使用示例

**示例1：基础使用（SD 1.5，英文）**
```bash
./diffusion_demo models/sd15 0 2 0 10 42 sunset.jpg "a beautiful sunset over the ocean"
```

**示例2：GPU加速（OpenCL）**
```bash
./diffusion_demo models/sd15 0 2 3 10 42 output.jpg "a cute cat playing with yarn"
```

#### 模型文件结构

```
models/sd15/
├── text_encoder.mnn    # CLIP文本编码器
├── unet.mnn            # UNet去噪模型
├── vae_decoder.mnn     # VAE解码器
└── tokenizer.mtok      # Tokenizer
```

#### Tokenizer 说明

- 运行 diffusion demo 需要开启 `-DMNN_BUILD_LLM=ON`。
- `resource_path` 目录下需要提供 `tokenizer.mtok`。
- `convert_mnn.py` 会把 HuggingFace tokenizer 导出为 `tokenizer.mtok`。

---

### Sana Diffusion Demo

#### 命令格式

```bash
./sana_diffusion_demo <resource_path> <mode> <prompt> [input_image] [output_image] [width] [height] [steps] [seed] [use_cfg] [cfg_scale]
```

#### 参数说明

| 参数 | 说明 | 默认值 | 备注 |
|------|------|--------|------|
| `resource_path` | 模型资源路径 | - | 必需 |
| `mode` | 生成模式 | - | `text2img`或`img2img` |
| `prompt` | 文本描述 | - | 支持中英文 |
| `input_image` | 输入图像路径 | `""` | img2img模式必需 |
| `output_image` | 输出图像路径 | `sana_out.jpg` | - |
| `width` | 输出宽度 | `512` | 必须是32的倍数 |
| `height` | 输出高度 | `512` | 必须是32的倍数 |
| `steps` | 推理步数 | `5` | 蒸馏加速，5步即可 |
| `seed` | 随机种子 | `42` | - |
| `use_cfg` | 是否使用CFG | `0` | `0`=否, `1`=是 |
| `cfg_scale` | CFG强度 | `4.5` | 仅use_cfg=1时生效 |

#### 使用示例

**示例1：基础文生图（512x512）**
```bash
./sana_diffusion_demo models/sana text2img "一只可爱的猫咪" "" cat.jpg 512 512 20 42 0 4.5
```

**示例2：使用CFG提升质量**
```bash
./sana_diffusion_demo models/sana text2img "夕阳下的海滩，细节丰富" "" beach.jpg 512 512 20 42 1 4.5
```

**示例3：图像编辑（img2img）**
```bash
./sana_diffusion_demo models/sana img2img "添加彩虹" input.jpg output.jpg 512 512 20 42 0 4.5
```


#### 模型文件结构

```
models/sana/
├── llm/                # Qwen3-0.6B LLM
│   ├── embeddings.mnn
│   ├── blocks_*.mnn
│   └── lm.mnn
├── connector.mnn       # 特征桥接
├── projector.mnn       # 特征投影
├── transformer.mnn     # DiT模型
├── vae_decoder.mnn     # VAE解码器
└── vae_encoder.mnn     # VAE编码器（img2img需要）
```

---

### SD3.5 Diffusion Demo

SD3.5 已集成到 MNN，用于移动端/边缘端的文生图推理。

English: [README.md](./README.md)

#### 效果展示

示例输出可参考 `Running.md` 中的 SD3.5 运行命令与结果保存路径。

#### 应用链接
- [Android MNN LLM Chat](../../apps/Android/MnnLlmChat/README.md)
- [iOS MNN LLM Chat](../../apps/iOS/MNNLLMChat/README.md)

#### 模型链接
Stable Diffusion 3.5 Medium 模型。

- HuggingFace: [https://huggingface.co/stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- ModelScope: [https://modelscope.cn/models/stabilityai/stable-diffusion-3.5-medium](https://modelscope.cn/models/stabilityai/stable-diffusion-3.5-medium)

#### 推荐设置
- 输入：文本提示词（prompt）。
- 输出：默认示例保存为 `demo.jpg`。
- 提示词：建议使用英文短句；中文也可用。
- 步数：建议使用 `20` 步作为平衡配置。
- 后端：`0`=CPU，`3`=OpenCL，`6`=Metal。
- 编译：需要开启 `-DMNN_BUILD_LLM=ON`。
- Tokenizer：`tokenizer`、`tokenizer_2`、`tokenizer_3` 目录下都需要提供 `tokenizer.mtok`。

#### 在 MNN Chat 应用中的使用
1. 完成模型转换并准备 SD3.5 MNN 模型目录。
2. 进入构建目录并确认 `diffusion_sd35_demo` 已编译。
3. 按以下命令运行生成：

```bash
./diffusion_sd35_demo <resource_path> <memory_mode> <backend_type> <iteration_num> <random_seed> <output_image_name> <prompt_text>
```

示例：

```bash
./diffusion_sd35_demo /path/to/stable-diffusion-3.5-medium-MNN 0 3 20 1 demo.jpg "a cute cat"
```

#### 参考
- 运行与导出示例: [Running.md](../../Running.md)

---

### Wan Diffusion Demo

Wan2.1-T2V-1.3B 支持以 v1 experimental 方式接入 MNN Diffusion。当前目标是提供可用的工程入口，便于后续用真实 checkpoint bring-up；脚本不会联网下载模型，也不会在缺少官方 Wan 代码或 checkpoint 时伪造导出成功。T2V 输出为帧序列，demo 侧会在输出目录中保存 `frame_0000.png` 这类文件。

#### 模型文件结构

转换后的资源目录建议如下：

```
models/wan2.1-t2v-1.3b/
├── text_encoder.mnn
├── transformer.mnn
├── vae_decoder.mnn
└── tokenizer/
    ├── tokenizer.mtok
    ├── tokenizer.json
    └── ...
```

中间 ONNX 目录由导出脚本生成：

```
wan_onnx/
├── text_encoder/model.onnx
├── transformer/model.onnx
├── vae_decoder/model.onnx
└── tokenizer/
```

#### 导出 ONNX

`wan_onnx_export.py` 会优先尝试从本地 diffusers Wan pipeline 加载；如果不可用，再尝试从 `--model_path` 或当前 Python 环境中导入官方 `wan` 包和 `WanT2V` 组件。

```bash
cd transformers/diffusion/export/wan
python wan_onnx_export.py \
  --model_path /path/to/Wan2.1-T2V-1.3B \
  --output_path /path/to/wan_onnx \
  --opset 17 \
  --dtype fp32 \
  --width 256 \
  --height 256 \
  --frames 9 \
  --text_len 512
```

导出的 ONNX 输入输出命名与 Wan runtime 对齐：

| 模块 | 输入 | 输出 |
|------|------|------|
| `text_encoder` | `input_ids` | `last_hidden_state` |
| `transformer` | `hidden_states`, `timestep`, `encoder_hidden_states`, `encoder_attention_mask` | `noise_pred` |
| `vae_decoder` | `latent_sample` | `sample` |

#### 转换 MNN

```bash
cd transformers/diffusion/export/wan
python wan_convert_mnn.py \
  --onnx_path /path/to/wan_onnx \
  --mnn_root /path/to/models/wan2.1-t2v-1.3b
```

如需传递额外 MNNConvert 参数，放在 `--extra` 后面：

```bash
python wan_convert_mnn.py \
  --onnx_path /path/to/wan_onnx \
  --mnn_root /path/to/models/wan2.1-t2v-1.3b \
  --extra --weightQuantBits=8
```

`wan_convert_mnn.py` 默认使用 `--saveExternalData=1`，并会把 tokenizer 源目录复制到 `mnn_root/tokenizer` 后导出 `tokenizer.mtok`。如果 tokenizer 不在 ONNX 目录下，可显式指定：

```bash
python wan_convert_mnn.py \
  --onnx_path /path/to/wan_onnx \
  --mnn_root /path/to/models/wan2.1-t2v-1.3b \
  --tokenizer_path /path/to/Wan2.1-T2V-1.3B/tokenizer
```

#### 运行建议

首次 bring-up 建议先用小分辨率和少帧数做 smoke，例如 `256x256`、`9` 帧、较少采样步数，确认 text encoder、transformer、VAE decoder 的输入输出形状和帧保存路径无误后，再提升到目标分辨率与帧数。当前 runtime 使用 CFG batch=2，因此导出脚本也按 batch=2 固化 `text_encoder` 和 `transformer` 的 ONNX 输入形状。

如果 Worker A 的运行时已编译 Wan demo，命令形式通常为：

```bash
./wan_diffusion_demo <resource_path> <memory_mode> <backend_type> <steps> <seed> <width> <height> <frames> <cfg_scale> <output_dir> <prompt_text>
```

参数含义：

| 参数 | 说明 |
|------|------|
| `resource_path` | Wan MNN 模型资源目录 |
| `memory_mode` | 内存模式，沿用 Diffusion demo 约定 |
| `backend_type` | `0`=CPU, `3`=OpenCL, `6`=Metal |
| `steps` | 采样步数，smoke 阶段建议先取较小值 |
| `seed` | 随机种子 |
| `width`, `height`, `frames` | 输出尺寸与帧数，宽高需要是 16 的倍数，并与导出 shape 对齐 |
| `cfg_scale` | CFG 引导强度 |
| `output_dir` | 输出帧序列目录 |
| `prompt_text` | 文生视频提示词 |

量化建议：先保留 `text_encoder` 和 `vae_decoder` 为 fp16/fp32，优先尝试对 `transformer.mnn` 做权重量化；每次量化后都用固定 seed、小尺寸、少帧数对比输出稳定性，再逐步扩大分辨率和帧数。
