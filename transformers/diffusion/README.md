# MNN Diffusion 模型使用指南

本目录包含MNN实现的Diffusion模型，支持Stable Diffusion和Sana Diffusion两种文生图模型。

## 目录

- [编译](#编译)
- [使用说明](#使用说明)
  - [Stable Diffusion Demo](#stable-diffusion-demo)
  - [Sana Diffusion Demo](#sana-diffusion-demo)
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
└── vocab.txt           # Tokenizer词表
```

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
