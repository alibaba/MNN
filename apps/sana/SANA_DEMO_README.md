# Sana Diffusion Demo

基于 MNN 的 Sana Diffusion 模型推理，支持 Mac 和 Android 平台。

## 功能特点

- **Qwen3-0.6B LLM**: 使用轻量级语言模型处理文本提示词
- **text2img**: 文本生成图像
- **img2img**: 图像风格转换/编辑
- **多后端支持**: CPU, Metal (Mac), OpenCL (Android/GPU)
- **CFG 引导**: 可选的 Classifier-Free Guidance

## 前置要求

### Mac
- cmake, python3
- Xcode Command Line Tools

### Android
- adb (Android Platform Tools)
- 已构建的 Android 可执行文件 (`build_android/sana_diffusion_demo`)

## 快速开始

### 1. 构建

```bash
# Mac 构建
cd /path/to/MNNNPU
mkdir -p build_sana && cd build_sana
cmake .. -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_LLM=ON -DMNN_METAL=ON
make -j8

# Android 构建
mkdir -p build_android && cd build_android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_STL=c++_static \
    -DMNN_BUILD_DIFFUSION=ON \
    -DMNN_BUILD_LLM=ON \
    -DMNN_OPENCL=ON
make -j8
```

### 2. 运行

```bash
cd apps/sana/

# Mac 文生图
./run_sana_demo.sh -m ~/models/sana -M text2img -p "一只可爱的猫咪" -o cat.jpg

# Mac 图像编辑
./run_sana_demo.sh -m ~/models/sana -M img2img -i input.jpg -p "转换为吉卜力风格" -o ghibli.jpg

# Android OpenCL
./run_sana_demo.sh -m ~/models/sana -t android -b opencl -M img2img -i photo.jpg -p "添加彩虹"
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | 模型目录路径 | (必需) |
| `-M, --mode` | 模式: text2img/img2img | text2img |
| `-i, --input` | 输入图像 (img2img 必需) | - |
| `-o, --output` | 输出图像路径 | sana_out.jpg |
| `-p, --prompt` | 文本提示词 | - |
| `-b, --backend` | 后端: cpu/metal/opencl | cpu |
| `-t, --target` | 平台: mac/android | mac |
| `-s, --steps` | 推理步数 | 5 |
| `-S, --seed` | 随机种子 | 42 |
| `-W, --width` | 输出宽度 | 512 |
| `-H, --height` | 输出高度 | 512 |
| `--cfg` | 启用 CFG 引导 | 默认启用 |
| `--no-cfg` | 禁用 CFG 引导 | - |
| `--cfg-scale` | CFG 强度 | 4.5 |

## 直接使用可执行文件

```bash
# 语法
./sana_diffusion_demo <model_path> <mode> <prompt> [input_image] [output] [width] [height] [steps] [seed] [use_cfg] [cfg_scale]

# 示例: 文生图
./sana_diffusion_demo models text2img "一只猫" "" out.jpg 512 512 5 42 1 4.5

# 示例: 图像编辑
./sana_diffusion_demo models img2img "添加彩虹" input.jpg out.jpg 512 512 5 42 1 4.5
```

## 后端选择

| 后端 | 平台 | 说明 |
|------|------|------|
| cpu | Mac/Android | 兼容性最好，速度较慢 |
| metal | Mac | Apple 芯片加速 |
| opencl | Android/GPU | GPU 加速，需要 `Precision_High` 配置 |

**注意**: OpenCL 后端默认使用 FP16 精度，可能导致 NaN 问题。代码中已配置 `Precision_High` 解决。

## 目录结构

```
apps/sana/
├── run_sana_demo.sh            # 主运行脚本
├── run_sana_on_android.sh      # Android 运行脚本
├── run_sana_benchmark_*.sh     # Benchmark 脚本
├── run_benchmark.py            # Python benchmark
├── docs/                       # 文档
├── SANA_DEMO_README.md         # 本文件
├── SANA_BENCHMARK_REPORT.md    # Benchmark 报告
└── SANA_OPTIMIZATION_REPORT.md # 优化报告
```

## 常见问题

### 1. OpenCL 生成灰色图片
已修复：在 `sana_diffusion.cpp` 中为 OpenCL 配置了 `Precision_High`。

### 2. Android 上找不到库
确保 `LD_LIBRARY_PATH` 包含库文件目录：
```bash
export LD_LIBRARY_PATH=/data/local/tmp/sana_demo/libs:$LD_LIBRARY_PATH
```

### 3. 模型加载失败
检查模型目录是否包含所有必需文件：
- `llm/` (LLM 模型)
- `transformer.mnn`
- `vae_decoder.mnn`
- `connector.mnn`
- `projector.mnn`
