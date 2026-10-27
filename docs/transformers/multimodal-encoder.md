# 多模态 Encoder

`Omni` 在加载时根据模型家族选择 processor。家族负责输入准备、媒体标记和位置规则；编码器执行、输出布局归一化、batch/时间组拆分及 embedding 插入共用。实现集中在 `omni.cpp` / `omni.hpp`，导出适配保留在 `vision.py` / `audio.py`。

## 家族与复用

| vision_processor.family | C++ 实现 | 复用与差异 |
| --- | --- | --- |
| `clip`、`deepseek_vl`、`internvl`、`qwen`、`gemma3`、`mobileclip` | `Fixed` | 共用固定尺寸输入和单段特征组装，尺寸与归一化由模型配置给出 |
| `qwen2_vl`、`glm_ocr` | `Qwen` | 复用 `Patch` 的动态缩放，构造网格位置和 attention mask |
| `qwen2_5_vl`、`qwen2_5_omni` | `Qwen25 : Qwen` | 覆盖窗口 mask/index 构造 |
| `qwen3_vl`、`qwen3_5` | `Qwen3 : Qwen` | 覆盖默认像素范围、位置插值输入和视频时间组标记 |
| `idefics3`、`lfm2_vl` | `Idefics` | 共用当前全局图/局部切块协议，集中处理特征顺序和切块标记 |
| `minicpmv` | `MiniCPM` | 保留 MiniCPM 切片、resampler 输入和图片 ID/切片标记 |
| `hunyuan_vl` | `Hunyuan : Patch` | 共用动态像素准备，覆盖尺寸取整、行网格输入和空间位置 |
| `gemma4` | `Gemma4` | 有效 patch 位置、长度补齐及输出 padding 裁剪 |

这些类型在 `VisionProcessor::create()` 中集中选择。`VisionPlan` 只保存批次、裁切网格、视频时间等本次输入的数据，不再用另一组布局类型决定模型行为。公共 resize、切片、patch 打包和分桶函数供家族调用。

音频同样由 `AudioProcessor::create()` 按家族选择：Qwen2Audio、Qwen2.5-Omni、Qwen3-ASR、FunAudioChat 复用 whisper 特征和输入准备流程；LFM2-Audio、Gemma4 分别覆盖 conformer、usm 特征提取。布局、长度限制、窗口 mask、编码器执行及占位 token 组装共用。Talker 的生成与声码器流程保持独立。

## 导出协议

`llm_config.json` 中的 `vision_processor` / `audio_processor` 使用 `version: 2` 和明确的 `family`。配置保留 patch、merge、像素/音频参数、图输入输出布局和注意力窗口大小，不再用 `type`、`position` 和隐含组合选择家族。新模型对应的 Python 子类声明 `processor_family`，在 C++ 工厂注册到兼容的已有实现，或注册需要覆盖局部规则的派生实现。

图像依赖尺寸的缩放、切片及 mask/index 循环保留在 C++。Qwen、Hunyuan、Gemma4 的 patch 重排和补齐在导出图内完成；所有新视觉/音频图通过 `EncoderExport` 输出 `[tokens, batch, hidden]`，并补偿语言模型实际的 `scale_emb`。Python 的 `patch_layout` 仅描述图内像素重排。

没有 descriptor 的历史模型和 `version: 1` 模型在工厂入口解析为对应家族，同时补齐旧布局和缩放约定。未知家族、缺失家族的 v2 描述及未知版本在加载时失败。

## 新模型接入

先逐项对照图像/音频处理、网络输入、媒体标记和位置规则。规则完全相同时注册到已有 processor，数值差异通过配置给出；算法或标记不同则覆盖相应函数。不要仅因模型同属一个家族就假定其所有版本兼容。例如 MiniCPM-V 4.7 的 merger 和 Canvas 位置需要单独适配，不能直接注册到旧 `MiniCPM` 实现。

回归需分别验证历史模型和新导出协议，核对 token、位置、完整拼接 embedding 与回答，覆盖切块、多图、视频/音频，以及关闭视觉/音频的构建。
