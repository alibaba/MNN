//
//  sana_diffusion.hpp
//
//  Created by MNN on 2025/01/12.
//  MNN
//
//  Sana Diffusion 架构说明：
//
//  Sana是一个高效的文生图模型，采用以下创新架构：
//
//  1. 文本编码器 (Text Encoder)
//     - 使用 Qwen3-0.6B 作为文本编码器（替代传统的CLIP）
//     - 优势：更强的语义理解能力，支持更复杂的文本描述
//     - 输出：文本特征向量 (llm_out)
//
//  2. 特征桥接 (Feature Bridge)
//     - Connector: 将LLM输出特征进行初步转换
//     - Projector: 将特征投影到Diffusion模型所需的空间
//     - 作用：桥接LLM特征空间和Diffusion特征空间
//
//  3. Diffusion模型 (Diffusion Model)
//     - 基于DiT (Diffusion Transformer) 架构
//     - 采用Flow Matching调度器进行采样
//     - 支持文生图(text2img)和图像编辑(img2img)两种模式
//
//  4. 加速技术
//     - 采用知识蒸馏技术，从大模型蒸馏到小模型
//     - 减少推理步数，提升生成速度
//     - 保持高质量的同时实现快速生成
//
//  5. 图像编码/解码
//     - VAE Encoder: 将输入图像编码为latent表示（用于img2img）
//     - VAE Decoder: 将latent解码为最终图像
//
#ifndef MNN_SANA_DIFFUSION_HPP
#define MNN_SANA_DIFFUSION_HPP

#include "diffusion.hpp"

namespace MNN
{
    namespace DIFFUSION
    {

        class MNN_PUBLIC SanaDiffusion : public Diffusion
        {
        public:
            SanaDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
            virtual ~SanaDiffusion() = default;

            // 加载所有模型组件
            virtual bool load() override;

            virtual bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) override;

            // 统一的生成接口
            // 参数说明：
            //   input_embeds: LLM输出的文本特征 (来自Qwen3-0.6B)
            //   mode: "text2img" 文生图模式, "img2img" 图像编辑模式
            //   inputImagePath: 输入图像路径（img2img模式必需，text2img模式忽略）
            //   outputImagePath: 输出图像路径
            //   width: 输出图像宽度（支持512, 1024等）
            //   height: 输出图像高度（支持512, 1024等）
            //   iterNum: 推理步数（通过蒸馏，可以用较少步数获得高质量结果）
            //   randomSeed: 随机种子
            //   use_cfg: 是否使用Classifier-Free Guidance（需要input_embeds包含正负样本）
            //   cfg_scale: CFG引导强度（仅use_cfg=true时生效）
            //   progressCallback: 进度回调函数
            virtual bool run(const VARP input_embeds,
                             const std::string &mode,
                             const std::string &inputImagePath,
                             const std::string &outputImagePath,
                             int width,
                             int height,
                             int iterNum,
                             int randomSeed,
                             bool use_cfg,
                             float cfg_scale,
                             std::function<void(int)> progressCallback) override;

        private:
            // VAE解码器：将latent解码为图像
            VARP vae_decoder(VARP latent);

            // VAE编码器：将图像编码为latent（用于img2img模式）
            VARP vae_encoder(VARP image);

            // 图像预处理：加载图像并进行padding和resize
            // 当宽高比>aspectRatioThreshold时，会padding到正方形以保持内容完整性
            VARP load_and_process_image(const std::string &imagePath, int &orig_width, int &orig_height, int &pad_left, int &pad_top);

        private:
            std::vector<float> mInitNoise;
            int mNumInferenceSteps = 20;
            int mNumTrainTimesteps = 1000;
            const float aspectRatioThreshold = 1.5f;
        };

    } // namespace DIFFUSION
} // namespace MNN

#endif // MNN_SANA_DIFFUSION_HPP
