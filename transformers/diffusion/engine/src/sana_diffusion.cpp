//
//  sana_diffusion.cpp
//
//  Sana Diffusion 实现
//
//  核心流程：
//  1. LLM特征处理：Qwen3-0.6B输出 -> Connector -> Projector -> prompt_embeds
//  2. Latent初始化：随机噪声或VAE编码的参考图像
//  3. Diffusion去噪：使用DiT模型和Flow Matching调度器逐步去噪
//  4. 图像解码：VAE Decoder将latent解码为最终图像
//

#include <random>
#include <fstream>
#include <chrono>
#include "diffusion/sana_diffusion.hpp"
#include "tokenizer.hpp"
#include "scheduler.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>
#include <fstream>
#include <sstream>
#include <MNN/expr/ExecutorScope.hpp>

#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif

using namespace CV;

namespace MNN
{
    namespace DIFFUSION
    {

        SanaDiffusion::SanaDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode)
            : Diffusion(modelPath, modelType, backendType, memoryMode)
        {
        }

        bool SanaDiffusion::load()
        {
            AUTOTIME;

            // 配置后端
            ScheduleConfig config;
            BackendConfig backendConfig;
            config.type = mBackendType;
            backendConfig.memory = BackendConfig::Memory_Low;

            if (config.type == MNN_FORWARD_CPU)
            {
                config.numThread = 4;
            }
            else if (config.type == MNN_FORWARD_OPENCL)
            {
                config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
            }
            else if (config.type == MNN_FORWARD_METAL)
            {
                backendConfig.precision = BackendConfig::Precision_High;
                config.numThread = 1;
            }
            else
            {
                config.numThread = 1;
            }

            config.backendConfig = &backendConfig;

            auto exe = ExecutorScope::Current();
            exe->lazyEval = false;
            exe->setGlobalExecutorConfig(config.type, backendConfig, config.numThread);

            Module::Config module_config;
            module_config.shapeMutable = false;
            runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));

            // 内存优化配置
            if (mMemoryMode == 0)
            {
                runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
            }
            else if (mMemoryMode == 2)
            {
                runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 1);
            }

            if (config.type == MNN_FORWARD_OPENCL)
            {
                const char *cacheFileName = ".tempcache";
                runtime_manager_->setCache(cacheFileName);
            }

            // 加载模型组件
            // Sana架构：LLM特征 -> Connector -> Projector -> DiT Transformer -> VAE Decoder
            mModules.resize(5);
            {
                // 模块0: Connector - 桥接LLM特征空间
                {
                    std::string model_path = mModelPath + "/connector.mnn";
                    mModules[0].reset(Module::load({"llm_out"}, {"connector_out"}, model_path.c_str(), runtime_manager_, &module_config));
                }

                // 模块1: Projector - 投影到Diffusion特征空间
                {
                    std::string model_path = mModelPath + "/projector.mnn";
                    mModules[1].reset(Module::load({"connector_out"}, {"prompt_embeds"}, model_path.c_str(), runtime_manager_, &module_config));
                }

                // 模块2: DiT Transformer - 核心去噪模型
                // 输入：sample(噪声latent), prompt_embeds(文本特征), timestep(时间步),
                //       encoder_attention_mask(注意力掩码), ref_latents(参考图像latent)
                {
                    std::string model_path = mModelPath + "/transformer.mnn";
                    MNN_PRINT("Loading transformer from %s\n", model_path.c_str());
                    mModules[2].reset(Module::load(
                        {"sample", "encoder_hidden_states", "timestep", "encoder_attention_mask", "ref_latents"},
                        {"noise_pred"},
                        model_path.c_str(), runtime_manager_, &module_config));

                    if (mModules[2] == nullptr)
                    {
                        MNN_ERROR("Failed to load transformer model\n");
                    }
                    else
                    {
                        MNN_PRINT("Transformer loaded successfully\n");
                    }
                }

                // 模块3: VAE Decoder - 将latent解码为图像
                {
                    std::string model_path = mModelPath + "/vae_decoder.mnn";
                    mModules[3].reset(Module::load({"latent_sample"}, {"sample"}, model_path.c_str(), runtime_manager_, &module_config));
                }
            }

            // 模块4: VAE Encoder - 延迟加载（仅img2img模式需要）
            mModules[4] = nullptr;

            return true;
        }

        // VAE Decoder: 将latent解码为图像
        VARP SanaDiffusion::vae_decoder(VARP latent)
        {
            AUTOTIME;

            // 反归一化latent (scaling_factor = 0.41407)
            latent = latent / _Const(0.41407f);

            auto outputs = mModules[3]->onForward({latent});
            auto output = _Convert(outputs[0], NCHW);

            // 后处理：归一化到[0,1]并转换为uint8图像
            auto image = output;
            image = _Minimum(_Maximum(image * _Const(0.5f) + _Const(0.5f), _Const(0.0f)), _Const(1.0f));
            image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
            image = _Cast(_Round(image * _Const(255.0f)), halide_type_of<uint8_t>());
            image = cvtColor(image, COLOR_BGR2RGB);

            return image;
        }

        // VAE Encoder: 将图像编码为latent（用于img2img模式）
        VARP SanaDiffusion::vae_encoder(VARP image)
        {
            AUTOTIME;

            // 延迟加载VAE Encoder（仅img2img模式需要）
            if (mModules[4] == nullptr)
            {
                MNN_PRINT("Loading VAE Encoder (lazy loading for img2img mode)...\n");

                if (runtime_manager_.get() == nullptr)
                {
                    MNN_ERROR("Error: runtime_manager_ is null, cannot load VAE Encoder\n");
                    return nullptr;
                }

                std::string model_path = mModelPath + "/vae_encoder.mnn";
                Module::Config module_config;
                module_config.shapeMutable = false;

                mModules[4].reset(Module::load({"image"}, {"latent"}, model_path.c_str(), runtime_manager_, &module_config));

                if (mModules[4] == nullptr)
                {
                    MNN_ERROR("Failed to load VAE Encoder from %s\n", model_path.c_str());
                    return nullptr;
                }
                MNN_PRINT("VAE Encoder loaded successfully.\n");
            }

            if (image.get() == nullptr)
            {
                MNN_ERROR("Error: input image is null\n");
                return nullptr;
            }

            auto input_var = _Input({1, 3, 512, 512}, NCHW, halide_type_of<float>());
            auto image_ptr = image->readMap<void>();
            auto input_ptr = input_var->writeMap<void>();

            if (image_ptr == nullptr || input_ptr == nullptr)
            {
                MNN_ERROR("Error: failed to map memory for VAE encoder input\n");
                return nullptr;
            }

            ::memcpy(input_ptr, image_ptr, 1 * 3 * 512 * 512 * sizeof(float));

            auto outputs = mModules[4]->onForward({input_var});
            if (outputs.empty() || outputs[0].get() == nullptr)
            {
                MNN_ERROR("Error: VAE encoder forward failed\n");
                return nullptr;
            }

            auto latent_dist = _Convert(outputs[0], NCHW);

            // 归一化latent (scaling_factor = 0.41407)
            auto ref_latents = latent_dist * _Const(0.41407f);
            return ref_latents;
        }

        // 图像预处理：加载、padding、resize
        // 当宽高比>1.5时，padding到正方形以保持内容完整性
        VARP SanaDiffusion::load_and_process_image(const std::string &imagePath, int &orig_width, int &orig_height, int &pad_left, int &pad_top)
        {
            VARP image = imread(imagePath);

            auto info = image->getInfo();
            orig_height = info->dim[0];
            orig_width = info->dim[1];

            MNN_PRINT("Original image size: %dx%d\n", orig_width, orig_height);

            image = cvtColor(image, COLOR_BGR2RGB);

            float aspect_ratio = (float)std::max(orig_width, orig_height) / (float)std::min(orig_width, orig_height);

            pad_left = 0;
            pad_top = 0;

            if (aspect_ratio > aspectRatioThreshold)
            {
                // Padding到正方形
                int max_dim = std::max(orig_width, orig_height);
                pad_left = (max_dim - orig_width) / 2;
                pad_top = (max_dim - orig_height) / 2;
                int pad_right = max_dim - orig_width - pad_left;
                int pad_bottom = max_dim - orig_height - pad_top;

                MNN_PRINT("Aspect ratio %.2f > %.2f, padding to square\n", aspect_ratio, aspectRatioThreshold);
                MNN_PRINT("Padding: left=%d, top=%d, right=%d, bottom=%d\n", pad_left, pad_top, pad_right, pad_bottom);

                std::vector<int> paddings = {pad_top, pad_bottom, pad_left, pad_right, 0, 0};
                auto paddings_var = _Const(paddings.data(), {3, 2}, NCHW, halide_type_of<int>());
                image = _Pad(image, paddings_var, CONSTANT);
            }
            else
            {
                MNN_PRINT("Aspect ratio %.2f <= %.2f, skipping padding (direct resize)\n", aspect_ratio, aspectRatioThreshold);
            }

            // Resize到512x512
            image = resize(image, Size(512, 512), 0, 0, INTER_LINEAR, -1, {0, 0, 0}, {1, 1, 1});

            // 归一化到[-1, 1]
            image = _Cast(image, halide_type_of<float>());
            image = image / _Const(255.0f);
            image = image * _Const(2.0f) - _Const(1.0f);

            // HWC -> NCHW
            image = _Transpose(image, {2, 0, 1});
            image = _Unsqueeze(image, {0});

            return image;
        }

    bool SanaDiffusion::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) {
        return true;
    }
        // 核心推理流程
        // 输入：LLM特征(来自Qwen3-0.6B) -> Connector -> Projector -> DiT去噪 -> VAE解码 -> 输出图像
        bool SanaDiffusion::run(const VARP input_embeds,
                                const std::string &mode,
                                const std::string &inputImagePath,
                                const std::string &outputImagePath,
                                int width,
                                int height,
                                int iterNum,
                                int randomSeed,
                                bool use_cfg,
                                float cfg_scale,
                                std::function<void(int)> progressCallback)
        {
            AUTOTIME;

            // 验证分辨率（必须是32的倍数，因为VAE scale factor是32）
            if (width % 32 != 0 || height % 32 != 0)
            {
                MNN_ERROR("Error: width and height must be multiples of 32. Got %dx%d\n", width, height);
                return false;
            }

            // 当前实现支持的分辨率范围
            if (width < 256 || width > 2048 || height < 256 || height > 2048)
            {
                MNN_ERROR("Error: width and height must be in range [256, 2048]. Got %dx%d\n", width, height);
                return false;
            }

            MNN_PRINT("Target resolution: %dx%d\n", width, height);

            bool is_img2img = (mode == "img2img" || mode == "image_edit");

            int orig_width = width, orig_height = height;
            int pad_left = 0, pad_top = 0;

            // ========== 步骤1: 处理参考图像（仅img2img模式） ==========
            VARP ref_latents;
            if (is_img2img)
            {
                if (inputImagePath.empty())
                {
                    MNN_ERROR("Error: img2img mode requires input image path.\n");
                    return false;
                }

                MNN_PRINT("Loading and processing input image: %s\n", inputImagePath.c_str());
                // 对于img2img，需要将输入图像resize到目标分辨率
                VARP pixel_values = load_and_process_image(inputImagePath, orig_width, orig_height, pad_left, pad_top);

                // TODO: 如果输入图像尺寸与目标尺寸不同，需要resize
                // 当前实现假设输入图像会被resize到目标分辨率

                MNN_PRINT("Running VAE Encoder...\n");
                ref_latents = vae_encoder(pixel_values);

                if (ref_latents.get() == nullptr)
                {
                    MNN_ERROR("Error: VAE encoder returned null\n");
                    return false;
                }

                if (mMemoryMode != 1)
                {
                    ((MNN::Tensor *)(ref_latents->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
                    // 只有当mModules[4]不为空时才reset
                    if (mModules[4] != nullptr)
                    {
                        mModules[4].reset();
                    }
                }
            }
            else
            {
                MNN_PRINT("Text-to-image mode: no reference image needed.\n");
            }

            // ========== 步骤2: LLM特征桥接 ==========
            // Qwen3-0.6B输出 -> Connector -> Projector -> prompt_embeds
            auto llm_out = input_embeds;

            // Connector: 初步转换LLM特征
            MNN_PRINT("Running Connector...\n");
            auto connector_res = mModules[0]->onForward({llm_out});
            auto connector_out = connector_res[0];

            if (mMemoryMode != 1)
            {
                ((MNN::Tensor *)(connector_out->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
                mModules[0].reset();
            }

            // Projector: 投影到Diffusion特征空间
            MNN_PRINT("Running Projector...\n");
            auto projector_res = mModules[1]->onForward({connector_out});
            auto prompt_embeds = projector_res[0];

            if (mMemoryMode != 1)
            {
                ((MNN::Tensor *)(prompt_embeds->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
                mModules[1].reset();
            }

            // ========== 步骤3: 处理CFG（Classifier-Free Guidance） ==========
            auto prompt_info = prompt_embeds->getInfo();
            int batch_size = 1;
            if (prompt_info && prompt_info->dim.size() > 0)
            {
                batch_size = prompt_info->dim[0];
            }
            int seq_len = 256;
            if (prompt_info && prompt_info->dim.size() > 1)
            {
                seq_len = prompt_info->dim[1];
            }

            // 处理CFG：根据use_cfg和batch_size调整prompt_embeds
            if (use_cfg)
            {
                if (batch_size != 2)
                {
                    MNN_ERROR("Error: CFG requires batch_size=2 (negative and positive prompts).\n");
                    return false;
                }

                MNN_PRINT("Using CFG with scale=%.2f\n", cfg_scale);

                // Reorder embeddings for CFG: [Neg, Pos]
                // Input order from LLM is [Pos, Neg], we need to swap
                auto split_res = _Split(prompt_embeds, {2}, 0);
                auto prompt_embeds_pos = split_res[0]; // First is positive
                auto prompt_embeds_neg = split_res[1]; // Second is negative

                // Reorder: [Neg, Pos]
                prompt_embeds = _Concat({prompt_embeds_neg, prompt_embeds_pos}, 0);
            }
            else
            {
                if (batch_size == 2)
                {
                    MNN_PRINT("Warning: batch_size=2 but use_cfg=false, using only first prompt.\n");
                    auto split_res = _Split(prompt_embeds, {2}, 0);
                    prompt_embeds = split_res[0]; // Use only positive prompt
                    batch_size = 1;
                    seq_len = prompt_embeds->getInfo()->dim[1];
                }
                MNN_PRINT("Not using CFG\n");
            }

            // ========== 步骤4: 初始化Latent ==========
            // Latent空间：32通道，尺寸为原图的1/32
            int latent_channels = 32;
            int vae_scale_factor = 32;
            int latent_h = height / vae_scale_factor;
            int latent_w = width / vae_scale_factor;

            MNN_PRINT("Latent size: %dx%d (channels: %d)\n", latent_w, latent_h, latent_channels);

            // 生成随机噪声作为初始latent
            std::vector<float> noise(1 * latent_channels * latent_h * latent_w);
            int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
            std::mt19937 rng(seed);
            std::normal_distribution<float> normal(0, 1);
            for (size_t i = 0; i < noise.size(); ++i)
                noise[i] = normal(rng);

            VARP latents = _Input({1, latent_channels, latent_h, latent_w}, NCHW, halide_type_of<float>());
            memcpy(latents->writeMap<float>(), noise.data(), noise.size() * sizeof(float));

            // CFG模式需要批处理latents
            if (use_cfg)
            {
                latents = _Concat({latents, latents}, 0); // [2, 32, 16, 16]
                MNN_PRINT("Batched latents for CFG: [2, %d, %d, %d]\n", latent_channels, latent_h, latent_w);
            }

            // 处理ref_latents批处理
            VARP ref_latents_batched;
            if (is_img2img)
            {
                // 图像编辑模式：使用实际的ref_latents
                if (ref_latents.get() == nullptr)
                {
                    MNN_ERROR("Error: ref_latents is null in img2img mode\n");
                    return false;
                }

                if (use_cfg)
                {
                    ref_latents_batched = _Concat({ref_latents, ref_latents}, 0); // [2, 32, 16, 16]
                    MNN_PRINT("Batched ref_latents for CFG: [2, %d, %d, %d]\n", latent_channels, latent_h, latent_w);
                }
                else
                {
                    ref_latents_batched = ref_latents; // [1, 32, 16, 16]
                }
            }
            else
            {
                // 文生图模式：创建零ref_latents
                int batch_for_ref = use_cfg ? 2 : 1;
                ref_latents_batched = _Const(0.0f, {batch_for_ref, latent_channels, latent_h, latent_w}, NCHW);
                MNN_PRINT("Created zero ref_latents for text2img mode: [%d, %d, %d, %d]\n",
                          batch_for_ref, latent_channels, latent_h, latent_w);
            }

            if (ref_latents_batched.get() == nullptr)
            {
                MNN_ERROR("Error: ref_latents_batched is null\n");
                return false;
            }

            // ========== 步骤5: 生成时间步序列 ==========
            // 使用Flow Matching调度器（蒸馏加速的关键）
            int num_inference_steps = iterNum;
            std::vector<double> timesteps;
            std::vector<double> sigmas;

            // 生成线性插值的时间步：从1.0到1/1000
            for (int i = 0; i < num_inference_steps; ++i)
            {
                double start = 1.0;
                double end = 1.0 / 1000.0;
                double alpha = start + i * (end - start) / (double)(num_inference_steps - 1);
                double sigma = 1.0 - alpha;

                // Flow shift: 调整噪声分布，加速收敛
                double flow_shift = 3.0;
                sigma = flow_shift * sigma / (1.0 + (flow_shift - 1.0) * sigma);

                sigmas.push_back(sigma);
                timesteps.push_back(sigma * 1000.0);
            }

            // 反转：从高噪声到低噪声
            std::reverse(timesteps.begin(), timesteps.end());
            std::reverse(sigmas.begin(), sigmas.end());

            // ========== 步骤6: DiT去噪循环 ==========
            // 通过蒸馏，可以用较少步数（如20步）达到高质量效果
            MNN_PRINT("Starting Denoising Loop (%d steps)...\n", num_inference_steps);
            VARP sample = latents;

            // Attention mask：根据batch_size调整
            int mask_batch = use_cfg ? 2 : 1;
            VARP encoder_attention_mask = _Input({mask_batch, seq_len}, NCHW, halide_type_of<float>());
            std::fill(encoder_attention_mask->writeMap<float>(),
                      encoder_attention_mask->writeMap<float>() + mask_batch * seq_len, 1.0f);

            for (int i = 0; i < num_inference_steps; ++i)
            {
                AUTOTIME;

                double t = timesteps[i];

                MNN_PRINT("Step %d/%d: t=%f\n", i + 1, num_inference_steps, t);

                // Timestep：根据batch_size调整
                VARP timestep_var = _Input({mask_batch}, NCHW, halide_type_of<float>());
                auto t_ptr = timestep_var->writeMap<float>();
                for (int b = 0; b < mask_batch; ++b)
                {
                    t_ptr[b] = (float)t;
                }

                // DiT Transformer推理：预测噪声

                auto res = mModules[2]->onForward({sample, prompt_embeds, timestep_var, encoder_attention_mask, ref_latents_batched});

                if (res.empty())
                {
                    MNN_ERROR("Error: Transformer returned empty result at step %d\n", i + 1);
                    return false;
                }

                if (res[0].get() == nullptr)
                {
                    MNN_ERROR("Error: Transformer returned null at step %d\n", i + 1);
                    return false;
                }

                auto noise_pred = _Convert(res[0], NCHW);

                // 应用CFG（Classifier-Free Guidance）
                VARP noise_pred_guided;
                if (use_cfg)
                {
                    // 分离条件和无条件预测
                    auto split_res = _Split(noise_pred, {2}, 0);
                    auto noise_pred_uncond = split_res[0]; // 负样本（无条件）
                    auto noise_pred_text = split_res[1];   // 正样本（有条件）

                    // CFG公式：guided = uncond + scale * (cond - uncond)
                    noise_pred_guided = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * _Const(cfg_scale);
                }
                else
                {
                    noise_pred_guided = noise_pred;
                }

                // Euler采样步骤
                float dt;
                if (i < num_inference_steps - 1)
                {
                    dt = timesteps[i + 1] - t;
                }
                else
                {
                    dt = -t;
                }

                sample = sample + noise_pred_guided * _Const(dt / 1000.0f);

                // CFG模式：保持batch维度一致
                if (use_cfg)
                {
                    auto sample_split = _Split(sample, {2}, 0);
                    sample = sample_split[0];
                    sample = _Concat({sample, sample}, 0);
                }
            }

            if (mMemoryMode != 1)
            {
                ((MNN::Tensor *)(sample->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
                mModules[2].reset();
            }

            // 提取最终latents
            VARP final_latents;
            if (use_cfg)
            {
                auto final_split = _Split(sample, {2}, 0);
                final_latents = final_split[0];
            }
            else
            {
                final_latents = sample;
            }

            // ========== 步骤7: VAE解码 ==========
            MNN_PRINT("Running VAE Decoder...\n");
            VARP image = vae_decoder(final_latents);
            image.fix(VARP::CONSTANT);

            // ========== 步骤8: 保存结果 ==========
            bool success = imwrite(outputImagePath, image);
            if (success)
            {
                MNN_PRINT("SUCCESS! Generated image saved to %s\n", outputImagePath.c_str());
            }

            if (mMemoryMode != 1)
            {
                mModules[3].reset();
            }

            return true;
        }

    } // namespace DIFFUSION
} // namespace MNN
