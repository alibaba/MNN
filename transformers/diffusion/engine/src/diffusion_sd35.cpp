//
//  diffusion_sd35.cpp
//
//  Created by zlaa on 2025/12/18.
//

#include <random>
#include <fstream>
#include <chrono>
#include "diffusion/diffusion_sd35.hpp"
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

namespace MNN {
namespace DIFFUSION {

DiffusionSD35* DiffusionSD35::createDiffusionSD35(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode) {
    return new DiffusionSD35(modelPath, modelType, backendType, memoryMode);
}

DiffusionSD35::DiffusionSD35(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode) :
mModelPath(modelPath), mModelType(modelType), mBackendType(backendType), mMemoryMode(memoryMode) {
    // Initialize tokenizers
    mTokenizer1.reset(new CLIPTokenizer);
    mTokenizer2.reset(new CLIPTokenizer);
    mTokenizer3.reset(new T5Tokenizer);
}

DiffusionSD35::~DiffusionSD35() {
    mModules.clear();
    runtime_manager_.reset();
}

bool DiffusionSD35::load() {
    AUTOTIME;
    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = mBackendType;
    if(config.type == MNN_FORWARD_CPU) {
        config.numThread = 4;
    } else if(config.type == MNN_FORWARD_OPENCL) {
        // Use only BUFFER mode for stability and memory saving
        config.mode = MNN_GPU_MEMORY_BUFFER; 
    } else {
        config.numThread = 1;
    }
    backendConfig.memory = BackendConfig::Memory_Low;
    backendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    
    auto exe = ExecutorScope::Current();
    exe->lazyEval = false;
    exe->setGlobalExecutorConfig(config.type, backendConfig, config.numThread);
    
    Module::Config module_config;
    module_config.shapeMutable = false;
    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    
    if (config.type == MNN_FORWARD_OPENCL) {
        const char* cacheFileName = ".tempcache_sd35";
        runtime_manager_->setCache(cacheFileName);
        // Force disable Winograd to save memory if needed, or keep level 0
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0); 
    }
    
    // Load models
    mModules.resize(5);
    std::vector<std::string> modelNames = {
        "text_encoder.mnn",
        "text_encoder_2.mnn",
        "text_encoder_3.mnn",
        "transformer.mnn",
        "vae_decoder.mnn"
    };

    for(int i=0; i<5; ++i) {
        std::string model_path = mModelPath + "/" + modelNames[i];
        MNN_PRINT("Load %s\n", model_path.c_str());
        
        std::vector<std::string> inputs, outputs;
        if (i == 0) { // text_encoder
            inputs = {"input_ids"};
            outputs = {"last_hidden_state", "text_embeds"};
        } else if (i == 1) { // text_encoder_2
            inputs = {"input_ids"};
            outputs = {"last_hidden_state", "text_embeds"};
        } else if (i == 2) { // text_encoder_3
            inputs = {"input_ids"};
            outputs = {"last_hidden_state"};
        } else if (i == 3) { // transformer
            inputs = {"hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"};
            outputs = {"out_hidden_states"};
        } else if (i == 4) { // vae_decoder
            inputs = {"latent_sample"};
            outputs = {"sample"};
        }
        
        mModules[i].reset(Module::load(inputs, outputs, model_path.c_str(), runtime_manager_, &module_config));
        if (!mModules[i]) {
            MNN_ERROR("Failed to load model %s\n", model_path.c_str());
            if (i == 2) {
                MNN_PRINT("Warning: T5 text encoder failed to load. Will use zeros.\n");
            } else {
                return false;
            }
        }
    }

    // Load tokenizers
    mTokenizer1->load(mModelPath + "/tokenizer");
    mTokenizer2->load(mModelPath + "/tokenizer_2");
    mTokenizer3->load(mModelPath + "/tokenizer_3");

    // Resize fix
    for (auto& m : mModules) {
        if(m) m->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    }

    return true;
}

std::pair<VARP, VARP> DiffusionSD35::encode_prompt(const std::string& prompt) {
    auto run_encoder = [&](int module_index, VARP input_ids, const char* name) {
        auto outputs = mModules[module_index]->onForward({input_ids});
        std::vector<VARP> safe_outputs;
        for (auto& out : outputs) {
            auto info = out->getInfo();
            const void* ptr = out->readMap<void>();
            auto new_var = _Const(ptr, info->dim, info->order, info->type);
            safe_outputs.push_back(new_var);
        }
        if (mMemoryMode != 1) {
            mModules[module_index].reset();
            MNN_PRINT("%s Module unloaded.\n", name);
            MNN::Express::Executor::getGlobalExecutor()->gc(MNN::Express::Executor::FULL);
        }
        return safe_outputs;
    };

    // 1. CLIP L
    auto ids1 = mTokenizer1->encode(prompt, mMaxTextLen);
    VARP input_ids1 = _Input({2, mMaxTextLen}, NCHW, halide_type_of<int>());
    int* inputs1_ptr = input_ids1->writeMap<int>();
    memset(inputs1_ptr, 0, 2 * mMaxTextLen * sizeof(int));
    if (ids1.size() > 0) {
        size_t copy_size = std::min(ids1.size(), (size_t)(2 * mMaxTextLen));
        memcpy(inputs1_ptr, ids1.data(), copy_size * sizeof(int));
    }
    
    auto out1 = run_encoder(0, input_ids1, "CLIP L");
    auto clip_l_hidden = out1[0]; // (2, 77, 768)
    auto clip_l_pooled = out1[1]; // (2, 768)
    
    // 2. CLIP G
    auto ids2 = mTokenizer2->encode(prompt, mMaxTextLen);
    VARP input_ids2 = _Input({2, mMaxTextLen}, NCHW, halide_type_of<int>());
    int* inputs2_ptr = input_ids2->writeMap<int>();
    memset(inputs2_ptr, 0, 2 * mMaxTextLen * sizeof(int));
    if (ids2.size() > 0) {
        size_t copy_size = std::min(ids2.size(), (size_t)(2 * mMaxTextLen));
        memcpy(inputs2_ptr, ids2.data(), copy_size * sizeof(int));
    }
    
    auto out2 = run_encoder(1, input_ids2, "CLIP G");
    auto clip_g_hidden = out2[0]; // (2, 77, 1280)
    auto clip_g_pooled = out2[1]; // (2, 1280)
    
    // 3. T5
    VARP t5_hidden;
    if (mModules[2]) {
        auto ids3 = mTokenizer3->encode(prompt, mMaxTextLenT5);
        VARP input_ids3 = _Input({2, mMaxTextLenT5}, NCHW, halide_type_of<int>());
        
        // Uncond for T5 is empty string
        auto ids3_uncond = mTokenizer3->encode("", mMaxTextLenT5);
        
        int* inputs3_ptr = input_ids3->writeMap<int>();
        memset(inputs3_ptr, 0, 2 * mMaxTextLenT5 * sizeof(int)); 
        memcpy(inputs3_ptr, ids3_uncond.data(), mMaxTextLenT5 * sizeof(int));
        memcpy(inputs3_ptr + mMaxTextLenT5, ids3.data(), mMaxTextLenT5 * sizeof(int));
        
        auto out3 = run_encoder(2, input_ids3, "T5");
        t5_hidden = out3[0];
    } else {
        t5_hidden = _Const(0.0f, {2, mMaxTextLenT5, 4096}, NCHW);
    }
    
    // Concatenate CLIP hidden states
    auto clip_hidden = _Concat({clip_l_hidden, clip_g_hidden}, -1); // (2, 77, 2048)
    
    // Pad CLIP hidden states to match T5 dimension (4096)
    // We need to pad the last dimension from 2048 to 4096
    auto padding = _Const(0.0f, {2, mMaxTextLen, 4096 - 2048}, NCHW);
    auto clip_hidden_padded = _Concat({clip_hidden, padding}, -1); // (2, 77, 4096)
    
    // Concatenate CLIP and T5 hidden states along sequence dimension
    auto encoder_hidden_states = _Concat({clip_hidden_padded, t5_hidden}, 1); // (2, 77+256, 4096)
    
    // Concatenate pooled projections
    auto pooled_projections = _Concat({clip_l_pooled, clip_g_pooled}, -1); // (2, 2048)
    
    return {encoder_hidden_states, pooled_projections};
}

VARP DiffusionSD35::step_flow_match(VARP sample, VARP model_output, int index, int num_steps) {
    // Flow Matching Euler Step
    // sigma goes from 1.0 to 0.0
    // t = sigma
    // dt = sigma_next - sigma
    // x_next = x + v * dt
    
    float sigma = mTimeSteps[index] / 1000.0f;
    float sigma_next = (index + 1 < mTimeSteps.size()) ? (mTimeSteps[index+1] / 1000.0f) : 0.0f;
    float dt = sigma_next - sigma;
    
    auto prev_sample = sample + model_output * _Const(dt);
    return prev_sample;
}

VARP DiffusionSD35::transformer(VARP hidden_states, VARP encoder_hidden_states, VARP pooled_projections, VARP timestep) {
    auto outputs = mModules[3]->onForward({hidden_states, encoder_hidden_states, pooled_projections, timestep});
    return outputs[0];
}

VARP DiffusionSD35::vae_decoder(VARP latent) {
    float scaling_factor = 1.5305f;
    float shift_factor = 0.0609f;
    
    latent = (latent * _Const(1.0f / scaling_factor)) + _Const(shift_factor);
    
    auto outputs = mModules[4]->onForward({latent});
    auto image = outputs[0];
    
    image = image * _Const(0.5f) + _Const(0.5f);
    image = _Maximum(_Minimum(image, _Const(1.0f)), _Const(0.0f));
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());
    image = cvtColor(image, COLOR_BGR2RGB);
    return image;
}

bool DiffusionSD35::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) {
    AUTOTIME;

    auto unload_module = [&](int index, const char* name) {
        if (mMemoryMode != 1) {
            mModules[index].reset();
            MNN_PRINT("%s Module unloaded.\n", name);
            MNN::Express::Executor::getGlobalExecutor()->gc(MNN::Express::Executor::FULL);
        }
    };
    
    mTimeSteps.resize(iterNum);
    if (iterNum == 1) {
        mTimeSteps[0] = 1000.0f;
    } else {
        float shift = 3.0f;
        float start = 1.0f;
        float end = 1.0f / 1000.0f;
        for(int i = 0; i < iterNum; i++) {
            float t_linear = start + i * (end - start) / (iterNum - 1);
            float t_shifted = (shift * t_linear) / (1.0f + (shift - 1.0f) * t_linear);
            mTimeSteps[i] = t_shifted * 1000.0f;
        }
    }
    
    // Encode prompt
    auto encoded = encode_prompt(prompt);
    auto encoder_hidden_states = encoded.first;
    auto pooled_projections = encoded.second;
    
    int height = 1024;
    int width = 1024;
    int channels = 16;
    int latents_height = 128;
    int latents_width = 128;
    
    // Random noise
    int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0, 1);
    
    int size = 1 * channels * latents_height * latents_width;
    std::vector<float> noise(size);
    for(int i=0; i<size; i++) {
        noise[i] = normal(rng);
    }
    
    mLatentVar = _Input({1, channels, latents_height, latents_width}, NCHW, halide_type_of<float>());
    memcpy((void *)mLatentVar->writeMap<float>(), noise.data(), size * sizeof(float));
    
    auto sample = mLatentVar;
    
    // Guidance scale for CFG
    float guidance_scale = 5.0f; // Default for SD3

    for (int i = 0; i < iterNum; i++) {
        AUTOTIME;
        float t = mTimeSteps[i];
        
        // Prepare inputs for CFG (Batch Size = 2)
        auto sample_input = _Concat({sample, sample}, 0); // (2, 16, 64, 64)
        auto timestep = _Const(t, {2}, NCHW); // (2)
        
        auto model_output = transformer(sample_input, encoder_hidden_states, pooled_projections, timestep);
        
        // Perform CFG
        auto split_output = _Split(model_output, {2}, 0);
        auto noise_pred_uncond = split_output[0];
        auto noise_pred_text = split_output[1];
        
        auto noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * _Const(guidance_scale);
        
        sample = step_flow_match(sample, noise_pred, i, iterNum);
        
        if (progressCallback) {
            progressCallback((i + 1) * 100 / iterNum);
        }
    }

    unload_module(3, "Transformer");
    
    auto image = vae_decoder(sample);
    bool res = imwrite(imagePath, image);
    if (res) {
        MNN_PRINT("SUCCESS! write generated image to %s\n", imagePath.c_str());
    }

    unload_module(4, "VAE Decoder");
    
    return true;
}

}
}
