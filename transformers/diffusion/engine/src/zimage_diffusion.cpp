//
//  zimage_diffusion.cpp
//  ZImage FlowMatch Euler diffusion model
//
//  Extracted from the monolithic diffusion.cpp during upstream rebase.
//  Implements: ZImage text-to-image with FlowMatch Euler scheduler,
//  Philox RNG, configurable GPU memory/precision modes.
//
#include <random>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include "diffusion/zimage_diffusion.hpp"
#include "diffusion/diffusion_config.hpp"
#include "tokenizer.hpp"
#include "scheduler.hpp"
#ifdef MNN_BUILD_LLM
#include "llm/tokenizer.hpp"
#include "llm/llm.hpp"
#endif
#include <rapidjson/document.h>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>
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

// ===== LlmTokenizerWrapper for ZImage =====
#ifdef MNN_BUILD_LLM
class LlmTokenizerWrapper : public Tokenizer {
public:
    class DiffusionHuggingfaceTokenizer : public MNN::Transformer::HuggingfaceTokenizer {
    public:
        bool loadFromFile(const std::string& filename) {
            std::ifstream file(filename.c_str());
            if (!file.is_open()) return false;
            load_special(file);
            return load_vocab(file);
        }
        std::vector<int> encodeWithSpecial(const std::string& str) {
            return MNN::Transformer::Tokenizer::encode(str);
        }
    };

    LlmTokenizerWrapper() = default;
    virtual ~LlmTokenizerWrapper() = default;

    virtual bool load(const std::string& filePath) override {
        std::string tokPath = filePath + "/tokenizer.txt";
        std::ifstream check(tokPath.c_str());
        if (!check.good()) {
            MNN_PRINT("Error: tokenizer.txt not found at %s\n", tokPath.c_str());
            return false;
        }
        check.close();
        std::shared_ptr<DiffusionHuggingfaceTokenizer> impl(new DiffusionHuggingfaceTokenizer);
        if (!impl->loadFromFile(tokPath)) {
            MNN_PRINT("Error: failed to load tokenizer vocab from %s\n", tokPath.c_str());
            return false;
        }
        mTokenizer = impl;
        return true;
    }

    virtual std::vector<int> encode(const std::string& sentence, int maxlen = 0) override {
        std::vector<int> packed;
        auto impl = std::static_pointer_cast<DiffusionHuggingfaceTokenizer>(mTokenizer);
        if (!impl) return packed;
        std::vector<int> baseIds = impl->encodeWithSpecial(sentence);
        if (maxlen <= 0) maxlen = (int)baseIds.size();
        packed.assign(maxlen * 2, 0);
        int n = std::min((int)baseIds.size(), maxlen);
        for (int i = 0; i < n; ++i) {
            packed[i] = baseIds[i];
            packed[maxlen + i] = 1;
        }
        return packed;
    }
private:
    std::shared_ptr<DiffusionHuggingfaceTokenizer> mTokenizer;
};
#endif

// ===== ZImageDiffusion Implementation =====

ZImageDiffusion::ZImageDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode,
                                 int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU,
                                 DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode,
                                 DiffusionCFGMode cfgMode, int numThreads)
    : Diffusion(modelPath, modelType, backendType, memoryMode) {
    mTextEncoderOnCPU = textEncoderOnCPU;
    mVaeOnCPU = vaeOnCPU;
    mGpuMemoryMode = gpuMemoryMode;
    mPrecisionMode = precisionMode;
    mCFGMode = cfgMode;
    mNumThreads = numThreads;
    mImageWidth = imageWidth;
    mImageHeight = imageHeight;
    
    mMaxTextLen = 128;
    mTrainTimestepsNum = 1000;
    mFlowShift = 3.0f;
    mUseDynamicShifting = false;
    
    // Load scheduler config if available
    std::string schedPath1 = mModelPath + "/scheduler/scheduler_config.json";
    std::string schedPath2 = mModelPath + "/scheduler_config.json";
    std::string schedPath;
    std::ifstream sfile;
    sfile.open(schedPath1.c_str());
    if (sfile.good()) { schedPath = schedPath1; }
    else { sfile.close(); sfile.open(schedPath2.c_str()); if (sfile.good()) schedPath = schedPath2; }
    if (!schedPath.empty()) {
        std::ostringstream oss;
        oss << sfile.rdbuf();
        sfile.close();
        rapidjson::Document doc;
        doc.Parse(oss.str().c_str());
        if (!doc.HasParseError() && doc.IsObject()) {
            if (doc.HasMember("num_train_timesteps") && doc["num_train_timesteps"].IsInt())
                mTrainTimestepsNum = doc["num_train_timesteps"].GetInt();
            if (doc.HasMember("shift") && (doc["shift"].IsFloat() || doc["shift"].IsDouble()))
                mFlowShift = static_cast<float>(doc["shift"].GetDouble());
            if (doc.HasMember("use_dynamic_shifting") && doc["use_dynamic_shifting"].IsBool())
                mUseDynamicShifting = doc["use_dynamic_shifting"].GetBool();
        }
    }
    
    // Set latent dimensions
    mLatentC = 16;
    if (mImageWidth > 0 && mImageHeight > 0) {
        int w = (mImageWidth / 8) * 8;
        int h = (mImageHeight / 8) * 8;
        if (w < 256) w = 256; if (h < 256) h = 256;
        if (w > 1280) w = 1280; if (h > 1280) h = 1280;
        mImageWidth = w; mImageHeight = h;
        mLatentW = w / 8; mLatentH = h / 8;
    } else {
        mLatentH = 128; mLatentW = 128;
        mImageWidth = 1024; mImageHeight = 1024;
    }
    MNN_PRINT("[ZImage] latent=(1,%d,%d,%d), image=%dx%d\n", mLatentC, mLatentH, mLatentW, mImageWidth, mImageHeight);
}

ZImageDiffusion::~ZImageDiffusion() {
}

bool ZImageDiffusion::load() {
    AUTOTIME;
    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = mBackendType;
    if (config.type == MNN_FORWARD_CPU) {
        config.numThread = mNumThreads;
    } else if (config.type == MNN_FORWARD_OPENCL) {
        int gpuMode = MNN_GPU_TUNING_FAST;
        if (mGpuMemoryMode == GPU_MEMORY_BUFFER) gpuMode |= MNN_GPU_MEMORY_BUFFER;
        else if (mGpuMemoryMode == GPU_MEMORY_IMAGE) gpuMode |= MNN_GPU_MEMORY_IMAGE;
        else gpuMode |= MNN_GPU_MEMORY_BUFFER;  // AUTO: BUFFER mode for ZImage
        config.mode = gpuMode;
    } else {
        config.numThread = 1;
    }
    backendConfig.memory = BackendConfig::Memory_Low;
    
    if (mPrecisionMode == PRECISION_LOW) backendConfig.precision = BackendConfig::Precision_Low;
    else if (mPrecisionMode == PRECISION_NORMAL) backendConfig.precision = BackendConfig::Precision_Normal;
    else if (mPrecisionMode == PRECISION_HIGH) backendConfig.precision = BackendConfig::Precision_High;
    else {
        // AUTO: ZImage requires FP32 for GPU (-inf handling in attention mask)
        if (config.type == MNN_FORWARD_OPENCL || config.type == MNN_FORWARD_VULKAN) {
            backendConfig.precision = BackendConfig::Precision_High;  // FP32
        } else {
            backendConfig.precision = BackendConfig::Precision_Normal;  // FP32 for CPU
        }
    }
    config.backendConfig = &backendConfig;
    
    auto exe = ExecutorScope::Current();
    exe->lazyEval = false;
    exe->setGlobalExecutorConfig(config.type, backendConfig, config.numThread);
    
    Module::Config module_config;
    module_config.shapeMutable = true;
    // module_config.rearrange = true;  // not set in old version
    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    if (runtime_manager_ == nullptr) {
        MNN_ERROR("ZImage: Failed to create runtime manager\n");
        return false;
    }
    
    if (config.type == MNN_FORWARD_OPENCL) {
        const char* cacheFileName = ".tempcache";
        runtime_manager_->setCache(cacheFileName);
    }
    if (mMemoryMode == 0) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
    } else if (mMemoryMode == 2) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 1);
    }
    if (config.type == MNN_FORWARD_CPU) {
        runtime_manager_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 0);
    }
    
    // CPU runtime for text encoder (avoid OpenCL buffer size limit)
    if (mTextEncoderOnCPU && (config.type == MNN_FORWARD_OPENCL || config.type == MNN_FORWARD_VULKAN)) {
        ScheduleConfig cpuConfig;
        cpuConfig.type = MNN_FORWARD_CPU;
        cpuConfig.numThread = mNumThreads;
        BackendConfig cpuBackendConfig;
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
        cpuBackendConfig.precision = BackendConfig::Precision_Normal;
        cpuConfig.backendConfig = &cpuBackendConfig;
        runtime_manager_cpu_.reset(Executor::RuntimeManager::createRuntimeManager(cpuConfig));
        runtime_manager_cpu_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 0);
    }
    
    // Create input variables
    mLatentVar = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
    mPromptVar = _Input({1, mMaxTextLen}, NCHW, halide_type_of<int>());
    mAttentionMaskVar = _Input({1, mMaxTextLen}, NCHW, halide_type_of<int>());
    mTimestepVar = _Input({1}, NCHW, halide_type_of<float>());
    mLatentVar->writeMap<float>();
    mPromptVar->writeMap<int>();
    mAttentionMaskVar->writeMap<int>();
    mTimestepVar->writeMap<float>();
    mSampleVar = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
    mSampleVar->writeMap<float>();
    
    DiffusionConfig diff_config(mModelPath);
    
    mModules.resize(3);
    // Load text encoder
    {
        std::string model_path = diff_config.text_encoder_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        auto& te_runtime = runtime_manager_cpu_ ? runtime_manager_cpu_ : runtime_manager_;
        mModules[0].reset(Module::load({"input_ids", "attention_mask"}, {"last_hidden_state"}, model_path.c_str(), te_runtime, &module_config));
    }
    // Load UNet
    {
        std::string model_path = diff_config.unet_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[1].reset(Module::load({"sample", "timestep", "encoder_hidden_states"}, {"out_sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    // Load VAE decoder
    {
        std::string model_path = diff_config.vae_decoder_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[2].reset(Module::load({"latent_sample"}, {"sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    
    // Load tokenizer
#ifdef MNN_BUILD_LLM
    mTokenizer.reset(new LlmTokenizerWrapper);
#else
    MNN_PRINT("Error: ZImage requires MNN_BUILD_LLM enabled.\n");
    return false;
#endif
    if (!mTokenizer->load(mModelPath)) {
        MNN_PRINT("Error: failed to load tokenizer from %s\n", mModelPath.c_str());
        return false;
    }
    
    // Resize fix (skip UNet due to dynamic shapes)
    for (int i = 0; i < (int)mModules.size(); ++i) {
        if (i == 1 || !mModules[i]) continue;
        mModules[i]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    }
    
    // UNet preprocessing: pass-through (ZImage outputs [B, C, H, W] directly)
    mUNetPreprocess = [](VARP unet_output) -> VARP { return unet_output; };
    mSchedulerType = SCHEDULER_EULER;
    
    return true;
}

VARP ZImageDiffusion::text_encoder(const std::vector<int>& ids) {
    AUTOTIME;
    int expected = mMaxTextLen * 2;
    memcpy((void*)mPromptVar->writeMap<int8_t>(), ids.data(), mMaxTextLen * sizeof(int));
    memcpy((void*)mAttentionMaskVar->writeMap<int8_t>(), ids.data() + mMaxTextLen, mMaxTextLen * sizeof(int));
    
    auto outputs = mModules[0]->onForward({mPromptVar, mAttentionMaskVar});
    auto output = outputs[0];  // keep original [B, L, D] layout for UNet
    output.fix(VARP::CONSTANT);
    return output;
}

VARP ZImageDiffusion::applyEulerUpdate(VARP sample, VARP noise_pred, float dt) {
    return sample + _Scalar(dt) * noise_pred;
}

VARP ZImageDiffusion::unet(VARP text_embeddings, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback) {
    if (mMemoryMode != 1) {
        mModules[0].reset();  // Unload text encoder
    }
    
    int latentSize = mLatentC * mLatentH * mLatentW;
    if ((int)mInitNoise.size() != latentSize) {
        mInitNoise.resize(latentSize);
    }
    
    // Generate random noise using Philox RNG (aligned with PyTorch)
    int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
    PhiloxRNG rng(seed);
    for (int i = 0; i < latentSize; i++) {
        mInitNoise[i] = rng.randn();
    }
    
    memcpy((void*)mLatentVar->writeMap<float>(), mInitNoise.data(), latentSize * sizeof(float));
    
    // Create a separate buffer for plms to allow in-place updates
    // Copy initial noise from mLatentVar using MNN's input() method (GPU-side copy)
    auto plms = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
    plms->input(mLatentVar);
    
    auto floatVar = _Input({1}, NCHW, halide_type_of<float>());
    auto ptr = floatVar->writeMap<float>();
    
    for (int i = 0; i < (int)mTimeSteps.size(); i++) {
        AUTOTIME;
        
        float sigma = mSigmas[i];
        float sigma_next = (i + 1 < (int)mSigmas.size()) ? mSigmas[i + 1] : 0.0f;
        float dt = sigma_next - sigma;
        
        // Set timestep: Z-image uses t = 1 - sigma
        float t = 1.0f - sigma;
        ptr[0] = t;
        mTimestepVar->input(floatVar);
        
        // Copy plms to pre-allocated mSampleVar (GPU-side copy for computation graph isolation)
        mSampleVar->input(plms);
        VARP sample_input = mSampleVar;
        VARP noise_pred;
        
        // No CFG: run UNet once
        std::vector<VARP> unet_inputs = {sample_input, mTimestepVar, text_embeddings};
        auto outputs = mModules[1]->onForward(unet_inputs);
        if (outputs.empty() || !outputs[0].get()) {
            MNN_PRINT("[ZImage UNet] ERROR: outputs empty!\n");
            return nullptr;
        }
        auto output = _Convert(outputs[0], NCHW);
        
        // ZImage: negate UNet output (Python convention)
        noise_pred = _Scalar(-1.0f) * output;
        
        // Apply CFG scaling if needed
        if (std::abs(cfgScale - 1.0f) > 0.001f) {
            noise_pred = _Scalar(cfgScale) * noise_pred;
        }
        
        // Layer 1: Preprocess (pass-through for ZImage)
        auto noise_pred_standard = mUNetPreprocess(noise_pred);
        
        // Layer 2: Euler update
        auto updated = applyEulerUpdate(plms, noise_pred_standard, dt);
        plms->input(updated);
        
        // Cleanup
        noise_pred = nullptr;
        noise_pred_standard = nullptr;
        
        if (mBackendType == MNN_FORWARD_OPENCL && (i + 1) % 2 == 0) {
            MNN::Express::ExecutorScope::Current()->gc(MNN::Express::Executor::PART);
        }
        
        if (progressCallback) {
            progressCallback((2 + i) * 100 / (iterNum + 3));
        }
    }
    plms.fix(VARP::CONSTANT);
    return plms;
}

VARP ZImageDiffusion::vae_decoder(VARP latent) {
    if (mMemoryMode != 1) {
        mModules[1].reset();  // Unload UNet
    }
    
    // Z-image uses standard SD VAE scaling
    latent = latent * _Const(1 / 0.18215);
    
    AUTOTIME;
    auto outputs = mModules[2]->onForward({latent});
    auto output = _Convert(outputs[0], NCHW);
    
    auto image = output;
    image = _Relu6(image * _Const(0.5) + _Const(0.5), 0, 1);
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());
    image = cvtColor(image, COLOR_BGR2RGB);
    image.fix(VARP::CONSTANT);
    return image;
}

bool ZImageDiffusion::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) {
    return run(prompt, imagePath, iterNum, randomSeed, 1.0f, progressCallback, "");
}

bool ZImageDiffusion::run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath) {
    AUTOTIME;
    
    if (iterNum > 50) { iterNum = 50; MNN_PRINT("Clamped iterations to 50\n"); }
    if (iterNum < 1) { iterNum = 10; MNN_PRINT("Set iterations to 10\n"); }
    
    // Build FlowMatch Euler sigma schedule
    FlowMatchEulerScheduler scheduler(mTrainTimestepsNum, mFlowShift, mUseDynamicShifting);
    mSigmas = scheduler.get_sigmas(iterNum);
    mTimeSteps.resize(iterNum);
    for (int i = 0; i < iterNum; ++i) mTimeSteps[i] = i;
    
    // Apply chat template for ZImage tokenizer
    std::string promptForTokenizer = std::string("<|im_start|>user\n") + prompt +
                                     std::string("<|im_end|>\n<|im_start|>assistant\n<think>\n");
    auto ids = mTokenizer->encode(promptForTokenizer, mMaxTextLen);
    auto text_embeddings = text_encoder(ids);
    
    if (progressCallback) progressCallback(1 * 100 / (iterNum + 3));
    
    auto latent = unet(text_embeddings, iterNum, randomSeed, cfgScale, progressCallback);
    auto image = vae_decoder(latent);
    bool res = imwrite(outputPath, image);
    if (res) MNN_PRINT("SUCCESS! Generated image: %s\n", outputPath.c_str());
    
    if (mMemoryMode != 1) mModules[2].reset();
    if (progressCallback) progressCallback(100);
    return res;
}

bool ZImageDiffusion::run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                          const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                          bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) {
    MNN_PRINT("Error: ZImage does not support input_embeds interface.\n");
    return false;
}

} // namespace DIFFUSION
} // namespace MNN
