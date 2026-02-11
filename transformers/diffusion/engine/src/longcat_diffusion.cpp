//
//  longcat_diffusion.cpp
//  LongCat Image Edit / T2I diffusion model (Flux-like architecture)
//
//  Extracted from the monolithic diffusion.cpp during upstream rebase.
//  Implements: LongCat text-to-image and image editing with FlowMatch Euler scheduler,
//  integrated LLM text encoder, VAE encoder/decoder, Flux-like packed latent format.
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
#include "diffusion/longcat_diffusion.hpp"
#include "diffusion/diffusion_config.hpp"
#include "tokenizer.hpp"
#include "scheduler.hpp"
#ifdef MNN_BUILD_LLM
#include "llm/tokenizer.hpp"
#include "llm/llm.hpp"
#include "../../../llm/engine/src/llmconfig.hpp"
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

// ===== LongCatDiffusion Implementation =====

LongCatDiffusion::LongCatDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode,
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
    
    // Load scheduler config
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
    
    // Load main config.json for LongCat-specific settings
    {
        std::string configPath = mModelPath + "/config.json";
        std::ifstream cf(configPath.c_str());
        if (cf.good()) {
            std::ostringstream oss;
            oss << cf.rdbuf();
            cf.close();
            rapidjson::Document doc;
            doc.Parse(oss.str().c_str());
            if (!doc.HasParseError() && doc.IsObject()) {
                if (doc.HasMember("text_encoder") && doc["text_encoder"].IsObject()) {
                    const auto& te = doc["text_encoder"];
                    if (te.HasMember("directory") && te["directory"].IsString())
                        mTextEncoderDir = te["directory"].GetString();
                    if (te.HasMember("cfg_scale") && (te["cfg_scale"].IsFloat() || te["cfg_scale"].IsDouble()))
                        mDefaultCfgScale = static_cast<float>(te["cfg_scale"].GetDouble());
                    if (te.HasMember("llm") && te["llm"].IsObject()) {
                        const auto& enc = te["llm"];
                        if (enc.HasMember("prefix_len") && enc["prefix_len"].IsInt())
                            mLlmEncoderConfig.prefixLen = enc["prefix_len"].GetInt();
                        if (enc.HasMember("suffix_len") && enc["suffix_len"].IsInt())
                            mLlmEncoderConfig.suffixLen = enc["suffix_len"].GetInt();
                        if (enc.HasMember("target_seq_len") && enc["target_seq_len"].IsInt())
                            mLlmEncoderConfig.targetSeqLen = enc["target_seq_len"].GetInt();
                        if (enc.HasMember("vision_resize_size") && enc["vision_resize_size"].IsInt())
                            mLlmEncoderConfig.visionResizeSize = enc["vision_resize_size"].GetInt();
                        if (enc.HasMember("hidden_size") && enc["hidden_size"].IsInt())
                            mLlmEncoderConfig.hiddenSize = enc["hidden_size"].GetInt();
                    }
                }
                MNN_PRINT("[LongCat] Config: text_encoder_dir=%s, llm_encoder(prefix=%d, suffix=%d, target=%d)\n",
                          mTextEncoderDir.c_str(), mLlmEncoderConfig.prefixLen, mLlmEncoderConfig.suffixLen, mLlmEncoderConfig.targetSeqLen);
            }
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
    MNN_PRINT("[LongCat] latent=(1,%d,%d,%d), image=%dx%d\n", mLatentC, mLatentH, mLatentW, mImageWidth, mImageHeight);
}

LongCatDiffusion::~LongCatDiffusion() {
#ifdef MNN_BUILD_LLM
    if (mLlm) {
        delete static_cast<MNN::Transformer::Llm*>(mLlm);
        mLlm = nullptr;
    }
#endif
}

bool LongCatDiffusion::load() {
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
        else gpuMode |= MNN_GPU_MEMORY_BUFFER;  // AUTO: BUFFER mode for LongCat
        config.mode = gpuMode;
    } else {
        config.numThread = 1;
    }
    backendConfig.memory = BackendConfig::Memory_Low;
    
    if (mPrecisionMode == PRECISION_LOW) backendConfig.precision = BackendConfig::Precision_Low;
    else if (mPrecisionMode == PRECISION_NORMAL) backendConfig.precision = BackendConfig::Precision_Normal;
    else if (mPrecisionMode == PRECISION_HIGH) backendConfig.precision = BackendConfig::Precision_High;
    else {
        // AUTO: LongCat requires FP32 for GPU (-inf handling in attention mask)
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
        MNN_ERROR("LongCat: Failed to create runtime manager\n");
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
    
    // LongCat needs 4 modules: text_encoder(skip), unet, vae_decoder, vae_encoder
    mModules.resize(4);
    MNN_PRINT("[LongCat] Skipping text_encoder.mnn (uses integrated LLM text encoder)\n");
    
    // Load UNet (with txt_ids and img_ids inputs)
    {
        std::string model_path = diff_config.unet_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[1].reset(Module::load(
            {"sample", "timestep", "encoder_hidden_states", "txt_ids", "img_ids"}, {"out_sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    // Load VAE decoder
    {
        std::string model_path = diff_config.vae_decoder_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[2].reset(Module::load({"latent_sample"}, {"sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    // Load VAE encoder
    {
        std::string model_path = diff_config.vae_encoder_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[3].reset(Module::load({"sample"}, {"latent_sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    
    // Resize fix (skip UNet due to dynamic shapes)
    for (int i = 0; i < (int)mModules.size(); ++i) {
        if (i == 1 || !mModules[i]) continue;
        mModules[i]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    }
    
    // UNet preprocessing: GPU slice + unpack for LongCat
    int packedSeq = (mLatentH / 2) * (mLatentW / 2);
    std::vector<int> starts_data = {0, 0, 0};
    std::vector<int> sizes_data = {1, packedSeq, 64};
    auto slice_starts = _Const(starts_data.data(), {3}, NCHW, halide_type_of<int>());
    auto slice_sizes = _Const(sizes_data.data(), {3}, NCHW, halide_type_of<int>());
    slice_starts.fix(VARP::CONSTANT);
    slice_sizes.fix(VARP::CONSTANT);
    int C = mLatentC, H = mLatentH, W = mLatentW;
    mUNetPreprocess = [this, slice_starts, slice_sizes, C, H, W](VARP unet_output) -> VARP {
        auto noise_part = _Slice(unet_output, slice_starts, slice_sizes);
        return this->unpackLatentsGPU(noise_part, 1, C, H, W);
    };
    mSchedulerType = SCHEDULER_EULER;
    
    return true;
}

VARP LongCatDiffusion::unpackLatentsGPU(VARP packed, int B, int C, int H, int W) {
    int packedH = H / 2;
    int packedW = W / 2;
    int packedC = C * 4;  // 64 for LongCat (C=16)
    
    // Step 1: Reshape [B, seq, 64] -> [B, H/2, W/2, C*4]
    auto reshaped = _Reshape(packed, {B, packedH, packedW, packedC});
    
    // Step 2: Reshape [B, H/2, W/2, C*4] -> [B, H/2, W/2, C, 2, 2]
    auto view6d = _Reshape(reshaped, {B, packedH, packedW, C, 2, 2});
    
    // Step 3: Permute [B, H/2, W/2, C, 2, 2] -> [B, C, H/2, 2, W/2, 2]
    auto permuted = _Transpose(view6d, {0, 3, 1, 4, 2, 5});
    
    // Step 4: Reshape [B, C, H/2, 2, W/2, 2] -> [B, C, H, W]
    auto unpacked = _Reshape(permuted, {B, C, H, W});
    
    return unpacked;
}

void LongCatDiffusion::getCFGSigmaRange(float& sigmaLow, float& sigmaHigh) const {
    switch (mCFGMode) {
        case CFG_MODE_WIDE:    sigmaLow = 0.1f; sigmaHigh = 0.9f; break;
        case CFG_MODE_STANDARD: sigmaLow = 0.1f; sigmaHigh = 0.8f; break;
        case CFG_MODE_MEDIUM:  sigmaLow = 0.15f; sigmaHigh = 0.7f; break;
        case CFG_MODE_NARROW:  sigmaLow = 0.2f; sigmaHigh = 0.6f; break;
        case CFG_MODE_MINIMAL: sigmaLow = 0.25f; sigmaHigh = 0.5f; break;
        default:               sigmaLow = 0.1f; sigmaHigh = 0.8f; break;
    }
}

VARP LongCatDiffusion::applyEulerUpdate(VARP sample, VARP noise_pred, float dt) {
    return sample + _Scalar(dt) * noise_pred;
}

#ifdef MNN_BUILD_LLM
VARP LongCatDiffusion::text_encoder_llm(const std::string& prompt, VARP preprocessedImage) {
    AUTOTIME;
    using namespace MNN::Transformer;
    
    const auto& cfg = mLlmEncoderConfig;
    bool isT2IMode = !preprocessedImage.get();
    
    // Lazy load LLM
    if (!mLlm) {
        std::string configPath = mModelPath + "/" + mTextEncoderDir + "/config.json";
        mLlm = Llm::createLLM(configPath);
        if (!mLlm) {
            MNN_PRINT("Error: Failed to create LLM from %s\n", configPath.c_str());
            return nullptr;
        }
        
        // Override mllm backend_type using set_config
        auto llm = static_cast<Llm*>(mLlm);
        const char* targetBackend = mTextEncoderOnCPU ? "cpu" : 
                                   (mBackendType == MNN_FORWARD_OPENCL ? "opencl" :
                                    mBackendType == MNN_FORWARD_VULKAN ? "vulkan" : "cpu");
        std::string overrideConfig = "{\"mllm\": {\"backend_type\": \"" + std::string(targetBackend) + "\"}}";
        llm->set_config(overrideConfig);
        
        if (mTextEncoderOnCPU) {
            MNN_PRINT("[LongCat] LLM backend forced to CPU (te_on_cpu=1)\n");
        } else {
            MNN_PRINT("[LongCat] LLM backend set to %s\n", targetBackend);
        }
        
        llm->load();
        MNN_PRINT("[LongCat] LLM text encoder loaded\n");
    }
    
    auto llm = static_cast<Llm*>(mLlm);
    
    VARP image = nullptr;
    if (!isT2IMode && preprocessedImage.get()) {
        // Image Edit mode: resize preprocessed image to vision encoder input size
        auto imgInfo = preprocessedImage->getInfo();
        int imgH = imgInfo->dim[0], imgW = imgInfo->dim[1];
        const int visionSize = cfg.visionResizeSize;
        if (imgW != visionSize || imgH != visionSize) {
            image = CV::resize(preprocessedImage, {visionSize, visionSize}, 0, 0, CV::INTER_LINEAR, -1, {}, {});
        } else {
            image = preprocessedImage;
        }
    }
    
    // Build prompt using LLM's apply_chat_template
    std::string systemPrompt;
    std::string userContent;
    if (isT2IMode) {
        systemPrompt = "As an image captioning expert, generate a descriptive text prompt based on an image content, "
                       "suitable for input to a text-to-image model.";
        userContent = prompt;
    } else {
        systemPrompt = "As an image editing expert, first analyze the content and attributes of the input image(s). "
                       "Then, based on the user's editing instructions, clearly and precisely determine how to modify "
                       "the given image(s), ensuring that only the specified parts are altered and all other aspects "
                       "remain consistent with the original(s).";
        userContent = "<|vision_start|><img>input</img><|vision_end|>" + prompt;
    }
    
    ChatMessages chatMessages;
    chatMessages.push_back({"system", systemPrompt});
    chatMessages.push_back({"user", userContent});
    std::string promptTemplate = llm->apply_chat_template(chatMessages);
    MNN_PRINT("[LongCat] Prompt template length: %d\n", (int)promptTemplate.size());
    
    // Create multimodal input
    MultimodalPrompt multimodalInput;
    multimodalInput.prompt_template = promptTemplate;
    if (!isT2IMode && image.get()) {
        auto imgInfo = image->getInfo();
        PromptImagePart imagePart;
        imagePart.image_data = image;
        imagePart.height = imgInfo->dim[0];
        imagePart.width = imgInfo->dim[1];
        multimodalInput.images["input"] = imagePart;
    }
    
    // Tokenize and forward
    auto inputIds = llm->tokenizer_encode(multimodalInput);
    MNN_PRINT("[LongCat] Tokenized: %d tokens\n", (int)inputIds.size());
    
    llm->generate_init();
    llm->forward(inputIds);
    
    // Get hidden states from outputs
    auto outputs = llm->getOutputs();
    MNN_PRINT("[LongCat] LLM outputs count: %d\n", (int)outputs.size());
    
    VARP hiddenStates = nullptr;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto info = outputs[i]->getInfo();
        if (info->dim.size() == 3 && info->dim[2] == cfg.hiddenSize) {
            hiddenStates = outputs[i];
            MNN_PRINT("[LongCat] Found hidden_states at output[%d]\n", (int)i);
            break;
        }
    }
    if (!hiddenStates.get()) {
        MNN_PRINT("Error: Failed to find hidden_states in LLM outputs\n");
        return nullptr;
    }
    
    // Slice hidden_states: remove prefix and suffix tokens
    auto hsInfo = hiddenStates->getInfo();
    int seqLen = hsInfo->dim[1];
    int hiddenSize = hsInfo->dim[2];
    
    int prefixLen, suffixLen;
    if (isT2IMode) {
        // T2I mode: dynamically calculate prefix length
        std::string t2iPrefixTemplate = "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n<|im_start|>user\n";
        auto prefixIds = llm->tokenizer_encode(t2iPrefixTemplate);
        prefixLen = (int)prefixIds.size();
        suffixLen = cfg.suffixLen;
        MNN_PRINT("[LongCat] T2I mode: calculated prefix=%d tokens\n", prefixLen);
    } else {
        prefixLen = cfg.prefixLen;
        suffixLen = cfg.suffixLen;
    }
    
    int sliceStart = prefixLen;
    int sliceEnd = seqLen - suffixLen;
    int outputSeqLen = sliceEnd - sliceStart;
    
    MNN_PRINT("[LongCat] Slicing: [%d:%d] -> %d tokens (prefix=%d, suffix=%d)\n",
              sliceStart, sliceEnd, outputSeqLen, prefixLen, suffixLen);
    
    if (outputSeqLen <= 0) {
        MNN_PRINT("Error: Invalid slice range for hidden_states\n");
        return nullptr;
    }
    
    // Slice using _GatherV2
    std::vector<int> indices;
    for (int i = sliceStart; i < sliceEnd; ++i) indices.push_back(i);
    auto indicesVar = _Const(indices.data(), {outputSeqLen}, NCHW, halide_type_of<int>());
    auto sliced = _GatherV2(hiddenStates, indicesVar, _Scalar<int>(1));
    
    // Pad to target sequence length
    auto slicedInfo = sliced->getInfo();
    int currentLen = slicedInfo->dim[1];
    int targetLen = isT2IMode ? cfg.tokenizerMaxLength : cfg.targetSeqLen;
    
    if (currentLen < targetLen) {
        int padLen = targetLen - currentLen;
        MNN_PRINT("[LongCat] %s: Padding %d -> %d (+%d zeros)\n",
                  isT2IMode ? "T2I" : "Image Edit", currentLen, targetLen, padLen);
        std::vector<int> padShape = {1, padLen, hiddenSize};
        auto padding = _Fill(_Const(padShape.data(), {3}, NCHW, halide_type_of<int>()), _Scalar<float>(0.0f));
        sliced = _Concat({sliced, padding}, 1);
    }
    
    sliced.fix(VARP::CONSTANT);
    
    auto finalInfo = sliced->getInfo();
    MNN_PRINT("[LongCat] Hidden states shape: [%d, %d, %d]\n",
              finalInfo->dim[0], finalInfo->dim[1], finalInfo->dim[2]);
    
    return sliced;
}
#else
VARP LongCatDiffusion::text_encoder_llm(const std::string& prompt, VARP preprocessedImage) {
    MNN_PRINT("Error: LongCat requires MNN_BUILD_LLM enabled.\n");
    return nullptr;
}
#endif

VARP LongCatDiffusion::unet(VARP text_embeddings, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback) {
    // Unload text_encoder and LLM to free memory before UNet inference
    if (mMemoryMode != 1 && mModules[0]) {
        mModules[0].reset();
    }
#ifdef MNN_BUILD_LLM
    if (mMemoryMode != 1 && mLlm) {
        delete static_cast<MNN::Transformer::Llm*>(mLlm);
        mLlm = nullptr;
        MNN_PRINT("[LongCat] LLM text encoder unloaded (memory_mode=%d)\n", mMemoryMode);
    }
#endif
    int latentSize = mLatentC * mLatentH * mLatentW;
    if ((int)mInitNoise.size() != latentSize) mInitNoise.resize(latentSize);
    
    // Generate random noise using Philox RNG (aligned with PyTorch)
    int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
    PhiloxRNG rng(seed);
    for (int i = 0; i < latentSize; i++) mInitNoise[i] = rng.randn();
    
    memcpy((void*)mLatentVar->writeMap<float>(), mInitNoise.data(), latentSize * sizeof(float));
    
    // For LongCat: pack noise latents and optionally image latents
    bool hasImageLatents = (mImageLatentsVar.get() != nullptr);
    bool isT2IMode = !hasImageLatents;
    
    int textSeqLen = text_embeddings->getInfo()->dim[1];
    int imgH = mLatentH / 2, imgW = mLatentW / 2;
    int singleSeq = imgH * imgW;
    int imgSeqLen = isT2IMode ? singleSeq : (singleSeq * 2);
    
    MNN_PRINT("[LongCat] %s mode: textSeqLen=%d, imgSeqLen=%d\n", isT2IMode ? "T2I" : "Image Edit", textSeqLen, imgSeqLen);
    
    // Build txt_ids and img_ids
    mTxtIdsVar = _Input({textSeqLen, 3}, NCHW, halide_type_of<float>());
    auto txtIdsPtr = mTxtIdsVar->writeMap<float>();
    for (int i = 0; i < textSeqLen; ++i) {
        txtIdsPtr[i * 3 + 0] = 0.0f;
        txtIdsPtr[i * 3 + 1] = static_cast<float>(i);
        txtIdsPtr[i * 3 + 2] = static_cast<float>(i);
    }
    
    mImgIdsVar = _Input({imgSeqLen, 3}, NCHW, halide_type_of<float>());
    auto imgIdsPtr = mImgIdsVar->writeMap<float>();
    int startOffset = isT2IMode ? mLlmEncoderConfig.tokenizerMaxLength : textSeqLen;
    
    if (isT2IMode) {
        for (int h = 0; h < imgH; ++h) {
            for (int w = 0; w < imgW; ++w) {
                int idx = h * imgW + w;
                imgIdsPtr[idx * 3 + 0] = 1.0f;
                imgIdsPtr[idx * 3 + 1] = static_cast<float>(startOffset + h);
                imgIdsPtr[idx * 3 + 2] = static_cast<float>(startOffset + w);
            }
        }
    } else {
        for (int half = 0; half < 2; ++half) {
            float modality = (half == 0) ? 1.0f : 2.0f;
            for (int h = 0; h < imgH; ++h) {
                for (int w = 0; w < imgW; ++w) {
                    int idx = half * singleSeq + h * imgW + w;
                    imgIdsPtr[idx * 3 + 0] = modality;
                    imgIdsPtr[idx * 3 + 1] = static_cast<float>(startOffset + h);
                    imgIdsPtr[idx * 3 + 2] = static_cast<float>(startOffset + w);
                }
            }
        }
    }
    
    // Pre-allocate mSampleVar as packed format for LongCat UNet input (reused in loop)
    int packedC4 = mLatentC * 4;
    int totalSeq = isT2IMode ? singleSeq : (singleSeq * 2);
    mSampleVar = _Input({1, totalSeq, packedC4}, NCHW, halide_type_of<float>());
    
    // Pre-create zero_embeddings for CFG
    VARP zero_embeddings;
    float cfgSigmaLow, cfgSigmaHigh;
    getCFGSigmaRange(cfgSigmaLow, cfgSigmaHigh);
    if (std::abs(cfgScale - 1.0f) > 0.001f) {
        zero_embeddings = text_embeddings * _Scalar<float>(0.0f);
        zero_embeddings.fix(VARP::CONSTANT);
    }
    
    // Create plms buffer for in-place Euler updates (GPU-side copy)
    auto plms = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
    plms->input(mLatentVar);
    
    auto floatVar = _Input({1}, NCHW, halide_type_of<float>());
    auto ptr = floatVar->writeMap<float>();
    
    for (int i = 0; i < (int)mTimeSteps.size(); i++) {
        AUTOTIME;
        
        float sigma = mSigmas[i];
        float sigma_next = (i + 1 < (int)mSigmas.size()) ? mSigmas[i + 1] : 0.0f;
        float dt = sigma_next - sigma;
        
        ptr[0] = sigma;
        mTimestepVar->input(floatVar);
        
        // Pack current latent state into pre-allocated mSampleVar for LongCat UNet input
        // CPU pack with writeMap/readMap (GPU pack causes computation graph accumulation)
        VARP sample_input;
        {
            auto samplePtr = mSampleVar->writeMap<float>();
            auto noisePtr = plms->readMap<float>();
            packLatents(noisePtr, samplePtr, 1, mLatentC, mLatentH, mLatentW, 0);
            if (!isT2IMode && mImageLatentsVar.get()) {
                auto imagePtr = mImageLatentsVar->readMap<float>();
                packLatents(imagePtr, samplePtr, 1, mLatentC, mLatentH, mLatentW, singleSeq);
            }
            sample_input = mSampleVar;
        }
        
        VARP noise_pred;
        
        // Limited Interval CFG for LongCat
        bool applyCFG = (std::abs(cfgScale - 1.0f) > 0.001f) && (sigma > cfgSigmaLow && sigma <= cfgSigmaHigh);
        
        if (applyCFG) {
            // Conditional pass
            std::vector<VARP> unet_inputs_cond = {sample_input, mTimestepVar, text_embeddings, mTxtIdsVar, mImgIdsVar};
            auto outputs_cond = mModules[1]->onForward(unet_inputs_cond);
            if (outputs_cond.empty()) { MNN_PRINT("[LongCat UNet] ERROR: cond outputs empty!\n"); return nullptr; }
            auto output_cond = _Convert(outputs_cond[0], NCHW);
            
            // Unconditional pass
            std::vector<VARP> unet_inputs_uncond = {sample_input, mTimestepVar, zero_embeddings, mTxtIdsVar, mImgIdsVar};
            auto outputs_uncond = mModules[1]->onForward(unet_inputs_uncond);
            if (outputs_uncond.empty()) { MNN_PRINT("[LongCat UNet] ERROR: uncond outputs empty!\n"); return nullptr; }
            auto output_uncond = _Convert(outputs_uncond[0], NCHW);
            
            noise_pred = output_uncond + _Scalar(cfgScale) * (output_cond - output_uncond);
        } else {
            std::vector<VARP> unet_inputs = {sample_input, mTimestepVar, text_embeddings, mTxtIdsVar, mImgIdsVar};
            auto outputs = mModules[1]->onForward(unet_inputs);
            if (outputs.empty()) { MNN_PRINT("[LongCat UNet] ERROR: outputs empty!\n"); return nullptr; }
            noise_pred = _Convert(outputs[0], NCHW);
        }
        
        // Layer 1: Preprocess (GPU slice + unpack for LongCat)
        auto noise_pred_standard = mUNetPreprocess(noise_pred);
        
        // Layer 2: Euler update
        auto updated = applyEulerUpdate(plms, noise_pred_standard, dt);
        plms->input(updated);
        
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

VARP LongCatDiffusion::vae_decoder(VARP latent) {
    if (mMemoryMode != 1) mModules[1].reset();
    
    // LongCat VAE scaling
    float scalingFactor = 0.3611f;
    float shiftFactor = 0.1159f;
    latent = latent / _Const(scalingFactor) + _Const(shiftFactor);
    
    AUTOTIME;
    auto outputs = mModules[2]->onForward({latent});
    auto output = _Convert(outputs[0], NCHW);
    
    auto image = output;
    image = _Relu6(image * _Const(0.5) + _Const(0.5), 0, 1);
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());
    image = cvtColor(image, COLOR_BGR2RGB);
    image.fix(VARP::CONSTANT);
    MNN_PRINT("[LongCat] VAE decoder output ready\n");
    return image;
}

bool LongCatDiffusion::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) {
    return run(prompt, imagePath, iterNum, randomSeed, mDefaultCfgScale, progressCallback, "");
}

bool LongCatDiffusion::run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath) {
    AUTOTIME;
    
    // Determine mode
    bool isT2IMode = inputImagePath.empty();
    VARP vaePreprocessedImage = nullptr;
    
    if (!isT2IMode) {
        MNN_PRINT("[LongCat] Image Edit mode: %s\n", inputImagePath.c_str());
        using namespace MNN::CV;
        auto rawImage = imread(inputImagePath);
        if (!rawImage.get()) { MNN_PRINT("Error: Failed to load input image\n"); return false; }
        
        auto processedImage = resizeAndCenterCrop(rawImage, mImageWidth, mImageHeight);
        VARP rgbImage = bgrToRgb(processedImage);
        vaePreprocessedImage = rgbImage;
        VARP inputImage = hwcToNchw(rgbImage, true);
        
        auto vaeOutputs = mModules[3]->onForward({inputImage});
        if (vaeOutputs.empty()) { MNN_PRINT("Error: VAE encoder failed\n"); return false; }
        
        float scalingFactor = 0.3611f, shiftFactor = 0.1159f;
        mImageLatentsVar = (vaeOutputs[0] - _Const(shiftFactor)) * _Const(scalingFactor);
        
        if (mMemoryMode != 1) mModules[3].reset();
    } else {
        MNN_PRINT("[LongCat] T2I mode\n");
        mImageLatentsVar = nullptr;
    }
    
    if (iterNum > 50) iterNum = 50;
    if (iterNum < 1) iterNum = 10;
    
    FlowMatchEulerScheduler scheduler(mTrainTimestepsNum, mFlowShift, mUseDynamicShifting);
    mSigmas = scheduler.get_sigmas(iterNum);
    mTimeSteps.resize(iterNum);
    for (int i = 0; i < iterNum; ++i) mTimeSteps[i] = i;
    
    auto text_embeddings = text_encoder_llm(prompt, vaePreprocessedImage);
    if (!text_embeddings.get()) { MNN_PRINT("Error: LLM text encoder failed\n"); return false; }
    
    if (progressCallback) progressCallback(1 * 100 / (iterNum + 3));
    
    auto latent = unet(text_embeddings, iterNum, randomSeed, cfgScale, progressCallback);
    auto image = vae_decoder(latent);
    bool res = imwrite(outputPath, image);
    if (res) MNN_PRINT("SUCCESS! Generated image: %s\n", outputPath.c_str());
    
    if (mMemoryMode != 1) mModules[2].reset();
    if (progressCallback) progressCallback(100);
    return res;
}

bool LongCatDiffusion::run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                           const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                           bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) {
    MNN_PRINT("Error: LongCat does not support input_embeds interface.\n");
    return false;
}

} // namespace DIFFUSION
} // namespace MNN
