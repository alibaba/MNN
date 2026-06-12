#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>

#include "core/MNNFileUtils.h"
#include "diffusion/wan_diffusion.hpp"
#include "tokenizer.hpp"

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <cv/cv.hpp>

using namespace CV;

namespace MNN {
namespace DIFFUSION {

WanDiffusion::WanDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType,
                           int memoryMode)
    : Diffusion(modelPath, modelType, backendType, memoryMode) {
    mTokenizer.reset(new MtokTokenizer(MtokTokenizer::Style::kSingle));
}

WanDiffusion::~WanDiffusion() = default;

bool WanDiffusion::load() {
    AUTOTIME;
#if !defined(MNN_DIFFUSION_WITH_LLM_TOKENIZER)
    MNN_ERROR("Wan diffusion requires MNN_BUILD_LLM=ON so diffusion can load tokenizer.mtok\n");
    return false;
#endif
    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = mBackendType;
    backendConfig.memory = BackendConfig::Memory_Low;
    backendConfig.precision = BackendConfig::Precision_Low;
    if (config.type == MNN_FORWARD_CPU) {
        config.numThread = 4;
    } else if (config.type == MNN_FORWARD_OPENCL) {
        config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
    } else {
        config.numThread = 1;
    }
    config.backendConfig = &backendConfig;

    auto exe = ExecutorScope::Current();
    exe->lazyEval = false;
    exe->setGlobalExecutorConfig(config.type, backendConfig, config.numThread);

    Module::Config moduleConfig;
    moduleConfig.shapeMutable = false;
    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    if (!runtime_manager_) {
        MNN_ERROR("Failed to create Wan runtime manager\n");
        return false;
    }

    if (config.type == MNN_FORWARD_OPENCL) {
        runtime_manager_->setCache(".tempcache_wan");
    }
    if (mMemoryMode == 0) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
    } else if (mMemoryMode == 2) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 1);
    }
    if (config.type == MNN_FORWARD_CPU) {
        runtime_manager_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 2);
    }

    if (!mTokenizer) {
        MNN_ERROR("Wan tokenizer is not initialized\n");
        return false;
    }
    if (!mTokenizer->load(mModelPath + "/tokenizer")) {
        MNN_PRINT("Warning: failed to load tokenizer at %s/tokenizer, trying %s\n", mModelPath.c_str(),
                  mModelPath.c_str());
        if (!mTokenizer->load(mModelPath)) {
            MNN_ERROR("Failed to load Wan tokenizer.mtok\n");
            return false;
        }
    }

    mModules.resize(3);
    {
        std::string modelPath = mModelPath + "/text_encoder.mnn";
        mModules[0].reset(
            Module::load({"input_ids"}, {"last_hidden_state"}, modelPath.c_str(), runtime_manager_, &moduleConfig));
        if (!mModules[0]) {
            MNN_ERROR("Failed to load Wan text_encoder from %s\n", modelPath.c_str());
            return false;
        }
    }
    {
        std::string modelPath = mModelPath + "/transformer.mnn";
        mModules[1].reset(Module::load({"hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask"},
                                       {"noise_pred"}, modelPath.c_str(), runtime_manager_, &moduleConfig));
        if (!mModules[1]) {
            MNN_ERROR("Failed to load Wan transformer from %s\n", modelPath.c_str());
            return false;
        }
    }
    {
        std::string modelPath = mModelPath + "/vae_decoder.mnn";
        mModules[2].reset(
            Module::load({"latent_sample"}, {"sample"}, modelPath.c_str(), runtime_manager_, &moduleConfig));
        if (!mModules[2]) {
            MNN_ERROR("Failed to load Wan vae_decoder from %s\n", modelPath.c_str());
            return false;
        }
    }

    for (auto& module : mModules) {
        module->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    }
    return true;
}

VARP WanDiffusion::encodePrompt(const std::string& prompt, int* seqLen, VARP* outMask) {
    if (!mTokenizer || mModules.size() < 1 || !mModules[0]) {
        MNN_ERROR("Wan text encoder is not ready\n");
        return nullptr;
    }

    // Defensive: peek at the un-padded token count first so we can warn the
    // caller when the prompt is truncated. mTokenizer->encode pads/truncates
    // silently when maxlen > 0; passing maxlen=0 returns the raw ids.
    std::vector<int> rawTokens = mTokenizer->encode(prompt, 0);
    if ((int)rawTokens.size() > mMaxTextLen) {
        MNN_PRINT("Wan: prompt truncated from %d to %d tokens\n", (int)rawTokens.size(), mMaxTextLen);
    }

    std::vector<int> uncond = mTokenizer->encode("", mMaxTextLen);
    std::vector<int> cond = mTokenizer->encode(prompt, mMaxTextLen);
    if ((int)uncond.size() < mMaxTextLen || (int)cond.size() < mMaxTextLen) {
        MNN_ERROR("Wan tokenizer returned too few ids: uncond=%d cond=%d expected=%d\n", (int)uncond.size(),
                  (int)cond.size(), mMaxTextLen);
        return nullptr;
    }

    VARP inputIds = _Input({2, mMaxTextLen}, NCHW, halide_type_of<int>());
    int* inputPtr = inputIds->writeMap<int>();
    ::memset(inputPtr, 0, 2 * mMaxTextLen * sizeof(int));
    ::memcpy(inputPtr, uncond.data(), mMaxTextLen * sizeof(int));
    ::memcpy(inputPtr + mMaxTextLen, cond.data(), mMaxTextLen * sizeof(int));

    // Build attention mask from token ids: 1 where id != 0, 0 for padding.
    // Dtype must be int32 to match the ONNX export (see wan_onnx_export.py).
    VARP mask = _Input({2, mMaxTextLen}, NCHW, halide_type_of<int>());
    int* maskData = mask->writeMap<int>();
    for (int i = 0; i < 2 * mMaxTextLen; ++i) {
        maskData[i] = (inputPtr[i] != 0) ? 1 : 0;
    }
    if (outMask) {
        *outMask = mask;
    }

    auto outputs = mModules[0]->onForward({inputIds});
    if (outputs.empty() || outputs[0].get() == nullptr) {
        MNN_ERROR("Wan text_encoder returned empty output\n");
        return nullptr;
    }
    auto hiddenStates = _Convert(outputs[0], NCHW);
    auto info = hiddenStates->getInfo();
    if (seqLen != nullptr && info != nullptr && info->dim.size() > 1) {
        *seqLen = info->dim[1];
    } else if (seqLen != nullptr) {
        *seqLen = mMaxTextLen;
    }
    hiddenStates.fix(VARP::CONSTANT);
    return hiddenStates;
}

VARP WanDiffusion::transformer(VARP hiddenStates, VARP timestep, VARP encoderHiddenStates, VARP encoderAttentionMask) {
    auto outputs = mModules[1]->onForward({hiddenStates, timestep, encoderHiddenStates, encoderAttentionMask});
    if (outputs.empty() || outputs[0].get() == nullptr) {
        return nullptr;
    }
    return _Convert(outputs[0], NCHW);
}

VARP WanDiffusion::vaeDecoder(VARP latent) {
    auto outputs = mModules[2]->onForward({latent});
    if (outputs.empty() || outputs[0].get() == nullptr) {
        return nullptr;
    }
    return _Convert(outputs[0], NCHW);
}

VARP WanDiffusion::stepFlowMatch(VARP sample, VARP modelOutput, int index) {
    float t = mTimesteps[index];
    float nextT = (index + 1 < (int)mTimesteps.size()) ? mTimesteps[index + 1] : 0.0f;
    return sample + modelOutput * _Const((nextT - t) / 1000.0f);
}

bool WanDiffusion::saveFrame(VARP image4d, const std::string& fileName) {
    if (image4d.get() == nullptr) {
        return false;
    }
    auto image = image4d * _Const(0.5f) + _Const(0.5f);
    image = _Maximum(_Minimum(image, _Const(1.0f)), _Const(0.0f));
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0f)), halide_type_of<uint8_t>());
    image = cvtColor(image, COLOR_BGR2RGB);
    image.fix(VARP::CONSTANT);
    return imwrite(fileName, image);
}

bool WanDiffusion::saveVideoFrames(VARP sample, const std::string& outputDir, int requestedFrames) {
    if (sample.get() == nullptr) {
        MNN_ERROR("Wan VAE output is null\n");
        return false;
    }
    if (!MNNCreateDir(outputDir.c_str())) {
        MNN_ERROR("Failed to create output dir %s\n", outputDir.c_str());
        return false;
    }

    auto info = sample->getInfo();
    if (info == nullptr) {
        MNN_ERROR("Wan VAE output has no shape info\n");
        return false;
    }

    int saved = 0;
    if (info->dim.size() == 4) {
        MNN_PRINT("Warning: Wan VAE returned 4D/single-frame output; saving only frame_0000.png\n");
        std::string fileName = MNNFilePathConcat(outputDir, "frame_0000.png");
        return saveFrame(sample, fileName);
    }

    if (info->dim.size() != 5) {
        MNN_ERROR("Wan VAE output rank %d is not supported for frame saving\n", (int)info->dim.size());
        return false;
    }

    if (info->dim[1] != 3) {
        MNN_PRINT(
            "Warning: Wan VAE 5D output shape is not NCTHW with C=3; frame saving is not implemented for "
            "this layout\n");
        return false;
    }

    int decodedFrames = info->dim[2];
    int saveFrames = std::min(decodedFrames, requestedFrames);
    if (decodedFrames < requestedFrames) {
        MNN_PRINT("Warning: Wan VAE returned %d frames, requested %d; saving decoded frames only\n", decodedFrames,
                  requestedFrames);
    }
    for (int i = 0; i < saveFrames; ++i) {
        int startsData[5] = {0, 0, i, 0, 0};
        int sizesData[5] = {1, 3, 1, info->dim[3], info->dim[4]};
        auto frame = _Slice(sample, _Const(startsData, {5}, NCHW, halide_type_of<int>()),
                            _Const(sizesData, {5}, NCHW, halide_type_of<int>()));
        frame = _Reshape(frame, {1, 3, info->dim[3], info->dim[4]}, NCHW);

        std::ostringstream name;
        name << "frame_" << std::setfill('0') << std::setw(4) << i << ".png";
        std::string fileName = MNNFilePathConcat(outputDir, name.str());
        if (saveFrame(frame, fileName)) {
            ++saved;
        } else {
            MNN_PRINT("Warning: failed to save %s\n", fileName.c_str());
        }
    }

    MNN_PRINT("Saved %d Wan frame(s) to %s\n", saved, outputDir.c_str());
    return saved > 0;
}

bool WanDiffusion::runVideo(const std::string& prompt, const std::string& outputDir, int width, int height, int frames,
                            int steps, int seed, float cfgScale, std::function<void(int)> progressCallback) {
    AUTOTIME;
    if (mModules.size() < 3 || !mModules[0] || !mModules[1] || !mModules[2]) {
        MNN_ERROR("Wan modules are not loaded. Please call load() first.\n");
        return false;
    }
    if (width <= 0 || height <= 0 || frames <= 0) {
        MNN_ERROR("Wan video shape must be positive, got width=%d height=%d frames=%d\n", width, height, frames);
        return false;
    }
    const int vaeScaleFactor = 8;
    const int transformerPatchSize = 2;
    const int spatialAlignment = vaeScaleFactor * transformerPatchSize;
    if (width % spatialAlignment != 0 || height % spatialAlignment != 0) {
        MNN_ERROR("Wan width and height must be multiples of %d, got %dx%d\n", spatialAlignment, width, height);
        return false;
    }
    if (steps <= 0) {
        MNN_PRINT("Warning: Wan steps must be positive; using 1\n");
        steps = 1;
    }

    int seqLen = mMaxTextLen;
    VARP encoderAttentionMask;
    auto encoderHiddenStates = encodePrompt(prompt, &seqLen, &encoderAttentionMask);
    if (encoderHiddenStates.get() == nullptr) {
        return false;
    }
    if (mMemoryMode != 1) {
        mModules[0].reset();
        MNN::Express::Executor::getGlobalExecutor()->gc(MNN::Express::Executor::FULL);
    }

    int latentFrames = std::max(1, (frames + 3) / 4);
    int latentH = height / vaeScaleFactor;
    int latentW = width / vaeScaleFactor;
    int latentChannels = mLatentChannels;
    int latentSize = latentChannels * latentFrames * latentH * latentW;

    std::vector<float> noise(latentSize);
    int realSeed = seed < 0 ? std::random_device()() : seed;
    std::mt19937 rng(realSeed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int i = 0; i < latentSize; ++i) {
        noise[i] = normal(rng);
    }

    VARP latent = _Input({1, latentChannels, latentFrames, latentH, latentW}, NCHW, halide_type_of<float>());
    ::memcpy(latent->writeMap<float>(), noise.data(), noise.size() * sizeof(float));

    mTimesteps.resize(steps);
    if (steps == 1) {
        mTimesteps[0] = 1000.0f;
    } else {
        float shift = 3.0f;
        for (int i = 0; i < steps; ++i) {
            float tLinear = 1.0f + i * (0.001f - 1.0f) / (float)(steps - 1);
            float tShifted = (shift * tLinear) / (1.0f + (shift - 1.0f) * tLinear);
            mTimesteps[i] = tShifted * 1000.0f;
        }
    }

    VARP sample = latent;
    for (int i = 0; i < steps; ++i) {
        VARP sampleInput = _Concat({sample, sample}, 0);
        VARP timestep = _Input({2}, NCHW, halide_type_of<float>());
        float* tPtr = timestep->writeMap<float>();
        tPtr[0] = mTimesteps[i];
        tPtr[1] = mTimesteps[i];

        auto noisePred = transformer(sampleInput, timestep, encoderHiddenStates, encoderAttentionMask);
        if (noisePred.get() == nullptr) {
            MNN_ERROR("Wan transformer returned empty output at step %d\n", i + 1);
            return false;
        }

        auto split = _Split(noisePred, {2}, 0);
        auto noiseUncond = split[0];
        auto noiseText = split[1];
        auto guidedNoise = noiseUncond + (noiseText - noiseUncond) * _Const(cfgScale);
        sample = stepFlowMatch(sample, guidedNoise, i);

        if (progressCallback) {
            progressCallback((i + 1) * 90 / steps);
        }
    }
    sample.fix(VARP::CONSTANT);

    if (mMemoryMode != 1) {
        mModules[1].reset();
        MNN::Express::Executor::getGlobalExecutor()->gc(MNN::Express::Executor::FULL);
    }

    auto decoded = vaeDecoder(sample);
    if (decoded.get() == nullptr) {
        MNN_ERROR("Wan VAE decoder returned empty output\n");
        return false;
    }
    decoded.fix(VARP::CONSTANT);
    bool ok = saveVideoFrames(decoded, outputDir, frames);

    if (mMemoryMode != 1) {
        mModules[2].reset();
    }
    if (progressCallback) {
        progressCallback(ok ? 100 : 90);
    }
    return ok;
}

bool WanDiffusion::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed,
                       std::function<void(int)> progressCallback) {
    (void)prompt;
    (void)imagePath;
    (void)iterNum;
    (void)randomSeed;
    (void)progressCallback;
    MNN_ERROR("Wan2.1-T2V is a video model. Please use WanDiffusion::runVideo().\n");
    return false;
}

bool WanDiffusion::run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                       const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                       bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) {
    (void)input_embeds;
    (void)mode;
    (void)inputImagePath;
    (void)outputImagePath;
    (void)width;
    (void)height;
    (void)iterNum;
    (void)randomSeed;
    (void)use_cfg;
    (void)cfg_scale;
    (void)progressCallback;
    MNN_ERROR("Wan2.1-T2V does not support image run() overrides. Please use WanDiffusion::runVideo().\n");
    return false;
}

} // namespace DIFFUSION
} // namespace MNN
