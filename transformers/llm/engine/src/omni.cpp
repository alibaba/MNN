//
//  omni.cpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
//#define MNN_OPEN_TIME_TRACE

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <regex>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/AttentionMaskConvention.hpp"
#include "omni.hpp"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "tokenizer/tokenizer.hpp"
#include "diskembedding.hpp"
#include "sampler.hpp"
#ifdef LLM_SUPPORT_HTTP_RESOURCE
#include "httplib.h"
#endif
#include <set>
#ifdef LLM_SUPPORT_VISION
#include <cv/cv.hpp>
#endif
#ifdef MNN_LLM_VIDEOIO_OPENCV
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif
#ifdef LLM_SUPPORT_AUDIO
#include <audio/audio.hpp>
#endif
namespace MNN {
using namespace Express;
namespace Transformer {

static int roundHalfToEven(float value) {
    float floorValue = std::floor(value);
    float diff = value - floorValue;
    int rounded = static_cast<int>(floorValue);
    constexpr float eps = 1e-6f;
    if (diff > 0.5f + eps) {
        return rounded + 1;
    }
    if (diff < 0.5f - eps) {
        return rounded;
    }
    return (rounded % 2 == 0) ? rounded : rounded + 1;
}

static std::pair<int, int> qwenVlSmartResize(int height, int width, int factor, int minPixels, int maxPixels) {
    if (factor <= 0 || height < factor || width < factor) {
        MNN_ERROR("Qwen-VL smart resize requires height and width >= factor, got %dx%d with factor %d\n", height, width,
                  factor);
        return std::make_pair(0, 0);
    }
    if (minPixels <= 0 || maxPixels < minPixels) {
        MNN_ERROR("Qwen-VL smart resize got invalid pixel limits: min=%d, max=%d\n", minPixels, maxPixels);
        return std::make_pair(0, 0);
    }
    const double aspectRatio = static_cast<double>(std::max(height, width)) / std::min(height, width);
    if (aspectRatio > 200.0) {
        MNN_ERROR("Qwen-VL smart resize requires an aspect ratio no larger than 200, got %.2f\n", aspectRatio);
        return std::make_pair(0, 0);
    }
    int resizedHeight = roundHalfToEven(static_cast<float>(height) / factor) * factor;
    int resizedWidth = roundHalfToEven(static_cast<float>(width) / factor) * factor;
    int64_t resizedPixels = static_cast<int64_t>(resizedHeight) * resizedWidth;
    if (resizedPixels > maxPixels) {
        double beta = std::sqrt(static_cast<double>(height) * width / maxPixels);
        resizedHeight = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
        resizedWidth = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
    } else if (resizedPixels < minPixels) {
        double beta = std::sqrt(static_cast<double>(minPixels) / (static_cast<double>(height) * width));
        resizedHeight = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        resizedWidth = static_cast<int>(std::ceil(width * beta / factor)) * factor;
    }
    return std::make_pair(resizedHeight, resizedWidth);
}

template <typename T>
static inline VARP _var(const std::vector<T>& vec, const std::vector<int>& dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

static constexpr int kQwenVideoTemporalPatchSize = 2;

static int qwenVideoAlignedFrameCount(int frameCount, int maxFrames, int factor) {
    if (frameCount <= 0) {
        return 0;
    }
    factor = std::max(factor, 1);
    if (factor == 1) {
        return maxFrames > 0 ? std::min(frameCount, maxFrames) : frameCount;
    }
    if (maxFrames > 0 && maxFrames < factor) {
        return 0;
    }
    int count = maxFrames > 0 ? std::min(frameCount, maxFrames) : frameCount;
    if (count < factor) {
        return factor;
    }
    return count / factor * factor;
}

static int qwenVideoEffectiveMaxPixels(int maxPixels, int maxVisionTokens, int frameCount, int factor, int patchSize) {
    factor = std::max(factor, 1);
    patchSize = std::max(patchSize, 1);
    int gridT = frameCount / factor;
    if (gridT <= 0) {
        return maxPixels;
    }
    int tokensPerTemporal = std::max(std::max(maxVisionTokens, 1) / gridT, 1);
    int64_t tokenPixels = static_cast<int64_t>(tokensPerTemporal) * patchSize * patchSize;
    int budgetPixels = static_cast<int>(std::min<int64_t>(tokenPixels, std::numeric_limits<int>::max()));
    return maxPixels > 0 ? std::min(maxPixels, budgetPixels) : budgetPixels;
}

static std::pair<int, int> qwenVideoResizeSize(int width, int height, int alignSize, int maxPixels) {
    alignSize = std::max(alignSize, 1);
    width = std::max(width, alignSize);
    height = std::max(height, alignSize);
    auto alignRound = [alignSize](int size) {
        int quotient = size / alignSize;
        int remainder = size % alignSize;
        if (remainder * 2 > alignSize || (remainder * 2 == alignSize && quotient % 2 != 0)) {
            quotient++;
        }
        return std::max(alignSize, quotient * alignSize);
    };
    int targetWidth = alignRound(width);
    int targetHeight = alignRound(height);
    if (maxPixels <= 0 || static_cast<int64_t>(targetWidth) * targetHeight <= maxPixels) {
        return std::make_pair(targetWidth, targetHeight);
    }
    double scale = std::sqrt(static_cast<double>(maxPixels) / (static_cast<double>(width) * height));
    targetWidth = std::max(alignSize, static_cast<int>(std::floor(width * scale / alignSize)) * alignSize);
    targetHeight = std::max(alignSize, static_cast<int>(std::floor(height * scale / alignSize)) * alignSize);
    while (static_cast<int64_t>(targetWidth) * targetHeight > maxPixels &&
           (targetWidth > alignSize || targetHeight > alignSize)) {
        if (targetWidth >= targetHeight && targetWidth > alignSize) {
            targetWidth -= alignSize;
        } else if (targetHeight > alignSize) {
            targetHeight -= alignSize;
        } else {
            break;
        }
    }
    return std::make_pair(targetWidth, targetHeight);
}

static std::vector<int> qwenVideoSampleIndices(int totalFrames, double nativeFps, float targetFps, int minFrames,
                                               int maxFrames) {
    if (totalFrames <= 0) {
        return {};
    }
    if (totalFrames == 1) {
        return std::vector<int>(qwenVideoAlignedFrameCount(1, maxFrames, kQwenVideoTemporalPatchSize), 0);
    }
    if (nativeFps <= 0.0) {
        nativeFps = 24.0;
    }
    if (targetFps <= 0.0f) {
        targetFps = static_cast<float>(nativeFps);
    }
    int sampleFrames = static_cast<int>(totalFrames / nativeFps * targetFps);
    sampleFrames = std::max(sampleFrames, minFrames);
    sampleFrames = qwenVideoAlignedFrameCount(sampleFrames, maxFrames, kQwenVideoTemporalPatchSize);
    if (sampleFrames <= 0) {
        return {};
    }
    std::vector<int> indices;
    indices.reserve(sampleFrames);
    for (int i = 0; i < sampleFrames; ++i) {
        double pos = static_cast<double>(totalFrames - 1) * i / (sampleFrames - 1);
        int index = static_cast<int>(std::nearbyint(pos));
        indices.push_back(std::min(std::max(index, 0), totalFrames - 1));
    }
    return indices;
}

static std::vector<int> qwen25VisionWindowIndex(int gridT, int llmGridH, int llmGridW, int mergeUnit, int windowSize,
                                                int* windowIndex) {
    // Same padding as the HF reference: a full extra window when the grid divides evenly; it
    // contributes zero-length windows, which only repeat a boundary.
    const int numWindowsH = (llmGridH + windowSize - llmGridH % windowSize) / windowSize;
    const int numWindowsW = (llmGridW + windowSize - llmGridW % windowSize) / windowSize;
    std::vector<int> cuWindowSeqlens = {0};
    int written = 0;
    for (int t = 0; t < gridT; ++t) {
        for (int winH = 0; winH < numWindowsH; ++winH) {
            for (int winW = 0; winW < numWindowsW; ++winW) {
                int count = 0;
                for (int i = 0; i < windowSize; ++i) {
                    int h = winH * windowSize + i;
                    if (h >= llmGridH) continue;
                    for (int j = 0; j < windowSize; ++j) {
                        int w = winW * windowSize + j;
                        if (w >= llmGridW) continue;
                        windowIndex[written++] = t * llmGridH * llmGridW + h * llmGridW + w;
                        ++count;
                    }
                }
                cuWindowSeqlens.push_back(cuWindowSeqlens.back() + count * mergeUnit);
            }
        }
    }
    return cuWindowSeqlens;
}

static std::vector<int> qwenVideoTemporalPositions(const std::vector<float>& frameTimes, int gridT,
                                                   int temporalPatchSize, float positionIdPerSecond) {
    std::vector<int> positions(std::max(gridT, 0));
    for (int t = 0; t < gridT; ++t) {
        if (positionIdPerSecond <= 0.0f) {
            positions[t] = t;
            continue;
        }
        // Epsilon so an exact product such as 1.0s * 25 is not truncated to 24 by float error.
        const float elapsed = frameTimes[static_cast<size_t>(t) * temporalPatchSize] - frameTimes[0];
        positions[t] = static_cast<int>(elapsed * positionIdPerSecond + 1e-3f);
    }
    return positions;
}

static bool needVarWithShape(VARP var, const std::vector<int>& dims) {
    if (var == nullptr || var->getInfo() == nullptr) {
        return true;
    }
    const auto& current = var->getInfo()->dim;
    return current != dims;
}

static bool isQwen3TTSTalker(const std::shared_ptr<LlmConfig>& config) {
    return config->talker_type() == "qwen3_tts" || config->config_.value("model_type", "") == "qwen3_tts";
}

template <typename T>
static VARP makeTensorInput(const std::vector<int>& dims) {
    return _Input(dims, NCHW, halide_type_of<T>());
}

template <typename T>
static VARP makeTensorInputFromVector(const std::vector<int>& dims, const std::vector<T>& values) {
    auto var = makeTensorInput<T>(dims);
    auto info = var->getInfo();
    auto ptr = var->template writeMap<T>();
    if (!info || !ptr || info->size != values.size()) {
        return nullptr;
    }
    std::copy(values.begin(), values.end(), ptr);
    return var;
}

template <typename T>
static std::vector<T> readTensorVector(VARP var) {
    std::vector<T> values;
    if (var.get() == nullptr) { return values; }
    auto info = var->getInfo();
    auto ptr = var->template readMap<T>();
    if (!info || !ptr) {
        return values;
    }
    values.assign(ptr, ptr + info->size);
    return values;
}

static VARP makeEmbeddingInput(DiskEmbedding* embedding, const std::vector<int>& ids, int hiddenSize) {
    auto var = makeTensorInput<float>({1, static_cast<int>(ids.size()), hiddenSize});
    if (!embedding || var.get() == nullptr || !var->writeMap<float>()) {
        return nullptr;
    }
    embedding->embedding(ids, var->writeMap<float>());
    return var;
}

static VARP materializeFloatVar(VARP var) {
    auto info = var.get() ? var->getInfo() : nullptr;
    auto values = readTensorVector<float>(var);
    if (!info || values.empty()) {
        return nullptr;
    }
    return makeTensorInputFromVector<float>(info->dim, values);
}

static VARP makeQwen3TTSCodecEmbeds(DiskEmbedding* embedding, const std::vector<int>& codecPrefix,
                                    VARP speakerEmbedding, int hiddenSize) {
    if (speakerEmbedding.get() == nullptr) {
        return makeEmbeddingInput(embedding, codecPrefix, hiddenSize);
    }
    if (codecPrefix.size() <= 2) {
        return nullptr;
    }
    std::vector<int> headIds(codecPrefix.begin(), codecPrefix.end() - 2);
    std::vector<int> tailIds(codecPrefix.end() - 2, codecPrefix.end());
    auto head = makeEmbeddingInput(embedding, headIds, hiddenSize);
    auto tail = makeEmbeddingInput(embedding, tailIds, hiddenSize);
    auto headValues = readTensorVector<float>(head);
    auto speakerValues = readTensorVector<float>(speakerEmbedding);
    auto tailValues = readTensorVector<float>(tail);
    if (headValues.empty() || speakerValues.size() != hiddenSize || tailValues.empty()) {
        return nullptr;
    }
    std::vector<float> values;
    values.reserve((codecPrefix.size() + 1) * hiddenSize);
    values.insert(values.end(), headValues.begin(), headValues.end());
    values.insert(values.end(), speakerValues.begin(), speakerValues.end());
    values.insert(values.end(), tailValues.begin(), tailValues.end());
    return makeTensorInputFromVector<float>({1, static_cast<int>(codecPrefix.size()) + 1, hiddenSize}, values);
}

#ifdef LLM_SUPPORT_AUDIO
static VARP makeQwen3TTSSpeakerEmbedding(Module* speakerEncoder, const std::string& refAudio, int sampleRate,
                                         int hiddenSize) {
    if (!speakerEncoder || refAudio.empty()) {
        return nullptr;
    }
    auto audioData = MNN::AUDIO::load(refAudio, sampleRate, 0, -1, MNN::AUDIO::RESAMPLE_SOXR_HQ);
    auto waveform = audioData.first;
    if (waveform.get() == nullptr || !waveform->getInfo() || audioData.second != sampleRate) {
        MNN_ERROR("[Error]: failed to load Qwen3-TTS ref audio: %s\n", refAudio.c_str());
        return nullptr;
    }

    constexpr int nFft = 1024;
    constexpr int hopSize = 256;
    constexpr int winSize = 1024;
    auto padded = _Pad(waveform, _var<int>({(nFft - hopSize) / 2, (nFft - hopSize) / 2}, {2}), REFLECT);

    MNN::AUDIO::MelscaleParams melParams;
    melParams.n_mels = 128;
    melParams.n_fft = nFft;
    melParams.sample_rate = sampleRate;
    melParams.htk = false;
    melParams.norm = true;
    melParams.f_min = 0.0f;
    melParams.f_max = 12000.0f;

    MNN::AUDIO::SpectrogramParams specParams;
    specParams.n_fft = nFft;
    specParams.hop_length = hopSize;
    specParams.win_length = winSize;
    specParams.window_type = MNN::AUDIO::HANNING;
    specParams.center = false;
    specParams.normalized = false;
    specParams.power = 1.0f;

    auto mel = MNN::AUDIO::mel_spectrogram(padded, &melParams, &specParams);
    mel = _Log(_Maximum(mel, _Scalar<float>(1e-5f)));
    mel = materializeFloatVar(mel);
    if (mel.get() == nullptr || !mel->getInfo() || mel->getInfo()->dim.size() != 2) {
        MNN_ERROR("[Error]: failed to build Qwen3-TTS ref mel\n");
        return nullptr;
    }
    auto outputs = speakerEncoder->onForward({_Unsqueeze(mel, {0})});
    if (outputs.size() != 1) {
        MNN_ERROR("[Error]: Qwen3-TTS speaker encoder output size mismatch: %zu\n", outputs.size());
        return nullptr;
    }
    auto speakerValues = readTensorVector<float>(outputs[0]);
    if (speakerValues.size() != hiddenSize) {
        MNN_ERROR("[Error]: invalid Qwen3-TTS speaker embedding size: %zu\n", speakerValues.size());
        return nullptr;
    }
    return makeTensorInputFromVector<float>({1, 1, hiddenSize}, speakerValues);
}
#endif

static VARP makeQwen3AudioCodePredictorCodecEmbeds(DiskEmbedding* firstEmbedding, DiskEmbedding* predictorEmbedding,
                                                const std::vector<int>& codes, int codeGroups, int vocabSize,
                                                int hiddenSize) {
    if (!firstEmbedding || !predictorEmbedding || static_cast<int>(codes.size()) < codeGroups - 1 || codeGroups <= 1) {
        return nullptr;
    }
    auto var = makeTensorInput<float>({1, codeGroups - 1, hiddenSize});
    if (var.get() == nullptr || !var->writeMap<float>()) {
        return nullptr;
    }
    auto ptr = var->writeMap<float>();
    firstEmbedding->embedding({codes[0]}, ptr);
    for (int group = 1; group < codeGroups - 1; ++group) {
        int embeddingId = (group - 1) * vocabSize + codes[group];
        predictorEmbedding->embedding({embeddingId}, ptr + group * hiddenSize);
    }
    return var;
}

static void fillPositionIds(VARP var, int rows, int seqLen) {
    auto ptr = var->writeMap<int>();
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < seqLen; ++i) {
            ptr[r * seqLen + i] = i;
        }
    }
}

// code_predictor.mnn's attention ops are exported with kv_cache=true, so the causal plane can be omitted.
static void makeCausalInputs(int seqLen, int positionRows, VARP& mask, VARP& positionIds,
                             const std::string& backendType) {
    MaskRequest req;
    req.type   = AttentionMaskType::Causal;
    req.seqLen = seqLen;
    req.rank   = 4;
    mask = buildAttentionMask(req, backendType);
    positionIds = makeTensorInput<int>({positionRows, seqLen});
    fillPositionIds(positionIds, positionRows, seqLen);
}

static VARP selectTextHidden(VARP trailingTextHidden, VARP ttsPadEmbed, int step, int hiddenSize) {
    auto trailingInfo = trailingTextHidden->getInfo();
    auto trailingPtr = trailingTextHidden->readMap<float>();
    auto padPtr = ttsPadEmbed->readMap<float>();
    auto out = makeTensorInput<float>({1, hiddenSize});
    auto outPtr = out->writeMap<float>();
    if (!trailingInfo || !trailingPtr || !padPtr || !outPtr) {
        return nullptr;
    }
    int trailingLen = trailingInfo->dim.size() > 1 ? trailingInfo->dim[1] : 0;
    const float* src = step < trailingLen ? trailingPtr + step * hiddenSize : padPtr;
    std::copy(src, src + hiddenSize, outPtr);
    return out;
}

static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu")
        return MNN_FORWARD_CPU;
    if (type_str == "metal")
        return MNN_FORWARD_METAL;
    if (type_str == "cuda")
        return MNN_FORWARD_CUDA;
    if (type_str == "opencl")
        return MNN_FORWARD_OPENCL;
    if (type_str == "opengl")
        return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan")
        return MNN_FORWARD_VULKAN;
    if (type_str == "hexagon")
        return MNN_FORWARD_HEXAGON;
    if (type_str == "qnn")
        return MNN_FORWARD_QNN;
    if (type_str == "npu")
        return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

// Model families select input preparation and token/position rules; execution stays in Omni.
struct Omni::VisionProcessor {
    struct Batch {
        VARPS inputs;
        std::vector<std::pair<int, int>> grids;
        int temporal = 1;
        int validTokens = 0;
    };
    struct Plan {
        std::vector<Batch> batches;
        bool video = false;
        int rows = 0, cols = 0;
        std::vector<float> timestamps;
        std::vector<int> temporalPositions;
    };
    struct Feature {
        VARP value, extra;
        std::pair<int, int> grid;
        int temporal = 1;
    };
    struct Fixed;
    struct Patch;
    struct Qwen;
    struct Qwen25;
    struct Qwen3;
    struct Idefics;
    struct MiniCPM;
    struct MiniCPM47;
    struct Hunyuan;
    struct Gemma4;
    explicit VisionProcessor(Omni& omni) : mOmni(omni) {}
    virtual ~VisionProcessor() = default;
    static std::unique_ptr<VisionProcessor> create(Omni& omni);
    virtual Plan prepare(VARP image) = 0;
    virtual std::vector<int> video(const std::vector<VARP>& frames, const std::vector<float>& timestamps);
    std::vector<int> encode(const Plan& plan);
    virtual bool splitVideoFrames() const { return false; }
    virtual bool relativeSpatialPositions() const { return false; }
    virtual void reset() {}
    virtual bool promptPositions(const std::vector<int>& ids) { return true; }
    virtual int lastPosition() const { return mOmni.mPositionIds.back(); }
    virtual bool validate(const Feature& feature) const { return true; }
    virtual bool assemble(const Plan& plan, std::vector<Feature>& features, std::vector<int>& ids);
    virtual void positions(const Plan& plan, const Feature& feature, int markers);
    void append(const Plan& plan, const Feature& feature, std::vector<int>& ids, int start, int end, int pad);
    void text(std::vector<int>& ids, const std::vector<int>& tokens);
    void marker(std::vector<int>& ids, int token);
    Omni& mOmni;
};

struct Omni::VisionProcessor::Fixed : VisionProcessor {
    using VisionProcessor::VisionProcessor;
    Plan prepare(VARP image) override;
};

struct Omni::AudioProcessor {
    struct Whisper;
    struct Conformer;
    struct Gemma4;
    explicit AudioProcessor(Omni& omni) : mOmni(omni) {}
    virtual ~AudioProcessor() = default;
    static std::unique_ptr<AudioProcessor> create(Omni& omni);
    virtual VARP extract(VARP waveform) const = 0;
    VARPS prepare(VARP waveform) const;
    Omni& mOmni;
};

Omni::Omni(std::shared_ptr<LlmConfig> config) : Embedding(config) {
    mIsEmbedding = config->config_.value("is_embedding", false);
    mVisionHeight = mVisionWidth = config->config_.value("image_size", 448);
    mVisionPad = config->config_.value("image_pad", 151859);
    mVideoPad = config->config_.value("video_pad", -1);
    mVisionStart = config->config_.value("vision_start", 151857);
    mVisionEnd = config->config_.value("vision_end", 151858);
    mVisionMean = config->config_.value("image_mean", std::vector<float>{122.7709383f, 116.7460125f, 104.09373615f});
    mVisionNorm = config->config_.value("image_norm", std::vector<float>{0.01459843f, 0.01500777f, 0.01422007f});
    mVisionSizeUnit = config->config_.value("image_size_unit", 1);
    mVisionMaxSize = config->config_.value("image_max_size", 2048);
    mVisionGlobal = config->config_.value("global_image", 49152);
    mVideoFps = config->config_.value("video_fps", 2.0f);
    mVideoMaxFrames = config->config_.value("video_max_frames", 768);
    mVideoMaxPixels = config->config_.value("video_max_pixels", 768 * 28 * 28);
    mVideoMaxVisionTokens = std::max(1, config->config_.value("video_max_vision_tokens", 4096));
    mVideoPositionIdPerSecond = config->config_.value("video_position_id_per_second", 0.0f);
    mAudioPad = config->config_.value("audio_pad", 151646);
}

Omni::~Omni() = default;

bool Omni::load() {
    MNN::Express::ExecutorScope s(mExecutor);
    bool res = true;
    if (mIsEmbedding || isQwen3TTSTalker(mConfig)) {
        initRuntime();
        mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
        if (mTokenizer == nullptr) {
            MNN_ERROR("[Error]: Failed to load tokenizer from: %s\n", mConfig->tokenizer_file().c_str());
            return false;
        }
        if (mIsEmbedding) {
            mDiskEmbedding.reset(new DiskEmbedding(mConfig));
        }
        setChatTemplate();
    } else {
        res = Llm::load();
        if (!res) {
            return false;
        }
    }
    res = initProcessorRuntime();
    if (!res) {
        return false;
    }
    if (mConfig->has_talker()) {
        mTalker.reset(new Talker(mConfig, this));
        mTalker->setProcessorRuntimeManager(mProcessorRuntimeManager);
        res = mTalker->load();
        if (!res) {
            return false;
        }
    }
    if (mConfig->has_deepstack()) {
        mExtraArgs.emplace_back(Express::_Fill(_var<int>({3, 1, 1}, {3}), _Scalar<float>(0.0)));
    }
    if (mIsEmbedding) {
        Module::Config module_config;
        if (mConfig->backend_type() == "npu" || mConfig->backend_type() == "qnn") {
            module_config.shapeMutable = false;
        } else {
            module_config.shapeMutable = true;
        }
        module_config.rearrange = true;
        auto model_path = mConfig->llm_model();
        auto weight_path = mConfig->llm_weight();
        MNN_PRINT("load %s ... ", model_path.c_str());
        mRuntimeManager->setExternalFile(weight_path);
        mModule.reset(Module::load({"input_ids", "attention_mask", "position_ids", "deepstack_embeds"},
                                   {"sentence_embeddings"}, model_path.c_str(), mRuntimeManager, &module_config));
        mRuntimeManager->setExternalFile("");
        if (nullptr == mModule.get()) {
            MNN_ERROR("[Omni:embedding] Module::load failed for %s\n", model_path.c_str());
            return false;
        }
        MNN_PRINT("Done!\n");
    }
    mContext->status = LlmStatus::RUNNING;
    return true;
}

bool Omni::initProcessorRuntime() {
    ScheduleConfig config;
    if (mConfig->mllm_config_.is_null()) {
        mProcessorRuntimeManager = mRuntimeManager;
    } else {
        BackendConfig cpuBackendConfig;
        config.type = backend_type_convert(mConfig->backend_type(true));
        config.numThread = mConfig->thread_num(true);
        if (config.type == 3) {
            config.numThread |= 64;
            config.numThread |= 512;
        }
        if (mConfig->power(true) == "high") {
            cpuBackendConfig.power = BackendConfig::Power_High;
        } else if (mConfig->power(true) == "low") {
            cpuBackendConfig.power = BackendConfig::Power_Low;
        }
        if (mConfig->memory(true) == "high") {
            cpuBackendConfig.memory = BackendConfig::Memory_High;
        } else if (mConfig->memory(true) == "low") {
            cpuBackendConfig.memory = BackendConfig::Memory_Low;
        }
        if (mConfig->precision(true) == "high") {
            cpuBackendConfig.precision = BackendConfig::Precision_High;
        } else if (mConfig->precision(true) == "low") {
            cpuBackendConfig.precision = BackendConfig::Precision_Low;
        }
        config.backendConfig = &cpuBackendConfig;
        mProcessorRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
        setRuntimeHint(mProcessorRuntimeManager, true);
    }
    Module::Config moduleConfig;
    auto backend = mConfig->mllm_config_.is_null() ? mConfig->backend_type() : mConfig->backend_type(true);
    moduleConfig.shapeMutable = backend != "npu" && backend != "qnn";
    moduleConfig.rearrange = moduleConfig.shapeMutable;
    auto load = [&](const std::string& path, const std::string& key, std::shared_ptr<Module>& module,
                    ujson::json& processor) {
        processor = mConfig->config_.contains(key) ? mConfig->config_[key] : ujson::json::object();
        if (!processor.is_object() || (processor.value("version", 1) < 1 || processor.value("version", 1) > 2)) {
            MNN_ERROR("Unsupported %s descriptor\n", key.c_str());
            return false;
        }
        module.reset(Module::load({}, {}, path.c_str(), mProcessorRuntimeManager, &moduleConfig));
        return module != nullptr;
    };
    if (mConfig->is_visual()) {
        if (!load(mConfig->visual_model(), "vision_processor", mVisionModule, mVisionConfig)) {
            return false;
        }
        mVisionProcessor = VisionProcessor::create(*this);
        if (!mVisionProcessor) {
            return false;
        }
        auto info = mVisionModule->getInfo();
        if (!info->inputs.empty()) {
            mVisionInputOrder = info->inputs[0].order;
        }
    }
    if (mConfig->is_audio()) {
        if (!load(mConfig->audio_model(), "audio_processor", mAudioModule, mAudioConfig)) {
            return false;
        }
        mAudioProcessor = AudioProcessor::create(*this);
        if (!mAudioProcessor) {
            return false;
        }
    }
    return true;
}

static std::vector<std::pair<int, int>> minicpmBestSize(std::pair<int, int> original_size, int patch_size) {
    constexpr int max_slice_nums = 9, scale_resolution = 448;
    auto _get_target_size = [&](std::pair<int, int> size, bool upscale) -> std::pair<int, int> {
        int h = size.first;
        int w = size.second;
        int target_w, target_h;
        if (!upscale &&
            (static_cast<long long>(w) * h <= static_cast<long long>(scale_resolution) * scale_resolution)) {
            target_w = w;
            target_h = h;
        } else {
            double r = (h != 0) ? static_cast<double>(w) / h : 0.0;
            if (r > 0) {
                target_h = static_cast<int>(scale_resolution / std::sqrt(r));
                target_w = static_cast<int>(target_h * r);
            } else {
                target_h = 0;
                target_w = scale_resolution;
            }
        }
        int final_h =
            std::max(static_cast<int>(std::round(static_cast<double>(target_h) / patch_size)) * patch_size, patch_size);
        int final_w =
            std::max(static_cast<int>(std::round(static_cast<double>(target_w) / patch_size)) * patch_size, patch_size);
        return std::make_pair(final_h, final_w);
    };
    int original_height = original_size.first;
    int original_width = original_size.second;
    double ratio = (static_cast<double>(original_width) * original_height) /
                   (static_cast<double>(scale_resolution) * scale_resolution);
    int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
    std::vector<std::pair<int, int>> candidates;
    std::set<int> nums_to_check;
    if (multiple > 1)
        nums_to_check.insert(multiple - 1);
    nums_to_check.insert(multiple);
    nums_to_check.insert(multiple + 1);
    for (std::set<int>::iterator it = nums_to_check.begin(); it != nums_to_check.end(); ++it) {
        int num = *it;
        if (num >= 1 && num <= max_slice_nums) {
            for (int m = 1; m * m <= num; ++m) {
                if (num % m == 0) {
                    candidates.push_back(std::make_pair(m, num / m));
                    if (m * m != num)
                        candidates.push_back(std::make_pair(num / m, m));
                }
            }
        }
    }
    if (candidates.empty()) {
        candidates.push_back(std::make_pair(1, 1));
    }
    double log_ratio = std::log(static_cast<double>(original_width) / original_height);
    std::pair<int, int> best_grid =
        *std::min_element(candidates.begin(), candidates.end(),
                          [log_ratio](const std::pair<int, int>& g1, const std::pair<int, int>& g2) {
                              auto key = [log_ratio](const std::pair<int, int>& g) -> double {
                                  if (g.first == 0)
                                      return std::numeric_limits<double>::max();
                                  return std::abs(log_ratio - std::log(static_cast<double>(g.second) / g.first));
                              };
                              return key(g1) < key(g2);
                          });
    std::pair<int, int> source_image_size = _get_target_size(original_size, false);
    double patch_h = static_cast<double>(original_height) / best_grid.first;
    double patch_w = static_cast<double>(original_width) / best_grid.second;
    std::pair<int, int> best_patch_size =
        _get_target_size(std::make_pair(static_cast<int>(patch_h), static_cast<int>(patch_w)), true);
    std::pair<int, int> refine_image_size =
        std::make_pair(best_patch_size.first * best_grid.first, best_patch_size.second * best_grid.second);
    std::vector<std::pair<int, int>> result;
    result.push_back(source_image_size);
    result.push_back(refine_image_size);
    result.push_back(best_grid);
    return result;
}

#ifdef LLM_SUPPORT_VISION
static VARP resizeImage(VARP image, int height, int width, const std::vector<float>& mean,
                        const std::vector<float>& norm, int interpolation, Dimensionformat order = NCHW) {
    auto pixels = MNN::CV::resize(image, {width, height}, 0, 0, interpolation, MNN::CV::COLOR_BGR2RGB, mean, norm);
    return _Convert(_Unsqueeze(pixels, {0}), order);
}

static VARP splitImage(VARP pixels, int rows, int cols) {
    auto shape = pixels->getInfo()->dim;
    pixels = _Reshape(pixels, {shape[0], shape[1], rows, shape[2] / rows, cols, shape[3] / cols});
    return _Reshape(_Permute(pixels, {0, 2, 4, 1, 3, 5}),
                    {shape[0] * rows * cols, shape[1], shape[2] / rows, shape[3] / cols});
}

static VARP packPatches(VARP pixels, int patch, int merge, int temporal, const std::string& order) {
    auto shape = pixels->getInfo()->dim;
    int batch = shape[0], channels = shape[1], gh = shape[2] / patch, gw = shape[3] / patch;
    if (order == "resampler") {
        pixels = _Reshape(pixels, {batch, channels, gh, patch, gw, patch});
        return _Reshape(_Permute(pixels, {0, 1, 3, 2, 4, 5}), {batch, channels, patch, gh * gw * patch});
    }
    if (order == "hwc") {
        pixels = _Reshape(pixels, {batch, channels, gh, patch, gw, patch});
        return _Reshape(_Permute(pixels, {0, 2, 4, 3, 5, 1}), {batch, gh * gw, channels * patch * patch});
    }
    pixels =
        _Reshape(pixels, {batch / temporal, temporal, channels, gh / merge, merge, patch, gw / merge, merge, patch});
    const std::vector<int> axes =
        order == "row" ? std::vector<int>{0, 3, 4, 6, 7, 2, 1, 5, 8} : std::vector<int>{0, 3, 6, 4, 7, 2, 1, 5, 8};
    return _Reshape(_Permute(pixels, axes), {batch / temporal * gh * gw, channels * temporal * patch * patch});
}

static VARP bucketPositions(const std::vector<std::pair<int, int>>& grids, int bins, int length) {
    auto positions = _Input({static_cast<int>(grids.size()), length}, NCHW, halide_type_of<int>());
    auto ptr = positions->writeMap<int>();
    std::fill(ptr, ptr + grids.size() * length, 0);
    for (int b = 0; b < grids.size(); ++b) {
        int height = grids[b].first, width = grids[b].second;
        for (int y = 0; y < height; ++y) {
            int row = static_cast<int>(std::floor(static_cast<float>(y) / height * bins));
            for (int x = 0; x < width; ++x) {
                int col = static_cast<int>(std::floor(static_cast<float>(x) / width * bins));
                ptr[b * length + y * width + x] = row * bins + col;
            }
        }
    }
    return positions;
}

Omni::VisionProcessor::Plan Omni::VisionProcessor::Fixed::prepare(VARP image) {
    Plan plan;
    mOmni.mVisionHeight = UP_DIV(mOmni.mVisionHeight, mOmni.mVisionSizeUnit) * mOmni.mVisionSizeUnit;
    mOmni.mVisionWidth = UP_DIV(mOmni.mVisionWidth, mOmni.mVisionSizeUnit) * mOmni.mVisionSizeUnit;
    Batch batch;
    batch.inputs = {resizeImage(image, mOmni.mVisionHeight, mOmni.mVisionWidth, mOmni.mVisionMean, mOmni.mVisionNorm,
                                MNN::CV::INTER_LINEAR, mOmni.mVisionInputOrder)};
    batch.grids = {{1, 1}};
    plan.batches.push_back(std::move(batch));
    return plan;
}

struct Omni::VisionProcessor::Patch : VisionProcessor {
    using VisionProcessor::VisionProcessor;
    virtual Batch patchInputs(VARP pixels, bool video) = 0;
    virtual int minPixels() const { return 3136; }
    virtual int maxPixels() const { return 12845056; }
    virtual std::pair<int, int> targetSize(int height, int width, int factor, int min, int max) const {
        return qwenVlSmartResize(height, width, factor, min, max);
    }
    Plan prepare(VARP image) override {
        Plan plan;
        const int patch = mOmni.mVisionConfig.value("patch_size", 14);
        const int merge = mOmni.mVisionConfig.value("spatial_merge_size", 2);
        const int temporal = mOmni.mVisionConfig.value("temporal_patch_size", 2);
        if (!mOmni.mVisionSizeOverridden) {
            mOmni.mVisionHeight = image->getInfo()->dim[0];
            mOmni.mVisionWidth = image->getInfo()->dim[1];
        }
        if (image->getInfo()->dim.back() == 4) {
            image = _Slice(image, _var<int>({0, 0, 0}, {3}), _var<int>({-1, -1, 3}, {3}));
        }
        auto size = targetSize(mOmni.mVisionHeight, mOmni.mVisionWidth, patch * merge,
                               mOmni.mConfig->config_.value("image_min_pixels", minPixels()),
                               mOmni.mConfig->config_.value("image_max_pixels", maxPixels()));
        if (size.first <= 0 || size.second <= 0) {
            return {};
        }
        mOmni.mVisionHeight = size.first;
        mOmni.mVisionWidth = size.second;
        auto pixels =
            resizeImage(image, size.first, size.second, mOmni.mVisionMean, mOmni.mVisionNorm, MNN::CV::INTER_CUBIC);
        plan.batches.push_back(patchInputs(_Concat(VARPS(temporal, pixels), 0), false));
        return plan;
    }
};

struct Omni::VisionProcessor::Qwen : Patch {
    using Patch::Patch;
    std::vector<int> video(const std::vector<VARP>& frames, const std::vector<float>& timestamps) override;
    VARP gridPositions(int temporal, int height, int width, int merge);
    virtual VARPS attentionInputs(int t, int h, int w, int merge, bool video);
    VARP mPositionIds;
    MaskCache mMask;
    virtual VARPS extraInputs(int t, int h, int w, int merge, bool video) {
        VARPS inputs{gridPositions(t, h, w, merge)};
        auto attention = attentionInputs(t, h, w, merge, video);
        inputs.insert(inputs.end(), attention.begin(), attention.end());
        return inputs;
    }
    Batch patchInputs(VARP pixels, bool video) override {
        const auto shape = pixels->getInfo()->dim;
        const int patch = mOmni.mVisionConfig.value("patch_size", 14);
        const int merge = mOmni.mVisionConfig.value("spatial_merge_size", 2);
        const int temporal = mOmni.mVisionConfig.value("temporal_patch_size", 2);
        const int gt = shape[0] / temporal, gh = shape[2] / patch, gw = shape[3] / patch;
        Batch batch;
        batch.grids = {{gh, gw}};
        batch.temporal = gt;
        if (mOmni.mVisionConfig.value("input_layout", "native") != "pixels") {
            pixels = packPatches(pixels, patch, merge, temporal, "merged");
        }
        batch.inputs.push_back(pixels);
        auto extra = extraInputs(gt, gh, gw, merge, video);
        batch.inputs.insert(batch.inputs.end(), extra.begin(), extra.end());
        return batch;
    }
    void positions(const Plan& plan, const Feature& feature, int markers) override;
};

struct Omni::VisionProcessor::Qwen25 : Qwen {
    using Qwen::Qwen;
    VARPS attentionInputs(int t, int h, int w, int merge, bool video) override;
    VARP mWindowIndex, mWindowMask;
};

struct Omni::VisionProcessor::Qwen3 : Qwen {
    explicit Qwen3(Omni& omni) : Qwen(omni), mGridPerSide(omni.mConfig->config_.value("num_grid_per_side", 1)) {}
    int mGridPerSide;
    VARPS interpolationInputs(int temporal, int height, int width, int merge);
    VARP mInterpolationIndices, mInterpolationWeights;
    int minPixels() const override { return 65536; }
    int maxPixels() const override { return 16777216; }
    bool splitVideoFrames() const override { return true; }
    VARPS extraInputs(int t, int h, int w, int merge, bool video) override {
        auto inputs = Qwen::extraInputs(t, h, w, merge, video);
        auto interpolation = interpolationInputs(t, h, w, merge);
        inputs.insert(inputs.end(), interpolation.begin(), interpolation.end());
        return inputs;
    }
    bool assemble(const Plan& plan, std::vector<Feature>& features, std::vector<int>& ids) override {
        if (!plan.video) {
            return Qwen::assemble(plan, features, ids);
        }
        if (features.size() != plan.timestamps.size()) {
            return false;
        }
        for (int i = 0; i < features.size(); ++i) {
            std::ostringstream timestamp;
            timestamp << "<" << std::fixed << std::setprecision(1) << plan.timestamps[i] << " seconds>";
            text(ids, mOmni.mTokenizer->encode(timestamp.str()));
            append(plan, features[i], ids, mOmni.mVisionStart, mOmni.mVisionEnd, mOmni.mVideoPad);
        }
        return true;
    }
};

struct Omni::VisionProcessor::Hunyuan : Patch {
    using Patch::Patch;
    int minPixels() const override { return mOmni.mVisionHeight * mOmni.mVisionWidth; }
    int maxPixels() const override { return mOmni.mVisionMaxSize * mOmni.mVisionMaxSize; }
    bool relativeSpatialPositions() const override { return true; }
    std::pair<int, int> targetSize(int height, int width, int factor, int min, int max) const override {
        int h = std::max(factor, static_cast<int>(std::round(static_cast<float>(height) / factor)) * factor);
        int w = std::max(factor, static_cast<int>(std::round(static_cast<float>(width) / factor)) * factor);
        if (h * w > max) {
            float beta = std::sqrt(static_cast<float>(height * width) / max);
            h = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
            w = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
        } else if (h * w < min) {
            float beta = std::sqrt(static_cast<float>(min) / (height * width));
            h = std::max(factor, static_cast<int>(std::ceil(height * beta / factor)) * factor);
            w = std::max(factor, static_cast<int>(std::ceil(width * beta / factor)) * factor);
        }
        return {h, w};
    }
    Batch patchInputs(VARP pixels, bool video) override {
        auto shape = pixels->getInfo()->dim;
        int patch = mOmni.mVisionConfig.value("patch_size", 16),
            merge = mOmni.mVisionConfig.value("spatial_merge_size", 2);
        int gh = shape[2] / patch, gw = shape[3] / patch;
        Batch batch;
        batch.grids = {{gh, gw}};
        if (mOmni.mVisionConfig.value("input_layout", "native") != "pixels") {
            pixels = packPatches(pixels, patch, merge, 1, "row");
        }
        batch.inputs = {pixels, _var<int>({shape[0], gh, gw}, {1, 3})};
        return batch;
    }
    bool validate(const Feature& feature) const override {
        int merge = mOmni.mVisionConfig.value("spatial_merge_size", 2);
        int expected = feature.grid.first / merge * (feature.grid.second / merge + 1);
        int length = feature.value->getInfo()->dim[0];
        return length == expected || length == expected + 2;
    }
    void positions(const Plan& plan, const Feature& feature, int markers) override {
        int merge = mOmni.mVisionConfig.value("spatial_merge_size", 2);
        int t = feature.value->getInfo()->dim[0];
        int h = feature.grid.first / merge, w = feature.grid.second / merge;
        int cur_idx = mOmni.mPositionIds.mT.empty() ? 0 : mOmni.mPositionIds.mT.back() + 1;
        int gridTokens = h * (w + 1);
        int extraTokens = t - gridTokens;
        if (extraTokens != 0 && extraTokens != 2) {
            MNN_ERROR("Hunyuan image token count mismatch: tokens=%d grid=%d\n", t, gridTokens);
            return;
        }
        mOmni.mPositionIds.push_back(cur_idx++);
        if (extraTokens == 2) {
            mOmni.mPositionIds.push_back(cur_idx++);
        }
        for (int h_i = 0; h_i < h; h_i++) {
            for (int w_i = 0; w_i <= w; w_i++) {
                mOmni.mPositionIds.push_back(cur_idx++, w_i, h_i, mOmni.mVisionNum);
            }
        }
        if (extraTokens == 2) {
            mOmni.mPositionIds.push_back(cur_idx++);
        }
        mOmni.mPositionIds.push_back(cur_idx++);
        return;
    }
};

struct Omni::VisionProcessor::Idefics : VisionProcessor {
    using VisionProcessor::VisionProcessor;
    Plan prepare(VARP image) override {
        Plan plan;
        auto resize = [&](int h, int w, int interpolation) {
            return resizeImage(image, h, w, mOmni.mVisionMean, mOmni.mVisionNorm, interpolation);
        };
        const int linear = MNN::CV::INTER_LINEAR;
        const bool split = mOmni.mVisionHeight > mOmni.mVisionSizeUnit || mOmni.mVisionWidth > mOmni.mVisionSizeUnit;
        Batch global;
        global.inputs = {resize(mOmni.mVisionSizeUnit, mOmni.mVisionSizeUnit, linear)};
        global.grids = {{1, 1}};
        plan.batches.push_back(std::move(global));
        if (split) {
            mOmni.mVisionHeight =
                std::min(mOmni.mVisionMaxSize,
                         static_cast<int>(std::round(static_cast<float>(mOmni.mVisionHeight) / mOmni.mVisionSizeUnit)) *
                             mOmni.mVisionSizeUnit);
            mOmni.mVisionWidth =
                std::min(mOmni.mVisionMaxSize,
                         static_cast<int>(std::round(static_cast<float>(mOmni.mVisionWidth) / mOmni.mVisionSizeUnit)) *
                             mOmni.mVisionSizeUnit);
            plan.rows = mOmni.mVisionHeight / mOmni.mVisionSizeUnit;
            plan.cols = mOmni.mVisionWidth / mOmni.mVisionSizeUnit;
            Batch tiles;
            tiles.inputs = {splitImage(resize(mOmni.mVisionHeight, mOmni.mVisionWidth, linear), plan.rows, plan.cols)};
            tiles.grids.assign(plan.rows * plan.cols, {1, 1});
            plan.batches.push_back(std::move(tiles));
        }
        return plan;
    }
    bool assemble(const Plan& plan, std::vector<Feature>& features, std::vector<int>& ids) override {
        if (features.size() != plan.rows * plan.cols + 1) {
            return false;
        }
        if (features.size() > 1) {
            std::rotate(features.begin(), features.begin() + 1, features.end());
        }
        int index = 0;
        auto newline = mOmni.mTokenizer->encode("\n", false);
        for (int y = 0; y < plan.rows; ++y) {
            for (int x = 0; x < plan.cols; ++x) {
                marker(ids, mOmni.mVisionStart);
                text(ids, mOmni.mTokenizer->encode(
                              "<row_" + std::to_string(y + 1) + "_col_" + std::to_string(x + 1) + ">", false));
                append(plan, features[index++], ids, -1, -1, mOmni.mVisionPad);
            }
            text(ids, newline);
        }
        if (plan.rows > 0) {
            text(ids, newline);
        }
        marker(ids, mOmni.mVisionStart);
        marker(ids, mOmni.mVisionGlobal);
        append(plan, features[index], ids, -1, mOmni.mVisionEnd, mOmni.mVisionPad);
        return true;
    }
};

struct Omni::VisionProcessor::MiniCPM : VisionProcessor {
    using VisionProcessor::VisionProcessor;
    Plan prepare(VARP image) override {
        Plan plan;
        const int patch = mOmni.mVisionSizeUnit;
        auto sizes = minicpmBestSize({mOmni.mVisionHeight, mOmni.mVisionWidth}, patch);
        auto globalSize = sizes[0], refinedSize = sizes[1], grid = sizes[2];
        plan.rows = grid.first;
        plan.cols = grid.second;
        auto global = resizeImage(image, globalSize.first, globalSize.second, mOmni.mVisionMean, mOmni.mVisionNorm,
                                  MNN::CV::INTER_LINEAR);
        auto slices = resizeImage(image, refinedSize.first, refinedSize.second, mOmni.mVisionMean, mOmni.mVisionNorm,
                                  MNN::CV::INTER_LINEAR);
        slices = splitImage(slices, grid.first, grid.second);
        global = packPatches(global, patch, 1, 1, "resampler");
        slices = packPatches(slices, patch, 1, 1, "resampler");
        int length = slices->getInfo()->dim[3] / patch;
        global =
            _Pad(global, _var<int>({0, 0, 0, 0, 0, 0, 0, length * patch - global->getInfo()->dim[3]}, {8}), CONSTANT);
        Batch batch;
        batch.grids.push_back({globalSize.first / patch, globalSize.second / patch});
        batch.grids.insert(batch.grids.end(), grid.first * grid.second,
                           {refinedSize.first / grid.first / patch, refinedSize.second / grid.second / patch});
        int count = batch.grids.size();
        std::vector<int> valid(count, length), targetSizes;
        valid[0] = batch.grids[0].first * batch.grids[0].second;
        for (auto size : batch.grids) {
            targetSizes.push_back(size.first);
            targetSizes.push_back(size.second);
        }
        MaskRequest request;
        request.type = AttentionMaskType::KeyPadding;
        request.batch = count;
        request.kvSeqLen = length;
        request.rank = 2;
        request.validLens = &valid;
        batch.inputs = {_Concat({global, slices}, 0), bucketPositions(batch.grids, 70, length),
                        buildAttentionMask(request), _var<int>(targetSizes, {count, 2})};
        plan.batches.push_back(std::move(batch));
        return plan;
    }
    bool assemble(const Plan& plan, std::vector<Feature>& features, std::vector<int>& ids) override {
        marker(ids, mOmni.mConfig->config_.value("vision_id_start_id", 113));
        text(ids, mOmni.mTokenizer->encode(std::to_string(mOmni.mVisionNum), false));
        marker(ids, mOmni.mConfig->config_.value("vision_id_end_id", 114));
        append(plan, features[0], ids, mOmni.mVisionStart, mOmni.mVisionEnd, mOmni.mVisionPad);
        for (int i = 1; i < features.size(); ++i) {
            append(plan, features[i], ids, mOmni.mConfig->config_.value("vision_slice_start_id", 111),
                   mOmni.mConfig->config_.value("vision_slice_end_id", 112), mOmni.mVisionPad);
        }
        return true;
    }
};

struct Omni::VisionProcessor::MiniCPM47 : MiniCPM {
    using MiniCPM::MiniCPM;
    void imageId(std::vector<int>& ids) {
        marker(ids, mOmni.mConfig->config_.value("vision_id_start_id", 113));
        text(ids, mOmni.mTokenizer->encode(std::to_string(mOmni.mVisionNum), false));
        marker(ids, mOmni.mConfig->config_.value("vision_id_end_id", 114));
    }
    std::vector<std::pair<int, int>> mCropSizes;
    int mNextPosition = 0;
    void reset() override {
        mCropSizes.clear();
        mNextPosition = 0;
    }
    int lastPosition() const override {
        return mCropSizes.empty() ? VisionProcessor::lastPosition() : std::max(mNextPosition - 1, 0);
    }
    bool promptPositions(const std::vector<int>& ids) override;
    void positions(const Plan& plan, const Feature& feature, int markers) override {
        mCropSizes.push_back(feature.grid);
        VisionProcessor::positions(plan, feature, markers);
    }
    bool validate(const Feature& feature) const override {
        int merge = mOmni.mConfig->config_.value("vision_merge_size", 4);
        return merge > 0 && feature.grid.first % merge == 0 && feature.grid.second % merge == 0 &&
               feature.value->getInfo()->dim[0] == (feature.grid.first / merge) * (feature.grid.second / merge);
    }
    Plan prepare(VARP image) override {
        Plan plan;
        const int patch = mOmni.mVisionSizeUnit;
        const int resolution = mOmni.mConfig->config_.value("image_size", 448);
        const int maxSlices = mOmni.mConfig->config_.value("vision_max_slice_nums", 9);
        const int bins = mOmni.mConfig->config_.value("vision_position_size", 70);
        int height = mOmni.mVisionHeight, width = mOmni.mVisionWidth;
        if (!mOmni.mVisionSizeOverridden) {
            height = image->getInfo()->dim[0];
            width = image->getInfo()->dim[1];
        }
        if (patch <= 0 || resolution <= 0 || maxSlices < 1 || bins <= 0 || height <= 0 || width <= 0) {
            return {};
        }
        auto divide = [](double value, int unit) {
            return std::max(unit, static_cast<int>(std::nearbyint(value / unit)) * unit);
        };
        auto bestSize = [&](double h, double w, bool upscale) {
            if (upscale || h * w > static_cast<double>(resolution) * resolution) {
                double ratio = w / h;
                h = static_cast<int>(resolution / std::sqrt(ratio));
                w = static_cast<int>(h * ratio);
            }
            return std::make_pair(divide(h, patch * 4), divide(w, patch * 4));
        };
        int multiple = std::min(
            maxSlices, static_cast<int>(std::ceil(static_cast<double>(height) * width / (resolution * resolution))));
        if (multiple > 1) {
            double logRatio = std::log(static_cast<double>(width) / height);
            double bestError = std::numeric_limits<double>::infinity();
            for (int count = multiple - 1; count <= multiple + 1; ++count) {
                if (count <= 1 || count > maxSlices) {
                    continue;
                }
                for (int rows = 1; rows <= count; ++rows) {
                    if (count % rows != 0) {
                        continue;
                    }
                    int cols = count / rows;
                    double error = std::abs(logRatio - std::log(static_cast<double>(cols) / rows));
                    if (error < bestError || (error == bestError && rows > plan.rows)) {
                        bestError = error;
                        plan.rows = rows;
                        plan.cols = cols;
                    }
                }
            }
        }
        auto resize = [&](std::pair<int, int> size) {
            return resizeImage(image, size.first, size.second, mOmni.mVisionMean, mOmni.mVisionNorm,
                               MNN::CV::INTER_CUBIC);
        };
        auto addBatch = [&](VARP pixels, std::pair<int, int> size, int count) {
            Batch batch;
            int gh = size.first / patch, gw = size.second / patch;
            batch.grids.assign(count, {gh, gw});
            batch.inputs = {pixels, bucketPositions(batch.grids, bins, gh * gw)};
            plan.batches.push_back(std::move(batch));
        };
        auto globalSize = bestSize(height, width, plan.rows == 0);
        auto global = resize(globalSize);
        if (plan.rows > 0) {
            auto sliceSize = bestSize(static_cast<double>(divide(height, plan.rows)) / plan.rows,
                                      static_cast<double>(divide(width, plan.cols)) / plan.cols, true);
            auto slices =
                splitImage(resize({sliceSize.first * plan.rows, sliceSize.second * plan.cols}), plan.rows, plan.cols);
            if (globalSize == sliceSize) {
                addBatch(_Concat({global, slices}, 0), globalSize, 1 + plan.rows * plan.cols);
            } else {
                addBatch(global, globalSize, 1);
                addBatch(slices, sliceSize, plan.rows * plan.cols);
            }
        } else {
            addBatch(global, globalSize, 1);
        }
        return plan;
    }
    bool assemble(const Plan& plan, std::vector<Feature>& features, std::vector<int>& ids) override {
        if (features.size() != plan.rows * plan.cols + 1) {
            return false;
        }
        if (mOmni.mConfig->config_.value("vision_use_image_id", true)) {
            imageId(ids);
        }
        append(plan, features[0], ids, mOmni.mVisionStart, mOmni.mVisionEnd, mOmni.mVisionPad);
        int index = 1;
        for (int row = 0; row < plan.rows; ++row) {
            if (row > 0) {
                marker(ids, mOmni.mConfig->config_.value("vision_newline_id", 198));
            }
            for (int col = 0; col < plan.cols; ++col) {
                append(plan, features[index++], ids, mOmni.mConfig->config_.value("vision_slice_start_id", 248088),
                       mOmni.mConfig->config_.value("vision_slice_end_id", 248089), mOmni.mVisionPad);
            }
        }
        return true;
    }
};

bool Omni::VisionProcessor::MiniCPM47::promptPositions(const std::vector<int>& inputIds) {
    struct Crop {
        int begin, end, height, width;
    };
    struct Frame {
        Crop thumbnail;
        std::vector<Crop> slices;
        int end = 0;
    };
    const int count = static_cast<int>(inputIds.size());
    const int merge = mOmni.mConfig->config_.value("vision_merge_size", 4);
    const int sliceStart = mOmni.mConfig->config_.value("vision_slice_start_id", 248088);
    const int sliceEnd = mOmni.mConfig->config_.value("vision_slice_end_id", 248089);
    const int newline = mOmni.mConfig->config_.value("vision_newline_id", 198);
    auto isMarker = [&](int id) {
        return id == sliceStart || id == sliceEnd || id == mOmni.mVisionStart || id == mOmni.mVisionEnd ||
               id == newline;
    };
    std::vector<std::vector<Frame>> groups;
    size_t cropIndex = 0;
    int previousEnd = 0;
    for (int i = 0; i < count;) {
        if (inputIds[i] != mOmni.mVisionPad) {
            ++i;
            continue;
        }
        int begin = i;
        while (i < count && inputIds[i] == mOmni.mVisionPad) {
            ++i;
        }
        if (begin == 0 || cropIndex >= mCropSizes.size()) {
            MNN_ERROR("MiniCPM-V visual tokens do not match image crops\n");
            return false;
        }
        auto size = mCropSizes[cropIndex++];
        Crop crop{begin, i, size.first / merge, size.second / merge};
        if (crop.height * crop.width != crop.end - crop.begin) {
            MNN_ERROR("MiniCPM-V visual token count does not match crop dimensions\n");
            return false;
        }
        if (!groups.empty() && inputIds[begin - 1] == sliceStart) {
            groups.back().back().slices.push_back(crop);
        } else {
            bool adjacent = !groups.empty();
            for (int gap = previousEnd; gap < begin - 1 && adjacent; ++gap) {
                adjacent = isMarker(inputIds[gap]);
            }
            if (!adjacent) {
                groups.emplace_back();
            }
            Frame frame;
            frame.thumbnail = crop;
            groups.back().push_back(std::move(frame));
        }
        previousEnd = i;
    }
    if (cropIndex != mCropSizes.size()) {
        MNN_ERROR("MiniCPM-V image crops are missing from the prompt\n");
        return false;
    }
    for (auto& group : groups) {
        for (auto& frame : group) {
            frame.end = frame.slices.empty() ? frame.thumbnail.end : frame.slices.back().end;
            while (frame.end < count && isMarker(inputIds[frame.end])) {
                ++frame.end;
            }
        }
    }
    mOmni.mPositionIds.mT.assign(count, 0);
    mOmni.mPositionIds.mH.assign(count, 0);
    mOmni.mPositionIds.mW.assign(count, 0);
    mOmni.mPositionIds.mX.assign(count, 0);
    auto setPosition = [&](int index, int t, int h, int w) {
        mOmni.mPositionIds.mT[index] = t;
        mOmni.mPositionIds.mH[index] = h;
        mOmni.mPositionIds.mW[index] = w;
        mOmni.mPositionIds.mX[index] = w;
    };
    int currentPosition = 0;
    auto textPositions = [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            int position = currentPosition++;
            setPosition(i, position, position, position);
        }
    };
    auto gridPositions = [&](const Crop& crop, int origin, int canvasHeight, int canvasWidth, int hOffset,
                             int wOffset) {
        for (int y = 0; y < crop.height; ++y) {
            int h =
                crop.height == 1
                    ? 0
                    : static_cast<int>(std::nearbyint(static_cast<float>(y) * (canvasHeight - 1) / (crop.height - 1)));
            for (int x = 0; x < crop.width; ++x) {
                int w = crop.width == 1 ? 0
                                        : static_cast<int>(std::nearbyint(static_cast<float>(x) * (canvasWidth - 1) /
                                                                          (crop.width - 1)));
                setPosition(crop.begin + y * crop.width + x, origin, origin + hOffset + h, origin + wOffset + w);
            }
        }
    };
    int cursor = 0;
    for (const auto& group : groups) {
        int groupBegin = group.front().thumbnail.begin - 1;
        int groupEnd = group.back().end;
        textPositions(cursor, groupBegin);
        int frameCursor = groupBegin;
        for (const auto& frame : group) {
            const auto& thumbnail = frame.thumbnail;
            int begin = thumbnail.begin - 1;
            int end = std::min(frame.end, groupEnd);
            if (begin > frameCursor) {
                textPositions(frameCursor, begin);
                ++currentPosition;
            }
            int origin = currentPosition;
            int canvasHeight = thumbnail.height, canvasWidth = thumbnail.width;
            int rows = 0, cols = 0;
            if (!frame.slices.empty()) {
                cols = static_cast<int>(frame.slices.size());
                for (int i = 0; i + 1 < frame.slices.size(); ++i) {
                    if (frame.slices[i + 1].begin - frame.slices[i].end > 2) {
                        cols = i + 1;
                        break;
                    }
                }
                rows = frame.slices.size() / cols;
                if (rows * cols != frame.slices.size()) {
                    rows = 1;
                    cols = frame.slices.size();
                }
                canvasHeight = rows * frame.slices[0].height;
                canvasWidth = cols * frame.slices[0].width;
            }
            for (int i = begin; i < end; ++i) {
                setPosition(i, origin, origin, origin);
            }
            setPosition(begin, origin, std::max(origin - 1, 0), std::max(origin - 1, 0));
            if (thumbnail.end < end) {
                setPosition(thumbnail.end, origin, origin + canvasHeight, origin + canvasWidth);
            }
            gridPositions(thumbnail, origin, canvasHeight, canvasWidth, 0, 0);
            for (int i = 0; i < frame.slices.size(); ++i) {
                const auto& crop = frame.slices[i];
                int hOffset = (i / cols) * frame.slices[0].height;
                int wOffset = (i % cols) * frame.slices[0].width;
                setPosition(crop.begin - 1, origin, origin + hOffset, origin + wOffset);
                if (crop.end < end) {
                    setPosition(crop.end, origin, origin + hOffset + crop.height - 1,
                                origin + wOffset + crop.width - 1);
                }
                gridPositions(crop, origin, crop.height, crop.width, hOffset, wOffset);
                if (i + 1 < frame.slices.size() && frame.slices[i + 1].begin - crop.end > 2) {
                    int boundaryHeight = (i / cols + 1) * frame.slices[0].height - 1;
                    for (int gap = crop.end + 1; gap < frame.slices[i + 1].begin - 1; ++gap) {
                        setPosition(gap, origin, origin + boundaryHeight, origin + canvasWidth);
                    }
                }
            }
            currentPosition = origin + std::max(canvasHeight, canvasWidth) + 1;
            frameCursor = end;
        }
        textPositions(frameCursor, groupEnd);
        cursor = groupEnd;
    }
    textPositions(cursor, count);
    mNextPosition = currentPosition;
    return true;
}

struct Omni::VisionProcessor::Gemma4 : VisionProcessor {
    using VisionProcessor::VisionProcessor;
    Plan prepare(VARP image) override {
        Plan plan;
        const int linear = MNN::CV::INTER_LINEAR;
        int patch = mOmni.mVisionConfig.value("patch_size", 16), pool = mOmni.mVisionConfig.value("pooling_size", 3);
        int limit = mOmni.mVisionConfig.value("max_tokens", 280) * pool * pool, align = patch * pool;
        mOmni.mVisionHeight =
            std::max(align, static_cast<int>(std::round(static_cast<float>(mOmni.mVisionHeight) / align)) * align);
        mOmni.mVisionWidth =
            std::max(align, static_cast<int>(std::round(static_cast<float>(mOmni.mVisionWidth) / align)) * align);
        while (mOmni.mVisionHeight / patch * (mOmni.mVisionWidth / patch) > limit) {
            (mOmni.mVisionHeight >= mOmni.mVisionWidth ? mOmni.mVisionHeight : mOmni.mVisionWidth) -= align;
        }
        int gh = mOmni.mVisionHeight / patch, gw = mOmni.mVisionWidth / patch, count = gh * gw;
        auto pixels = resizeImage(image, mOmni.mVisionHeight, mOmni.mVisionWidth, {0, 0, 0},
                                  {1.0f / 255, 1.0f / 255, 1.0f / 255}, linear);
        if (mOmni.mVisionConfig.value("input_layout", "native") != "pixels") {
            pixels = packPatches(pixels, patch, 1, 1, "hwc");
            pixels = _Pad(pixels, _var<int>({0, 0, 0, limit - count, 0, 0}, {6}), CONSTANT);
        }
        auto positions = _Input({1, limit, 2}, NCHW, halide_type_of<int>());
        auto ptr = positions->writeMap<int>();
        std::fill(ptr, ptr + limit * 2, -1);
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                ptr[(y * gw + x) * 2] = x;
                ptr[(y * gw + x) * 2 + 1] = y;
            }
        }
        Batch batch;
        batch.inputs = {pixels, positions};
        batch.grids = {{gh, gw}};
        batch.validTokens = count / (pool * pool);
        plan.batches.push_back(std::move(batch));
        return plan;
    }
};

VARP Omni::VisionProcessor::Qwen::gridPositions(int temporal, int height, int width, int merge) {
    int count = temporal * height * width;
    if (needVarWithShape(mPositionIds, {2, count})) {
        mPositionIds = _Input({2, count}, NCHW, halide_type_of<int>());
    }
    auto data = mPositionIds->writeMap<int>();
    for (int t = 0; t < temporal; ++t) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = t * height * width + (y / merge * (width / merge) + x / merge) * merge * merge +
                            y % merge * merge + x % merge;
                data[index] = y;
                data[count + index] = x;
            }
        }
    }
    return mPositionIds;
}

VARPS Omni::VisionProcessor::Qwen3::interpolationInputs(int temporal, int height, int width, int merge) {
    const int grid = mGridPerSide;
    const int count = temporal * height * width;
    if (needVarWithShape(mInterpolationIndices, {4, count})) {
        mInterpolationIndices = _Input({4, count}, NCHW, halide_type_of<int>());
        mInterpolationWeights = _Input({4, count}, NCHW, halide_type_of<float>());
    }
    auto indices = mInterpolationIndices->writeMap<int>();
    auto weights = mInterpolationWeights->writeMap<float>();
    for (int t = 0; t < temporal; ++t) {
        for (int y = 0; y < height; ++y) {
            float h = height > 1 ? static_cast<float>(y) * (grid - 1) / (height - 1) : 0.0f;
            int h0 = static_cast<int>(h), h1 = std::min(h0 + 1, grid - 1);
            float dh = h - h0;
            for (int x = 0; x < width; ++x) {
                float w = width > 1 ? static_cast<float>(x) * (grid - 1) / (width - 1) : 0.0f;
                int w0 = static_cast<int>(w), w1 = std::min(w0 + 1, grid - 1);
                float dw = w - w0;
                int index = t * height * width + y * width + x;
                indices[index] = h0 * grid + w0;
                indices[count + index] = h0 * grid + w1;
                indices[2 * count + index] = h1 * grid + w0;
                indices[3 * count + index] = h1 * grid + w1;
                weights[index] = (1 - dh) * (1 - dw);
                weights[count + index] = (1 - dh) * dw;
                weights[2 * count + index] = dh * (1 - dw);
                weights[3 * count + index] = dh * dw;
            }
        }
    }
    auto reorder = [&](VARP value) {
        value = _Reshape(value, {4, temporal, height / merge, merge, width / merge, merge});
        return _Reshape(_Permute(value, {0, 1, 2, 4, 3, 5}), {4, -1});
    };
    return {reorder(mInterpolationIndices), reorder(mInterpolationWeights)};
}

VARPS Omni::VisionProcessor::Qwen::attentionInputs(int grid_t, int grid_h, int grid_w, int merge_size, bool video) {
    const int seq_len = grid_t * grid_h * grid_w;
    auto boundaries = uniformBoundaries(seq_len, grid_h * grid_w);
    MaskRequest request;
    request.type = video ? AttentionMaskType::BlockDiagonal : AttentionMaskType::Bidirectional;
    request.seqLen = seq_len;
    request.rank = 3;
    request.cuSeqlens = &boundaries;
    return {mMask.obtain(request, (mOmni.mConfig->mllm_config_.is_null() ? mOmni.mConfig->backend_type()
                                                                         : mOmni.mConfig->backend_type(true)))};
}
VARPS Omni::VisionProcessor::Qwen25::attentionInputs(int grid_t, int grid_h, int grid_w, int merge_size, bool video) {
    const int seq_len = grid_t * grid_h * grid_w;
    if (needVarWithShape(mWindowIndex, {seq_len / (merge_size * merge_size)})) {
        mWindowIndex = Express::_Input({seq_len / (merge_size * merge_size)}, NCHW, halide_type_of<int>());
    }
    auto window_index = mWindowIndex;
    auto window_index_ptr = window_index->writeMap<int>();
    const int merge_unit = merge_size * merge_size;
    auto cu_window_seqlens = qwen25VisionWindowIndex(grid_t, grid_h / merge_size, grid_w / merge_size, merge_unit,
                                                     mOmni.mVisionConfig.value("window_size", 4), window_index_ptr);
    // Global attention stays within each temporal group; local attention follows merger windows.
    if (needVarWithShape(mWindowMask, {2, 1, seq_len, seq_len})) {
        mWindowMask = Express::_Input({2, 1, seq_len, seq_len}, NCHW);
    }
    auto attention_mask = mWindowMask;
    MaskRequest maskReq;
    maskReq.seqLen = seq_len;
    maskReq.rank = 3; // -> {2, 1, seq_len, seq_len}
    MaskPlane visible, blocked;
    auto temporalBoundaries = uniformBoundaries(seq_len, grid_h * grid_w);
    visible.type = video ? AttentionMaskType::BlockDiagonal : AttentionMaskType::Bidirectional;
    visible.cuSeqlens = &temporalBoundaries;
    blocked.type = AttentionMaskType::BlockDiagonal;
    blocked.cuSeqlens = &cu_window_seqlens;
    fillTwoPlane(attention_mask->writeMap<float>(), maskReq, visible, blocked);
    return {attention_mask, window_index};
}
void Omni::VisionProcessor::Qwen::positions(const Plan& plan, const Feature& feature, int markers) {
    const int merge = mOmni.mVisionConfig.value("spatial_merge_size", 2);
    const int h = feature.grid.first / merge, w = feature.grid.second / merge;
    auto& positions = mOmni.mPositionIds;
    int start = positions.currentIdx();
    positions.push_back(start++);
    int extent = std::max(h, w);
    for (int t = 0; t < feature.temporal; ++t) {
        const int time = plan.temporalPositions.empty() ? t : plan.temporalPositions[t];
        if (!plan.temporalPositions.empty()) {
            extent = std::max(extent, time + 1);
        }
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                positions.push_back(start + time, start + y, start + x);
            }
        }
    }
    // Text resumes after the largest spatial or explicit temporal extent.
    positions.push_back(start + extent);
}
std::vector<int> Omni::VisionProcessor::Qwen::video(const std::vector<VARP>& frames,
                                                    const std::vector<float>& timestamps) {
    if (frames.empty()) {
        MNN_PRINT("Omni video has no decoded frames\n");
        mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
        return std::vector<int>(0);
    }
    if (mOmni.mVideoPad < 0) {
        MNN_PRINT("Omni video_pad is missing in llm_config.json\n");
        mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
        return std::vector<int>(0);
    }
    MNN::Express::ExecutorScope s(mOmni.mExecutor);
    Timer _t;
    const int patch_size = mOmni.mVisionConfig.value("patch_size", 14);
    const int temporal_patch_size = mOmni.mVisionConfig.value("temporal_patch_size", kQwenVideoTemporalPatchSize);
    const int merge_size = mOmni.mVisionConfig.value("spatial_merge_size", 2);
    const int align_size = patch_size * merge_size;
    int frameCount =
        qwenVideoAlignedFrameCount(static_cast<int>(frames.size()), mOmni.mVideoMaxFrames, temporal_patch_size);
    if (frameCount <= 0) {
        MNN_PRINT("Omni video requires max_frames >= %d\n", temporal_patch_size);
        mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
        return std::vector<int>(0);
    }
    int maxPixels = qwenVideoEffectiveMaxPixels(mOmni.mVideoMaxPixels, mOmni.mVideoMaxVisionTokens, frameCount,
                                                temporal_patch_size, patch_size);
    auto videoSize = qwenVideoResizeSize(mOmni.mVisionWidth, mOmni.mVisionHeight, align_size, maxPixels);
    mOmni.mVisionWidth = videoSize.first;
    mOmni.mVisionHeight = videoSize.second;

    std::vector<VARP> processedFrames;
    std::vector<float> frameTimes;
    frameTimes.reserve(frameCount);
    for (int i = 0; i < frameCount; ++i) {
        size_t frameIndex = std::min(static_cast<size_t>(i), frames.size() - 1);
        auto image = resizeImage(frames[frameIndex], mOmni.mVisionHeight, mOmni.mVisionWidth, mOmni.mVisionMean,
                                 mOmni.mVisionNorm, MNN::CV::INTER_CUBIC);
        processedFrames.push_back(image);
        if (frameIndex < timestamps.size()) {
            frameTimes.push_back(timestamps[frameIndex]);
        } else {
            frameTimes.push_back(static_cast<float>(i) / std::max(mOmni.mVideoFps, 1.0f));
        }
    }

    Plan plan;
    plan.video = true;
    plan.batches.push_back(patchInputs(_Concat(processedFrames, 0), true));
    const auto& batch = plan.batches[0];
    int seqLen = batch.temporal * batch.grids[0].first * batch.grids[0].second;
    if (seqLen > mOmni.mVideoMaxVisionTokens) {
        MNN_ERROR("Video visual tokens exceed video_max_vision_tokens\n");
        mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    if (!splitVideoFrames()) {
        plan.temporalPositions = qwenVideoTemporalPositions(frameTimes, batch.temporal, temporal_patch_size,
                                                            mOmni.mVideoPositionIdPerSecond);
    }
    for (int t = 0; t < batch.temporal; ++t) {
        float timestamp = 0.0f;
        for (int i = 0; i < temporal_patch_size; ++i) {
            timestamp += frameTimes[t * temporal_patch_size + i];
        }
        plan.timestamps.push_back(timestamp / temporal_patch_size);
    }
    auto videoIds = encode(plan);
    mOmni.mContext->vision_us += _t.durationInUs();
    mOmni.mContext->pixels_mp +=
        processedFrames.size() * (mOmni.mVisionWidth / 1000.0f) * (mOmni.mVisionHeight / 1000.0f);
    return videoIds;
}
#else
Omni::VisionProcessor::Plan Omni::VisionProcessor::Fixed::prepare(VARP image) {
    return {};
}
#endif

std::unique_ptr<Omni::VisionProcessor> Omni::VisionProcessor::create(Omni& o) {
    auto& config = o.mVisionConfig;
    std::string family = config.value("family", "");
    if (config.value("version", 1) == 2 && family.empty()) {
        MNN_ERROR("Vision processor family is missing\n");
        return nullptr;
    }
    if (family.empty()) {
        // Resolve historical graph signatures once; inference below uses model families.
        std::string type = config.value("type", "");
        if (type.empty()) {
            const auto& names = o.mVisionModule->getInfo()->inputNames;
            type = "fixed";
            if (o.mConfig->config_.value("vision_type", "") == "hunyuan_vl") {
                type = "row_grid";
            } else if (names.size() >= 3 && names[0] == "patches") {
                type = "patches";
            } else if (!names.empty() && names[0] == "pixel_values") {
                type = names.size() == 1 ? "tiles" : "resampler";
            } else if (!names.empty() && names[0] == "input_patches") {
                type = "padded_patches";
            }
            config["patch_size"] = names.size() == 5 ? 16 : 14;
            config["window_index"] = names.size() == 4 && names.back() == "window_index";
            config["position_interpolation"] = names.size() == 5 && names[3] == "idx_tensor";
            config["spatial_merge_size"] = o.mConfig->config_.value("vision_merge_size", 2);
            config["window_size"] = o.mConfig->config_.value("vision_merger_window_size", 4);
        }
        if (type == "fixed") {
            family = "clip";
        } else if (type == "patches") {
            family = config.value("position_interpolation", false)
                         ? "qwen3_vl"
                         : (config.value("window_index", false) ? "qwen2_5_vl" : "qwen2_vl");
        } else if (type == "tiles") {
            family = "idefics3";
        } else if (type == "resampler") {
            family = o.mConfig->config_.value("vision_type", "") == "minicpmv4_7" ? "minicpmv4_7" : "minicpmv";
        } else if (type == "row_grid") {
            family = "hunyuan_vl";
        } else if (type == "padded_patches") {
            family = "gemma4";
        } else {
            MNN_ERROR("Unsupported historical vision processor: %s\n", type.c_str());
            return nullptr;
        }
        if (!config.contains("output_layout")) {
            config["output_layout"] =
                type == "tiles" ? "bt1c" : (type == "resampler" || type == "padded_patches" ? "btc" : "tbc");
            if (type == "padded_patches") {
                config["output_scale"] = 1.0f / 39.25f;
            }
        }
        if (family == "hunyuan_vl" && !config.contains("input_layout")) {
            config["patch_size"] = o.mConfig->config_.value("hunyuan_patch_size", 16);
            config["spatial_merge_size"] = o.mConfig->config_.value("hunyuan_spatial_merge_size", 2);
            config["temporal_patch_size"] = o.mConfig->config_.value("hunyuan_temporal_patch_size", 1);
        }
        config["family"] = family;
    }
    if (config.value("patch_size", 14) <= 0 || config.value("spatial_merge_size", 2) <= 0 ||
        config.value("temporal_patch_size", 2) <= 0 || config.value("window_size", 4) <= 0 ||
        (family == "hunyuan_vl" && config.value("temporal_patch_size", 1) != 1)) {
        MNN_ERROR("Invalid vision patch dimensions\n");
        return nullptr;
    }
    if (family == "clip" || family == "deepseek_vl" || family == "internvl" || family == "qwen" || family == "gemma3" ||
        family == "mobileclip") {
        return std::unique_ptr<VisionProcessor>(new Fixed(o));
    }
#ifdef LLM_SUPPORT_VISION
    if (family == "qwen2_vl" || family == "glm_ocr") {
        return std::unique_ptr<VisionProcessor>(new Qwen(o));
    }
    if (family == "qwen2_5_vl" || family == "qwen2_5_omni") {
        return std::unique_ptr<VisionProcessor>(new Qwen25(o));
    }
    if (family == "qwen3_vl" || family == "qwen3_5") {
        return std::unique_ptr<VisionProcessor>(new Qwen3(o));
    }
    if (family == "idefics3" || family == "lfm2_vl") {
        return std::unique_ptr<VisionProcessor>(new Idefics(o));
    }
    if (family == "minicpmv4_7") {
        return std::unique_ptr<VisionProcessor>(new MiniCPM47(o));
    }
    if (family == "minicpmv") {
        return std::unique_ptr<VisionProcessor>(new MiniCPM(o));
    }
    if (family == "hunyuan_vl") {
        return std::unique_ptr<VisionProcessor>(new Hunyuan(o));
    }
    if (family == "gemma4") {
        return std::unique_ptr<VisionProcessor>(new Gemma4(o));
    }
#endif
    MNN_ERROR("Unsupported vision processor family: %s\n", family.c_str());
    return nullptr;
}

void Omni::VisionProcessor::text(std::vector<int>& ids, const std::vector<int>& tokens) {
    ids.insert(ids.end(), tokens.begin(), tokens.end());
    mOmni.addPositionIds(tokens.size());
}
void Omni::VisionProcessor::marker(std::vector<int>& ids, int token) {
    if (token >= 0) {
        text(ids, {token});
    }
}
void Omni::VisionProcessor::positions(const Plan& plan, const Feature& feature, int markers) {
    mOmni.addPositionIds(feature.value->getInfo()->dim[0] + markers);
}
void Omni::VisionProcessor::append(const Plan& plan, const Feature& feature, std::vector<int>& ids, int start, int end,
                                   int pad) {
    if (start >= 0) {
        ids.push_back(start);
    }
    ids.insert(ids.end(), feature.value->getInfo()->dim[0], pad);
    if (end >= 0) {
        ids.push_back(end);
    }
    positions(plan, feature, (start >= 0) + (end >= 0));
    mOmni.mVisionEmbeddings.push_back(feature.value);
    if (feature.extra.get()) {
        mOmni.mDeepStackEmbeddings.push_back(feature.extra);
    }
}
bool Omni::VisionProcessor::assemble(const Plan& plan, std::vector<Feature>& features, std::vector<int>& ids) {
    if (features.size() != 1) {
        return false;
    }
    append(plan, features[0], ids, mOmni.mVisionStart, mOmni.mVisionEnd,
           plan.video ? mOmni.mVideoPad : mOmni.mVisionPad);
    return true;
}

VARPS Omni::runEncoder(Module* module, const VARPS& inputs, const ujson::json& processor, bool preserve) {
    if (!module) {
        return {};
    }
    auto outputs = module->onForward(inputs);
    if (outputs.empty() || outputs[0].get() == nullptr || outputs[0]->getInfo() == nullptr) {
        MNN_ERROR("Media encoder returned no features\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    auto layout = processor.value("output_layout", "tbc");
    if (layout == "bt1c") {
        outputs[0] = _Squeeze(outputs[0], {2});
    }
    if (layout == "btc" || layout == "bt1c") {
        outputs[0] = _Permute(outputs[0], {1, 0, 2});
    }
    float scale = processor.value("output_scale", 1.0f);
    if (scale != 1.0f) {
        outputs[0] = outputs[0] * _Scalar<float>(scale);
    }
    auto info = outputs[0]->getInfo();
    if (!info || info->dim.size() != 3) {
        MNN_ERROR("Media encoder must return [tokens, batch, hidden] features\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    for (auto& output : outputs) {
        // A later encoder call can resize the module and reuse its output storage.
        if (preserve) {
            output = _Clone(output, true);
        }
        if (!mConfig->config_.value("async", true)) {
            output->readMap<float>();
        }
    }
    return outputs;
}

std::vector<int> Omni::VisionProcessor::encode(const Plan& plan) {
    if (plan.batches.empty()) {
        mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    std::vector<Feature> features;
    for (const auto& batch : plan.batches) {
        auto outputs = mOmni.runEncoder(mOmni.mVisionModule.get(), batch.inputs, mOmni.mVisionConfig, true);
        if (outputs.empty()) {
            return {};
        }
        auto values = outputs[0];
        int count = batch.grids.size(), length = values->getInfo()->dim[0];
        int groups = plan.video && !splitVideoFrames() ? 1 : batch.temporal;
        if (groups <= 0 || values->getInfo()->dim[1] != count || length % groups != 0) {
            MNN_ERROR("Vision output batch does not match its crop geometry\n");
            mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
            return {};
        }
        int tokens = batch.validTokens > 0 ? batch.validTokens : length / groups;
        for (int b = 0; b < count; ++b) {
            for (int t = 0; t < groups; ++t) {
                Feature feature;
                feature.value = _Slice(values, _var<int>({t * tokens, b, 0}, {3}), _var<int>({tokens, 1, -1}, {3}));
                feature.grid = batch.grids[b];
                feature.temporal = batch.temporal / groups;
                if (outputs.size() > 1) {
                    feature.extra =
                        _Slice(outputs[1], _var<int>({0, t * tokens, 0}, {3}), _var<int>({-1, tokens, -1}, {3}));
                }
                if (!validate(feature)) {
                    MNN_ERROR("Vision feature length does not match the model's crop geometry\n");
                    mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
                    return {};
                }
                features.push_back(std::move(feature));
            }
        }
    }
    std::vector<int> ids;
    if (!assemble(plan, features, ids)) {
        MNN_ERROR("Vision features do not match the model's prompt layout\n");
        mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    return ids;
}
void Omni::setImageSize(int height, int width) {
    if (height > 0 && width > 0) {
        mVisionHeight = height;
        mVisionWidth = width;
        mVisionSizeOverridden = true;
    }
}
void Omni::addPositionIds(int count) {
    const int start = mPositionIds.currentIdx();
    for (int i = 0; i < count; ++i) {
        mPositionIds.push_back(start + i);
    }
}

std::vector<int> Omni::visionProcess(VARP image) {
#ifdef LLM_SUPPORT_VISION
    ExecutorScope scope(mExecutor);
    if (image.get() == nullptr || !mVisionModule || !mVisionProcessor) {
        mVisionSizeOverridden = false;
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    Timer timer;
    auto ids = mVisionProcessor->encode(mVisionProcessor->prepare(image));
    mContext->vision_us += timer.durationInUs();
    mContext->pixels_mp += (mVisionWidth / 1000.0f) * (mVisionHeight / 1000.0f);
    mVisionSizeOverridden = false;
    ++mVisionNum;
    return ids;
#else
    return {};
#endif
}
std::vector<int> Omni::VisionProcessor::video(const std::vector<VARP>& frames, const std::vector<float>& timestamps) {
    MNN_ERROR("Video is not supported by this vision processor\n");
    mOmni.mContext->status = LlmStatus::INTERNAL_ERROR;
    return {};
}

std::vector<int> Omni::visionProcess(const std::string& file) {
#if defined(LLM_SUPPORT_VISION) && defined(MNN_IMGCODECS)
    VARP image = MNN::CV::imread(file);
    return visionProcess(image);
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::videoProcess(const PromptVideoPart& video) {
    if (!video.frames.empty()) {
        int oldVisionHeight = mVisionHeight;
        int oldVisionWidth = mVisionWidth;
        if (video.width > 0 && video.height > 0) {
            mVisionWidth = video.width;
            mVisionHeight = video.height;
        }
        float oldFps = mVideoFps;
        int oldMaxFrames = mVideoMaxFrames;
        mVideoFps = video.fps > 0.0f ? video.fps : mVideoFps;
        mVideoMaxFrames = video.max_frames > 0 ? video.max_frames : mVideoMaxFrames;
        auto ids = mVisionProcessor ? mVisionProcessor->video(video.frames, video.timestamps) : std::vector<int>{};
        if (!mVisionProcessor) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
        }
        mVideoFps = oldFps;
        mVideoMaxFrames = oldMaxFrames;
        mVisionHeight = oldVisionHeight;
        mVisionWidth = oldVisionWidth;
        return ids;
    }
    if (!video.file_path.empty()) {
        return videoProcess(video.file_path);
    }
    MNN_PRINT("Omni video part has no frames or file path\n");
    mContext->status = LlmStatus::INTERNAL_ERROR;
    return std::vector<int>(0);
}

std::vector<int> Omni::videoProcess(const std::string& file) {
#if defined(LLM_SUPPORT_VISION) && defined(MNN_LLM_VIDEOIO_OPENCV)
    cv::VideoCapture cap(file);
    if (!cap.isOpened()) {
        MNN_PRINT("Omni Can't open video: %s\n", file.c_str());
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return std::vector<int>(0);
    }
    double nativeFps = cap.get(cv::CAP_PROP_FPS);
    if (nativeFps <= 0.0) {
        nativeFps = 24.0;
    }
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    auto sampleIndices = qwenVideoSampleIndices(totalFrames, nativeFps, mVideoFps, 4, mVideoMaxFrames);
    std::vector<VARP> frames;
    std::vector<float> timestamps;
    int frameIdx = 0;
    int sampleIdx = 0;
    int oldVisionHeight = mVisionHeight;
    int oldVisionWidth = mVisionWidth;
    const int patchSize = mVisionConfig.value("patch_size", 14);
    const int alignSize = patchSize * mVisionConfig.value("spatial_merge_size", 2);
    int frameCount = qwenVideoAlignedFrameCount(static_cast<int>(sampleIndices.size()), mVideoMaxFrames,
                                                kQwenVideoTemporalPatchSize);
    bool hasVideoSize = false;
    cv::Mat frame;
    while (sampleIdx < sampleIndices.size() && cap.read(frame)) {
        if (frameIdx == sampleIndices[sampleIdx] && !frame.empty()) {
            cv::Mat bgr = frame;
            if (frame.channels() == 4) {
                cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
            } else if (frame.channels() == 1) {
                cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
            }
            if (!bgr.isContinuous()) {
                bgr = bgr.clone();
            }
            if (!hasVideoSize) {
                int maxPixels = qwenVideoEffectiveMaxPixels(mVideoMaxPixels, mVideoMaxVisionTokens, frameCount,
                                                            kQwenVideoTemporalPatchSize, patchSize);
                auto videoSize = qwenVideoResizeSize(bgr.cols, bgr.rows, alignSize, maxPixels);
                mVisionWidth = videoSize.first;
                mVisionHeight = videoSize.second;
                hasVideoSize = true;
            }
            cv::Mat resizedBgr = bgr;
            if (bgr.cols != mVisionWidth || bgr.rows != mVisionHeight) {
                cv::resize(bgr, resizedBgr, cv::Size(mVisionWidth, mVisionHeight), 0, 0, cv::INTER_CUBIC);
            }
            if (!resizedBgr.isContinuous()) {
                resizedBgr = resizedBgr.clone();
            }
            while (sampleIdx < sampleIndices.size() && frameIdx == sampleIndices[sampleIdx]) {
                auto var = Express::_Input({resizedBgr.rows, resizedBgr.cols, 3}, NHWC, halide_type_of<uint8_t>());
                ::memcpy(var->writeMap<uint8_t>(), resizedBgr.data, resizedBgr.total() * resizedBgr.elemSize());
                frames.push_back(var);
                timestamps.push_back(static_cast<float>(frameIdx / nativeFps));
                sampleIdx++;
            }
        }
        frameIdx++;
    }
    cap.release();
    auto ids = mVisionProcessor ? mVisionProcessor->video(frames, timestamps) : std::vector<int>{};
    if (!mVisionProcessor) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
    }
    mVisionHeight = oldVisionHeight;
    mVisionWidth = oldVisionWidth;
    return ids;
#else
    MNN_PRINT("Omni video decode unsupported: build with MNN_LLM_VIDEOIO_OPENCV=ON and OpenCV videoio\n");
    mContext->status = LlmStatus::INTERNAL_ERROR;
    return std::vector<int>(0);
#endif
}

struct Omni::AudioProcessor::Whisper : AudioProcessor {
    using AudioProcessor::AudioProcessor;
    VARP extract(VARP waveform) const override;
};
VARP Omni::AudioProcessor::Whisper::extract(VARP waveform) const {
#ifdef LLM_SUPPORT_AUDIO
    const auto& config = mOmni.mAudioConfig;
    return MNN::AUDIO::whisper_fbank(waveform, config.value("sample_rate", 16000), config.value("mel_bins", 128),
                                     config.value("n_fft", 400), config.value("hop_length", 160));
#else
    return nullptr;
#endif
}
struct Omni::AudioProcessor::Conformer : AudioProcessor {
    using AudioProcessor::AudioProcessor;
    VARP extract(VARP waveform) const override {
#ifdef LLM_SUPPORT_AUDIO
        const auto& config = mOmni.mAudioConfig;
        return MNN::AUDIO::conformer_fbank(waveform, config.value("sample_rate", 16000), config.value("mel_bins", 128),
                                           config.value("n_fft", 512), config.value("hop_length", 160),
                                           config.value("window_length", 400));
#else
        return nullptr;
#endif
    }
};
struct Omni::AudioProcessor::Gemma4 : AudioProcessor {
    using AudioProcessor::AudioProcessor;
    VARP extract(VARP waveform) const override {
#ifdef LLM_SUPPORT_AUDIO
        const auto& config = mOmni.mAudioConfig;
        return MNN::AUDIO::usm_fbank(waveform, config.value("sample_rate", 16000), config.value("mel_bins", 128),
                                     config.value("n_fft", 512), config.value("hop_length", 160),
                                     config.value("window_length", 320));
#else
        return nullptr;
#endif
    }
};

std::unique_ptr<Omni::AudioProcessor> Omni::AudioProcessor::create(Omni& o) {
    auto& config = o.mAudioConfig;
    std::string family = config.value("family", "");
    if (config.value("version", 1) == 2 && family.empty()) {
        MNN_ERROR("Audio processor family is missing\n");
        return nullptr;
    }
    if (family.empty()) {
        if (config.empty()) {
            auto type = o.mConfig->audio_type();
            bool windowed = o.mAudioModule->getInfo()->inputNames.size() > 1 && type != "qwen3_asr";
            config["type"] = type == "conformer" || type == "usm" ? type : "whisper";
            config["input_layout"] = type == "qwen3_asr" ? "ct" : (type == "usm" ? "btc" : "bct");
            config["window"] = windowed ? 100 : 0;
            config["max_frames"] = type == "whisper" && !windowed ? 3000 : 0;
            config["output_layout"] = "btc";
            config["output_scale"] =
                type == "usm" ? 1.0f / std::sqrt(static_cast<float>(o.mConfig->hidden_size())) : 1.0f;
        }
        auto type = config.value("type", "whisper");
        if (type == "whisper") {
            family = config.value("input_layout", "bct") == "ct" ? "qwen3_asr" : "qwen2_audio";
        } else if (type == "conformer") {
            family = "lfm2_audio";
        } else if (type == "usm") {
            family = "gemma4";
        } else {
            MNN_ERROR("Unsupported historical audio features: %s\n", type.c_str());
            return nullptr;
        }
        config["family"] = family;
    }
    const auto input = config.value("input_layout", "bct");
    const auto output = config.value("output_layout", "btc");
    if ((input != "bct" && input != "btc" && input != "ct") || (output != "btc" && output != "tbc") ||
        config.value("sample_rate", 16000) <= 0 || config.value("mask_stride", 2) <= 0) {
        MNN_ERROR("Invalid audio processor layout or sampling parameters\n");
        return nullptr;
    }
    if (family == "qwen2_audio" || family == "qwen2_5_omni" || family == "qwen3_asr" || family == "funaudiochat") {
        return std::unique_ptr<AudioProcessor>(new Whisper(o));
    }
    if (family == "lfm2_audio") {
        return std::unique_ptr<AudioProcessor>(new Conformer(o));
    }
    if (family == "gemma4") {
        return std::unique_ptr<AudioProcessor>(new Gemma4(o));
    }
    MNN_ERROR("Unsupported audio processor family: %s\n", family.c_str());
    return nullptr;
}

VARPS Omni::AudioProcessor::prepare(VARP waveform) const {
    // Feature extraction and the encoder can use different runtimes.
    auto features = materializeFloatVar(extract(waveform));
    if (features.get() == nullptr) {
        MNN_ERROR("Audio feature extraction failed\n");
        return {};
    }
    auto layout = mOmni.mAudioConfig.value("input_layout", "bct");
    int axis = layout == "bct" ? 2 : 1;
    if (layout == "ct") {
        features = _Squeeze(features, {0});
    }
    int maxFrames = mOmni.mAudioConfig.value("max_frames", 0);
    if (maxFrames > 0 && features->getInfo()->dim[axis] > maxFrames) {
        auto shape = features->getInfo()->dim;
        std::vector<int> start(shape.size(), 0), size(shape.size(), -1);
        size[axis] = maxFrames;
        features = _Slice(features, _Const(start.data(), {static_cast<int>(start.size())}, NCHW, halide_type_of<int>()),
                          _Const(size.data(), {static_cast<int>(size.size())}, NCHW, halide_type_of<int>()));
    }
    VARPS inputs{features};
    int window = mOmni.mAudioConfig.value("window", 0);
    if (window > 0) {
        int length = UP_DIV(features->getInfo()->dim[axis], mOmni.mAudioConfig.value("mask_stride", 2));
        auto boundaries = uniformBoundaries(length, window);
        MaskRequest request;
        request.type = AttentionMaskType::BlockDiagonal;
        request.seqLen = length;
        request.rank = 3;
        request.cuSeqlens = &boundaries;
        inputs.push_back(
            buildAttentionMask(request, (mOmni.mConfig->mllm_config_.is_null() ? mOmni.mConfig->backend_type()
                                                                               : mOmni.mConfig->backend_type(true))));
    }
    return inputs;
}

std::vector<int> Omni::audioProcess(const std::string& file) {
#ifdef LLM_SUPPORT_AUDIO
    ExecutorScope scope(mExecutor);
    return audioProcess(MNN::AUDIO::load(file, mAudioConfig.value("sample_rate", 16000)).first);
#else
    mContext->status = LlmStatus::INTERNAL_ERROR;
    return {};
#endif
}
std::vector<int> Omni::audioProcess(VARP waveform) {
    ExecutorScope scope(mExecutor);
    Timer timer;
    if (!mAudioModule || !mAudioProcessor || waveform.get() == nullptr || waveform->getInfo() == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    mContext->audio_input_s +=
        static_cast<float>(waveform->getInfo()->size) / mAudioConfig.value("sample_rate", 16000);
    auto inputs = mAudioProcessor->prepare(waveform);
    if (inputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    auto outputs = runEncoder(mAudioModule.get(), inputs, mAudioConfig);
    if (outputs.empty()) {
        return {};
    }
    auto embedding = outputs[0];
    mAudioEmbeddings.push_back(embedding);
    std::vector<int> ids;
    int start = mConfig->config_.value("audio_start", -1), end = mConfig->config_.value("audio_end", -1);
    if (start >= 0) {
        ids.push_back(start);
    }
    ids.insert(ids.end(), embedding->getInfo()->dim[0], mAudioPad);
    if (end >= 0) {
        ids.push_back(end);
    }
    addPositionIds(ids.size());
    mContext->audio_us += timer.durationInUs();
    return ids;
}

std::vector<int> Omni::processMediaFile(const std::string& mode, std::string info) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto file_info = info;
    if (mode == "img") {
        std::smatch match;
        if (std::regex_search(info, match, std::regex(R"(<hw>(.*?)</hw>)"))) {
            std::stringstream size(match[1].str());
            char comma;
            int height = 0, width = 0;
            if (size >> height >> comma >> width && height > 0 && width > 0) {
                setImageSize(height, width);
            }
            file_info = match.prefix().str() + match.suffix().str();
        }
    }
#ifdef LLM_SUPPORT_HTTP_RESOURCE
    if (file_info.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))");
        std::smatch url_match_result;
        std::string host, path;
        if (std::regex_search(file_info, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();
            path = url_match_result[2].str();
        }
        httplib::Client cli(host);
        auto res = cli.Get(path);
        file_info = "downloaded_file";
        if (res && res->status == 200) {
            std::ofstream file(file_info, std::ios::binary);
            if (file.is_open()) {
                file.write(res->body.c_str(), res->body.size());
                std::cout << "File has been downloaded successfully." << std::endl;
                file.close();
            } else {
                std::cerr << "Unable to open file to write." << std::endl;
            }
        } else {
            std::cerr << "Failed to download file. Status code: " << (res ? res->status : 0) << std::endl;
        }
    }
#endif
    if (mode == "img" && mConfig->is_visual()) {
        return visionProcess(file_info);
    }
    if (mode == "audio" && mConfig->is_audio()) {
        return audioProcess(file_info);
    }
    if (mode == "video" && mConfig->is_visual()) {
        return videoProcess(file_info);
    }
    return std::vector<int>(0);
}

std::vector<int> Omni::tokenizer_encode(const MultimodalPrompt& multimodal_input) {
    std::string prompt = multimodal_input.prompt_template;
    std::regex multimode_regex(kOmniMultimodalRegex);
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<int> ids{};
    mPositionIds.clear();
    mVisionEmbeddings.clear();
    mAudioEmbeddings.clear();
    mDeepStackEmbeddings.clear();
    mVisionNum = 0;
    if (mVisionProcessor) {
        mVisionProcessor->reset();
    }
    if (mConfig->has_deepstack() && mExtraArgs.size() == 1) {
        mExtraArgs[0] = Express::_Fill(_var<int>({3, 1, 1}, {3}), _Scalar<float>(0.0));
    }

    // Text-only requests use the same compression and post-processing as Llm.
    if (mConfig->token_compress() != 0 && !std::regex_search(prompt, multimode_regex)) {
        ids = Llm::tokenizer_encode(prompt);
        addPositionIds(static_cast<int>(ids.size()));
        return ids;
    }

    while (std::regex_search(searchStart, prompt.cend(), match, multimode_regex)) {
        auto txt_ids = mTokenizer->encode(match.prefix().str(), false);
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        auto mul_ids = processMediaContent(match[1].str(), match[2].str(), multimodal_input);

        ids.insert(ids.end(), mul_ids.begin(), mul_ids.end());
        searchStart = match.suffix().first;
    }
    if (searchStart != prompt.cend()) {
        auto txt_ids = mTokenizer->encode(std::string(searchStart, prompt.cend()), false);
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    const auto postProcessorInfo = mTokenizer->post_process(ids);
    const auto positionIdsDims = mModule->getInfo()->inputs[2].dim;
    if (positionIdsDims[0] != 1) {
        if (!postProcessorInfo.has_single_sequence_a) {
            MNN_ERROR("[Omni] Multimodal mRoPE post-processor must contain exactly one Sequence_A.\n");
            return {};
        }
        mPositionIds.prependTextPositions(postProcessorInfo.prefix_size);
        if (postProcessorInfo.suffix_size > 0) {
            addPositionIds(postProcessorInfo.suffix_size);
        }
    }
    if (mVisionProcessor && !mVisionProcessor->promptPositions(ids)) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    return ids;
}

std::vector<int> Omni::tokenizer_encode(const std::string& prompt) {
    MultimodalPrompt multimodal_input;
    multimodal_input.prompt_template = prompt;
    return tokenizer_encode(multimodal_input);
}

std::vector<int> Omni::processMediaContent(const std::string& mode, const std::string& content,
                                           const MultimodalPrompt& prompt) {
    if (mode == "img") {
        auto it = prompt.images.find(content);
        if (it != prompt.images.end()) {
            setImageSize(it->second.height, it->second.width);
            return visionProcess(it->second.image_data);
        }
    } else if (mode == "audio") {
        auto it = prompt.audios.find(content);
        if (it != prompt.audios.end()) {
            if (it->second.waveform.get()) {
                return audioProcess(it->second.waveform);
            }
            if (!it->second.file_path.empty()) {
                return audioProcess(it->second.file_path);
            }
            MNN_ERROR("Audio part has no waveform or file path\n");
            return {};
        }
    } else if (mode == "video") {
        auto it = prompt.videos.find(content);
        if (it != prompt.videos.end()) {
            return videoProcess(it->second);
        }
    }
    return processMediaFile(mode, content);
}

VARP Omni::embedding(const std::vector<int>& input_ids) {
    MNN::Express::ExecutorScope s(mExecutor);
    bool hasMultimodalEmbeds = !mVisionEmbeddings.empty() || !mAudioEmbeddings.empty();
    if (!hasMultimodalEmbeds) {
        if (mConfig->has_deepstack() && mExtraArgs.size() == 1) {
            mExtraArgs[0] = Express::_Fill(
                _var<int>({3, static_cast<int>(input_ids.size()), mConfig->hidden_size()}, {3}), _Scalar<float>(0.0));
        }
        return Llm::embedding(input_ids);
    }
    // PLE for the full input_ids (gemma4); applied after the text segments below, whose
    // Llm::embedding calls overwrite mPleInput. HF looks up multimodal pads as pad_token_id(0).
    VARP fullPle;
    if (mPleEmbedding) {
        int ple_dim = mConfig->ple_embed_dim();
        float ple_scale = mConfig->ple_embed_scale();
        int seq_len = static_cast<int>(input_ids.size());
        // Replace vision/audio pad tokens with pad_token_id=0 for PLE
        std::vector<int> ple_ids = input_ids;
        for (auto& id : ple_ids) {
            if (id == mVisionPad || id == mAudioPad || id == mVideoPad) {
                id = 0; // pad_token_id
            }
        }
        fullPle = _Input({1, seq_len, ple_dim}, NCHW);
        mPleEmbedding->embedding(ple_ids, fullPle->writeMap<float>());
        if (ple_scale != 1.0f) {
            fullPle = fullPle * _Scalar<float>(ple_scale);
        }
    }
    VARPS embeddings, deepstacks;
    const bool hasDeepStack = !mDeepStackEmbeddings.empty();
    auto appendZeros = [&](int length) {
        if (hasDeepStack && length > 0) {
            auto shape = mDeepStackEmbeddings[0]->getInfo()->dim;
            shape[1] = length;
            deepstacks.push_back(_Fill(_var<int>(shape, {static_cast<int>(shape.size())}), _Scalar<float>(0)));
        }
    };
    auto appendText = [&](int begin, int end) {
        if (begin == end) {
            return true;
        }
        auto feature = Llm::embedding(std::vector<int>(input_ids.begin() + begin, input_ids.begin() + end));
        if (feature.get() == nullptr) {
            return false;
        }
        embeddings.push_back(feature);
        appendZeros(end - begin);
        return true;
    };
    int visionIndex = 0, audioIndex = 0, textBegin = 0;
    const int count = static_cast<int>(input_ids.size());
    for (int i = 0; i < count;) {
        const int id = input_ids[i];
        bool audio = id == mAudioPad;
        bool vision = id == mVisionPad || (mVideoPad >= 0 && id == mVideoPad);
        if (!audio && !vision) {
            ++i;
            continue;
        }
        if (!appendText(textBegin, i)) {
            return nullptr;
        }
        auto& features = audio ? mAudioEmbeddings : mVisionEmbeddings;
        int& index = audio ? audioIndex : visionIndex;
        if (index >= features.size() || features[index].get() == nullptr || features[index]->getInfo() == nullptr) {
            MNN_ERROR("Multimodal placeholder has no encoder feature at token %d\n", i);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return nullptr;
        }
        auto feature = features[index];
        int length = feature->getInfo()->dim[0];
        if (length <= 0 || i + length > count ||
            !std::all_of(input_ids.begin() + i, input_ids.begin() + i + length, [id](int token) { return token == id; })) {
            MNN_ERROR("Multimodal feature length does not match its placeholders at token %d\n", i);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return nullptr;
        }
        embeddings.push_back(feature);
        if (hasDeepStack) {
            if (audio) {
                appendZeros(length);
            } else {
                if (index >= mDeepStackEmbeddings.size()) {
                    mContext->status = LlmStatus::INTERNAL_ERROR;
                    return nullptr;
                }
                deepstacks.push_back(mDeepStackEmbeddings[index]);
            }
        }
        ++index;
        i += length;
        textBegin = i;
    }
    if (!appendText(textBegin, count) || visionIndex != mVisionEmbeddings.size() || audioIndex != mAudioEmbeddings.size()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    auto mergedEmbed = Express::_Concat(embeddings, 0);
    if (mergedEmbed == nullptr || mergedEmbed->getInfo() == nullptr) {
        MNN_ERROR(
            "[Error]: failed to build merged multimodal embeddings. text_chunks=%zu, vision_embeds=%zu, "
            "audio_embeds=%zu\n",
            embeddings.size(), mVisionEmbeddings.size(), mAudioEmbeddings.size());
        return nullptr;
    }
    // Deep copy: materialize the lazy concat so vision data persists after clear
    {
        auto cInfo = mergedEmbed->getInfo();
        auto cPtr = mergedEmbed->readMap<float>();
        if (cPtr == nullptr) {
            MNN_ERROR("[Error]: merged multimodal embeddings are not readable. dims=[%d,%d,%d,%d]\n",
                      cInfo->dim.size() > 0 ? cInfo->dim[0] : -1, cInfo->dim.size() > 1 ? cInfo->dim[1] : -1,
                      cInfo->dim.size() > 2 ? cInfo->dim[2] : -1, cInfo->dim.size() > 3 ? cInfo->dim[3] : -1);
            return nullptr;
        }
        auto freshEmbed = _Input(cInfo->dim, cInfo->order);
        ::memcpy(freshEmbed->writeMap<float>(), cPtr, cInfo->size * sizeof(float));
        mergedEmbed = freshEmbed;
    }
    mVisionEmbeddings.clear();
    mAudioEmbeddings.clear();
    mDeepStackEmbeddings.clear();
    if (fullPle.get()) {
        mPleInput = fullPle;
    }
    // Qwen3-VL
    if (hasDeepStack) {
        mExtraArgs[0] = Express::_Concat(deepstacks, 1);
    }
    return mergedEmbed;
}

bool Omni::prepareHistoryReplay(const std::vector<int>& input_ids) {
    // Encoder embeddings and MRoPE layouts of past multimodal spans are dropped after their prefill.
    if (mTalker != nullptr) {
        return false;
    }
    for (int id : input_ids) {
        if (id == mVisionPad || id == mAudioPad || id == mVideoPad) {
            return false;
        }
    }
    mPositionIds.clear();
    mVisionEmbeddings.clear();
    mAudioEmbeddings.clear();
    mDeepStackEmbeddings.clear();
    return true;
}

static inline bool needNewVar(VARP var, int axis, int seq_len) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    return false;
}

VARP Omni::gen_attention_mask(int seq_len) {
    return mIsEmbedding ? Embedding::gen_attention_mask(seq_len) : Llm::gen_attention_mask(seq_len);
}

VARP Omni::gen_position_ids(int seq_len) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto positionIdsDims = mModule->getInfo()->inputs[2].dim;
    if (positionIdsDims[0] == 1) {
        return Llm::gen_position_ids(seq_len);
    }
    // mrope
    int axes = mConfig->mrope_axes();
    if (positionIds == nullptr || positionIds->getInfo()->dim[0] != axes || needNewVar(positionIds, 1, seq_len)) {
        positionIds = _Input({axes, seq_len}, NCHW, halide_type_of<int>());
    }
    auto ptr = positionIds->writeMap<int>();
    if (mContext->gen_seq_len > 0) {
        const int lastPosition = mVisionProcessor ? mVisionProcessor->lastPosition() : mPositionIds.back();
        for (int i = 0; i < seq_len; ++i) {
            auto pos = mContext->gen_seq_len + lastPosition + i;
            for (int axis = 0; axis < axes; axis++) {
                ptr[i + seq_len * axis] = pos;
            }
        }
    } else {
        bool relativeSpatial = mVisionProcessor && mVisionProcessor->relativeSpatialPositions();
        auto axisValue = [this](int axis, int i) {
            const std::vector<int>* values = nullptr;
            if (axis == 0) {
                values = &mPositionIds.mT;
            } else if (axis == 1) {
                values = &mPositionIds.mH;
            } else if (axis == 2) {
                values = &mPositionIds.mW;
            } else if (axis == 3) {
                values = &mPositionIds.mX;
            }
            if (values != nullptr && i < static_cast<int>(values->size())) {
                return (*values)[i];
            }
            return i;
        };
        for (int i = 0; i < seq_len; i++) {
            for (int axis = 0; axis < axes; axis++) {
                int offset = (relativeSpatial && axis > 0) ? 0 : mContext->all_seq_len;
                ptr[i + seq_len * axis] = axisValue(axis, i) + offset;
            }
        }
        if (mTalker) {
            mTalker->setPostionIds(mPositionIds);
        }
    }
    return positionIds;
}

std::vector<Express::VARP> Omni::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs) {
    MNN::Express::ExecutorScope s(mExecutor);
    extraArgs.insert(extraArgs.end(), mExtraArgs.begin(), mExtraArgs.end());
    if (mIsEmbedding) {
        std::vector<VARP> inputs{hiddenState, mask, inputPos};
        if (!extraArgs.empty()) {
            inputs.insert(inputs.end(), extraArgs.begin(), extraArgs.end());
        }
        return mModule->onForward(inputs);
    }
    auto outputs = Llm::forwardRaw(hiddenState, mask, inputPos, extraArgs);
    if (mTalker && outputs.size() > 1) {
        mTalker->addTalkerEmbeds(outputs[1]);
    }
    return outputs;
}

void Omni::responseInterleaved(const std::vector<int>& input_ids, std::ostream* os, const char* end_with,
                               int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    MNN::Timer ttfa_timer;

    if (max_new_tokens < 0) {
        max_new_tokens = mConfig->max_new_tokens();
    }

    // ---- 1. Thinker Prefill ----
    auto input_embeds = embedding(input_ids);
    if (input_embeds == nullptr) {
        MNN_ERROR("[Error]: Omni embedding returned nullptr. input_ids=%zu, audio_embeds=%zu, vision_embeds=%zu\n",
                  input_ids.size(), mAudioEmbeddings.size(), mVisionEmbeddings.size());
        return;
    }
    int seqLen = input_embeds->getInfo()->dim[mSeqLenIndex];
    mContext->prompt_len = seqLen;
    {
        std::lock_guard<std::mutex> _l(mContext->mutex);
        mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end());
    }

    MNN::Timer _t;
    auto outputs = forwardVec(input_embeds);
    if (outputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    updateContext(seqLen, 0);
    mContext->prefill_us += _t.durationInUs();

    // Sample first thinker token from prefill logits
    mContext->current_token = sample(outputs[0]);
    {
        std::lock_guard<std::mutex> _l(mContext->mutex);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);
    }
    updateContext(0, 1);

    // Output first token
    if (!is_stop(mContext->current_token)) {
        auto decodeStr = tokenizer_decode(mContext->current_token);
        {
            std::lock_guard<std::mutex> _l(mContext->mutex);
            mContext->generate_str += decodeStr;
        }
        if (nullptr != os) {
            *os << decodeStr << std::flush;
        }
    }

    // ---- 1.5 Run one Thinker decode step to populate mTalkerEmbeds[1] ----
    // This is needed so stepPrefill() has the thinker's first decode hidden state
    // (mTalkerEmbeds[1]) instead of mTextEos.
    // Only run if the first token is NOT a stop token.
    int thinker_tokens = 1;
    if (!is_stop(mContext->current_token)) {
        MNN::Timer t_decode;
        auto decode_outputs = forwardVec({mContext->current_token});
        if (decode_outputs.empty()) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return;
        }
        updateContext(1, 0);

        int next_token = sample(decode_outputs[0]);
        mContext->current_token = next_token;
        {
            std::lock_guard<std::mutex> _l(mContext->mutex);
            mContext->history_tokens.push_back(next_token);
            mContext->output_tokens.push_back(next_token);
        }
        updateContext(0, 1);
        mContext->decode_us += t_decode.durationInUs();
        thinker_tokens = 2;

        if (!is_stop(next_token)) {
            auto decodeStr = tokenizer_decode(next_token);
            {
                std::lock_guard<std::mutex> _l(mContext->mutex);
                mContext->generate_str += decodeStr;
            }
            if (nullptr != os) {
                *os << decodeStr << std::flush;
            }
        }
    }

    // ---- 2. Talker Prefill ----
    // Now mTalkerEmbeds has at least 2 thinker embed entries (from prefill + decode)
    if (!mTalker->hasEmbeds()) {
        MNN_ERROR("Talker embeds empty before prefill in interleaved mode\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    mTalker->stepPrefill();
    mContext->ttfa_us = ttfa_timer.durationInUs();

    // ---- 3. Interleaved Decode Loop ----
    bool thinker_done = is_stop(mContext->current_token);
    bool talker_done = !mTalker->doGenerate();
    int talker_step = 2;
    int64_t talker_decode_us = 0;

    while (!thinker_done || !talker_done) {
        if (mContext->status == LlmStatus::USER_CANCEL || mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }

        // ---- Thinker Decode Step ----
        if (!thinker_done && thinker_tokens < max_new_tokens) {
            MNN::Timer t_decode;
            auto decode_outputs = forwardVec({mContext->current_token});
            if (decode_outputs.empty()) {
                break;
            }
            updateContext(1, 0);

            int next_token = sample(decode_outputs[0]);
            mContext->current_token = next_token;
            {
                std::lock_guard<std::mutex> _l(mContext->mutex);
                mContext->history_tokens.push_back(next_token);
                mContext->output_tokens.push_back(next_token);
            }
            updateContext(0, 1);
            mContext->decode_us += t_decode.durationInUs();
            thinker_tokens++;

            if (is_stop(next_token)) {
                thinker_done = true;
                if (nullptr != os) {
                    *os << end_with << std::flush;
                }
            } else {
                auto decodeStr = tokenizer_decode(next_token);
                {
                    std::lock_guard<std::mutex> _l(mContext->mutex);
                    mContext->generate_str += decodeStr;
                }
                if (nullptr != os) {
                    *os << decodeStr << std::flush;
                }
            }
        } else if (!thinker_done) {
            thinker_done = true;
            if (nullptr != os) {
                *os << end_with << std::flush;
            }
        }

        // ---- Talker Decode Step ----
        if (!talker_done) {
            if (talker_step >= mTalker->maxNewTokens()) {
                talker_done = true;
            } else {
                MNN::Timer t_talker;
                mTalker->stepForward(talker_step++);
                talker_decode_us += t_talker.durationInUs();

                int talker_token = mTalker->getContext()->current_token;
                if (talker_token == 8292 || talker_token == 8294) {
                    talker_done = true;
                }
            }
        }
    }

    // Accumulate talker decode time collected during interleaved loop
    mTalker->mContext->decode_us += talker_decode_us;

    // ---- 4. Final Talker flush ----
    mTalker->finalize();

    if (thinker_tokens >= max_new_tokens) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
#ifdef DUMP_TALKER_PERFORMANCE
    {
        auto ctx = mTalker->getContext();
        float ttfa_s = mContext->ttfa_us / 1e6;
        float thinker_prefill_s = mContext->prefill_us / 1e6;
        float thinker_decode_s = mContext->decode_us / 1e6;
        float talker_prefill_s = ctx->prefill_us / 1e6;
        float talker_decode_s = ctx->decode_us / 1e6;
        float token2wav_s = ctx->audio_us / 1e6;
        float audio_duration = ctx->gen_seq_len / 50.0;
        printf("\n#################################\n");
        printf(" [interleaved mode]\n");
        printf("  thinker tokens num = %d\n", thinker_tokens);
        printf("   talker tokens num = %d\n", ctx->gen_seq_len);
        printf("    thinker prefill = %.2f s\n", thinker_prefill_s);
        printf("     thinker decode = %.2f s\n", thinker_decode_s);
        printf("     talker prefill = %.2f s\n", talker_prefill_s);
        printf("      talker decode = %.2f s\n", talker_decode_s);
        printf("       ttfa (total) = %.2f s\n", ttfa_s);
        printf("      token2wav     = %.2f s\n", token2wav_s);
        printf("       tts rtf      = %.2f \n", (talker_decode_s + token2wav_s) / audio_duration);
        printf("##################################\n");
    }
#endif
}

void Omni::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    if (mTalker) {
        mTalker->generate_init();
    }
    CHECK_LLM_RUNNING(mContext);
    if (!mTalker || !mTalker->mInterleaved) {
        MNN::Timer thinker_timer;
        generate(input_ids, max_new_tokens);
        mThinkerElapsedUs = thinker_timer.durationInUs();
    } else {
        responseInterleaved(input_ids, os, end_with, max_new_tokens);
    }
}

void Omni::setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) {
    if (mTalker) {
        mTalker->setWavformCallback(callback);
    }
}

void Omni::generateWavform() {
    if (mTalker) {
        if (!mTalker->mInterleaved) {
            mTalker->generate();
#ifdef DUMP_TALKER_PERFORMANCE
            auto context = mTalker->getContext();
            float prefill_s = context->prefill_us / 1e6;
            float decode_s = context->decode_us / 1e6;
            float ttfa_s = (mThinkerElapsedUs + context->ttfa_us) / 1e6;
            float token2wav_s = context->audio_us / 1e6;
            float dit_s = context->vision_us / 1e6;
            float tts_s = token2wav_s;
            if (mTalker->mStreamWithDecode) {
                tts_s += decode_s;
            }
            float audio_duration = context->gen_seq_len / 50.0;
            printf("\n#################################\n");
            printf("prompt tokens num = %d\n", context->prompt_len);
            printf("decode tokens num = %d\n", context->gen_seq_len);
            printf("  prefill time = %.2f s\n", prefill_s);
            printf("   decode time = %.2f s\n", decode_s);
            printf("      ttfa time = %.2f s\n", ttfa_s);
            printf("      dit time = %.2f s\n", dit_s);
            printf("token2wav time = %.2f s\n", token2wav_s);
            printf("      tts time = %.2f s\n", tts_s);
            printf("  prefill speed = %.2f tok/s\n", context->prompt_len / prefill_s);
            printf("   decode speed = %.2f tok/s\n", context->gen_seq_len / decode_s);
            printf("token2wav speed = %.2f tok/s\n", context->gen_seq_len / token2wav_s);
            printf("      tts rtf   = %.2f \n", tts_s / audio_duration);
            printf("##################################\n");
#endif
        }
    }
}

static std::string qwen3TTSNormalizeLanguage(std::string language) {
    std::transform(language.begin(), language.end(), language.begin(), [](unsigned char c) { return std::tolower(c); });
    static const std::vector<std::string> supported = {"auto",    "chinese",    "english", "german",
                                                       "italian", "portuguese", "spanish", "japanese",
                                                       "korean",  "french",     "russian"};
    for (const auto& item : supported) {
        if (language == item) {
            return language;
        }
    }
    return "";
}

bool Omni::generateTTS(const std::string& text, const std::string& language, int max_new_tokens,
                       const std::string& ref_audio) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!mTalker || !isQwen3TTSTalker(mConfig)) {
        MNN_ERROR("[Error]: current model does not support Qwen3-TTS generation\n");
        return false;
    }
    if (!mTokenizer) {
        MNN_ERROR("[Error]: Qwen3-TTS tokenizer is not loaded\n");
        return false;
    }
    std::string normalizedLanguage = qwen3TTSNormalizeLanguage(language);
    if (normalizedLanguage.empty()) {
        MNN_ERROR("[Error]: unsupported Qwen3-TTS language: %s\n", language.c_str());
        return false;
    }
    mTokenizer->set_chat_template_context("{\"qwen3_tts_language\":\"" + normalizedLanguage + "\"}");
    ChatMessages messages;
    messages.emplace_back("assistant", text);
    std::string prompt = mTokenizer->apply_chat_template(messages, true);
    if (prompt.empty()) {
        MNN_ERROR("[Error]: Qwen3-TTS chat template produced empty prompt\n");
        return false;
    }
    bool ok = mTalker->generateQwen3TTS(prompt, max_new_tokens, ref_audio);
    const auto* talkerContext = mTalker->getContext();
    if (talkerContext) {
        mContext->prompt_len = talkerContext->prompt_len;
        mContext->gen_seq_len = talkerContext->gen_seq_len;
        mContext->prefill_us = talkerContext->prefill_us;
        mContext->audio_us = talkerContext->audio_us;
        mContext->decode_us = talkerContext->decode_us;
        mContext->status = talkerContext->status;
        mContext->output_tokens = talkerContext->output_tokens;
    }
    return ok;
}

VARP Omni::ids_embedding(const std::vector<int>& ids) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (mContext->status == LlmStatus::INTERNAL_ERROR) {
        return nullptr;
    }
    auto inputs_ids = embedding(ids);
    if (inputs_ids == nullptr || inputs_ids->getInfo() == nullptr) {
        return nullptr;
    }
    int seqLen = inputs_ids->getInfo()->dim[mSeqLenIndex];
    mMeta->add = seqLen;
    mContext->prompt_len = seqLen;
    auto attention_mask = gen_attention_mask(seqLen);
    auto position_ids = gen_position_ids(seqLen);
    auto outputs = forwardRaw(inputs_ids, attention_mask, position_ids);
    if (outputs.empty()) {
        return nullptr;
    }
    return outputs[0];
}

bool Talker::load() {
    MNN::Express::ExecutorScope s(mExecutor);
    initRuntime();
    mSeqLenIndex = 1;
    if (isQwen3TTSTalker(mConfig)) {
        // The talker runs on the config-driven mRuntimeManager (created by initRuntime()
        // with KVMeta attached), so backend_type / thread_num / precision in config.json
        // take effect and the KV cache can be reused across decode frames.
        // The aux modules (prompt embedder, audio_code_predictor, codec embedder, speech decoder,
        // speaker encoder) are stateless full-sequence graphs: give them a manager on the
        // same user backend but WITHOUT KVCACHE_INFO, so their attention ops keep the
        // stateless path instead of fighting over mMeta with the talker.
        ScheduleConfig auxScheduleConfig;
        BackendConfig auxBackendConfig;
        auxScheduleConfig.type      = backend_type_convert(mConfig->backend_type());
        auxScheduleConfig.numThread = mConfig->thread_num();
        if (auxScheduleConfig.type == 3) {
            // opencl need set numThread = 64(buffer mode)
            auxScheduleConfig.numThread |= 64;
            auxScheduleConfig.numThread |= 512;
        }
        if (mConfig->power() == "high") {
            auxBackendConfig.power = BackendConfig::Power_High;
        } else if (mConfig->power() == "low") {
            auxBackendConfig.power = BackendConfig::Power_Low;
        }
        if (mConfig->memory() == "high") {
            auxBackendConfig.memory = BackendConfig::Memory_High;
        } else if (mConfig->memory() == "low") {
            auxBackendConfig.memory = BackendConfig::Memory_Low;
        }
        if (mConfig->precision() == "high") {
            auxBackendConfig.precision = BackendConfig::Precision_High;
        } else if (mConfig->precision() == "low") {
            auxBackendConfig.precision = BackendConfig::Precision_Low;
        }
        auxScheduleConfig.backendConfig = &auxBackendConfig;
        mQwen3RuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(auxScheduleConfig));
        setRuntimeHint(mQwen3RuntimeManager, true);
        if (backend_type_convert(mConfig->backend_type()) != 0) { // not cpu
            // mRuntimeManager already claims "<tmp>/mnn_cachefile.bin"; a second manager
            // on the same path would clobber the talker's tuned shader cache.
            std::string tmpPath = mConfig->tmp_path().empty() ? "." : mConfig->tmp_path();
            mQwen3RuntimeManager->setCache(tmpPath + "/mnn_cachefile_qwen3_aux.bin");
        }
        Module::Config module_config;
        module_config.shapeMutable = true;
        module_config.rearrange = true;
        constexpr int hiddenSize = 1024;

        mRuntimeManager->setExternalFile(mConfig->talker_weight().c_str());
        mModule.reset(Module::load({"inputs_embeds", "attention_mask", "position_ids"}, {"logits", "hidden_states"},
                                   mConfig->talker_model().c_str(), mRuntimeManager, &module_config),
                      Module::destroy);
        mRuntimeManager->setExternalFile("");

        mQwen3RuntimeManager->setExternalFile(mConfig->talker_weight().c_str());
        mQwen3PromptEmbedder.reset(Module::load({"codec_embeds", "text_raw_embeds", "tts_raw_embeds"},
                                                {"prompt_inputs_embeds", "trailing_text_hidden", "tts_pad_embed"},
                                                mConfig->talker_model().c_str(), mQwen3RuntimeManager, &module_config),
                                   Module::destroy);
        mQwen3RuntimeManager->setExternalFile("");

        const bool cpIncremental = mConfig->audio_code_predictor_incremental();
        if (cpIncremental) {
            // The incremental audio_code_predictor decodes one codec row at a time through
            // a KV cache, so it needs a manager whose KVCACHE_INFO points at its own
            // KVMeta. It cannot share the talker's meta: both models' attention ops
            // would read the same previous/remove/add channel while holding KV of
            // different lengths. And it cannot ride the aux manager either: that one
            // must stay stateless for the prompt embedder / speech decoder graphs.
            mQwen3CpMeta.reset(new KVMeta);
            // The predictor issues ~15 tiny seq-1 forwards per frame; on GPU backends
            // the per-forward submit/sync overhead dwarfs the compute of a 63M-param
            // model (measured on one OpenCL device: 50.5s vs 11.8s all-CPU decode for
            // 10s audio), so pin it to CPU. The talker and every stateless module keep
            // honoring the user's backend.
            ScheduleConfig cpScheduleConfig = auxScheduleConfig;
            BackendConfig cpBackendConfig = auxBackendConfig;
            // fp16-unsafe submodel: the talker's hidden states (absmax ~70) overflow
            // the predictor's RMSNorm/attention intermediates in fp16, collapsing its
            // logits to NaN, so always compute it in fp32 (the PyTorch model's fp16
            // forward collapses the same way).
            cpBackendConfig.precision = BackendConfig::Precision_High;
            cpScheduleConfig.backendConfig = &cpBackendConfig;
            if (auxScheduleConfig.type != MNN_FORWARD_CPU) {
                cpScheduleConfig.type = MNN_FORWARD_CPU;
                cpScheduleConfig.numThread = mConfig->thread_num();
            }
            mQwen3CpRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(cpScheduleConfig));
            setRuntimeHint(mQwen3CpRuntimeManager, true);
            mQwen3CpRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mQwen3CpMeta.get());
        }
        auto cpManager = cpIncremental ? mQwen3CpRuntimeManager : mQwen3RuntimeManager;
        cpManager->setExternalFile(mConfig->audio_code_predictor_weight().c_str());
        if (cpIncremental) {
            mQwen3AudioCodePredictor.reset(
                Module::load({"inputs_embeds", "attention_mask", "position_ids"}, {"logits"},
                             mConfig->audio_code_predictor_model().c_str(), cpManager, &module_config),
                Module::destroy);
        } else {
            mQwen3AudioCodePredictor.reset(
                Module::load({"talker_hidden_states", "codec_embeds", "attention_mask", "position_ids"}, {"logits"},
                             mConfig->audio_code_predictor_model().c_str(), cpManager, &module_config),
                Module::destroy);
        }
        cpManager->setExternalFile("");

        mQwen3RuntimeManager->setExternalFile(mConfig->codec_embedder_weight().c_str());
        mQwen3CodecEmbedder.reset(Module::load({"codec_embeds", "text_hidden"}, {"inputs_embeds"},
                                               mConfig->codec_embedder_model().c_str(), mQwen3RuntimeManager,
                                               &module_config),
                                  Module::destroy);
        mQwen3RuntimeManager->setExternalFile("");

        mQwen3RuntimeManager->setExternalFile(mConfig->speech_decoder_weight().c_str());
        mQwen3SpeechDecoder.reset(Module::load({"codes"}, {"waveform"}, mConfig->speech_decoder_model().c_str(),
                                               mQwen3RuntimeManager, &module_config),
                                  Module::destroy);
        mQwen3RuntimeManager->setExternalFile("");

        mQwen3RuntimeManager->setExternalFile(mConfig->speaker_encoder_weight().c_str());
        mQwen3SpeakerEncoder.reset(Module::load({"mels"}, {"speaker_embedding"},
                                                mConfig->speaker_encoder_model().c_str(), mQwen3RuntimeManager,
                                                &module_config),
                                   Module::destroy);
        mQwen3RuntimeManager->setExternalFile("");
        if (!mModule || !mQwen3PromptEmbedder || !mQwen3AudioCodePredictor || !mQwen3CodecEmbedder || !mQwen3SpeechDecoder ||
            !mQwen3SpeakerEncoder) {
            return false;
        }
        mDiskEmbedding.reset(new DiskEmbedding(mConfig, mConfig->talker_embedding_file()));
        mQwen3TextEmbedding.reset(
            new DiskEmbedding(mConfig, mConfig->talker_text_embedding_file(), mConfig->talker_text_hidden_size()));
        mQwen3AudioCodePredictorEmbedding.reset(
            new DiskEmbedding(mConfig, mConfig->audio_code_predictor_embedding_file(), hiddenSize));
        mMaxNewTokens = mConfig->talker_max_new_tokens();
        set_config("{\"sampler_type\":\"greedy\"}");
        mSampler.reset(Sampler::createSampler(mContext, mConfig));
        mContext->status = LlmStatus::RUNNING;
        return true;
    }
    // Talker codec tokens need a minimal sampler pipeline; default mixed_samplers includes
    // tfs/typical/min_p which over-filter the codec distribution causing greedy collapse.
    set_config(
        "{\"sampler_type\": \"mixed\", \"mixed_samplers\": [\"penalty\", \"topK\", \"topP\", \"temperature\"], "
        "\"temperature\": 0.9, \"topK\": 40, \"topP\": 0.8, \"penalty\": 1.05, \"penalty_window\": 64}");
    mSampler.reset(Sampler::createSampler(mContext, mConfig));
    mDiskEmbedding.reset(new DiskEmbedding(mConfig, mConfig->talker_embedding_file()));
    // some embeddings
    mMaxNewTokens = mConfig->talker_max_new_tokens();
    mInterleaved = mConfig->interleaved();
    std::string speaker = mConfig->talker_speaker();
    auto spk_dict = Express::Variable::loadMap(mConfig->spk_dict().c_str());
    mSpk = spk_dict[speaker + "_spk"];
    mCond = spk_dict[speaker + "_cond"];
    mTextBosToken = int(spk_dict[speaker + "_bos_token"]->readMap<float>()[0]);
    mTextBos = mThinker->embedding({mTextBosToken});
    mTextEos = mThinker->embedding({mTextEosToken});
    mTextPad = mThinker->embedding({mTextPadToken});
    mCodecBos = embedding({mCodecBosToken});
    mCodecPad = embedding({mCodecPadToken});

    Module::Config module_config;
    module_config.shapeMutable = false;
    module_config.rearrange    = true;
    std::vector<std::string> inputNames {"inputs_embeds", "attention_mask", "position_ids", "logits_index"};

    mModule.reset(Module::load(inputNames,
                                    {"logits"}, mConfig->talker_model().c_str(), mRuntimeManager, &module_config));
    if (mModule.get() == nullptr) {
        return false;
    }
    auto module_runtime = mProcessorRuntimeManager ? mProcessorRuntimeManager : mRuntimeManager;
    auto dit_runtime = module_runtime;
    // DiT requires fp32 precision on GPU backends; fp16 causes numerical collapse
    // (ODE solver iterates 4 steps, each step's error accumulates → output all zeros).
    auto processorBackendType = backend_type_convert(mConfig->backend_type(true));
    if (processorBackendType == MNN_FORWARD_OPENCL || processorBackendType == MNN_FORWARD_METAL ||
        processorBackendType == MNN_FORWARD_VULKAN || processorBackendType == MNN_FORWARD_CUDA) {
        ScheduleConfig ditSchedConfig;
        BackendConfig ditBackendConfig;
        ditBackendConfig.precision = BackendConfig::Precision_High;
        ditBackendConfig.memory = BackendConfig::Memory_Low;
        ditSchedConfig.type = processorBackendType;
        ditSchedConfig.numThread = mConfig->thread_num(true);
        if (processorBackendType == MNN_FORWARD_OPENCL) {
            ditSchedConfig.numThread |= 64;  // buffer mode
            ditSchedConfig.numThread |= 512; // tuning
        }
        ditSchedConfig.backendConfig = &ditBackendConfig;
        dit_runtime.reset(Executor::RuntimeManager::createRuntimeManager(ditSchedConfig));
    }
    // dit
    mPreDit.reset(Module::load({"cond", "spk", "code"}, {"code_embeds", "rope", "mask"},
                                mConfig->predit_model().c_str(), module_runtime, &module_config));
    mDit.reset(Module::load({"x", "code_embeds", "rope", "mask", "time"}, {"mel"}, mConfig->dit_model().c_str(),
                            dit_runtime, &module_config));
    // bigvgan
    mBigvgan.reset(Module::load({"generated_mel"},
                                {"waveform"}, mConfig->bigvgan_model().c_str(), module_runtime, &module_config));
    // autoregressive decode module
    mModulePool[std::make_pair(1, false)].reset(Module::clone(mModule.get()));
    // prefill module
    mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModule;
    if (mBigvgan.get() == nullptr || mPreDit.get() == nullptr || mDit.get() == nullptr) {
        return false;
    }
    mAsyncToken2Wav = (module_runtime.get() != mRuntimeManager.get());
    // GPU backends don't support cross-thread async token2wav; run synchronously.
    auto talkerBackend = backend_type_convert(mConfig->backend_type());
    if (talkerBackend == MNN_FORWARD_OPENCL || talkerBackend == MNN_FORWARD_METAL ||
        talkerBackend == MNN_FORWARD_VULKAN || talkerBackend == MNN_FORWARD_CUDA) {
        mAsyncToken2Wav = false;
    }

    if (mAsyncToken2Wav && doGenerate()) {
        startAsyncWorker();
    }
    
    mContext->status = LlmStatus::RUNNING;  // Set status to RUNNING after successful load
    return true;
}

Talker::~Talker() {
    if (mWavWorkerRunning) {
        stopAsyncWorker();
    }
}

void Talker::generate_init(std::ostream* os, const char* end_with) {
    if (!doGenerate()) { return; }
    Llm::generate_init(os, end_with);
    // stream generate init
    mTalkerEmbeds.clear();
    if (mInitialNoise.empty()) {
        mInitialNoise.resize(mMaxNewTokens * 2 * 80);
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < mMaxNewTokens * 2 * 80; ++i) {
            mInitialNoise[i] = distribution(generator);
        }
    }
    mWaveformBuffer.reserve(mMaxNewTokens * 2 * 240);
    mMelBuffer = nullptr;
    dit_start_index = 0;
    dit_left_padding = 0;
    vocoder_left_pad = 0;
    mWavLastDone.store(false);
    {
        std::lock_guard<std::mutex> lock(mWavQueueMutex);
        std::queue<WavChunk>().swap(mWavQueue);
    }
    {
        std::lock_guard<std::mutex> lock(mMelQueueMutex);
        std::queue<WavChunk>().swap(mMelQueue);
    }
}

Express::VARP Talker::embedding(const std::vector<int>& input_ids) {
    return Llm::embedding(input_ids);
}

Express::VARP Talker::gen_position_ids(int seq_len) {
    MNN::Express::ExecutorScope s(mExecutor);
    // mrope
    if (needNewVar(positionIds, 2, seq_len)) {
        positionIds = _Input({3, 1, seq_len}, NCHW, halide_type_of<int>());
    }
    auto ptr = positionIds->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = mContext->gen_seq_len + mPositionIds.back();
        ptr[1] = ptr[0];
        ptr[2] = ptr[0];
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = mPositionIds.mT[i];
            ptr[i + seq_len] = mPositionIds.mH[i];
            ptr[i + seq_len * 2] = mPositionIds.mW[i];
        }
    }
    return positionIds;
}

void Talker::setProcessorRuntimeManager(std::shared_ptr<Executor::RuntimeManager> processorRuntimeManager) {
    mProcessorRuntimeManager = processorRuntimeManager;
}

void Talker::setWavformCallback(const std::function<bool(const float*, size_t, bool)> callback) {
    mWavformCallback = callback;
}

static VARP runDit(Module* preDit, Module* dit, VARP condition, VARP speaker, int steps, int solver, int codec_size,
                   const int* codec_tokens, const float* initial_noise) {
    auto code = _Const(codec_tokens, {1, codec_size}, NCHW, halide_type_of<int>());
    const int max_duration = codec_size * 2;
    auto outputs = preDit->onForward({condition, speaker, code});
    auto code_embeds = outputs[0];
    auto rope = outputs[1];
    auto mask = outputs[2];
    const float step_ratio = 1.0 / (steps - 1);
    auto forward_dit = [&](float t, Express::VARP x) {
        auto pred = dit->onForward({x, code_embeds, rope, mask, _Const(t, {1}, NCHW)})[0];
        return pred;
    };
    auto y0 = _Input({1, max_duration, 80}, NCHW, halide_type_of<float>());
    auto noise = y0->writeMap<float>();
    if (initial_noise) {
        for (int i = 0; i < max_duration * 80; ++i) {
            noise[i] = initial_noise[i];
        }
    } else {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < max_duration * 80; ++i) {
            noise[i] = distribution(generator);
        }
    }
    for (int i = 0; i < steps - 1; i++) {
        float t0 = 1 - std::cos(M_PI / 2 * i * step_ratio);
        float t1 = 1 - std::cos(M_PI / 2 * (i + 1) * step_ratio);
        float dt = t1 - t0;
        auto k1 = dit->onForward({y0, code_embeds, rope, mask, _Const(t0, {1}, NCHW)})[0];
        if (solver == 1) {
            y0 = y0 + k1 * _Scalar<float>(dt);
        } else {
            constexpr float one_third = 1.0 / 3.0;
            constexpr float two_third = 2.0 / 3.0;
            auto kk1 = _Clone(k1, true);
            auto k2 = forward_dit(t0 + dt * one_third, y0 + k1 * _Scalar<float>(dt * one_third));
            auto kk2 = _Clone(k2, true);
            auto k3 = forward_dit(t0 + dt * two_third, y0 + _Scalar<float>(dt) * (k2 - k1 * _Scalar<float>(two_third)));
            auto kk3 = _Clone(k3, true);
            auto k4 = forward_dit(t1, y0 + _Scalar<float>(dt) * (k1 - k2 + k3));
            auto kk4 = _Clone(k4, true);
            auto dy = (kk1 + _Scalar<float>(3.0) * (kk2 + kk3) + kk4) * _Scalar<float>(dt * 0.125);
            y0 = y0 + dy;
        }
    }
    return _Permute(y0, {0, 2, 1});
}

VARP Talker::ditForward(const int codec_size, const int* codec_tokens, const float* initial_noise) {
    ExecutorScope scope(mExecutor);
    return runDit(mPreDit.get(), mDit.get(), mCond, mSpk, mConfig->dit_steps(), mConfig->dit_solver(), codec_size,
                  codec_tokens, initial_noise);
}

VARP Talker::bigvganForward(VARP mel) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto waveform = mBigvgan->forward(mel);
    return waveform;
}

void Talker::token2wav(bool talker_done) {
    MNN::Express::ExecutorScope s(mExecutor);
    int codec_size = mContext->gen_seq_len - dit_start_index;
    int chunk_size = dit_left_padding + dit_chunk_size + dit_right_padding;
    bool last_chunk = talker_done && (codec_size <= chunk_size);
    if (!last_chunk && codec_size < chunk_size) {
        return;
    }
    auto codec_ptr = mContext->output_tokens.data() + dit_start_index;
    auto noise_ptr = mInitialNoise.data() + dit_start_index * 160;
    int real_size = last_chunk ? codec_size : chunk_size;
    int mel_size = last_chunk ? -1 : dit_chunk_size * 2;
    MNN::Timer _t;
    // dit
    auto generated_mel = ditForward(real_size, codec_ptr, noise_ptr);
    generated_mel = _Slice(generated_mel, _var<int>({0, 0, dit_left_padding * 2}, {3}), _var<int>({-1, -1, mel_size}, {3}));
    mMelBuffer = (mMelBuffer == nullptr) ? generated_mel : _Concat({mMelBuffer, generated_mel}, -1);
    dit_left_padding = dit_left_context;
    dit_start_index += (chunk_size - dit_left_padding - dit_right_padding);
    // bigvga
    auto generated_waveform = bigvganForward(mMelBuffer);
    // append waveform to mWaveformBuffer
    auto ptr = generated_waveform->readMap<float>() + vocoder_left_pad * vocoder_upsample_rate;
    auto size = generated_waveform->getInfo()->size - (vocoder_left_pad + vocoder_right_pad) * vocoder_upsample_rate;
    mWaveformBuffer.insert(mWaveformBuffer.end(), ptr, ptr + size);
    vocoder_left_pad = vocoder_left_context;
    mMelBuffer = _Slice(mMelBuffer, _var<int>({0, 0, -vocoder_left_pad - vocoder_right_pad}, {3}), _var<int>({-1, -1, -1}, {3}));
    mContext->audio_us += _t.durationInUs();
    if (mWavformCallback) {
        bool res = mWavformCallback(ptr, size, last_chunk);
        if (!res) { return; }
    }
    if (talker_done && !last_chunk) {
        token2wav(true);
    }
}

VARP Talker::token2wav(const std::vector<int>& codec_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto generated_mel = ditForward(codec_tokens.size(), codec_tokens.data());
    auto waveform = bigvganForward(generated_mel);
    return waveform;
}

void Talker::startAsyncWorker() {
    if (mWavWorkerRunning.exchange(true)) return; // already running
    mDitWorkerThread = std::thread(&Talker::ditWorkerLoop, this);
    mVocoderWorkerThread = std::thread(&Talker::vocoderWorkerLoop, this);
}

void Talker::stopAsyncWorker() {
    mWavWorkerRunning.store(false);
    mWavQueueCond.notify_all();
    mMelQueueCond.notify_all();
    if (mDitWorkerThread.joinable()) {
        mDitWorkerThread.join();
    }
    if (mVocoderWorkerThread.joinable()) {
        mVocoderWorkerThread.join();
    }
}

void Talker::ditWorkerLoop() {
    BackendConfig backendConfig;
    auto forwardType = backend_type_convert(mConfig->backend_type(true));
    int numThread = mConfig->thread_num(true);
    if (forwardType == 3) {
        // OpenCL: force buffer memory mode (|64) + tuning (|512), matching Llm::initRuntime / Omni::load.
        // Without this the async executor defaults to IMAGE mode on non-MALI/INTEL GPUs (e.g. Adreno),
        // and the buffer-only quantized conv passes image tensors to __global args -> CL_INVALID_MEM_OBJECT.
        numThread |= 64;
        numThread |= 512;
    }
    auto executor = Express::Executor::newExecutor(forwardType, backendConfig, numThread);
    Express::ExecutorScope scope(executor);
    mPreDit_async.reset(Module::clone(mPreDit.get()));
    mDit_async.reset(Module::clone(mDit.get()));
    mSpk_async = _Clone(mSpk, true);
    mCond_async = _Clone(mCond, true);

    while (true) {
        WavChunk chunk;
        {
            std::unique_lock<std::mutex> lock(mWavQueueMutex);
            mWavQueueCond.wait(lock, [this] {
                return !mWavQueue.empty() || !mWavWorkerRunning;
            });
            
            if (!mWavWorkerRunning && mWavQueue.empty()) {
                break;
            }
            
            if (mWavQueue.empty()) {
                continue;
            }
            
            chunk = std::move(mWavQueue.front());
            mWavQueue.pop();
        }

        if (!chunk.codec_tokens.empty()) {
            auto generated_mel = ditForwardAsync((int)chunk.codec_tokens.size(),
                chunk.codec_tokens.data(), chunk.noise.data());
            generated_mel = _Slice(generated_mel,
                _var<int>({0, 0, chunk.mel_slice_start}, {3}),
                _var<int>({-1, -1, chunk.mel_slice_size}, {3}));
            auto mel_info = generated_mel->getInfo();
            chunk.mel_dims = mel_info->dim;
            chunk.mel.assign(generated_mel->readMap<float>(),
                             generated_mel->readMap<float>() + mel_info->size);
        }

        {
            std::lock_guard<std::mutex> lock(mMelQueueMutex);
            mMelQueue.push(std::move(chunk));
        }
        mMelQueueCond.notify_one();
    }

    {
        WavChunk sentinel;
        sentinel.is_last = true;
        std::lock_guard<std::mutex> lock(mMelQueueMutex);
        mMelQueue.push(std::move(sentinel));
    }
    mMelQueueCond.notify_one();

    mPreDit_async.reset();
    mDit_async.reset();
    mSpk_async = nullptr;
    mCond_async = nullptr;
}

void Talker::vocoderWorkerLoop() {
    BackendConfig backendConfig;
    auto forwardType = backend_type_convert(mConfig->backend_type(true));
    int numThread = mConfig->thread_num(true);
    if (forwardType == 3) {
        // OpenCL: force buffer memory mode (|64) + tuning (|512); see ditWorkerLoop for rationale.
        numThread |= 64;
        numThread |= 512;
    }
    auto executor = Express::Executor::newExecutor(forwardType, backendConfig, numThread);
    Express::ExecutorScope scope(executor);
    mBigvgan_async.reset(Module::clone(mBigvgan.get()));

    while (true) {
        WavChunk chunk;
        {
            std::unique_lock<std::mutex> lock(mMelQueueMutex);
            mMelQueueCond.wait(lock, [this] {
                return !mMelQueue.empty() || !mWavWorkerRunning;
            });
            if (!mWavWorkerRunning && mMelQueue.empty()) {
                break;
            }
            if (mMelQueue.empty()) {
                continue;
            }
            chunk = std::move(mMelQueue.front());
            mMelQueue.pop();
        }

        processWavChunk(chunk);

        if (chunk.is_last) {
            mWavLastDone.store(true);
            mWavQueueCond.notify_all();
        }
    }

    mBigvgan_async.reset();
}

VARP Talker::ditForwardAsync(const int codec_size, const int* codec_tokens, const float* initial_noise) {
    return runDit(mPreDit_async.get(), mDit_async.get(), mCond_async, mSpk_async, mConfig->dit_steps(),
                  mConfig->dit_solver(), codec_size, codec_tokens, initial_noise);
}

void Talker::processWavChunk(WavChunk& chunk) {
    if (chunk.mel.empty() || chunk.mel_dims.empty()) {
        if (chunk.is_last && mWavformCallback) {
            mWavformCallback(nullptr, 0, true);
        }
        return;
    }
    MNN::Timer _t;
    auto generated_mel = _Const(chunk.mel.data(), chunk.mel_dims, NCHW, halide_type_of<float>());
    mMelBuffer = (mMelBuffer == nullptr) ?
        generated_mel : _Concat({mMelBuffer, generated_mel}, -1);

    auto generated_waveform = mBigvgan_async->forward(mMelBuffer);

    auto ptr = generated_waveform->readMap<float>()
        + vocoder_left_pad * vocoder_upsample_rate;
    auto size = generated_waveform->getInfo()->size
        - (vocoder_left_pad + vocoder_right_pad) * vocoder_upsample_rate;
    mWaveformBuffer.insert(mWaveformBuffer.end(), ptr, ptr + size);
    vocoder_left_pad = vocoder_left_context;
    mMelBuffer = _Slice(mMelBuffer,
        _var<int>({0, 0, -vocoder_left_pad - vocoder_right_pad}, {3}),
        _var<int>({-1, -1, -1}, {3}));
    mContext->audio_us += _t.durationInUs();
    if (mWavformCallback) {
        mWavformCallback(ptr, size, chunk.is_last);
    }
}

void Talker::trySubmitChunkAsync(bool talker_done) {
    while (true) {
        int codec_size = mContext->gen_seq_len - dit_start_index;
        int chunk_size = dit_left_padding + dit_chunk_size + dit_right_padding;
        bool last_chunk = talker_done && (codec_size <= chunk_size);

        if (!last_chunk && codec_size < chunk_size) {
            return;
        }
        if (codec_size <= 0) {
            if (talker_done) {
                WavChunk wav_chunk;
                wav_chunk.is_last = true;
                {
                    std::lock_guard<std::mutex> lock(mWavQueueMutex);
                    mWavQueue.push(std::move(wav_chunk));
                }
                mWavQueueCond.notify_one();
            }
            return;
        }

        int real_size = last_chunk ? codec_size : chunk_size;

        WavChunk wav_chunk;
        wav_chunk.codec_tokens.assign(
            mContext->output_tokens.begin() + dit_start_index,
            mContext->output_tokens.begin() + dit_start_index + real_size);
        int noise_start = dit_start_index * 160;
        wav_chunk.noise.assign(
            mInitialNoise.begin() + noise_start,
            mInitialNoise.begin() + noise_start + real_size * 160);
        wav_chunk.mel_slice_start = dit_left_padding * 2;
        wav_chunk.mel_slice_size = last_chunk ? -1 : dit_chunk_size * 2;
        wav_chunk.is_last = last_chunk;

        dit_left_padding = dit_left_context;
        dit_start_index += (chunk_size - dit_left_padding - dit_right_padding);

        {
            std::lock_guard<std::mutex> lock(mWavQueueMutex);
            mWavQueue.push(std::move(wav_chunk));
        }
        mWavQueueCond.notify_one();

        if (last_chunk) {
            return;
        }
    }
}

int Talker::sample(Express::VARP logits, int offset, int size) {
    MNN::Express::ExecutorScope s(mExecutor);
    int token = Llm::sample(logits, offset, size);
    if (mAsyncToken2Wav) {
        if (!mWavWorkerRunning) {
            startAsyncWorker();
        }
        trySubmitChunkAsync(false);
    } else if (mStreamWithDecode) {
        token2wav();
    }
    return token;
}

bool Talker::generateQwen3TTS(const std::string& prompt, int maxFrames, const std::string& refAudio) {
    CHECK_LLM_RUNNING_RET(mContext, false);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!isQwen3TTSTalker(mConfig)) {
        MNN_ERROR("[Error]: generateQwen3TTS called on non-qwen3_tts talker\n");
        return false;
    }
    if (!mWavformCallback) {
        MNN_ERROR("[Error]: Qwen3-TTS waveform callback is not set\n");
        return false;
    }
    if (maxFrames <= 0) {
        maxFrames = mMaxNewTokens;
    }
    constexpr int hiddenSize = 1024;
    constexpr int codecEosToken = 2150;
    const int codeGroups = mConfig->audio_code_predictor_groups();
    const int audioCodePredictorSeq = codeGroups;
    std::vector<int> firstCodeCandidates(2049);
    for (int i = 0; i < 2048; ++i) {
        firstCodeCandidates[i] = i;
    }
    firstCodeCandidates[2048] = codecEosToken;

    auto inputIds = mThinker->Llm::tokenizer_encode(prompt);
    if (inputIds.empty()) {
        MNN_ERROR("[Error]: empty qwen3_tts prompt ids\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    mContext->prompt_len = static_cast<int>(inputIds.size());
    mContext->history_tokens = inputIds;
    mContext->output_tokens.clear();
    mContext->gen_seq_len = 0;

    if (inputIds.size() <= 6) {
        MNN_ERROR("[Error]: qwen3_tts prompt ids too short: %zu\n", inputIds.size());
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    std::vector<int> codecPrefix(inputIds.begin(), inputIds.begin() + 6);
    std::vector<int> textIds(inputIds.begin() + 6, inputIds.end());
    std::vector<int> ttsIds{mConfig->tts_bos_token_id(), mConfig->tts_eos_token_id(), mConfig->tts_pad_token_id()};
    VARP speakerEmbedding = nullptr;
    if (!refAudio.empty()) {
#ifdef LLM_SUPPORT_AUDIO
        speakerEmbedding = makeQwen3TTSSpeakerEmbedding(mQwen3SpeakerEncoder.get(), refAudio,
                                                        mConfig->speaker_encoder_sample_rate(), hiddenSize);
        if (speakerEmbedding.get() == nullptr) {
            MNN_ERROR("[Error]: failed to build Qwen3-TTS speaker embedding\n");
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return false;
        }
#else
        MNN_ERROR("[Error]: Qwen3-TTS ref_audio requires LLM_SUPPORT_AUDIO\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
#endif
    }
    auto codecEmbeds = makeQwen3TTSCodecEmbeds(mDiskEmbedding.get(), codecPrefix, speakerEmbedding, hiddenSize);
    auto textRawEmbeds = makeEmbeddingInput(mQwen3TextEmbedding.get(), textIds, mConfig->talker_text_hidden_size());
    auto ttsRawEmbeds = makeEmbeddingInput(mQwen3TextEmbedding.get(), ttsIds, mConfig->talker_text_hidden_size());
    if (codecEmbeds.get() == nullptr || textRawEmbeds.get() == nullptr || ttsRawEmbeds.get() == nullptr) {
        MNN_ERROR("[Error]: failed to build qwen3_tts prompt embeddings\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    auto promptOutputs = mQwen3PromptEmbedder->onForward({codecEmbeds, textRawEmbeds, ttsRawEmbeds});
    if (promptOutputs.size() != 3) {
        MNN_ERROR("[Error]: qwen3_tts prompt embedder output size mismatch: %zu\n", promptOutputs.size());
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    auto promptInfo = promptOutputs[0]->getInfo();
    auto embeddings = readTensorVector<float>(promptOutputs[0]);
    if (!promptInfo || promptInfo->dim.size() != 3 || embeddings.empty()) {
        MNN_ERROR("[Error]: invalid qwen3_tts prompt embeddings\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    int seqLen = promptInfo->dim[1];
    std::vector<int> generatedCodes;
    generatedCodes.reserve(maxFrames * codeGroups);

    const bool cpIncremental = mConfig->audio_code_predictor_incremental();
    VARP cpMask, cpPos;
    if (!cpIncremental) {
        makeCausalInputs(audioCodePredictorSeq, 1, cpMask, cpPos, mConfig->backend_type());
    }

    // talker.mnn's attention ops are exported with kv_cache=true, so the causal plane can be omitted.
    const std::string backendType = mConfig->backend_type();
    auto talkerMaskRequest = [](int len) {
        MaskRequest req;
        req.type   = AttentionMaskType::Causal;
        req.seqLen = len;
        req.rank   = 4;
        return req;
    };

    // Prefill: run the whole prompt sequence once to fill the KV cache; every later
    // frame decodes a single token, so the per-frame talker cost stays O(1) instead of
    // recomputing the growing prompt each frame.
    auto prefillEmbeds = makeTensorInputFromVector<float>({1, seqLen, hiddenSize}, embeddings);
    auto talkerPos     = makeTensorInput<int>({3, seqLen});
    fillPositionIds(talkerPos, 3, seqLen);
    VARP talkerMask = buildAttentionMask(talkerMaskRequest(seqLen), backendType);
    mMeta->remove = mMeta->previous;
    mMeta->add    = seqLen;
    MNN::Timer prefillTimer;
    auto talkerOutputs = mModule->onForward({prefillEmbeds, talkerMask, talkerPos});
    mContext->prefill_us += prefillTimer.durationInUs();
    mMeta->sync();
    if (talkerOutputs.size() != 2) {
        MNN_ERROR("[Error]: qwen3_tts talker prefill output size mismatch: %zu\n", talkerOutputs.size());
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }

    // Decode-step inputs: fixed shapes, only the position value changes per frame.
    VARP decodeMask = buildAttentionMask(talkerMaskRequest(1), backendType);
    VARP decodePos = makeTensorInput<int>({3, 1});

    // Incremental audio_code_predictor inputs. Per frame the predictor sequence is
    // [talker hidden, c0, c1, ..., c14]: prefill feeds rows 0..1 once, then every group
    // appends exactly one row through the predictor's KV cache. The old path re-ran the
    // whole padded sequence for every group (15 x 16 rows per frame); this runs each row
    // exactly once (16 rows per frame). All 15 lm heads read the last row, and the head
    // scoring code g is head g-1 at flat offset (g-1)*vocab -- the same offsets the old
    // graph produced, since head i was trained on row i+1.
    VARP cpPrefillMask, cpDecodeMask, cpPrefillPos, cpDecodePos;
    if (cpIncremental) {
        cpPrefillMask = buildAttentionMask(talkerMaskRequest(2), backendType);
        cpDecodeMask  = buildAttentionMask(talkerMaskRequest(1), backendType);
        cpPrefillPos = makeTensorInput<int>({1, 2});
        auto prefillPosPtr = cpPrefillPos->writeMap<int>();
        prefillPosPtr[0] = 0;
        prefillPosPtr[1] = 1;
        cpDecodePos = makeTensorInput<int>({1, 1});
    }

    MNN::Timer decodeTimer;
    for (int step = 0; step < maxFrames; ++step) {
        int firstCode = mSampler->sample(talkerOutputs[0], firstCodeCandidates);
        if (firstCode == codecEosToken) {
            break;
        }
        if (firstCode < 0) {
            MNN_ERROR("[Error]: invalid qwen3_tts first code at frame %d\n", step);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return false;
        }

        std::vector<int> frameCodes(codeGroups, 0);
        frameCodes[0] = firstCode;
        if (cpIncremental) {
            auto talkerHidden = readTensorVector<float>(talkerOutputs[1]);
            if (talkerHidden.size() != hiddenSize) {
                MNN_ERROR("[Error]: invalid qwen3_tts talker hidden size at frame %d: %zu\n", step,
                          talkerHidden.size());
                mContext->status = LlmStatus::INTERNAL_ERROR;
                return false;
            }
            auto cpPrefill = makeTensorInput<float>({1, 2, hiddenSize});
            auto prefillPtr = cpPrefill->writeMap<float>();
            std::copy(talkerHidden.begin(), talkerHidden.end(), prefillPtr);
            mDiskEmbedding->embedding({firstCode}, prefillPtr + hiddenSize);
            mQwen3CpMeta->remove = mQwen3CpMeta->previous;
            mQwen3CpMeta->add    = 2;
            auto cpOutputs = mQwen3AudioCodePredictor->onForward({cpPrefill, cpPrefillMask, cpPrefillPos});
            mQwen3CpMeta->sync();
            if (cpOutputs.size() != 1) {
                MNN_ERROR("[Error]: qwen3_tts audio_code_predictor prefill output size mismatch at frame %d: %zu\n", step,
                          cpOutputs.size());
                mContext->status = LlmStatus::INTERNAL_ERROR;
                return false;
            }
            int token = sample(cpOutputs[0], 0, 2048);
            frameCodes[1] = token < 0 ? 0 : token;
            for (int group = 2; group < codeGroups; ++group) {
                auto cpStep    = makeTensorInput<float>({1, 1, hiddenSize});
                auto stepPtr   = cpStep->writeMap<float>();
                mQwen3AudioCodePredictorEmbedding->embedding(
                    {(group - 2) * mConfig->audio_code_predictor_vocab_size() + frameCodes[group - 1]}, stepPtr);
                auto posPtr = cpDecodePos->writeMap<int>();
                posPtr[0]   = group;
                mQwen3CpMeta->add = 1;
                cpOutputs = mQwen3AudioCodePredictor->onForward({cpStep, cpDecodeMask, cpDecodePos});
                mQwen3CpMeta->sync();
                if (cpOutputs.size() != 1) {
                    MNN_ERROR("[Error]: qwen3_tts audio_code_predictor output size mismatch at frame %d group %d: %zu\n",
                              step, group, cpOutputs.size());
                    mContext->status = LlmStatus::INTERNAL_ERROR;
                    return false;
                }
                token = sample(cpOutputs[0], (group - 1) * 2048, 2048);
                frameCodes[group] = token < 0 ? 0 : token;
            }
        } else {
            for (int group = 1; group < codeGroups; ++group) {
                auto audioCodePredictorEmbeds =
                    makeQwen3AudioCodePredictorCodecEmbeds(mDiskEmbedding.get(), mQwen3AudioCodePredictorEmbedding.get(),
                                                      frameCodes, codeGroups,
                                                      mConfig->audio_code_predictor_vocab_size(), hiddenSize);
                if (audioCodePredictorEmbeds.get() == nullptr) {
                    MNN_ERROR("[Error]: failed to build qwen3_tts audio_code_predictor embeddings at frame %d group %d\n",
                              step, group);
                    mContext->status = LlmStatus::INTERNAL_ERROR;
                    return false;
                }
                auto cpOutputs =
                    mQwen3AudioCodePredictor->onForward({talkerOutputs[1], audioCodePredictorEmbeds, cpMask, cpPos});
                if (cpOutputs.size() != 1) {
                    MNN_ERROR("[Error]: qwen3_tts audio_code_predictor output size mismatch at frame %d group %d: %zu\n",
                              step, group, cpOutputs.size());
                    mContext->status = LlmStatus::INTERNAL_ERROR;
                    return false;
                }
                int token = sample(cpOutputs[0], (group - 1) * 2048, 2048);
                frameCodes[group] = token < 0 ? 0 : token;
            }
        }
        generatedCodes.insert(generatedCodes.end(), frameCodes.begin(), frameCodes.end());
        mContext->output_tokens.insert(mContext->output_tokens.end(), frameCodes.begin(), frameCodes.end());
        mContext->gen_seq_len = step + 1;

        if (step + 1 >= maxFrames) {
            break;
        }
        auto textHidden = selectTextHidden(promptOutputs[1], promptOutputs[2], step, hiddenSize);
        auto codecFrameEmbeds =
            makeQwen3AudioCodePredictorCodecEmbeds(mDiskEmbedding.get(), mQwen3AudioCodePredictorEmbedding.get(), frameCodes,
                                              codeGroups + 1, mConfig->audio_code_predictor_vocab_size(), hiddenSize);
        auto codecOutputs = mQwen3CodecEmbedder->onForward({codecFrameEmbeds, textHidden});
        if (codecOutputs.size() != 1) {
            MNN_ERROR("[Error]: qwen3_tts codec embedder output size mismatch at frame %d: %zu\n", step,
                      codecOutputs.size());
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return false;
        }
        auto nextEmbed = readTensorVector<float>(codecOutputs[0]);
        if (nextEmbed.size() != hiddenSize) {
            MNN_ERROR("[Error]: invalid qwen3_tts codec embed size at frame %d: %zu\n", step, nextEmbed.size());
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return false;
        }
        embeddings.insert(embeddings.end(), nextEmbed.begin(), nextEmbed.end());

        auto decodeEmbeds = makeTensorInputFromVector<float>({1, 1, hiddenSize}, nextEmbed);
        {
            auto posPtr = decodePos->writeMap<int>();
            posPtr[0] = posPtr[1] = posPtr[2] = seqLen;
        }
        mMeta->add = 1;
        talkerOutputs = mModule->onForward({decodeEmbeds, decodeMask, decodePos});
        mMeta->sync();
        if (talkerOutputs.size() != 2) {
            MNN_ERROR("[Error]: qwen3_tts talker output size mismatch at step %d: %zu\n", step, talkerOutputs.size());
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return false;
        }
        seqLen += 1;
    }
    mContext->decode_us += decodeTimer.durationInUs();

    int frames = static_cast<int>(generatedCodes.size() / codeGroups);
    if (frames <= 0) {
        MNN_ERROR("[Error]: no qwen3_tts codec frame generated\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    auto decoderCodes = makeTensorInput<int>({1, codeGroups, frames});
    auto codePtr = decoderCodes->writeMap<int>();
    for (int g = 0; g < codeGroups; ++g) {
        for (int t = 0; t < frames; ++t) {
            codePtr[g * frames + t] = generatedCodes[t * codeGroups + g];
        }
    }

    MNN::Timer wavTimer;
    auto wavOutputs = mQwen3SpeechDecoder->onForward({decoderCodes});
    mContext->audio_us += wavTimer.durationInUs();
    if (wavOutputs.size() != 1 || !wavOutputs[0]->getInfo() || !wavOutputs[0]->readMap<float>()) {
        MNN_ERROR("[Error]: invalid qwen3_tts speech decoder output\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    auto info = wavOutputs[0]->getInfo();
    bool callbackOk = mWavformCallback(wavOutputs[0]->readMap<float>(), info->size, true);
    mContext->status = callbackOk ? LlmStatus::NORMAL_FINISHED : LlmStatus::USER_CANCEL;
    return callbackOk;
}

void Talker::stepPrefill() {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!doGenerate()) { return; }

    mTalkerEmbeds.push_back(mTextEos);
    auto input_embeds = _Concat({mTalkerEmbeds[0], mTextBos + mCodecPad, mTalkerEmbeds[1] + mCodecBos}, 1);
    mPositionIds.push_back();
    mPositionIds.push_back();
    mContext->prompt_len = input_embeds->getInfo()->dim[1];
    MNN::Timer _t;
    auto logits = forward(input_embeds);
    mContext->current_token = sample(logits);
    {
        std::lock_guard<std::mutex> _l(mContext->mutex);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);
    }
    mContext->prefill_us += _t.durationInUs();
}

void Talker::stepForward(int stepIdx) {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!doGenerate()) {
        return;
    }

    auto input_embeds = embedding({mContext->current_token});
    if (stepIdx + 1 < mTalkerEmbeds.size()) {
        input_embeds = input_embeds + mTalkerEmbeds[stepIdx + 1];
    } else {
        mTalkerEmbeds.clear();
        input_embeds = input_embeds + mTextPad;
    }

    auto logits = forward(input_embeds);
    int token = sample(logits);

    mContext->current_token = token;
    {
        std::lock_guard<std::mutex> _l(mContext->mutex);
        mContext->history_tokens.push_back(token);
        mContext->output_tokens.push_back(token);
    }

    if (mAsyncToken2Wav) {
        trySubmitChunkAsync(false);
    }
}

void Talker::finalize() {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!doGenerate()) {
        return;
    }

    if (mAsyncToken2Wav) {
        trySubmitChunkAsync(true);
        std::unique_lock<std::mutex> lock(mWavQueueMutex);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!mWavLastDone.load()) {
            if (!mWavQueueCond.wait_until(lock, deadline, [this] { return mWavLastDone.load(); })) {
                MNN_ERROR("Talker async worker timeout; audio may be incomplete\n");
                break;
            }
        }
    } else {
        token2wav(true);
    }
}

void Talker::generate() {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!doGenerate()) {
        return;
    }

    MNN::Timer ttfa_timer;
    stepPrefill();
    mContext->ttfa_us = ttfa_timer.durationInUs();

    if (mAsyncToken2Wav && !mWavWorkerRunning) {
        startAsyncWorker();
    }

    MNN::Timer _t;
    for (int i = 1; i < mMaxNewTokens; i++) {
        stepForward(i);

        int token = mContext->current_token;
        if (token == 8292 || token == 8294) {
            break;
        }
    }
    mContext->decode_us += _t.durationInUs();

    finalize();
}

void Talker::setPostionIds(const MropeInfo& positionIds) {
    if (!doGenerate()) { return; }
    mPositionIds = MropeInfo(positionIds);
}

void Talker::addTalkerEmbeds(VARP talker_embeds) {
    if (!doGenerate()) { return; }
    mTalkerEmbeds.push_back(_Clone(talker_embeds, true));
}

} // namespace Transformer
} // namespace MNN
