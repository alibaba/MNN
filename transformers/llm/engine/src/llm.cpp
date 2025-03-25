//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <unordered_set>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/FileLoader.hpp"
#include "cpp/ExprDebug.hpp"
#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include "tokenizer.hpp"
#include "sampler.hpp"
#include "prompt.hpp"
// 0: no debug, 1: test op time, 2: print tensor info, 3: print tensor in output
#define DEBUG_MODE 0
//#define DEBUG_IMAGE

#include "httplib.h"
#ifdef LLM_SUPPORT_VISION
#include <cv/cv.hpp>
#endif
#ifdef LLM_SUPPORT_AUDIO
#include <audio/audio.hpp>
#endif

namespace MNN {
using namespace Express;
namespace Transformer {
struct KVMeta {
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::vector<int> reserveHost;
    void sync() {
        int revertNumber = 0;
        for (int i=0; i<n_reserve; ++i) {
            revertNumber += reserve[2*i+1];
        }
        previous = previous - remove + add + revertNumber;
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
};
typedef void (*DequantFunction)(const uint8_t*, float*, float, float, int);

static void q41_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size / 2; i++) {
        int x          = src[i];
        int x1         = x / 16;
        int x2         = x % 16;
        float w1       = x1 * scale + zero;
        float w2       = x2 * scale + zero;
        dst[2 * i]     = w1;
        dst[2 * i + 1] = w2;
    }
}

static void q81_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = (src[i]) * scale + zero;
    }
}

class DiskEmbedding {
public:
    explicit DiskEmbedding(const std::shared_ptr<LlmConfig>& config);
    ~DiskEmbedding() {}
    void embedding(const std::vector<int>& input_ids, float* ptr);

private:
    void seek_read(uint8_t* dst, size_t size, size_t offset);
    std::unique_ptr<uint8_t[]> mAlpha  = nullptr;
    std::unique_ptr<uint8_t[]> mWeight = nullptr;
    std::unique_ptr<FileLoader> mFile;
    DequantFunction mDequantFunc;
    int mHiddenSize, mTokenSize;
    float mOffset = 0.0f;
    bool mAsymc = true;
    int64_t mWeightOffset, mBlockNum, mQuantBlock, mQuantBit;
};

void DiskEmbedding::seek_read(uint8_t* dst, size_t size, size_t offset) {
    mFile->offset(offset);
    mFile->read((char*)dst, size);
}

DiskEmbedding::DiskEmbedding(const std::shared_ptr<LlmConfig>& config) {
    auto tie_embeddings = config->tie_embeddings();
    mHiddenSize        = config->hidden_size();
    if (tie_embeddings.size() == 5) {
        mWeightOffset     = tie_embeddings[0];
        mQuantBit         = tie_embeddings[3];
        mQuantBlock       = tie_embeddings[4];
        mBlockNum         = mHiddenSize / mQuantBlock;
        mTokenSize = mHiddenSize * mQuantBit / 8;
        mFile.reset(new FileLoader(config->llm_weight().c_str(), true));
        // TODO: optimize dequant function
        mDequantFunc      = mQuantBit == 8 ? q81_dequant_ref : q41_dequant_ref;
        auto a_offset   = tie_embeddings[1];
        auto alpha_size = tie_embeddings[2];
        size_t oc = (a_offset - mWeightOffset) / mHiddenSize * (8 / mQuantBit);
        
        mAlpha.reset(new uint8_t[alpha_size]);
        seek_read(mAlpha.get(), alpha_size, a_offset);
        mOffset = -(1 << (mQuantBit-1));
        if (alpha_size == sizeof(float) * mBlockNum * oc) {
            mAsymc = false;
        } else {
            MNN_ASSERT(alpha_size == 2 * sizeof(float) * mBlockNum * oc);
            mAsymc = true;
            auto alphaPtr = (float*)mAlpha.get();
            for (int i=0; i<mBlockNum * oc; ++i) {
                alphaPtr[2*i] = alphaPtr[2*i] + alphaPtr[2*i+1] * mOffset;
            }
        }
    } else {
        mTokenSize = mHiddenSize * sizeof(int16_t);
        mFile.reset(new FileLoader(config->embedding_file().c_str(), true));
    }
    if(mFile == nullptr || (!mFile->valid())) {
        MNN_ERROR("Failed to open embedding file!\n");
    }
    mWeight.reset(new uint8_t[mTokenSize]);
}

void DiskEmbedding::embedding(const std::vector<int>& input_ids, float* dst) {
    if (mAlpha.get()) {
        // quant
        if (mAsymc) {
            for (size_t i = 0; i < input_ids.size(); i++) {
                int token = input_ids[i];
                seek_read(mWeight.get(), mTokenSize, mWeightOffset + token * mTokenSize);
                auto dptr      = dst + i * mHiddenSize;
                auto alpha_ptr = reinterpret_cast<float*>(mAlpha.get()) + token * mBlockNum * 2;
                for (int n = 0; n < mBlockNum; n++) {
                    auto dst_ptr     = dptr + n * mQuantBlock;
                    uint8_t* src_ptr = mWeight.get() + n * (mQuantBlock * mQuantBit / 8);
                    float zero       = (alpha_ptr + n * 2)[0];
                    float scale      = (alpha_ptr + n * 2)[1];
                    mDequantFunc(src_ptr, dst_ptr, scale, zero, mQuantBlock);
                }
            }
        } else {
            for (size_t i = 0; i < input_ids.size(); i++) {
                int token = input_ids[i];
                seek_read(mWeight.get(), mTokenSize, mWeightOffset + token * mTokenSize);
                auto dptr      = dst + i * mHiddenSize;
                auto alpha_ptr = reinterpret_cast<float*>(mAlpha.get()) + token * mBlockNum;
                for (int n = 0; n < mBlockNum; n++) {
                    auto dst_ptr     = dptr + n * mQuantBlock;
                    uint8_t* src_ptr = mWeight.get() + n * (mQuantBlock * mQuantBit / 8);
                    float scale      = (alpha_ptr + n)[0];
                    float zero       = mOffset * scale;
                    mDequantFunc(src_ptr, dst_ptr, scale, zero, mQuantBlock);
                }
            }
        }
    } else {
        // bf16
        for (size_t i = 0; i < input_ids.size(); i++) {
            seek_read(mWeight.get(), mTokenSize, input_ids[i] * mTokenSize);
            int16_t* dst_ptr = reinterpret_cast<int16_t*>(dst + i * mHiddenSize);
            for (int j = 0; j < mHiddenSize; j++) {
                dst_ptr[j * 2]     = 0;
                dst_ptr[j * 2 + 1] = reinterpret_cast<int16_t*>(mWeight.get())[j];
            }
        }
    }
}

class Mllm : public Llm {
public:
    Mllm(std::shared_ptr<LlmConfig> config) : Llm(config) {
        if (config->is_visual()) {
            mVisionHeight = config->llm_config_.value("image_size", mVisionHeight);
            mVisionWidth  = mVisionHeight;
            mVisionPad    = config->llm_config_.value("image_pad", mVisionPad);
            mVisionStart  = config->llm_config_.value("vision_start", mVisionStart);
            mVisionEnd    = config->llm_config_.value("vision_end", mVisionEnd);
            mVisionMean   = config->llm_config_.value("image_mean", mVisionMean);
            mVisionNorm   = config->llm_config_.value("image_norm", mVisionNorm);
        }
        if (config->is_audio()) {
        }
    }
    ~Mllm() {
        mMulModule.reset();
    }
    virtual void load() override;
    virtual std::vector<int> tokenizer_encode(const std::string& query) override;
    virtual Express::VARP embedding(const std::vector<int>& input_ids) override;

private:
    int mVisionHeight = 448, mVisionWidth = 448, mVisionStart = 151857,
        mVisionEnd = 151858, mVisionPad = 151859, mAudioPad = 151646;
    std::vector<float> mVisionMean{122.7709383, 116.7460125, 104.09373615};
    std::vector<float> mVisionNorm{0.01459843, 0.01500777, 0.01422007};
    std::vector<int> multimode_process(const std::string& mode, std::string info);
    std::vector<int> vision_process(const std::string& file);
    std::vector<int> audio_process(const std::string& file);
    std::shared_ptr<Module> mMulModule;
    std::vector<VARP> mMulEmbeddings;
};

// Llm start
Llm* Llm::createLLM(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;
    if (config->is_visual() || config->is_audio()) {
        llm = new Mllm(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
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
    if (type_str == "npu")
        return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

std::string Llm::dump_config() {
    return mConfig->config_.dump();
}

bool Llm::set_config(const std::string& content) {
    return mConfig->config_.merge(content.c_str());
}

int file_size_m(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return -1;
    }
    long long fileSize = file.tellg();
    file.close();
    return fileSize / (1024 * 1024);
}

void Llm::initRuntime() {
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type      = backend_type_convert(mConfig->backend_type());
    config.numThread = mConfig->thread_num();
    if(config.type == 3){
        // opencl need set numThread = 64(buffer mode)
        config.numThread |= 64;
    }
    ExecutorScope::Current()->setGlobalExecutorConfig(config.type, cpuBackendConfig, config.numThread);
    if (mConfig->power() == "high") {
        cpuBackendConfig.power = BackendConfig::Power_High;
    } else if (mConfig->power() == "low") {
        cpuBackendConfig.power = BackendConfig::Power_Low;
    }
    if (mConfig->memory() == "high") {
        cpuBackendConfig.memory = BackendConfig::Memory_High;
    } else if (mConfig->memory() == "low") {
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
    }
    if (mConfig->precision() == "high") {
        cpuBackendConfig.precision = BackendConfig::Precision_High;
    } else if (mConfig->precision() == "low") {
        cpuBackendConfig.precision = BackendConfig::Precision_Low;
    }
    config.backendConfig = &cpuBackendConfig;

    mRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
    // Use 4 thread to load llm
    mRuntimeManager->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 4);

    mRuntimeManager->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
    mRuntimeManager->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, mConfig->quant_qkv());
    mRuntimeManager->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, mConfig->kvcache_limit());
    if (mConfig->use_cached_mmap()) {
        mRuntimeManager->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
    }
    std::string tmpPath = mConfig->tmp_path();
    if (mConfig->kvcache_mmap()) {
        mRuntimeManager->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    if (mConfig->use_mmap()) {
        mRuntimeManager->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }
    mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());

#if DEBUG_MODE == 1
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initTimeTrace();
#endif
#if DEBUG_MODE == 2
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initTensorStatic();
#endif
#if DEBUG_MODE == 3
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initDebug();
#endif
    {
        std::string cacheFilePath = tmpPath.length() != 0 ? tmpPath : ".";
        mRuntimeManager->setCache(cacheFilePath + "/mnn_cachefile.bin");
    }
}

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

void Llm::load() {
    initRuntime();
    // init module status
    // 1. load vocab
    mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
    mDiskEmbedding.reset(new DiskEmbedding(mConfig));
    mPrompt.reset(Prompt::createPrompt(mContext, mConfig));
    mSampler.reset(Sampler::createSampler(mContext, mConfig));
    // 3. load model
    Module::Config module_config;
    if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan") {
        module_config.shapeMutable = false;
    } else {
        module_config.shapeMutable = false;
    }
    module_config.rearrange    = true;
    // using base module for lora module
    if (mBaseModule != nullptr) {
        module_config.base = mBaseModule;
    }
    int layer_nums = mConfig->layer_nums();
    // load single model
    mModules.resize(1);
    std::string model_path = mConfig->llm_model();
    MNN_PRINT("load %s ... ", model_path.c_str());
    mRuntimeManager->setExternalFile(mConfig->llm_weight());
    mModules[0].reset(Module::load({"input_ids", "attention_mask", "position_ids", "logits_index"},
                                   {"logits"}, model_path.c_str(), mRuntimeManager, &module_config));
    MNN_PRINT("Load Module Done!\n");
    mDecodeModules.resize(mModules.size());
    for (int v = 0; v < mModules.size(); ++v) {
        mDecodeModules[v].reset(Module::clone(mModules[v].get()));
    }
    MNN_PRINT("Clone Decode Module Done!\n");

    mPrefillModules = mModules;
}

size_t Llm::apply_lora(const std::string& lora_path) {
    std::string model_path = mConfig->base_dir_ + "/" + lora_path;
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    module_config.base         = mModules.begin()->get();
    size_t lora_index          = mModules.size();
    mRuntimeManager->setHint(MNN::Interpreter::USE_CACHED_MMAP, 0);
    mModules.emplace_back(Module::load({"input_ids", "attention_mask", "position_ids"}, {"logits"},
                                        model_path.c_str(), mRuntimeManager, &module_config));
    select_module(lora_index);
    return lora_index;
}

Llm* Llm::create_lora(const std::string& lora_path) {
    auto llm = new Llm(std::make_shared<LlmConfig>(*mConfig));
    llm->set_config("{\"llm_model\": \"" + lora_path + "\", \"use_mmap\": false, \"use_cached_mmap\": false}");
    llm->mBaseModule = mModules.begin()->get();
    llm->load();
    return llm;
}

bool Llm::release_module(size_t index) {
    if (index >= mModules.size()) {
        return false;
    }
    if (mPrefillModules[0] == mModules[index]) {
        select_module(0);
    }
    mModules[index].reset();
    return true;
}

bool Llm::select_module(size_t index) {
    if (index >= mModules.size()) {
        return false;
    }
    if (mModules[index] == nullptr) {
        return false;
    }
    if (mDecodeModules.empty()) {
        mDecodeModules.resize(mModules.size());
        mPrefillModules.resize(mModules.size());
    }
    mDecodeModules[0].reset(Module::clone(mModules[index].get()));
    mPrefillModules[0] = mModules[index];
    return true;
}

void Llm::tuning(TuneType type, std::vector<int> candidates) {
    if (type != OP_ENCODER_NUMBER) {
        MNN_ERROR("tuning type not supported\n");
        return;
    }
    if (mConfig->backend_type() != "metal") {
        return;
    }

    mCurrentModules     = mDecodeModules;
    int64_t min_time     = INT64_MAX;
    int prefer_candidate = 10;
    for (auto& candidate : candidates) {
        mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, candidate);
        Timer _t;
        auto logits = forward({0});
        if (nullptr == logits.get()) {
            return;
        }
        if (logits->getInfo()->size == 0) {
            return;
        }
        auto token   = sample(logits);
        auto time = _t.durationInUs();
        if (time < min_time) {
            prefer_candidate = candidate;
            min_time         = time;
            // MNN_PRINT("op encode number:%d, decode time: %lld us\n", candidate, time);
        }
    }
    mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, prefer_candidate);
    // clear dirty tuning kv history
    setKVCacheInfo(0, getCurrentHistory());
    reset();
}

void Llm::switchMode(Llm::Stage stage) {
    switch (stage) {
        case Prefill:
            mCurrentModules = mPrefillModules;
            break;
        case Decode:
            mCurrentModules = mDecodeModules;
            break;
        default:
            break;
    }
}

void Llm::setKVCacheInfo(size_t add, size_t remove, int* reserve, int n_reserve) {
    if (remove > mMeta->previous) {
        remove = mMeta->previous;
    }
    mMeta->remove = remove;
    mMeta->reserve = reserve;
    mMeta->n_reserve = n_reserve;
    mMeta->add = add;
}

Express::VARP Llm::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos) {
    VARP logits;
    auto logitsIndex = _var<int>({-1}, {1});
    if (mConfig->all_logits()) {
        logitsIndex = _var<int>({0}, {1});
    }
    std::vector<Express::VARP> outputs;
    outputs = mCurrentModules.back()->onForward({hiddenState, mask, inputPos, logitsIndex});
    if (outputs.empty()) {
        return nullptr;
    }
    logits = outputs[0];
    mMeta->sync();
    return logits;
}

VARP Llm::forward(const std::vector<int>& input_ids, bool is_prefill) {
    int seq_len         = input_ids.size();
    mMeta->add          = seq_len;
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    auto hidden_states = embedding(input_ids);
    auto logits = forwardRaw(hidden_states, attention_mask, position_ids);
    mContext->all_seq_len += seq_len;
    mContext->gen_seq_len++;
    return logits;
}

int Llm::sample(VARP logits, int offset, int size) {
    auto logitsShape = logits->getInfo()->dim;
    if (logitsShape.size() == 3 && logitsShape[1] > 1) {
        // get last logits
        logits = _GatherV2(logits, _var<int>({logitsShape[1]-1}, {1}), _var<int>({1}, {1}));
    }
    if (offset && size) {
        logits = _Const(logits->readMap<float>() + offset, {size}, NHWC, halide_type_of<float>());
    }
    auto token_id = mSampler->sample(logits);
    mContext->history_tokens.push_back(token_id);
    mContext->output_tokens.push_back(token_id);
    return token_id;
}

void Llm::reset() {
    mContext->output_tokens.clear();
    mContext->history_tokens.clear();
    mContext->all_seq_len = 0;
}

void Llm::generate_init(std::ostream* os, const char* end_with) {
    // init status
    mContext->os = os;
    if (nullptr != end_with) {
        mContext->end_with = end_with;
    }
    if (!mContext->generate_str.empty()) {
        mContext->generate_str.clear();
    }
    mContext->gen_seq_len = 0;
    mContext->vision_us   = 0;
    mContext->audio_us    = 0;
    mContext->prefill_us  = 0;
    mContext->decode_us   = 0;
    mContext->current_token = 0;
    if (!mConfig->reuse_kv()) {
        mContext->all_seq_len = 0;
        mContext->history_tokens.clear();
        mMeta->remove = mMeta->previous;
    }
    mContext->output_tokens.clear();
    mCurrentModules = mPrefillModules;
}

size_t Llm::getCurrentHistory() const {
    return mMeta->previous;
}
void Llm::eraseHistory(size_t begin, size_t end) {
    if (0 == end) {
        end = mMeta->previous;
    }
    if (end > mMeta->previous || begin >= end) {
        MNN_ERROR("Invalid erase range history larger than current\n");
        return;
    }
    if (mMeta->remove != 0) {
        MNN_ERROR("MNN-LLM: erase history hasn't been executed by response, override erase info\n");
    }
    mMeta->remove = mMeta->previous - begin;
    if (end != mMeta->previous) {
        mMeta->reserveHost.resize(2);
        mMeta->reserve = mMeta->reserveHost.data();
        mMeta->n_reserve = 1;
        mMeta->reserve[0] = end - begin;
        mMeta->reserve[1] = mMeta->previous - end;
    }
}

bool Llm::stoped() {
    return is_stop(mContext->current_token);
}

void Llm::generate(int max_token) {
    int len = 0;
    while (len < max_token) {
        MNN::Timer _t;
        auto decodeStr = tokenizer_decode(mContext->current_token);
        mContext->generate_str += decodeStr;
        if (nullptr != mContext->os) {
            *mContext->os << decodeStr;
            *mContext->os << std::flush;
        }
        // mContext->history_tokens.push_back(mContext->current_token);
        mMeta->remove = 0;
        auto logits = forward({mContext->current_token});
        len++;
        if (nullptr == logits.get()) {
            break;
        }
        if (logits->getInfo()->size == 0) {
            break;
        }
        mContext->current_token = sample(logits);
        mContext->decode_us += _t.durationInUs();
        if (is_stop(mContext->current_token) && nullptr != mContext->os) {
            *mContext->os << mContext->end_with << std::flush;
            break;
        }
    }
}

std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_tokens) {
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }
    mContext->prompt_len = static_cast<int>(input_ids.size());
    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    Timer _t;
    mCurrentModules = mPrefillModules;
    auto logits      = forward(input_ids);
    if (nullptr == logits.get()) {
        return {};
    }
    mContext->current_token = sample(logits);
    logits = nullptr;
    mCurrentModules = mDecodeModules;
    mContext->prefill_us = _t.durationInUs();
    generate(max_tokens - 1);

    return mContext->output_tokens;
}

std::vector<int> Llm::tokenizer_encode(const std::string& user_content) {
    return mTokenizer->encode(user_content);
}

void Llm::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    generate(input_ids, max_new_tokens);
}

void Llm::response(const std::string& user_content, std::ostream* os, const char* end_with, int max_new_tokens) {
    auto prompt = user_content;
    if (mConfig->use_template()) {
        prompt = mPrompt->applyTemplate(user_content);
    }
    std::vector<int> input_ids = tokenizer_encode(prompt);
    response(input_ids, os, end_with, max_new_tokens);
}

void Llm::response(const ChatMessages& chat_prompts, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (chat_prompts.empty()) {
        return;
    }
    auto prompt = mPrompt->applyTemplate(chat_prompts);
    std::vector<int> input_ids = tokenizer_encode(prompt);
    response(input_ids, os, end_with, max_new_tokens);
}

Llm::Llm(std::shared_ptr<LlmConfig> config) : mConfig(config) {
    mContext.reset(new LlmContext);
    mMeta.reset(new KVMeta);
}

Llm::~Llm() {
#if DEBUG_MODE == 1
    if (nullptr != gTimeTraceInfo) {
        float opSummer       = 0.0f;
        float opFlopsSummber = 0.0f;
        for (auto& iter : gTimeTraceInfo->mTypes) {
            float summer      = 0.0f;
            float summerflops = 0.0f;
            for (auto& t : iter.second) {
                for (auto& t0 : t.second) {
                    summer += t0.first;
                    summerflops += t0.second;
                }
            }
            summer      = summer;
            summerflops = summerflops;
            MNN_PRINT("%s : %.7f, FLOP: %.7f, Speed: %.7f GFlops\n", iter.first.c_str(), summer, summerflops,
                      summerflops / summer);
            opSummer += summer;
            opFlopsSummber += summerflops;
        }
        MNN_PRINT("OP Summer: %.7f, Flops: %.7f, Speed: %.7f GFlops\n", opSummer, opFlopsSummber,
                  opFlopsSummber / opSummer);
    }
#endif
    mCurrentModules.clear();
    mDecodeModules.clear();
    mPrefillModules.clear();
    mModules.clear();
    mRuntimeManager.reset();
    mProcessorRuntimeManager.reset();
}

bool Llm::reuse_kv() { return mConfig->reuse_kv(); }

static inline bool needNewVar(VARP var, int axis, int seq_len) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    return false;
}

VARP Llm::embedding(const std::vector<int>& input_ids) {
    AUTOTIME;
    int hidden_size = mConfig->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());
    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    // disk embedding to save memory
    mDiskEmbedding->embedding(input_ids, res->writeMap<float>());
    return res;
}

std::string Llm::tokenizer_decode(int id) {
    std::string word = mTokenizer->decode(id);
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length() - 1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word    = static_cast<char>(num);
    }
    return word;
}

VARP Llm::gen_attention_mask(int seq_len) {
    int kv_seq_len = mContext->all_seq_len + seq_len;
    if (seq_len == 1) {
        kv_seq_len = seq_len;
    }
    if (mConfig->attention_mask() == "float") {
        if (needNewVar(attentionMask, 2, seq_len)) {
            attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
        } else {
            return attentionMask;
        }
        auto ptr = attentionMask->writeMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                int row = i + mContext->all_seq_len;
                ptr[kv_seq_len * i + j] = (j > row) * std::numeric_limits<float>::lowest();
            }
        }
        return attentionMask;
    } else {
        if (needNewVar(attentionMask, 2, seq_len)) {
            attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<int>());
        } else {
            return attentionMask;
        }
        auto ptr = attentionMask->writeMap<int>();
        if (mConfig->attention_mask() == "glm") {
            // chatglm
            for (int i = 0; i < seq_len * kv_seq_len; i++) {
                ptr[i] = 0;
            }
            if (seq_len > 1) {
                for (int i = 1; i < seq_len; i++) {
                    ptr[seq_len * i - 1] = 1;
                }
            }
        } else {
            bool is_glm2 = mConfig->attention_mask() == "glm2";
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    int row              = i + mContext->all_seq_len;
                    ptr[seq_len * i + j] = is_glm2 ? j > row : j <= row;
                }
            }
        }
        return attentionMask;
    }
}

VARP Llm::gen_position_ids(int seq_len) {
    if (mConfig->attention_mask() == "glm") {
        // chatglm
        if (needNewVar(positionIds, 2, seq_len)) {
            positionIds = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = positionIds->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = mContext->all_seq_len - mContext->gen_seq_len - 2;
            ptr[1] = mContext->gen_seq_len + 1;
        } else {
            for (int i = 0; i < seq_len - 1; i++) {
                ptr[i]           = i;
                ptr[seq_len + i] = 0;
            }
            ptr[seq_len - 1]     = seq_len - 2;
            ptr[2 * seq_len - 1] = 1;
        }
        return positionIds;
    } else {
        bool is_glm2 = mConfig->attention_mask() == "glm2";
        if (needNewVar(positionIds, 0, seq_len)) {
            positionIds = _Input({seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = positionIds->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
        } else {
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mContext->all_seq_len;
            }
        }
        return positionIds;
    }
}

bool Llm::is_stop(int token_id) {
    return mTokenizer->is_stop(token_id);
}

void Mllm::load() {
    Llm::load();
    if (mConfig->mllm_config_.empty()) {
        mProcessorRuntimeManager = mRuntimeManager;
    } else {
        ScheduleConfig config;
        BackendConfig cpuBackendConfig;
        config.type      = backend_type_convert(mConfig->backend_type(true));;
        config.numThread = mConfig->thread_num(true);
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
        mProcessorRuntimeManager->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
        mProcessorRuntimeManager->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
        mProcessorRuntimeManager->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, mConfig->quant_qkv());
        mProcessorRuntimeManager->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, mConfig->kvcache_limit());
        std::string tmpPath = mConfig->tmp_path();
        if (mConfig->kvcache_mmap()) {
            mProcessorRuntimeManager->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
        }
        if (mConfig->use_mmap()) {
            mProcessorRuntimeManager->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
        }
    }
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    if (mConfig->is_visual()) {
        mProcessorRuntimeManager->setExternalFile(mConfig->visual_model() + ".weight");
        mMulModule.reset(Module::load({}, {}, mConfig->visual_model().c_str(), mProcessorRuntimeManager, &module_config));
    }
    if (mConfig->is_audio()) {
        mProcessorRuntimeManager->setExternalFile(mConfig->audio_model() + ".weight");
        mMulModule.reset(Module::load({}, {}, mConfig->audio_model().c_str(), mProcessorRuntimeManager, &module_config));
    }
}

std::vector<int> Mllm::vision_process(const std::string& file) {
#ifdef LLM_SUPPORT_VISION
    VARP image = MNN::CV::imread(file);
    if (image == nullptr) {
        MNN_PRINT("Mllm Can't open image: %s\n", file.c_str());
        return std::vector<int>(0);
    }
    Timer _t;
    VARP image_embedding;

    if (mMulModule->getInfo()->inputNames[0] == "patches") {
        // Qwen2-VL
        mVisionHeight = round(mVisionHeight / 28.0) * 28;
        mVisionWidth = round(mVisionWidth / 28.0) * 28;
        image        = MNN::CV::resize(image, {mVisionHeight, mVisionWidth}, 0, 0,
                                     MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                     mVisionMean, mVisionNorm);
        image        = Express::_Unsqueeze(image, {0});
        image        = Express::_Convert(image, NCHW);
        auto patches = Express::_Concat({image, image}, 0);
        auto patches_dim = patches->getInfo()->dim;
        int temporal = patches_dim[0];
        int channel  = patches_dim[1];
        int height   = patches_dim[2];
        int width    = patches_dim[3];
        constexpr int temporal_patch_size = 2;
        constexpr int patch_size = 14;
        constexpr int merge_size = 2;
        int grid_t = temporal / temporal_patch_size;
        int grid_h = height / patch_size;
        int grid_w = width / patch_size;
        // build patches
        patches = Express::_Reshape(patches, {
            grid_t, temporal_patch_size,
            channel,
            grid_h / merge_size, merge_size, patch_size,
            grid_w / merge_size, merge_size, patch_size,
        });
        patches = Express::_Permute(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8});
        patches = Express::_Reshape(patches, {
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size
        });
        const int seq_len = grid_t * grid_h * grid_w;
        // build position_ids
        const int wblock_size = merge_size * merge_size;
        const int hblock_size = wblock_size * grid_w / merge_size;
        VARP position_ids = Express::_Input({2, seq_len}, NCHW, halide_type_of<int>());
        auto hpos_ptr = position_ids->writeMap<int>();
        auto wpos_ptr = hpos_ptr + seq_len;
        for (int i = 0; i < grid_h; i++) {
            int h_idx = i / merge_size, h_off = i % merge_size;
            for (int j = 0; j < grid_w; j++) {
                int w_idx = j / merge_size, w_off = j % merge_size;
                int index = h_idx * hblock_size + w_idx * wblock_size + h_off * 2 + w_off;
                hpos_ptr[index] = i;
                wpos_ptr[index] = j;
            }
        }
        // build attention_mask
        VARP attention_mask = Express::_Input({1, seq_len, seq_len}, NCHW);
        ::memset(attention_mask->writeMap<float>(), 0, seq_len * seq_len * sizeof(float));
#ifdef DEBUG_IMAGE
        patches.fix(MNN::Express::VARP::CONSTANT);
        patches->setName("patches");
        position_ids.fix(MNN::Express::VARP::CONSTANT);
        position_ids->setName("position_ids");
        attention_mask.fix(MNN::Express::VARP::CONSTANT);
        attention_mask->setName("attention_mask");
        MNN::Express::Variable::save({patches, position_ids, attention_mask}, "input.mnn");
#endif
        image_embedding = mMulModule->onForward({patches, position_ids, attention_mask})[0];
#ifdef DEBUG_IMAGE
        image_embedding->setName("image_embeds");
        MNN::Express::Variable::save({image_embedding}, "output.mnn");
#endif
    } else {
        image           = MNN::CV::resize(image, {mVisionHeight, mVisionWidth}, 0, 0,
                                          MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                          mVisionMean, mVisionNorm);
        image           = Express::_Unsqueeze(image, {0});
        image           = Express::_Convert(image, NC4HW4);
        image_embedding = mMulModule->forward(image);
    }
    mContext->vision_us = _t.durationInUs();
    mMulEmbeddings.push_back(image_embedding);
    int visual_len = image_embedding->getInfo()->dim[0];
    std::vector<int> img_ids(visual_len, mVisionPad);
    img_ids.insert(img_ids.begin(), mVisionStart);
    img_ids.push_back(mVisionEnd);
    return img_ids;
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Mllm::audio_process(const std::string& file) {
#ifdef LLM_SUPPORT_AUDIO
    constexpr int sample_rate = 16000;
    auto load_res        = MNN::AUDIO::load(file, sample_rate);
    VARP waveform        = load_res.first;
    if (waveform == nullptr) {
        MNN_PRINT("Mllm Can't open audio: %s\n", file.c_str());
        return std::vector<int>(0);
    }
    // int sample_rate      = load_res.second;
    int wav_len          = waveform->getInfo()->dim[0];
    int hop_length       = 160;
    Timer _t;
    auto input_features  = MNN::AUDIO::whisper_fbank(waveform);
    auto audio_embedding = mMulModule->forward(input_features);
    audio_embedding = _Permute(audio_embedding, {1, 0, 2});
    mContext->audio_us = _t.durationInUs();
    mMulEmbeddings.push_back(audio_embedding);
    int embed_len = audio_embedding->getInfo()->dim[0];
    std::vector<int> audio_ids(embed_len, mAudioPad);
    return audio_ids;
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Mllm::multimode_process(const std::string& mode, std::string info) {
    auto file_info = info;
    if (mode == "img") {
        std::regex hw_regex(R"(<hw>(.*?)</hw>)");
        std::sregex_iterator iter(info.begin(), info.end(), hw_regex);
        std::sregex_iterator end;
        file_info = "";

        size_t currentPosition = 0;
        if (iter != end) {
            std::smatch match = *iter;
            size_t matchPosition = match.position();
            if (matchPosition > currentPosition) {
                file_info.append(info.substr(currentPosition, matchPosition - currentPosition));
            }

            std::stringstream hw_ss(match.str(1));
            char comma;
            hw_ss >> mVisionHeight >> comma >> mVisionWidth;
            currentPosition = matchPosition + match.length();
        }
        if (currentPosition < info.length()) {
            file_info.append(info.substr(currentPosition));
        }
        // std::cout << "hw: " << mVisionHeight << ", " << mVisionWidth << std::endl;
        // std::cout << "file: " << file_info << std::endl;
    }
    if (file_info.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))");
        std::smatch url_match_result;
        std::string host, path;
        if (std::regex_search(file_info, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();
            path = url_match_result[2].str();
        }
        // std::cout << host << "#" << path << std::endl;
        httplib::Client cli(host);
        auto res  = cli.Get(path);
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
    if (mode == "img" && mConfig->is_visual()) {
        return vision_process(file_info);
    }
    if (mode == "audio" && mConfig->is_audio()) {
        return audio_process(file_info);
    }
    return std::vector<int>(0);
}

std::vector<int> Mllm::tokenizer_encode(const std::string& prompt) {
    // split query
    std::regex multimode_regex("<(img|audio)>(.*?)</\\1>");
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<std::string> img_infos;
    std::vector<int> ids{};

    while (std::regex_search(searchStart, prompt.cend(), match, multimode_regex)) {
        // std::cout << "img match: " << match[1].str() << std::endl;
        auto txt_ids = mTokenizer->encode(match.prefix().str());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        auto mul_ids = multimode_process(match[1].str(), match[2].str());
        ids.insert(ids.end(), mul_ids.begin(), mul_ids.end());
        searchStart = match.suffix().first;
    }
    if (searchStart != prompt.cend()) {
        auto txt_ids = mTokenizer->encode(std::string(searchStart, prompt.cend()));
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    return ids;
}

VARP Mllm::embedding(const std::vector<int>& input_ids) {
    if (input_ids.size() == 1) {
        return Llm::embedding(input_ids);
    }
    std::vector<VARP> embeddings;
    int mul_idx = 0;
    std::vector<int> cur_txt_ids;
    bool in_audio = false;
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        // audio
        if (in_audio) {
            if (id == mAudioPad) {
                continue;
            } else {
                cur_txt_ids.clear();
                in_audio = false;
            }
        } else if (id == mAudioPad) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mMulEmbeddings[mul_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            in_audio = true;
        }
        // vision
        if (id == mVisionPad) {
            continue;
        }
        cur_txt_ids.push_back(id);
        if (id == mVisionStart) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mMulEmbeddings[mul_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
        } else if (id == mVisionEnd) {
            cur_txt_ids.clear();
            cur_txt_ids.push_back(id);
        }
    }
    mMulEmbeddings.clear();
    if (!cur_txt_ids.empty()) {
        auto txt_embedding = Llm::embedding(cur_txt_ids);
        embeddings.push_back(txt_embedding);
    }
    auto embedding = Express::_Concat(embeddings, 0);
    return embedding;
}
// Llm end

// Embedding start
float Embedding::dist(VARP var0, VARP var1) {
    auto distVar = _Sqrt(_ReduceSum(_Square(var0 - var1)));
    auto dist    = distVar->readMap<float>()[0];
    return dist;
}

Embedding* Embedding::createEmbedding(const std::string& config_path, bool load) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Embedding* embedding = new Embedding(config);
    if (load) {
        embedding->load();
    }
    return embedding;
}

Embedding::Embedding(std::shared_ptr<LlmConfig> config) : Llm(config) {
}

int Embedding::dim() const {
    return mConfig->hidden_size();
}

void Embedding::load() {
    initRuntime();
    printf("load tokenizer\n");
    std::cout << mConfig->tokenizer_file() << std::endl;
    // 1. load vocab
    mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
    printf("load tokenizer Done\n");
    mDiskEmbedding.reset(new DiskEmbedding(mConfig));
    // 2. load model
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    auto model_path            = mConfig->llm_model();
    MNN_PRINT("load %s ... ", model_path.c_str());
    mModules.resize(1);
    mModules[0].reset(Module::load({"input_ids", "attention_mask", "position_ids"}, {"sentence_embeddings"},
                                   model_path.c_str(), mRuntimeManager, &module_config));
    MNN_PRINT("Done!\n");
}

VARP Embedding::ids_embedding(const std::vector<int>& ids) {
    int prompt_len           = ids.size();
    auto inputs_ids          = embedding(ids);
    auto attention_mask      = gen_attention_mask(prompt_len);
    auto position_ids        = gen_position_ids(prompt_len);
    auto outputs             = mModules[0]->onForward({inputs_ids, attention_mask, position_ids});
    auto sentence_embeddings = outputs[0];
    return sentence_embeddings;
}

VARP Embedding::txt_embedding(const std::string& txt) {
    return ids_embedding(tokenizer_encode(txt));
}

VARP Embedding::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, 1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr            = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = 1;
    }
    return attention_mask;
}

VARP Embedding::gen_position_ids(int seq_len) {
    auto position_ids = _Input({1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr          = position_ids->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = i;
    }
    return position_ids;
}
// Embedding end
} // namespace Transformer
} // namespace MNN
