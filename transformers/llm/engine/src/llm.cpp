//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "cpp/ExprDebug.hpp"
#include "llm/llm.hpp"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "prompt.hpp"
#include "tokenizer.hpp"
#include "diskembedding.hpp"
#include "sampler.hpp"
#include "omni.hpp"
#include "speculative_decoding/generate.hpp"

// 0: no debug, 1: test op time, 2: print tensor info, 3: print tensor in output
#define DEBUG_MODE 0
//#define DEBUG_IMAGE

namespace MNN {
using namespace Express;
namespace Transformer {

void KVMeta::sync() {
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

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

Llm* Llm::createLLM(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;
    if (config->is_visual() || config->is_audio() || config->has_talker()) {
        llm = new Omni(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
}
void Llm::destroy(Llm* llm) {
    delete llm;
}

std::string Llm::dump_config() {
    return mConfig->config_.dump();
}

bool Llm::set_config(const std::string& content) {
    auto res = mConfig->config_.merge(content.c_str());
    // update prompt
    if(mPrompt != nullptr) {
        mPrompt->setParams(mConfig);
    } else {
        mPrompt.reset(Prompt::createPrompt(mContext, mConfig));
    }
    return res;
}

void Llm::setRuntimeHint(std::shared_ptr<Express::Executor::RuntimeManager> &rtg) {
    rtg->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 4);

    rtg->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
    rtg->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, mConfig->quant_qkv());
    rtg->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, mConfig->kvcache_limit());
    if (mConfig->use_cached_mmap()) {
        rtg->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
    }
    std::string tmpPath = mConfig->tmp_path();
    if (mConfig->kvcache_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    if (mConfig->use_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }
    if (mConfig->dynamic_option()) {
        rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, mConfig->dynamic_option());
    }
    if (mConfig->thread_num() > 7) { // if thread_num > 7, cpu dynamic quant use Arm86 kernels
        rtg->setHint(MNN::Interpreter::CPU_SME2_INSTRUCTIONS, 0);
    } else {
        rtg->setHint(MNN::Interpreter::CPU_SME2_INSTRUCTIONS, 1);

    }
    rtg->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
    if (backend_type_convert(mConfig->backend_type()) != 0) { // not cpu
        std::string cacheFilePath = tmpPath.length() != 0 ? tmpPath : ".";
        rtg->setCache(cacheFilePath + "/mnn_cachefile.bin");
    }
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
    setRuntimeHint(mRuntimeManager);

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
    // get linear input thresholds and max values
    if (mConfig->config_.value("enable_debug", false)) {
        mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    }
}
    
static bool canSpecDecode(std::shared_ptr<Express::Module> module) {
    bool canSpec = false;
    auto info = module->getInfo();
    // check from mnn model 
    for (int i=0; i<info->inputNames.size(); ++i) {
        auto& varInfo = info->inputs[i];
        if(info->inputNames[i] == "logits_index") {
            if (varInfo.dim.size() > 0) {
                canSpec = true;
            }
        }
    }
    return canSpec;
}
void Llm::setSpeculativeConfig() {
    auto specultive_type = mConfig->speculative_type();
    if(!specultive_type.empty()) {
        if(!canSpecDecode(mModules[0])) {
            mInSpec = false;
            return;
        }
        mDraftLength = mConfig->draft_predict_length();
        mInSpec = true;
    }
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
        module_config.shapeMutable = true;
    }
    module_config.rearrange    = true;
    // using base module for lora module
    if (mBaseModule != nullptr) {
        module_config.base = mBaseModule;
    }
    // load single model
    mModules.resize(1);
    std::string model_path = mConfig->llm_model();
    
    std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
    std::vector<std::string> outputNames {"logits"};
    if (mConfig->has_talker()) {
        outputNames.emplace_back("talker_embeds");
    }
    bool needHiddenState = false;
    if (mConfig->config_.document.HasMember("hidden_states")) {
        needHiddenState = mConfig->config_.document["hidden_states"].GetBool();
    }
    if(mConfig->speculative_type() == "mtp") {
        needHiddenState = true;
    }
    if (needHiddenState) {
        outputNames.emplace_back("hidden_states");
    }
    mRuntimeManager->setExternalFile(mConfig->llm_weight());
    mModules[0].reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
    mRuntimeManager->setExternalFile("");
    if(nullptr == mModules[0]) {
        MNN_ERROR("[Error]: Load module failed, please check model.\n");
        if(outputNames.size() > 1) {
            MNN_ERROR("[Warning]: Set module multi outputs, please double check.\n");
        }
        return;
    }
    // set speculative decoding params
    setSpeculativeConfig();
    // create generation strategy
    mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);

    int decode_type_num = 1;
    int verify_length = 1;
    if(mInSpec) {
        // decode one token or mDraftLength token
        decode_type_num = 2;
        verify_length = mDraftLength + 1;
        // speculative decode module
        mModulePool[std::make_pair(verify_length, true)].reset(Module::clone(mModules[0].get()));
    }

    // autoregressive decode module
    mModulePool[std::make_pair(1, false)].reset(Module::clone(mModules[0].get()));
    // prefill module
    mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModules[0];
    
    // module input varp setting
    logitsLastIdx = _var<int>({-1}, {1});
    logitsAllIdx = _var<int>({0}, {1});
    // index match with seq_len
    mAttentionMaskVarVec.resize(decode_type_num);
    mPositionIdsVarVec.resize(decode_type_num);
    for(int i = 0; i < decode_type_num; i++) {
        int index = 1;
        if(i > 0) {
            index = verify_length;
        }
        // attentiion mask var
        {
            mAttentionMaskVarVec[i] = _Input({1, 1, index, index}, NCHW, halide_type_of<float>());
            auto ptr = mAttentionMaskVarVec[i]->writeMap<float>();
            for (int i = 0; i < index; i++) {
                for (int j = 0; j < index; j++) {
                    ptr[index * i + j] = (j > i) * std::numeric_limits<float>::lowest();
                }
            }
        }
        
        mPositionIdsVarVec[i] = _Input({index}, NCHW, halide_type_of<int>());
    }
    
    // MTP model load
    mGenerationStrategy->load(module_config);
}

Llm* Llm::create_lora(const std::string& lora_path) {
    auto llm = new Llm(std::make_shared<LlmConfig>(*mConfig));
    llm->set_config("{\"llm_model\": \"" + lora_path + "\", \"use_mmap\": false, \"use_cached_mmap\": false}");
    llm->mBaseModule = mModules.begin()->get();
    llm->load();
    return llm;
}

void Llm::tuning(TuneType type, std::vector<int> candidates) {
    if (type != OP_ENCODER_NUMBER) {
        MNN_ERROR("tuning type not supported\n");
        return;
    }
    // FIXME: Currently OpenCL Don't support KVMeta
    if (mConfig->backend_type() == "opencl") {
        return;
    }
    int decode_seq = 1;
    // Set to decode mode
    mContext->gen_seq_len = 1;
    if(mInSpec) {
        // start autoregressive decoding
        std::vector<int> input_ids = {0};
        auto logits = forwardVec(input_ids);
        int verify_length = mDraftLength + 1;
        decode_seq = verify_length;
    }
    int64_t min_time     = INT64_MAX;
    int prefer_candidate = 10;
    for (auto& candidate : candidates) {
        mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, candidate);
        Timer _t;
        std::vector<int> input_ids(decode_seq, 0);
        auto outputs = forwardVec(input_ids);
        if(outputs.empty()) {
            return;
        }
        auto logits = outputs[0];
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
    // do nothing, only reserve api
    return;
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

std::vector<Express::VARP> Llm::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos) {
    Express::VARP logitsIndex;
    bool inDecode = mContext->gen_seq_len > 0;
    
    bool isAllLogists = mConfig->all_logits() ? true : (inDecode ? mInSpec : false);
    int seqLenKey = inDecode ? hiddenState->getInfo()->dim[0] : mPrefillKey;
    isAllLogists = seqLenKey == 1 ? false : isAllLogists;
    auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
    if(mModulePool.find(moduleKey) == mModulePool.end()) {
        MNN_PRINT("Warning: module need new clone, cloning now.\n");
        mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
        mModulePool[moduleKey].reset(Module::clone(mModules[0].get()));
    }

    if (isAllLogists) {
        logitsIndex = logitsAllIdx;
    } else {
        logitsIndex = logitsLastIdx;
    }
    mGenerateParam->input_embeds = nullptr;
    mGenerateParam->outputs.clear();
    std::vector<Express::VARP> outputs;
    outputs = mModulePool[moduleKey]->onForward({hiddenState, mask, inputPos, logitsIndex});

    if (outputs.empty()) {
        return outputs;
    }
    mGenerateParam->input_embeds = hiddenState;
    mGenerateParam->outputs = outputs;

#if DEBUG_MODE == 3
    VARP logits = outputs[0];
    if(logits->getInfo()->dim[1] < 10 && logits->getInfo()->dim[1] >= 1) {
        for (int j = 0; j < logits->getInfo()->dim[1]; j++) {
            {
                int length = hiddenState->getInfo()->dim[2];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = hiddenState->readMap<float>()[index];
                    total += temp;
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                MNN_PRINT("\nhiddenState statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }
            
            {
                int length = mask->getInfo()->dim[3];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = mask->readMap<float>()[index];
                    total += (temp / length);
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                MNN_PRINT("mask statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }
            MNN_PRINT("position statistic value:%d\n", inputPos->readMap<int>()[j]);
            {
                int length = logits->getInfo()->dim[2];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = logits->readMap<float>()[index];
                    total += temp;
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                auto ptr = logits->readMap<float>() + j * logits->getInfo()->dim[2];
                //            MNN_PRINT("\noutput data value:%6f %6f %6f %6f %6f\n", ptr[0], ptr[length/5], ptr[length/10], ptr[length/20], ptr[length/100]);
                MNN_PRINT("output statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }
        }
    }
#endif
    mMeta->sync();
    return outputs;
}

VARP Llm::forward(const std::vector<int>& input_ids, bool is_prefill) {
    auto hidden_states = embedding(input_ids);
    return forward(hidden_states);
}
VARP Llm::forward(MNN::Express::VARP input_embeds) {
    int seq_len         = input_embeds->getInfo()->dim[mSeqLenIndex];
    auto out = forwardVec(input_embeds);
    if (out.empty()) {
        return nullptr;
    }
    auto logits = out[0];
    updateContext(seq_len, 1);
    return logits;
}

std::vector<VARP> Llm::forwardVec(const std::vector<int>& input_ids) {
    auto input_embeds = embedding(input_ids);
    auto outputs = forwardVec(input_embeds);
    return outputs;
}

std::vector<VARP> Llm::forwardVec(MNN::Express::VARP input_embeds) {
    int seq_len         = input_embeds->getInfo()->dim[mSeqLenIndex];
    mMeta->add         = seq_len;
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    auto logits = forwardRaw(input_embeds, attention_mask, position_ids);
    return logits;
}

void Llm::updateContext(int seq_len, int gen_len) {
    mContext->all_seq_len += seq_len;
    mContext->gen_seq_len += gen_len;
}

int Llm::sample(VARP logits, int offset, int size) {
    auto logitsShape = logits->getInfo()->dim;
    if (offset && size) {
        logits = _Const(logits->readMap<float>() + offset, {size}, NHWC, halide_type_of<float>());
    }
    auto token_id = mSampler->sample(logits);
    return token_id;
}

void Llm::reset() {
    mContext->output_tokens.clear();
    mContext->history_tokens.clear();
    mContext->all_seq_len = 0;
    mMeta->remove = mMeta->previous;
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
    mContext->prefill_us  = 0;
    mContext->decode_us   = 0;
    mContext->current_token = 0;
    if (!mConfig->reuse_kv()) {
        mContext->all_seq_len = 0;
        mContext->history_tokens.clear();
        mMeta->remove = mMeta->previous;
    }
    mContext->output_tokens.clear();
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
    if (is_stop(mContext->current_token)) {
        return;
    }
    mGenerateParam->max_new_tokens = max_token;
    mGenerationStrategy->generate(*mGenerateParam);
}

std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_tokens) {
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }
    mContext->prompt_len = static_cast<int>(input_ids.size());
    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    auto hidden_states = embedding(input_ids);
    return generate(hidden_states, max_tokens);
}

std::vector<int> Llm::tokenizer_encode(const std::string& user_content) {
    return mTokenizer->encode(user_content);
}

std::vector<int> Llm::generate(MNN::Express::VARP input_embeds, int max_tokens) {
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }
    int seqLen = input_embeds->getInfo()->dim[mSeqLenIndex];
    mContext->prompt_len = seqLen;
    Timer _t;
    auto outputs      = forwardVec(input_embeds);
    if(outputs.size() < 1) {
        return {};
    }
    auto logits = outputs[0];
    updateContext(seqLen, 0);

    if (nullptr == logits.get()) {
        return {};
    }
    // logits compute sync for correct timer
    logits->readMap<void>();
    mContext->prefill_us = _t.durationInUs();
    
#if DEBUG_MODE == 3
    {
        std::ofstream outFile("input_embeds.txt");
        auto temp = input_embeds->readMap<float>();
        for (size_t i = 0; i < input_embeds->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("logits.txt");
        auto temp = outputs[0]->readMap<float>();
        for (size_t i = 0; i < outputs[0]->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
#endif
    
    _t.reset();
    mContext->current_token = sample(logits);
    mContext->sample_us += _t.durationInUs();
    logits = nullptr;

    // call generation function
    mGenerateParam->max_new_tokens = max_tokens;
    mGenerationStrategy->generate(*mGenerateParam);
    return mContext->output_tokens;
}

void Llm::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    generate(input_ids, max_new_tokens);
}

void Llm::response(MNN::Express::VARP input_embeds, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    generate(input_embeds, max_new_tokens);
}

void Llm::response(const std::string& user_content, std::ostream* os, const char* end_with, int max_new_tokens) {
    auto prompt = user_content;
    if (mConfig->use_template()) {
        prompt = mPrompt->applyTemplate(user_content, true);
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
    mGenerateParam.reset(new GenerationParams);
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
    mGenerateParam.reset();
    mModules.clear();
    mRuntimeManager.reset();
    mProcessorRuntimeManager.reset();
}
int Llm::getOutputIndex(const std::string& name) const {
    if (mModulePool.empty()) {
        return -1;
    }
    auto info = mModulePool.begin()->second->getInfo();
    for (int i=0; i<info->outputNames.size(); ++i) {
        if (info->outputNames[i] == name) {
            return i;
        }
    }
    return -1;
}
std::vector<Express::VARP> Llm::getOutputs() const {
    return mGenerateParam->outputs;
}

bool Llm::reuse_kv() { return mConfig->reuse_kv(); }

static inline bool needNewVar(VARP var, int axis, int seq_len, int kv_seq_len = 0) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    if (kv_seq_len != 0 && var->getInfo()->dim[axis + 1] != kv_seq_len) {
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
    if (mConfig->attention_mask() == "float") {
        // Use square mask
        kv_seq_len = seq_len;
        if (mAttentionMaskVarVec.size() > 0) {
            if(seq_len == 1) {
                return mAttentionMaskVarVec[0];
            }
            if (mAttentionMaskVarVec.size() > 1 && seq_len == mDraftLength) {
                return mAttentionMaskVarVec[1];
            }
        }
        
        attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
        auto ptr = attentionMask->writeMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                ptr[kv_seq_len * i + j] = (j > i) * std::numeric_limits<float>::lowest();
            }
        }
        return attentionMask;
    } else {
        if (needNewVar(attentionMask, 2, seq_len, kv_seq_len)) {
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
        if (seq_len == 1) {
            auto ptr = mPositionIdsVarVec[0]->writeMap<int>();
            ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
            return mPositionIdsVarVec[0];
        }
        if(mPositionIdsVarVec.size() > 1 && seq_len == mDraftLength) {
            auto ptr = mPositionIdsVarVec[1]->writeMap<int>();
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mContext->all_seq_len;
            }
            return mPositionIdsVarVec[1];
        }
        
        positionIds = _Input({seq_len}, NCHW, halide_type_of<int>());
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
} // namespace Transformer
} // namespace MNN
