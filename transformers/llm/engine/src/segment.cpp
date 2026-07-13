#ifdef MNN_LLM_SUPPORT_SEGMENT

#include "segment.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

#include <MNN/AutoTime.hpp>
#include "core/MNNFileUtils.h"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "tokenizer/tokenizer.hpp"

namespace MNN {
namespace Transformer {
namespace {

using namespace Express;
using RuntimeManager = Express::Executor::RuntimeManager;

static bool segmentCheckFile(const std::string& path, const char* name) {
    if (!MNNFileExist(path.c_str())) {
        MNN_ERROR("[Error]: segment %s not found: %s\n", name, path.c_str());
        return false;
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        MNN_ERROR("[Error]: failed to open segment %s: %s\n", name, path.c_str());
        return false;
    }
    return true;
}

static std::string segmentPath(const LlmConfig& config, const std::string& name) {
    return config.base_dir_ + name;
}

static MNNForwardType segmentForwardType(std::shared_ptr<LlmConfig> config) {
    if (config->config_.contains("forwardtype")) {
        return static_cast<MNNForwardType>(config->config_.value("forwardtype", 0));
    }
    const auto type = config->backend_type();
    if (type == "metal")
        return MNN_FORWARD_METAL;
    if (type == "cuda")
        return MNN_FORWARD_CUDA;
    if (type == "opencl")
        return MNN_FORWARD_OPENCL;
    if (type == "opengl")
        return MNN_FORWARD_OPENGL;
    if (type == "vulkan")
        return MNN_FORWARD_VULKAN;
    if (type == "npu")
        return MNN_FORWARD_NN;
    return MNN_FORWARD_CPU;
}

static void segmentApplyBackendConfig(std::shared_ptr<LlmConfig> config, BackendConfig* backend) {
    if (backend == nullptr) {
        return;
    }
    if (config->config_.contains("precision")) {
        backend->precision = static_cast<BackendConfig::PrecisionMode>(config->config_.value("precision", 2));
    } else if (config->precision() == "high") {
        backend->precision = BackendConfig::Precision_High;
    } else if (config->precision() == "low") {
        backend->precision = BackendConfig::Precision_Low;
    }
    if (config->config_.contains("memory")) {
        backend->memory = static_cast<BackendConfig::MemoryMode>(config->config_.value("memory", 2));
    } else if (config->memory() == "high") {
        backend->memory = BackendConfig::Memory_High;
    } else if (config->memory() == "low") {
        backend->memory = BackendConfig::Memory_Low;
    }
}

static VARP segmentTakeLastHidden(VARP hidden) {
    if (hidden == nullptr) {
        return nullptr;
    }
    auto info = hidden->getInfo();
    if (info == nullptr || info->dim.size() < 3) {
        return hidden;
    }
    const int seqLen = info->dim[1];
    const int hiddenSize = info->dim[2];
    if (seqLen <= 0 || hiddenSize <= 0 || seqLen == 1) {
        return hidden;
    }
    const size_t bytes = static_cast<size_t>(hiddenSize) * info->type.bytes();
    const uint8_t* src = hidden->readMap<uint8_t>();
    if (src == nullptr || bytes == 0) {
        return hidden;
    }
    auto out = _Input({1, 1, hiddenSize}, info->order, info->type);
    ::memcpy(out->writeMap<uint8_t>(), src + static_cast<size_t>(seqLen - 1) * bytes, bytes);
    out.fix(VARP::CONSTANT);
    return out;
}

static void segmentWait(VARP var) {
    if (var == nullptr || var->getTensor() == nullptr) {
        return;
    }
    ((MNN::Tensor*)var->getTensor())->wait(MNN::Tensor::MAP_TENSOR_READ, true);
}

} // namespace

class SegmentLlm final : public Llm {
public:
    explicit SegmentLlm(std::shared_ptr<LlmConfig> config) : Llm(config) {
        mSeqLenIndex = 1;
        mMeta->layer_nums = mConfig->config_.value("layer_nums", 0);
    }

    bool load() override;
    VARP embedding(const std::vector<int>& input_ids) override;
    VARP gen_attention_mask(int seq_len) override;
    VARP gen_position_ids(int seq_len) override;
    std::vector<VARP> forwardRaw(VARP hiddenState, VARP mask, VARP inputPos, VARPS extraArgs = {}) override;
    int sample(VARP logits, int offset = 0, int size = 0) override;
    void response(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr,
                  int max_new_tokens = -1) override;
    void generate(int max_token) override;

private:
    bool loadTokenizer();
    bool loadModules();
    bool prefill(const std::vector<int>& input_ids);
    int sampleFromHidden(VARP hidden);
    VARP embeddingToken(int token);
    VARP decodeAttentionMask();
    VARP decodePositionId();
    VARP decoderForward(VARP input, VARP mask = nullptr, VARP positionIds = nullptr);
    void updateSegmentContext(int seqLen, int genLen);

private:
    std::shared_ptr<Module> mEmbedModule;
    std::shared_ptr<Module> mDecoderModule;
    std::shared_ptr<Module> mDecoderPrefillModule;
    std::shared_ptr<Module> mLogitBaseModule;
    std::shared_ptr<Module> mLogitModule;
    VARP mLastHidden;
    VARP mTokenInput;
    VARP mDecodeMaskInput;
    VARP mDecodePositionInput;
    int mMaxDecodeTokens = 1024;
};

bool SegmentLlm::loadTokenizer() {
    std::string tokenizerPath = mConfig->tokenizer_file();
    if (!segmentCheckFile(tokenizerPath, "tokenizer file")) {
        return false;
    }
    mTokenizer.reset(Tokenizer::createTokenizer(tokenizerPath));
    if (mTokenizer == nullptr) {
        MNN_ERROR("[Error]: failed to load segment tokenizer: %s\n", tokenizerPath.c_str());
        return false;
    }

    if (mConfig->config_.contains("jinja")) {
        setChatTemplate();
        return true;
    }

    std::ifstream tokenConfig(segmentPath(*mConfig, "token_config.json"));
    if (tokenConfig.is_open()) {
        std::ostringstream ostr;
        ostr << tokenConfig.rdbuf();
        auto json = ujson::json::parse(ostr.str());
        if (json.contains("chat_template")) {
            mTokenizer->set_chat_template(json["chat_template"].get<std::string>(), json.value("eos_token", ""));
        }
    }
    return true;
}

bool SegmentLlm::loadModules() {
    const std::string embedPath = segmentPath(*mConfig, "embed.mnn");
    const std::string decoderPath = segmentPath(*mConfig, "decoder.mnn");
    const std::string decoderWeightPath = decoderPath + ".weight";
    const std::string logitPath = segmentPath(*mConfig, "logit.mnn");
    const std::string logitWeightPath = logitPath + ".weight";
    const std::string logitTopkPath = segmentPath(*mConfig, "logit_topkv_1.mnn");

    if (!segmentCheckFile(embedPath, "embed model") || !segmentCheckFile(decoderPath, "decoder model") ||
        !segmentCheckFile(decoderWeightPath, "decoder weight") || !segmentCheckFile(logitPath, "logit model") ||
        !segmentCheckFile(logitWeightPath, "logit weight") || !segmentCheckFile(logitTopkPath, "logit topk model")) {
        return false;
    }

    BackendConfig backendConfig;
    segmentApplyBackendConfig(mConfig, &backendConfig);

    ScheduleConfig decoderSchedule;
    decoderSchedule.backendConfig = &backendConfig;
    decoderSchedule.type = segmentForwardType(mConfig);
    decoderSchedule.numThread = mConfig->config_.value("thread_num", 1);
    if (decoderSchedule.type == MNN_FORWARD_OPENCL) {
        decoderSchedule.numThread |= 64;
    }

    mRuntimeManager.reset(RuntimeManager::createRuntimeManager(decoderSchedule), RuntimeManager::destroy);
    mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
    Module::Config decoderConfig;
    decoderConfig.rearrange = true;
    if (decoderSchedule.type == MNN_FORWARD_OPENCL || decoderSchedule.type == MNN_FORWARD_VULKAN) {
        decoderConfig.shapeMutable = false;
    }

    const std::vector<std::string> decoderInputs = {"input_embedding", "mask", "position_ids"};
    mDecoderModule.reset(
        Module::load(decoderInputs, {"last_hidden_state"}, decoderPath.c_str(), mRuntimeManager, &decoderConfig));
    if (!mDecoderModule) {
        mDecoderModule.reset(
            Module::load(decoderInputs, {"hidden_state"}, decoderPath.c_str(), mRuntimeManager, &decoderConfig));
    }
    if (!mDecoderModule) {
        MNN_ERROR("[Error]: load segment decoder.mnn failed\n");
        return false;
    }
    mDecoderPrefillModule.reset(Module::clone(mDecoderModule.get()));
    if (!mDecoderPrefillModule) {
        MNN_ERROR("[Error]: clone segment decoder prefill module failed\n");
        return false;
    }

    BackendConfig otherBackendConfig = backendConfig;
    if (mConfig->config_.contains("otherPrecision")) {
        otherBackendConfig.precision = static_cast<BackendConfig::PrecisionMode>(
            mConfig->config_.value("otherPrecision", (int)otherBackendConfig.precision));
    }
    ScheduleConfig otherSchedule = decoderSchedule;
    otherSchedule.backendConfig = &otherBackendConfig;
    mProcessorRuntimeManager.reset(RuntimeManager::createRuntimeManager(otherSchedule), RuntimeManager::destroy);
    mProcessorRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, nullptr);

    Module::Config moduleConfig;
    moduleConfig.rearrange = true;
    mLogitBaseModule.reset(Module::load({}, {}, logitPath.c_str(), mProcessorRuntimeManager, &moduleConfig));
    if (!mLogitBaseModule) {
        MNN_ERROR("[Error]: load segment logit.mnn failed\n");
        return false;
    }

    Module::Config depConfig = moduleConfig;
    depConfig.base = mLogitBaseModule.get();
    mLogitModule.reset(Module::load({}, {}, logitTopkPath.c_str(), mProcessorRuntimeManager, &depConfig));
    mEmbedModule.reset(Module::load({}, {}, embedPath.c_str(), mProcessorRuntimeManager, &depConfig));
    if (!mLogitModule || !mEmbedModule) {
        MNN_ERROR("[Error]: load segment logit_topkv_1.mnn/embed.mnn failed\n");
        return false;
    }
    return true;
}

bool SegmentLlm::load() {
    MNN::Express::ExecutorScope s(mExecutor);
    Timer _t;
    mMaxDecodeTokens = mConfig->config_.value("max_decode_tokens", mConfig->max_new_tokens());
    if (!loadTokenizer() || !loadModules()) {
        return false;
    }
    mContext->load_us += _t.durationInUs();
    mContext->status = LlmStatus::RUNNING;
    return true;
}

VARP SegmentLlm::embedding(const std::vector<int>& input_ids) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (input_ids.empty() || !mEmbedModule) {
        return nullptr;
    }
    auto var = _Input({1, static_cast<int>(input_ids.size())}, NCHW, halide_type_of<int>());
    ::memcpy(var->writeMap<int>(), input_ids.data(), input_ids.size() * sizeof(int));
    auto outputs = mEmbedModule->onForward({var});
    return outputs.empty() ? nullptr : outputs[0];
}

VARP SegmentLlm::embeddingToken(int token) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!mEmbedModule) {
        return nullptr;
    }
    if (mTokenInput == nullptr) {
        mTokenInput = _Input({1, 1}, NCHW, halide_type_of<int>());
    }
    *mTokenInput->writeMap<int>() = token;
    auto outputs = mEmbedModule->onForward({mTokenInput});
    return outputs.empty() ? nullptr : outputs[0];
}

VARP SegmentLlm::gen_attention_mask(int seq_len) {
    auto mask = _Input({}, NCHW, halide_type_of<float>());
    *mask->writeMap<float>() = 0.0f;
    mask.fix(VARP::CONSTANT);
    return mask;
}

VARP SegmentLlm::gen_position_ids(int seq_len) {
    auto positionIds = _Input({1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = positionIds->writeMap<int>();
    const int start = static_cast<int>(mMeta->previous) - static_cast<int>(mMeta->remove);
    for (int i = 0; i < seq_len; ++i) {
        ptr[i] = start + i;
    }
    positionIds.fix(VARP::CONSTANT);
    return positionIds;
}

VARP SegmentLlm::decodeAttentionMask() {
    if (mDecodeMaskInput == nullptr) {
        mDecodeMaskInput = _Input({}, NCHW, halide_type_of<float>());
        *mDecodeMaskInput->writeMap<float>() = 0.0f;
        mDecodeMaskInput.fix(VARP::CONSTANT);
    }
    return mDecodeMaskInput;
}

VARP SegmentLlm::decodePositionId() {
    if (mDecodePositionInput == nullptr) {
        mDecodePositionInput = _Input({1, 1}, NCHW, halide_type_of<int>());
    }
    const int start = static_cast<int>(mMeta->previous) - static_cast<int>(mMeta->remove);
    *mDecodePositionInput->writeMap<int>() = start;
    return mDecodePositionInput;
}

VARP SegmentLlm::decoderForward(VARP input, VARP mask, VARP positionIds) {
    if (input == nullptr) {
        return nullptr;
    }
    auto info = input->getInfo();
    if (info == nullptr || info->dim.size() < 3) {
        return nullptr;
    }
    const int seqLen = info->dim[1];
    if (mask == nullptr) {
        mask = (seqLen == 1) ? decodeAttentionMask() : gen_attention_mask(seqLen);
    }
    if (positionIds == nullptr) {
        positionIds = (seqLen == 1) ? decodePositionId() : gen_position_ids(seqLen);
    }
    auto module = (seqLen == 1) ? mDecoderModule : mDecoderPrefillModule;
    mMeta->add = seqLen;
    auto outputs = module->onForward({input, mask, positionIds});
    mMeta->sync();
    if (outputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    segmentWait(outputs[0]);
    return outputs[0];
}

std::vector<VARP> SegmentLlm::forwardRaw(VARP hiddenState, VARP mask, VARP inputPos, VARPS extraArgs) {
    auto hidden = decoderForward(hiddenState, mask, inputPos);
    if (hidden == nullptr || !mLogitModule) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    mLastHidden = segmentTakeLastHidden(hidden);
    auto outputs = mLogitModule->onForward({hidden});
    if (outputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    return outputs;
}

int SegmentLlm::sample(VARP logits, int offset, int size) {
    if (logits == nullptr) {
        return -1;
    }
    auto info = logits->getInfo();
    if (info == nullptr || info->size <= 0) {
        return -1;
    }
    const int* topk = logits->readMap<int>();
    return topk == nullptr ? -1 : topk[info->size - 1];
}

int SegmentLlm::sampleFromHidden(VARP hidden) {
    if (hidden == nullptr || !mLogitModule) {
        return -1;
    }
    auto outputs = mLogitModule->onForward({hidden});
    if (outputs.empty()) {
        return -1;
    }
    return sample(outputs[0]);
}

void SegmentLlm::updateSegmentContext(int seqLen, int genLen) {
    mContext->all_seq_len += seqLen;
    mContext->gen_seq_len += genLen;
}

bool SegmentLlm::prefill(const std::vector<int>& input_ids) {
    if (input_ids.empty()) {
        return false;
    }
    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end());
    Timer _t;
    auto emb = embedding(input_ids);
    auto hidden = decoderForward(emb);
    if (hidden == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    mLastHidden = segmentTakeLastHidden(hidden);
    if (mLastHidden.get() != nullptr) {
        mLastHidden.fix(VARP::CONSTANT);
    }
    updateSegmentContext(static_cast<int>(input_ids.size()), 0);
    mContext->prompt_len = static_cast<int>(input_ids.size());
    mContext->prefill_us += _t.durationInUs();
    return true;
}

void SegmentLlm::generate(int max_token) {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (max_token < 0) {
        max_token = mMaxDecodeTokens;
    }
    max_token = std::min(max_token, mMaxDecodeTokens);
    int len = 0;
    while (len < max_token) {
        if (mContext->status == LlmStatus::USER_CANCEL || mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }
        Timer _t;
        int token = sampleFromHidden(mLastHidden);
        if (token < 0) {
            mContext->decode_us += _t.durationInUs();
            mContext->status = LlmStatus::INTERNAL_ERROR;
            break;
        }
        mContext->current_token = token;
        if (is_stop(token)) {
            mContext->decode_us += _t.durationInUs();
            if (mContext->os != nullptr) {
                *mContext->os << mContext->end_with << std::flush;
            }
            break;
        }

        mContext->history_tokens.push_back(token);
        mContext->output_tokens.push_back(token);
        auto decodeStr = tokenizer_decode(token);
        mContext->generate_str += decodeStr;
        if (mContext->os != nullptr) {
            *mContext->os << decodeStr << std::flush;
        }

        auto emb = embeddingToken(token);
        auto hidden = decoderForward(emb);
        if (hidden == nullptr) {
            mContext->decode_us += _t.durationInUs();
            mContext->status = LlmStatus::INTERNAL_ERROR;
            break;
        }
        mLastHidden = segmentTakeLastHidden(hidden);
        if (mLastHidden.get() != nullptr) {
            mLastHidden.fix(VARP::CONSTANT);
        }
        updateSegmentContext(1, 1);
        mContext->decode_us += _t.durationInUs();
        ++len;
    }
    if (len >= max_token) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
}

void SegmentLlm::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with,
                          int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!end_with) {
        end_with = "\n";
    }
    generate_init(os, end_with);
    if (!prefill(input_ids)) {
        return;
    }
    if (max_new_tokens < 0) {
        max_new_tokens = mMaxDecodeTokens;
    }
    if (max_new_tokens > 0) {
        generate(max_new_tokens);
    }
}

Llm* createSegmentLlm(std::shared_ptr<LlmConfig> config) {
    return new SegmentLlm(std::move(config));
}

} // namespace Transformer
} // namespace MNN

#endif // MNN_LLM_SUPPORT_SEGMENT
