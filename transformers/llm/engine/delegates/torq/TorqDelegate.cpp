#include "../LlmDelegate.hpp"
#include "llmconfig.hpp"
#include "kvmeta.hpp"
#include "sampler.hpp"
#include "speculative_decoding/generate.hpp"
#include "tokenizer/tokenizer.hpp"
#include <tllm/exec.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <sstream>

namespace MNN {
namespace Transformer {
using namespace Express;

static ujson::json readManifest(const std::string& file, std::initializer_list<const char*> requiredKeys) {
    std::ifstream stream(file);
    if (!stream) {
        MNN_ERROR("TORQ: cannot open manifest %s\n", file.c_str());
        return {};
    }
    std::ostringstream text;
    text << stream.rdbuf();
    auto manifest = ujson::json::parse(text.str());
    if (!manifest.is_object()) {
        MNN_ERROR("TORQ: expected a JSON object in %s\n", file.c_str());
        return {};
    }
    for (auto key : requiredKeys) {
        if (!manifest.contains(key)) {
            MNN_ERROR("TORQ: missing field '%s' in %s\n", key, file.c_str());
            return {};
        }
    }
    return manifest;
}

class TorqGeneration final : public ArGeneration {
public:
    TorqGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, int capacity)
        : ArGeneration(llm, context, config), mCapacity(capacity) {}

    void generate(GenerationParams& param) override {
        // Each sampled token consumes one graph position, including the last decode step.
        const int remaining = mCapacity - mContext->all_seq_len;
        if (param.max_new_tokens > remaining) {
            MNN_PRINT("TORQ: limiting max_new_tokens from %d to %d (max_seq_len=%d)\n",
                      param.max_new_tokens, remaining, mCapacity);
            param.max_new_tokens = remaining;
        }
        ArGeneration::generate(param);
    }

private:
    const int mCapacity;
};

// TORQ embeds the token IDs inside its compiled graph. The MNN generation
// loop still owns sampling, stopping, streaming and the response context.
class TorqDelegate final : public LlmDelegate {
public:
    using LlmDelegate::LlmDelegate;
    ~TorqDelegate() override { close(); }

    bool load() override {
        Timer timer;
        close();
        mContext->status = LlmStatus::NOT_LOADED;
        if (mConfig->backend_type() != "cpu") {
            MNN_ERROR("TORQ models require a CPU host executor\n");
            return false;
        }
        if (mConfig->reuse_kv() || mConfig->prompt_cache() ||
            mConfig->is_visual() || mConfig->is_audio() || mConfig->has_talker() || mConfig->all_logits() ||
            !mConfig->speculative_type().empty() || mBlockSize != 0) {
            MNN_ERROR("TORQ supports fresh text prompts and last-token logits only\n");
            return false;
        }
        auto dir = mConfig->npu_model_dir();
        auto meta = readManifest(dir + "/torq.llm.json", {"config", "forge_option"});
        auto bm = readManifest(dir + "/torq.bm.json", {"extension_files"});
        if (meta.is_null() || bm.is_null()) {
            return false;
        }
        auto cfg = meta["config"];
        auto opt = meta["forge_option"];
        auto files = bm["extension_files"];
        mVocab = cfg.value("vocab_size", 0);
        mCapacity = opt.value("max_seq_len", 0);
        if (cfg.value("architectures", "") != "qwen3_5" || opt.value("base_dtype", -1) != 12 ||
            !files.contains("graph") || !files["graph"].contains("linear_output.bin") ||
            mVocab <= 0 || mCapacity <= 0) {
            MNN_ERROR("Expected Qwen3.5 BF16 logits and a separate linear_output graph\n");
            return false;
        }
        mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
        if (!mTokenizer) {
            return false;
        }
        setChatTemplate();
        mSampler.reset(Sampler::createSampler(mContext, mConfig));
        mGenerationStrategy = std::make_shared<TorqGeneration>(this, mContext, mConfig, mCapacity);
        mTorq = static_cast<torq_llm_ctx*>(torq_llm_init_ctx(&dir[0]));
        if (!mTorq) {
            MNN_ERROR("TORQ: torq_llm_init_ctx failed for model directory %s\n", dir.c_str());
            return false;
        }
        auto graph = mTorq->linear_output_ctx;
        auto runtime = graph ? graph->runtime_ctx : nullptr;
        if (!runtime || runtime->n_output < 1 || !runtime->out_attr) {
            MNN_ERROR("TORQ: missing linear_output tensor descriptor\n");
            close();
            return false;
        }
        const auto& out = runtime->out_attr[0];
        const auto width = out.n_dims > 0 && out.n_dims <= TORQ_MAX_DIMS ? out.dims[out.n_dims - 1] : 0;
        const auto vocab = static_cast<uint32_t>(mVocab);
        if (out.type != TORQ_TENSOR_BFLOAT16 || width != vocab || out.n_elems != vocab ||
            out.size / sizeof(uint16_t) < vocab) {
            MNN_ERROR("TORQ: expected one BF16 logits row of width %d, got type=%d width=%u elements=%u bytes=%u\n",
                      mVocab, out.type, width, out.n_elems, out.size);
            close();
            return false;
        }
        mContext->load_us = timer.durationInUs();
        mContext->status = LlmStatus::RUNNING;
        MNN_PRINT("MNN-LLM TORQ: Qwen3.5 text, vocab=%d, context=%d\n", mVocab, mCapacity);
        return true;
    }

    VARP embedding(const std::vector<int>& ids) override {
        return _Const(ids.data(), {static_cast<int>(ids.size())}, NCHW, halide_type_of<int>());
    }
    VARP gen_attention_mask(int) override { return nullptr; }
    VARP gen_position_ids(int) override { return nullptr; }

    VARPS forwardRaw(VARP ids, VARP, VARP, VARPS) override {
        mGenerateParam->outputs.clear();
        const auto info = ids.get() ? ids->getInfo() : nullptr;
        if (!info || info->type != halide_type_of<int>() || info->dim.size() != 1 ||
            mConfig->reuse_kv() || mConfig->prompt_cache() || mBlockSize || mPrefixCacheMode) {
            return fail();
        }
        const int n = info->size;
        const int pos = mContext->all_seq_len;
        const int* tokens = ids->readMap<int>();
        if (!mTorq || !tokens || n < 1 || pos < 0 || n > mCapacity || pos > mCapacity - n) {
            return fail();
        }
        for (int i = 0; i < n; ++i) {
            if (tokens[i] < 0 || tokens[i] >= mVocab) {
                return fail();
            }
        }

        const uint16_t* raw;
        if (mContext->gen_seq_len == 0) {
            if (pos != 0) {
                return fail();
            }
            if (mInput) {
                torq_llm_release_input(mInput);
            }
            mInput = static_cast<tllm_input*>(torq_llm_create_input());
            if (!mInput) {
                return fail();
            }
            // SDK 2.1.5 probes image slot zero even when the logical count is zero.
            int zero = 0;
            torq_llm_set_images(mInput, 1, &zero, &zero, &zero, &zero);
            if (!mInput->mm_input || !mInput->mm_input->images || mInput->mm_input->images[0].img_embedding) {
                return fail();
            }
            mInput->mm_input->n_images = 0;
            mPrompt.assign(tokens, tokens + n);
            torq_llm_set_input_tokens(mInput, mPrompt.data(), n);
            torq_llm_set_input_position(mInput, 0);
            raw = static_cast<const uint16_t*>(torq_llm_vlm_exec_prefill(mTorq, mInput));
            mNeedSync = true;
            mNext = n;
        } else {
            if (n != 1 || !mInput || pos != mNext) {
                return fail();
            }
            if (mNeedSync) {
                torq_llm_sync_kvcache(mTorq);
                torq_llm_sync_linear_attn_state(mTorq);
                mNeedSync = false;
            }
            torq_llm_iterate_input_token(mInput, tokens[0]);
            torq_llm_set_input_position(mInput, pos);
            raw = static_cast<const uint16_t*>(torq_llm_vlm_exec_decode(mTorq, mInput));
            ++mNext;
        }
        if (!raw) {
            return fail();
        }
        auto logits = _Input({1, 1, mVocab}, NCHW, halide_type_of<float>());
        float* dst = logits->writeMap<float>();
        if (!dst) {
            return fail();
        }
        for (int i = 0; i < mVocab; ++i) {
            uint32_t bits = uint32_t(raw[i]) << 16;
            std::memcpy(dst + i, &bits, sizeof(bits));
        }
        mGenerateParam->outputs = {logits};
        mMeta->sync();
        return mGenerateParam->outputs;
    }

private:
    VARPS fail() {
        MNN_ERROR("TORQ forward failed: position=%d, limit=%d\n", mContext->all_seq_len, mCapacity);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    void close() {
        if (mInput) {
            torq_llm_release_input(mInput);
        }
        if (mTorq) {
            torq_llm_release_ctx(mTorq);
        }
        mInput = nullptr;
        mTorq = nullptr;
    }
    torq_llm_ctx* mTorq = nullptr;
    tllm_input* mInput = nullptr;
    std::vector<int> mPrompt;
    int mCapacity = 0, mVocab = 0, mNext = 0;
    bool mNeedSync = false;
};

Llm* createTorqDelegate(const std::shared_ptr<LlmConfig>& config) {
    if (config->resources_) {
        MNN_ERROR("TORQ requires a filesystem model directory\n");
        return nullptr;
    }
    return new TorqDelegate(config);
}
} // namespace Transformer
} // namespace MNN
