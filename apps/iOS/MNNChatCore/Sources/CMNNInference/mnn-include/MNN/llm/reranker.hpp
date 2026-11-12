//
//  reranker.hpp
//
//  Created by MNN on 2025/07/29.
//  ZhaodeWang
//

#ifndef MNN_LLM_RERANKER_hpp
#define MNN_LLM_RERANKER_hpp

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include "llm.hpp"

namespace MNN {
namespace Transformer {

using namespace MNN::Express;

/**
 * @brief Abstract base class for Rerankers.
 * This class defines the interface for any reranker implementation.
 */
class RerankerBase {
public:
    /**
     * @brief Virtual destructor to ensure proper cleanup of derived classes.
     */
    virtual ~RerankerBase() = default;

    /**
     * @brief Initializes the reranker with a configuration path.
     * @param config_path The path to the configuration file.
     */
    virtual void initialize(const std::string& config_path) = 0;

    /**
     * @brief Loads the reranker model after initialization.
     */
    virtual void load() = 0;

    /**
     * @brief Sets the instruction for the reranker.
     * @param instruct The instruction string.
     */
    virtual void setInstruct(const std::string& instruct) = 0;

    /**
     * @brief Computes scores for a list of documents based on a query and instruct.
     * @param query The input query.
     * @param documents A vector of document strings.
     * @return A vector of float scores, one for each document.
     */
    virtual std::vector<float> compute_scores(const std::string& query, const std::vector<std::string>& documents) = 0;

    /**
     * @brief Gets the underlying LLM instance for configuration purposes.
     * @return A pointer to the LLM instance, or nullptr if not available.
     */
    Llm* get_llm() {
        return mLlm.get();
    }

protected:
    std::unique_ptr<Llm> mLlm;
};


class Qwen3Reranker : public RerankerBase {
public:
    /**
     * @brief Constructor for Qwen3Reranker.
     * @param config_path The path to the LLM configuration.
     */
    Qwen3Reranker(const std::string& config_path) {
        initialize(config_path);
        }

    /**
     * @brief Initializes the LLM.
     * @param config_path The path to the LLM configuration.
     */
    void initialize(const std::string& config_path) override {
        mLlm.reset(Llm::createLLM(config_path));
        mLlm->set_config("{\"all_logits\":true}");
    }

    /**
     * @brief Loads the LLM and initializes token IDs.
     */
    void load() override {
        mLlm->load();
        mTokenTrueId = mLlm->tokenizer_encode("yes")[0];
        mTokenFalseId = mLlm->tokenizer_encode("no")[0];
    }

    /**
     * @brief Sets the instruction for the reranker.
     * @param instruct The instruction string.
     */
    void setInstruct(const std::string& instruct) override {
        mInstruct = instruct;
    }

    /**
     * @brief Computes scores for a list of documents using the Qwen3 model.
     * @param query The input query.
     * @param documents A vector of document strings.
     * @return A vector of float scores, one for each document.
     */
    std::vector<float> compute_scores(const std::string& query, const std::vector<std::string>& documents) override {
        CHECK_LLM_RUNNING_RET(mLlm->getContext(), std::vector<float>());
        std::string prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
        prefix = prefix + "<Instruct>: " + mInstruct + "\n<Query>: " + query + "\n<Document>: ";
        auto suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        auto input_ids = mLlm->tokenizer_encode(prefix);
        auto suffix_ids = mLlm->tokenizer_encode(suffix);
        std::vector<int> position_ids_;
        std::vector<std::pair<int, int>> spans;
        int prefix_len = input_ids.size();
        int suffix_len = suffix_ids.size();
        int current_len = prefix_len;
        for (int i = 0; i < current_len; i++) {
            position_ids_.push_back(i);
        }
        for (const auto& doc : documents) {
            auto doc_ids = mLlm->tokenizer_encode(doc);
            input_ids.insert(input_ids.end(), doc_ids.begin(), doc_ids.end());
            input_ids.insert(input_ids.end(), suffix_ids.begin(), suffix_ids.end());
            int doc_len = doc_ids.size() + suffix_len;
            int start_idx = current_len;
            int end_idx = current_len + doc_len;
            spans.emplace_back(start_idx, end_idx);
            current_len = end_idx;
            for (int i = prefix_len; i < prefix_len + doc_len; i++) {
                position_ids_.push_back(i);
            }
        }
        int total_len = input_ids.size();
        // printf("Total input length: %d\n", total_len);
        // input_embeds
        auto input_embeds = mLlm->embedding(input_ids);
        // Causal; beyond the shared prompt prefix every document sees only itself.
        std::vector<int> docBounds;
        docBounds.reserve(spans.size() + 1);
        for (const auto& span : spans) {
            if (docBounds.empty() || docBounds.back() != span.first) {
                docBounds.push_back(span.first);
            }
            docBounds.push_back(span.second);
        }
        MaskRequest maskReq;
        maskReq.type         = AttentionMaskType::CausalBlockDiagonal;
        maskReq.seqLen       = total_len;
        maskReq.rank         = 4;
        maskReq.sharedPrefix = prefix_len;
        maskReq.cuSeqlens    = &docBounds;
        auto attention_mask = buildAttentionMask(maskReq);
        // position ids
        auto position_ids = _Input({1, total_len}, NCHW, halide_type_of<int>());
        auto pos_ptr = position_ids->writeMap<int>();
        for (int i = 0; i < total_len; i++) {
            pos_ptr[i] = position_ids_[i];
        }

        // clear history cache
        mLlm->setKVCacheInfo(total_len, mLlm->getCurrentHistory());
        
        // forward
        auto outputs = mLlm->forwardRaw(input_embeds, attention_mask, position_ids);
        if(outputs.empty()) {
            return {};
        }
        auto logits = outputs[0];
        auto logits_dim = logits->getInfo()->dim[2];
        auto logits_ptr = logits->readMap<float>();

        mLlm->reset();
        std::vector<float> scores;
        for (int i = 0; i < spans.size(); i++) {
            int logits_idx = spans[i].second - 1;
            float true_logit = logits_ptr[logits_idx * logits_dim + mTokenTrueId];
            float false_logit = logits_ptr[logits_idx * logits_dim + mTokenFalseId];
            // logsoftmax
            float max_logit = std::max(true_logit, false_logit);
            float stable_true_logit = true_logit - max_logit;
            float stable_false_logit = false_logit - max_logit;
            float exp_true = std::exp(stable_true_logit);
            float exp_false = std::exp(stable_false_logit);
            float score = exp_true / (exp_true + exp_false);
            scores.push_back(score);
        }

        return scores;
    }

private:
    std::string mInstruct;
    int mTokenTrueId;
    int mTokenFalseId;
};

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

class GteReranker : public RerankerBase {
public:
    /**
     * @brief Constructor for Qwen3Reranker.
     * @param config_path The path to the LLM configuration.
     */
    GteReranker(const std::string& config_path) {
        initialize(config_path);
    }

    /**
     * @brief Initializes the LLM and token IDs.
     * @param config_path The path to the LLM configuration.
     */
    void initialize(const std::string& config_path) override {
        mLlm.reset(Embedding::createEmbedding(config_path, false));
    }

    /**
     * @brief Loads the reranker model after initialization.
     */
    void load() override {
        mLlm->load();
    }

    /**
     * @brief Sets the instruction for the reranker.
     * @param instruct The instruction string.
     */
    void setInstruct(const std::string& instruct) override {}

    /**
     * @brief Computes scores for a list of documents using the Qwen3 model.
     * @param query The input query.
     * @param documents A vector of document strings.
     * @return A vector of float scores, one for each document.
     */
    std::vector<float> compute_scores(const std::string& query, const std::vector<std::string>& documents) override {
        CHECK_LLM_RUNNING_RET(mLlm->getContext(), std::vector<float>());
        constexpr int pad_token_id = 1;
        constexpr int eos_token_id = 2;
        std::vector<int> query_ids {0, 6};
        auto q_ids = mLlm->tokenizer_encode(query);
        query_ids.insert(query_ids.end(), q_ids.begin(), q_ids.end());
        query_ids.push_back(eos_token_id);
        query_ids.push_back(eos_token_id);
        std::vector<std::vector<int>> all_ids;
        int max_len = 0;
        for (const auto& doc : documents) {
            auto doc_ids = mLlm->tokenizer_encode(doc);
            std::vector<int> ids(query_ids);
            ids.insert(ids.end(), doc_ids.begin(), doc_ids.end());
            ids.push_back(eos_token_id);
            max_len = std::max(max_len, (int)ids.size());
            all_ids.push_back(std::move(ids));
        }
        const int batch = static_cast<int>(documents.size());
        const int total_len = max_len * batch;
        // Key padding [batch,1,1,max_len], broadcast over queries. Rerankers are exported without
        // transformer fusion, so this feeds a plain MatMul/Add/Softmax chain, not an Attention op.
        std::vector<int> validLens;
        validLens.reserve(all_ids.size());
        for (const auto& ids : all_ids) {
            validLens.push_back(static_cast<int>(ids.size()));
        }
        MaskRequest maskReq;
        maskReq.type      = AttentionMaskType::KeyPadding;
        maskReq.batch     = batch;
        maskReq.kvSeqLen  = max_len;
        maskReq.rank      = 4;  // -> {batch, 1, 1, max_len}
        maskReq.validLens = &validLens;
        auto attention_mask = buildAttentionMask(maskReq);
        std::vector<int> input_ids(total_len);
        int idx = 0;
        for (auto& ids : all_ids) {
            size_t i = 0;
            for (; i < ids.size(); i++, idx++) {
                input_ids[idx] = ids[i];
            }
            for (; (int)i < max_len; i++, idx++) {
                input_ids[idx] = pad_token_id;
            }
        }
        
        auto input_embeds = mLlm->embedding(input_ids);
        input_embeds = _Reshape(input_embeds, _var<int>({batch, max_len, -1}, {3}));
        auto position_ids = mLlm->gen_position_ids(max_len);
        auto outputs = mLlm->forwardRaw(input_embeds, attention_mask, position_ids);
        if(outputs.empty()) {
            return {};
        }
        std::vector<float> scores(outputs[0]->readMap<float>(), outputs[0]->readMap<float>() + batch);
        return scores;
    }

};

}
}
#endif /* MNN_LLM_RERANKER_hpp */
