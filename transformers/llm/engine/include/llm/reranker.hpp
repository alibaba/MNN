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
     * @brief Initializes the LLM and token IDs.
     * @param config_path The path to the LLM configuration.
     */
    void initialize(const std::string& config_path) override {
        mLlm.reset(Llm::createLLM(config_path));
        mLlm->set_config("{\"all_logits\":true}");
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
        // attention mask
        auto attention_mask = _Input({1, 1, total_len, total_len}, NCHW, halide_type_of<float>());
        auto ptr = attention_mask->writeMap<float>();
        for (int i = 0; i < total_len; i++) {
            for (int j = 0; j < total_len; j++) {
                ptr[total_len * i + j] = (j > i) * std::numeric_limits<float>::lowest();
            }
        }
        for (int i = 0; i < spans.size(); i++) {
            for (int j = 0; j < spans.size(); j++) {
                if (i == j) {
                    continue;
                }
                int start_i = spans[i].first;
                int end_i = spans[i].second;
                int start_j = spans[j].first;
                int end_j = spans[j].second;
                for (int k = start_i; k < end_i; k++) {
                    for (int l = start_j; l < end_j; l++) {
                        ptr[total_len * k + l] = std::numeric_limits<float>::lowest();
                    }
                }
            }
        }
        // position ids
        auto position_ids = _Input({1, total_len}, NCHW, halide_type_of<int>());
        auto pos_ptr = position_ids->writeMap<int>();
        for (int i = 0; i < total_len; i++) {
            pos_ptr[i] = position_ids_[i];
        }
        // forward
        mLlm->setKVCacheInfo(total_len, 0);
        auto logits = mLlm->forwardRaw(input_embeds, attention_mask, position_ids)[0];
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
    std::unique_ptr<Llm> mLlm;
    std::string mInstruct;
    int mTokenTrueId;
    int mTokenFalseId;
};

}
}
#endif /* MNN_LLM_RERANKER_hpp */