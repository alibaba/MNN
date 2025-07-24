//
//  reranker_demo.cpp
//
//  Created by MNN on 2024/07/10.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#include <fstream>
#include <stdlib.h>

using namespace MNN::Transformer;

class Qwen3Reranker {
public:
    Qwen3Reranker(const std::string& config_path) {
        mLlm.reset(Llm::createLLM(config_path));
        mLlm->load();
        mTokenTrueId = mLlm->tokenizer_encode("yes")[0];
        mTokenFalseId = mLlm->tokenizer_encode("no")[0];
    }
    void setInstruct(const std::string& instruct)  {
        mInstruct = instruct;
    }
    float compute_score(const std::string& query, const std::string& document) {
        auto prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
        auto suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        auto content = "<Instruct>: " + mInstruct + "\n<Query>: " + query + "\n<Document>: " + document;
        auto prompt = prefix + content + suffix;
        auto ids = mLlm->tokenizer_encode(prompt);
        return compute_score_from_ids(ids);
    }
    std::vector<float> compute_scores(const std::string& query, const std::vector<std::string>& documents) {
        std::vector<float> scores;
        scores.reserve(documents.size());
        for (const auto& doc : documents) {
            scores.push_back(compute_score(query, doc));
        }
        return scores;
    }
private:
    float compute_score_from_ids(const std::vector<int>& ids) {
        auto logits = mLlm->forward(ids, true);
        auto logits_ptr = logits->readMap<float>();
        mLlm->reset();
        if (nullptr == logits_ptr) {
            MNN_ERROR("Reranker failed to read logits from model output.\n");
            return 0.0f;
        }
        float true_logit  = logits_ptr[mTokenTrueId];
        float false_logit = logits_ptr[mTokenFalseId];
        // logsoftmax
        float max_logit = std::max(true_logit, false_logit);
        float stable_true_logit = true_logit - max_logit;
        float stable_false_logit = false_logit - max_logit;
        float exp_true = std::exp(stable_true_logit);
        float exp_false = std::exp(stable_false_logit);
        float score = exp_true / (exp_true + exp_false);
        return score;
    }
private:
    std::unique_ptr<Llm> mLlm;
    std::string mInstruct;
    int mTokenTrueId, mTokenFalseId;
};


int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json" << std::endl;
        return 0;
    }
    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Qwen3Reranker> reranker(new Qwen3Reranker(config_path));
    reranker->setInstruct("Given a web search query, retrieve relevant passages that answer the query");
    std::string query = "What is the capital of China?";
    std::vector<std::string> documents = {
        "The capital of China is Beijing.",
        "春天到了，樱花树悄然绽放",
        "北京是中国的首都",
        "在炎热的夏日里，沙滩上的游客们穿着泳装享受着海水的清凉。"
    };
    auto scores = reranker->compute_scores(query, documents);
    // sorted_documents by scores
    std::vector<int> documents_index(documents.size());
    for (int i = 0; i < documents.size(); i++) {
        documents_index[i] = i;
    }
    std::sort(documents_index.begin(), documents_index.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });
    for (auto index : documents_index) {
        std::cout << documents[index] << " " << scores[index] << std::endl;
    }
    return 0;
}
