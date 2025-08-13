//
//  reranker_demo.cpp
//
//  Created by MNN on 2024/07/10.
//  ZhaodeWang
//

#include "llm/reranker.hpp"
#include <stdlib.h>
#include <chrono>

using namespace MNN::Transformer;

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
    auto t0 = std::chrono::high_resolution_clock::now();
    auto scores = reranker->compute_scores(query, documents);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Reranker compute time: " << duration << " ms" << std::endl;
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
