//
//  embedding_demo.cpp
//
//  Created by MNN on 2024/01/10.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#include <fstream>
#include <stdlib.h>

using namespace MNN::Express;
using namespace MNN::Transformer;

#define DUMP_NUM_DATA(type)                          \
    auto data = var->readMap<type>();                \
    for (int z = 0; z < outside; ++z) {              \
        for (int x = 0; x < width; ++x) {            \
            outputOs << data[x + z * width] << "\n"; \
        }                                            \
    }

static void dumpVar2File(VARP var, const char* file) {
    std::ofstream outputOs(file);
    
    auto dimension = var->getInfo()->dim.size();
    int width     = 1;
    if (dimension > 1) {
        width = var->getInfo()->dim[dimension - 1];
    }
    
    auto outside = var->getInfo()->size / width;
    DUMP_NUM_DATA(float);
    
}

static void dumpVARP(VARP var) {
    auto size = static_cast<int>(var->getInfo()->size);
    auto ptr = var->readMap<float>();
    printf("[ ");
    for (int i = 0; i < 5; i++) {
        printf("%f, ", ptr[i]);
    }
    printf("... ");
    for (int i = size - 5; i < size; i++) {
        printf("%f, ", ptr[i]);
    }
    printf(" ]\n");
}

static void unittest(std::unique_ptr<Embedding> &embedding, std::string prompt) {
    auto vec_0 = embedding->txt_embedding(prompt);
    float sum = 0;
    auto ptr = vec_0->readMap<float>();
    for (int i = 0;i < vec_0->getInfo()->size; ++i) {
        sum += ptr[i];
    }
    MNN_PRINT("%s\n", prompt.c_str());
    MNN_PRINT("sum = %f\n", sum);
    MNN_PRINT("\n");
}
static void benchmark(std::unique_ptr<Embedding> &embedding, std::string prompt_file) {
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    prompt_fs.close();
    for (auto& p: prompts) {
        unittest(embedding, p);
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json" << std::endl;
        return 0;
    }
    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Embedding> embedding(Embedding::createEmbedding(config_path, true));
    if (argc > 2) {
        benchmark(embedding, argv[2]);
        return 0;
    }
    unittest(embedding, "这个东西，这。");
//    dumpVar2File(vec_0, filename.c_str());
//    dumpVARP(vec_0);
    return 0;
}
