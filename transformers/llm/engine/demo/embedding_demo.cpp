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
    auto vec_0 = embedding->txt_embedding("在春暖花开的季节，走在樱花缤纷的道路上，人们纷纷拿出手机拍照留念。樱花树下，情侣手牵手享受着这绝美的春光。孩子们在树下追逐嬉戏，脸上洋溢着纯真的笑容。春天的气息在空气中弥漫，一切都显得那么生机勃勃，充满希望。");
    auto vec_1 = embedding->txt_embedding("春天到了，樱花树悄然绽放，吸引了众多游客前来观赏。小朋友们在花瓣飘落的树下玩耍，而恋人们则在这浪漫的景色中尽情享受二人世界。每个人的脸上都挂着幸福的笑容，仿佛整个世界都被春天温暖的阳光和满树的樱花渲染得更加美好。");
    auto vec_2 = embedding->txt_embedding("在炎热的夏日里，沙滩上的游客们穿着泳装享受着海水的清凉。孩子们在海边堆沙堡，大人们则在太阳伞下品尝冷饮，享受悠闲的时光。远处，冲浪者们挑战着波涛，体验着与海浪争斗的刺激。夏天的海滩，总是充满了活力和热情。");
    dumpVARP(vec_0);
    dumpVARP(vec_1);
    dumpVARP(vec_2);
    printf("dist_0_1: %f\n", Embedding::dist(vec_0, vec_1));
    printf("dist_0_2: %f\n", Embedding::dist(vec_0, vec_2));
    printf("dist_1_2: %f\n", Embedding::dist(vec_1, vec_2));
    printf("cos_sim_0_1: %f\n", Embedding::cos_sim(vec_0, vec_1));
    printf("cos_sim_0_2: %f\n", Embedding::cos_sim(vec_0, vec_2));
    printf("cos_sim_1_2: %f\n", Embedding::cos_sim(vec_1, vec_2));
    return 0;
}
