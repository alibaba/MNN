#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "llmconfig.hpp"
#include "omni.hpp"

using namespace MNN::Express;
using namespace MNN::Transformer;

static std::shared_ptr<LlmConfig> makeConfig(bool isEmbedding) {
    auto config = std::make_shared<LlmConfig>();
    config->config_ = ujson::json::parse(R"({
        "attention_mask": "float",
        "attention_type": "mix",
        "backend_type": "cpu",
        "sliding_window": 2
    })");
    config->config_["is_embedding"] = isEmbedding;
    return config;
}

static bool checkGenerationMask() {
    Omni omni(makeConfig(false));
    auto mask = omni.gen_attention_mask(4);
    auto info = mask.get() ? mask->getInfo() : nullptr;
    const std::vector<int> expectedDims{2, 1, 1, 4, 4};
    if (info == nullptr || info->dim != expectedDims) {
        std::cerr << "Generation Omni must use the LLM mixed-attention mask" << std::endl;
        return false;
    }

    const float blocked = std::numeric_limits<float>::lowest();
    auto ptr = mask->readMap<float>();
    if (ptr == nullptr || ptr[0] != 0.0f || ptr[1] != blocked) {
        std::cerr << "Unexpected full-attention mask values" << std::endl;
        return false;
    }

    const int slidingOffset = 16;
    if (ptr[slidingOffset + 8] != blocked || ptr[slidingOffset + 9] != 0.0f || ptr[slidingOffset + 10] != 0.0f ||
        ptr[slidingOffset + 11] != blocked) {
        std::cerr << "Unexpected sliding-attention mask values" << std::endl;
        return false;
    }
    return true;
}

static bool checkEmbeddingMask() {
    Omni omni(makeConfig(true));
    auto mask = omni.gen_attention_mask(4);
    auto info = mask.get() ? mask->getInfo() : nullptr;
    auto ptr = mask.get() ? mask->readMap<float>() : nullptr;
    if (info == nullptr || !info->dim.empty() || ptr == nullptr || ptr[0] != 0.0f) {
        std::cerr << "Embedding Omni must preserve the scalar causal mask" << std::endl;
        return false;
    }
    return true;
}

int main() {
    return checkGenerationMask() && checkEmbeddingMask() ? 0 : 1;
}
