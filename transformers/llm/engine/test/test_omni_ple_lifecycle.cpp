#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "diskembedding.hpp"
#include "omni.hpp"

using namespace MNN::Transformer;

namespace {

struct TempEmbeddingFiles {
    std::string token;
    std::string ple;

    ~TempEmbeddingFiles() {
        std::remove(token.c_str());
        std::remove(ple.c_str());
    }
};

static bool writeBf16Embedding(const std::string& path, int tokenCount, int dimension) {
    std::vector<uint16_t> data(tokenCount * dimension, 0);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint16_t));
    return output.good();
}

class TestOmni : public Omni {
public:
    explicit TestOmni(const std::shared_ptr<LlmConfig>& config) : Omni(config) {
        mDiskEmbedding.reset(new DiskEmbedding(mConfig));
        mPleEmbedding.reset(
            new DiskEmbedding(mConfig, mConfig->ple_embed_file(), mConfig->ple_embed_dim(), mConfig->ple_quant()));
    }

    int pleSequenceLength() const {
        auto info = mPleInput.get() ? mPleInput->getInfo() : nullptr;
        if (info == nullptr || info->dim.size() != 3) {
            return -1;
        }
        return info->dim[1];
    }
};

} // namespace

int main() {
    std::ostringstream prefix;
    prefix << "test_omni_ple_" << std::chrono::high_resolution_clock::now().time_since_epoch().count();
    TempEmbeddingFiles files{prefix.str() + "_token.bin", prefix.str() + "_ple.bin"};

    const int tokenCount = 8;
    const int hiddenSize = 2;
    const int pleDimension = 3;
    if (!writeBf16Embedding(files.token, tokenCount, hiddenSize) ||
        !writeBf16Embedding(files.ple, tokenCount, pleDimension)) {
        std::cerr << "Failed to create temporary embedding files" << std::endl;
        return 1;
    }

    auto config = std::make_shared<LlmConfig>();
    config->config_ = ujson::json::parse("{}");
    config->config_["hidden_size"] = hiddenSize;
    config->config_["embedding_file"] = files.token;
    config->config_["ple_embed_file"] = files.ple;
    config->config_["ple_embed_dim"] = pleDimension;

    TestOmni omni(config);
    auto decodeEmbedding = omni.embedding({1});
    if (decodeEmbedding == nullptr || omni.pleSequenceLength() != 1) {
        std::cerr << "Expected decode to leave a one-token PLE input" << std::endl;
        return 1;
    }

    auto nextPrefillEmbedding = omni.embedding({2, 3, 4});
    if (nextPrefillEmbedding == nullptr || omni.pleSequenceLength() != 3) {
        std::cerr << "Text prefill reused the stale one-token PLE input" << std::endl;
        return 1;
    }
    return 0;
}
