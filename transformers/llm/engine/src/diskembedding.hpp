//
//  diskembedding.hpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DISKEMBEDDING_hpp
#define DISKEMBEDDING_hpp

#include "llmconfig.hpp"
#include "core/FileLoader.hpp"

namespace MNN {
namespace Transformer {

typedef void (*DequantFunction)(const uint8_t*, float*, float, float, int);

class DiskEmbedding {
public:
    explicit DiskEmbedding(const std::shared_ptr<LlmConfig>& config, std::string fileName = "");
    ~DiskEmbedding() {}
    void embedding(const std::vector<int>& input_ids, float* ptr);

private:
    void seek_read(uint8_t* dst, size_t size, size_t offset);
    std::unique_ptr<uint8_t[]> mAlpha  = nullptr;
    std::unique_ptr<uint8_t[]> mWeight = nullptr;
    std::unique_ptr<FileLoader> mFile;
    DequantFunction mDequantFunc;
    int mHiddenSize, mTokenSize;
    float mOffset = 0.0f;
    bool mAsymc = true;
    int64_t mWeightOffset = 0;
    int64_t mBlockNum = 0, mQuantBlock = 0, mQuantBit = 0;
};

}
}
#endif // DISKEMBEDDING_hpp
