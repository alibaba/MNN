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
#include "core/MNNFileUtils.h"

namespace MNN {
namespace Transformer {

typedef void (*DequantFunction)(const uint8_t*, float*, float, float, int);

class DiskEmbedding {
public:
    explicit DiskEmbedding(const std::shared_ptr<LlmConfig>& config, std::string fileName = "", int hiddenSize = 0,
                           std::vector<int64_t> quant_info = {});
    ~DiskEmbedding();
    bool embedding(const std::vector<int>& input_ids, float* ptr);
    bool valid() const { return mValid; }
    bool mapped() const { return mMappedData != nullptr; }

private:
    bool openSourceFile(const std::string& fileName, bool useMmap);
    bool readBytes(uint8_t* dst, size_t size, size_t offset) const;
    const uint8_t* readToken(size_t token);
    bool getAlpha(size_t token, size_t block, float* scale, float* zero) const;

    std::unique_ptr<uint8_t[]> mAlpha = nullptr;
    std::unique_ptr<uint8_t[]> mWeight = nullptr;
    std::unique_ptr<FileLoader> mFile;
    uint8_t* mMappedData = nullptr;
    size_t mMappedSize = 0;
    size_t mSourceSize = 0;
    size_t mAlphaOffset = 0;
    size_t mAlphaDiskSize = 0;
    size_t mAlphaElemBytes = 0;
    size_t mTokenCount = 0;
    DequantFunction mDequantFunc = nullptr;
    size_t mHiddenSize = 0, mTokenSize = 0;
    float mOffset = 0.0f;
    bool mAsymc = true;
    bool mValid = false;
    size_t mWeightOffset = 0;
    size_t mBlockNum = 0, mQuantBlock = 0, mQuantBit = 0;
};

}
}
#endif // DISKEMBEDDING_hpp
