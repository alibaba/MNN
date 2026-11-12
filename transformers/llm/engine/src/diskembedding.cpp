//
//  diskembedding.cpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <memory>
#include <unordered_map>
#include <climits>
#include <cstring>
#include <limits>
#include "diskembedding.hpp"
#include "half.hpp"

namespace MNN {
namespace Transformer {

static void q41_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size / 2; i++) {
        int x          = src[i];
        int x1         = x / 16;
        int x2         = x % 16;
        float w1       = x1 * scale + zero;
        float w2       = x2 * scale + zero;
        dst[2 * i]     = w1;
        dst[2 * i + 1] = w2;
    }
}

static void q81_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = (src[i]) * scale + zero;
    }
}

static void q21_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size / 4; i++) {
        int x = src[i];
        dst[4 * i]     = ((x >> 6) & 3) * scale + zero;
        dst[4 * i + 1] = ((x >> 4) & 3) * scale + zero;
        dst[4 * i + 2] = ((x >> 2) & 3) * scale + zero;
        dst[4 * i + 3] = (x & 3) * scale + zero;
    }
}

static void q31_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    // 3-bit MSB-first bit-packed (mirrors SplitBufToArray): value i at bit offset 3*i.
    int bitpos = 0;
    for (int i = 0; i < size; i++) {
        int r = bitpos % 8;
        int shift = 8 - 3 - r;
        int v;
        if (shift < 0) {
            v = ((src[bitpos / 8] << (-shift)) | (src[bitpos / 8 + 1] >> (8 + shift))) & 7;
        } else {
            v = (src[bitpos / 8] >> shift) & 7;
        }
        dst[i] = v * scale + zero;
        bitpos += 3;
    }
}

static float readAlphaValue(const uint8_t* src, size_t elemBytes) {
    if (elemBytes == sizeof(uint16_t)) {
        uint16_t raw;
        memcpy(&raw, src, sizeof(raw));
        half_float::half value;
        memcpy(&value, &raw, sizeof(value));
        return float(value);
    }
    float value;
    memcpy(&value, src, sizeof(value));
    return value;
}

static bool checkedAdd(size_t a, size_t b, size_t* result) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    *result = a + b;
    return true;
}

static bool checkedMul(size_t a, size_t b, size_t* result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    *result = a * b;
    return true;
}

static bool toSize(int64_t value, size_t* result) {
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) {
        return false;
    }
    *result = static_cast<size_t>(value);
    return true;
}

DiskEmbedding::~DiskEmbedding() {
    if (mMappedData != nullptr) {
        MNNUnmapFile(mMappedData, mMappedSize);
    }
}

bool DiskEmbedding::openSourceFile(const std::string& fileName, bool useMmap) {
    if (useMmap && !FileLoader::isFileRegion(fileName)) {
        // The mapped file must stay immutable while it backs embedding reads.
        file_t file = MNNOpenFile(fileName.c_str(), MNN_FILE_READ);
        if (file != INVALID_FILE) {
            const size_t fileSize = MNNGetFileSize(file);
            if (fileSize > 0) {
                mMappedData = reinterpret_cast<uint8_t*>(MNNMmapFile(file, fileSize, true));
                if (mMappedData != nullptr) {
                    mMappedSize = fileSize;
                    mSourceSize = fileSize;
                }
            }
            MNNCloseFile(file);
            if (mMappedData != nullptr) {
                return true;
            }
        }
    }
    mFile.reset(new FileLoader(fileName.c_str(), true));
    if (mFile == nullptr || !mFile->valid()) {
        return false;
    }
    const int64_t fileSize = mFile->contentSize();
    if (!toSize(fileSize, &mSourceSize) || mSourceSize == 0) {
        return false;
    }
    return true;
}

bool DiskEmbedding::readBytes(uint8_t* dst, size_t size, size_t offset) const {
    const size_t limit = mMappedData != nullptr ? mMappedSize : std::numeric_limits<size_t>::max();
    if (offset > limit) {
        return false;
    }
    if (size > limit - offset) {
        return false;
    }
    if (mMappedData != nullptr) {
        memcpy(dst, mMappedData + offset, size);
        return true;
    }
    if (mFile == nullptr || mFile->offset(offset) != 0) {
        return false;
    }
    return mFile->read(reinterpret_cast<char*>(dst), size);
}

const uint8_t* DiskEmbedding::readToken(size_t token) {
    if (token >= mTokenCount || mTokenSize == 0) {
        return nullptr;
    }
    size_t tokenOffset = 0;
    size_t offset = 0;
    if (!checkedMul(token, mTokenSize, &tokenOffset) || !checkedAdd(mWeightOffset, tokenOffset, &offset)) {
        return nullptr;
    }
    if (mMappedData != nullptr) {
        size_t end = 0;
        if (!checkedAdd(offset, mTokenSize, &end) || end > mMappedSize) {
            return nullptr;
        }
        return mMappedData + offset;
    }
    if (mFile->offset(offset) != 0 || !mFile->read(reinterpret_cast<char*>(mWeight.get()), mTokenSize)) {
        return nullptr;
    }
    return mWeight.get();
}

bool DiskEmbedding::getAlpha(size_t token, size_t block, float* scale, float* zero) const {
    if (token >= mTokenCount || block >= mBlockNum) {
        return false;
    }
    size_t valuesPerToken = 0;
    if (!checkedMul(mBlockNum, mAsymc ? 2 : 1, &valuesPerToken) || valuesPerToken == 0) {
        return false;
    }
    size_t tokenValue = 0;
    size_t blockValue = 0;
    size_t valueIndex = 0;
    size_t byteIndex = 0;
    if (!checkedMul(token, valuesPerToken, &tokenValue) || !checkedMul(block, mAsymc ? 2 : 1, &blockValue) ||
        !checkedAdd(tokenValue, blockValue, &valueIndex) || !checkedMul(valueIndex, mAlphaElemBytes, &byteIndex)) {
        return false;
    }
    size_t valueEnd = 0;
    if (!checkedAdd(byteIndex, mAsymc ? mAlphaElemBytes * 2 : mAlphaElemBytes, &valueEnd) ||
        valueEnd > mAlphaDiskSize) {
        return false;
    }

    if (mMappedData != nullptr) {
        const auto src = mMappedData + mAlphaOffset + byteIndex;
        if (mAsymc) {
            float alphaZero = readAlphaValue(src, mAlphaElemBytes);
            *scale = readAlphaValue(src + mAlphaElemBytes, mAlphaElemBytes);
            *zero = alphaZero + *scale * mOffset;
        } else {
            *scale = readAlphaValue(src, mAlphaElemBytes);
            *zero = mOffset * *scale;
        }
        return true;
    }

    const auto alpha = reinterpret_cast<const float*>(mAlpha.get());
    if (mAsymc) {
        *zero = alpha[valueIndex];
        *scale = alpha[valueIndex + 1];
    } else {
        *scale = alpha[valueIndex];
        *zero = mOffset * *scale;
    }
    return true;
}

DiskEmbedding::DiskEmbedding(const std::shared_ptr<LlmConfig>& config, std::string fileName, int hiddenSize,
                             std::vector<int64_t> quant_info) {
    // External positional list (e.g. ple_quant) takes precedence; otherwise read structured form from config.
    LlmConfig::TieEmbeddingsInfo info =
        quant_info.size() >= 5 ? LlmConfig::parseTieEmbeddings(quant_info) : config->tie_embeddings_info();
    const int resolvedHiddenSize = hiddenSize > 0 ? hiddenSize : config->hidden_size();
    if (resolvedHiddenSize <= 0) {
        MNN_ERROR("Invalid embedding hidden size\n");
        return;
    }
    mHiddenSize = static_cast<size_t>(resolvedHiddenSize);
    if (mHiddenSize > std::numeric_limits<size_t>::max() / sizeof(float)) {
        MNN_ERROR("Embedding hidden size is too large\n");
        return;
    }
    if (hiddenSize > 0 && !fileName.empty() && quant_info.empty()) {
        // PLE bf16 path
        if (!checkedMul(mHiddenSize, sizeof(int16_t), &mTokenSize)) {
            MNN_ERROR("Invalid embedding token size\n");
            return;
        }
        if (!openSourceFile(fileName, config->use_mmap())) {
            MNN_ERROR("Failed to open embedding file!\n");
            return;
        }
        const size_t payloadSize = mSourceSize;
        if (payloadSize == 0 || payloadSize % mTokenSize != 0) {
            MNN_ERROR("Invalid embedding file size\n");
            return;
        }
        mTokenCount = payloadSize / mTokenSize;
    } else if (info.valid) {
        size_t weightOffset = 0;
        size_t alphaOffset = 0;
        size_t alphaSize = 0;
        size_t quantBit = 0;
        if (!toSize(info.weight_offset, &weightOffset) || !toSize(info.alpha_offset, &alphaOffset) ||
            !toSize(info.alpha_size, &alphaSize) || !toSize(info.quant_bit, &quantBit) ||
            (quantBit != 8 && quantBit != 4 && quantBit != 3 && quantBit != 2 && quantBit != 16) ||
            alphaOffset < weightOffset) {
            MNN_ERROR("Invalid embedding quantization metadata\n");
            return;
        }
        mWeightOffset = weightOffset;
        mQuantBit = quantBit;
        mAlphaOffset = alphaOffset;
        mAlphaDiskSize = alphaSize;
        const bool alphaIsFp16 = info.alpha_fp16;
        if (mWeightOffset == 0) {
            // embedding_int8/4.bin
            if (fileName.empty()) {
                fileName = config->embedding_file();
            }
        } else {
            fileName = config->llm_weight();
        }
        if (!openSourceFile(fileName, config->use_mmap())) {
            MNN_ERROR("Failed to open embedding file!\n");
            return;
        }
        size_t tokenBits = 0;
        if (!checkedMul(mHiddenSize, mQuantBit, &tokenBits) || tokenBits % 8 != 0) {
            MNN_ERROR("Invalid embedding token size\n");
            return;
        }
        mTokenSize = tokenBits / 8;
        if (mTokenSize == 0 || mAlphaOffset > mSourceSize) {
            MNN_ERROR("Invalid embedding file size\n");
            return;
        }
        const size_t weightBytes = mAlphaOffset - mWeightOffset;
        if (weightBytes == 0 || weightBytes % mTokenSize != 0) {
            MNN_ERROR("Invalid embedding weight size\n");
            return;
        }
        mTokenCount = weightBytes / mTokenSize;
        size_t alphaEnd = 0;
        if (!checkedAdd(mAlphaOffset, mAlphaDiskSize, &alphaEnd) || alphaEnd > mSourceSize) {
            MNN_ERROR("Embedding alpha is outside the source file\n");
            return;
        }
        if (mQuantBit != 16) {
            size_t quantBlock = 0;
            if (!toSize(info.quant_block, &quantBlock)) {
                MNN_ERROR("Invalid embedding quantization metadata\n");
                return;
            }
            if (quantBlock == 0) {
                mBlockNum = 1;
                mQuantBlock = mHiddenSize; // be used for mDequantFunc.
            } else {
                mQuantBlock = quantBlock;
                if (mQuantBlock > mHiddenSize || mHiddenSize % mQuantBlock != 0) {
                    MNN_ERROR("Invalid embedding quant block\n");
                    return;
                }
                mBlockNum = mHiddenSize / mQuantBlock;
            }
            if (mQuantBlock > INT_MAX || mBlockNum == 0) {
                MNN_ERROR("Embedding quant block is too large\n");
                return;
            }
            size_t blockBits = 0;
            if (!checkedMul(mQuantBlock, mQuantBit, &blockBits) || blockBits % 8 != 0) {
                MNN_ERROR("Invalid embedding quant block\n");
                return;
            }
            if (mQuantBit == 8) {
                mDequantFunc = q81_dequant_ref;
            } else if (mQuantBit == 4) {
                mDequantFunc = q41_dequant_ref;
            } else if (mQuantBit == 3) {
                mDequantFunc = q31_dequant_ref;
            } else {
                mDequantFunc = q21_dequant_ref;
            }
            mAlphaElemBytes = alphaIsFp16 ? sizeof(uint16_t) : sizeof(float);
            if (mAlphaDiskSize % mAlphaElemBytes != 0) {
                MNN_ERROR("Invalid embedding quantization metadata\n");
                return;
            }
            const size_t alphaElems = mAlphaDiskSize / mAlphaElemBytes;
            size_t alphaSizeFp32 = 0;
            size_t alphaValues = 0;
            size_t doubleAlphaValues = 0;
            if (!checkedMul(alphaElems, sizeof(float), &alphaSizeFp32) ||
                !checkedMul(mBlockNum, mTokenCount, &alphaValues) || !checkedMul(alphaValues, 2, &doubleAlphaValues)) {
                MNN_ERROR("Invalid embedding alpha size\n");
                return;
            }
            if (alphaElems == alphaValues) {
                mAsymc = false;
            } else if (alphaElems == doubleAlphaValues) {
                mAsymc = true;
            } else {
                MNN_ERROR("Invalid embedding alpha size\n");
                return;
            }
            mOffset = -(1 << (mQuantBit - 1));

            if (mMappedData == nullptr) {
                mAlpha.reset(new uint8_t[alphaSizeFp32]);
                if (alphaIsFp16) {
                    std::vector<uint16_t> tmp(alphaElems);
                    if (!readBytes(reinterpret_cast<uint8_t*>(tmp.data()), mAlphaDiskSize, mAlphaOffset)) {
                        MNN_ERROR("Failed to read embedding alpha\n");
                        return;
                    }
                    auto src = reinterpret_cast<half_float::half*>(tmp.data());
                    auto dst = reinterpret_cast<float*>(mAlpha.get());
                    for (size_t i = 0; i < alphaElems; ++i) {
                        dst[i] = float(src[i]);
                    }
                } else if (!readBytes(mAlpha.get(), mAlphaDiskSize, mAlphaOffset)) {
                    MNN_ERROR("Failed to read embedding alpha\n");
                    return;
                }
                if (mAsymc) {
                    auto alphaPtr = reinterpret_cast<float*>(mAlpha.get());
                    for (size_t i = 0; i < alphaValues; ++i) {
                        alphaPtr[2 * i] = alphaPtr[2 * i] + alphaPtr[2 * i + 1] * mOffset;
                    }
                }
            }
        } else if (mAlphaDiskSize != 0) {
            MNN_ERROR("FP16 embedding must not have alpha data\n");
            return;
        }
    } else {
        if (fileName.empty()) {
            fileName = config->embedding_file();
        }
        if (!checkedMul(mHiddenSize, sizeof(int16_t), &mTokenSize)) {
            MNN_ERROR("Invalid embedding token size\n");
            return;
        }
        if (!openSourceFile(fileName, config->use_mmap())) {
            MNN_ERROR("Failed to open embedding file!\n");
            return;
        }
        if (mSourceSize == 0 || mSourceSize % mTokenSize != 0) {
            MNN_ERROR("Invalid embedding file size\n");
            return;
        }
        mTokenCount = mSourceSize / mTokenSize;
    }
    if (mMappedData == nullptr) {
        mWeight.reset(new uint8_t[mTokenSize]);
    }
    mValid = true;
}

bool DiskEmbedding::embedding(const std::vector<int>& input_ids, float* dst) {
    if (dst == nullptr || !mValid) {
        return false;
    }
    size_t outputSize = 0;
    if (!checkedMul(input_ids.size(), mHiddenSize, &outputSize) ||
        !checkedMul(outputSize, sizeof(float), &outputSize)) {
        return false;
    }
    for (int id : input_ids) {
        if (id < 0 || static_cast<size_t>(id) >= mTokenCount) {
            MNN_ERROR("Invalid embedding token: %d, token count: %zu\n", id, mTokenCount);
            return false;
        }
    }
    std::unordered_map<int, size_t> cache_tokens;
    cache_tokens.reserve(input_ids.size());
#define TRY_CACHE_TOKEN                                                                        \
    if (cache_tokens.find(token) != cache_tokens.end()) {                                      \
        size_t index = cache_tokens[token];                                                    \
        memcpy(dst + i * mHiddenSize, dst + index * mHiddenSize, mHiddenSize * sizeof(float)); \
        continue;                                                                              \
    } else {                                                                                   \
        cache_tokens.insert(std::make_pair(token, i));                                         \
    }

    if (mDequantFunc != nullptr) {
        // quant
        if (mAsymc) {
            for (size_t i = 0; i < input_ids.size(); i++) {
                const int tokenId = input_ids[i];
                size_t token = static_cast<size_t>(tokenId);
                TRY_CACHE_TOKEN;
                auto src = readToken(token);
                if (src == nullptr) {
                    MNN_ERROR("Invalid embedding token: %zu\n", token);
                    return false;
                }
                auto dptr = dst + i * mHiddenSize;
                for (size_t n = 0; n < mBlockNum; n++) {
                    auto dst_ptr = dptr + n * mQuantBlock;
                    auto src_ptr = src + n * (mQuantBlock * mQuantBit / 8);
                    float zero = 0.0f;
                    float scale = 0.0f;
                    if (!getAlpha(token, n, &scale, &zero)) {
                        MNN_ERROR("Invalid embedding alpha for token: %zu\n", token);
                        return false;
                    }
                    mDequantFunc(src_ptr, dst_ptr, scale, zero, static_cast<int>(mQuantBlock));
                }
            }
        } else {
            for (size_t i = 0; i < input_ids.size(); i++) {
                const int tokenId = input_ids[i];
                size_t token = static_cast<size_t>(tokenId);
                TRY_CACHE_TOKEN;
                auto src = readToken(token);
                if (src == nullptr) {
                    MNN_ERROR("Invalid embedding token: %zu\n", token);
                    return false;
                }
                auto dptr = dst + i * mHiddenSize;
                for (size_t n = 0; n < mBlockNum; n++) {
                    auto dst_ptr = dptr + n * mQuantBlock;
                    auto src_ptr = src + n * (mQuantBlock * mQuantBit / 8);
                    float zero = 0.0f;
                    float scale = 0.0f;
                    if (!getAlpha(token, n, &scale, &zero)) {
                        MNN_ERROR("Invalid embedding alpha for token: %zu\n", token);
                        return false;
                    }
                    mDequantFunc(src_ptr, dst_ptr, scale, zero, static_cast<int>(mQuantBlock));
                }
            }
        }
        return true;
    }
    if (mQuantBit == 16) {
        // FP16
        for (size_t i = 0; i < input_ids.size(); i++) {
            const int tokenId = input_ids[i];
            size_t token = static_cast<size_t>(tokenId);
            TRY_CACHE_TOKEN;
            auto src = readToken(token);
            if (src == nullptr) {
                MNN_ERROR("Invalid embedding token: %zu\n", token);
                return false;
            }
            auto srcPtr = reinterpret_cast<const half_float::half*>(src);
            auto dst_ptr = reinterpret_cast<float*>(dst + i * mHiddenSize);
            for (int j = 0; j < mHiddenSize; j++) {
                dst_ptr[j] = srcPtr[j];
            }
        }
        return true;
    }
    // bf16
    for (size_t i = 0; i < input_ids.size(); i++) {
        const int tokenId = input_ids[i];
        size_t token = static_cast<size_t>(tokenId);
        TRY_CACHE_TOKEN;
        auto src = readToken(token);
        if (src == nullptr) {
            MNN_ERROR("Invalid embedding token: %zu\n", token);
            return false;
        }
        int16_t* dst_ptr = reinterpret_cast<int16_t*>(dst + i * mHiddenSize);
        for (int j = 0; j < mHiddenSize; j++) {
            dst_ptr[j * 2] = 0;
            dst_ptr[j * 2 + 1] = reinterpret_cast<const int16_t*>(src)[j];
        }
    }
#undef TRY_CACHE_TOKEN
    return true;
}
}
}
