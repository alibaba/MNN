//
//  diskembedding.cpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <memory>
#include <unordered_map>
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

void DiskEmbedding::seek_read(uint8_t* dst, size_t size, size_t offset) {
    mFile->offset(offset);
    mFile->read((char*)dst, size);
}

DiskEmbedding::DiskEmbedding(const std::shared_ptr<LlmConfig>& config, std::string fileName) {
    auto tie_embeddings = config->tie_embeddings();
    mHiddenSize        = config->hidden_size();
    if (tie_embeddings.size() == 5) {
        mWeightOffset     = tie_embeddings[0];
        mQuantBit         = tie_embeddings[3];
        mQuantBlock       = tie_embeddings[4];
        if (mWeightOffset == 0) {
            // embedding_int8/4.bin
            if (fileName.empty()) {
                fileName = config->embedding_file();
            }
        } else {
            fileName = config->llm_weight();
        }
        mFile.reset(new FileLoader(fileName.c_str(), true));
        mTokenSize = mHiddenSize * mQuantBit / 8;
        // TODO: optimize dequant function
        if (mQuantBit != 16) {
            if (mQuantBlock == 0) {
                mBlockNum = 1;
                mQuantBlock = mHiddenSize; // be used for mDequantFunc.
            } else {
                mBlockNum = mHiddenSize / mQuantBlock;
            }
            mDequantFunc      = mQuantBit == 8 ? q81_dequant_ref : q41_dequant_ref;
            auto a_offset   = tie_embeddings[1];
            auto alpha_size = tie_embeddings[2];
            size_t oc = (a_offset - mWeightOffset) / mHiddenSize * (8 / mQuantBit);

            mAlpha.reset(new uint8_t[alpha_size]);
            seek_read(mAlpha.get(), alpha_size, a_offset);
            mOffset = -(1 << (mQuantBit-1));
            if (alpha_size == sizeof(float) * mBlockNum * oc) {
                mAsymc = false;
            } else {
                MNN_ASSERT(alpha_size == 2 * sizeof(float) * mBlockNum * oc);
                mAsymc = true;
                auto alphaPtr = (float*)mAlpha.get();
                for (int i=0; i<mBlockNum * oc; ++i) {
                    alphaPtr[2*i] = alphaPtr[2*i] + alphaPtr[2*i+1] * mOffset;
                }
            }
        }
    } else {
        if (fileName.empty()) {
            fileName = config->embedding_file();
        }
        mTokenSize = mHiddenSize * sizeof(int16_t);
        mFile.reset(new FileLoader(fileName.c_str(), true));
    }
    if(mFile == nullptr || (!mFile->valid())) {
        MNN_ERROR("Failed to open embedding file!\n");
    }
    mWeight.reset(new uint8_t[mTokenSize]);
}

void DiskEmbedding::embedding(const std::vector<int>& input_ids, float* dst) {
    std::unordered_map<int, int> cache_tokens;
#define TRY_CACHE_TOKEN \
    if (cache_tokens.find(token) != cache_tokens.end()) {\
        int index = cache_tokens[token];\
        memcpy(dst + i * mHiddenSize, dst + index * mHiddenSize, mHiddenSize * sizeof(float));\
        continue;\
    } else {\
        cache_tokens.insert(std::make_pair(token, i));\
    }

    if (mAlpha.get()) {
        // quant
        if (mAsymc) {
            for (size_t i = 0; i < input_ids.size(); i++) {
                int token = input_ids[i];
                TRY_CACHE_TOKEN;
                seek_read(mWeight.get(), mTokenSize, mWeightOffset + token * mTokenSize);
                auto dptr      = dst + i * mHiddenSize;
                auto alpha_ptr = reinterpret_cast<float*>(mAlpha.get()) + token * mBlockNum * 2;
                for (int n = 0; n < mBlockNum; n++) {
                    auto dst_ptr     = dptr + n * mQuantBlock;
                    uint8_t* src_ptr = mWeight.get() + n * (mQuantBlock * mQuantBit / 8);
                    float zero       = (alpha_ptr + n * 2)[0];
                    float scale      = (alpha_ptr + n * 2)[1];
                    mDequantFunc(src_ptr, dst_ptr, scale, zero, mQuantBlock);
                }
            }
        } else {
            for (size_t i = 0; i < input_ids.size(); i++) {
                int token = input_ids[i];
                TRY_CACHE_TOKEN;
                seek_read(mWeight.get(), mTokenSize, mWeightOffset + token * mTokenSize);
                auto dptr      = dst + i * mHiddenSize;
                auto alpha_ptr = reinterpret_cast<float*>(mAlpha.get()) + token * mBlockNum;
                for (int n = 0; n < mBlockNum; n++) {
                    auto dst_ptr     = dptr + n * mQuantBlock;
                    uint8_t* src_ptr = mWeight.get() + n * (mQuantBlock * mQuantBit / 8);
                    float scale      = (alpha_ptr + n)[0];
                    float zero       = mOffset * scale;
                    mDequantFunc(src_ptr, dst_ptr, scale, zero, mQuantBlock);
                }
            }
        }
        return;
    }
    if (mQuantBit == 16) {
        // FP16
        for (size_t i = 0; i < input_ids.size(); i++) {
            int token = input_ids[i];
            TRY_CACHE_TOKEN;
            seek_read(mWeight.get(), mTokenSize, mWeightOffset + token * mTokenSize);
            auto src = (half_float::half*)mWeight.get();
            auto dst_ptr = reinterpret_cast<float*>(dst + i * mHiddenSize);
            for (int j = 0; j < mHiddenSize; j++) {
                dst_ptr[j] = src[j];
            }
        }
        return;
    }
    // bf16
    for (size_t i = 0; i < input_ids.size(); i++) {
        int token = input_ids[i];
        TRY_CACHE_TOKEN;
        seek_read(mWeight.get(), mTokenSize, token * mTokenSize);
        int16_t* dst_ptr = reinterpret_cast<int16_t*>(dst + i * mHiddenSize);
        for (int j = 0; j < mHiddenSize; j++) {
            dst_ptr[j * 2]     = 0;
            dst_ptr[j * 2 + 1] = reinterpret_cast<int16_t*>(mWeight.get())[j];
        }
    }
#undef TRY_CACHE_TOKEN
}

}
}
