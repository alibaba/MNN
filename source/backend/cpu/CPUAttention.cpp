//
//  CPUAttention.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <limits>
#include "CPUAttention.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"

#if defined (__aarch64__)
#define FLOAT16_T __fp16
#else
#define FLOAT16_T float
#endif

#define MNN_FLASH_ATTENTION_BLOCK_SIZE 64

namespace MNN {

template <typename T>
void CPUAttention::pack_query(Tensor* query, int8_t* pack_q, int8_t* sum_q, int seq_len, int h, float q_scale) {
    if (mUseGemmInt8) { // Shape of Query: numhead, [seqlen/eP8, headdim/lP8, eP8, lP8]
        mMinQ[h] = query->host<T>()[h * mHeadDim];
        mMaxQ[h] = query->host<T>()[h * mHeadDim];
        for (int i = 0; i < seq_len; i++) {
            T * query_src = query->host<T>() + i * mNumHead * mHeadDim + h * mHeadDim;
            for (int j = 0; j < mHeadDim; j++) {
                mMinQ[h] = ALIMIN(mMinQ[h], query_src[j]);
                mMaxQ[h] = ALIMAX(mMaxQ[h], query_src[j]);
            }
        }
        mQueryScale[h] = (mMaxQ[h] - mMinQ[h]) / 255.0f;
        mQueryZeroPoint[h] = -255.0f * mMinQ[h] / (mMaxQ[h] - mMinQ[h]) - 128.0;
        for (int i = 0; i < seq_len; i++) {
            T * query_src = query->host<T>() + i * mNumHead * mHeadDim + h * mHeadDim;
            float sumQ = 0;
            int out_index = i / eP8;
            int in_index  = i % eP8;
            for (int j = 0; j < mHeadDim; j++) {
                int a = j / lP8;
                int b = j % lP8;
                int quant_res = (int)roundf(query_src[j] / mQueryScale[h] + mQueryZeroPoint[h]);
                sumQ += quant_res;
                *((int8_t*)pack_q + out_index * UP_DIV(mHeadDim, lP8) * eP8 * lP8 + a * eP8 * lP8 + in_index * lP8 + b) = quant_res;
            }
            *((float*)sum_q + out_index * eP8 + in_index) = sumQ * mQueryScale[h];
        }
    }
    else {
        // target: [seq_len/eP, mHeadDim/lP, eP, lP]
        T * query_src = query->host<T>();
        T * query_dst = reinterpret_cast<T*>(pack_q);
        auto stride0 = ROUND_UP(mHeadDim, lP) * eP;
        auto stride1 = eP * lP;
        if (mHeadDim % lP) {
            memset(query_dst, 0, ROUND_UP(mHeadDim, lP) * bytes * ROUND_UP(seq_len, eP));
        }
        for (int i = 0; i < seq_len; i++) {
            int out_index = i / eP;
            int in_index  = i % eP;
            for (int j = 0; j < mHeadDim; j++) {
                query_dst[out_index * stride0 + (j / lP) * stride1 + in_index * lP + (j % lP)] = query_src[i * mNumHead * mHeadDim + h * mHeadDim + j] * q_scale;
            }
        }
    }
}

template <typename T>
void CPUAttention::unpack_QK(float * unpack_qk_dst, int8_t * pack_qk_src, int seq_len, int kv_seq_len) {
    float * dst = unpack_qk_dst;
    T * src = (T *)(pack_qk_src);
    // [kv_seq_len/mPack, seq_len, mPack] -> [seq_len, kv_seq_len]
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_seq_len; j++) {
            int out_index = j / mPack;
            int in_index  = j % mPack;
            dst[i * kv_seq_len + j] = src[out_index * seq_len * mPack + i * mPack + in_index];
        }
    }
}

template <typename T>
static void pack_QK(int8_t * pack_qk_dst, float * qk_src, int seq_len, int kv_seq_len, int eP, int lP, int bytes) {
    T * dst = reinterpret_cast<T*>(pack_qk_dst);
    float * src = reinterpret_cast<float*>(qk_src);
    // [seq_len, kv_seq_len] -> [seq_len/eP, kv_seq_len/lP, eP, lP]
    auto stride0 = ROUND_UP(kv_seq_len, lP) * eP;
    auto stride1 = eP * lP;
    if (kv_seq_len % lP) {
        memset(dst, 0, ROUND_UP(kv_seq_len, lP) * ROUND_UP(seq_len, eP) * bytes);
    }
    for (int i = 0; i < seq_len; i++) {
        int out_index = i / eP;
        int in_index  = i % eP;
        for (int j = 0; j < kv_seq_len; j++) {
            dst[out_index * stride0 + (j / lP) * stride1 + in_index * lP + (j % lP)] = src[i * kv_seq_len + j];
        }
    }
}

template <typename T>
static void mask_QK(float * unpack_qk, int seq_len, int kv_seq_len, float mScale, float min_val, const Tensor* maskTensor, int offset, int startIndx, int processedKvLen) {
    
    int endIndx = startIndx + processedKvLen;
    if (maskTensor == nullptr) {
        for (int i = 0; i < processedKvLen; i++) {
            unpack_qk[i] = unpack_qk[i] * mScale;
        }
        return;
    }
    const int8_t* mask = maskTensor->host<int8_t>();
    halide_type_t htype = maskTensor->getType();
    int maskSize = maskTensor->elementSize();
    
    if (htype == halide_type_of<float>()) {
        // float mask
        T* fpmask_ptr = (T*)mask;
        if (maskSize == seq_len * kv_seq_len) { // sliding attention, mask shape: [seq_len, kv_seq_len]
            for (int i = 0; i < seq_len; ++i) {
                auto unpack_qki = unpack_qk + i * processedKvLen;
                auto fpmask_ptri = fpmask_ptr + i * kv_seq_len;
                for (int j = startIndx; j < endIndx; ++j) {
                    unpack_qki[j - startIndx] = unpack_qki[j - startIndx] * mScale + fpmask_ptri[j];
                }
            }
        } else { // mask shape: [seq_len, seq_len]
            for (int i = 0; i < seq_len; ++i) {
                auto unpack_qki = unpack_qk + i * processedKvLen;
                auto fpmask_ptri = fpmask_ptr + i * seq_len;

                auto notMaskIndx = ALIMIN(endIndx, offset);
                auto stMaskIndx = ALIMAX(startIndx, offset);
                for (int j = startIndx; j < notMaskIndx; ++j) {
                    unpack_qki[j - startIndx] = unpack_qki[j - startIndx] * mScale;
                }
                for (int j = stMaskIndx; j < endIndx; ++j) {
                    unpack_qki[j - startIndx] = unpack_qki[j - startIndx] * mScale + fpmask_ptri[j - offset];
                }
            }
        }
    } else {
        // int mask
        int* mask_ptr = (int*)mask;
        for (int i = 0; i < seq_len * processedKvLen; i++) {
            if (mask_ptr[i / processedKvLen * kv_seq_len + i % processedKvLen]) {
                unpack_qk[i] = unpack_qk[i] * mScale;
            } else {
                unpack_qk[i] = min_val;
            }
        }
    }
}

typedef void(softmaxFunc)(float* softmaxDst, float* input, float* runningMax, float* runningSum, float* updateScale, int outside, int reduceSize);
template <typename T>
static void softmaxQK(float* softmax_qk_addr, float* unpack_qk_addr, float* runningMax, float* runningSum, float* diffScale, const float* sinkPtr, softmaxFunc* sffunc, int seq_len, int kv_seq_len, int headIdx, bool isLastKvBlock) {
    
    // not sliding attention
    if (sinkPtr == nullptr) {
        sffunc(softmax_qk_addr, unpack_qk_addr, runningMax, runningSum, diffScale, seq_len, kv_seq_len);
        return;
    }
    
    float sink = ((T*)sinkPtr)[headIdx];
    if (!runningMax && !runningSum) { // Do not use flash attention
        
        for (int i = 0; i < seq_len; ++i) {
            float exprOffset[4] = {1, 0, -sink, 1.f};
            MNNExp(softmax_qk_addr + i * kv_seq_len, unpack_qk_addr + i * kv_seq_len, exprOffset, kv_seq_len);
            for (int j = 0; j < kv_seq_len; ++j) {
                softmax_qk_addr[i * kv_seq_len + j] /= exprOffset[3];
            }
        }
        return;
    }

    // Use flash attention    
    if (isLastKvBlock) {
        for (int i = 0; i < seq_len; ++i) {
            runningSum[i] += expf(sink - runningMax[i]);
        }
    }
    MNNSoftmax(softmax_qk_addr, unpack_qk_addr, runningMax, runningSum, diffScale, seq_len, kv_seq_len);
}

template <typename T>
static void unpack_QKV(int8_t* pack_qkv, int8_t* unpack_qkv, int mNumHead, int mHeadDim, int mPack, int seq_len) {
    auto src_ptr = reinterpret_cast<T*>(pack_qkv);
    auto dst_ptr = reinterpret_cast<T*>(unpack_qkv);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < mHeadDim; j++) {
            int a = j / mPack;
            int b = j % mPack;
            dst_ptr[i * mNumHead * mHeadDim + j] = src_ptr[a * seq_len * mPack + i * mPack + b];
        }
    }
}

ErrorCode CPUAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend *>(backend())->functions();
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    mPack  = core->pack;
    bytes = core->bytes;
    int qkvQuantOptions = static_cast<CPUBackend *>(backend())->getRuntime()->hint().qkvQuantOption;
    mUseGemmInt8 = (qkvQuantOptions % 8 == 4);
    if (mUseGemmInt8) {
        static_cast<CPUBackend*>(backend())->int8Functions()->MNNGetGemmUnit(&hP8, &lP8, &eP8);
    }
    auto query = inputs[0];
    auto key   = inputs[1];
    int seq_len = query->length(1);
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    mKVCacheManager->onResize(mKvNumHead, mHeadDim);
    if (mUseGemmInt8) {
        mPackQ.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(seq_len, eP8), UP_DIV(mHeadDim, lP8), eP8 * lP8}));
        mSumQ.reset(Tensor::createDevice<int32_t>({mThreadNum, UP_DIV(seq_len, eP8), eP8}));
        mPackQKV.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(mHeadDim, mPack), seq_len, mPack}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mSumQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mSumQ.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackQKV.get(), Backend::DYNAMIC);
        mMinQ.resize(mNumHead);
        mMaxQ.resize(mNumHead);
        mQueryScale.resize(mNumHead);
        mQueryZeroPoint.resize(mNumHead);
    } else {
        mPackQ.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(seq_len, eP), ROUND_UP(mHeadDim, lP), eP * bytes}));
        mPackQKV.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack), seq_len, mPack * bytes}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
        
        // flash attention
        if (qkvQuantOptions / 8 == 1) {
            mRunningMax.reset(Tensor::createDevice<int8_t>({mThreadNum, seq_len * 4}));
            mRunningSum.reset(Tensor::createDevice<int8_t>({mThreadNum, seq_len * 4}));
            mExpfDiffMax.reset(Tensor::createDevice<int8_t>({mThreadNum, seq_len * 4}));
            mTempOut.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack), seq_len, mPack * bytes}));

            backend()->onAcquireBuffer(mRunningMax.get(), Backend::DYNAMIC);
            backend()->onAcquireBuffer(mRunningSum.get(), Backend::DYNAMIC);
            backend()->onAcquireBuffer(mExpfDiffMax.get(), Backend::DYNAMIC);
            backend()->onAcquireBuffer(mTempOut.get(), Backend::DYNAMIC);
        }

        backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackQKV.get(), Backend::DYNAMIC);

        if (qkvQuantOptions / 8 == 1) {
            backend()->onReleaseBuffer(mRunningMax.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mRunningSum.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mExpfDiffMax.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mTempOut.get(), Backend::DYNAMIC);
        }
    }
    return NO_ERROR;
}

ErrorCode CPUAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core  = static_cast<CPUBackend *>(backend())->functions();
    auto qkvQuantOptions = static_cast<CPUBackend *>(backend())->getRuntime()->hint().qkvQuantOption;
    auto query = inputs[0];
    auto key   = inputs[1];
    auto value = inputs[2];
    const Tensor* mask = nullptr;
    int seq_len = query->length(1);
    if (inputs.size() > 3) {
        mask = inputs[3];
    }
    const Tensor* sinks = nullptr;
    if (inputs.size() > 4) {
        sinks = inputs[4];
        MNN_ASSERT(sinks != nullptr);
        MNN_ASSERT(sinks->elementSize() == mNumHead)
    }
    int tileCount = UP_DIV(mNumHead, mThreadNum);
    int group_size = mNumHead / mKvNumHead;
    // reduce the value of 'query' to avoid fp16 overflow
    float mScale = 1.0 / sqrt(mHeadDim);
    float q_scale = 1.0;
    if (bytes == 2) {
        // reduce the value of 'query' to 'query * FP16_QSCALE', avoid fp16 overflow
        FLOAT16_T minValue;
        FLOAT16_T maxValue;
        core->MNNCountMaxMinValue(query->host<float>(), (float*)(&minValue), (float*)(&maxValue), query->elementSize());
        float maxV = maxValue;
        float minV = minValue;
        float absMax = ALIMAX(fabsf(maxV), fabsf(minV));
        if (absMax > 1.0f) {
            q_scale = 1.0f / absMax;
        }
        mScale /= q_scale;
    }

    if (mKVCache && mMeta != nullptr) {
        if (mMeta->previous == mMeta->remove) {
            mKVCacheManager->onClear();
            mKVCacheManager->onAlloc(mMeta->add);
        } else {
            MNN_ASSERT(mMeta->previous == mKVCacheManager->kvLength());
            mKVCacheManager->onRealloc(mMeta);
        }
    } else {
        mKVCacheManager->onClear();
        mKVCacheManager->onAlloc(seq_len);
    }
    // Add the new kv to the kvcache
    mKVCacheManager->onPushBack(key, value);
    int kv_seq_len  = mKVCacheManager->kvLength();
    int max_len = mKVCacheManager->maxLength();
    bool quant_key = mKVCacheManager->config()->mQuantKey;
    bool quant_value = mKVCacheManager->config()->mQuantValue;

    mBlockKV = (qkvQuantOptions / 8 == 1) ? ALIMIN(MNN_FLASH_ATTENTION_BLOCK_SIZE, kv_seq_len) : kv_seq_len;
    int32_t units[2] = {eP, lP};

    // Temporary tensors for intermediate results
    std::shared_ptr<Tensor> unpackQK(Tensor::createDevice<int32_t>({mThreadNum, seq_len, mBlockKV}));
    std::shared_ptr<Tensor> softmMaxQ(Tensor::createDevice<int32_t>({mThreadNum, seq_len, mBlockKV}));
    std::shared_ptr<Tensor> newPackQK(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(seq_len, eP), ROUND_UP(mBlockKV, lP), eP * bytes}));
    std::shared_ptr<Tensor> dequantV(Tensor::createDevice<int8_t>({mKvNumHead, UP_DIV(mHeadDim, hP), kv_seq_len, hP * bytes}));
    // mTempQKBlock.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mBlockKV, mPack), seq_len, mPack * bytes}));
    std::shared_ptr<Tensor> tempQKBlock(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mBlockKV, mPack), seq_len, mPack * bytes}));
    backend()->onAcquireBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onAcquireBuffer(newPackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(tempQKBlock.get(), Backend::STATIC);
    if (quant_value) {
        backend()->onAcquireBuffer(dequantV.get(), Backend::STATIC);
        mKVCacheManager->onDequantValue(dequantV.get());
    }
    const float* sinksPtr = sinks ? sinks->host<float>() : nullptr;
    std::function<void(int)> mCompute = [=](int tId) {
        auto qReordered      = mPackQ->host<int8_t>() + tId * mPackQ->stride(0);
        auto qkPacked     = tempQKBlock->host<int8_t>() + tId * tempQKBlock->stride(0);
        int8_t * sum_q     = nullptr;
        auto qkFlatten   = unpackQK->host<float>() + tId * unpackQK->stride(0);
        auto qkSoftmax  = softmMaxQ->host<float>() + tId * softmMaxQ->stride(0);
        auto qkReordered = newPackQK->host<int8_t>() + tId * newPackQK->stride(0);
        auto qkvPacked    = mPackQKV->host<int8_t>() + tId * mPackQKV->stride(0);
        auto QxK         = quant_key ? core->MNNPackedMatMul_int8 : core->MNNPackedMatMul;
        auto QxK_remain  = quant_key ? core->MNNPackedMatMulRemain_int8 : core->MNNPackedMatMulRemain;

        // Flash Attention
        auto runningMax = mRunningMax ? (float*)(mRunningMax->host<int8_t>() + tId * mRunningMax->stride(0)) : nullptr;
        auto runningSum = mRunningSum ? (float*)(mRunningSum->host<int8_t>() + tId * mRunningSum->stride(0)) : nullptr;
        auto diffScale = mExpfDiffMax ? (float*)(mExpfDiffMax->host<int8_t>() + tId * mExpfDiffMax->stride(0)) : nullptr;
        auto outputPacked = mTempOut ? mTempOut->host<int8_t>() + tId * mTempOut->stride(0) : qkvPacked;
        int  head_index  = tId * tileCount;
        int  kvBlocks = UP_DIV(kv_seq_len, mBlockKV);

        if (mUseGemmInt8) {
            qReordered  = mPackQ->host<int8_t>() + tId * UP_DIV(seq_len, eP8) * UP_DIV(mHeadDim, lP8) * eP8 * lP8;
            sum_q   = mSumQ->host<int8_t>() + tId * UP_DIV(seq_len, eP8) * eP8 * 4;
        }
        for (int h = head_index; h < head_index + tileCount && h < mNumHead; h++) {
            if (runningSum && runningMax) {
                memset(runningSum, 0, mRunningSum->stride(0));
                if (sinksPtr == nullptr) {
                    for (int k = 0; k < seq_len; ++k) {
                        runningMax[k] = -std::numeric_limits<float>::infinity();
                    }
                } else {
                    float sinkVal;
                    if (bytes == 2) {
                        sinkVal = ((FLOAT16_T*)sinksPtr)[h];
                    } else {
                        sinkVal =sinksPtr[h];
                    }
                    for (int k = 0; k < seq_len; ++k) {
                        runningMax[k] = sinkVal;
                    }
                }
            }
            int    kv_h              = h / group_size;
            int8_t * key_addr        = mKVCacheManager->addrOfKey(kv_h);
            int8_t * scale_addr      = mKVCacheManager->addrOfScale(kv_h);
            int8_t * zero_point_addr = mKVCacheManager->addrOfZeroPoint(kv_h);
            int8_t * key_sum_addr    = mKVCacheManager->addrOfKeySum(kv_h);
            int8_t * value_addr      = quant_value ? (dequantV->host<int8_t>() + kv_h * UP_DIV(mHeadDim, hP) * ROUND_UP(kv_seq_len, lP) * hP * bytes) : mKVCacheManager->addrOfValue(kv_h);
            if (mUseGemmInt8) {
                if (bytes == 2) {
                    pack_query<FLOAT16_T>(query, qReordered, sum_q, seq_len, h, q_scale);
                } else {
                    pack_query<float>(query, qReordered, sum_q, seq_len, h, q_scale);
                }
            } else {
                core->MNNAttenPackAndScaleSingleHead((float*)qReordered, (float*)(query->host<int8_t>() + h * mHeadDim * bytes), mHeadDim * mNumHead, &q_scale, units, seq_len, mHeadDim);
            }
            for (int i = 0; i < kvBlocks; ++i) {
                int subKvSeqLen = ALIMIN(mBlockKV, kv_seq_len - i * mBlockKV);
                auto keyPtr = key_addr + i * UP_DIV(mBlockKV, hP) * ROUND_UP(mHeadDim, lP) * hP * bytes;
                auto valuePtr = value_addr + i * UP_DIV(mBlockKV, lP) * hP * lP * bytes;
                // query @ key
                {
                    int loop_e = seq_len / eP;
                    int remain = seq_len % eP;
                    auto qStride0 = ROUND_UP(mHeadDim, lP) * eP * bytes;
                    size_t shapeParameters[7] = {(size_t)eP * bytes, ROUND_UP((size_t)mHeadDim, lP), (size_t)subKvSeqLen, (size_t)seq_len * mPack * bytes, 0, 0, 0};
                    for (int ei = 0 ; ei < loop_e; ei++) {
                        QxK((float*)(qkPacked + (ei * eP * mPack) * bytes), (float*)(qReordered + ei * qStride0), (float*)keyPtr, shapeParameters, nullptr, nullptr, (float*)scale_addr, (float*)zero_point_addr);
                    }
                    QxK_remain((float*)(qkPacked + (loop_e * eP * mPack) * bytes), (float*)(qReordered + loop_e * qStride0), (float*)keyPtr, remain, shapeParameters, nullptr, nullptr, (float*)scale_addr, (float*)zero_point_addr);
                }
                // qk: [kv_seq_len/mPack, seq_len, mPack] -> [seq_len/eP, kv_seq_len, eP]
                {
                    if(bytes == 2) {
                        if (seq_len == 1) {
                            core->MNNLowpToFp32((int16_t*)qkPacked, qkFlatten, seq_len * subKvSeqLen);
                        } else {
                            core->MNNAttenUnpackAndConvertFp16(qkFlatten, (float*)qkPacked, subKvSeqLen, seq_len, mPack);
                        }
                        mask_QK<FLOAT16_T>(qkFlatten, seq_len, kv_seq_len, mScale, std::numeric_limits<float>::lowest(), mask, kv_seq_len - seq_len, i * mBlockKV, subKvSeqLen);
                        softmaxQK<FLOAT16_T>(qkSoftmax, qkFlatten, runningMax, runningSum, diffScale, sinksPtr, core->MNNSoftmax, seq_len, subKvSeqLen, h, i == kvBlocks - 1);
                        core->MNNAttenPackAndConvertFp32((float*)qkReordered, qkSoftmax, units, seq_len, subKvSeqLen);
                    } else {
                        if (seq_len > 1) {
                            int32_t areaOffset[2] = {seq_len, seq_len};
                            core->MNNUnpackCUnitTranspose(qkFlatten, (float*)qkPacked, seq_len, subKvSeqLen, areaOffset);
                        } else {
                            memcpy(qkFlatten, qkPacked, subKvSeqLen * sizeof(float));
                        }
                        mask_QK<float>(qkFlatten, seq_len, kv_seq_len, mScale, std::numeric_limits<float>::lowest(), mask, kv_seq_len - seq_len, i * mBlockKV, subKvSeqLen);
                        softmaxQK<float>(qkSoftmax, qkFlatten, runningMax, runningSum, diffScale, sinksPtr, core->MNNSoftmax, seq_len, subKvSeqLen, h, i == kvBlocks - 1);
                        packKvCache((float*)qkReordered, qkSoftmax, seq_len, subKvSeqLen, eP);
                    }
                }
                // qk @ v
                // TODO: update qkvPacked using diffScale
                size_t shapeParameters[7] = {(size_t)eP * bytes, ROUND_UP((size_t)subKvSeqLen, lP), (size_t)mHeadDim, (size_t)seq_len * mPack * bytes, 0, 0, 0};
                size_t bExtraStride = (UP_DIV(max_len, lP) - UP_DIV(subKvSeqLen + i * mBlockKV, lP) + UP_DIV(i * mBlockKV, lP)) * hP * lP * bytes;
                shapeParameters[5] = quant_value ? 0 : bExtraStride;
                int loop_e = seq_len / eP;
                int remain = seq_len % eP;
                auto qkStride0 = ROUND_UP(subKvSeqLen, lP) * eP * bytes;
                for (int ei = 0 ; ei < loop_e; ei++) {
                    core->MNNPackedMatMul((float*)(qkvPacked + (ei * eP * mPack) * bytes), (float*)(qkReordered + ei * qkStride0), (float*)valuePtr, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                }
                core->MNNPackedMatMulRemain((float*)(qkvPacked + (loop_e * eP * mPack) * bytes), (float*)(qkReordered + loop_e * qkStride0), (float*)valuePtr, remain, shapeParameters, nullptr, nullptr, nullptr, nullptr);

                if (runningMax != nullptr && runningSum != nullptr && diffScale != nullptr) {
                    core->MNNFlashAttentionUpdateBlockOutput((float*)outputPacked, (float*)qkvPacked, diffScale, runningSum, UP_DIV(mHeadDim, mPack), seq_len, mPack, i, kvBlocks, mPackQKV->stride(0) / bytes, bytes);
                }
            }
            // unpack: [head_dim/mPack, seq_len, mPack] -> [seq_len, num_head, head_dim]
            auto dst_ptr = outputs[0]->host<int8_t>() + h * mHeadDim * bytes;
            if (bytes == 2) {
                unpack_QKV<int16_t>((int8_t*)outputPacked, dst_ptr, mNumHead, mHeadDim, mPack, seq_len);
            } else {
                unpack_QKV<float>((int8_t*)outputPacked, dst_ptr, mNumHead, mHeadDim, mPack, seq_len);
            }
            
        }
    };

    MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
        mCompute((int)tId);
    }
    MNN_CONCURRENCY_END();

    backend()->onReleaseBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(tempQKBlock.get(), Backend::STATIC);
    if (quant_value){
        backend()->onReleaseBuffer(dequantV.get(), Backend::STATIC);
    }
    if (!mKVCache) {
        mKVCacheManager->onClear();
    }
    return NO_ERROR;
}

bool CPUAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new CPUAttention(bn, mKVCache);
    tmp->mKVCacheManager = mKVCacheManager;
    *dst = tmp;
    return true;
}

CPUAttention::CPUAttention(Backend *backend, bool kv_cache) : Execution(backend), mKVCache(kv_cache) {
    mMeta = (KVMeta*)(backend->getMetaPtr());
    mPackQ.reset(Tensor::createDevice<float>({1, 1, 1, 1}));
    mPackQKV.reset(Tensor::createDevice<float>({1, 1, 1, 1}));
    MNN::KVCacheManager::KVCacheConfig kvconfig;
    int qkvQuantOptions = static_cast<CPUBackend *>(backend)->getRuntime()->hint().qkvQuantOption;
    kvconfig.mUseInt8Kernel = (qkvQuantOptions % 8 == 4);

    // qkvQuantOption % 8:
    // 0: Do not quantize
    // 1: Only quantize key, use int8 asymmetric quantization
    // 2: Only quantize value, use fp8 quantization
    // 3: quantize both key and value
    // 4: quantize query, key and value, and use gemm int8 kernel to compute K*V

    // qkvQuantOption / 8:
    // 1: use flash attention
    kvconfig.mQuantKey   = (qkvQuantOptions % 8 == 4) || (qkvQuantOptions % 8 == 1) || (qkvQuantOptions % 8 == 3);
    kvconfig.mQuantValue = (qkvQuantOptions % 8 == 4) || (qkvQuantOptions % 8 == 2);
    kvconfig.mKVCacheDir = static_cast<CPUBackend *>(backend)->getRuntime()->hint().kvcacheDirPath;
    kvconfig.mKVCacheSizeLimit = static_cast<CPUBackend *>(backend)->getRuntime()->hint().kvcacheSizeLimit;
    kvconfig.mExpandChunk = 64;
    mKVCacheManager.reset(new KVCacheManager(backend, kvconfig));
}

CPUAttention::~CPUAttention() {

}

class CPUAttentionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_AttentionParam();
        return new CPUAttention(backend, param->kv_cache());
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUAttentionCreator, OpType_Attention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

