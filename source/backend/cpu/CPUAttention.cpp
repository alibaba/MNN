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
#include "compute/TurboQuant.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/BufferAllocator.hpp"
#include "compute/ConvolutionTiledExecutor.hpp"

#if defined (__aarch64__)
#define FLOAT16_T __fp16
#else
#define FLOAT16_T float
#endif



namespace MNN {

template <typename T>
static void _maskQK(float * qkPacked, const float* scale, size_t seqLen, size_t processedKvSeq, int pack, int kvSeqLen, int kvoffset, int padKvSeqLen, const float* sinksPtr, const Tensor* mask, bool quantKey, bool isLowerTriangular) {
    /*
     * FIGURE 1: mask->elementSize() == seqLen * maskStride
     * Context: Cross Attention or Prefill stage (Full Context).
     * Logic:   gapLen = 0. The mask tensor dimensions match the logical QK matrix exactly.
     *          Direct access: mask[row * stride + col]
     * Row\Col   0   1   2   3
     *
     *   0       0   X   X   X    (Can only see Col 0)
     *
     *   1       0   0   X   X    (Can see Col 0, 1)
     *
     *   2       0   0   0   X    (Can see Col 0, 1, 2)
     *
     *   3       0   0   0   0    (Fully visible)
     *
     * Legend:
     *   '0' : Visible (Value = Scale * QK)
     *   'X' : Masked  (Value = -inf)
     */


    /*
     * FIGURE 2: mask->elementSize() != seqLen * maskStride
     * Context: Self-Attention Inference (Decoding stage).
     * Logic:   gapLen = maskStride - seqLen (Right Alignment).
     *          The "Gap" represents History KV Cache, which is implicitly visible.
     *          The Mask Tensor only covers the current sequence window.
     *
     * Example: maskStride (Total KV) = 6
     *          seqLen (Current Q)    = 4
     *          gapLen                = 6 - 4 = 2
     *
     * Structure:
     *   - Cols [0, 1]: "Gap" / History region. Code logic: `if (col < gapLen) continue;`.
     *                  No mask is added, so they remain Visible ('0').
     *   - Cols [2-5]:  "Current" region. Code logic: `mask[col - gapLen]`.
     *
     * Row\Col   0   1   |   2   3   4   5
     *          (Gap)    |   (Mask Tensor Region)
     *
     *   0       0   0   |   0   X   X   X    <-- Mask row 0 applies to Col 2~5
     *                   |
     *   1       0   0   |   0   0   X   X    <-- Mask row 1 applies to Col 2~5
     *                   |
     *   2       0   0   |   0   0   0   X    <-- Mask row 2 applies to Col 2~5
     *                   |
     *   3       0   0   |   0   0   0   0    <-- Mask row 3 applies to Col 2~5
     *
     * Legend:
     *   '0' (Left)  : History KV, implicitly visible (code skips mask addition).
     *   '0' (Right) : Current KV, visible according to Mask Tensor.
     *   'X'         : Masked by Mask Tensor (-inf).
     */

    if (isLowerTriangular && quantKey) {
        return;
    }
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    auto source = (T*)qkPacked;
    float scaleVal = scale[0];
    int gapLen = (mask->elementSize() == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen)) ? 0 : static_cast<int>(kvSeqLen - seqLen);

    auto kvBlockCount = UP_DIV(processedKvSeq, pack);
    auto qkSize = ROUND_UP(processedKvSeq, pack) * seqLen;

    if (isLowerTriangular) {
        for (int i = 0; i < qkSize; ++i) {
            source[i] *= scaleVal;
        }
        return;
    }

    if (mask == nullptr) {
        return;
    }

    auto maskPtr = mask->host<T>();

    // not lower triangular
    auto maskCols = (mask->elementSize() == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen)) ? kvSeqLen + padKvSeqLen : seqLen + padKvSeqLen;
    for (int i = 0; i < kvBlockCount; ++i) {
        T* blockDataPtr = source + (i * seqLen * pack);

        for (int j = 0; j < seqLen; ++j) {
            T* dataPtr = blockDataPtr + (j * pack);
            const T* currentMaskRow = maskPtr + j * maskCols;

            for (int k = 0; k < pack; ++k) {
                float val = (float)dataPtr[k];
                if (!quantKey) {
                    val *= scaleVal;
                    dataPtr[k] = (T)val;
                }
                int currentKvSeqIndx = kvoffset + i * pack + k; // kvoffset=i*mBlockKv

                if (currentKvSeqIndx < gapLen) {
                    continue;
                }
                if (currentKvSeqIndx - gapLen >= maskCols) {
                    break;
                }

                val += (float)currentMaskRow[currentKvSeqIndx - gapLen];
                dataPtr[k] = (T)val;

            }
        }
    }
}

ErrorCode CPUAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto gcore = static_cast<CPUBackend *>(backend())->functions();
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    gcore->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    mPack  = gcore->pack;
    mBytes = gcore->bytes;
    int attentionOption = static_cast<CPUBackend *>(backend())->getRuntime()->hint().attentionOption;
    mUseFlashAttention = (attentionOption / 8 == 1);

    // attentionOption % 8:
    // 0: no quant, 1: K int8, 2: K+V int8, 3: K TQ3, 4: K+V TQ3, 5: K TQ4, 6: K+V TQ4
    int quantMode = attentionOption % 8;
    mKeyQuantMode = KVQuantMode::None;
    mValueQuantMode = KVQuantMode::None;
    if (inputs.size() < 5) {
        switch (quantMode) {
            case 1:
                mKeyQuantMode = KVQuantMode::Int8;
                break;
            case 2:
                mKeyQuantMode = KVQuantMode::Int8;
                mValueQuantMode = KVQuantMode::Int8;
                break;
            case 3:
                mKeyQuantMode = KVQuantMode::TQ3;
                break;
            case 4:
                mKeyQuantMode = KVQuantMode::TQ3;
                mValueQuantMode = KVQuantMode::TQ3;
                break;
            case 5:
                mKeyQuantMode = KVQuantMode::TQ4;
                break;
            case 6:
                mKeyQuantMode = KVQuantMode::TQ4;
                mValueQuantMode = KVQuantMode::TQ4;
                break;
            default:
                break;
        }
        if (mValueQuantMode == KVQuantMode::Int8 && !mUseFlashAttention) {
            mValueQuantMode = KVQuantMode::None;
        }
    }
    static_cast<CPUBackend*>(backend())->int8Functions()->MNNGetGemmUnit(&hP8, &lP8, &eP8);

    auto query = inputs[0];
    auto key   = inputs[1];
    int seqLen = query->length(1);
    int mBlockNum = 1;
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    if (!mIsKVShared) {
        mKVCacheManager->setKVQuantMode(mUseFlashAttention, mKeyQuantMode, mValueQuantMode);
        mKVCacheManager->onResize(mKvNumHead, mHeadDim);
    }

    // Common buffer allocated
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    mPackQKV.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack), seqLen, mPack * mBytes}));
    backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
    if (inputs.size() > 4 || mUseFlashAttention) { // needed by flash attention and sliding attention with sink
        mRunningMax.reset(Tensor::createDevice<int8_t>({mThreadNum, seqLen * 4}));
        mRunningSum.reset(Tensor::createDevice<int8_t>({mThreadNum, seqLen * 4}));
        backend()->onAcquireBuffer(mRunningMax.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mRunningSum.get(), Backend::DYNAMIC);
    }
    if (mUseFlashAttention) { // extra buffer need by flash attention
        mExpfDiffMax.reset(Tensor::createDevice<int8_t>({mThreadNum, seqLen * 4}));
        mTempOut.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack), seqLen, mPack * mBytes}));
        backend()->onAcquireBuffer(mExpfDiffMax.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mTempOut.get(), Backend::DYNAMIC);
    }
    if (mKeyQuantMode == KVQuantMode::TQ3 || mKeyQuantMode == KVQuantMode::TQ4 || mValueQuantMode == KVQuantMode::TQ3 ||
        mValueQuantMode == KVQuantMode::TQ4) {
        // Vec_dot fusion buffers (per thread, shared by TQ3/TQ4):
        // Q_rotated: seqLen * headDim floats (WHT_forward of scaled Q)
        // V_acc_rotated: headDim floats (accumulator in rotated domain)
        // weights: blockKV floats (extracted softmax weights for one query)
        int blockKV = mUseFlashAttention ? MNN_FLASH_ATTENTION_BLOCK_SIZE : (seqLen + 64);
        int qRotatedSize = seqLen * mHeadDim * sizeof(float);
        int vAccSize = mHeadDim * sizeof(float);
        int weightsSize = blockKV * sizeof(float);
        mTQ3DequantBuf.reset(Tensor::createDevice<int8_t>({mThreadNum, qRotatedSize + vAccSize + weightsSize}));
        backend()->onAcquireBuffer(mTQ3DequantBuf.get(), Backend::DYNAMIC);
    }
    if (mKeyQuantMode == KVQuantMode::Int8) {
        int outterSeqLen = UP_DIV(seqLen, eP8);
        int outterHeadDim = UP_DIV(mHeadDim, lP8);

        size_t packedQSize = 0;
        if (outterSeqLen > 0) {
            int fullSeqBlocks = (seqLen / eP8);
            packedQSize += (size_t)fullSeqBlocks * outterHeadDim * eP8 * lP8;

            int lastEUnit = seqLen % eP8;
            if (lastEUnit != 0) {
                packedQSize += (size_t)outterHeadDim * lastEUnit * lP8;
            }
        }
        mPackQ.reset(Tensor::createDevice<int8_t>({mNumHead, (int32_t)packedQSize}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);

        mSumQ = bufferAlloc->alloc(mThreadNum * ROUND_UP(seqLen, eP8) * mBlockNum * sizeof(int32_t));
        mQueryScale = bufferAlloc->alloc(mNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryZeroPoint = bufferAlloc->alloc(mNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryQuantZero = bufferAlloc->alloc(mNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryQuantScale = bufferAlloc->alloc(mNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQuantQuery = bufferAlloc->alloc(seqLen * mNumHead * UP_DIV(mHeadDim, gcore->pack) * gcore->pack);

        if (mBlockNum > 1) {
            mAccumBuffer = bufferAlloc->alloc(eP8 * hP8 * mThreadNum * QUANT_INFO_BYTES);
            if (mAccumBuffer.invalid()) {
                return OUT_OF_MEMORY;
            }
        }

        if (mSumQ.invalid() || mQueryScale.invalid() || mQueryQuantZero.invalid() || mQueryZeroPoint.invalid() || mQueryQuantScale.invalid() || mQuantQuery.invalid()) {
            return OUT_OF_MEMORY;
        }

        // post parameters for int8 gemm
        mGemmRelu.reset(2 * sizeof(int32_t));
        if (!mGemmRelu.get()) {
            MNN_ERROR("Allocate mGemmRelu buffer failed in CPU Attention");
            return OUT_OF_MEMORY;
        }
        ((float*)mGemmRelu.get())[0] = -std::numeric_limits<float>().max();
        ((float*)mGemmRelu.get())[1] = std::numeric_limits<float>().max();
        if (mBytes == 2) {
            gcore->MNNFp32ToLowp((float*)mGemmRelu.get(), reinterpret_cast<int16_t*>(mGemmRelu.get()), 2);
        }

        // GemmInt8 kernels
        if (mBytes == 4) {
            mInt8GemmKernel = core->Int8GemmKernel;
        } else {
            mInt8GemmKernel = core->MNNGemmInt8AddBiasScale_Unit_FP16;
        }

        if (mValueQuantMode == KVQuantMode::Int8) {
            mQuantQK = bufferAlloc->alloc(mThreadNum * eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, mPack));
            mQKScale = bufferAlloc->alloc(eP8 * QUANT_INFO_BYTES);
            mQKBias = bufferAlloc->alloc(eP8 * QUANT_INFO_BYTES);
            mSumQK = bufferAlloc->alloc(mThreadNum * eP8 * QUANT_INFO_BYTES);

            if (mQuantQK.invalid() || mQKScale.invalid() || mQKBias.invalid() || mSumQK.invalid()) {
                return OUT_OF_MEMORY;
            }
        }
    } else {
        mPackQ.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(seqLen, eP), ROUND_UP(mHeadDim, lP), eP * mBytes}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
    }

    // release tensor
    backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackQKV.get(), Backend::DYNAMIC);

    if (inputs.size() > 4 || mUseFlashAttention) {
        backend()->onReleaseBuffer(mRunningMax.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mRunningSum.get(), Backend::DYNAMIC);
    }
    if (mUseFlashAttention) {
        backend()->onReleaseBuffer(mExpfDiffMax.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mTempOut.get(), Backend::DYNAMIC);
    }
    if (mKeyQuantMode == KVQuantMode::TQ3 || mKeyQuantMode == KVQuantMode::TQ4 || mValueQuantMode == KVQuantMode::TQ3 ||
        mValueQuantMode == KVQuantMode::TQ4) {
        backend()->onReleaseBuffer(mTQ3DequantBuf.get(), Backend::DYNAMIC);
    }

    // release memchunk
    if (mKeyQuantMode == KVQuantMode::Int8) {
        bufferAlloc->free(mSumQ);
        bufferAlloc->free(mQueryScale);
        bufferAlloc->free(mQueryZeroPoint);
        bufferAlloc->free(mQueryQuantScale);
        bufferAlloc->free(mQueryQuantZero);
        bufferAlloc->free(mQuantQuery);
        if (mBlockNum > 1) {
            bufferAlloc->free(mAccumBuffer);
        }
        if (mValueQuantMode == KVQuantMode::Int8) {
            bufferAlloc->free(mQuantQK);
            bufferAlloc->free(mQKScale);
            bufferAlloc->free(mQKBias);
            bufferAlloc->free(mSumQK);
        }
    }

    // Only allocated for quantized Q&K
    if (mKeyQuantMode == KVQuantMode::Int8) {
        if (mBytes == 4) {
            mQuantFunc = core->MNNFloat2Int8;
        } else {
            mQuantFunc = core->DynamicQuanInput_ARM82;
        }
    }
    return NO_ERROR;
}

ErrorCode CPUAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto gcore  = static_cast<CPUBackend *>(backend())->functions();
    auto core   = static_cast<CPUBackend*>(backend())->int8Functions();
    auto query = inputs[0];
    auto key   = inputs[1];
    auto value = inputs[2];
    int seqLen = query->length(1);
    const Tensor* mask = nullptr;
    if (inputs.size() > 3) {
        mask = inputs[3];
    }
    const Tensor* sinks = nullptr;
    if (inputs.size() > 4) {
        sinks = inputs[4];
        MNN_ASSERT(sinks != nullptr);
        MNN_ASSERT(sinks->elementSize() == mNumHead)
    }
    int numHeadDiv = UP_DIV(mNumHead, mThreadNum);
    int group_size = mNumHead / mKvNumHead;
    // reduce the value of 'query' to avoid fp16 overflow
    float mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0 / sqrt(mHeadDim));
    float q_scale = 1.0;
    if (mBytes == 2 && mKeyQuantMode != KVQuantMode::Int8) {
        // reduce the value of 'query' to 'query * FP16_QSCALE', avoid fp16 overflow
        FLOAT16_T minValue;
        FLOAT16_T maxValue;
        gcore->MNNCountMaxMinValue(query->host<float>(), (float*)(&minValue), (float*)(&maxValue), query->elementSize());
        float maxV = maxValue;
        float minV = minValue;
        float absMax = ALIMAX(fabsf(maxV), fabsf(minV));
        if (absMax > 1.0f) {
            q_scale = 1.0f / absMax;
        }
        mScale /= q_scale;
    }
    int insertLen = seqLen;

    if (!mIsKVShared) {
        if (mKVCache && mMeta != nullptr) {
            if (mMeta->previous == mMeta->remove) {
                mKVCacheManager->onClear();
                mKVCacheManager->onAlloc(mMeta, seqLen);
            } else {
                MNN_ASSERT(mMeta->previous == mKVCacheManager->kvLength());
                mKVCacheManager->onRealloc(mMeta);
            }
            insertLen = (int)mMeta->add;
        } else {
            mKVCacheManager->onClear();
            mKVCacheManager->onAlloc(mMeta, seqLen);
        }
        // Add the new kv to the kvcache
        mKVCacheManager->onUpdateKV(key, value, (int)insertLen);
    } else {
        // Shared layer: KV cache is shared via onClone, skip KV update
        insertLen = (int)mMeta->add;
    }

    if (mUseFlashAttention) {
        mBlockKV = ALIMIN(MNN_FLASH_ATTENTION_BLOCK_SIZE, mKVCacheManager->kvLength());
    } else {
        mBlockKV = mKVCacheManager->kvLength();
    }

    // Constant Initialization
    auto padSeqLength = seqLen - insertLen;
    seqLen = insertLen;
    int kvSeqLen  = mKVCacheManager->kvLength();
    int maxLen = mKVCacheManager->maxLength();
    int32_t units[2] = {eP, lP};
    const float* sinksPtr = sinks ? sinks->host<float>() : nullptr;
    int kvValidOffset = kvSeqLen - seqLen; // reuse_kv=true or decode, kvValidOffset>0

    // Temporary tensors for intermediate results
    std::shared_ptr<Tensor> unpackQK(Tensor::createDevice<int32_t>({mThreadNum, seqLen, mBlockKV}));
    std::shared_ptr<Tensor> softmMaxQ(Tensor::createDevice<int32_t>({mThreadNum, seqLen, ROUND_UP(mBlockKV, mPack)})); // [mBlockKV/mPack, seqLen, mPack ]
    std::shared_ptr<Tensor> newPackQK;
    if (mValueQuantMode != KVQuantMode::Int8) {
        newPackQK.reset(Tensor::createDevice<int8_t>({mThreadNum, eP * ROUND_UP(mBlockKV, lP) * mBytes}));
    } else {
        newPackQK.reset(Tensor::createDevice<int8_t>({mThreadNum, eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8)}));
    }
    std::shared_ptr<Tensor> mTempQKBlock(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mBlockKV, mPack), seqLen, mPack * mBytes}));
    backend()->onAcquireBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onAcquireBuffer(newPackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(mTempQKBlock.get(), Backend::STATIC);

    // Quantize Q and initialize bias 0
    if (mKeyQuantMode == KVQuantMode::Int8) {
        mGemmBias.reset(ROUND_UP(ALIMAX(mBlockKV, mHeadDim), hP8) * QUANT_INFO_BYTES);
        if (!mGemmBias.get()) {
            MNN_ERROR("Allocate bias buffer failed in CPU Attention\n");
            return OUT_OF_MEMORY;
        }
        memset(mGemmBias.get(), 0, ROUND_UP(ALIMAX(mBlockKV, mHeadDim), hP8) * QUANT_INFO_BYTES);

        // Q: [seqLen,numHead,headDim]
        // maxQ, minQ: [seqLen,numHead]
        // scaleQ, zeroQ: [numHead, seqLen]
        // quantQ: [seqLen,numHead,headDim]
        auto queryPtr = query->host<int8_t>();
        int divPart = UP_DIV(seqLen * mNumHead, mThreadNum);
        MNN_CONCURRENCY_BEGIN (tId, mThreadNum) {
            size_t info[9] = {1, (size_t)mHeadDim, 1, 1, 1, 1, 1, 1, 0};
            auto remainLu = seqLen * mNumHead - tId * divPart;
            if (remainLu > 0) {
                remainLu = ALIMIN(divPart, remainLu);
                for (int i = tId * divPart; i < tId * divPart + remainLu; ++i) {

                    // address
                    auto srcFloatPtr = (float*)(queryPtr + i * mHeadDim * mBytes);
                    auto dstInt8Ptr = (int8_t*)(mQuantQuery.ptr() + i * mHeadDim);
                    auto quantScalePtr = (float*)(mQueryQuantScale.ptr() + i * QUANT_INFO_BYTES);
                    auto quantZeroPtr = (float*)(mQueryQuantZero.ptr() + i * QUANT_INFO_BYTES);

                    // scaleQ, zeroQ, [seqLen,numHead]->[numHead,seqLen]
                    int indexQ = (i / mNumHead) + (i % mNumHead) * seqLen;
                    auto scalePtr = (float*)(mQueryScale.ptr() + indexQ * QUANT_INFO_BYTES);
                    auto zeroPtr = (float*)(mQueryZeroPoint.ptr() + indexQ * QUANT_INFO_BYTES);


                    // compute the quant/dequant scale/bias
                    gcore->MNNAsyQuantInfo(scalePtr, zeroPtr, quantScalePtr, quantZeroPtr, nullptr, nullptr, srcFloatPtr, info);
                    scalePtr[0] *= mScale;
                    zeroPtr[0] *= mScale;

                    // quantize the float query to int8_t query
                    mQuantFunc(srcFloatPtr, dstInt8Ptr, UP_DIV(mHeadDim, gcore->pack), quantScalePtr, -128, 127, quantZeroPtr, 0);
                }
            }
        } MNN_CONCURRENCY_END();

        // source int8_t query: [seqLen,numHead,headDim]
        // dest int8_t query: [numHead,seqLen/eP,headDim/lP,eP,lP]

        int outterSeqLen = UP_DIV(seqLen, eP8);
        int outterHeadDim = UP_DIV(mHeadDim, lP8);
        size_t outputOffset = 0;

        const int8_t* src_base_ptr = (const int8_t*)mQuantQuery.ptr();
        int8_t* dst_base_ptr = mPackQ->host<int8_t>();

        for (int h = 0; h < mNumHead; ++h) {
            for (int seqBlock = 0; seqBlock < outterSeqLen; ++seqBlock) {
                int seqBase = seqBlock * eP8;
                int eunit = std::min(eP8, seqLen - seqBase);
                size_t currentSeqBlockSize = (size_t)outterHeadDim * eunit * lP8;

                for (int dimBlock = 0; dimBlock < outterHeadDim; ++dimBlock) {
                    int dimBase = dimBlock * lP8;
                    int headDimRemain = mHeadDim - dimBase;
                    int copyLen = std::min(lP8, headDimRemain);

                    if (copyLen <= 0) {
                        continue;
                    }

                    int8_t* dst_block_ptr = dst_base_ptr +
                                          outputOffset +
                                          (size_t)dimBlock * (eunit * lP8);

                    const size_t src_row_stride = (size_t)mNumHead * mHeadDim;

                    for (int seqLocal = 0; seqLocal < eunit; ++seqLocal) {
                        int innerSeq = seqBase + seqLocal;

                        const int8_t* src_row_ptr = src_base_ptr +
                                                    (size_t)innerSeq * src_row_stride +
                                                    (size_t)h * mHeadDim +
                                                    dimBase;

                        int8_t* dst_row_ptr = dst_block_ptr + seqLocal * lP8;

                        std::memcpy(dst_row_ptr, src_row_ptr, copyLen);
                    }
                    if (copyLen < lP8) {
                        for (int seqLocal = 0; seqLocal < eunit; ++seqLocal) {
                            int8_t* dst_pad_ptr = dst_block_ptr + seqLocal * lP8 + copyLen;
                            std::memset(dst_pad_ptr, 0, lP8 - copyLen);
                        }
                    }
                }
                outputOffset += currentSeqBlockSize;
            }
        } // Finish quantize Q

        if (mValueQuantMode == KVQuantMode::Int8) {
            auto scalePtr = (float*)(mQKScale.ptr());
            auto zeroPtr = (float*)(mQKBias.ptr());
            for (int k = 0; k < eP8; ++k) {
                scalePtr[k] = 1.f / 255.f;
#ifdef MNN_USE_SSE
                zeroPtr[k] =0;
#else
                zeroPtr[k] = 128.f / 255.f;
#endif
            }
        }
    }

    std::function<void(int)> mCompute = [=](int tId) {
        int8_t* qReordered = nullptr;
        auto qkPacked     = mTempQKBlock->host<int8_t>() + tId * mTempQKBlock->stride(0);
        auto qkFlatten   = unpackQK->host<float>() + tId * unpackQK->stride(0);
        auto qkSoftmax  = softmMaxQ->host<float>() + tId * softmMaxQ->stride(0);
        auto qkReordered = newPackQK->host<int8_t>() + tId * newPackQK->stride(0);
        auto qkvPacked    = mPackQKV->host<int8_t>() + tId * mPackQKV->stride(0);
        int  headIndex  = tId * numHeadDiv;
        int  headsToCompute = ALIMIN(numHeadDiv, mNumHead - headIndex);

        // Flash Attention
        auto runningMax = mRunningMax ? (float*)(mRunningMax->host<int8_t>() + tId * mRunningMax->stride(0)) : nullptr;
        auto runningSum = mRunningSum ? (float*)(mRunningSum->host<int8_t>() + tId * mRunningSum->stride(0)) : nullptr;
        auto diffScale = mExpfDiffMax ? (float*)(mExpfDiffMax->host<int8_t>() + tId * mExpfDiffMax->stride(0)) : nullptr;
        auto outputPacked = mTempOut ? mTempOut->host<int8_t>() + tId * mTempOut->stride(0) : qkvPacked;
        
        int  kvBlocks = UP_DIV(kvSeqLen, mBlockKV);

        bool isLowerTriangular = (mask == nullptr);
        if (mask != nullptr && mask->shape().empty()) {
            if (mBytes == 2) {
                auto maskPtr = mask->host<FLOAT16_T>();
                if (maskPtr[0] < 1e-6) {
                    isLowerTriangular = true;
                }
            } else {
                auto maskPtr = mask->host<float>();
                if (maskPtr[0] < 1e-6f) {
                    isLowerTriangular = true;
                }
            }
        }
        bool useMaskInSoftmax = (isLowerTriangular && sinksPtr == nullptr);

        QuanPostTreatParameters gemmParam4QxK, gemmParam4QKxV; // used by int8 gemm, allocated per thread.
        SumByAxisParams sumParams4QxK, sumParams4QKxV = {};
        float* qSumAddr = nullptr;
        float* qScale = nullptr;
        float* qBias = nullptr;
        float* accumbuff = nullptr;
        int32_t unitColBufferSize = 0;
        if (mKeyQuantMode == KVQuantMode::Int8) {
            // parameters shared by all mBlockKV
            gemmParam4QxK.blockNum = mBlockNum;
            gemmParam4QxK.biasFloat = reinterpret_cast<float*>(mGemmBias.get());
            gemmParam4QxK.useInt8 = 0;
            gemmParam4QxK.fp32minmax = reinterpret_cast<float*>(mGemmRelu.get());

            sumParams4QxK.oneScale = 0;
            sumParams4QxK.SRC_UNIT = lP8;
            sumParams4QxK.blockNum = mBlockNum;
            sumParams4QxK.DST_XUNIT = eP8;
            sumParams4QxK.inputBlock = 0;
            sumParams4QxK.kernelxy = 1;
            // fixed
            sumParams4QxK.LU = UP_DIV(mHeadDim, lP8);
            sumParams4QxK.unitColBufferSize = ROUND_UP(mHeadDim, lP8) * eP8;
            sumParams4QxK.kernelCountUnitDouble = UP_DIV(mHeadDim, lP8);
            sumParams4QxK.valid = mHeadDim % lP8;


            if (mBlockNum > 1) {
                accumbuff = (float*)(mAccumBuffer.ptr() + tId * eP8 * hP8 * QUANT_INFO_BYTES);
            }
            unitColBufferSize = eP8 * ROUND_UP(mHeadDim, lP8);

            if (mValueQuantMode == KVQuantMode::Int8) {
                gemmParam4QKxV.blockNum = mBlockNum;
                gemmParam4QKxV.biasFloat = reinterpret_cast<float*>(mGemmBias.get());
                gemmParam4QKxV.useInt8 = 0;
                gemmParam4QKxV.fp32minmax = reinterpret_cast<float*>(mGemmRelu.get());
                gemmParam4QKxV.inputScale = (float*)mQKScale.ptr();
                gemmParam4QKxV.inputBias = (float*)mQKBias.ptr();
                gemmParam4QKxV.srcKernelSum = (float*)(mSumQK.ptr() + tId * eP8 * QUANT_INFO_BYTES);

                sumParams4QKxV.oneScale = 0;
                sumParams4QKxV.SRC_UNIT = lP8;
                sumParams4QKxV.blockNum = mBlockNum;
                sumParams4QKxV.DST_XUNIT = eP8;
                sumParams4QKxV.inputBlock = 0;
                sumParams4QKxV.kernelxy = 1;
                sumParams4QKxV.unitColBufferSize = ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8) * eP8;
                sumParams4QKxV.kernelCountUnitDouble = UP_DIV(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8);
            }
        }

        size_t vstride0 = ROUND_UP(mHeadDim, hP) * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP);
        if (mValueQuantMode == KVQuantMode::Int8) {
            vstride0 = (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mKVCacheManager->getFlashAttentionBlockKv(), lP8) + 2 * QUANT_INFO_BYTES * mBlockNum * ROUND_UP(mHeadDim, hP8));
        }

        // use for V
        float const* srcPtr[1];
        // only used for quantized V
        float vQuantScale[1] = {255.f};
        float vQuantBias[1] = {-128.f};
        int32_t infoInt8V[5];
        infoInt8V[0] = 1;       // number
        infoInt8V[2] = static_cast<int32_t>(sumParams4QKxV.unitColBufferSize);
        infoInt8V[3] = 1;       // stride
        int32_t elInt8V[4] = {eP8, ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8), 0, 0};

        // only used for float V
        int32_t infoFloatV[4];
        infoFloatV[0] = 1;      // number
        infoFloatV[1] = seqLen; // eReal
        infoFloatV[3] = 1;      // stride
        int32_t elFloatV[4] = {seqLen, ROUND_UP(kvSeqLen, lP), 0, 0};

        int offset[2] = {seqLen, mNumHead * mHeadDim};

        for (int h = headIndex; h < headIndex + headsToCompute; h++) {
            // Prepare for flash attention
            if (runningSum && runningMax) {
                if (sinksPtr == nullptr) {
                    memset(runningSum, 0, mRunningSum->stride(0));
                    for (int k = 0; k < seqLen; ++k) {
                        runningMax[k] = std::numeric_limits<float>::lowest();
                    }
                } else {
                    for (int k = 0; k < seqLen; ++k) {
                        runningSum[k] = 1.f; // exp(sink-sink)
                    }
                    float sinkVal;
                    if (mBytes == 2) {
                        sinkVal = ((FLOAT16_T*)sinksPtr)[h];
                    } else {
                        sinkVal = sinksPtr[h];
                    }
                    for (int k = 0; k < seqLen; ++k) {
                        runningMax[k] = sinkVal;
                    }
                }
            }

            // Compute the current addresses
            int    kvHeadIndex = h / group_size;
            int8_t * keyAddr   = mKVCacheManager->addrOfKey(kvHeadIndex);
            int8_t * keySum    = mKVCacheManager->addrOfKeySum(kvHeadIndex);
            int8_t * valueAddr = mKVCacheManager->addrOfValue(kvHeadIndex);
            float* valueSum    = (float*)mKVCacheManager->addrOfValueSum(kvHeadIndex);

            // Get packed Q
            if (mKeyQuantMode != KVQuantMode::Int8) {
                qReordered      = mPackQ->host<int8_t>() + tId * mPackQ->stride(0);
                gcore->MNNAttenPackAndScaleSingleHead((float*)qReordered, (float*)(query->host<int8_t>() + h * mHeadDim * mBytes), mHeadDim * mNumHead, &q_scale, units, seqLen, mHeadDim);
            } else {
                qReordered = mPackQ->host<int8_t>() + h * mPackQ->stride(0);
                qSumAddr = (float*)(mSumQ.ptr() + tId * ROUND_UP(seqLen, eP8) * mBlockNum * QUANT_INFO_BYTES);
                qScale = (float*)(mQueryScale.ptr() + h * seqLen * mBlockNum * QUANT_INFO_BYTES);
                qBias = (float*)(mQueryZeroPoint.ptr() + h * seqLen * mBlockNum * QUANT_INFO_BYTES);
                gcore->MNNSumByAxisLForMatmul_A(qSumAddr, qReordered, qScale, seqLen, sumParams4QxK);
            }

            // Start computing
            for (int i = 0; i < kvBlocks; ++i) {
                int subKvSeqLen = ALIMIN(mBlockKV, kvSeqLen - i * mBlockKV);

                // 1. query @ key
                if (mKeyQuantMode == KVQuantMode::TQ3) {
                    // Vec_dot fusion: Q_rotated · TQ3_compressed_K directly (no dequant buffer)
                    // Q_rotated = WHT_forward(Q * q_scale) computed once per KV block iteration
                    int tq3BytesPerSeq = (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
                    int numBlocks = mHeadDim / TQ3_BLOCK_SIZE;
                    auto tq3Buf = mTQ3DequantBuf->host<int8_t>() + tId * mTQ3DequantBuf->stride(0);
                    auto qRotated = (float*)tq3Buf; // seqLen * headDim floats

                    // Pre-rotate Q vectors (only on first KV block)
                    if (i == 0) {
                        float qScale = 1.0f / sqrtf((float)mHeadDim);
                        auto queryBase = (float*)(query->host<int8_t>() + h * mHeadDim * mBytes);
                        int qStride = mHeadDim * mNumHead; // stride between seq positions
                        for (int q = 0; q < seqLen; q++) {
                            for (int b = 0; b < numBlocks; b++) {
                                float scaled[TQ3_BLOCK_SIZE];
                                if (mBytes == 2) {
                                    auto src16 = (FLOAT16_T*)(query->host<int8_t>() + h * mHeadDim * mBytes) +
                                                 q * mHeadDim * mNumHead;
                                    for (int d = 0; d < TQ3_BLOCK_SIZE; d++) {
                                        scaled[d] = (float)src16[b * TQ3_BLOCK_SIZE + d] * qScale;
                                    }
                                } else {
                                    auto srcF = queryBase + q * qStride;
                                    for (int d = 0; d < TQ3_BLOCK_SIZE; d++) {
                                        scaled[d] = srcF[b * TQ3_BLOCK_SIZE + d] * qScale;
                                    }
                                }
                                tq3_wht_forward_32(qRotated + q * mHeadDim + b * TQ3_BLOCK_SIZE, scaled);
                            }
                        }
                    }

                    // Compute QK scores directly: score[q][s] = Σ_b vec_dot_block(Q_rot, K_tq3)
                    // Output format: qkPacked [kvSeq/mPack, seqLen, mPack]
                    for (int s = 0; s < subKvSeqLen; s++) {
                        int seqIdx = i * mBlockKV + s;
                        auto kPtr = (uint8_t*)keyAddr + seqIdx * tq3BytesPerSeq;
                        for (int q = 0; q < seqLen; q++) {
                            float score = 0.0f;
                            auto qr = qRotated + q * mHeadDim;
                            for (int b = 0; b < numBlocks; b++) {
                                score += tq3_vec_dot_block(qr + b * TQ3_BLOCK_SIZE, kPtr + b * TQ3_BYTES_PER_BLOCK);
                            }
                            // Write to [kvSeq/mPack, seqLen, mPack] format
                            int packIdx = (s / mPack) * seqLen * mPack + q * mPack + s % mPack;
                            if (mBytes == 2) {
                                ((FLOAT16_T*)qkPacked)[packIdx] = (FLOAT16_T)score;
                            } else {
                                ((float*)qkPacked)[packIdx] = score;
                            }
                        }
                    }
                } else if (mKeyQuantMode == KVQuantMode::TQ4) {
                    // Vec_dot fusion for TQ4 (4-bit): same logic as TQ3, different bytesPerBlock + functions
                    int tq4BytesPerSeq = (mHeadDim / TQ4_BLOCK_SIZE) * TQ4_BYTES_PER_BLOCK;
                    int numBlocks = mHeadDim / TQ4_BLOCK_SIZE;
                    auto tq4Buf = mTQ3DequantBuf->host<int8_t>() + tId * mTQ3DequantBuf->stride(0);
                    auto qRotated = (float*)tq4Buf;

                    if (i == 0) {
                        float qScale = 1.0f / sqrtf((float)mHeadDim);
                        for (int q = 0; q < seqLen; q++) {
                            for (int b = 0; b < numBlocks; b++) {
                                float scaled[TQ4_BLOCK_SIZE];
                                if (mBytes == 2) {
                                    auto src16 = (FLOAT16_T*)(query->host<int8_t>() + h * mHeadDim * mBytes) +
                                                 q * mHeadDim * mNumHead;
                                    for (int d = 0; d < TQ4_BLOCK_SIZE; d++)
                                        scaled[d] = (float)src16[b * TQ4_BLOCK_SIZE + d] * qScale;
                                } else {
                                    auto srcF = (float*)(query->host<int8_t>() + h * mHeadDim * mBytes) +
                                                q * mHeadDim * mNumHead;
                                    for (int d = 0; d < TQ4_BLOCK_SIZE; d++)
                                        scaled[d] = srcF[b * TQ4_BLOCK_SIZE + d] * qScale;
                                }
                                tq3_wht_forward_32(qRotated + q * mHeadDim + b * TQ4_BLOCK_SIZE, scaled);
                            }
                        }
                    }

                    for (int s = 0; s < subKvSeqLen; s++) {
                        int seqIdx = i * mBlockKV + s;
                        auto kPtr = (uint8_t*)keyAddr + seqIdx * tq4BytesPerSeq;
                        for (int q = 0; q < seqLen; q++) {
                            float score = 0.0f;
                            auto qr = qRotated + q * mHeadDim;
                            for (int b = 0; b < numBlocks; b++) {
                                score += tq4_vec_dot_block(qr + b * TQ4_BLOCK_SIZE, kPtr + b * TQ4_BYTES_PER_BLOCK);
                            }
                            int packIdx = (s / mPack) * seqLen * mPack + q * mPack + s % mPack;
                            if (mBytes == 2) {
                                ((FLOAT16_T*)qkPacked)[packIdx] = (FLOAT16_T)score;
                            } else {
                                ((float*)qkPacked)[packIdx] = score;
                            }
                        }
                    }
                } else if (mKeyQuantMode != KVQuantMode::Int8) {
                    auto keyPtr = keyAddr + i * UP_DIV(mBlockKV, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
                    int loop_e = seqLen / eP;
                    int remain = seqLen % eP;
                    auto qStride0 = ROUND_UP(mHeadDim, lP) * eP * mBytes;
                    size_t shapeParameters[7] = {(size_t)eP * lP *  mBytes, ROUND_UP((size_t)mHeadDim, lP), (size_t)subKvSeqLen, (size_t)seqLen * mPack * mBytes, 0, 0, 0};
                    for (int ei = 0 ; ei < loop_e; ei++) {
                        gcore->MNNPackedMatMul((float*)(qkPacked + (ei * eP * mPack) * mBytes), (float*)(qReordered + ei * qStride0), (float*)keyPtr, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                    if (remain > 0) {
                        gcore->MNNPackedMatMulRemain((float*)(qkPacked + (loop_e * eP * mPack) * mBytes), (float*)(qReordered + loop_e * qStride0), (float*)keyPtr, remain, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                } else {
                    auto eRemain = seqLen;
                    auto srcInt8 = qReordered;
                    auto dstInt8 = qkPacked;
                    auto keyPtr = keyAddr + i * UP_DIV(mBlockKV, hP8) * (ROUND_UP(mHeadDim, lP8) * hP8 + 2 * hP8 * QUANT_INFO_BYTES);
                    gemmParam4QxK.weightKernelSum = (float*)(keySum + i * mBlockKV * QUANT_INFO_BYTES);
                    gemmParam4QxK.inputScale   = qScale;
                    gemmParam4QxK.inputBias    = qBias;
                    gemmParam4QxK.srcKernelSum = qSumAddr;
                    while (eRemain > 0) {
                        auto eSize = ALIMIN(eP8, eRemain);
                        mInt8GemmKernel(dstInt8, srcInt8, keyPtr, UP_DIV(mHeadDim, lP8), mBytes * seqLen * mPack, UP_DIV(subKvSeqLen, mPack), &gemmParam4QxK, eSize);
                        eRemain -= eP8;
                        gemmParam4QxK.inputScale += eP8;
                        gemmParam4QxK.inputBias += eP8;
                        gemmParam4QxK.srcKernelSum += eP8;
                        srcInt8 += unitColBufferSize;
                        dstInt8 += eP8 * mPack * mBytes;
                        if (mBlockNum > 1) {
                            memset(accumbuff, 0, eP8 * hP8 * QUANT_INFO_BYTES);
                            gemmParam4QxK.accumBuffer = accumbuff;
                        }
                    }
                }
                // 2. softmax scores, softmax src/dst shape: [kv_seq_len/mPack, seq_len, mPack]
                {
                    if (mKeyQuantMode != KVQuantMode::Int8 || isLowerTriangular == false || sinksPtr != nullptr) {
                        if (mBytes == 2) {
                            _maskQK<FLOAT16_T>((float*)qkPacked, &mScale, seqLen, subKvSeqLen, mPack, kvSeqLen,
                                               i * mBlockKV, padSeqLength, sinksPtr, mask,
                                               (mKeyQuantMode == KVQuantMode::Int8), isLowerTriangular);
                        } else {
                            _maskQK<float>((float*)qkPacked, &mScale, seqLen, subKvSeqLen, mPack, kvSeqLen,
                                           i * mBlockKV, padSeqLength, sinksPtr, mask,
                                           (mKeyQuantMode == KVQuantMode::Int8), isLowerTriangular);
                        }
                    }
                    gcore->MNNSoftmax(qkSoftmax, (float*)qkPacked, runningMax, runningSum, diffScale, seqLen, subKvSeqLen, i * mBlockKV, kvValidOffset, mPack, useMaskInSoftmax);
                }
                // 3. qk @ v
                auto qkStride0 = ROUND_UP(subKvSeqLen, lP) * eP * mBytes;
                auto rowStart = (!isLowerTriangular || i * mBlockKV < kvValidOffset)? 0 : (i * mBlockKV - kvValidOffset);

                if (mValueQuantMode == KVQuantMode::TQ3) {
                    // Vec_dot Value fusion: accumulate in rotated domain, WHT_inverse once
                    // qkSoftmax format: [kvSeq/mPack, seqLen, mPack], element (s,q) at (s/mPack)*seqLen*mPack + q*mPack
                    // + s%mPack
                    int tq3BytesPerSeq = (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
                    int numBlocks = mHeadDim / TQ3_BLOCK_SIZE;
                    auto tq3Buf = mTQ3DequantBuf->host<int8_t>() + tId * mTQ3DequantBuf->stride(0);
                    auto vAccRotated = (float*)(tq3Buf + seqLen * mHeadDim * sizeof(float));

                    auto weightsPtr = (float*)(tq3Buf + seqLen * mHeadDim * sizeof(float) + mHeadDim * sizeof(float));
                    for (int q = rowStart; q < seqLen; q++) {
                        // Extract softmax weights for this query position (float)
                        float* weights = weightsPtr;
                        for (int s = 0; s < subKvSeqLen; s++) {
                            int packIdx = (s / mPack) * seqLen * mPack + q * mPack + s % mPack;
                            if (mBytes == 2) {
                                weights[s] = (float)((FLOAT16_T*)qkSoftmax)[packIdx];
                            } else {
                                weights[s] = ((float*)qkSoftmax)[packIdx];
                            }
                        }

                        // For each dim block: accumulate weighted codebook values in rotated domain
                        for (int b = 0; b < numBlocks; b++) {
                            memset(vAccRotated, 0, TQ3_BLOCK_SIZE * sizeof(float));
                            for (int s = 0; s < subKvSeqLen; s++) {
                                int seqIdx = i * mBlockKV + s;
                                const uint8_t* block =
                                    (uint8_t*)valueAddr + seqIdx * tq3BytesPerSeq + b * TQ3_BYTES_PER_BLOCK;
                                uint16_t scaleFp16;
                                memcpy(&scaleFp16, block, 2);
                                float w = weights[s] * tq3_fp16_to_float(scaleFp16);
                                tq3_weighted_acc_block(vAccRotated, w, block + 2);
                            }
                            // WHT_inverse to get final output values
                            float reconstructed[TQ3_BLOCK_SIZE];
                            tq3_wht_inverse_32(reconstructed, vAccRotated);

                            // Write to qkvPacked: [headDim/mPack, seqLen, mPack]
                            for (int d = 0; d < TQ3_BLOCK_SIZE; d++) {
                                int dimIdx = b * TQ3_BLOCK_SIZE + d;
                                int outIdx = (dimIdx / mPack) * seqLen * mPack + q * mPack + dimIdx % mPack;
                                if (mBytes == 2) {
                                    ((FLOAT16_T*)qkvPacked)[outIdx] = (FLOAT16_T)reconstructed[d];
                                } else {
                                    ((float*)qkvPacked)[outIdx] = reconstructed[d];
                                }
                            }
                        }
                    }
                } else if (mValueQuantMode == KVQuantMode::TQ4) {
                    // Vec_dot Value fusion for TQ4: same structure as TQ3
                    int tq4BytesPerSeq = (mHeadDim / TQ4_BLOCK_SIZE) * TQ4_BYTES_PER_BLOCK;
                    int numBlocks = mHeadDim / TQ4_BLOCK_SIZE;
                    auto tqBuf = mTQ3DequantBuf->host<int8_t>() + tId * mTQ3DequantBuf->stride(0);
                    auto vAccRotated = (float*)(tqBuf + seqLen * mHeadDim * sizeof(float));
                    auto weightsPtr = (float*)(tqBuf + seqLen * mHeadDim * sizeof(float) + mHeadDim * sizeof(float));

                    for (int q = rowStart; q < seqLen; q++) {
                        float* weights = weightsPtr;
                        for (int s = 0; s < subKvSeqLen; s++) {
                            int packIdx = (s / mPack) * seqLen * mPack + q * mPack + s % mPack;
                            if (mBytes == 2) {
                                weights[s] = (float)((FLOAT16_T*)qkSoftmax)[packIdx];
                            } else {
                                weights[s] = ((float*)qkSoftmax)[packIdx];
                            }
                        }
                        for (int b = 0; b < numBlocks; b++) {
                            memset(vAccRotated, 0, TQ4_BLOCK_SIZE * sizeof(float));
                            for (int s = 0; s < subKvSeqLen; s++) {
                                int seqIdx = i * mBlockKV + s;
                                const uint8_t* block =
                                    (uint8_t*)valueAddr + seqIdx * tq4BytesPerSeq + b * TQ4_BYTES_PER_BLOCK;
                                uint16_t scaleFp16;
                                memcpy(&scaleFp16, block, 2);
                                float w = weights[s] * tq3_fp16_to_float(scaleFp16);
                                tq4_weighted_acc_block(vAccRotated, w, block + 2);
                            }
                            float reconstructed[TQ4_BLOCK_SIZE];
                            tq3_wht_inverse_32(reconstructed, vAccRotated);
                            for (int d = 0; d < TQ4_BLOCK_SIZE; d++) {
                                int dimIdx = b * TQ4_BLOCK_SIZE + d;
                                int outIdx = (dimIdx / mPack) * seqLen * mPack + q * mPack + dimIdx % mPack;
                                if (mBytes == 2) {
                                    ((FLOAT16_T*)qkvPacked)[outIdx] = (FLOAT16_T)reconstructed[d];
                                } else {
                                    ((float*)qkvPacked)[outIdx] = reconstructed[d];
                                }
                            }
                        }
                    }
                } else if (mValueQuantMode != KVQuantMode::Int8) {
                    auto valuePtr = valueAddr + i * vstride0 * mBytes;
                    size_t shapeParameters[7] = {(size_t)eP * lP * mBytes, ROUND_UP((size_t)subKvSeqLen, lP), (size_t)mHeadDim, (size_t)seqLen * mPack * mBytes, 0, 0, 0};
                    size_t bExtraStride = (i < kvBlocks - 1) ? 0 : (ROUND_UP(mKVCacheManager->getFlashAttentionBlockKv(), lP) - ROUND_UP(subKvSeqLen, lP)) * hP * mBytes;
                    shapeParameters[5] = bExtraStride;

                    int loop_e = (seqLen - rowStart) / eP;
                    int remain = (seqLen - rowStart) % eP;

                    int ei = 0;
                    elFloatV[0] = eP;
                    elFloatV[1] = ROUND_UP(subKvSeqLen, lP);
                    infoFloatV[2] = eP;
                    for ( ; ei < loop_e; ei++) {
                        srcPtr[0] = (float const*)((int8_t*)qkSoftmax + (ei * eP + rowStart) * mPack * mBytes);
                        gcore->MNNPackC4ForMatMul_A((float*)qkReordered, srcPtr, infoFloatV, elFloatV);
                        gcore->MNNPackedMatMul((float*)(qkvPacked + (ei * eP + rowStart) * mPack * mBytes), (float*)qkReordered, (float*)valuePtr, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                    if (remain > 0) {
                        elFloatV[0] = remain;
                        infoFloatV[2] = remain;
                        srcPtr[0] = (float const*)((int8_t*)qkSoftmax + (loop_e * eP + rowStart) * mPack * mBytes);
                        shapeParameters[0] = remain * lP * mBytes;
                        gcore->MNNPackC4ForMatMul_A((float*)qkReordered, srcPtr, infoFloatV, elFloatV);
                        gcore->MNNPackedMatMulRemain((float*)(qkvPacked + (loop_e * eP + rowStart) * mPack * mBytes), (float*)qkReordered, (float*)valuePtr, remain, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                } else { // use int8 kernel to compute qk@ v
                    auto valuePtr = valueAddr + i * vstride0;
                    auto eRemain = seqLen - rowStart;
                    auto qkPtr = (int8_t*)(qkSoftmax) + rowStart * mPack * mBytes; // [UP_DIV(subKvSeqLen,pack),seqLen,pack]
                    auto qkvFloat = qkvPacked + rowStart * mPack * mBytes;
                    gemmParam4QKxV.weightKernelSum = valueSum + i * ROUND_UP(mHeadDim, hP8);
                    sumParams4QKxV.valid = subKvSeqLen % lP8;
                    sumParams4QKxV.LU = UP_DIV(subKvSeqLen, lP8);

                    auto dstInt8Ptr = (int8_t*)mQuantQK.ptr() + tId * eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, mPack);
                    srcPtr[0] = (const float*)(dstInt8Ptr);

                    while (eRemain > 0) {
                        auto eSize = ALIMIN(eRemain, eP8);

                        memset(dstInt8Ptr, 0, eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, mPack));

                        infoInt8V[1] = eSize; // eReal
                        infoInt8V[4] = eSize; // e to process
                        elInt8V[0] = eSize;   // e to process


                        for (int qi = 0; qi < UP_DIV(subKvSeqLen, mPack); ++qi) {
                            mQuantFunc((float*)(qkPtr + qi * seqLen * mPack * mBytes), dstInt8Ptr + qi * eSize * mPack, eSize, vQuantScale, -128, 127, vQuantBias, 0);
                        }
                        core->MNNPackC4Int8ForMatMul_A(qkReordered, (int8_t const **)srcPtr, infoInt8V, elInt8V);
                        // mSumQK
                        gcore->MNNSumByAxisLForMatmul_A(gemmParam4QKxV.srcKernelSum, qkReordered, (float*)mQKScale.ptr(), eSize, sumParams4QKxV);
                        mInt8GemmKernel(qkvFloat, qkReordered, valuePtr, UP_DIV(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8), mBytes * seqLen * mPack, UP_DIV(mHeadDim, mPack), &gemmParam4QKxV, eSize);

                        eRemain -= eSize;
                        qkPtr += (eSize * mPack * mBytes);
                        qkvFloat += (eSize * mPack * mBytes);
                    }
                }

                // 4. flash attention, update each sub kvSeq's final results
                if (runningMax != nullptr && runningSum != nullptr && diffScale != nullptr) {
                    gcore->MNNFlashAttentionUpdateBlockOutput((float*)outputPacked, (float*)qkvPacked, diffScale, runningSum, UP_DIV(mHeadDim, mPack), seqLen, mPack, i, kvBlocks, mPackQKV->stride(0) / mBytes, mBytes, rowStart);
                }
            }

            // Final results writing: [head_dim/mPack, seq_len, mPack] -> [seq_len, num_head, head_dim]
            auto dstPtr = outputs[0]->host<int8_t>() + h * mHeadDim * mBytes;
            // offset = {seqLen, mNumHead * mHeadDim};
            gcore->MNNUnpackCUnitTranspose((float*)dstPtr, (float*)outputPacked, seqLen, mHeadDim, offset);
        }
    };

    MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
        mCompute((int)tId);
    }
    MNN_CONCURRENCY_END();

    backend()->onReleaseBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mTempQKBlock.get(), Backend::STATIC);

    if (!mKVCache) {
        mKVCacheManager->onClear();
    }
    auto ptr = outputs[0]->host<float>();
    if (seqLen < outputs[0]->length(1)) {
        ::memset(outputs[0]->host<uint8_t>() + seqLen * mHeadDim * mNumHead * mBytes, 0, (outputs[0]->length(1)-seqLen) * mHeadDim * mNumHead * mBytes);
    }
    return NO_ERROR;
}

bool CPUAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new CPUAttention(bn, mKVCache);
    // Share KV cache when cloning within the same session (same meta pointer)
    if (bn->getMetaPtr() == mMeta) {
        tmp->mKVCacheManager = mKVCacheManager;
        // Mark as KV-shared if the target op requests KV reuse
        auto param = op->main_as_AttentionParam();
        if (param && param->kv_shared_layer_index() >= 0) {
            tmp->mIsKVShared = true;
        }
    }
    *dst = tmp;
    return true;
}

CPUAttention::CPUAttention(Backend* backend, bool kv_cache) : Execution(backend), mKVCache(kv_cache) {
    mMeta = (KVMeta*)(backend->getMetaPtr());
    mPackQ.reset(Tensor::createDevice<float>({1, 1, 1, 1}));
    mPackQKV.reset(Tensor::createDevice<float>({1, 1, 1, 1}));
    MNN::KVCacheManager::KVCacheConfig kvconfig;

    // attentionOption % 8:
    // 0: Do not quantize
    // 1: Q,K: Int8, V: Float32
    // 2: Q,K,V: Int8
    // 3: K: TQ3, V: Float32
    // 4: K,V: TQ3
    // 5: K: TQ4, V: Float32
    // 6: K,V: TQ4

    // attentionOption / 8:
    // 0: do not use flash attention
    // 1: use flash attention
    kvconfig.mKVCacheDir = static_cast<CPUBackend *>(backend)->getRuntime()->hint().kvcacheDirPath;
    kvconfig.mPrefixCacheDir = static_cast<CPUBackend *>(backend)->getRuntime()->hint().prefixcacheDirPath;
    kvconfig.mExpandChunk = 64;
    kvconfig.mBlockNum = 1;
    mKVCacheManager.reset(new CPUKVCacheManager(backend, kvconfig));
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

