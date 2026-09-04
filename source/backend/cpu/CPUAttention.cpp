//
//  CPUAttention.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <atomic>
#include <limits>
#include "CPUAttention.hpp"
#include "CPUBackend.hpp"
#include "compute/CPUExtension.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/TurboQuant.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/BufferAllocator.hpp"
#include "compute/ConvolutionTiledExecutor.hpp"

#if defined(__aarch64__)
#define FLOAT16_T __fp16
#else
#define FLOAT16_T float
#endif

namespace MNN {

template <typename T>
static void _maskQK(float* qkPacked, const float* scale, size_t seqLen, size_t processedKvSeq, int pack, int kvSeqLen,
                    int kvoffset, int padKvSeqLen, const float* sinksPtr, const Tensor* mask, bool scaleApplied,
                    bool isLowerTriangular) {
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

    if (isLowerTriangular && scaleApplied) {
        return;
    }
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    auto source = (T*)qkPacked;
    float scaleVal = scale[0];

    auto processedKvSeqDivPack = UP_DIV(processedKvSeq, pack);
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

    int gapLen = (mask->elementSize() == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen))
                     ? 0
                     : static_cast<int>(kvSeqLen - seqLen);
    auto maskPtr = mask->host<T>();
    auto maskCols = (mask->elementSize() == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen)) ? kvSeqLen + padKvSeqLen
                                                                                               : seqLen + padKvSeqLen;
    for (int i = 0; i < processedKvSeqDivPack; ++i) {
        T* blockDataPtr = source + (i * seqLen * pack);

        for (int j = 0; j < seqLen; ++j) {
            T* dataPtr = blockDataPtr + (j * pack);
            const T* currentMaskRow = maskPtr + j * maskCols;

            for (int k = 0; k < pack; ++k) {
                float val = (float)dataPtr[k];
                if (!scaleApplied) {
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
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    gcore->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mThreadNum = ((CPUBackend*)backend())->computeThreadNumber(inputs[0]->length(1));
    mPack = gcore->pack;
    mBytes = gcore->bytes;
    int attentionOption = static_cast<CPUBackend*>(backend())->getRuntime()->hint().attentionOption;
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
    auto key = inputs[1];
    int seqLen = query->length(1);
    int mBlockNum = 1;
    mQNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    if (!mIsKVShared) {
        mKVCacheManager->setKVQuantMode(mUseFlashAttention, mKeyQuantMode, mValueQuantMode);
        mKVCacheManager->onResize(mKvNumHead, mHeadDim);
    }

    // Decode GQA batching: query heads sharing one KV head run as a single e=group GEMM,
    // so each K/V block is read once per group instead of once per head. Safe for fp32 too
    // because mixed SME/NEON dispatch is fp16-only (see onExecute): the e=group fp32 GEMM
    // only regressed when NEON compat threads ran it against SME-packed KV.
    int groupSize = (mKvNumHead > 0 && mQNumHead % mKvNumHead == 0) ? mQNumHead / mKvNumHead : 1;
    mDecodeGqaBatch = mUseFlashAttention && seqLen == 1 && groupSize > 1 &&
                      mKeyQuantMode == KVQuantMode::None && mValueQuantMode == KVQuantMode::None;
    const int qRows = mDecodeGqaBatch ? groupSize : seqLen; // row count of QK / QKV tiles

    // Common buffer allocated
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    mPackQKV.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack), qRows, mPack * mBytes}));
    backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
    if (inputs.size() > 4 || mUseFlashAttention) { // needed by flash attention and sliding attention with sink
        mRunningMax.reset(Tensor::createDevice<int8_t>({mThreadNum, qRows * 4}));
        mRunningSum.reset(Tensor::createDevice<int8_t>({mThreadNum, qRows * 4}));
        backend()->onAcquireBuffer(mRunningMax.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mRunningSum.get(), Backend::DYNAMIC);
    }
    if (mUseFlashAttention) { // extra buffer need by flash attention
        mExpfDiffMax.reset(Tensor::createDevice<int8_t>({mThreadNum, qRows * 4}));
        mTempOut.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack), qRows, mPack * mBytes}));
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
        mPackQ.reset(Tensor::createDevice<int8_t>({mQNumHead, (int32_t)packedQSize}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);

        mSumQ = bufferAlloc->alloc(mThreadNum * ROUND_UP(seqLen, eP8) * mBlockNum * sizeof(int32_t));
        mQueryScale = bufferAlloc->alloc(mQNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryZeroPoint = bufferAlloc->alloc(mQNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryQuantZero = bufferAlloc->alloc(mQNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryQuantScale = bufferAlloc->alloc(mQNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQuantQuery = bufferAlloc->alloc(seqLen * mQNumHead * UP_DIV(mHeadDim, gcore->pack) * gcore->pack);

        if (mBlockNum > 1) {
            mAccumBuffer = bufferAlloc->alloc(eP8 * hP8 * mThreadNum * QUANT_INFO_BYTES);
            if (mAccumBuffer.invalid()) {
                return OUT_OF_MEMORY;
            }
        }

        if (mSumQ.invalid() || mQueryScale.invalid() || mQueryQuantZero.invalid() || mQueryZeroPoint.invalid() ||
            mQueryQuantScale.invalid() || mQuantQuery.invalid()) {
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
        mPackQ.reset(
            Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(qRows, eP), ROUND_UP(mHeadDim, lP), eP * mBytes}));
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
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    bool outputC4 = TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    int seqLen = query->length(1);
    mQNumHead = query->length(2);
#ifdef MNN_SME2
    // fp16 only: for fp32 the NEON compat threads (deinterleaving SME-packed KV per tile)
    // cost more than the extra SME core earns; all-SME2 + GQA batching + wide blocks wins
    // at every kv length (0.6B t4 kv512/1024/2048: 0.060/0.103/0.259 vs 0.084/0.145/0.296 ms).
    mUseMixedSmeNeonMatMul = seqLen == 1 && mUseFlashAttention && mKeyQuantMode == KVQuantMode::None &&
                              mValueQuantMode == KVQuantMode::None && gcore->bytes == 2 &&
                              gcore->supportSME2 &&
                              gcore->smeCoreNumber > 0 && mThreadNum > gcore->smeCoreNumber;
    mSmeThreadCount = 0;
    // Decode-side per-core work ratio of one SME core vs one NEON core, from the light
    // proportion of divisionRatio (same encoding as ConvInt8's _getProportions). It only
    // shifts a fixed item boundary, so the item -> kernel-variant mapping stays deterministic.
    int smeDecodeRatio = 1;
    if (mUseMixedSmeNeonMatMul) {
        mSmeThreadCount = ALIMIN(mThreadNum, gcore->smeCoreNumber);
        const int totalProp = static_cast<CPUBackend*>(backend())->getRuntime()->hint().divisionRatio;
        int lightProp = totalProp % 8;
        const int intensiveProp = totalProp / 8 % 8;
        if (lightProp == 0) {
            lightProp = 1;
        } else if (intensiveProp != 0 && lightProp > intensiveProp) {
            lightProp = 1;
        }
        smeDecodeRatio = lightProp;
    }
#endif
    auto queryPtr = query->host<int8_t>();
    const Tensor* mask = nullptr;
    if (inputs.size() > 3) {
        mask = inputs[3];
    }
    const Tensor* sinks = nullptr;
    if (inputs.size() > 4) {
        sinks = inputs[4];
        MNN_ASSERT(sinks != nullptr);
        MNN_ASSERT(sinks->elementSize() == mQNumHead)
    }
    int group_size = mQNumHead / mKvNumHead;
    // reduce the value of 'query' to avoid fp16 overflow
    float mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0 / sqrt(mHeadDim));
    float q_scale = 1.0;
    if (mBytes == 2 && (mKeyQuantMode == KVQuantMode::TQ3 || mKeyQuantMode == KVQuantMode::TQ4)) {
        // reduce the value of 'query' to 'query * FP16_QSCALE', avoid fp16 overflow
        FLOAT16_T minValue;
        FLOAT16_T maxValue;
        gcore->MNNCountMaxMinValue(reinterpret_cast<float*>(queryPtr), (float*)(&minValue), (float*)(&maxValue),
                                   (size_t)seqLen * mQNumHead * mHeadDim);
        float maxV = maxValue;
        float minV = minValue;
        float absMax = ALIMAX(fabsf(maxV), fabsf(minV));
        if (absMax > 1.0f) {
            q_scale = 1.0f / absMax;
        }
        mScale /= q_scale;
    }
    // For the float key path the whole attention scale is folded into Q packing,
    // so QK products are already final logits and _maskQK needs no extra scaling.
    float packScale = q_scale * mScale;
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
        // Decode (1 new token) has no causal-mask waste, so a wider kv block amortizes
        // per-block fixed costs (softmax setup, PV prologue, flash rescale). Single-thread
        // widens both logical block and physical V block to 2048. Multithread keeps the
        // 64-row physical V block (a wide one regresses ~5% via L2/TLB pressure) but still
        // runs 256-row logical blocks: the PV matmul below loops over 64-row physical
        // sub-blocks and accumulates. hP must divide the logical block (flat-K QK offset)
        // and mPack must divide the physical V block (A-panel sub-block offset). K-int8/V-float
        // joins both tiers: its int8 K cache is flat (offsets scale with mKvBlockSize) and V stays
        // on the V-block-parameterized float path; hP8 must also divide the logical block for the
        // int8-K QK offset. V-int8 stays at 64 (its PV call sites hardcode the block).
        // fp32 joins the multithread wide block only because fp32 decode never uses mixed
        // SME/NEON dispatch: with compat NEON threads in play the wide block gave no
        // stable fp32 gain, but on the all-SME2 fp32 path it wins at every kv length
        // (0.6B decode kv512/1024/2048 t4: 0.084/0.145/0.296 -> 0.060/0.103/0.259 ms).
        int blockCap = MNN_FLASH_ATTENTION_BLOCK_SIZE;
        const bool wideBlockKv = mValueQuantMode == KVQuantMode::None &&
                                 (mKeyQuantMode == KVQuantMode::None || mKeyQuantMode == KVQuantMode::Int8);
        if (insertLen == 1 && wideBlockKv) {
            if (static_cast<CPUBackend*>(backend())->threadNumber() == 1) {
                blockCap = MNN_FLASH_ATTENTION_BLOCK_DECODE;
            } else if (0 == (4 * MNN_FLASH_ATTENTION_BLOCK_SIZE) % hP && 0 == MNN_FLASH_ATTENTION_BLOCK_SIZE % mPack &&
                       (mKeyQuantMode == KVQuantMode::None ||
                        0 == (4 * MNN_FLASH_ATTENTION_BLOCK_SIZE) % hP8) &&
                       mKVCacheManager->getFlashAttentionBlockKv() == MNN_FLASH_ATTENTION_BLOCK_SIZE) {
                blockCap = 4 * MNN_FLASH_ATTENTION_BLOCK_SIZE;
            }
        }
        mKvBlockSize = ALIMIN(blockCap, mKVCacheManager->kvLength());
    } else {
        mKvBlockSize = mKVCacheManager->kvLength();
    }

    // Constant Initialization
    auto padSeqLength = seqLen - insertLen;
    seqLen = insertLen;
    int kvSeqLen = mKVCacheManager->kvLength();
    int maxLen = mKVCacheManager->maxLength();
    int32_t units[2] = {eP, lP};
    const float* sinksPtr = sinks ? sinks->host<float>() : nullptr;
    int kvValidOffset = kvSeqLen - seqLen; // reuse_kv=true or decode, kvValidOffset>0

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
    const bool directC4Output = outputC4 && mHeadDim % mPack == 0;
    // Decode GQA batch: QK/QKV tile rows become the group's query heads (seqLen == 1, causal),
    // heads sharing one KV head are partitioned to the same thread
    const int qHeadsPerUnit = (mDecodeGqaBatch && seqLen == 1 && isLowerTriangular) ? group_size : 1;
    const int qRows = (qHeadsPerUnit > 1) ? qHeadsPerUnit : seqLen;
    const int headUnitCount = mQNumHead / qHeadsPerUnit;
    int numHeadDiv = UP_DIV(headUnitCount, mThreadNum);

    // Flash-decoding KV split: decode GQA batching produces only headUnitCount(=kvHeads) work items,
    // which quantizes badly over mThreadNum (e.g. 8 units on 6 threads -> 2,2,2,1,1,0). Split each
    // unit's KV-block range into splits so item count headUnitCount*kvSplitsPerUnit divides evenly by mThreadNum.
    int kvSplitsPerUnit = 1;
    if (qHeadsPerUnit > 1 && mThreadNum > 1) {
        int kvBlockNums = UP_DIV(kvSeqLen, mKvBlockSize);
        if (kvBlockNums > 1) {
            int a = headUnitCount, b = mThreadNum;
            while (b > 0) {
                int t = a % b;
                a = b;
                b = t;
            }
            kvSplitsPerUnit = ALIMIN(mThreadNum / a, kvBlockNums);
        }
    }

#ifdef MNN_USE_RVV
    if (tryExecuteFastPath(queryPtr, outputs[0]->host<int8_t>(), seqLen, kvSeqLen, padSeqLength, q_scale, mScale,
                           isLowerTriangular, sinksPtr != nullptr, outputC4, directC4Output)) {
        if (!mKVCache) {
            mKVCacheManager->onClear();
        }
        if (!outputC4 && seqLen < outputs[0]->length(1)) {
            ::memset(outputs[0]->host<uint8_t>() + seqLen * mHeadDim * mQNumHead * mBytes, 0,
                     (outputs[0]->length(1) - seqLen) * mHeadDim * mQNumHead * mBytes);
        }
        return NO_ERROR;
    }
#endif

    // Temporary tensors for intermediate results
    std::shared_ptr<Tensor> softmMaxQ(Tensor::createDevice<int32_t>(
        {mThreadNum, qRows, ROUND_UP(mKvBlockSize, mPack)})); // [mKvBlockSize/mPack, qRows, mPack ]
    std::shared_ptr<Tensor> newPackQK;
    if (mValueQuantMode != KVQuantMode::Int8) {
        newPackQK.reset(Tensor::createDevice<int8_t>({mThreadNum, eP * ROUND_UP(mKvBlockSize, lP) * mBytes}));
    } else {
        newPackQK.reset(
            Tensor::createDevice<int8_t>({mThreadNum, eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8)}));
    }
    std::shared_ptr<Tensor> mTempQKBlock(
        Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mKvBlockSize, mPack), qRows, mPack * mBytes}));
    std::shared_ptr<Tensor> kvSplitPartials;
    if (kvSplitsPerUnit > 1) {
        // Per (unit, split) item: packed output tile + runningMax/runningSum per row
        int partialSlotBytes = UP_DIV(mHeadDim, mPack) * qRows * mPack * mBytes + 2 * qRows * sizeof(float);
        kvSplitPartials.reset(Tensor::createDevice<int8_t>({headUnitCount * kvSplitsPerUnit * partialSlotBytes}));
    }
    std::shared_ptr<Tensor> pvSubBlockScratch;
    if (mValueQuantMode != KVQuantMode::Int8 && mKvBlockSize > (int)mKVCacheManager->getFlashAttentionBlockKv()) {
        // Wide logical block over smaller physical V blocks: per-sub-block PV tile to accumulate
        // from (packedMatMul overwrites C, there is no accumulate mode). Same layout as qkvPacked.
        pvSubBlockScratch.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack) * qRows * mPack * mBytes}));
    }
    if (!backend()->onAcquireBuffer(softmMaxQ.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    if (!backend()->onAcquireBuffer(newPackQK.get(), Backend::STATIC)) {
        backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
        return OUT_OF_MEMORY;
    }
    if (!backend()->onAcquireBuffer(mTempQKBlock.get(), Backend::STATIC)) {
        backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
        backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
        return OUT_OF_MEMORY;
    }
    if (kvSplitPartials.get() && !backend()->onAcquireBuffer(kvSplitPartials.get(), Backend::STATIC)) {
        backend()->onReleaseBuffer(mTempQKBlock.get(), Backend::STATIC);
        backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
        backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
        return OUT_OF_MEMORY;
    }
    if (pvSubBlockScratch.get() && !backend()->onAcquireBuffer(pvSubBlockScratch.get(), Backend::STATIC)) {
        if (kvSplitPartials.get()) {
            backend()->onReleaseBuffer(kvSplitPartials.get(), Backend::STATIC);
        }
        backend()->onReleaseBuffer(mTempQKBlock.get(), Backend::STATIC);
        backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
        backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
        return OUT_OF_MEMORY;
    }

    // Quantize Q and initialize bias 0
    if (mKeyQuantMode == KVQuantMode::Int8) {
        mGemmBias.reset(ROUND_UP(ALIMAX(mKvBlockSize, mHeadDim), hP8) * QUANT_INFO_BYTES);
        if (!mGemmBias.get()) {
            MNN_ERROR("Allocate bias buffer failed in CPU Attention\n");
            backend()->onReleaseBuffer(mTempQKBlock.get(), Backend::STATIC);
            backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
            backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
            return OUT_OF_MEMORY;
        }
        memset(mGemmBias.get(), 0, ROUND_UP(ALIMAX(mKvBlockSize, mHeadDim), hP8) * QUANT_INFO_BYTES);

        // Q: [seqLen,numHead,headDim]
        // maxQ, minQ: [seqLen,numHead]
        // scaleQ, zeroQ: [numHead, seqLen]
        // quantQ: [seqLen,numHead,headDim]
        int divPart = UP_DIV(seqLen * mQNumHead, mThreadNum);
        MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
            size_t info[9] = {1, (size_t)mHeadDim, 1, 1, 1, 1, 1, 1, 0};
            auto remainLu = seqLen * mQNumHead - tId * divPart;
            if (remainLu > 0) {
                remainLu = ALIMIN(divPart, remainLu);
                for (int i = tId * divPart; i < tId * divPart + remainLu; ++i) {
                    // address
                    auto srcFloatPtr = (float*)(queryPtr + i * mHeadDim * mBytes);
                    auto dstInt8Ptr = (int8_t*)(mQuantQuery.ptr() + i * mHeadDim);
                    auto quantScalePtr = (float*)(mQueryQuantScale.ptr() + i * QUANT_INFO_BYTES);
                    auto quantZeroPtr = (float*)(mQueryQuantZero.ptr() + i * QUANT_INFO_BYTES);

                    // scaleQ, zeroQ, [seqLen,numHead]->[numHead,seqLen]
                    int indexQ = (i / mQNumHead) + (i % mQNumHead) * seqLen;
                    auto scalePtr = (float*)(mQueryScale.ptr() + indexQ * QUANT_INFO_BYTES);
                    auto zeroPtr = (float*)(mQueryZeroPoint.ptr() + indexQ * QUANT_INFO_BYTES);

                    // compute the quant/dequant scale/bias
                    gcore->MNNAsyQuantInfo(scalePtr, zeroPtr, quantScalePtr, quantZeroPtr, nullptr, nullptr,
                                           srcFloatPtr, info);
                    scalePtr[0] *= mScale;
                    zeroPtr[0] *= mScale;

                    // quantize the float query to int8_t query
                    mQuantFunc(srcFloatPtr, dstInt8Ptr, UP_DIV(mHeadDim, gcore->pack), quantScalePtr, -128, 127,
                               quantZeroPtr, 0);
                }
            }
        }
        MNN_CONCURRENCY_END();

        // source int8_t query: [seqLen,numHead,headDim]
        // dest int8_t query: [numHead,seqLen/eP,headDim/lP,eP,lP]

        int outterSeqLen = UP_DIV(seqLen, eP8);
        int outterHeadDim = UP_DIV(mHeadDim, lP8);
        size_t outputOffset = 0;

        const int8_t* src_base_ptr = (const int8_t*)mQuantQuery.ptr();
        int8_t* dst_base_ptr = mPackQ->host<int8_t>();

        for (int h = 0; h < mQNumHead; ++h) {
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

                    int8_t* dst_block_ptr = dst_base_ptr + outputOffset + (size_t)dimBlock * (eunit * lP8);

                    const size_t src_row_stride = (size_t)mQNumHead * mHeadDim;

                    for (int seqLocal = 0; seqLocal < eunit; ++seqLocal) {
                        int innerSeq = seqBase + seqLocal;

                        const int8_t* src_row_ptr =
                            src_base_ptr + (size_t)innerSeq * src_row_stride + (size_t)h * mHeadDim + dimBase;

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
                zeroPtr[k] = 0;
#else
                zeroPtr[k] = 128.f / 255.f;
#endif
            }
        }
    }

    int offset[2] = {seqLen, mQNumHead * mHeadDim};
    // Final results writing: [head_dim/mPack, seq_len, mPack] -> [seq_len, num_head, head_dim]
    std::function<void(int, int8_t*)> writeOut = [&](int h, int8_t* outputPacked) {
        if (!outputC4) {
            auto dstPtr = outputs[0]->host<int8_t>() + h * mHeadDim * mBytes;
            if (qHeadsPerUnit > 1) {
                // Batched rows are the group's heads: unpack to consecutive head slots
                int offsetGqa[2] = {qRows, mHeadDim};
                gcore->MNNUnpackCUnitTranspose((float*)dstPtr, (float*)outputPacked, qRows, mHeadDim, offsetGqa);
            } else {
                // offset = {seqLen, mQNumHead * mHeadDim};
                gcore->MNNUnpackCUnitTranspose((float*)dstPtr, (float*)outputPacked, seqLen, mHeadDim, offset);
            }
        } else if (directC4Output) {
            if (qHeadsPerUnit > 1) {
                // Scatter [headDim/mPack, qHeadsPerUnit, mPack] into each head's C4 plane
                for (int hg = 0; hg < qHeadsPerUnit; ++hg) {
                    auto dstHead = outputs[0]->host<int8_t>() + (h + hg) * mHeadDim * seqLen * mBytes;
                    for (int dq = 0; dq < mHeadDim / mPack; ++dq) {
                        ::memcpy(dstHead + dq * seqLen * mPack * mBytes,
                                 outputPacked + (dq * qRows + hg) * mPack * mBytes, mPack * mBytes);
                    }
                }
            }
        } else {
            auto outputPtr = outputs[0]->host<int8_t>();
            for (int hg = 0; hg < qHeadsPerUnit; ++hg) {
                for (int d = 0; d < mHeadDim; ++d) {
                    const int channel = (h + hg) * mHeadDim + d;
                    for (int s = 0; s < seqLen; ++s) {
                        const size_t srcOffset =
                            ((size_t)(d / mPack) * qRows * mPack + (hg * seqLen + s) * mPack + d % mPack) * mBytes;
                        const size_t dstOffset =
                            ((size_t)(channel / mPack) * seqLen * mPack + s * mPack + channel % mPack) * mBytes;
                        ::memcpy(outputPtr + dstOffset, outputPacked + srcOffset, mBytes);
                    }
                }
            }
        }
    };
    std::atomic<int> nextSplitItem(0);
    std::atomic<int> splitNextNeon(0);
    // With mixed SME/NEON matmul, which thread computes an item determines its numerics, so a
    // single shared counter would make results timing-dependent. Give each kernel-variant group
    // its own counter over a fixed item range; the grab within a group stays dynamic.
#ifdef MNN_SME2
    // SME group share is weighted by smeDecodeRatio per SME core vs 1 per NEON core.
    const int smeWeight = mSmeThreadCount * smeDecodeRatio;
    const int totalWeight = smeWeight + (mThreadNum - mSmeThreadCount);
    const bool groupSplitDispatch =
        kvSplitsPerUnit > 1 && mUseMixedSmeNeonMatMul && mSmeThreadCount > 0 && mSmeThreadCount < mThreadNum;
    const int smeSplitItems =
        groupSplitDispatch ? ALIMAX(1, (headUnitCount * kvSplitsPerUnit * smeWeight + totalWeight / 2) / totalWeight) : 0;
    const int smeUnitItems = (mUseMixedSmeNeonMatMul && kvSplitsPerUnit == 1)
                                 ? ALIMIN(headUnitCount, ALIMAX(1, (headUnitCount * smeWeight + totalWeight / 2) / totalWeight))
                                 : 0;
#endif
    std::function<void(int)> mCompute = [=, &nextSplitItem, &splitNextNeon](int tId) {
        int8_t* qReordered = nullptr;
        auto qkPacked = mTempQKBlock->host<int8_t>() + tId * mTempQKBlock->stride(0);
        auto qkSoftmax = softmMaxQ->host<float>() + tId * softmMaxQ->stride(0);
        auto qkReordered = newPackQK->host<int8_t>() + tId * newPackQK->stride(0);
        auto qkvBuffer = mPackQKV->host<int8_t>() + tId * mPackQKV->stride(0);
#ifdef MNN_SME2
        bool useNeonMatMul = false;
        int headIndex = 0;
        int headsToCompute = 0;
        if (mUseMixedSmeNeonMatMul) {
            // Kernel variant is fixed per thread; items are grabbed dynamically within each
            // variant group below, so the item -> variant mapping stays deterministic.
            useNeonMatMul = tId >= mSmeThreadCount;
        } else {
            headIndex = tId * numHeadDiv * qHeadsPerUnit;
            headsToCompute = headIndex < mQNumHead ? ALIMIN(numHeadDiv * qHeadsPerUnit, mQNumHead - headIndex) : 0;
        }
        auto packedMatMul = useNeonMatMul ? gcore->MNNPackedMatMulWithSme2PackedB : gcore->MNNPackedMatMul;
        auto packedMatMulRemain =
            useNeonMatMul ? gcore->MNNPackedMatMulRemainWithSme2PackedB : gcore->MNNPackedMatMulRemain;
#else
        const int headIndex = tId * numHeadDiv * qHeadsPerUnit;
        const int headsToCompute = headIndex < mQNumHead ? ALIMIN(numHeadDiv * qHeadsPerUnit, mQNumHead - headIndex) : 0;
#endif

        // Flash Attention
        auto runningMax = mRunningMax ? (float*)(mRunningMax->host<int8_t>() + tId * mRunningMax->stride(0)) : nullptr;
        auto runningSum = mRunningSum ? (float*)(mRunningSum->host<int8_t>() + tId * mRunningSum->stride(0)) : nullptr;
        auto diffScale =
            mExpfDiffMax ? (float*)(mExpfDiffMax->host<int8_t>() + tId * mExpfDiffMax->stride(0)) : nullptr;
        auto outputBuffer = mTempOut ? mTempOut->host<int8_t>() + tId * mTempOut->stride(0) : qkvBuffer;

        int kvBlockNums = UP_DIV(kvSeqLen, mKvBlockSize);

        QuanPostTreatParameters gemmParam4QxK, gemmParam4QKxV; // used by int8 gemm, allocated per thread.
        SumByAxisParams sumParams4QxK, sumParams4QKxV = {};
        float* qSumAddr = nullptr;
        float* qScale = nullptr;
        float* qBias = nullptr;
        float* accumbuff = nullptr;
        int32_t unitColBufferSize = 0;
        if (mKeyQuantMode == KVQuantMode::Int8) {
            // parameters shared by all mKvBlockSize
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

        size_t vBlockElements = ROUND_UP(mHeadDim, hP) * ROUND_UP(mKVCacheManager->getFlashAttentionBlockKv(), lP);
        if (mValueQuantMode == KVQuantMode::Int8) {
            vBlockElements = (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mKVCacheManager->getFlashAttentionBlockKv(), lP8) +
                        2 * QUANT_INFO_BYTES * mBlockNum * ROUND_UP(mHeadDim, hP8));
        }

        // use for V
        float const* srcPtr[1];
        // only used for quantized V
        float vQuantScale[1] = {255.f};
        float vQuantBias[1] = {-128.f};
        int32_t infoInt8V[5];
        infoInt8V[0] = 1; // number
        infoInt8V[2] = static_cast<int32_t>(sumParams4QKxV.unitColBufferSize);
        infoInt8V[3] = 1; // stride
        int32_t elInt8V[4] = {eP8, ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8), 0, 0};

        // only used for float V
        int32_t infoFloatV[4];
        infoFloatV[0] = 1;     // number
        infoFloatV[1] = qRows; // eReal
        infoFloatV[3] = 1;     // stride
        int32_t elFloatV[4] = {qRows, ROUND_UP(kvSeqLen, lP), 0, 0};

        auto runBlocks = [&](int h, int blkBegin, int blkEnd, int8_t* outputPacked, int8_t* qkvPacked) {
            auto dstStep = mBytes * qRows * mPack;
            // Prepare for flash attention; an attention sink is counted only by the first KV block
            if (runningSum && runningMax) {
                if (sinksPtr == nullptr || blkBegin > 0) {
                    memset(runningSum, 0, mRunningSum->stride(0));
                    for (int k = 0; k < qRows; ++k) {
                        runningMax[k] = std::numeric_limits<float>::lowest();
                    }
                } else {
                    for (int k = 0; k < qRows; ++k) {
                        runningSum[k] = 1.f; // exp(sink-sink)
                    }
                    for (int k = 0; k < qRows; ++k) {
                        // batched rows map to consecutive heads of the group
                        const int sinkHead = (qHeadsPerUnit > 1) ? (h + k) : h;
                        float sinkVal;
                        if (mBytes == 2) {
                            sinkVal = ((FLOAT16_T*)sinksPtr)[sinkHead];
                        } else {
                            sinkVal = sinksPtr[sinkHead];
                        }
                        runningMax[k] = sinkVal;
                    }
                }
            }

            // Compute the current addresses
            int kvHeadIndex = h / group_size;
            int8_t* keyAddr = mKVCacheManager->addrOfKey(kvHeadIndex);
            int8_t* keySum = mKVCacheManager->addrOfKeySum(kvHeadIndex);
            int8_t* valueAddr = mKVCacheManager->addrOfValue(kvHeadIndex);
            float* valueSum = (float*)mKVCacheManager->addrOfValueSum(kvHeadIndex);

            // Get packed Q
            if (mKeyQuantMode == KVQuantMode::None) {
                qReordered = mPackQ->host<int8_t>() + tId * mPackQ->stride(0);
                // Batched: pack the group's consecutive heads as rows (row stride = mHeadDim)
                gcore->MNNAttenPackAndScaleSingleHead((float*)qReordered, (float*)(queryPtr + h * mHeadDim * mBytes),
                                                      (qHeadsPerUnit > 1) ? mHeadDim : (mHeadDim * mQNumHead), &packScale,
                                                      units, qRows, mHeadDim);
            } else if (mKeyQuantMode == KVQuantMode::Int8) {
                qReordered = mPackQ->host<int8_t>() + h * mPackQ->stride(0);
                qSumAddr = (float*)(mSumQ.ptr() + tId * ROUND_UP(seqLen, eP8) * mBlockNum * QUANT_INFO_BYTES);
                qScale = (float*)(mQueryScale.ptr() + h * seqLen * mBlockNum * QUANT_INFO_BYTES);
                qBias = (float*)(mQueryZeroPoint.ptr() + h * seqLen * mBlockNum * QUANT_INFO_BYTES);
                gcore->MNNSumByAxisLForMatmul_A(qSumAddr, qReordered, qScale, seqLen, sumParams4QxK);
            }

            // Start computing
            const int localBlocks = blkEnd - blkBegin;
            for (int i = blkBegin; i < blkEnd; ++i) {
                const int localBlockIdx = i - blkBegin;
                int curKvBlockSize = ALIMIN(mKvBlockSize, kvSeqLen - i * mKvBlockSize);
                // Rows [0, rowStart) are fully masked by causality for this KV block
                int rowStart =
                    (!isLowerTriangular || i * mKvBlockSize < kvValidOffset) ? 0 : (i * mKvBlockSize - kvValidOffset);

                // 1. query @ key
                if (mKeyQuantMode == KVQuantMode::TQ3) {
                    // Vec_dot fusion: Q_rotated · TQ3_compressed_K directly (no dequant buffer)
                    // Q_rotated = WHT_forward(Q * q_scale) computed once per KV block iteration
                    int tq3BytesPerSeq = (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
                    int numBlocks = mHeadDim / TQ3_BLOCK_SIZE;
                    auto tq3Buf = mTQ3DequantBuf->host<int8_t>() + tId * mTQ3DequantBuf->stride(0);
                    auto qRotated = (float*)tq3Buf; // seqLen * headDim floats

                    // Pre-rotate Q vectors (only on first KV block)
                    if (localBlockIdx == 0) {
                        float qScale = 1.0f / sqrtf((float)mHeadDim);
                        auto queryBase = (float*)(queryPtr + h * mHeadDim * mBytes);
                        int qStride = mHeadDim * mQNumHead; // stride between seq positions
                        for (int q = 0; q < seqLen; q++) {
                            for (int b = 0; b < numBlocks; b++) {
                                float scaled[TQ3_BLOCK_SIZE];
                                if (mBytes == 2) {
                                    auto src16 =
                                        (FLOAT16_T*)(queryPtr + h * mHeadDim * mBytes) + q * mHeadDim * mQNumHead;
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
                    for (int s = 0; s < curKvBlockSize; s++) {
                        int seqIdx = i * mKvBlockSize + s;
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

                    if (localBlockIdx == 0) {
                        float qScale = 1.0f / sqrtf((float)mHeadDim);
                        for (int q = 0; q < seqLen; q++) {
                            for (int b = 0; b < numBlocks; b++) {
                                float scaled[TQ4_BLOCK_SIZE];
                                if (mBytes == 2) {
                                    auto src16 =
                                        (FLOAT16_T*)(queryPtr + h * mHeadDim * mBytes) + q * mHeadDim * mQNumHead;
                                    for (int d = 0; d < TQ4_BLOCK_SIZE; d++)
                                        scaled[d] = (float)src16[b * TQ4_BLOCK_SIZE + d] * qScale;
                                } else {
                                    auto srcF = (float*)(queryPtr + h * mHeadDim * mBytes) + q * mHeadDim * mQNumHead;
                                    for (int d = 0; d < TQ4_BLOCK_SIZE; d++)
                                        scaled[d] = srcF[b * TQ4_BLOCK_SIZE + d] * qScale;
                                }
                                tq3_wht_forward_32(qRotated + q * mHeadDim + b * TQ4_BLOCK_SIZE, scaled);
                            }
                        }
                    }

                    for (int s = 0; s < curKvBlockSize; s++) {
                        int seqIdx = i * mKvBlockSize + s;
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
                    auto keyPtr = keyAddr + i * UP_DIV(mKvBlockSize, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
                    int loop_e = qRows / eP;
                    int remain = qRows % eP;
                    // Skip eP tiles whose rows are all masked; softmax memsets them without reading src
                    int eStart = useMaskInSoftmax ? (rowStart / eP) : 0;
                    auto qStride0 = ROUND_UP(mHeadDim, lP) * eP * mBytes;
                    size_t shapeParameters[7] = {(size_t)eP * lP * mBytes,
                                                 ROUND_UP((size_t)mHeadDim, lP),
                                                 (size_t)curKvBlockSize,
                                                 (size_t)qRows * mPack * mBytes,
                                                 0,
                                                 0,
                                                 0};
                    for (int ei = eStart; ei < loop_e; ei++) {
#ifdef MNN_SME2
                        packedMatMul((float*)(qkPacked + (ei * eP * mPack) * mBytes),
#else
                        gcore->MNNPackedMatMul((float*)(qkPacked + (ei * eP * mPack) * mBytes),
#endif
                                               (float*)(qReordered + ei * qStride0), (float*)keyPtr, shapeParameters,
                                               nullptr, nullptr, nullptr, nullptr);
                    }
                    if (remain > 0) {
#ifdef MNN_SME2
                        packedMatMulRemain((float*)(qkPacked + (loop_e * eP * mPack) * mBytes),
#else
                        gcore->MNNPackedMatMulRemain((float*)(qkPacked + (loop_e * eP * mPack) * mBytes),
#endif
                                                     (float*)(qReordered + loop_e * qStride0), (float*)keyPtr, remain,
                                                     shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                } else {
                    auto eRemain = seqLen;
                    auto srcInt8 = qReordered;
                    auto dstInt8 = qkPacked;
                    auto keyPtr = keyAddr + i * UP_DIV(mKvBlockSize, hP8) *
                                                (ROUND_UP(mHeadDim, lP8) * hP8 + 2 * hP8 * QUANT_INFO_BYTES);
                    gemmParam4QxK.weightKernelSum = (float*)(keySum + i * mKvBlockSize * QUANT_INFO_BYTES);
                    gemmParam4QxK.inputScale = qScale;
                    gemmParam4QxK.inputBias = qBias;
                    gemmParam4QxK.srcKernelSum = qSumAddr;
                    while (eRemain > 0) {
                        auto eSize = ALIMIN(eP8, eRemain);
                        mInt8GemmKernel(dstInt8, srcInt8, keyPtr, UP_DIV(mHeadDim, lP8), mBytes * seqLen * mPack,
                                        UP_DIV(curKvBlockSize, mPack), &gemmParam4QxK, eSize);
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
                    bool scaleApplied =
                        (mKeyQuantMode == KVQuantMode::Int8 || mKeyQuantMode == KVQuantMode::None);
                    if (!scaleApplied || isLowerTriangular == false || sinksPtr != nullptr) {
                        if (mBytes == 2) {
                            _maskQK<FLOAT16_T>((float*)qkPacked, &mScale, qRows, curKvBlockSize, mPack, kvSeqLen,
                                               i * mKvBlockSize, padSeqLength, sinksPtr, mask, scaleApplied,
                                               isLowerTriangular);
                        } else {
                            _maskQK<float>((float*)qkPacked, &mScale, qRows, curKvBlockSize, mPack, kvSeqLen,
                                           i * mKvBlockSize, padSeqLength, sinksPtr, mask, scaleApplied,
                                           isLowerTriangular);
                        }
                    }
                    gcore->MNNSoftmax(qkSoftmax, (float*)qkPacked, runningMax, runningSum, diffScale, qRows,
                                      curKvBlockSize, i * mKvBlockSize, kvValidOffset, mPack, useMaskInSoftmax);
                }
                // 3. qk @ v
                auto qkStride0 = ROUND_UP(curKvBlockSize, lP) * eP * mBytes;

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
                        for (int s = 0; s < curKvBlockSize; s++) {
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
                            for (int s = 0; s < curKvBlockSize; s++) {
                                int seqIdx = i * mKvBlockSize + s;
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
                        for (int s = 0; s < curKvBlockSize; s++) {
                            int packIdx = (s / mPack) * seqLen * mPack + q * mPack + s % mPack;
                            if (mBytes == 2) {
                                weights[s] = (float)((FLOAT16_T*)qkSoftmax)[packIdx];
                            } else {
                                weights[s] = ((float*)qkSoftmax)[packIdx];
                            }
                        }
                        for (int b = 0; b < numBlocks; b++) {
                            memset(vAccRotated, 0, TQ4_BLOCK_SIZE * sizeof(float));
                            for (int s = 0; s < curKvBlockSize; s++) {
                                int seqIdx = i * mKvBlockSize + s;
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
                    // V cache is physically split into vBlockSize-row blocks. A logical block wider
                    // than the physical one (multithread decode) is computed as one PV matmul per
                    // physical sub-block: sub-block 0 writes qkvPacked directly, later sub-blocks go
                    // through a per-thread scratch and are added onto qkvPacked (packedMatMul
                    // overwrites C, there is no accumulate mode).
                    const size_t vBlockSize = mKVCacheManager->getFlashAttentionBlockKv();
                    const int kvStart = i * mKvBlockSize;
                    const int subBlockNums = UP_DIV(curKvBlockSize, (int)vBlockSize);
                    int8_t* pvAccumBase = nullptr;
                    if (subBlockNums > 1) {
                        pvAccumBase = pvSubBlockScratch->host<int8_t>() + tId * pvSubBlockScratch->stride(0);
                    }
                    // dst += src over one packed C tile: nBlk blocks x (rows x mPack) elements
                    auto addTile = [&](int8_t* dst, const int8_t* src, int rows) {
                        const int nBlk = UP_DIV(mHeadDim, mPack);
                        const int count = rows * mPack;
                        for (int nb = 0; nb < nBlk; ++nb) {
                            auto d = dst + nb * dstStep;
                            auto s = src + nb * dstStep;
                            if (mBytes == 2) {
                                auto d16 = (FLOAT16_T*)d;
                                auto s16 = (const FLOAT16_T*)s;
                                for (int v = 0; v < count; ++v) {
                                    d16[v] = (FLOAT16_T)((float)d16[v] + (float)s16[v]);
                                }
                            } else {
                                auto d32 = (float*)d;
                                auto s32 = (const float*)s;
                                for (int v = 0; v < count; ++v) {
                                    d32[v] += s32[v];
                                }
                            }
                        }
                    };
                    for (int subBlockIdx = 0; subBlockIdx < subBlockNums; ++subBlockIdx) {
                        // Sub-block starts are block-aligned (mKvBlockSize is a multiple of vBlockSize),
                        // so inBlock is 0 for subBlockIdx > 0.
                        const int subBlockStartRow = kvStart + subBlockIdx * (int)vBlockSize;
                        const int subBlockSize = ALIMIN(curKvBlockSize - subBlockIdx * (int)vBlockSize, (int)vBlockSize);
                        const int inBlock = subBlockStartRow % (int)vBlockSize;
                        auto valuePtr = valueAddr + (subBlockStartRow / (int)vBlockSize) * vBlockElements * mBytes +
                                        ((inBlock / lP) * hP * lP + inBlock % lP) * mBytes;
                        size_t shapeParameters[7] = {(size_t)eP * lP * mBytes,
                                                     ROUND_UP((size_t)subBlockSize, lP),
                                                     (size_t)mHeadDim,
                                                     (size_t)dstStep,
                                                     0,
                                                     0,
                                                     0};
                        // Physical N-group stride covers vBlockSize rows, logical only subBlockSize.
                        size_t vRowPadStride = (ROUND_UP(vBlockSize, lP) - ROUND_UP(subBlockSize, lP)) * hP * mBytes;
                        shapeParameters[5] = vRowPadStride;
                        // Sub-block's A panel starts subBlockIdx*vBlockSize columns into qkSoftmax
                        // ([kv/mPack][qRows][mPack]; vBlockSize is divisible by mPack).
                        const int softmaxColOffsetBytes = subBlockIdx * (int)vBlockSize * qRows * mBytes;

                        int loop_e = (qRows - rowStart) / eP;
                        int remain = (qRows - rowStart) % eP;

                        int ei = 0;
                        elFloatV[0] = eP;
                        elFloatV[1] = ROUND_UP(subBlockSize, lP);
                        infoFloatV[2] = eP;
                        for (; ei < loop_e; ei++) {
                            srcPtr[0] =
                                (float const*)((int8_t*)qkSoftmax + softmaxColOffsetBytes + (ei * eP + rowStart) * mPack * mBytes);
                            gcore->MNNPackC4ForMatMul_A((float*)qkReordered, srcPtr, infoFloatV, elFloatV);
                            auto cTile = qkvPacked + (ei * eP + rowStart) * mPack * mBytes;
                            if (subBlockIdx > 0) {
                                cTile = pvAccumBase;
                            }
#ifdef MNN_SME2
                            packedMatMul((float*)cTile,
#else
                            gcore->MNNPackedMatMul((float*)cTile,
#endif
                                                   (float*)qkReordered, (float*)valuePtr, shapeParameters, nullptr, nullptr,
                                                   nullptr, nullptr);
                            if (subBlockIdx > 0) {
                                addTile(qkvPacked + (ei * eP + rowStart) * mPack * mBytes, pvAccumBase, eP);
                            }
                        }
                        if (remain > 0) {
                            elFloatV[0] = remain;
                            infoFloatV[2] = remain;
                            srcPtr[0] = (float const*)((int8_t*)qkSoftmax + softmaxColOffsetBytes +
                                                       (loop_e * eP + rowStart) * mPack * mBytes);
                            shapeParameters[0] = remain * lP * mBytes;
                            gcore->MNNPackC4ForMatMul_A((float*)qkReordered, srcPtr, infoFloatV, elFloatV);
                            auto cTile = qkvPacked + (loop_e * eP + rowStart) * mPack * mBytes;
                            if (subBlockIdx > 0) {
                                cTile = pvAccumBase;
                            }
#ifdef MNN_SME2
                            packedMatMulRemain((float*)cTile,
#else
                            gcore->MNNPackedMatMulRemain((float*)cTile,
#endif
                                                         (float*)qkReordered, (float*)valuePtr, remain, shapeParameters,
                                                         nullptr, nullptr, nullptr, nullptr);
                            if (subBlockIdx > 0) {
                                addTile(qkvPacked + (loop_e * eP + rowStart) * mPack * mBytes, pvAccumBase, remain);
                            }
                        }
                    }
                } else { // use int8 kernel to compute qk@ v
                    auto valuePtr = valueAddr + i * vBlockElements;
                    auto eRemain = seqLen - rowStart;
                    auto qkPtr =
                        (int8_t*)(qkSoftmax) + rowStart * mPack * mBytes; // [UP_DIV(curKvBlockSize,pack),seqLen,pack]
                    auto qkvFloat = qkvPacked + rowStart * mPack * mBytes;
                    gemmParam4QKxV.weightKernelSum = valueSum + i * ROUND_UP(mHeadDim, hP8);
                    sumParams4QKxV.valid = curKvBlockSize % lP8;
                    sumParams4QKxV.LU = UP_DIV(curKvBlockSize, lP8);

                    auto dstInt8Ptr =
                        (int8_t*)mQuantQK.ptr() + tId * eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, mPack);
                    srcPtr[0] = (const float*)(dstInt8Ptr);

                    while (eRemain > 0) {
                        auto eSize = ALIMIN(eRemain, eP8);

                        memset(dstInt8Ptr, 0, eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, mPack));

                        infoInt8V[1] = eSize; // eReal
                        infoInt8V[4] = eSize; // e to process
                        elInt8V[0] = eSize;   // e to process

                        for (int qi = 0; qi < UP_DIV(curKvBlockSize, mPack); ++qi) {
                            mQuantFunc((float*)(qkPtr + qi * seqLen * mPack * mBytes), dstInt8Ptr + qi * eSize * mPack,
                                       eSize, vQuantScale, -128, 127, vQuantBias, 0);
                        }
                        core->MNNPackC4Int8ForMatMul_A(qkReordered, (int8_t const**)srcPtr, infoInt8V, elInt8V);
                        // mSumQK
                        gcore->MNNSumByAxisLForMatmul_A(gemmParam4QKxV.srcKernelSum, qkReordered,
                                                         (float*)mQKScale.ptr(), eSize, sumParams4QKxV);
                        mInt8GemmKernel(qkvFloat, qkReordered, valuePtr, UP_DIV(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8),
                                        dstStep, UP_DIV(mHeadDim, mPack), &gemmParam4QKxV, eSize);

                        eRemain -= eSize;
                        qkPtr += (eSize * mPack * mBytes);
                        qkvFloat += (eSize * mPack * mBytes);
                    }
                }

                // 4. flash attention, update each sub kvSeq's final results
                if (runningMax != nullptr && runningSum != nullptr && diffScale != nullptr) {
                    gcore->MNNFlashAttentionUpdateBlockOutput((float*)outputPacked, (float*)qkvPacked, diffScale,
                                                              runningSum, UP_DIV(mHeadDim, mPack), qRows, mPack, localBlockIdx,
                                                              localBlocks, mPackQKV->stride(0) / mBytes, mBytes, rowStart);
                }
            }
        };

        if (kvSplitsPerUnit > 1) {
            // Flash-decoding: work item = (KV-head unit, KV-block split), grabbed dynamically so slow
            // threads simply take fewer items. Each item stores a normalized partial output plus its
            // running max/sum; partials are merged after the barrier below.
            const int totalItems = headUnitCount * kvSplitsPerUnit;
            const int blocksBase = kvBlockNums / kvSplitsPerUnit;
            const int blocksRem = kvBlockNums % kvSplitsPerUnit;
            const int oElems = UP_DIV(mHeadDim, mPack) * qRows * mPack;
            const int partialSlotBytes = oElems * mBytes + 2 * qRows * sizeof(float);
            int8_t* partials = kvSplitPartials->host<int8_t>();
            for (;;) {
                int item;
#ifdef MNN_SME2
                if (groupSplitDispatch) {
                    auto& counter = useNeonMatMul ? splitNextNeon : nextSplitItem;
                    const int base = useNeonMatMul ? smeSplitItems : 0;
                    const int end = useNeonMatMul ? totalItems : smeSplitItems;
                    const int n = counter.fetch_add(1, std::memory_order_relaxed);
                    if (n >= end - base) {
                        break;
                    }
                    item = base + n;
                } else
#endif
                {
                    item = nextSplitItem.fetch_add(1, std::memory_order_relaxed);
                    if (item >= totalItems) {
                        break;
                    }
                }
                int unit = item / kvSplitsPerUnit;
                int splitIdx = item - unit * kvSplitsPerUnit;
                int blkBegin = splitIdx * blocksBase + ALIMIN(splitIdx, blocksRem);
                int blkEnd = blkBegin + blocksBase + (splitIdx < blocksRem ? 1 : 0);
                int8_t* slot = partials + (size_t)item * partialSlotBytes;
                runBlocks(unit * qHeadsPerUnit, blkBegin, blkEnd, slot, qkvBuffer);
                float* pstat = (float*)(slot + oElems * mBytes);
                ::memcpy(pstat, runningMax, qRows * sizeof(float));
                ::memcpy(pstat + qRows, runningSum, qRows * sizeof(float));
            }
            return;
        }
        auto runWholeUnit = [&](int h) {
            auto qkvPacked = qkvBuffer;
            auto outputPacked = outputBuffer;
            if (directC4Output && qHeadsPerUnit == 1) {
                outputPacked = outputs[0]->host<int8_t>() + h * mHeadDim * seqLen * mBytes;
                if (!mUseFlashAttention) {
                    qkvPacked = outputPacked;
                }
            }
            runBlocks(h, 0, kvBlockNums, outputPacked, qkvPacked);
            writeOut(h, outputPacked);
        };
#ifdef MNN_SME2
        if (mUseMixedSmeNeonMatMul) {
            // Same grouped dynamic dispatch as the kvSplitsPerUnit>1 path, with whole units as items.
            for (;;) {
                auto& counter = useNeonMatMul ? splitNextNeon : nextSplitItem;
                const int base = useNeonMatMul ? smeUnitItems : 0;
                const int end = useNeonMatMul ? headUnitCount : smeUnitItems;
                const int n = counter.fetch_add(1, std::memory_order_relaxed);
                if (n >= end - base) {
                    break;
                }
                runWholeUnit((base + n) * qHeadsPerUnit);
            }
            return;
        }
#endif
        for (int h = headIndex; h < headIndex + headsToCompute; h += qHeadsPerUnit) {
            runWholeUnit(h);
        }
    };

    MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
        mCompute((int)tId);
    }
    MNN_CONCURRENCY_END();

    if (kvSplitsPerUnit > 1) {
        // Merge split partials per unit: O = sum_c( w_c * O_c ), w_c = sum_c*exp(max_c - max) / sum_c(...)
        const int oElems = UP_DIV(mHeadDim, mPack) * qRows * mPack;
        const int partialSlotBytes = oElems * mBytes + 2 * qRows * sizeof(float);
        const int dQuad = UP_DIV(mHeadDim, mPack);
        const int8_t* partials = kvSplitPartials->host<int8_t>();
        MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
            const int unitsPerThread = UP_DIV(headUnitCount, mThreadNum);
            const int uBegin = tId * unitsPerThread;
            const int uEnd = ALIMIN(headUnitCount, uBegin + unitsPerThread);
            int8_t* dstPacked = mTempOut->host<int8_t>() + tId * mTempOut->stride(0);
            float* mstar = (float*)(mRunningMax->host<int8_t>() + tId * mRunningMax->stride(0));
            float* invSum = (float*)(mRunningSum->host<int8_t>() + tId * mRunningSum->stride(0));
            for (int u = uBegin; u < uEnd; ++u) {
                const int8_t* unitPart = partials + (size_t)(u * kvSplitsPerUnit) * partialSlotBytes;
                for (int q = 0; q < qRows; ++q) {
                    float mx = std::numeric_limits<float>::lowest();
                    for (int c = 0; c < kvSplitsPerUnit; ++c) {
                        const float* pstat = (const float*)(unitPart + (size_t)c * partialSlotBytes + oElems * mBytes);
                        mx = ALIMAX(mx, pstat[q]);
                    }
                    float s = 0.f;
                    for (int c = 0; c < kvSplitsPerUnit; ++c) {
                        const float* pstat = (const float*)(unitPart + (size_t)c * partialSlotBytes + oElems * mBytes);
                        s += pstat[qRows + q] * expf(pstat[q] - mx);
                    }
                    mstar[q] = mx;
                    invSum[q] = 1.f / s;
                }
                for (int c = 0; c < kvSplitsPerUnit; ++c) {
                    const int8_t* slot = unitPart + (size_t)c * partialSlotBytes;
                    const float* pstat = (const float*)(slot + oElems * mBytes);
                    for (int q = 0; q < qRows; ++q) {
                        float w = pstat[qRows + q] * expf(pstat[q] - mstar[q]) * invSum[q];
                        for (int d = 0; d < dQuad; ++d) {
                            int base = (d * qRows + q) * mPack;
                            if (mBytes == 2) {
                                auto dst16 = (FLOAT16_T*)dstPacked + base;
                                auto src16 = (const FLOAT16_T*)slot + base;
                                if (c == 0) {
                                    for (int l = 0; l < mPack; ++l) {
                                        dst16[l] = (FLOAT16_T)(w * (float)src16[l]);
                                    }
                                } else {
                                    for (int l = 0; l < mPack; ++l) {
                                        dst16[l] = (FLOAT16_T)((float)dst16[l] + w * (float)src16[l]);
                                    }
                                }
                            } else {
                                auto dstF = (float*)dstPacked + base;
                                auto srcF = (const float*)slot + base;
                                if (c == 0) {
                                    for (int l = 0; l < mPack; ++l) {
                                        dstF[l] = w * srcF[l];
                                    }
                                } else {
                                    for (int l = 0; l < mPack; ++l) {
                                        dstF[l] += w * srcF[l];
                                    }
                                }
                            }
                        }
                    }
                }
                writeOut(u * qHeadsPerUnit, dstPacked);
            }
        }
        MNN_CONCURRENCY_END();
        backend()->onReleaseBuffer(kvSplitPartials.get(), Backend::STATIC);
    }

    backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mTempQKBlock.get(), Backend::STATIC);
    if (pvSubBlockScratch.get()) {
        backend()->onReleaseBuffer(pvSubBlockScratch.get(), Backend::STATIC);
    }

    if (!mKVCache) {
        mKVCacheManager->onClear();
    }
    if (!outputC4) {
        auto ptr = outputs[0]->host<float>();
        if (seqLen < outputs[0]->length(1)) {
            ::memset(outputs[0]->host<uint8_t>() + seqLen * mHeadDim * mQNumHead * mBytes, 0,
                     (outputs[0]->length(1) - seqLen) * mHeadDim * mQNumHead * mBytes);
        }
    }
    return NO_ERROR;
}

bool CPUAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = createClone(bn);
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

CPUAttention::CPUAttention(Backend* backend, bool kvCache) : Execution(backend), mKVCache(kvCache) {
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
    kvconfig.mKVCacheDir = static_cast<CPUBackend*>(backend)->getRuntime()->hint().kvcacheDirPath;
    kvconfig.mPrefixCacheDir = static_cast<CPUBackend*>(backend)->getRuntime()->hint().prefixcacheDirPath;
    kvconfig.mExpandChunk = 64;
    kvconfig.mBlockNum = 1;
    mKVCacheManager.reset(new CPUKVCacheManager(backend, kvconfig));
}

bool CPUAttention::tryExecuteFastPath(const int8_t* query, int8_t* output, int seqLen, int kvSeqLen, int paddingLength,
                                      float qScale, float attentionScale, bool lowerTriangular, bool hasSinks,
                                      bool outputC4, bool directC4Output) {
    return false;
}

CPUAttention* CPUAttention::createClone(Backend* backend) const {
    return new CPUAttention(backend, mKVCache);
}

class CPUAttentionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_AttentionParam();
        auto extension = static_cast<CPUBackend*>(backend)->functions()->extension;
        if (extension != nullptr && extension->createAttentionExecution != nullptr) {
            auto execution = extension->createAttentionExecution(backend, param->kv_cache());
            if (execution != nullptr) {
                return execution;
            }
        }
        return new CPUAttention(backend, param->kv_cache());
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUAttentionCreator, OpType_Attention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
