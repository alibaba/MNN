//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MNN_KLEIDIAI_ENABLED
#include "KleidiAIConvInt8.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"

#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"

#define QUANT_INFO_BYTES 4
namespace MNN {

KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant, KleidiAI &kai, KleidiAI::AccelType accelType, int32_t blockNum) : CPUConvolution(op->main_as_Convolution2D()->common(), backend), kai(kai), mAccelType(accelType), mBlockNum(blockNum) {
    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();

    // backend info
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int pack = core->pack;

    // compute info
    int ocUp4 = ROUND_UP(oc, pack);
    int scaleSize = ocUp4 * mBlockNum;

    // kleidia info
    bool bFP16 = core->bytes == 2 ? true : false;
    bool bAsym = quanCommon->asymmetric;
    size_t blkSize = mBlockNum == 1 ? 0 : ic / mBlockNum;

    AutoStorage<int8_t> reorderedQuantInfo;
    reorderedQuantInfo.reset(2 * scaleSize * QUANT_INFO_BYTES + oc * QUANT_INFO_BYTES);
    if (reorderedQuantInfo.get() == nullptr) {
        MNN_ERROR("Memory not enough\n");
        return;
    }

    //Prepare scale and zero data.
    {
        int outputCount = convOp->common()->outputCount();
        int originOffset = -8;
        auto quanInfoPtr = quanCommon->alpha.get();
        auto scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
        auto zeroPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalePtr) + scaleSize * QUANT_INFO_BYTES);
        auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(zeroPtr) + scaleSize * QUANT_INFO_BYTES);
        if (quanCommon->asymmetric) {
            for (int i = 0; i < blockNum; ++i) {
                auto dstScale = scalePtr + i * ocUp4;
                auto dstZero  = zeroPtr + i * ocUp4;
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstScale[j] = quanInfoPtr[2 * scaleIndex + 1];
                    dstZero[j] = quanInfoPtr[2 * scaleIndex] + (float)originOffset * dstScale[j];
                }
            }
        } else {
            for (int i = 0; i < blockNum; ++i) {
                auto dstScale = scalePtr + i * ocUp4;
                auto dstZero  = zeroPtr + i * ocUp4;
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstScale[j] = quanInfoPtr[scaleIndex];
                    dstZero[j] = (float)originOffset * dstScale[j];
                }
            }
        }
        ::memcpy(biasPtr, convOp->bias()->data(), oc * QUANT_INFO_BYTES);
    }

    int n = oc;
    int k = ic;
    int packedWeightSize = kai.getRhsPackedSize(mAccelType, n, k, blkSize);

    //Alloc packed weight tensor.
    mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
    bool success = backend->onAcquireBuffer(mWeightInt8.get(), Backend::STATIC);

    if (!success) {
        MNN_ERROR("Out of static memory!\n");
        return;
    }

    size_t paraNum = scaleSize;
    float *scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
    float *zeroPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + paraNum;
    float *biasPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + 2 * paraNum;
    //Reload some parameters to fit ukernels' layout.
    auto quanInfoPtr = quanCommon->alpha.get();
    auto alphaSize = quanCommon->alpha.size();
    if(bAsym) {
        for(int i = 0; i < paraNum; i++) {
            if(i*2 >= alphaSize){
                zeroPtr[i] = 0;
                scalePtr[i] = 0;
            }
            else{
                zeroPtr[i] = quanInfoPtr[i * 2];
                scalePtr[i] = quanInfoPtr[i * 2 + 1];
            }
        }
    } else {
        if(blkSize != 0) {
            memcpy(scalePtr, (uint8_t*)quanInfoPtr, paraNum * sizeof(float));
        }
    }

    //Run rhs pack.
    auto weightPackedData = mWeightInt8->host<uint8_t>();
    kai.runRhsPack(mAccelType, 1, n, k, blkSize, 0/*unused*/,
                    (uint8_t*)quanCommon->weight.get(),
                    (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
                    weightPackedData);
    return;
}


KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op, const KleidiAIConvInt8& exe)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend), kai(exe.kai), mAccelType(exe.mAccelType),
    mWeightInt8(exe.mWeightInt8),mBlockNum(exe.mBlockNum),
      mTempIm2ColBuffer(exe.mTempIm2ColBuffer) {
}

KleidiAIConvInt8::~KleidiAIConvInt8() {
    // Do nothing
}

bool KleidiAIConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new KleidiAIConvInt8(bn, op, *this);
    if (!exe->valid()) {
        return false;
    }
    *dst = exe;
    return true;
}

// need
ErrorCode KleidiAIConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Initialize.
    auto output = outputs[0];
    auto core =static_cast<CPUBackend*>(backend())->functions();

    MNN_ASSERT(kai.isLoaded(mAccelType));
    const size_t m = inputs[0]->batch(); //lhs vector number.
    const size_t n = outputs[0]->channel(); //rhs vector number.
    const size_t k = inputs[0]->channel(); //vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    int packedSize = kai.getLhsQuantedPackedSize(mAccelType, m, k, blkSize);
    int elementSize = kai.isHalf() ? sizeof(__fp16) : sizeof(float);
    if(m > 1 && !kai.isLinear()) {
        int srcSize = m * k * elementSize;
        int dstSize = m * n * elementSize;
        int extraSize = srcSize > dstSize ? srcSize : dstSize;
        packedSize += extraSize;
    }

    //Split mTempIm2ColBuffer as two parts for linear/tile transfer:
    //Part0: Lhs_packed.
    //Part1: Lhs/Dst before transfer.
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({packedSize}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        MNN_ERROR("Out of dynamic memory!\n");
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode KleidiAIConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

     MNN_ASSERT(kai.isLoaded(mAccelType));
    const size_t m = input->batch(); //lhs vector number.
    const size_t n = output->channel(); //rhs vector number.
    const size_t k = input->channel(); //vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    bool bHalf = kai.isHalf();
    size_t elementSize = bHalf ? sizeof(__fp16) : sizeof(float);
    size_t lhsPackedSize = kai.getLhsQuantedPackedSize(mAccelType, m, k, blkSize);

    auto lhs = input->host<uint8_t>();
    auto lhsPacked = mTempIm2ColBuffer->host<int8_t>();
    auto rhsPacked = mWeightInt8->host<uint8_t>();
    auto dst = output->host<uint8_t>();

    uint8_t *linearLhs, *linearDst;
    if(m > 1 && !kai.isLinear()) {
        linearLhs = (uint8_t *)lhsPacked + lhsPackedSize;
        linearDst = linearLhs;
    } else {
        linearLhs = lhs;
        linearDst = dst;
    }

    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    int threadNeed, vecPerThread;

    //Dynamic quant pack lhs.
    if(m == 1) {
        kai.runLhsQuantPack(mAccelType, 1, k, blkSize, 1, linearLhs, lhsPacked);
    } else {
        if(!kai.isLinear()) {
            if(bHalf) {
                KleidiAIUtil::transferNC4HW4ToNCHW((__fp16 *)lhs, (__fp16 *)linearLhs, m, k);
            } else {
                KleidiAIUtil::transferNC4HW4ToNCHW((float *)lhs, (float *)linearLhs, m, k);
            }
        }

        vecPerThread = kai.getVecNumPerThread(m, threadNum, kai.getMr(mAccelType, m));
        threadNeed = m % vecPerThread == 0 ? m / vecPerThread : (m / vecPerThread + 1);
        size_t srcStride = vecPerThread * k * elementSize;

        auto BatchDynamicQuant = [=](int tId) {
            auto threadSrc = linearLhs + tId * srcStride;
            auto threadDst = lhsPacked + kai.getLhsQuantedPackedOffset(mAccelType, m, tId * vecPerThread, k, blkSize);
            int vecNum = (tId == threadNeed - 1) ? (m - vecPerThread * tId) : vecPerThread; //Last threadN may less than vecPerThread.
            kai.runLhsQuantPack(mAccelType, vecNum, k, blkSize, kai.getMr(mAccelType, m), threadSrc, threadDst);
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            BatchDynamicQuant((int)tId);
        }
        MNN_CONCURRENCY_END();
    }

    //Run matmul.
    if(kai.bSupportSme2() && mAccelType == KleidiAI::AccelType::QI4_SYM_CHNLQT) {
        //SME prefer running on single thread to obtain better performance/power consumption ratio.
        threadNum = 1;
    }

    vecPerThread = kai.getVecNumPerThread(n, threadNum, kai.getNStep(mAccelType));
    threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);

    auto ThreadFunction = [=](int tId) {
        auto threadRhsPacked = rhsPacked + kai.getRhsPackedOffset(mAccelType, tId * vecPerThread, k, blkSize);
        auto threadDst = linearDst + kai.getDstOffset(0, tId * vecPerThread, n, elementSize);
        int vecNum = (tId == threadNeed - 1) ? (n - vecPerThread * tId) : vecPerThread; //Last threadN may less than vecPerThread.
        float scalarMax = bHalf ? FLT16_MAX : FLT_MAX;
        kai.runMatmul(mAccelType, m, vecNum, k, blkSize, lhsPacked, threadRhsPacked, threadDst, n * elementSize, elementSize, scalarMax, -scalarMax);
    };

    MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
        ThreadFunction((int)tId);
    }
    MNN_CONCURRENCY_END();

    if(m > 1 && !kai.isLinear()) {
        if(bHalf) {
            KleidiAIUtil::transferNCHWToNC4HW4((__fp16 *)linearDst, (__fp16 *)dst, m, n);
        } else {
            KleidiAIUtil::transferNCHWToNC4HW4((float *)linearDst, (float *)dst, m, n);
        }
    }

    return NO_ERROR;
}

}
#endif