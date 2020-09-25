//
//  Convolution1x1Strassen.cpp
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Convolution1x1Strassen.hpp"
#include <string.h>
#include "core/BufferAllocator.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
namespace MNN {
Convolution1x1Strassen::Convolution1x1Strassen(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                               size_t originWeightSize, const float *bias, size_t biasSize)
    : CPUConvolution(common, b) {
    auto outputCount = (int)biasSize;
    auto mSrcCount   = (int)originWeightSize / outputCount;
    int ePack, lPack, hPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, hPack), mSrcCount, hPack}));
    mValid = b->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    ::memset(mWeight->host<float>(), 0, mWeight->size());
    MNNPackForMatMul_B(mWeight->host<float>(), originWeight, outputCount, mSrcCount, true);

    mBias.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, 4), 4}));
    mValid = b->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));
}

Convolution1x1Strassen::~Convolution1x1Strassen() {
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

ErrorCode Convolution1x1Strassen::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    int ePack, lPack, hPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);

    auto CONVOLUTION_TILED_NUMBER = ePack;
    auto input       = inputs[0];
    auto output      = outputs[0];
    int numberThread = ((CPUBackend *)backend())->threadNumber();
    auto ic = input->channel();
    auto icC4        = UP_DIV(ic, 4);
    auto ocC4        = UP_DIV(output->channel(), 4);
    auto outputPlane = output->height() * output->width();
    mUnits.clear();
    auto inputPtr  = input->host<float>();
    auto outputPtr = output->host<float>();
    mTempOutputBatch.reset();
    mTempInputBatch.reset();
    std::shared_ptr<char> __autoFunction;
    auto padY     = mPadY;
    auto padX     = mPadX;
    auto strideX  = mCommon->strideX();
    auto strideY  = mCommon->strideY();
    mNeedPretreat = input->batch() > 1 || (!(padX == 0 && padY == 0 && strideY == 1 && strideX == 1));
    auto postParameters = getPostParameters();
    if (mNeedPretreat) {
        mTempInputBatch.reset(Tensor::createDevice<float>(std::vector<int>{icC4, outputPlane, 4}));
        mTempOutputBatch.reset(Tensor::createDevice<float>(std::vector<int>{ocC4, outputPlane, 4}));
        bool success = backend()->onAcquireBuffer(mTempOutputBatch.get(), Backend::DYNAMIC);
        success      = success && backend()->onAcquireBuffer(mTempInputBatch.get(), Backend::DYNAMIC);
        if (!success) {
            return OUT_OF_MEMORY;
        }
        inputPtr       = mTempInputBatch->host<float>();
        outputPtr      = mTempOutputBatch->host<float>();
        __autoFunction = std::shared_ptr<char>(nullptr, [this](void *ptr) {
            backend()->onReleaseBuffer(mTempOutputBatch.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mTempInputBatch.get(), Backend::DYNAMIC);
        });
        auto ow        = output->width();
        auto oh        = output->height();
        auto iw        = input->width();
        auto ih        = input->height();
        if (padX == 0 && padY == 0 && strideY == 1 && strideX == 1) {
            mPretreatFunction = [outputPlane, icC4](const float *srcBatch, float *dstBatch) {
                ::memcpy(dstBatch, srcBatch, outputPlane * sizeof(float) * 4 * icC4);
            };
        } else if (strideY == 1 && strideX == 1) {
            mPretreatFunction = [outputPlane, padY, padX, ow, oh, iw, ih, icC4](const float *srcBatch,
                                                                                float *dstBatch) {
                ::memset(dstBatch, 0, outputPlane * sizeof(float) * 4 * icC4);
                for (int z = 0; z < icC4; ++z) {
                    auto srcZ = srcBatch + z * iw * ih * 4;
                    auto dstZ = dstBatch + z * ow * oh * 4;
                    for (int y = 0; y < ih; ++y) {
                        auto src = srcZ + iw * y * 4;
                        auto dst = dstZ + (ow * (y + padY) + padX) * 4;
                        ::memcpy(dst, src, iw * 4 * sizeof(float));
                    }
                }
            };
        } else {
            int oyStart, oyEnd, oxStart, oxEnd;
            for (oyStart = 0; oyStart * strideY - padY < 0; ++oyStart) {
                // do nothing
            }
            for (oyEnd = oh - 1; oyEnd * strideY - padY >= ih; --oyEnd) {
                // do nothing
            }
            for (oxStart = 0; oxStart * strideX - padX < 0; ++oxStart) {
                // do nothing
            }
            for (oxEnd = ow - 1; oxEnd * strideX - padX >= iw; --oxEnd) {
                // do nothing
            }
            int oyCount       = oyEnd - oyStart + 1;
            int oxCount       = oxEnd - oxStart + 1;
            mPretreatFunction = [outputPlane, padY, padX, strideX, strideY, ow, oh, iw, ih, icC4, oxStart, oyStart,
                                 oxCount, oyCount](const float *srcBatch, float *dstBatch) {
                ::memset(dstBatch, 0, outputPlane * sizeof(float) * 4 * icC4);
                auto srcStride = strideX * 4;
                auto dstStride = 4;
                int syStart    = oyStart * strideY - padY;
                int sxStart    = oxStart * strideX - padX;
                for (int z = 0; z < icC4; ++z) {
                    auto srcZ = srcBatch + (z * iw * ih + syStart * iw + sxStart) * 4;
                    auto dstZ = dstBatch + (z * ow * oh + oyStart * ow + oxStart) * 4;
                    for (int y = 0; y < oyCount; ++y) {
                        auto dstY = dstZ + y * ow * 4;
                        auto srcY = srcZ + y * strideY * iw * 4;
                        MNNCopyC4WithStride(srcY, dstY, srcStride, dstStride, oxCount);
                    }
                }
            };
        }
    }
    auto memoryPool = ((CPUBackend *)backend())->getBufferAllocator();
    memoryPool->barrierBegin();
    std::shared_ptr<void> __a(nullptr, [memoryPool](void *) { memoryPool->barrierEnd(); });
    int maxDepth = 5;
    if (outputPlane > CONVOLUTION_TILED_NUMBER * 8 * numberThread && outputPlane > ocC4) {
        // Divide in plane, in this case the divide equal numberThread
        int divideStep = UP_DIV(outputPlane, numberThread);
        mUnits.resize(numberThread);
        for (int i = 0; i < numberThread; ++i) {
            int planeStart = i * divideStep;
            int planeEnd   = std::min(planeStart + divideStep, outputPlane);
            int planeSize  = planeEnd - planeStart;
            Unit &unit     = mUnits[i];
            if (planeSize <= 0) {
                unit.mValid = false;
                continue;
            }
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth));
            unit.mTempInput.reset(
                Tensor::create<float>(std::vector<int>{icC4, planeSize, 4}, inputPtr + 4 * planeStart));
            unit.mTempInput->setStride(0, outputPlane * 4);
            unit.mTempOutput.reset(
                Tensor::create<float>(std::vector<int>{ocC4, planeSize, 4}, outputPtr + 4 * planeStart));
            unit.mTempOutput->setStride(0, outputPlane * 4);
            unit.mTempInputVector  = std::vector<Tensor *>{unit.mTempInput.get(), mWeight.get(), mBias.get()};
            unit.mTempOutputVector = std::vector<Tensor *>{unit.mTempOutput.get()};
            memoryPool->beginGroup();
            std::shared_ptr<void> __b(nullptr, [memoryPool](void *) { memoryPool->endGroup(); });
            unit.mStracssenComputor->onReset();
            auto code = unit.mStracssenComputor->onEncode(unit.mTempInputVector, unit.mTempOutputVector, postParameters);
            if (NO_ERROR != code) {
                return code;
            }
        }
    } else {
        // Divide in ocC4
        auto hDiv = MNNGetC4DivNumber(hPack);
        auto ocDiv = UP_DIV(ocC4, hDiv);
        numberThread   = std::min(numberThread, ocDiv);
        int divideStep = (ocDiv / numberThread) * hDiv;
        mUnits.resize(numberThread);
        for (int i = 0; i < numberThread; ++i) {
            int ocStart = i * divideStep;
            int ocSize  = divideStep;
            if (i == numberThread - 1) {
                ocSize = ocC4 - i * divideStep;
            }
            Unit &unit  = mUnits[i];
            if (ocSize <= 0) {
                unit.mValid = false;
                continue;
            }
            auto ocStartWeight = (ocStart * 4) / hPack;
            auto ocWeightSize = std::min(UP_DIV((ocSize * 4), hPack), mWeight->length(0) - ocStartWeight);
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth));
            unit.mTempInput.reset(Tensor::create<float>(std::vector<int>{icC4, outputPlane, 4}, inputPtr));
            unit.mTempBias.reset(Tensor::create<float>({ocSize, 1, 4}, mBias->host<float>() + 4 * ocStart));
            unit.mTempOutput.reset(
                Tensor::create<float>(std::vector<int>{ocSize, outputPlane, 4}, outputPtr + 4 * outputPlane * ocStart));
            unit.mTempWeight.reset(Tensor::create<float>(std::vector<int>{ocWeightSize, ic, hPack},
                                                         mWeight->host<float>() + hPack * ic * ocStartWeight));
            unit.mTempInputVector  = std::vector<Tensor *>{unit.mTempInput.get(), unit.mTempWeight.get(), unit.mTempBias.get()};
            unit.mTempOutputVector = std::vector<Tensor *>{unit.mTempOutput.get()};
            memoryPool->beginGroup();
            std::shared_ptr<void> __b(nullptr, [memoryPool](void *) { memoryPool->endGroup(); });
            unit.mStracssenComputor->onReset();
            auto code = unit.mStracssenComputor->onEncode(unit.mTempInputVector, unit.mTempOutputVector, postParameters);
            if (NO_ERROR != code) {
                return code;
            }
        }
    }
    return NO_ERROR;
}

ErrorCode Convolution1x1Strassen::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto size   = mUnits.size();
    auto input  = inputs[0];
    auto output = outputs[0];

    if (!mNeedPretreat) {
        MNN_CONCURRENCY_BEGIN(tId, size) {
            auto &unit = mUnits[tId];
            if (unit.mValid) {
                unit.mStracssenComputor->onExecute();
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        mPretreatFunction(input->host<float>() + batchIndex * input->stride(0), mTempInputBatch->host<float>());
        MNN_CONCURRENCY_BEGIN(tId, size) {
            auto &unit = mUnits[tId];
            if (unit.mValid) {
                unit.mStracssenComputor->onExecute();
            }
        }
        MNN_CONCURRENCY_END();

        ::memcpy(output->host<float>() + batchIndex * output->stride(0), mTempOutputBatch->host<float>(),
                 output->stride(0) * sizeof(float));
    }
    return NO_ERROR;
}
} // namespace MNN
