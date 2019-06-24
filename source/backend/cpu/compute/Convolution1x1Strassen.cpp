//
//  Convolution1x1Strassen.cpp
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Convolution1x1Strassen.hpp"
#include <string.h>
#include "BufferAllocator.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "ConvOpt.h"
#include "Macro.h"
namespace MNN {
Convolution1x1Strassen::Convolution1x1Strassen(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                               size_t originWeightSize, const float *bias, size_t biasSize)
    : CPUConvolution(common, b) {
    mPostFunction    = CPUConvolution::getPostFunction();
    auto outputCount = (int)biasSize;
    auto mSrcCount   = (int)originWeightSize / outputCount;
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, 4), UP_DIV(mSrcCount, 4), 16}));
    std::shared_ptr<Tensor> cacheWeight(
        Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, 4), UP_DIV(mSrcCount, 4), 16}));
    mValid =
        b->onAcquireBuffer(mWeight.get(), Backend::STATIC) && b->onAcquireBuffer(cacheWeight.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    ::memset(mWeight->host<float>(), 0, mWeight->size());
    CPUConvolution::reorderWeight(mWeight->host<float>(), originWeight, mSrcCount, outputCount, 1,
                                  cacheWeight->host<float>());
    b->onReleaseBuffer(cacheWeight.get(), Backend::STATIC);

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

ErrorCode Convolution1x1Strassen::onReleaseCache() {
    bool cacheB = ((CPUBackend *)backend())->memoryMode() == BackendConfig::Memory_High;
    if (cacheB) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
        mWeight = nullptr;
    }
    return NO_ERROR;
}

ErrorCode Convolution1x1Strassen::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input       = inputs[0];
    auto output      = outputs[0];
    int numberThread = ((CPUBackend *)backend())->threadNumber();
    auto icC4        = UP_DIV(input->channel(), 4);
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
    bool cacheB   = ((CPUBackend *)backend())->memoryMode() == BackendConfig::Memory_High;
    mNeedPretreat = input->batch() > 1 || (!(padX == 0 && padY == 0 && strideY == 1 && strideX == 1));
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
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), maxDepth, cacheB));
            unit.mTempInput.reset(
                Tensor::create<float>(std::vector<int>{icC4, planeSize, 4}, inputPtr + 4 * planeStart));
            unit.mTempInput->setStride(0, outputPlane * 4);
            unit.mTempOutput.reset(
                Tensor::create<float>(std::vector<int>{ocC4, planeSize, 4}, outputPtr + 4 * planeStart));
            unit.mTempOutput->setStride(0, outputPlane * 4);
            unit.mTempInputVector  = std::vector<Tensor *>{unit.mTempInput.get(), mWeight.get()};
            unit.mTempOutputVector = std::vector<Tensor *>{unit.mTempOutput.get()};
            memoryPool->beginGroup();
            std::shared_ptr<void> __b(nullptr, [memoryPool](void *) { memoryPool->endGroup(); });
            unit.mStracssenComputor->onReset();
            auto code = unit.mStracssenComputor->onEncode(unit.mTempInputVector, unit.mTempOutputVector);
            if (NO_ERROR != code) {
                return code;
            }
            unit.mPostExecutor = [&]() {
                auto dst    = unit.mTempOutput->host<float>();
                auto stride = unit.mTempOutput->stride(0);
                auto oZ4    = unit.mTempOutput->length(0);
                auto plane  = unit.mTempOutput->length(1);
                auto bias   = mBias->host<float>();
                for (int oz = 0; oz < oZ4; ++oz) {
                    auto dstOz = dst + stride * oz;
                    auto biasZ = bias + 4 * oz;
                    mPostFunction(dstOz, biasZ, plane, 1);
                }
            };
        }
    } else {
        // Divide in ocC4
        numberThread   = std::min(numberThread, ocC4);
        int divideStep = UP_DIV(ocC4, numberThread);
        mUnits.resize(numberThread);
        for (int i = 0; i < numberThread; ++i) {
            int ocStart = i * divideStep;
            int ocEnd   = std::min(ocStart + divideStep, ocC4);
            int ocSize  = ocEnd - ocStart;
            Unit &unit  = mUnits[i];
            if (ocSize <= 0) {
                unit.mValid = false;
                continue;
            }
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), maxDepth, cacheB));
            unit.mTempInput.reset(Tensor::create<float>(std::vector<int>{icC4, outputPlane, 4}, inputPtr));
            unit.mTempOutput.reset(
                Tensor::create<float>(std::vector<int>{ocSize, outputPlane, 4}, outputPtr + 4 * outputPlane * ocStart));
            unit.mTempWeight.reset(Tensor::create<float>(std::vector<int>{ocSize, icC4, 16},
                                                         mWeight->host<float>() + 16 * icC4 * ocStart));
            unit.mTempInputVector  = std::vector<Tensor *>{unit.mTempInput.get(), unit.mTempWeight.get()};
            unit.mTempOutputVector = std::vector<Tensor *>{unit.mTempOutput.get()};
            memoryPool->beginGroup();
            std::shared_ptr<void> __b(nullptr, [memoryPool](void *) { memoryPool->endGroup(); });
            unit.mStracssenComputor->onReset();
            auto code = unit.mStracssenComputor->onEncode(unit.mTempInputVector, unit.mTempOutputVector);
            if (NO_ERROR != code) {
                return code;
            }
            unit.mPostExecutor = [ocStart, ocSize, this, &unit]() {
                auto dst   = unit.mTempOutput->host<float>();
                auto plane = unit.mTempOutput->length(1);
                auto bias  = mBias->host<float>() + ocStart * 4;
                mPostFunction(dst, bias, plane, ocSize);
            };
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
                unit.mPostExecutor();
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
                unit.mPostExecutor();
            }
        }
        MNN_CONCURRENCY_END();
        ::memcpy(output->host<float>() + batchIndex * output->stride(0), mTempOutputBatch->host<float>(),
                 output->stride(0) * sizeof(float));
    }
    return NO_ERROR;
}
} // namespace MNN
