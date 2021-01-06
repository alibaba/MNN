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
    mResource.reset(new CPUConvolution::Resource);
    mResource->backend = b;
    mResource->mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, hPack), mSrcCount, hPack}));
    mValid = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    MNNPackForMatMul_B(mResource->mWeight->host<float>(), originWeight, outputCount, mSrcCount, true);
    mResource->mBias.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV((int)biasSize, 4), 4}));
    if (!(backend()->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC))) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
    ::memcpy(mResource->mBias->host<float>(), bias, biasSize * sizeof(float));
    auto remain = mResource->mBias->size() - biasSize * sizeof(float);
    if (remain > 0) {
        ::memset(mResource->mBias->host<float>() + biasSize, 0, remain);
    }
}
Convolution1x1Strassen::Convolution1x1Strassen(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b) : CPUConvolution(common, b) {
    mResource = resource;
}

Convolution1x1Strassen::~Convolution1x1Strassen() {
    // Do nothing
}

bool Convolution1x1Strassen::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new Convolution1x1Strassen(mResource, op->main_as_Convolution2D()->common(), bn);
    return true;
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
    auto batch       = input->batch();
    auto matrixSizeE = output->height() * output->width() * input->batch();
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
        mTempInputBatch.reset(Tensor::createDevice<float>(std::vector<int>{icC4, matrixSizeE, 4}));
        mTempOutputBatch.reset(Tensor::createDevice<float>(std::vector<int>{ocC4, matrixSizeE, 4}));
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
            mPretreatFunction = [outputPlane, icC4, batch, numberThread, this](const float *srcBatch, float *dstBatch) {
                MNN_CONCURRENCY_BEGIN(y, icC4) {
                    auto srcY = srcBatch + outputPlane * y * 4;
                    auto dstY = dstBatch + y * outputPlane * batch * 4;
                    for (int x = 0; x < batch; ++x) {
                        auto srcX = srcY + x * outputPlane * icC4 * 4;
                        auto dstX = dstY + x * outputPlane * 4;
                        ::memcpy(dstX, srcX, outputPlane * 4 * sizeof(float));
                    }
                }
                MNN_CONCURRENCY_END();
            };
        } else if (strideY == 1 && strideX == 1) {
            mPretreatFunction = [outputPlane, padY, padX, ow, oh, iw, ih, icC4, batch, this](const float *srcOrigin,
                                                                                float *dstOrigin) {
                ::memset(dstOrigin, 0, outputPlane * batch * sizeof(float) * 4 * icC4);
                MNN_CONCURRENCY_BEGIN(z, icC4) {
                    auto srcZ = srcOrigin + z * iw * ih * 4;
                    auto dstZ = dstOrigin + z * ow * oh * batch * 4;
                    for (int b = 0; b < batch; ++b) {
                        auto srcBatch = srcZ + b * iw * ih * icC4 * 4;
                        auto dstBatch = dstZ + b * ow * oh * 4;
                        for (int y = 0; y < ih; ++y) {
                            auto src = srcBatch + iw * y * 4;
                            auto dst = dstBatch + (ow * (y + padY) + padX) * 4;
                            ::memcpy(dst, src, iw * 4 * sizeof(float));
                        }
                    }
                }
                MNN_CONCURRENCY_END();
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
                                 oxCount, oyCount, batch, this](const float *srcOrigin, float *dstOrigin) {
                ::memset(dstOrigin, 0, outputPlane * batch * sizeof(float) * 4 * icC4);
                auto srcStride = strideX * 4;
                auto dstStride = 4;
                int syStart    = oyStart * strideY - padY;
                int sxStart    = oxStart * strideX - padX;
                MNN_CONCURRENCY_BEGIN(z, icC4) {
                    auto srcZ = srcOrigin + (z * iw * ih + syStart * iw + sxStart) * 4;
                    auto dstZ = dstOrigin + (z * ow * oh * batch + oyStart * ow + oxStart) * 4;
                    for (int b = 0; b < batch; ++b) {
                        auto srcBatch = srcZ + b * iw * ih * icC4 * 4;
                        auto dstBatch = dstZ + b * ow * oh * 4;
                        for (int y = 0; y < oyCount; ++y) {
                            auto dstY = dstBatch + y * ow * 4;
                            auto srcY = srcBatch + y * strideY * iw * 4;
                            MNNCopyC4WithStride(srcY, dstY, srcStride, dstStride, oxCount);
                        }
                    }
                }
                MNN_CONCURRENCY_END();
            };
        }
    }
    auto memoryPool = ((CPUBackend *)backend())->getBufferAllocator();
    memoryPool->barrierBegin();
    std::shared_ptr<void> __a(nullptr, [memoryPool](void *) { memoryPool->barrierEnd(); });
    int maxDepth = 5;
    if (matrixSizeE > CONVOLUTION_TILED_NUMBER * 8 * numberThread && matrixSizeE > ocC4) {
        // Divide in plane, in this case the divide equal numberThread
        int divideStep = UP_DIV(matrixSizeE, numberThread);
        mUnits.resize(numberThread);
        for (int i = 0; i < numberThread; ++i) {
            int planeStart = i * divideStep;
            int planeEnd   = std::min(planeStart + divideStep, matrixSizeE);
            int planeSize  = planeEnd - planeStart;
            Unit &unit     = mUnits[i];
            if (planeSize <= 0) {
                unit.mValid = false;
                continue;
            }
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth));
            unit.mTempInput.reset(
                Tensor::create<float>(std::vector<int>{icC4, planeSize, 4}, inputPtr + 4 * planeStart));
            unit.mTempInput->setStride(0, matrixSizeE * 4);
            unit.mTempOutput.reset(
                Tensor::create<float>(std::vector<int>{ocC4, planeSize, 4}, outputPtr + 4 * planeStart));
            unit.mTempOutput->setStride(0, matrixSizeE * 4);
            unit.mTempInputVector  = std::vector<Tensor *>{unit.mTempInput.get(), mResource->mWeight.get(), mResource->mBias.get()};
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
            auto ocWeightSize = std::min(UP_DIV((ocSize * 4), hPack), mResource->mWeight->length(0) - ocStartWeight);
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth));
            unit.mTempInput.reset(Tensor::create<float>(std::vector<int>{icC4, matrixSizeE, 4}, inputPtr));
            unit.mTempBias.reset(Tensor::create<float>({ocSize, 1, 4}, mResource->mBias->host<float>() + 4 * ocStart));
            unit.mTempOutput.reset(
                Tensor::create<float>(std::vector<int>{ocSize, matrixSizeE, 4}, outputPtr + 4 * matrixSizeE * ocStart));
            unit.mTempWeight.reset(Tensor::create<float>(std::vector<int>{ocWeightSize, ic, hPack},
                                                         mResource->mWeight->host<float>() + hPack * ic * ocStartWeight));
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
    mPretreatFunction(input->host<float>(), mTempInputBatch->host<float>());
    MNN_CONCURRENCY_BEGIN(tId, size) {
        auto &unit = mUnits[tId];
        if (unit.mValid) {
            unit.mStracssenComputor->onExecute();
        }
    }
    MNN_CONCURRENCY_END();

    auto batch       = input->batch();
    auto outputPlane = output->height() * output->width();
    auto ocC4        = UP_DIV(output->channel(), 4);
    MNN_CONCURRENCY_BEGIN(y, ocC4) {
        auto srcY = mTempOutputBatch->host<float>() + outputPlane * y * 4 * batch;
        auto dstY = output->host<float>() + y * outputPlane * 4;
        for (int x = 0; x < batch; ++x) {
            auto srcX = srcY + x * outputPlane * 4;
            auto dstX = dstY + x * outputPlane * ocC4 * 4;
            ::memcpy(dstX, srcX, outputPlane * 4 * sizeof(float));
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
} // namespace MNN
