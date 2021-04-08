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
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
#include "CommonOptFunction.h"

namespace MNN {
Convolution1x1Strassen::Convolution1x1Strassen(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                               size_t originWeightSize, const float *bias, size_t biasSize)
    : CPUConvolution(common, b) {
    auto outputCount = (int)biasSize;
    auto mSrcCount   = (int)originWeightSize / outputCount;
    mResource.reset(new CPUConvolution::Resource);
    mResource->backend = b;
    if (!mResource->copyBiasAlign(bias, biasSize)) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
    auto core = static_cast<CPUBackend*>(b)->functions();
    int ePack, lPack, hPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    mResource->mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, hPack), UP_DIV(mSrcCount, lPack) * lPack, hPack}));
    mValid = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    if (core->bytes < 4) {
        AutoRelease<Tensor> tempTensor(Tensor::createDevice<float>({outputCount * mSrcCount}));
        mValid = b->onAcquireBuffer(tempTensor.get(), Backend::STATIC);
        if (!mValid) {
            MNN_ERROR("Not Enough Memory\n");
            return;
        }
        core->MNNFp32ToLowp(originWeight, tempTensor->host<int16_t>(), outputCount * mSrcCount);
        core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), tempTensor->host<float>(), outputCount, mSrcCount, true);
        b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
    } else {
        core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), originWeight, outputCount, mSrcCount, true);
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
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int ePack, lPack, hPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    int bytes = core->bytes;
    auto CONVOLUTION_TILED_NUMBER = ePack;
    auto input       = inputs[0];
    auto output      = outputs[0];
    int numberThread = ((CPUBackend *)backend())->threadNumber();
    auto ic = input->channel();
    auto oc = output->channel();
    auto icC4        = UP_DIV(ic, core->pack);
    auto ocC4        = UP_DIV(oc, core->pack);
    auto batch       = input->batch();
    auto matrixSizeE = output->height() * output->width() * input->batch();
    auto outputPlane = output->height() * output->width();
    mUnits.clear();
    auto inputPtr  = input->host<uint8_t>();
    auto outputPtr = output->host<uint8_t>();
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
        mTempInputBatch.reset(Tensor::createDevice<float>(std::vector<int>{icC4, matrixSizeE, core->pack}));
        mTempOutputBatch.reset(Tensor::createDevice<float>(std::vector<int>{ocC4, matrixSizeE, core->pack}));
        bool success = backend()->onAcquireBuffer(mTempOutputBatch.get(), Backend::DYNAMIC);
        success      = success && backend()->onAcquireBuffer(mTempInputBatch.get(), Backend::DYNAMIC);
        if (!success) {
            return OUT_OF_MEMORY;
        }
        inputPtr       = mTempInputBatch->host<uint8_t>();
        outputPtr      = mTempOutputBatch->host<uint8_t>();
        __autoFunction = std::shared_ptr<char>(nullptr, [this](void *ptr) {
            backend()->onReleaseBuffer(mTempOutputBatch.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mTempInputBatch.get(), Backend::DYNAMIC);
        });
        auto ow        = output->width();
        auto oh        = output->height();
        auto iw        = input->width();
        auto ih        = input->height();
        if (padX == 0 && padY == 0 && strideY == 1 && strideX == 1) {
            mPretreatFunction = [outputPlane, icC4, batch, numberThread, this, core](const uint8_t *srcBatch, uint8_t *dstBatch) {
                MNN_CONCURRENCY_BEGIN(y, icC4) {
                    auto srcY = srcBatch + outputPlane * y * core->pack * core->bytes;
                    auto dstY = dstBatch + y * outputPlane * batch * core->pack * core->bytes;
                    for (int x = 0; x < batch; ++x) {
                        auto srcX = srcY + x * outputPlane * icC4 * core->pack * core->bytes;
                        auto dstX = dstY + x * outputPlane * core->pack * core->bytes;
                        ::memcpy(dstX, srcX, outputPlane * core->pack * core->bytes);
                    }
                }
                MNN_CONCURRENCY_END();
            };
        } else if (strideY == 1 && strideX == 1) {
            mPretreatFunction = [outputPlane, padY, padX, ow, oh, iw, ih, icC4, batch, this, core](const uint8_t *srcOrigin,
                                                                                                    uint8_t *dstOrigin) {
                auto unitBytes = core->bytes * core->pack;
                ::memset(dstOrigin, 0, outputPlane * batch * unitBytes * icC4);
                MNN_CONCURRENCY_BEGIN(z, icC4) {
                    auto srcZ = srcOrigin + z * iw * ih * unitBytes;
                    auto dstZ = dstOrigin + z * ow * oh * batch * unitBytes;
                    for (int b = 0; b < batch; ++b) {
                        auto srcBatch = srcZ + b * iw * ih * icC4 * unitBytes;
                        auto dstBatch = dstZ + b * ow * oh * unitBytes;
                        for (int y = 0; y < ih; ++y) {
                            auto src = srcBatch + iw * y * unitBytes;
                            auto dst = dstBatch + (ow * (y + padY) + padX) * unitBytes;
                            ::memcpy(dst, src, iw * unitBytes);
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
                                 oxCount, oyCount, batch, this, core](const uint8_t *srcOrigin, uint8_t *dstOrigin) {
                ::memset(dstOrigin, 0, outputPlane * batch * core->bytes * core->pack * icC4);
                auto srcStride = strideX;
                auto dstStride = 1;
                int syStart    = oyStart * strideY - padY;
                int sxStart    = oxStart * strideX - padX;
                MNN_CONCURRENCY_BEGIN(z, icC4) {
                    auto srcZ = srcOrigin + (z * iw * ih + syStart * iw + sxStart) * core->bytes * core->pack;
                    auto dstZ = dstOrigin + (z * ow * oh * batch + oyStart * ow + oxStart) * core->bytes * core->pack;
                    for (int b = 0; b < batch; ++b) {
                        auto srcBatch = srcZ + b * iw * ih * icC4 * core->bytes * core->pack;
                        auto dstBatch = dstZ + b * ow * oh * core->bytes * core->pack;
                        for (int y = 0; y < oyCount; ++y) {
                            auto dstY = dstBatch + y * ow * core->bytes * core->pack;
                            auto srcY = srcBatch + y * strideY * iw * core->bytes * core->pack;
                            core->MNNCopyC4WithStride((const float*)(srcY), (float*)(dstY), strideX * core->pack, core->pack, oxCount);
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
    auto icAlign = UP_DIV(ic, lPack) * lPack;
    auto weightTensor = mResource->mWeight.get();
    AutoRelease<Tensor> tempWeight;
    if (icAlign != ic) {
        tempWeight.reset(Tensor::create<float>(std::vector<int>{oc, ic, hPack}, mResource->mWeight->host<uint8_t>()));
        weightTensor = tempWeight.get();
    }
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
            AutoRelease<Tensor> mTempInput(
                Tensor::create<float>(std::vector<int>{icC4, planeSize, core->pack}, inputPtr + core->pack * planeStart * bytes));
            mTempInput->setStride(0, matrixSizeE * core->pack);
            AutoRelease<Tensor> mTempOutput(
                Tensor::create<float>(std::vector<int>{ocC4, planeSize, core->pack}, outputPtr + core->pack * planeStart * bytes));
            mTempOutput->setStride(0, matrixSizeE * core->pack);
            unit.mTempInputVector  = std::vector<Tensor *>{mTempInput.get(), weightTensor, mResource->mBias.get()};
            unit.mTempOutputVector = std::vector<Tensor *>{mTempOutput.get()};
            memoryPool->beginGroup();
            unit.mStracssenComputor->onReset();
            auto code = unit.mStracssenComputor->onEncode(unit.mTempInputVector, unit.mTempOutputVector, postParameters);
            if (NO_ERROR != code) {
                memoryPool->endGroup();
                return code;
            }
            memoryPool->endGroup();
        }
    } else {
        // Divide in ocC4
        auto hDiv = hPack / core->pack;
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
            auto ocStartWeight = (ocStart * core->pack) / hPack;
            auto ocWeightSize = std::min(UP_DIV((ocSize * core->pack), hPack), mResource->mWeight->length(0) - ocStartWeight);
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth));
            AutoRelease<Tensor> mTempInput(Tensor::create<float>(std::vector<int>{icC4, matrixSizeE, core->pack}, inputPtr));
            AutoRelease<Tensor> mTempBias(Tensor::create<float>({ocSize, 1, core->pack}, mResource->mBias->host<uint8_t>() + core->pack * ocStart * bytes));
            AutoRelease<Tensor> mTempOutput(
                Tensor::create<float>(std::vector<int>{ocSize, matrixSizeE, core->pack}, outputPtr + core->pack * matrixSizeE * ocStart * bytes));
            AutoRelease<Tensor> mTempWeight(Tensor::create<float>(std::vector<int>{ocWeightSize, ic, hPack},
                                                         mResource->mWeight->host<uint8_t>() + hPack * icAlign * ocStartWeight * bytes));
            unit.mTempInputVector  = std::vector<Tensor *>{mTempInput.get(), mTempWeight.get(), mTempBias.get()};
            unit.mTempOutputVector = std::vector<Tensor *>{mTempOutput.get()};
            memoryPool->beginGroup();
            unit.mStracssenComputor->onReset();
            auto code = unit.mStracssenComputor->onEncode(unit.mTempInputVector, unit.mTempOutputVector, postParameters);
            if (NO_ERROR != code) {
                memoryPool->endGroup();
                return code;
            }
            memoryPool->endGroup();
        }
    }
    return NO_ERROR;
}

ErrorCode Convolution1x1Strassen::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto size   = mUnits.size();
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();

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
    int bytes = core->bytes;
    mPretreatFunction(input->host<uint8_t>(), mTempInputBatch->host<uint8_t>());
    MNN_CONCURRENCY_BEGIN(tId, size) {
        auto &unit = mUnits[tId];
        if (unit.mValid) {
            unit.mStracssenComputor->onExecute();
        }
    }
    MNN_CONCURRENCY_END();

    auto batch       = input->batch();
    auto outputPlane = output->height() * output->width();
    auto ocC4        = UP_DIV(output->channel(), core->pack);
    MNN_CONCURRENCY_BEGIN(y, ocC4) {
        auto srcY = mTempOutputBatch->host<uint8_t>() + outputPlane * y * core->pack * batch * bytes;
        auto dstY = output->host<uint8_t>() + y * outputPlane * core->pack * bytes;
        for (int x = 0; x < batch; ++x) {
            auto srcX = srcY + x * outputPlane * core->pack * bytes;
            auto dstX = dstY + x * outputPlane * ocC4 * core->pack * bytes;
            ::memcpy(dstX, srcX, outputPlane * core->pack * bytes);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
} // namespace MNN
