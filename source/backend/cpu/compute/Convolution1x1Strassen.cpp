//
//  Convolution1x1Strassen.cpp
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Convolution1x1Strassen.hpp"
#include "DenseConvolutionTiledExecutor.hpp"
#include <string.h>
#include "core/BufferAllocator.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
#include "CommonOptFunction.h"
#include "core/TensorUtils.hpp"

namespace MNN {
#ifndef MNN_REDUCE_SIZE
Convolution1x1Strassen::Convolution1x1Strassen(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                               size_t originWeightSize, const float *bias, size_t biasSize)
    : CPUConvolution(common, b) {
    auto outputCount = (int)biasSize;
    int ePack, lPack, hPack;
    auto core = static_cast<CPUBackend*>(b)->functions();
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    mResource.reset(new CPUConvolution::Resource);
    mResource->backend = b;
    auto mSrcCount   = (int)originWeightSize / outputCount;
    if (!mResource->copyBiasAlign(bias, (int)biasSize)) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
    // Use Float Weight.
    mResource->mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, hPack), UP_DIV(mSrcCount, lPack) * lPack, hPack}));
    mValid = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    if (b->getRuntime()->hint().useCachedMmap > 1) {
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
#ifdef MNN_KLEIDIAI_ENABLED
        if (core->bytes == 2) {
            if (!KleidiAI::mKaiInitialized) {
                KleidiAI& kai = KleidiAI::getInstance(*MNNGetCPUInfo(), true, false);
            }
            KleidiAI::AccelType accelType = KleidiAI::AccelType::FP16;
            KleidiAI& kai = KleidiAI::getInstance();
            if (!kai.isLoaded(accelType)) {
                kai.setLoaded(accelType);
                kai.printInfo(accelType);
            }

            if (kai.canAccelerate(accelType)) {
                mAccelType = accelType;
                AutoRelease<Tensor> tempBiasTensor(Tensor::createDevice<float>({outputCount}));
                mValid = b->onAcquireBuffer(tempBiasTensor.get(), Backend::STATIC);
                if (!mValid) {
                    b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
                    MNN_ERROR("Not Enough Memory\n");
                    return;
                }
                core->MNNFp32ToLowp(bias, tempBiasTensor->host<int16_t>(), outputCount);

                int packedSize = kai.getRhsPackedSize(mAccelType, outputCount, mSrcCount, 0);
                //Alloc packed weight tensor.
                mResource->mWeight.reset(Tensor::createDevice<float>({packedSize}));
                bool success = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
                if (!success) {
                    b->onReleaseBuffer(tempBiasTensor.get(), Backend::STATIC);
                    b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
                    MNN_ERROR("Out of static memory!\n");
                    return;
                }

                //Run rhs pack.
                kai.runRhsPack(mAccelType, 1, outputCount, mSrcCount, 0, mSrcCount * sizeof(__fp16),
                               tempTensor->host<void>(), nullptr, nullptr, tempBiasTensor->host<void>(),
                               mResource->mWeight->host<void>());
                b->onReleaseBuffer(tempBiasTensor.get(), Backend::STATIC);
            } else {
                core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), tempTensor->host<float>(), outputCount, mSrcCount, true);
            }
        } else {
            core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), tempTensor->host<float>(), outputCount, mSrcCount, true);
        }
#else
        core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), tempTensor->host<float>(), outputCount, mSrcCount, true);
#endif
        b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
    } else {
#ifdef MNN_KLEIDIAI_ENABLED
        if (!KleidiAI::mKaiInitialized) {
            KleidiAI& kai = KleidiAI::getInstance(*MNNGetCPUInfo(), false, false);
        }

        KleidiAI::AccelType accelType = KleidiAI::AccelType::FP32;
        KleidiAI& kai = KleidiAI::getInstance();
        if(!kai.isLoaded(accelType)) {
            kai.setLoaded(accelType);
            kai.printInfo(accelType);
        }

        if (kai.canAccelerate(accelType)) {
            mAccelType = accelType;
            int packedSize = kai.getRhsPackedSize(mAccelType, outputCount, mSrcCount, 0);
            //Alloc packed weight tensor.
            mResource->mWeight.reset(Tensor::createDevice<float>({packedSize}));
            bool success = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
            if (!success) {
                MNN_ERROR("Out of static memory!\n");
                return;
            }

            //Run rhs pack.
            kai.runRhsPack(mAccelType, 1, outputCount, mSrcCount, 0, mSrcCount * sizeof(float),
                        originWeight, nullptr, nullptr, bias, mResource->mWeight->host<void>());
        } else {
            core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), originWeight, outputCount, mSrcCount, true);
        }
#else
        core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), originWeight, outputCount, mSrcCount, true);
#endif
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
    auto exe = new Convolution1x1Strassen(mResource, op->main_as_Convolution2D()->common(), bn);
#ifdef MNN_KLEIDIAI_ENABLED
    exe->mAccelType = this->mAccelType;
#endif
    *dst = exe;
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
    const int numberThread = ((CPUBackend *)backend())->threadNumber();
    auto ic = input->channel();
    auto oc = output->channel();
    auto ocC4        = UP_DIV(oc, core->pack);
    auto batch       = input->batch();
    auto matrixSizeE = output->height() * output->width() * input->batch();
    mUnits.clear();
    std::shared_ptr<char> __autoFunction;
    auto postParameters = getPostParameters();
    auto memoryPool = ((CPUBackend *)backend())->getBufferAllocator();
    memoryPool->barrierBegin();
    std::shared_ptr<void> __a(nullptr, [memoryPool](void *) { memoryPool->barrierEnd(); });
    int maxDepth = 5;
    auto icAlign = UP_DIV(ic, lPack) * lPack;
    auto weightTensor = mResource->mWeight.get();

#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if (kai.canAccelerate(mAccelType)) {
        if (batch != 1) {
            int packedSize = kai.getLhsPackedSize(mAccelType, batch, ic);

            mInputResource.reset(Tensor::createDevice<float>({packedSize}));
            bool success = backend()->onAcquireBuffer(mInputResource.get(), Backend::DYNAMIC);
            if (!success) {
                MNN_ERROR("Out of dynamic memory!\n");
                return OUT_OF_MEMORY;
            }

            backend()->onReleaseBuffer(mInputResource.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }
#endif

    mWeightBytes = bytes;
    if (matrixSizeE > CONVOLUTION_TILED_NUMBER * 8 * numberThread && matrixSizeE > ocC4) {
        std::vector<int> divides(numberThread+1);
        divides[0] = 0;
        static_cast<CPUBackend *>(backend())->computeDivideSizes(matrixSizeE, divides.data()+1);
        mUnits.resize(numberThread);
        for (int i = 0; i < numberThread; ++i) {
            int planeStart = divides[i];
            int planeEnd   = divides[i+1];
            int planeSize  = planeEnd - planeStart;
            Unit &unit     = mUnits[i];
            if (planeSize <= 0) {
                unit.mValid = false;
                continue;
            }
            unit.offset[1] = 0;
            unit.offset[2] = 0;
            unit.offset[0] = core->pack * planeStart * bytes;
            unit.offset[3] = core->pack * planeStart * bytes;
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth));
            int e = planeSize;
            int l = ic;
            int h = oc;
            uint8_t* aPtr = nullptr;
            auto bPtr = TensorUtils::getDescribeOrigin(weightTensor)->mem->chunk();;
            uint8_t* cPtr = nullptr;
            auto biasPtr = TensorUtils::getDescribeOrigin(mResource->mBias.get())->mem->chunk();
            memoryPool->beginGroup();
            auto code = unit.mStracssenComputor->onEncode(e, l, h, matrixSizeE * core->pack, UP_DIV(l, lPack) * lPack * hPack, matrixSizeE * core->pack, aPtr, bPtr, cPtr, true, biasPtr, postParameters);
            if (NO_ERROR != code) {
                memoryPool->endGroup();
                return code;
            }
            memoryPool->endGroup();
        }
    } else {
        // Divide in ocC4
        auto hDiv = 1;
        if (hPack > core->pack) {
            hDiv = hPack / core->pack;
        }
        auto ocDiv = UP_DIV(ocC4, hDiv);
        std::vector<int> divides(numberThread+1);
        divides[0] = 0;
        static_cast<CPUBackend *>(backend())->computeDivideSizes(ocDiv, divides.data()+1);
        mUnits.resize(numberThread);
        for (int i = 0; i < numberThread; ++i) {
            int ocStart = divides[i] * hDiv;
            int ocEnd = divides[i+1] * hDiv;
            if (ocEnd >= ocC4) {
                ocEnd = ocC4;
            }
            int ocSize  = ocEnd - ocStart;
            Unit &unit  = mUnits[i];
            if (ocSize <= 0) {
                unit.mValid = false;
                continue;
            }
            auto ocStartWeight = (ocStart * core->pack) / hPack;
            auto ocWeightSize = std::min(UP_DIV((ocSize * core->pack), hPack), mResource->mWeight->length(0) - ocStartWeight);
            unit.offset[1] = hPack * icAlign * ocStartWeight * mWeightBytes;
            unit.offset[2] = core->pack * ocStart * bytes;
            unit.offset[0] = 0;
            unit.offset[3] = core->pack * matrixSizeE * ocStart * bytes;

            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth));
            int e = matrixSizeE;
            int l = ic;
            int h = std::min(ocSize * core->pack, ocWeightSize * hPack);
            uint8_t* aPtr = nullptr;
            auto bPtr = TensorUtils::getDescribeOrigin(mResource->mWeight.get())->mem->chunk() + hPack * icAlign * ocStartWeight * mWeightBytes;
            uint8_t* cPtr = nullptr;
            auto biasPtr = TensorUtils::getDescribeOrigin(mResource->mBias.get())->mem->chunk() + core->pack * ocStart * bytes;
            memoryPool->beginGroup();
            auto code = unit.mStracssenComputor->onEncode(e, l, h, matrixSizeE * core->pack, UP_DIV(l, lPack) * lPack * hPack, matrixSizeE * core->pack, aPtr, bPtr, cPtr, true, biasPtr, postParameters);
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
    auto inputPtr = input->host<uint8_t>();
    auto outputPtr = output->host<uint8_t>();
    auto weightPtr = mResource->mWeight->host<uint8_t>();
    auto biasPtr = mResource->mBias->host<uint8_t>();

#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if (kai.canAccelerate(mAccelType)) {
        const size_t m = input->batch(); //lhs vector number.
        const size_t n = output->channel(); //rhs vector number.
        const size_t k = input->channel(); //vector size.
        auto lhsPacked = inputPtr;
        auto dst = output->host<uint8_t>();
        size_t elementSize = kai.isFP16() ? sizeof(__fp16) : sizeof(float);
        if(m != 1) {
            lhsPacked = mInputResource->host<uint8_t>();
            kai.runLhsPack(mAccelType, m, k, 0, inputPtr, k * elementSize, lhsPacked);
        }
        auto postPtr = getPostParameters();
        kai.runMatmul(mAccelType, m, n, k, 0, lhsPacked, weightPtr, dst, n * elementSize, elementSize, postPtr[3], postPtr[2]);
        return NO_ERROR;
    }
#endif
    MNN_CONCURRENCY_BEGIN(tId, size) {
        auto &unit = mUnits[tId];
        if (unit.mValid) {
            unit.mStracssenComputor->onExecute(inputPtr + unit.offset[0], weightPtr + unit.offset[1], biasPtr + unit.offset[2], outputPtr + unit.offset[3]);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
#endif
} // namespace MNN
