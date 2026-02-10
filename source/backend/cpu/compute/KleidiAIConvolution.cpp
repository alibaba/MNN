//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef MNN_KLEIDIAI_ENABLED
#include "KleidiAIConvolution.hpp"
#include <string.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"

namespace MNN {
KleidiAIConvolution::KleidiAIConvolution(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                        size_t originWeightSize, const float *bias, size_t biasSize)
    : CPUConvolution(common, b) {

        auto outputCount = (int)biasSize;
        auto core = static_cast<CPUBackend*>(b)->functions();
        mResource.reset(new CPUConvolution::Resource);
        mResource->backend = b;
        auto mSrcCount   = (int)originWeightSize / outputCount;
        if (!mResource->copyBiasAlign(bias, (int)biasSize)) {
            MNN_ERROR("Not Enough Memory\n");
            mValid = false;
            return;
        }
        if (b->getRuntime()->hint().useCachedMmap > 1) {
            return;
        }
        KleidiAI& kai = KleidiAI::getInstance(*MNNGetCPUInfo());

        if (core->bytes == 2) {
            AutoRelease<Tensor> tempTensor(Tensor::createDevice<float>({outputCount * mSrcCount}));
            mValid = b->onAcquireBuffer(tempTensor.get(), Backend::STATIC);
            if (!mValid) {
                MNN_ERROR("Not Enough Memory\n");
                return;
            }
            core->MNNFp32ToLowp(originWeight, tempTensor->host<int16_t>(), outputCount * mSrcCount);

            KleidiAI::AccelType accelType = KleidiAI::AccelType::FP16;
            if (!kai.isLoaded(accelType)) {
                kai.setLoaded(accelType);
                kai.printInfo(accelType);
            }

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
            mResource->mWeight.reset(Tensor::createDevice<int8_t>({packedSize}));
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
            b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
        } else {
            KleidiAI::AccelType accelType = KleidiAI::AccelType::FP32;
            if(!kai.isLoaded(accelType)) {
                kai.setLoaded(accelType);
                kai.printInfo(accelType);
            }
            mAccelType = accelType;
            int packedSize = kai.getRhsPackedSize(mAccelType, outputCount, mSrcCount, 0);
            //Alloc packed weight tensor.
            mResource->mWeight.reset(Tensor::createDevice<int8_t>(std::vector<int>{packedSize}));
            mValid  = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
            if (!mValid) {
                MNN_ERROR("Out of static memory!\n");
                return;
            }

            //Run rhs pack.
            kai.runRhsPack(mAccelType, 1, outputCount, mSrcCount, 0, mSrcCount * sizeof(float),
                        originWeight, nullptr, nullptr, bias, mResource->mWeight->host<void>());
        }

}

KleidiAIConvolution::KleidiAIConvolution(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b) : CPUConvolution(common, b) {
    mResource = resource;
}

KleidiAIConvolution::~KleidiAIConvolution() {
    // Do nothing
}

bool KleidiAIConvolution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto exe = new KleidiAIConvolution(mResource, op->main_as_Convolution2D()->common(), bn);
    exe->mAccelType = this->mAccelType;
    *dst = exe;
    return true;
}

ErrorCode KleidiAIConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int bytes = core->bytes;
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto inputDes       = TensorUtils::getDescribe(inputs[0]);
    auto outputDes      = TensorUtils::getDescribe(outputs[0]);
    auto ic = input->channel();
    auto oc = output->channel();
    auto batch       = input->batch();
    auto b = backend();

    KleidiAI& kai = KleidiAI::getInstance(*MNNGetCPUInfo());
    auto inputOriginFmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto outputOriginFmt = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();
    if(inputOriginFmt != MNN_DATA_FORMAT_NHWC){
        mInputConvertBuffer.reset(Tensor::createDevice(std::vector<int>{input->batch(), input->height(), input->width(), input->channel()}, dataType, Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        mOutputConvertBuffer.reset(Tensor::createDevice(std::vector<int>{output->batch(), output->height(), output->width(), output->channel()}, dataType, Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }

    auto m = batch * input->width() * input->height();
    if (m != 1) {
        int packedSize = kai.getLhsPackedSize(mAccelType, m, ic);

        mInputResource.reset(Tensor::createDevice<float>({packedSize}));
        bool success = backend()->onAcquireBuffer(mInputResource.get(), Backend::DYNAMIC);
        if (!success) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }

        b->onReleaseBuffer(mInputResource.get(), Backend::DYNAMIC);
    }

    if(inputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
    }

    mPostParameters = getPostParameters();
    return NO_ERROR;
}

ErrorCode KleidiAIConvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto inputPtr = input->host<uint8_t>();
    auto weightPtr = mResource->mWeight->host<uint8_t>();
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();

    KleidiAI& kai = KleidiAI::getInstance(*MNNGetCPUInfo());
    const size_t m = input->batch() * input->width() * input->height(); //lhs vector number.
    const size_t n = output->channel(); //rhs vector number.
    const size_t k = input->channel(); //vector size.
    auto dst = output->host<uint8_t>();
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();
    size_t elementSize = core->bytes;
    auto b = backend();

    auto inputDes = TensorUtils::getDescribe(inputs[0]);
    if(inputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC){
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(input, mInputConvertBuffer.get(), core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
        inputPtr = mInputConvertBuffer->host<uint8_t>();
    }
    auto lhsPacked = inputPtr;
    if(m != 1) {
        lhsPacked = mInputResource->host<uint8_t>();
        kai.runLhsPack(mAccelType, m, k, 0, inputPtr, k * elementSize, lhsPacked);
    }

    auto outputDes = TensorUtils::getDescribe(outputs[0]);
    auto outputPtr = output->host<uint8_t>();
    if(outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC){
        outputPtr = mOutputConvertBuffer->host<uint8_t>();
    }

    kai.runMatmul(mAccelType, m, n, k, 0, lhsPacked, weightPtr, outputPtr, n * elementSize, elementSize, mPostParameters[3], mPostParameters[2]);

    if(outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC){
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(mOutputConvertBuffer.get(), output, core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}
} // namespace MNN
#endif //MNN_KLEIDIAI_ENABLED
