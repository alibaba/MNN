//
//  CPUDepthwiseConvInt8.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUDepthwiseConvInt8.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "compute/Int8FunctionsOpt.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include <math.h>
#ifdef MNN_USE_SSE
#define BASIC_TYPE int16_t
#else
#define BASIC_TYPE int8_t
#endif
namespace MNN {

CPUDepthwiseConvInt8::CPUDepthwiseConvInt8(Backend* backend, const Convolution2DCommon* common, std::shared_ptr<ResourceInt8> res): CPUConvolution(common, backend), mResource(res) {
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    const int kernelSize      = common->kernelX() * common->kernelY();
    const int outputCount     = common->outputCount();
    const int ocDivUnit       = UP_DIV(outputCount, UNIT);
    const int weightSizeAlign = ocDivUnit * UNIT * kernelSize;

    std::shared_ptr<Tensor> weight(Tensor::createDevice<BASIC_TYPE>({weightSizeAlign}));
    auto allocRes = backend->onAcquireBuffer(weight.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    auto originWeight = mResource->mWeightInt8->host<int8_t>();
    auto weightPtr = weight->host<BASIC_TYPE>();
    memset(weightPtr, 0, weightSizeAlign * sizeof(BASIC_TYPE));
    
    for (int dz = 0; dz < outputCount; ++dz) {
        const int dzDivUnit = dz / UNIT;
        const int my        = dz % UNIT;
        auto dstDz          = weightPtr + dzDivUnit * kernelSize * UNIT;
        for (int i = 0; i < kernelSize; ++i) {
            dstDz[i * UNIT + my] = (BASIC_TYPE)(originWeight[dz * kernelSize + i]);
        }
    }
    mResource->mWeightInt8.swap(weight);
    backend->onReleaseBuffer(weight.get(), Backend::STATIC);
    
#ifdef MNN_USE_SSE
    if (!mResource->offsets.empty()) {
        for (int i = 0; i < outputCount; ++i) {
            mResource->mBiasInt32->host<int32_t>()[i] -= mResource->offsets[i];
        }
    }
    mResource->offsets.clear();
#endif
}

CPUDepthwiseConvInt8::CPUDepthwiseConvInt8(Backend* backend, const Convolution2DCommon* common, const CPUDepthwiseConvInt8& exe) : CPUConvolution(common, backend), mResource(exe.mResource) {

}

CPUDepthwiseConvInt8::~CPUDepthwiseConvInt8() {
    // Do nothing
}

bool CPUDepthwiseConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new CPUDepthwiseConvInt8(bn, op->main_as_Convolution2D()->common(), *this);
    *dst = exe;
    return true;
}

ErrorCode CPUDepthwiseConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    std::vector<float> inputQuantInfo = TensorUtils::getQuantInfo(input);
    std::vector<float> outputQuantInfo = TensorUtils::getQuantInfo(output);
    mResource->updateInputOutputScale(inputQuantInfo, outputQuantInfo);
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mCommon);

    int padX = std::get<0>(pads);
    int padY = std::get<1>(pads);
    mPads = std::make_pair(padX, padY);
    
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    const int src_width      = input->width();
    const int src_height     = input->height();
    const int dst_width      = output->width();
    const int dst_height     = output->height();
    const int dst_depth_quad = UP_DIV(output->channel(), UNIT);
    const int strideY        = mCommon->strideY();
    const int strideX        = mCommon->strideX();
    const int dilateY        = mCommon->dilateY();
    const int dilateX        = mCommon->dilateX();
    const int kernel_height  = mCommon->kernelY();
    const int kernel_width   = mCommon->kernelX();
    const int threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    mThreadNumber          = std::min(threadNumber, dst_depth_quad * input->batch());
    int paddedWidth = std::get<0>(pads) + std::get<2>(pads) + input->width();
    int paddedHeight = std::get<1>(pads) + std::get<3>(pads) + input->height();
    mInputPad.reset(Tensor::createDevice<BASIC_TYPE>({mThreadNumber, paddedWidth * paddedHeight * UNIT}));
    mPaddedSize = std::make_pair(paddedWidth, paddedHeight);
    mStrides = std::make_pair(strideX, strideY);
    mDilates = std::make_pair(dilateX, dilateY);
    mKernels = std::make_pair(kernel_width, kernel_height);

    bool res = backend()->onAcquireBuffer(mInputPad.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mInputPad.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUDepthwiseConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    auto input           = inputs[0];
    auto output          = outputs[0];
    const int batch      = input->batch();
    const int src_b_step = input->stride(0);
    const int dst_b_step = output->stride(0);

    const auto inputPtr = input->host<int8_t>();
    auto outputPtr      = output->host<int8_t>();
    const int dst_depth_quad = UP_DIV(output->channel(), UNIT);
    const int src_width      = input->width();
    const int src_height     = input->height();
    const int dst_width      = output->width();
    const int dst_height     = output->height();
    const int dst_z_step     = dst_width * dst_height * UNIT;
    const int src_z_step     = src_width * src_height * UNIT;
    const auto weightPtr   = mResource->mWeightInt8->host<BASIC_TYPE>();
    const auto biasPtr     = mResource->mBiasInt32->host<int32_t>();
    const auto scalePtr    = mResource->mScaleFloat->host<float>();
    auto totalCount = batch * dst_depth_quad;

    MNN_CONCURRENCY_BEGIN(tId, mThreadNumber) {
        const auto inputPadPtr = mInputPad->host<BASIC_TYPE>() + mInputPad->stride(0) * tId;
        QuanPostTreatParameters quanParameters;
        if (mResource->mRelu) {
            quanParameters.maxValue = mResource->mClampMax;
            quanParameters.minValue = mResource->mOutputZeroPoint;
        } else {
            quanParameters.maxValue = mResource->mClampMax;
            quanParameters.minValue = mResource->mClampMin;
        }
        for (int index = tId; index < totalCount; index += mThreadNumber) {
            int dz = index / batch;
            const auto srcOrigin = inputPtr + index * src_z_step;
            auto dstOrigin       = outputPtr + index * dst_z_step;
#ifdef MNN_USE_SSE
            auto inputPadPtrCopy = (int8_t*)inputPadPtr + mInputPad->stride(0);
#else
            auto inputPadPtrCopy = inputPadPtr;
#endif
            ::memset(inputPadPtrCopy, mResource->mInputZeroPoint, mInputPad->stride(0) * sizeof(int8_t));
            // Pad inputs
            for (int y = 0; y < src_height; ++y) {
                auto src = srcOrigin + y * src_width * UNIT;
                auto dst = inputPadPtrCopy + ((y + mPads.second) * mPaddedSize.first + mPads.first) * UNIT;
                ::memcpy(dst, src, src_width * UNIT * sizeof(int8_t));
            }
#ifdef MNN_USE_SSE
            // Int8_t -> Int16_t
            MNNInt8ToInt16(inputPadPtr, inputPadPtrCopy, mInputPad->stride(0));
#endif
            // Compute
            const auto weight_dz = weightPtr + dz * mKernels.first * mKernels.second * UNIT;
            const auto bias_dz   = biasPtr + dz * UNIT;
            const auto scale_dz  = scalePtr + dz * UNIT;
            quanParameters.bias = bias_dz;
            quanParameters.scale = scale_dz;
            for (int dy = 0; dy < dst_height; ++dy) {
                const int srcStartY = dy * mStrides.second;
                const auto src_dy   = inputPadPtr + srcStartY * mPaddedSize.first * UNIT;
                auto dst_y          = dstOrigin + dy * dst_width * UNIT;
                core->ConvDepthwiseLineInt8(dst_y, (const int8_t*)src_dy, (const int8_t*)weight_dz, &quanParameters, dst_width, mStrides.first * UNIT, mKernels.first, mKernels.second, mDilates.first * UNIT, mDilates.second * UNIT * mPaddedSize.first);
            }
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUDepthwiseConvInt8Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        std::vector<float> inputQuantInfo;
        std::vector<float> outputQuantInfo;
        if (inputs.size() > 0) {
            inputQuantInfo = TensorUtils::getQuantInfo(inputs[0]);
            outputQuantInfo = TensorUtils::getQuantInfo(outputs[0]);
        }
        auto convOp = op->main_as_Convolution2D();
        auto res = CPUConvolution::makeResourceInt8(backend, convOp, inputQuantInfo, outputQuantInfo);
        return new CPUDepthwiseConvInt8(backend, convOp->common(), res);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDepthwiseConvInt8Creator, OpType_DepthwiseConvInt8);

} // namespace MNN
