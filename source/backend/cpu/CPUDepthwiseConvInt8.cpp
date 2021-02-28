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
#define UNIT 4
#ifdef MNN_USE_SSE
#define BASIC_TYPE int16_t
#else
#define BASIC_TYPE int8_t
#endif
namespace MNN {
CPUDepthwiseConvInt8::CPUDepthwiseConvInt8(Backend* backend, const MNN::Convolution2D* dwConvParam)
    : Execution(backend), mCommon(dwConvParam->common()) {
    auto common               = dwConvParam->common();
    mResource.reset(new CPUConvInt8::ResourceInt8);
    mResource->mRelu                     = common->relu6() || common->relu();
    mResource->backend = backend;
    const int kx              = common->kernelX();
    const int ky              = common->kernelY();
    const int kernelSize      = kx * ky;
    const int outputCount     = common->outputCount();
    const int ocDivUnit       = UP_DIV(outputCount, UNIT);
    const int weightSizeAlign = ocDivUnit * UNIT * kernelSize;

    mResource->mWeightInt8.reset(Tensor::createDevice<BASIC_TYPE>({weightSizeAlign}));
    const auto *originWeight = dwConvParam->symmetricQuan()->weight()->data();
    auto allocRes = backend->onAcquireBuffer(mResource->mWeightInt8.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    auto weightPtr = mResource->mWeightInt8->host<BASIC_TYPE>();
    memset(weightPtr, 0, weightSizeAlign * sizeof(BASIC_TYPE));
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (dwConvParam->quanParameter() != nullptr) {
        quanCommon = ConvolutionCommon::load(dwConvParam->quanParameter(), false);
        originWeight = quanCommon->weight.get();
    }
    int cur = 0;
    for (int dz = 0; dz < outputCount; ++dz) {
        const int dzDivUnit = dz / UNIT;
        const int my        = dz % UNIT;
        auto dstDz          = weightPtr + dzDivUnit * kernelSize * UNIT;
        for (int i = 0; i < kernelSize; ++i) {
            dstDz[i * UNIT + my] = originWeight[cur++];
        }
    }

    mResource->mBiasInt32.reset(Tensor::createDevice<int32_t>({ocDivUnit * UNIT}));
    allocRes = backend->onAcquireBuffer(mResource->mBiasInt32.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    auto biasPtr = mResource->mBiasInt32->host<int32_t>();
    memset(biasPtr, 0, ocDivUnit * UNIT * sizeof(int32_t));
    memcpy(biasPtr, dwConvParam->symmetricQuan()->bias()->data(), outputCount * sizeof(int32_t));

    mResource->mScaleFloat.reset(Tensor::createDevice<int32_t>({ocDivUnit * UNIT}));
    allocRes = backend->onAcquireBuffer(mResource->mScaleFloat.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    auto scalePtr = mResource->mScaleFloat->host<float>();
    memset(scalePtr, 0, ocDivUnit * UNIT * sizeof(float));
    memcpy(scalePtr, dwConvParam->symmetricQuan()->scale()->data(), outputCount * sizeof(float));

    mResource->mInputZeroPoint = dwConvParam->symmetricQuan()->zeroPoint();
    mResource->mOutputZeroPoint = dwConvParam->symmetricQuan()->outputZeroPoint();
    mResource->mClampMin = dwConvParam->symmetricQuan()->clampMin();
    mResource->mClampMax = dwConvParam->symmetricQuan()->clampMax();
}

CPUDepthwiseConvInt8::~CPUDepthwiseConvInt8() {
    // Do nothing
}

bool CPUDepthwiseConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new CPUDepthwiseConvInt8(mResource, op->main_as_Convolution2D()->common(), bn);
    *dst = exe;
    return true;
}

ErrorCode CPUDepthwiseConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mCommon);

    int padX = std::get<0>(pads);
    int padY = std::get<1>(pads);
    mPads = std::make_pair(padX, padY);

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
    mInputPad.buffer().type = halide_type_of<BASIC_TYPE>();
    mInputPad.buffer().dimensions = 2;
    int paddedWidth = std::get<0>(pads) + std::get<2>(pads) + input->width();
    int paddedHeight = std::get<1>(pads) + std::get<3>(pads) + input->height();
    mPaddedSize = std::make_pair(paddedWidth, paddedHeight);
    mInputPad.setLength(0, mThreadNumber);
    mInputPad.setLength(1, paddedWidth * paddedHeight * UNIT);
    TensorUtils::setLinearLayout(&mInputPad);
    mStrides = std::make_pair(strideX, strideY);
    mDilates = std::make_pair(dilateX, dilateY);
    mKernels = std::make_pair(kernel_width, kernel_height);

    bool res = backend()->onAcquireBuffer(&mInputPad, Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mInputPad, Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPUDepthwiseConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input           = inputs[0];
    auto output          = outputs[0];
    const int batch      = input->batch();
    const int src_b_step = input->stride(0);
    const int dst_b_step = output->stride(0);

    const auto inputPtr = input->host<int8_t>();
    auto outputPtr      = output->host<int8_t>();
    const int dst_depth_quad = UP_DIV(output->channel(), 4);
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
        const auto inputPadPtr = mInputPad.host<BASIC_TYPE>() + mInputPad.stride(0) * tId;
        QuanPostTreatParameters quanParameters;
        if (mResource->mRelu) {
            quanParameters.maxValue = mResource->mClampMax;
            quanParameters.minValue = mResource->mOutputZeroPoint;
        } else {
            quanParameters.maxValue = mResource->mClampMax;
            quanParameters.minValue = mResource->mClampMin;
        }
        for (int index = tId; index < totalCount; index += mThreadNumber) {
            int bIndex = index / dst_depth_quad;
            int dz = index % dst_depth_quad;
            const auto srcOrigin = inputPtr + index * src_z_step;
            auto dstOrigin       = outputPtr + index * dst_z_step;
#ifdef MNN_USE_SSE
            auto inputPadPtrCopy = (int8_t*)inputPadPtr + mInputPad.stride(0);
#else
            auto inputPadPtrCopy = inputPadPtr;
#endif
            ::memset(inputPadPtrCopy, mResource->mInputZeroPoint, mInputPad.stride(0) * sizeof(int8_t));
            // Pad inputs
            for (int y = 0; y < src_height; ++y) {
                auto src = srcOrigin + y * src_width * 4;
                auto dst = inputPadPtrCopy + ((y + mPads.second) * mPaddedSize.first + mPads.first) * UNIT;
                ::memcpy(dst, src, src_width * 4 * sizeof(int8_t));
            }
#ifdef MNN_USE_SSE
            // Int8_t -> Int16_t
            MNNInt8ToInt16(inputPadPtr, inputPadPtrCopy, mInputPad.stride(0));
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
                auto dst_y          = dstOrigin + dy * dst_width * 4;
                MNNLineDepthWiseInt8AddBiasScaleUnit(dst_y, (const int8_t*)src_dy, (const int8_t*)weight_dz, &quanParameters, dst_width, mStrides.first * UNIT, mKernels.first, mKernels.second, mDilates.first * UNIT, mDilates.second * UNIT * mPaddedSize.first);
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
        return new CPUDepthwiseConvInt8(backend, op->main_as_Convolution2D());
    }
};

REGISTER_CPU_OP_CREATOR(CPUDepthwiseConvInt8Creator, OpType_DepthwiseConvInt8);

} // namespace MNN
