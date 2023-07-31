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

void CPUDepthwiseConvInt8::fastDepthwiseInt8(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT = mPack;
    if (mUse3x3Kernel) {
        UNIT = 4;
    }

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
    const auto weightPtr     = mResource->mWeightInt8->host<BASIC_TYPE>();
    // const auto biasPtr       = mMutableResource.mBiasInt32->host<int32_t>();
    const auto biasPtr       = mBiasExtend.data();
    const auto scalePtr      = mMutableResource.mScaleFloat->host<float>();
    auto totalCount = batch * dst_depth_quad;

    MNN_CONCURRENCY_BEGIN(tId, mThreadNumber) {
        const auto inputPadPtr = mInputPad->host<BASIC_TYPE>() + mInputPad->stride(0) * tId;
        QuanPostTreatParameters quanParameters;
        if (mResource->mRelu) {
            quanParameters.maxValue = mMutableResource.mClampMax;
            quanParameters.minValue = mMutableResource.mOutputZeroPoint;
        } else {
            quanParameters.maxValue = mMutableResource.mClampMax;
            quanParameters.minValue = mMutableResource.mClampMin;
        }
        for (int index = tId; index < totalCount; index += mThreadNumber) {
            int dz = index / batch;
            const auto srcOrigin = inputPtr + index * src_z_step;
            auto dstOrigin       = outputPtr + index * dst_z_step;
#ifdef MNN_USE_SSE
            auto inputPadPtrCopy = (int8_t*)inputPadPtr + mInputPad->stride(0);
            ::memset(inputPadPtrCopy, mMutableResource.mInputZeroPoint + 128, mInputPad->stride(0) * sizeof(int8_t));
#else
            auto inputPadPtrCopy = inputPadPtr;
            ::memset(inputPadPtrCopy, mMutableResource.mInputZeroPoint, mInputPad->stride(0) * sizeof(int8_t));
#endif
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
            const auto bias_dz   = biasPtr + dz * 16;
            const auto scale_dz  = scalePtr + dz * UNIT;
            quanParameters.bias = bias_dz;
            quanParameters.scale = scale_dz;
            for (int dy = 0; dy < dst_height; ++dy) {
                const int srcStartY = dy * mStrides.second;
                const auto src_dy   = inputPadPtr + srcStartY * mPaddedSize.first * UNIT;
                auto dst_y          = dstOrigin + dy * dst_width * UNIT;
                mThreadFunction(dst_y, (const int8_t*)src_dy, (const int8_t*)weight_dz, &quanParameters, dst_width, mStrides.first * UNIT, mKernels.first, mKernels.second, mDilates.first * UNIT, mDilates.second * UNIT * mPaddedSize.first, mOrder.data());
            }
        }
    }
    MNN_CONCURRENCY_END();
}

CPUDepthwiseConvInt8::CPUDepthwiseConvInt8(Backend* backend, const Convolution2DCommon* common, std::shared_ptr<ResourceInt8> res): CPUConvolution(common, backend), mResource(res), mMutableResource(res, backend) {
    mValid        = mMutableResource.mValid;
}
CPUDepthwiseConvInt8::~CPUDepthwiseConvInt8() {
    // Do nothing
}

bool CPUDepthwiseConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new CPUDepthwiseConvInt8(bn, op->main_as_Convolution2D()->common(), mResource);
    *dst = exe;
    return true;
}

ErrorCode CPUDepthwiseConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    std::vector<float> inputQuantInfo = TensorUtils::getQuantInfo(input);
    std::vector<float> outputQuantInfo = TensorUtils::getQuantInfo(output);
    mMutableResource.updateInputOutputScale(inputQuantInfo, outputQuantInfo);
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mCommon);

    int padX = std::get<0>(pads);
    int padY = std::get<1>(pads);
    mPads = std::make_pair(padX, padY);
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    int UNIT = mPack;
    mThreadFunction = core->ConvDepthwiseLineInt8;

    const int src_width      = input->width();
    const int src_height     = input->height();
    const int dst_width      = output->width();
    const int dst_height     = output->height();
    const int strideY        = mCommon->strideY();
    const int strideX        = mCommon->strideX();
    const int dilateY        = mCommon->dilateY();
    const int dilateX        = mCommon->dilateX();
    const int kernel_height  = mCommon->kernelY();
    const int kernel_width   = mCommon->kernelX();

    int size_ = mMutableResource.mBiasInt32->length(0);
    if (core->ConvDepthwise3x3LineInt8_ARM82) {
        if (kernel_width == 3 && kernel_height == 3 && strideX == 1 && strideY == 1 && dilateX == 1 && dilateY == 1 && gcore->MNNMultiAndDestTransformCommon23 != nullptr && dst_width >= 2 && dst_height >= 2) {
            mUse3x3Kernel   = true;
            mThreadFunction = core->ConvDepthwise3x3LineInt8_ARM82;
            UNIT = 4;
            mOrder.resize(64);
            mOrder = { 0, 4, 8, 16, 1, 5, 9, 17, 2, 6, 10, 18, 3, 7, 11, 19, 
                       4, 8, 16, 20, 5, 9, 17, 21, 6, 10, 18, 22, 7, 11, 19, 23,
                       4, 8, 12, 20, 5, 9, 13, 21, 6, 10, 14, 22, 7, 11, 15, 23,
                       8, 12, 20, 24, 9, 13, 21, 25, 10, 14, 22, 26, 11, 15, 23, 27
                    };
            
            int32_t* biasPtr = mMutableResource.mBiasInt32->host<int32_t>();
            mBiasExtend.resize(size_ * 4);
            int32_t* dstPtr = mBiasExtend.data();
            for (int i = 0; i < size_ / 4; ++i) {
                ::memcpy(dstPtr, biasPtr, sizeof(int32_t) * 4);
                ::memcpy(dstPtr + 4, biasPtr, sizeof(int32_t) * 4);
                ::memcpy(dstPtr + 8, biasPtr, sizeof(int32_t) * 4);
                ::memcpy(dstPtr + 12, biasPtr, sizeof(int32_t) * 4);
                biasPtr += 4;
                dstPtr += 16;
            }
        }
    }
    if (!mUse3x3Kernel) {
        mBiasExtend.resize(size_);
        int32_t* biasPtr = mMutableResource.mBiasInt32->host<int32_t>();
        int32_t* dstPtr  = mBiasExtend.data();
        ::memcpy(dstPtr, biasPtr, sizeof(int32_t) * size_);
    }


    const int dst_depth_quad = UP_DIV(output->channel(), UNIT);
    const int threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    mThreadNumber          = std::min(threadNumber, dst_depth_quad * input->batch());
    int paddedWidth = std::get<0>(pads) + std::get<2>(pads) + input->width();
    int paddedHeight = std::get<1>(pads) + std::get<3>(pads) + input->height();
    mInputPad.reset(Tensor::createDevice<BASIC_TYPE>({mThreadNumber, paddedWidth * paddedHeight * UNIT}));
    mPaddedSize = std::make_pair(paddedWidth, paddedHeight);
    mStrides = std::make_pair(strideX, strideY);
    mDilates = std::make_pair(dilateX, dilateY);
    mKernels = std::make_pair(kernel_width, kernel_height);

    bool succ = backend()->onAcquireBuffer(mInputPad.get(), Backend::DYNAMIC);
    if (!succ) {
        return OUT_OF_MEMORY;
    }

    mInputTemp.reset(Tensor::createDevice<BASIC_TYPE>({input->batch(), src_height, src_width, UP_DIV(output->channel(), UNIT) * UNIT}));
    mOutputTemp.reset(Tensor::createDevice<BASIC_TYPE>({output->batch(), dst_height, dst_width, UP_DIV(output->channel(), UNIT) * UNIT}));
    bool res = backend()->onAcquireBuffer(mInputTemp.get(), Backend::DYNAMIC)
            && backend()->onAcquireBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mInputTemp.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mInputPad.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPUDepthwiseConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto coreInt8 = static_cast<CPUBackend*>(backend())->int8Functions();
    auto input = inputs[0];
    auto output = outputs[0];
    auto plane_in = input->width() * input->height() * input->batch();
    auto plane_out = output->width() * output->height() * output->batch();
    auto depth = UP_DIV(input->channel(), core->pack);

    if (mUse3x3Kernel) {
        CPUDepthwiseConvInt8::fastDepthwiseInt8(inputs, outputs);
        return NO_ERROR;
    }

    if (core->pack == 4) {
        MNNPackC4Origin(mInputTemp.get()->host<float>(), input->host<float>(), plane_in, depth, plane_in);
        CPUDepthwiseConvInt8::fastDepthwiseInt8({mInputTemp.get()}, {mOutputTemp.get()});
        MNNUnpackC4Origin(output->host<float>(), mOutputTemp.get()->host<float>(), plane_out, depth, plane_out);
    }
    else if (core->pack == 8) {
        MNNPackC2Origin(mInputTemp.get()->host<double>(), input->host<double>(), plane_in, depth, plane_in);
        CPUDepthwiseConvInt8::fastDepthwiseInt8({mInputTemp.get()}, {mOutputTemp.get()});
        MNNUnpackC2Origin(output->host<double>(), mOutputTemp.get()->host<double>(), plane_out, depth, plane_out);
    } else if (core->pack == 16) {
        CPUDepthwiseConvInt8::fastDepthwiseInt8(inputs, outputs);
    }
    
    return NO_ERROR;
}

class CPUDepthwiseConvInt8Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto convOp = op->main_as_Convolution2D();
        auto core = static_cast<CPUBackend*>(backend)->int8Functions();
        auto gcore = static_cast<CPUBackend*>(backend)->functions();
        auto common = convOp->common();
        bool use3x3kernel = false;
        int UNIT = 16;
        
        if (core->ConvDepthwise3x3LineInt8_ARM82) {
           if (common->kernelX() == 3 && common->kernelY() == 3 && common->strideX() == 1 && common->strideY() == 1 && common->dilateX() == 1
               && common->dilateY() == 1 && gcore->MNNMultiAndDestTransformCommon23 != nullptr && outputs[0]->width() >= 2 && outputs[0]->height() >= 2) {
               use3x3kernel = true;
               UNIT = 4;
           }
        }
        auto res = CPUConvolution::makeResourceInt8(backend, convOp, UNIT);
        const int kernelSize      = common->kernelX() * common->kernelY();
        const int outputCount     = common->outputCount();
        const int ocDivUnit       = UP_DIV(outputCount, UNIT);
        const int weightSizeAlign = ocDivUnit * UNIT * kernelSize;

        std::shared_ptr<Tensor> weight(Tensor::createDevice<BASIC_TYPE>({weightSizeAlign}));
        auto allocRes = backend->onAcquireBuffer(weight.get(), Backend::STATIC);
        if (!allocRes) {
            return nullptr;
        }
        // Reorder the weight
        auto originWeight = res->mWeightInt8->host<int8_t>();
        auto weightPtr = weight->host<BASIC_TYPE>();
        memset(weightPtr, 0, weightSizeAlign * sizeof(BASIC_TYPE));

        if (use3x3kernel) {
            int kernelOrder[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            for (int dz = 0; dz < outputCount; ++dz) {
                const int dzDivUnit = dz / gcore->pack;
                const int my        = dz % gcore->pack;
                auto dstDz          = weightPtr + dzDivUnit * kernelSize * gcore->pack;
                for (int i = 0; i < 4; ++i) { // kernelSize = 9
                    int k = kernelOrder[i];
                    dstDz[i + my * 4] = (BASIC_TYPE)(originWeight[dz * kernelSize + k]);
                }
                for (int i = 0; i < 4; ++i) {
                    int k = kernelOrder[i + 4];
                    dstDz[16 + i + my * 4] = (BASIC_TYPE)(originWeight[dz * kernelSize + k]);
                }
                dstDz[8 * gcore->pack + my] = (BASIC_TYPE)(originWeight[dz * kernelSize + 8]);
            }
             res->mWeightInt8.swap(weight);
             backend->onReleaseBuffer(weight.get(), Backend::STATIC);
             return new CPUDepthwiseConvInt8(backend, convOp->common(), res);
        }

        for (int dz = 0; dz < outputCount; ++dz) {
            const int dzDivUnit = dz / UNIT;
            const int my        = dz % UNIT;
            auto dstDz          = weightPtr + dzDivUnit * kernelSize * UNIT;
            for (int i = 0; i < kernelSize; ++i) {
                dstDz[i * UNIT + my] = (BASIC_TYPE)(originWeight[dz * kernelSize + i]);
            }
        }
        res->mWeightInt8.swap(weight);
        backend->onReleaseBuffer(weight.get(), Backend::STATIC);
        return new CPUDepthwiseConvInt8(backend, convOp->common(), res);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDepthwiseConvInt8Creator, OpType_DepthwiseConvInt8);

} // namespace MNN
