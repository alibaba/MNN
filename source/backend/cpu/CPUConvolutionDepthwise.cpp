//
//  CPUConvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUConvolutionDepthwise.hpp"
#include <string.h>
#include "core/Concurrency.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"

namespace MNN {
CPUConvolutionDepthwise::FloatExecution::FloatExecution(const Convolution2DCommon* common, Backend* b,
                                                        const float* originWeight, size_t originWeightSize,
                                                        const float* bias, size_t biasSize)
    : MNN::CPUConvolution(common, b) {
    auto layer = common;
    mOrigin.reset(new BasicFloatExecution(common, b));
    mResource.reset(new Resource);
    mResource->backend = backend();
    auto core = static_cast<CPUBackend*>(b)->functions();
    int bytes = core->bytes;
    int unit = core->pack;
    int kw          = layer->kernelX();
    int kh          = layer->kernelY();
    int outputCount = (int)biasSize;
    int depthQuad   = UP_DIV(outputCount, unit);
    int kernelSize  = depthQuad * unit * kw * kh;
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>(std::vector<int>{kernelSize * bytes}));
    bool success = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Error for alloc memory for CPUConvolutionDepthwise\n");
        mValid = false;
        return;
    }
    success = mResource->copyBiasAlign(bias, static_cast<int>(biasSize));
    if (!success) {
        mValid = false;
        return;
    }
    const float* tempWeight = originWeight;
    // Reorder weight from whc -> pwhc4
    auto weight = mResource->mWeight->host<float>();
    int offset[] = {
        (int)(kh * kw),
        (int)(kh * kw)
    };
    if (bytes < 4) {
        AutoStorage<uint8_t> tempW(kh * kw * outputCount * bytes);
        if (tempW.get() == nullptr) {
            mValid = false;
            return;
        }
        core->MNNFp32ToLowp(tempWeight, (int16_t*)tempW.get(), kh * kw * outputCount);
        core->MNNPackCUnit(weight, (const float*)tempW.get(), kh * kw, outputCount, offset);
    } else {
        core->MNNPackCUnit(weight, tempWeight, kh * kw, outputCount, offset);
    }
}
CPUConvolutionDepthwise::FloatExecution::~FloatExecution() {
    // Do nothing
}
bool CPUConvolutionDepthwise::FloatExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new CPUConvolutionDepthwise::FloatExecution(mResource, op->main_as_Convolution2D()->common(), bn);
    *dst = dstExe;
    return true;
}

ErrorCode CPUConvolutionDepthwise::MultiInputFloatExecution::onResize(const std::vector<Tensor*>& inputs,
                                                                      const std::vector<Tensor*>& outputs) {
    auto layer = mCommon;
    auto kw    = layer->kernelX();
    auto kh    = layer->kernelY();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int bytes = core->bytes;
    int unit = core->pack;
    auto ic4 = UP_DIV(inputs[0]->channel(), unit);
    mWeight.reset(Tensor::createDevice<uint8_t>({ic4, kh, kw, unit * bytes}));
    mBias.reset(Tensor::createDevice<uint8_t>({ic4 * unit * bytes}));
    mTempInputs = {inputs[0], mWeight.get(), mBias.get()};
    bool success = backend()->onAcquireBuffer(mWeight.get(), Backend::DYNAMIC);
    success = success && backend()->onAcquireBuffer(mBias.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto code = CPUConvolutionDepthwise::BasicFloatExecution::onResize(mTempInputs, outputs);
    backend()->onReleaseBuffer(mWeight.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mBias.get(), Backend::DYNAMIC);
    return code;
}

ErrorCode CPUConvolutionDepthwise::MultiInputFloatExecution::onExecute(const std::vector<Tensor*>& inputs,
                                                                       const std::vector<Tensor*>& outputs) {
    auto kh = mWeight->length(1);
    auto kw = mWeight->length(2);
    // Reorder weight from whc -> pwhc4
    auto outputCount = inputs[0]->channel();
    auto weight      = mWeight->host<float>();
    auto tempWeight  = inputs[1]->host<float>();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int bytes = core->bytes;
    int unit = core->pack;
    int offset[] = {
        (int)(kh * kw),
        (int)(kh * kw)
    };
    core->MNNPackCUnit(weight, tempWeight, kh * kw, outputCount, offset);
    ::memset(mBias->host<float>(), 0, mBias->size());
    if (inputs.size() > 2) {
        ::memcpy(mBias->host<float>(), inputs[2]->host<float>(), outputCount * bytes);
    }
    return CPUConvolutionDepthwise::BasicFloatExecution::onExecute(mTempInputs, outputs);
}

ErrorCode CPUConvolutionDepthwise::BasicFloatExecution::onResize(const std::vector<Tensor*>& inputs,
                                                                 const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto layer         = mCommon;
    auto core          = static_cast<CPUBackend*>(backend())->functions();
    int bytes          = core->bytes;
    int unit           = core->pack;
    auto kernelFunc = core->MNNConvRunForLineDepthwise;
    auto postFunc = core->MNNAxByClampBroadcastUnit;
    auto inputTensor   = inputs[0];
    auto outputTensor  = outputs[0];
    int src_width      = inputTensor->width();
    int src_height     = inputTensor->height();
    int dst_width      = outputTensor->width();
    int dst_height     = outputTensor->height();
    int dst_depth_quad = UP_DIV(layer->outputCount(), unit);
    int strideY        = layer->strideY();
    int strideX        = layer->strideX();
    int dilateX        = layer->dilateX();
    int dilateY        = layer->dilateY();
    int kernel_height  = layer->kernelY();
    int kernel_width   = layer->kernelX();
    int padX           = mPadX;
    int padY           = mPadY;
    if (src_width == 1 && dst_width == 1 && dst_height > 1 && kernel_width == 1) {
        // Swap x, y
        dst_width = dst_height;
        dst_height = 1;
        padX = mPadY;
        padY = mPadX;
        strideX = strideY;
        strideY = 1;// Don't need stride
        src_width = src_height;
        src_height = 1;
        dilateX = dilateY;
        dilateY = 1;
        kernel_width = kernel_height;
        kernel_height = 1;
    }
    int dst_z_step     = dst_width * dst_height * unit;
    int src_z_step     = src_width * src_height * unit;
    int dst_y_step     = dst_width * unit;
    int src_y_step     = src_width * unit;
    int weight_z_step  = kernel_height * kernel_width * unit;
    int dilateY_step   = dilateY * src_width * unit;
    int dilateX_step   = dilateX * unit;

    auto batch = inputs[0]->batch();
    int total = batch * dst_depth_quad;
    int numberThread = ((CPUBackend*)backend())->threadNumber();
    std::vector<int> divides(numberThread+1);
    divides[0] = 0;
    static_cast<CPUBackend *>(backend())->computeDivideSizes(total, divides.data()+1);
    mNumber = numberThread;
    for (int i=1; i<numberThread; ++i) {
        if (divides[i+1] <= divides[i]) {
            // Only 0-(i-1) thread has work
            mNumber = i;
            break;
        }
    }
    MNN_ASSERT(mNumber > 0);
    auto postData = getPostParameters();
    if (static_cast<CPUBackend*>(backend())->functions()->bytes < 4) {
        static_cast<CPUBackend*>(backend())->functions()->MNNFp32ToLowp(postData.data() + 2, (int16_t*)(postData.data() + 2), 2);
    }
    mFastKernelApply = (dilateX == 1 && dilateY == 1 && strideX == 1 && strideY == 1 && core->MNNDepthwiseConvFastKernel);
    if (mFastKernelApply ) { // Only support ARM kernel
        kernelFunc = core->MNNDepthwiseConvFastKernel;
    }
    auto pads = ConvolutionCommon::convolutionPadFull(inputs[0], outputs[0], mCommon);
    int paddedWidth = std::get<0>(pads) + std::get<2>(pads) + src_width;
    int paddedHeight = std::get<1>(pads) + std::get<3>(pads) + src_height;
    mInputPad.reset(Tensor::createDevice<float>({mNumber, paddedWidth * paddedHeight * unit}));
    bool succ = backend()->onAcquireBuffer(mInputPad.get(), Backend::DYNAMIC);
    if (!succ) {
        return OUT_OF_MEMORY;
    }
    if (paddedWidth != src_width) {
        dilateY_step   = dilateY * paddedWidth * unit;
        src_y_step     = paddedWidth * unit;
    }
    mExecutor   = [=](const uint8_t* inputPtr, uint8_t* outputPtr, int tId) {
        MNN_ASSERT(divides[tId] < divides[tId+1]);
        const auto inputPadPtr = mInputPad->host<uint8_t>() + mInputPad->stride(0) * tId * bytes;
        ::memset(inputPadPtr, 0, mInputPad->stride(0) * bytes);
        auto biasP   = inputs[2]->host<uint8_t>();
        auto weightP = inputs[1]->host<uint8_t>();
        for (int index = divides[tId]; index < divides[tId+1]; ++index) {
            
            int dz = index / batch;
            auto dstOrigin           = outputPtr + dst_z_step * index * bytes;
            const auto srcOrigin     = inputPtr + src_z_step * index * bytes;
            auto bias_z          = biasP + unit * dz * bytes;
            const auto weight_dz = weightP + dz * weight_z_step * bytes;
            
            auto srcPtr = srcOrigin;
            // Pad inputs
            for (int y = 0; y < src_height; ++y) {
                auto src = srcOrigin + y * src_width * unit * bytes;
                auto dst = inputPadPtr + ((y + padY) * paddedWidth + padX) * unit * bytes;
                ::memcpy(dst, src, src_width * unit * bytes);
            }

            // Compute
            kernelFunc((float*)dstOrigin, (const float*)(inputPadPtr), (const float*)weight_dz, dst_width, strideX * unit, kernel_width, kernel_height, dilateX_step, dilateY_step, dst_height, src_y_step * strideY, dst_y_step, (const float*)bias_z, postData.data() + 2);
        }
    };
    backend()->onReleaseBuffer(mInputPad.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUConvolutionDepthwise::BasicFloatExecution::onExecute(const std::vector<Tensor*>& inputs,
                                                                  const std::vector<Tensor*>& outputs) {
    auto inputTensor  = inputs[0];
    auto outputTensor = outputs[0];
    const auto srcOrigin = inputTensor->host<uint8_t>();
    auto dstOrigin       = outputTensor->host<uint8_t>();
    MNN_CONCURRENCY_BEGIN(tId, mNumber) {
        mExecutor(srcOrigin, dstOrigin, (int)tId);
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
class CPUConvolutionDepthwiseCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto conv2d = op->main_as_Convolution2D();
        auto conv   = op->main_as_Convolution2D()->common();
        if (1 < inputs.size()) {
            return new CPUConvolutionDepthwise::MultiInputFloatExecution(conv, backend);
        }
        const float* originWeight = nullptr;
        const float* originBias = nullptr;
        int originWeightSize   = 0;
        int originBiasSize   = 0;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
        if (nullptr != conv2d->quanParameter()) {
            quanCommon = ConvolutionCommon::load(op, backend, true);
            // Back to float
            originWeight     = quanCommon->weightFloat.get();
            originWeightSize = quanCommon->weightFloat.size();
        }
        if (nullptr == originWeight) {
            originWeight     = conv2d->weight()->data();
            originWeightSize = conv2d->weight()->size();
        }
        if (nullptr == originBias) {
            originBias     = op->main_as_Convolution2D()->bias()->data();
            originBiasSize = op->main_as_Convolution2D()->bias()->size();
        }
        if (inputs.empty()) {
            return new CPUConvolutionDepthwise::FloatExecution(conv2d->common(), backend, originWeight, originWeightSize, originBias, originBiasSize);
        }
        return new CPUConvolutionDepthwise::FloatExecution(conv2d->common(), backend, originWeight, originWeightSize, originBias, originBiasSize);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConvolutionDepthwiseCreator, OpType_ConvolutionDepthwise);
} // namespace MNN
