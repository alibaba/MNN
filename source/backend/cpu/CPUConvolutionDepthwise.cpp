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
#include "backend/cpu/compute/ConvolutionDepthwise3x3.hpp"

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
    auto unitFunc = core->MNNConvRunForUnitDepthWise;
    auto lineFunc = core->MNNConvRunForLineDepthwise;
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
    // Compute Mid Rect
    int l = 0, t = 0, r = dst_width, b = dst_height;
    for (; l * strideX - padX < 0 && l < dst_width; l++) {
        // do nothing
    }
    for (; t * strideY - padY < 0 && t < dst_height; t++) {
        // do nothing
    }
    for (; (r - 1) * strideX - padX + (kernel_width - 1) * dilateX >= src_width && r > l; r--) {
        // do nothing
    }
    for (; (b - 1) * strideY - padY + (kernel_height - 1) * dilateY >= src_height && b > t; b--) {
        // do nothing
    }

    auto postData = getPostParameters();
    auto batch = inputs[0]->batch();
    int total = batch * dst_depth_quad;
    int numberThread = ((CPUBackend*)backend())->threadNumber();
    auto rt = static_cast<const CPURuntime*>(backend()->getRuntime());
    auto runBasic     = [=](uint8_t* dst_z, const uint8_t* src_z, const uint8_t* weight_dz, int L, int T, int R, int B) {
        for (int dy = T; dy < B; ++dy) {
            auto dst_y        = dst_z + dy * dst_y_step * bytes;
            int srcStartY       = dy * strideY - padY;
            const auto src_dy = src_z + srcStartY * src_y_step * bytes;
            int sfy             = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
            int efy             = ALIMIN(kernel_height, UP_DIV(src_height - srcStartY, dilateY));
            for (int dx = L; dx < R; ++dx) {
                auto dst_x        = dst_y + unit * dx * bytes;
                int srcStartX       = dx * strideX - padX;
                const auto src_dx = src_dy + srcStartX * unit * bytes;
                int sfx             = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                int efx             = ALIMIN(kernel_width, UP_DIV(src_width - srcStartX, dilateX));
                unitFunc((float*)dst_x, (const float*)(src_dx + (sfx * dilateX + sfy * dilateY * src_width) * unit * bytes),
                         (const float*)(weight_dz + unit * (kernel_width * sfy + sfx) * bytes), efx - sfx, efy - sfy,
                         unit * kernel_width, dilateX_step, dilateY_step);
            }
        }
    };
    std::vector<int> divides(numberThread+1);
    divides[0] = 0;
    rt->computeDivideSizes(total, divides.data()+1);
    mExecutor   = [=](const uint8_t* srcOrigin, uint8_t* dstOrigin, int tId) {
        auto biasP   = inputs[2]->host<uint8_t>();
        auto weightP = inputs[1]->host<uint8_t>();
        for (int index = divides[tId]; index < divides[tId+1]; ++index) {
            int dz = index / batch;
            auto dst_z           = dstOrigin + dst_z_step * index * bytes;
            const auto src_z     = srcOrigin + src_z_step * index * bytes;
            auto bias_z          = biasP + unit * dz * bytes;
            const auto weight_dz = weightP + dz * weight_z_step * bytes;
            runBasic(dst_z, src_z, weight_dz, 0, 0, dst_width, t);
            runBasic(dst_z, src_z, weight_dz, 0, b, dst_width, dst_height);
            runBasic(dst_z, src_z, weight_dz, 0, t, l, b);
            runBasic(dst_z, src_z, weight_dz, r, t, dst_width, b);
            if (r > l && b > t) {
                lineFunc((float*)(dst_z + (t * dst_y_step + l * unit) * bytes),
                                           (const float*)(src_z + ((t * strideY - padY) * src_y_step + (l * strideX - padX) * unit) * bytes),
                                           (const float*)weight_dz, r - l, strideX * unit, kernel_width, kernel_height, dilateX_step,
                                           dilateY_step, b - t, src_y_step * strideY, dst_y_step);
            }
            postFunc((float*)dst_z, (float*)dst_z, (const float*)bias_z, dst_width * dst_height, 0, 0, 1, postData.data());
        }
    };
    mNumber = numberThread;

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
            quanCommon = ConvolutionCommon::load(conv2d, backend, true);
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
        auto core = static_cast<CPUBackend*>(backend)->functions();
        if (conv->dilateX() == 1 && conv->dilateY() == 1 && conv->strideX() == 1 && conv->strideY() == 1 &&
            conv->kernelX() == 3 && conv->kernelY() == 3 && outputs[0]->width() >= 2 && outputs[0]->height() >= 2 && core->MNNMultiAndDestTransformCommon23 != nullptr) {
            return new ConvolutionDepthwise3x3(conv, backend, originWeight, originWeightSize, originBias, originBiasSize);
        }
        return new CPUConvolutionDepthwise::FloatExecution(conv2d->common(), backend, originWeight, originWeightSize, originBias, originBiasSize);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConvolutionDepthwiseCreator, OpType_ConvolutionDepthwise);
} // namespace MNN
