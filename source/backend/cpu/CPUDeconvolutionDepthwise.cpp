//
//  CPUDeconvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUDeconvolutionDepthwise.hpp"
#include <string.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "compute/CommonOptFunction.h"
#include "core/Concurrency.h"


namespace MNN {
CPUDeconvolutionDepthwise::CPUDeconvolutionDepthwise(const Tensor* input, const Op* convOp, Backend* b)
    : MNN::CPUDeconvolutionCommon(input, convOp, b, false) {
    auto conv               = convOp->main_as_Convolution2D();
    auto layer              = convOp->main_as_Convolution2D()->common();
    int kw                  = layer->kernelX();
    int kh                  = layer->kernelY();
    int outputCount         = layer->outputCount();
    auto core               = static_cast<CPUBackend*>(backend())->functions();
    int depthQuad           = UP_DIV(outputCount, core->pack);
    const float* tempWeight = nullptr;
    int tempWeightSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, b, conv, &tempWeight, &tempWeightSize);

    // Reorder weight from whc -> pwhc4
    int kernelSize = depthQuad * core->pack * kw * kh;
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{kernelSize}));
    auto sucess = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!sucess) {
        mValid = false;
        return;
    }
    AutoStorage<uint8_t> weightTempStorage;
    if (core->bytes < 4) {
        weightTempStorage.reset(kernelSize * core->bytes);
        if (weightTempStorage.get() == nullptr) {
            mValid = false;
            return;
        }
        core->MNNFp32ToLowp(tempWeight, (int16_t*)weightTempStorage.get(), kernelSize);
        tempWeight = (const float*)weightTempStorage.get();
    }
    auto weight = mWeight->host<float>();
    int offset[] = {
        kw * kh,
        kw * kh
    };
    core->MNNPackCUnit(weight, tempWeight, kw * kh, outputCount, offset);
    mOrigin.reset(new CPUDeconvolutionDepthwiseBasic(input, convOp, b));
}

CPUDeconvolutionDepthwise::~CPUDeconvolutionDepthwise() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
}

ErrorCode CPUDeconvolutionDepthwiseMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                                        const std::vector<Tensor*>& outputs) {
    auto kw = mCommon->kernelX();
    auto kh = mCommon->kernelY();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mWeight.reset(Tensor::createDevice<float>({UP_DIV(inputs[0]->channel(), core->pack), kh, kw, core->pack}));
    mBias.reset(Tensor::createDevice<float>({UP_DIV(inputs[0]->channel(), core->pack), core->pack}));
    backend()->onAcquireBuffer(mWeight.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mBias.get(), Backend::DYNAMIC);
    mInputs   = {inputs[0], mWeight.get(), mBias.get()};
    auto code = CPUDeconvolutionDepthwiseBasic::onResize(mInputs, outputs);
    backend()->onReleaseBuffer(mWeight.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mBias.get(), Backend::DYNAMIC);
    return code;
}

ErrorCode CPUDeconvolutionDepthwiseMultiInput::onExecute(const std::vector<Tensor*>& inputs,
                                                         const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    ::memset(mBias->host<float>(), 0, mBias->elementSize() * core->bytes);
    if (inputs.size() > 2) {
        ::memcpy(mBias->host<float>(), inputs[2]->host<float>(), inputs[2]->elementSize() * core->bytes);
    }
    ::memset(mWeight->host<float>(), 0, mWeight->elementSize() * core->bytes);
    auto weight      = mWeight->host<float>();
    auto outputCount = inputs[0]->channel();
    auto kh          = mWeight->length(1);
    auto kw          = mWeight->length(2);
    auto tempWeight  = inputs[1]->host<float>();
    int offset[] = {
        kw * kh,
        kw * kh
    };
    core->MNNPackCUnit(weight, tempWeight, kw * kh, outputCount, offset);
    return CPUDeconvolutionDepthwiseBasic::onExecute(mInputs, outputs);
}

ErrorCode CPUDeconvolutionDepthwiseBasic::onResize(const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    CPUDeconvolutionBasic::onResize(inputs, outputs);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto layer         = mCommon;
    auto inputTensor   = outputs[0];
    auto outputTensor  = inputs[0];
    int src_width      = inputTensor->width();
    int src_height     = inputTensor->height();
    int dst_width      = outputTensor->width();
    int dst_height     = outputTensor->height();
    int dst_depth_quad = UP_DIV(layer->outputCount(), core->pack);
    int dst_z_step     = dst_width * dst_height * core->pack;
    int src_z_step     = src_width * src_height * core->pack;
    int dst_y_step     = dst_width * core->pack;
    int src_y_step     = src_width * core->pack;
    int strideY        = layer->strideY();
    int strideX        = layer->strideX();
    int dilateX        = layer->dilateX();
    int dilateY        = layer->dilateY();
    int dilateY_step   = dilateY * src_width * core->pack;
    int dilateX_step   = dilateX * core->pack;
    int kernel_height  = layer->kernelY();
    int kernel_width   = layer->kernelX();
    int padX           = mPadX;
    int padY           = mPadY;
    int weight_z_step  = kernel_height * kernel_width * core->pack;
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

#define RUN_BASIC(L, T, R, B)                                                                              \
    for (int dy = T; dy < B; ++dy) {                                                                       \
        auto dst_y = dst_z + dy * dst_y_step * core->bytes;                                                      \
        int srcStartY      = dy * strideY - padY;                                                          \
        auto src_dy      = src_z + srcStartY * src_y_step * core->bytes;                                               \
        int sfy            = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));                                     \
        int efy            = ALIMIN(kernel_height, UP_DIV(src_height - srcStartY, dilateY));               \
        for (int dx = L; dx < R; ++dx) {                                                                   \
            auto dst_x = dst_y + core->pack * core->bytes * dx;                                                           \
            int srcStartX      = dx * strideX - padX;                                                      \
            auto src_dx      = src_dy + srcStartX * core->pack * core->bytes;                                                   \
            int sfx            = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));                                 \
            int efx            = ALIMIN(kernel_width, UP_DIV(src_width - srcStartX, dilateX));             \
            core->MNNDeconvRunForUnitDepthWise((const float*)dst_x, (float*)(src_dx + (sfx * dilateX + sfy * dilateY * src_width) * core->bytes * core->pack),  \
                                         (const float*)(weight_dz + core->pack * core->bytes * (kernel_width * sfy + sfx)), efx - sfx, efy - sfy, \
                                         core->pack * kernel_width, dilateX_step, dilateY_step);                    \
        }                                                                                                  \
    }
    auto weight = inputs[1];
    auto bias   = inputs[2];
    int batch = inputs[0]->batch();
    int totalSize = batch * dst_depth_quad;
    int numberThread = ((CPUBackend*)backend())->threadNumber();

    mFunction = [=](const uint8_t* dstOrigin, uint8_t* srcOrigin, int tId) {
        for (int dz = tId; dz < totalSize; dz+=numberThread) {
            auto zPos = dz / batch;
            auto dst_z     = dstOrigin + dst_z_step * dz * core->bytes;
            auto src_z           = srcOrigin + src_z_step * dz * core->bytes;
            auto weight_dz = weight->host<uint8_t>() + zPos * weight_z_step * core->bytes;
            ::memset(src_z, 0, src_width * src_height * core->bytes * core->pack);

            RUN_BASIC(0, 0, dst_width, t);
            RUN_BASIC(0, b, dst_width, dst_height);

            RUN_BASIC(0, t, l, b);
            RUN_BASIC(r, t, dst_width, b);

            if (r > l) {
                for (int dy = t; dy < b; ++dy) {
                    auto dst_y = dst_z + dy * dst_y_step * core->bytes;
                    int srcStartY      = dy * strideY - padY;
                    auto src_dy      = src_z + srcStartY * src_y_step * core->bytes;
                    core->MNNDeconvRunForLineDepthwise((const float*)(dst_y + l * core->pack * core->bytes), (float*)(src_dy + (l * strideX - padX) * core->bytes * core->pack), (const float*)weight_dz, r - l,
                                                 strideX * core->pack, kernel_width, kernel_height, dilateX_step, dilateY_step);
                }
            }
            core->MNNAxByClampBroadcastUnit((float*)src_z, (float*)src_z, (const float*)(bias->host<uint8_t>() + zPos * core->pack * core->bytes), src_width * src_height, 0, 0, 1, mPostParameters.data());
        }
    };
#undef RUN_BASIC

    return NO_ERROR;
}

ErrorCode CPUDeconvolutionDepthwiseBasic::onExecute(const std::vector<Tensor*>& inputs,
                                                    const std::vector<Tensor*>& outputs) {
    // Revert input and output, do deconvolution
    auto inputTensor  = outputs[0];
    auto outputTensor = inputs[0];
    int numberThread = ((CPUBackend*)backend())->threadNumber();
    auto srcOrigin = inputTensor->host<uint8_t>();
    auto dstOrigin = outputTensor->host<uint8_t>();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        mFunction(dstOrigin, srcOrigin, tId);
    };
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUDeconvolutionDepthwiseCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        if (1 < inputs.size()) {
            return new CPUDeconvolutionDepthwiseMultiInput(inputs[0], op, backend);
        }
        return new CPUDeconvolutionDepthwise(inputs[0], op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolutionDepthwiseCreator, OpType_DeconvolutionDepthwise);

} // namespace MNN
