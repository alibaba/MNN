//
//  CPUDeconvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUDeconvolutionDepthwise.hpp"
#include <string.h>
#include "CPUBackend.hpp"
#include "MNN_generated.h"
#include "Macro.h"
#include "compute/ConvOpt.h"

namespace MNN {
CPUDeconvolutionDepthwise::CPUDeconvolutionDepthwise(const Tensor* input, const Op* convOp, Backend* b)
    : MNN::CPUDeconvolutionCommon(input, convOp, b) {
    auto conv               = convOp->main_as_Convolution2D();
    auto layer              = convOp->main_as_Convolution2D()->common();
    int kw                  = layer->kernelX();
    int kh                  = layer->kernelY();
    int outputCount         = layer->outputCount();
    int depthQuad           = UP_DIV(outputCount, 4);
    int planeStride         = kw * kh * 4;
    const float* tempWeight = conv->weight()->data();
    // Reorder weight from whc -> pwhc4
    int kernelSize = depthQuad * 4 * kw * kh;
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{kernelSize}));
    auto sucess = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!sucess) {
        mValid = false;
        return;
    }
    ::memset(mWeight->host<float>(), 0, mWeight->size());
    auto weight = mWeight->host<float>();
    int cur     = 0;
    for (int c = 0; c < outputCount; ++c) {
        int plane  = c / 4;
        int offset = c % 4;
        for (int y = 0; y < kh; ++y) {
            for (int x = 0; x < kw; ++x) {
                float* dst = weight + offset + (x + y * kw) * 4 + planeStride * plane;
                *dst       = tempWeight[cur++];
            }
        }
    }
    mOrigin.reset(new CPUDeconvolutionDepthwiseBasic(input, convOp, b));
}

CPUDeconvolutionDepthwise::~CPUDeconvolutionDepthwise() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
}

ErrorCode CPUDeconvolutionDepthwiseMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                                        const std::vector<Tensor*>& outputs) {
    auto kw = mCommon->kernelX();
    auto kh = mCommon->kernelY();
    mWeight.reset(Tensor::createDevice<float>({UP_DIV(inputs[0]->channel(), 4), kh, kw, 4}));
    mBias.reset(Tensor::createDevice<float>({UP_DIV(inputs[0]->channel(), 4), 4}));
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
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), inputs[2]->host<float>(), inputs[2]->size());
    ::memset(mWeight->host<float>(), 0, mWeight->size());
    auto weight      = mWeight->host<float>();
    auto outputCount = inputs[0]->channel();
    auto kh          = mWeight->length(1);
    auto kw          = mWeight->length(2);
    auto tempWeight  = inputs[1]->host<float>();
    auto planeStride = kw * kh * 4;
    int cur          = 0;
    for (int c = 0; c < outputCount; ++c) {
        int plane  = c / 4;
        int offset = c % 4;
        for (int y = 0; y < kh; ++y) {
            for (int x = 0; x < kw; ++x) {
                float* dst = weight + offset + (x + y * kw) * 4 + planeStride * plane;
                *dst       = tempWeight[cur++];
            }
        }
    }
    return CPUDeconvolutionDepthwiseBasic::onExecute(mInputs, outputs);
}

ErrorCode CPUDeconvolutionDepthwiseBasic::onResize(const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    CPUDeconvolutionBasic::onResize(inputs, outputs);
    auto layer         = mCommon;
    auto inputTensor   = outputs[0];
    auto outputTensor  = inputs[0];
    int src_width      = inputTensor->width();
    int src_height     = inputTensor->height();
    int dst_width      = outputTensor->width();
    int dst_height     = outputTensor->height();
    int dst_depth_quad = UP_DIV(layer->outputCount(), 4);
    int dst_z_step     = dst_width * dst_height * 4;
    int src_z_step     = src_width * src_height * 4;
    int dst_y_step     = dst_width * 4;
    int src_y_step     = src_width * 4;
    int strideY        = layer->strideY();
    int strideX        = layer->strideX();
    int dilateX        = layer->dilateX();
    int dilateY        = layer->dilateY();
    int dilateY_step   = dilateY * src_width * 4;
    int dilateX_step   = dilateX * 4;
    int kernel_height  = layer->kernelY();
    int kernel_width   = layer->kernelX();
    int padX           = mPadX;
    int padY           = mPadY;
    int weight_z_step  = kernel_height * kernel_width * 4;
    // Compute Mid Rect
    int l = 0, t = 0, r = dst_width, b = dst_height;
    for (; l * strideX - padX < 0; l++) {
        // do nothing
    }
    for (; t * strideY - padY < 0; t++) {
        // do nothing
    }
    for (; (r - 1) * strideX - padX + kernel_width * dilateX > src_width && r > l; r--) {
        // do nothing
    }
    for (; (b - 1) * strideY - padY + kernel_height * dilateY > src_height && b > t; b--) {
        // do nothing
    }

    auto postFunction = getPostFunction();
#define RUN_BASIC(L, T, R, B)                                                                              \
    for (int dy = T; dy < B; ++dy) {                                                                       \
        const float* dst_y = dst_z + dy * dst_y_step;                                                      \
        int srcStartY      = dy * strideY - padY;                                                          \
        float* src_dy      = src_z + srcStartY * src_y_step;                                               \
        int sfy            = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));                                     \
        int efy            = ALIMIN(kernel_height, UP_DIV(src_height - srcStartY, dilateY));               \
        for (int dx = L; dx < R; ++dx) {                                                                   \
            const float* dst_x = dst_y + 4 * dx;                                                           \
            int srcStartX      = dx * strideX - padX;                                                      \
            float* src_dx      = src_dy + srcStartX * 4;                                                   \
            int sfx            = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));                                 \
            int efx            = ALIMIN(kernel_width, UP_DIV(src_width - srcStartX, dilateX));             \
            MNNDeconvRunForUnitDepthWise(dst_x, src_dx + (sfx * dilateX + sfy * dilateY * src_width) * 4,  \
                                         weight_dz + 4 * (kernel_width * sfy + sfx), efx - sfx, efy - sfy, \
                                         4 * kernel_width, dilateX_step, dilateY_step);                    \
        }                                                                                                  \
    }
    auto weight = inputs[1];
    auto bias   = inputs[2];

    mFunction = [=](const float* dstOrigin, float* srcOrigin) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const float* dst_z     = dstOrigin + dst_z_step * dz;
            float* src_z           = srcOrigin + src_z_step * dz;
            const float* weight_dz = weight->host<float>() + dz * weight_z_step;

            RUN_BASIC(0, 0, dst_width, t);
            RUN_BASIC(0, b, dst_width, dst_height);

            RUN_BASIC(0, t, l, b);
            RUN_BASIC(r, t, dst_width, b);

            if (r > l) {
                for (int dy = t; dy < b; ++dy) {
                    const float* dst_y = dst_z + dy * dst_y_step;
                    int srcStartY      = dy * strideY - padY;
                    float* src_dy      = src_z + srcStartY * src_y_step;
                    MNNDeconvRunForLineDepthwise(dst_y + l * 4, src_dy + (l * strideX - padX) * 4, weight_dz, r - l,
                                                 strideX * 4, kernel_width, kernel_height, dilateX_step, dilateY_step);
                }
            }
        }
        postFunction(srcOrigin, bias->host<float>(), src_width * src_height, dst_depth_quad);
    };
#undef RUN_BASIC

    return NO_ERROR;
}

ErrorCode CPUDeconvolutionDepthwiseBasic::onExecute(const std::vector<Tensor*>& inputs,
                                                    const std::vector<Tensor*>& outputs) {
    // Revert input and output, do deconvolution
    auto inputTensor  = outputs[0];
    auto outputTensor = inputs[0];
    for (int batchIndex = 0; batchIndex < inputTensor->batch(); ++batchIndex) {
        float* srcOrigin = inputTensor->host<float>() + batchIndex * inputTensor->stride(0);
        ::memset(srcOrigin, 0, inputTensor->stride(0) * sizeof(float));
        const float* dstOrigin = outputTensor->host<float>() + batchIndex * outputTensor->stride(0);
        mFunction(dstOrigin, srcOrigin);
    }

    return NO_ERROR;
}

class CPUDeconvolutionDepthwiseCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        if (3 == inputs.size()) {
            return new CPUDeconvolutionDepthwiseMultiInput(inputs[0], op, backend);
        }
        return new CPUDeconvolutionDepthwise(inputs[0], op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolutionDepthwiseCreator, OpType_DeconvolutionDepthwise);

} // namespace MNN
