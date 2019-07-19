//
//  CPUConvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUConvolutionDepthwise.hpp"
#include <string.h>
#include "Concurrency.h"
#include "Int8FunctionsOpt.h"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/ConvOpt.h"
#include "compute/ConvolutionDepthwise3x3.hpp"
static const int gIntUnit = 4;
extern "C" {
void MNNConvRunForLineDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t width,
                                    size_t src_w_setup, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step,
                                    const float* alpha_z);
}

#ifndef MNN_USE_NEON
void MNNConvRunForLineDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t width,
                                    size_t src_w_setup, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step,
                                    const float* alpha_z) {
    int dx, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        float* dst_x  = dst + dx * 4;
        dst_x[0]      = 0.0f;
        dst_x[1]      = 0.0f;
        dst_x[2]      = 0.0f;
        dst_x[3]      = 0.0f;
        auto src_z    = src + src_w_setup * dx;
        auto weight_z = weight;
        for (fy = 0; fy < fh; ++fy) {
            auto src_y    = src_z + fy * dilateY_step;
            auto weight_y = weight_z + fy * fw * 4;
            for (fx = 0; fx < fw; ++fx) {
                auto weight_x = weight_y + 4 * fx;
                auto src_x    = src_y + fx * dilateX_step;
                for (int j = 0; j < 4; ++j) {
                    dst_x[j] += (float)src_x[j] * (float)weight_x[j];
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            dst_x[i] *= alpha_z[i];
        }
    }
}
#endif

namespace MNN {
CPUConvolutionDepthwise::CPUConvolutionDepthwise(const Op* op, Backend* backend) : Execution(backend) {
    auto conv2d               = op->main_as_Convolution2D();
    const float* originWeight = nullptr;
    size_t originWeightSize   = 0;
    std::shared_ptr<ConvolutionIntFactory::Int8Common> quanCommon;
    if (nullptr != conv2d->quanParameter()) {
        quanCommon = ConvolutionIntFactory::load(conv2d->quanParameter(), false);
        if (quanCommon->weightFloat.get() == nullptr) {
            mSubExecution.reset(new Int8Execution(conv2d->common(), backend, quanCommon.get(), conv2d->bias()->data(),
                                                  conv2d->bias()->size()));
            return;
        }
        // Back to float
        originWeight     = quanCommon->weightFloat.get();
        originWeightSize = quanCommon->weightFloat.size();
    }
    if (nullptr == originWeight) {
        originWeight     = conv2d->weight()->data();
        originWeightSize = conv2d->weight()->size();
    }

    mSubExecution.reset(new FloatExecution(conv2d->common(), backend, originWeight, originWeightSize,
                                           conv2d->bias()->data(), conv2d->bias()->size()));
}
ErrorCode CPUConvolutionDepthwise::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mSubExecution->onResize(inputs, outputs);
}

ErrorCode CPUConvolutionDepthwise::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mSubExecution->onExecute(inputs, outputs);
}

CPUConvolutionDepthwise::FloatExecution::FloatExecution(const Convolution2DCommon* common, Backend* b,
                                                        const float* originWeight, size_t originWeightSize,
                                                        const float* bias, size_t biasSize)
    : MNN::CPUConvolution(common, b) {
    auto layer = common;
    mOrigin.reset(new BasicFloatExecution(common, b));

    int kw          = layer->kernelX();
    int kh          = layer->kernelY();
    int outputCount = (int)biasSize;
    mBias.reset(Tensor::createDevice<float>(std::vector<int>{ALIGN_UP4(outputCount)}));
    int depthQuad   = UP_DIV(outputCount, 4);
    int planeStride = kw * kh * 4;
    int kernelSize  = depthQuad * 4 * kw * kh;
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{kernelSize}));
    bool success =
        b->onAcquireBuffer(mBias.get(), Backend::STATIC) && b->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Error for alloc memory for CPUConvolutionDepthwise\n");
        mValid = false;
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));

    const float* tempWeight = originWeight;
    // Reorder weight from whc -> pwhc4
    ::memset(mWeight->host<float>(), 0, kernelSize * sizeof(float));
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
}
CPUConvolutionDepthwise::FloatExecution::~FloatExecution() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}
ErrorCode CPUConvolutionDepthwise::MultiInputFloatExecution::onResize(const std::vector<Tensor*>& inputs,
                                                                      const std::vector<Tensor*>& outputs) {
    auto layer = mCommon;
    auto kw    = layer->kernelX();
    auto kh    = layer->kernelY();

    mWeight.reset(Tensor::createDevice<float>({UP_DIV(inputs[0]->channel(), 4), kh, kw, 4}));
    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4(inputs[0]->channel())}));
    mTempInputs = {inputs[0], mWeight.get(), mBias.get()};
    backend()->onAcquireBuffer(mWeight.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mBias.get(), Backend::DYNAMIC);
    auto code = CPUConvolutionDepthwise::BasicFloatExecution::onResize(mTempInputs, outputs);
    backend()->onReleaseBuffer(mWeight.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mBias.get(), Backend::DYNAMIC);
    return code;
}

ErrorCode CPUConvolutionDepthwise::MultiInputFloatExecution::onExecute(const std::vector<Tensor*>& inputs,
                                                                       const std::vector<Tensor*>& outputs) {
    auto kh = mWeight->length(1);
    auto kw = mWeight->length(2);
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), inputs[2]->host<float>(), inputs[2]->size());
    // Reorder weight from whc -> pwhc4
    ::memset(mWeight->host<float>(), 0, mWeight->size());
    auto outputCount = inputs[0]->channel();
    auto weight      = mWeight->host<float>();
    auto tempWeight  = inputs[1]->host<float>();
    int cur          = 0;
    for (int c = 0; c < outputCount; ++c) {
        int plane  = c / 4;
        int offset = c % 4;
        for (int y = 0; y < kh; ++y) {
            for (int x = 0; x < kw; ++x) {
                float* dst = weight + offset + (x + y * kw) * 4 + kw * kh * 4 * plane;
                *dst       = tempWeight[cur++];
            }
        }
    }
    return CPUConvolutionDepthwise::BasicFloatExecution::onExecute(mTempInputs, outputs);
}

ErrorCode CPUConvolutionDepthwise::BasicFloatExecution::onResize(const std::vector<Tensor*>& inputs,
                                                                 const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto layer         = mCommon;
    auto inputTensor   = inputs[0];
    auto outputTensor  = outputs[0];
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
    int numberThread  = std::min(((CPUBackend*)backend())->threadNumber(), dst_depth_quad);
    auto runBasic     = [=](float* dst_z, const float* src_z, const float* weight_dz, int L, int T, int R, int B) {
        for (int dy = T; dy < B; ++dy) {
            float* dst_y        = dst_z + dy * dst_y_step;
            int srcStartY       = dy * strideY - padY;
            const float* src_dy = src_z + srcStartY * src_y_step;
            int sfy             = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
            int efy             = ALIMIN(kernel_height, UP_DIV(src_height - srcStartY, dilateY));
            for (int dx = L; dx < R; ++dx) {
                float* dst_x        = dst_y + 4 * dx;
                int srcStartX       = dx * strideX - padX;
                const float* src_dx = src_dy + srcStartX * 4;
                int sfx             = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                int efx             = ALIMIN(kernel_width, UP_DIV(src_width - srcStartX, dilateX));
                MNNConvRunForUnitDepthWise(dst_x, src_dx + (sfx * dilateX + sfy * dilateY * src_width) * 4,
                                           weight_dz + 4 * (kernel_width * sfy + sfx), efx - sfx, efy - sfy,
                                           4 * kernel_width, dilateX_step, dilateY_step);
            }
        }
    };
    auto bias   = inputs[2];
    auto weight = inputs[1];
    mExecutor   = [=](const float* srcOrigin, float* dstOrigin, int tId) {
        for (int dz = tId; dz < dst_depth_quad; dz += numberThread) {
            float* dst_z           = dstOrigin + dst_z_step * dz;
            const float* src_z     = srcOrigin + src_z_step * dz;
            float* bias_z          = bias->host<float>() + 4 * dz;
            const float* weight_dz = weight->host<float>() + dz * weight_z_step;
            runBasic(dst_z, src_z, weight_dz, 0, 0, dst_width, t);
            runBasic(dst_z, src_z, weight_dz, 0, b, dst_width, dst_height);
            runBasic(dst_z, src_z, weight_dz, 0, t, l, b);
            runBasic(dst_z, src_z, weight_dz, r, t, dst_width, b);
            if (r > l && b > t) {
                MNNConvRunForLineDepthwise(dst_z + t * dst_y_step + l * 4,
                                           src_z + (t * strideY - padY) * src_y_step + (l * strideX - padX) * 4,
                                           weight_dz, r - l, strideX * 4, kernel_width, kernel_height, dilateX_step,
                                           dilateY_step, b - t, src_y_step * strideY, dst_y_step);
            }
            postFunction(dst_z, bias_z, dst_width * dst_height, 1);
        }
    };
    mNumber = numberThread;

    return NO_ERROR;
}

ErrorCode CPUConvolutionDepthwise::BasicFloatExecution::onExecute(const std::vector<Tensor*>& inputs,
                                                                  const std::vector<Tensor*>& outputs) {
    auto inputTensor  = inputs[0];
    auto outputTensor = outputs[0];
    for (int batchIndex = 0; batchIndex < inputTensor->batch(); ++batchIndex) {
        const float* srcOrigin = inputTensor->host<float>() + batchIndex * inputTensor->stride(0);
        float* dstOrigin       = outputTensor->host<float>() + batchIndex * outputTensor->stride(0);
        MNN_CONCURRENCY_BEGIN(tId, mNumber) {
            mExecutor(srcOrigin, dstOrigin, (int)tId);
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

CPUConvolutionDepthwise::Int8Execution::Int8Execution(const Convolution2DCommon* convOp, Backend* b,
                                                      const ConvolutionIntFactory::Int8Common* common,
                                                      const float* bias, size_t biasSize)
    : MNN::CPUConvolution(convOp, b) {
    mQuan = common->quan;
    MNN_ASSERT(nullptr != mQuan);
    mBias.reset(ALIGN_UP4((int)biasSize));
    mBias.clear();
    ::memcpy(mBias.get(), bias, biasSize * sizeof(float));

    mAlpha.reset(ALIGN_UP4((int)biasSize));
    mAlpha.clear();
    ::memcpy(mAlpha.get(), common->alpha.get(), biasSize * sizeof(float));

    auto layer = mCommon;
    int kx     = layer->kernelX();
    int ky     = layer->kernelY();

    int outputCount = (int)biasSize;
    int dstCountD8  = UP_DIV(outputCount, gIntUnit);

    int cur = 0;
    mWeight.reset(dstCountD8 * gIntUnit * kx * ky);
    mWeight.clear();
    int8_t* reorderedWeight = mWeight.get();
    auto originWeight       = common->weight.get();
    for (int dz = 0; dz < outputCount; ++dz) {
        int dzD8   = dz / gIntUnit;
        int my     = dz % gIntUnit;
        auto dstDz = reorderedWeight + dzD8 * kx * ky * gIntUnit;

        for (int i = 0; i < kx * ky; ++i) {
            auto index        = i * gIntUnit;
            dstDz[index + my] = originWeight[cur++];
        }
    }
}

ErrorCode CPUConvolutionDepthwise::Int8Execution::onResize(const std::vector<Tensor*>& inputs,
                                                           const std::vector<Tensor*>& outputs) {
    auto result      = CPUConvolution::onResize(inputs, outputs);
    auto originInput = inputs[0];
    auto& ib         = mInputTempBuffer.buffer();
    ib.type          = halide_type_of<int8_t>();
    ib.dim[0].extent = UP_DIV(originInput->channel(), gIntUnit);
    ib.dim[3].extent = gIntUnit;
    ib.dim[1].extent = originInput->height();
    ib.dim[2].extent = originInput->width();
    TensorUtils::setLinearLayout(&mInputTempBuffer);

    backend()->onAcquireBuffer(&mInputTempBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mInputTempBuffer, Backend::DYNAMIC);

    return result;
}

ErrorCode CPUConvolutionDepthwise::Int8Execution::onExecute(const std::vector<Tensor*>& inputs,
                                                            const std::vector<Tensor*>& outputs) {
    auto layer         = mCommon;
    auto inputTensor   = inputs[0];
    auto outputTensor  = outputs[0];
    int src_width      = inputTensor->width();
    int src_height     = inputTensor->height();
    int dst_width      = outputTensor->width();
    int dst_height     = outputTensor->height();
    int dst_depth_quad = UP_DIV(layer->outputCount(), gIntUnit);
    int dst_z_step     = dst_width * dst_height * gIntUnit;
    int src_z_step     = mInputTempBuffer.buffer().dim[0].stride;
    int dst_y_step     = dst_width * gIntUnit;
    int src_y_step     = src_width * gIntUnit;
    int strideY        = layer->strideY();
    int strideX        = layer->strideX();
    int dilateX        = layer->dilateX();
    int dilateY        = layer->dilateY();
    int dilateY_step   = dilateY * src_width * gIntUnit;
    int dilateX_step   = dilateX * gIntUnit;
    int kernel_height  = layer->kernelY();
    int kernel_width   = layer->kernelX();
    int padX           = mPadX;
    int padY           = mPadY;
    int weight_z_step  = kernel_height * kernel_width * gIntUnit;

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
    for (int i=0; i<4; ++i) {
        mQuanScale[i] = mQuan->quantScale();
    }

    auto runBasic = [=](float* dst_z, const int8_t* src_z, const int8_t* weight_dz, const float* alpha_z, int L, int T,
                        int R, int B) {
        for (int dy = T; dy < B; ++dy) {
            float* dst_y  = dst_z + dy * dst_y_step;
            int srcStartY = dy * strideY - padY;
            auto src_dy   = src_z + srcStartY * src_y_step;
            int sfy       = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
            int efy       = ALIMIN(kernel_height, UP_DIV(src_height - srcStartY, dilateY));
            for (int dx = L; dx < R; ++dx) {
                float* dst_x  = dst_y + 4 * dx;
                int srcStartX = dx * strideX - padX;
                auto src_dx   = src_dy + srcStartX * 4;
                int sfx       = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                int efx       = ALIMIN(kernel_width, UP_DIV(src_width - srcStartX, dilateX));
                MNNConvRunForUnitDepthWiseInt8(dst_x, src_dx + (sfx * dilateX + sfy * dilateY * src_width) * 4,
                                               weight_dz + 4 * (kernel_width * sfy + sfx), efx - sfx, efy - sfy,
                                               4 * kernel_width, dilateX_step, dilateY_step, alpha_z);
            }
        }
    };
    for (int batchIndex = 0; batchIndex < inputTensor->batch(); ++batchIndex) {
        const float* srcOrigin = inputTensor->host<float>() + batchIndex * src_z_step * dst_depth_quad;
        float* dstOrigin       = outputTensor->host<float>() + batchIndex * dst_z_step * dst_depth_quad;

        MNN_CONCURRENCY_BEGIN(dz, dst_depth_quad) {
            float* dst_z_float       = dstOrigin + dst_z_step * dz;
            const float* src_z_float = srcOrigin + src_z_step * dz;

            auto dst_z = dst_z_float;
            auto src_z = (int8_t*)mInputTempBuffer.buffer().host + dz * mInputTempBuffer.buffer().dim[0].stride;

            MNNFloat2Int8(src_z_float, src_z, src_z_step / 4, mQuanScale, mQuan->aMin(), mQuan->aMax());

            const float* bias_z     = mBias.get() + gIntUnit * dz;
            const float* alpha_z    = mAlpha.get() + gIntUnit * dz;
            const int8_t* weight_dz = mWeight.get() + dz * weight_z_step;
            runBasic(dst_z, src_z, weight_dz, alpha_z, 0, 0, dst_width, t);
            runBasic(dst_z, src_z, weight_dz, alpha_z, 0, b, dst_width, dst_height);
            runBasic(dst_z, src_z, weight_dz, alpha_z, 0, t, l, b);
            runBasic(dst_z, src_z, weight_dz, alpha_z, r, t, dst_width, b);
            if (r > l) {
                for (int dy = t; dy < b; ++dy) {
                    float* dst_y  = dst_z + dy * dst_y_step;
                    int srcStartY = dy * strideY - padY;
                    auto src_dy   = src_z + srcStartY * src_y_step;
                    MNNConvRunForLineDepthWiseInt8(dst_y + l * 4, src_dy + (l * strideX - padX) * 4, weight_dz, r - l,
                                                   strideX * 4, kernel_width, kernel_height, dilateX_step, dilateY_step,
                                                   alpha_z);
                }
            }

            postFunction(dst_z_float, bias_z, dst_width * dst_height, 1);
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}

class CPUConvolutionDepthwiseCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto conv2D = op->main_as_Convolution2D();
        auto conv   = op->main_as_Convolution2D()->common();
        if (3 == inputs.size()) {
            return new CPUConvolutionDepthwise::MultiInputFloatExecution(conv, backend);
        }
        if (conv->dilateX() == 1 && conv->dilateY() == 1 && conv->strideX() == 1 && conv->strideY() == 1 &&
            conv->kernelX() == 3 && conv->kernelY() == 3 && conv2D->quanParameter() == nullptr) {
            return new ConvolutionDepthwise3x3(conv, backend, conv2D->weight()->data(), conv2D->weight()->size(),
                                               conv2D->bias()->data(), conv2D->bias()->size());
        }
        return new CPUConvolutionDepthwise(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConvolutionDepthwiseCreator, OpType_ConvolutionDepthwise);
} // namespace MNN
