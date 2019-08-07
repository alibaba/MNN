//
//  CPUDepthwiseConvInt8.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUDepthwiseConvInt8.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "Macro.h"
#include <math.h>

#define UNIT 4

extern "C" {
void MNNDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                      size_t fw, size_t fh, size_t weight_y_step, size_t dilateX_step,
                                      size_t dilateY_step, const float* scale);
void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z,
                                          size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
                                          size_t dilateY_step, const float* scale_z);
}

namespace MNN {

#ifndef MNN_USE_NEON
inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(roundf(value));
}

static void MNNDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                             size_t fw, size_t fh, size_t weight_y_step, size_t dilateX_step,
                                             size_t dilateY_step, const float* scale) {
    int fx, fy;

    int dst_temp[UNIT] = {0, 0, 0, 0};

    for (fy = 0; fy < fh; ++fy) {
        const auto src_y    = src + fy * dilateY_step;
        const auto weight_y = weight + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            const auto weight_x = weight_y + fx * UNIT;
            const auto src_x    = src_y + fx * dilateX_step;
            for (int j = 0; j < UNIT; ++j) {
                dst_temp[j] += (int32_t)src_x[j] * (int32_t)weight_x[j];
            }
        }
    }
    for (int i = 0; i < UNIT; ++i) {
        dst[i] = int32ToInt8(dst_temp[i], bias[i], scale[i]);
    }
}

static void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                 const int32_t* bias_z, size_t width, size_t src_w_step, size_t fw,
                                                 size_t fh, size_t dilateX_step, size_t dilateY_step,
                                                 const float* scale_z) {
    int dx, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        auto dst_x          = dst + dx * 4;
        int32_t dstInt32[4] = {0, 0, 0, 0};
        const auto src_z    = src + src_w_step * dx;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * 4;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                const auto weight_x = weight_y + 4 * fx;
                for (int j = 0; j < UNIT; ++j) {
                    dstInt32[j] += (int32_t)src_x[j] * (int32_t)weight_x[j];
                }
            }
        }

        for (int i = 0; i < UNIT; ++i) {
            dst_x[i] = int32ToInt8(dstInt32[i], bias_z[i], scale_z[i]);
        }
    }
}

#endif

CPUDepthwiseConvInt8::CPUDepthwiseConvInt8(Backend* backend, const MNN::Convolution2D* dwConvParam)
    : CPUConvolution(dwConvParam->common(), backend) {
    auto common               = dwConvParam->common();
    mRelu                     = common->relu6() || common->relu();
    const int kx              = common->kernelX();
    const int ky              = common->kernelY();
    const int kernelSize      = kx * ky;
    const int outputCount     = common->outputCount();
    const int ocDivUnit       = UP_DIV(outputCount, UNIT);
    const int weightSizeAlign = ocDivUnit * UNIT * kernelSize;
    mWeightInt8.reset(Tensor::createDevice<int8_t>({weightSizeAlign}));
    auto allocRes = backend->onAcquireBuffer(mWeightInt8.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    auto weightPtr = mWeightInt8->host<int8_t>();
    memset(weightPtr, 0, weightSizeAlign * sizeof(int8_t));
    const auto originWeight = dwConvParam->symmetricQuan()->weight()->data();
    int cur                 = 0;
    for (int dz = 0; dz < outputCount; ++dz) {
        const int dzDivUnit = dz / UNIT;
        const int my        = dz % UNIT;
        auto dstDz          = weightPtr + dzDivUnit * kernelSize * UNIT;
        for (int i = 0; i < kernelSize; ++i) {
            dstDz[i * UNIT + my] = originWeight[cur++];
        }
    }

    mBiasInt32.reset(Tensor::createDevice<int32_t>({ocDivUnit * UNIT}));
    allocRes = backend->onAcquireBuffer(mBiasInt32.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    auto biasPtr = mBiasInt32->host<int32_t>();
    memset(biasPtr, 0, ocDivUnit * UNIT * sizeof(int32_t));
    memcpy(biasPtr, dwConvParam->symmetricQuan()->bias()->data(), outputCount * sizeof(int32_t));

    mScaleFloat.reset(Tensor::createDevice<int32_t>({ocDivUnit * UNIT}));
    allocRes = backend->onAcquireBuffer(mScaleFloat.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    auto scalePtr = mScaleFloat->host<float>();
    memset(scalePtr, 0, ocDivUnit * UNIT * sizeof(float));
    memcpy(scalePtr, dwConvParam->symmetricQuan()->scale()->data(), outputCount * sizeof(float));
}

ErrorCode CPUDepthwiseConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    CPUConvolution::onResize(inputs, outputs);

    int padX = mPadX;
    int padY = mPadY;

    const int src_width      = input->width();
    const int src_height     = input->height();
    const int dst_width      = output->width();
    const int dst_height     = output->height();
    const int dst_depth_quad = UP_DIV(output->channel(), UNIT);
    const int dst_z_step     = dst_width * dst_height * UNIT;
    const int src_z_step     = src_width * src_height * UNIT;
    const int dst_y_step     = dst_width * UNIT;
    const int src_y_step     = src_width * UNIT;
    const int strideY        = mCommon->strideY();
    const int strideX        = mCommon->strideX();
    const int dilateY        = mCommon->dilateY();
    const int dilateX        = mCommon->dilateX();
    const int dilateY_step   = dilateY * src_width * UNIT;
    const int dilateX_step   = dilateX * UNIT;
    const int kernel_height  = mCommon->kernelY();
    const int kernel_width   = mCommon->kernelX();
    const int weight_z_step  = kernel_width * kernel_height * UNIT;
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

    const auto weightPtr   = mWeightInt8->host<int8_t>();
    const auto biasPtr     = mBiasInt32->host<int32_t>();
    const auto scalePtr    = mScaleFloat->host<float>();
    const int threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    mThreadNumber          = std::min(threadNumber, dst_depth_quad);

    auto runBasic = [=](int8_t* dst_z, const int8_t* src_z, const int8_t* weight_dz, const int32_t* bias_z,
                        const float* scale_z, int L, int T, int R, int B) {
        for (int dy = T; dy < B; ++dy) {
            auto dst_y          = dst_z + dy * dst_y_step;
            const int srcStartY = dy * strideY - padY;
            const auto src_y    = src_z + srcStartY * src_y_step;
            const int sfy       = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
            const int efy       = ALIMIN(kernel_height, (UP_DIV(src_height - srcStartY, dilateY)));
            for (int dx = L; dx < R; ++dx) {
                auto dst_x            = dst_y + 4 * dx;
                const int srcStartX   = dx * strideX - padX;
                const auto src_x      = src_y + srcStartX * 4;
                const int sfx         = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                const int efx         = ALIMIN(kernel_width, (UP_DIV(src_width - srcStartX, dilateX)));
                const int srcIndex    = (sfx * dilateX + sfy * dilateY * src_width) * 4;
                const int weightIndex = (kernel_width * sfy + sfx) * 4;

                MNNDepthWiseInt8AddBiasScaleUnit(dst_x, src_x + srcIndex, weight_dz + weightIndex, bias_z, efx - sfx,
                                                 efy - sfy, 4 * kernel_width, dilateX_step, dilateY_step, scale_z);
            }
        }
    };

    mThreadFunction = [=](int tId, const int8_t* src, int8_t* dst) {
        for (int dz = tId; dz < dst_depth_quad; dz += mThreadNumber) {
            const auto src_z     = src + dz * src_z_step;
            const auto weight_dz = weightPtr + dz * weight_z_step;
            const auto bias_dz   = biasPtr + dz * UNIT;
            const auto scale_dz  = scalePtr + dz * UNIT;
            auto dst_z           = dst + dz * dst_z_step;
            runBasic(dst_z, src_z, weight_dz, bias_dz, scale_dz, 0, 0, dst_width, t);
            runBasic(dst_z, src_z, weight_dz, bias_dz, scale_dz, 0, b, dst_width, dst_height);
            runBasic(dst_z, src_z, weight_dz, bias_dz, scale_dz, 0, t, l, b);
            runBasic(dst_z, src_z, weight_dz, bias_dz, scale_dz, r, t, dst_width, b);
            if (r > l) {
                for (int dy = t; dy < b; ++dy) {
                    const int srcStartY = dy * strideY - padY;
                    const auto src_dy   = src_z + srcStartY * src_y_step;
                    auto dst_y          = dst_z + dy * dst_y_step;
                    MNNLineDepthWiseInt8AddBiasScaleUnit(dst_y + l * 4, src_dy + (l * strideX - padX) * 4, weight_dz,
                                                         bias_dz, r - l, strideX * 4, kernel_width, kernel_height,
                                                         dilateX_step, dilateY_step, scale_dz);
                }
            }

            if (mRelu) {
                MNNReluInt8(dst_z, dst_z, dst_z_step);
            }
        }
    };

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

    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto srcOrigin = inputPtr + bIndex * src_b_step;
        auto dstOrigin       = outputPtr + bIndex * dst_b_step;

        MNN_CONCURRENCY_BEGIN(tId, mThreadNumber) {
            mThreadFunction((int)tId, srcOrigin, dstOrigin);
        }
        MNN_CONCURRENCY_END();
    }
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
