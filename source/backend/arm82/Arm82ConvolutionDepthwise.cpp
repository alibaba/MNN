//
//  Arm82ConvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2020/01/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/arm82/Arm82ConvolutionDepthwise.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "backend/arm82/Arm82OptFunc.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

extern "C" {
void MNNLineDepthWiseFp16C8Unit(FLOAT16* dst, const FLOAT16* src, const FLOAT16* weight, const FLOAT16* bias_z,
                                size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
                                size_t dilateY_step, size_t relu, size_t relu6);
}

namespace MNN {

static void MNNDepthWiseFp16C8Unit(FLOAT16* dst, const FLOAT16* src, const FLOAT16* weight, const FLOAT16* bias,
                                   size_t fw, size_t fh, size_t weight_y_step, size_t dilateX_step, size_t dilateY_step,
                                   size_t relu, size_t relu6) {
    int fx, fy;

#ifdef MNN_USE_NEON
    float16x8_t acc_value = vld1q_f16(bias);
#else
    FLOAT16 acc_value[ARMV82_CHANNEL_UNIT];
    memcpy(acc_value, bias, sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);
#endif

    for (fy = 0; fy < fh; ++fy) {
        const auto src_y    = src + fy * dilateY_step;
        const auto weight_y = weight + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            const auto weight_x = weight_y + fx * ARMV82_CHANNEL_UNIT;
            const auto src_x    = src_y + fx * dilateX_step;

#ifdef MNN_USE_NEON
            float16x8_t src_x_value    = vld1q_f16(src_x);
            float16x8_t weight_x_value = vld1q_f16(weight_x);
            acc_value                  = vfmaq_f16(acc_value, src_x_value, weight_x_value);
#else
            for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
                acc_value[j] += src_x[j] * weight_x[j];
            }
#endif
        }
    }

#ifdef MNN_USE_NEON
    if (relu) {
        float16x8_t zero_value = vdupq_n_f16(float16_t(0.0));
        acc_value              = vmaxq_f16(acc_value, zero_value);
    }
    if (relu6) {
        float16x8_t zero_value = vdupq_n_f16(float16_t(0.0));
        float16x8_t six_value  = vdupq_n_f16(float16_t(6.0));
        acc_value              = vmaxq_f16(acc_value, zero_value);
        acc_value              = vminq_f16(acc_value, six_value);
    }
    vst1q_f16(dst, acc_value);
#else
    if (relu) {
        for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
            if (acc_value[j] < 0) {
                acc_value[j] = 0;
            }
        }
    }
    if (relu6) {
        for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
            if (acc_value[j] < 0) {
                acc_value[j] = 0;
            }
            if (acc_value[j] > 6) {
                acc_value[j] = 6.0;
            }
        }
    }
    memcpy(dst, acc_value, sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);
#endif
}

#ifndef MNN_USE_NEON
static void MNNLineDepthWiseFp16C8Unit(FLOAT16* dst, const FLOAT16* src, const FLOAT16* weight, const FLOAT16* bias_z,
                                       size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
                                       size_t dilateY_step, size_t relu, size_t relu6) {
    int dx, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        auto dst_x = dst + dx * ARMV82_CHANNEL_UNIT;
        FLOAT16 dst_temp[ARMV82_CHANNEL_UNIT];
        memcpy(dst_temp, bias_z, sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);

        const auto src_z = src + src_w_step * dx;

        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * ARMV82_CHANNEL_UNIT;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                const auto weight_x = weight_y + fx * ARMV82_CHANNEL_UNIT;

                for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
                    dst_temp[j] += src_x[j] * weight_x[j];
                }
            }
        }

        if (relu) {
            for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
                if (dst_temp[j] < 0) {
                    dst_temp[j] = 0;
                }
            }
        }
        if (relu6) {
            for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
                if (dst_temp[j] < 0) {
                    dst_temp[j] = 0;
                }
                if (dst_temp[j] > 6) {
                    dst_temp[j] = 6.0;
                }
            }
        }

        memcpy(dst_x, dst_temp, sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);
    }
}
#endif

Arm82ConvolutionDepthwise::Arm82ConvolutionDepthwise(const MNN::Convolution2D* convParam, Backend* bn) : Execution(bn) {
    const auto commonParam = convParam->common();
    mCommon                = commonParam;
    mRelu = commonParam->relu();
    mRelu6 = commonParam->relu6();
    const int kx           = commonParam->kernelX();
    const int ky           = commonParam->kernelY();
    const int kernelSize   = kx * ky;

    const int outputChannel      = commonParam->outputCount();
    const int ocDivUnit          = UP_DIV(outputChannel, ARMV82_CHANNEL_UNIT);
    const int weightSizeAlignLen = ocDivUnit * ARMV82_CHANNEL_UNIT * kernelSize;
    mWeightFp16.reset(Tensor::createDevice<uint16_t>({weightSizeAlignLen}));
    auto success = bn->onAcquireBuffer(mWeightFp16.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    auto weightDstPtr = mWeightFp16->host<FLOAT16>();
    memset(weightDstPtr, 0, weightSizeAlignLen * sizeof(FLOAT16));
    
    const FLOAT16* fp16WeightPtr = nullptr;
    std::vector<FLOAT16> weightFp16;
    if(convParam->quanParameter()){
        MNN_ASSERT(convParam->quanParameter()->type() == 3);
        // the data type of weight is fp16
        fp16WeightPtr = reinterpret_cast<const FLOAT16*>(convParam->quanParameter()->buffer()->data());
    }else{
        // the data type of weight is fp32, then quantize weight to be fp16 data type
        int size = convParam->weight()->size();
        weightFp16.resize(size);
        MNNQuantizeFP16(weightFp16.data(), convParam->weight()->data(), size);
        fp16WeightPtr = weightFp16.data();
    }
    
    const auto weightSrcPtr = fp16WeightPtr;
    int cur                 = 0;
    for (int dz = 0; dz < outputChannel; ++dz) {
        const int dzi = dz / ARMV82_CHANNEL_UNIT;
        const int dzj = dz % ARMV82_CHANNEL_UNIT;

        auto dstDz = weightDstPtr + dzi * kernelSize * ARMV82_CHANNEL_UNIT + dzj;
        for (int k = 0; k < kernelSize; ++k) {
            dstDz[k * ARMV82_CHANNEL_UNIT] = weightSrcPtr[cur++];
        }
    }
    mBiasFp16.reset(Tensor::createDevice<uint16_t>({ocDivUnit * ARMV82_CHANNEL_UNIT}));
    success = bn->onAcquireBuffer(mBiasFp16.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }

    // TODO, bias is fp32, save bias also in fp16?
    auto biasDstPtr = mBiasFp16->host<FLOAT16>();
    memset(biasDstPtr, 0, mBiasFp16->size());

    MNNQuantizeFP16(biasDstPtr, convParam->bias()->data(), outputChannel);
}

Arm82ConvolutionDepthwise::~Arm82ConvolutionDepthwise() {
    if (mWeightFp16 != nullptr) {
        backend()->onReleaseBuffer(mWeightFp16.get(), Backend::STATIC);
    }
    if (mBiasFp16 != nullptr) {
        backend()->onReleaseBuffer(mBiasFp16.get(), Backend::STATIC);
    }
}

ErrorCode Arm82ConvolutionDepthwise::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    int padX = mCommon->padX();
    int padY = mCommon->padY();

    if (mCommon->padMode() == PadMode_SAME) {
        int kernelWidthSize  = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

        int padNeededWidth  = (output->width() - 1) * mCommon->strideX() + kernelWidthSize - input->width();
        int padNeededHeight = (output->height() - 1) * mCommon->strideY() + kernelHeightSize - input->height();
        padX                = padNeededWidth / 2;
        padY                = padNeededHeight / 2;
    }

    const int src_width      = input->width();
    const int src_height     = input->height();
    const int dst_width      = output->width();
    const int dst_height     = output->height();
    const int dst_depth_quad = UP_DIV(output->channel(), ARMV82_CHANNEL_UNIT);
    const int dst_z_step     = dst_width * dst_height * ARMV82_CHANNEL_UNIT;
    const int src_z_step     = src_width * src_height * ARMV82_CHANNEL_UNIT;
    const int dst_y_step     = dst_width * ARMV82_CHANNEL_UNIT;
    const int src_y_step     = src_width * ARMV82_CHANNEL_UNIT;
    const int strideY        = mCommon->strideY();
    const int strideX        = mCommon->strideX();
    const int dilateY        = mCommon->dilateY();
    const int dilateX        = mCommon->dilateX();
    const int dilateY_step   = dilateY * src_width * ARMV82_CHANNEL_UNIT;
    const int dilateX_step   = dilateX * ARMV82_CHANNEL_UNIT;
    const int kernel_height  = mCommon->kernelY();
    const int kernel_width   = mCommon->kernelX();
    const int weight_z_step  = kernel_width * kernel_height * ARMV82_CHANNEL_UNIT;
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

    const auto weightPtr   = mWeightFp16->host<FLOAT16>();
    const auto biasPtr     = mBiasFp16->host<FLOAT16>();
    const int threadNumber = static_cast<Arm82Backend*>(backend())->numberThread();
    mThreadNumber          = std::min(threadNumber, dst_depth_quad);
    auto runBasic = [=](FLOAT16* dst_z, const FLOAT16* src_z, const FLOAT16* weight_dz, const FLOAT16* bias_z, int L,
                        int T, int R, int B) {
        for (int dy = T; dy < B; ++dy) {
            auto dst_y          = dst_z + dy * dst_y_step;
            const int srcStartY = dy * strideY - padY;
            const auto src_y    = src_z + srcStartY * src_y_step;
            const int sfy       = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
            const int efy       = ALIMIN(kernel_height, (UP_DIV(src_height - srcStartY, dilateY)));
            for (int dx = L; dx < R; ++dx) {
                auto dst_x            = dst_y + ARMV82_CHANNEL_UNIT * dx;
                const int srcStartX   = dx * strideX - padX;
                const auto src_x      = src_y + srcStartX * ARMV82_CHANNEL_UNIT;
                const int sfx         = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                const int efx         = ALIMIN(kernel_width, (UP_DIV(src_width - srcStartX, dilateX)));
                const int srcIndex    = (sfx * dilateX + sfy * dilateY * src_width) * ARMV82_CHANNEL_UNIT;
                const int weightIndex = (kernel_width * sfy + sfx) * ARMV82_CHANNEL_UNIT;

                MNNDepthWiseFp16C8Unit(dst_x, src_x + srcIndex, weight_dz + weightIndex, bias_z, efx - sfx, efy - sfy,
                                       ARMV82_CHANNEL_UNIT * kernel_width, dilateX_step, dilateY_step,
                                       (size_t)mRelu, (size_t)mRelu6);
            }
        }
    };

    mThreadFunction = [=](int tId, const FLOAT16* src, FLOAT16* dst) {
        for (int dz = tId; dz < dst_depth_quad; dz += mThreadNumber) {
            const auto src_z     = src + dz * src_z_step;
            const auto weight_dz = weightPtr + dz * weight_z_step;
            const auto bias_dz   = biasPtr + dz * ARMV82_CHANNEL_UNIT;
            auto dst_z           = dst + dz * dst_z_step;
            runBasic(dst_z, src_z, weight_dz, bias_dz, 0, 0, dst_width, t);
            runBasic(dst_z, src_z, weight_dz, bias_dz, 0, b, dst_width, dst_height);
            runBasic(dst_z, src_z, weight_dz, bias_dz, 0, t, l, b);
            runBasic(dst_z, src_z, weight_dz, bias_dz, r, t, dst_width, b);
            if (r > l) {
                for (int dy = t; dy < b; ++dy) {
                    const int srcStartY = dy * strideY - padY;
                    const auto src_dy   = src_z + srcStartY * src_y_step;
                    auto dst_y          = dst_z + dy * dst_y_step;
                    MNNLineDepthWiseFp16C8Unit(
                        dst_y + l * ARMV82_CHANNEL_UNIT, src_dy + (l * strideX - padX) * ARMV82_CHANNEL_UNIT, weight_dz,
                        bias_dz, r - l, strideX * ARMV82_CHANNEL_UNIT, kernel_width, kernel_height, dilateX_step,
                        dilateY_step, (size_t)mRelu, (size_t)mRelu6);
                }
            }
        }
    };

    return NO_ERROR;
}

ErrorCode Arm82ConvolutionDepthwise::onExecute(const std::vector<Tensor*>& inputs,
                                               const std::vector<Tensor*>& outputs) {

    auto input           = inputs[0];
    auto output          = outputs[0];
    const int batch      = input->batch();
    
    const int inBatchStride = ROUND_UP(input->channel(), ARMV82_CHANNEL_UNIT) * input->height() * input->width();
    const int outBatchStride = ROUND_UP(output->channel(), ARMV82_CHANNEL_UNIT) * output->height() * output->width();

    const auto inputPtr = input->host<FLOAT16>();
    auto outputPtr      = output->host<FLOAT16>();

    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto srcOrigin = inputPtr + bIndex * inBatchStride;
        auto dstOrigin       = outputPtr + bIndex * outBatchStride;

        MNN_CONCURRENCY_BEGIN(tId, mThreadNumber)
            mThreadFunction((int)tId, srcOrigin, dstOrigin);
#ifdef MNN_USE_THREAD_POOL
        MNN_CONCURRENCY_ARM82_END();
#else
        MNN_CONCURRENCY_END();
#endif
    }
    return NO_ERROR;
}

class Arm82ConvolutionDepthwiseCreator : public Arm82Backend::Arm82Creator {
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new Arm82ConvolutionDepthwise(op->main_as_Convolution2D(), backend);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_ConvolutionDepthwise, Arm82ConvolutionDepthwiseCreator);

} // namespace MNN
