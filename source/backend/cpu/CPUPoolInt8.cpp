//
//  CPUPoolInt8.cpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUPoolInt8.hpp"
#include "Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#include "Concurrency.h"

#define UNIT 4

namespace MNN {

static void poolingInt8Max(int8_t *dst, int outputWidth, int outputHeight, const int8_t *src, int inputWidth,
                           int inputHeight, int kernelWidth, int kernelHeight, int strideWidth, int strideHeight,
                           int padWidth, int padHeight) {
    int8_t maxValue[2 * UNIT];
    for (int oy = 0; oy < outputHeight; ++oy) {
        for (int ox = 0; ox < outputWidth; ++ox) {
            memset(maxValue, INT8_MIN, 2 * UNIT * sizeof(int8_t));
            const int srcOriginX  = ox * strideWidth - padWidth;
            const int srcOriginY  = oy * strideHeight - padHeight;
            const int kxs         = std::max(0, -srcOriginX);
            const int kxe         = std::min(kernelWidth, inputWidth - srcOriginX);
            const int kys         = std::max(0, -srcOriginY);
            const int kye         = std::min(kernelHeight, inputHeight - srcOriginY);

            const auto srcPtr = src + (srcOriginY * inputWidth + srcOriginX) * UNIT;
            auto dstPtr       = dst + (oy * outputWidth + ox) * UNIT;
            // find kernel_w * kernel_h max value
            for (int ky = kys; ky < kye; ++ky) {
                const auto srcPtrRow = srcPtr + ky * inputWidth + kxs;
                int kx               = kxs;
#ifdef MNN_USE_NEON
                // process two data together
                int8x8_t max_reg = vld1_s8(maxValue);
                for (; kx < kxe - 2; kx += 2) {
                    const auto srcPtrStart = srcPtrRow + kx * UNIT;
                    int8x8_t input_reg     = vld1_s8(srcPtrStart);
                    max_reg                = vmax_s8(max_reg, input_reg);
                    vst1_s8(maxValue, max_reg);
                }
                for (int j = 0; j < UNIT; ++j) {
                    maxValue[j] = std::max(maxValue[j], maxValue[j + UNIT]);
                }
#else
                for (; kx < kxe; ++kx) {
                    const auto srcPtrStart = srcPtrRow + kx * UNIT;
                    for (int j = 0; j < UNIT; ++j) {
                        maxValue[j] = std::max(maxValue[j], srcPtrStart[j]);
                    }
                }
#endif
            }
            // output
            memcpy(dstPtr, maxValue, UNIT * sizeof(int8_t));
        }
    }
}

static void poolingInt8Avg(int8_t *dst, int outputWidth, int outputHeight, const int8_t *src, int inputWidth,
                           int inputHeight, int kernelWidth, int kernelHeight, int strideWidth, int strideHeight,
                           int padWidth, int padHeight) {
    int16_t sum[2 * UNIT];
    for (int oy = 0; oy < outputHeight; ++oy) {
        for (int ox = 0; ox < outputWidth; ++ox) {
            memset(sum, 0, 2 * UNIT * sizeof(int16_t));
            const int srcOriginX  = ox * strideWidth - padWidth;
            const int srcOriginY  = oy * strideHeight - padHeight;
            const int kxs         = std::max(0, -srcOriginX);
            const int kxe         = std::min(kernelWidth, inputWidth - srcOriginX);
            const int kys         = std::max(0, -srcOriginY);
            const int kye         = std::min(kernelHeight, inputHeight - srcOriginY);
            const int kernelCount = (kxe - kxs) * (kye - kys);

            const auto srcPtr = src + (srcOriginY * inputWidth + srcOriginX) * UNIT;
            auto dstPtr       = dst + (oy * outputWidth + ox) * UNIT;
            // compute kernel_w * kernel_h sum
            for (int ky = kys; ky < kye; ++ky) {
                const auto srcPtrRow = srcPtr + ky * inputWidth + kxs;
                int kx               = kxs;
#ifdef MNN_USE_NEON
                // process two data together
                int16x8_t sum_reg = vld1q_s16(sum);
                for (; kx < kxe - 2; kx += 2) {
                    const auto srcPtrStart = srcPtrRow + kx * UNIT;
                    int8x8_t input_reg     = vld1_s8(srcPtrStart);
                    sum_reg                = vaddw_s8(sum_reg, input_reg);
                    vst1_s16(sum, vadd_s16(vget_high_s16(sum_reg), vget_low_s16(sum_reg)));
                }
#else
                for (; kx < kxe; ++kx) {
                    const auto srcPtrStart = srcPtrRow + kx * UNIT;
                    for (int j = 0; j < UNIT; ++j) {
                        sum[j] += srcPtrStart[j];
                    }
                }
#endif
            }
            // avg
            for (int j = 0; j < UNIT; ++j) {
                sum[j + UNIT] =
                    sum[j] > 0 ? (sum[j] + kernelCount / 2) / kernelCount : (sum[j] - kernelCount / 2) / kernelCount;
                dstPtr[j] = static_cast<int8_t>(sum[j + UNIT]);
            }
        }
    }
}

CPUPoolInt8::CPUPoolInt8(Backend *b, const Pool *parameter) : Execution(b), mParameter(parameter) {
}

ErrorCode CPUPoolInt8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    int strideWidth  = mParameter->strideX();
    int strideHeight = mParameter->strideY();
    int padWidth     = mParameter->padX();
    int padHeight    = mParameter->padY();
    int kernelWidth  = mParameter->kernelX();
    int kernelHeight = mParameter->kernelY();

    const int inputWidth   = input->width();
    const int inputHeight  = input->height();
    const int outputWidth  = output->width();
    const int outputHeight = output->height();

    kernelWidth  = std::min(kernelWidth, inputWidth);
    kernelHeight = std::min(kernelHeight, inputHeight);
    if (mParameter->isGlobal()) {
        kernelWidth  = inputWidth;
        kernelHeight = inputHeight;
        strideWidth  = inputWidth;
        strideHeight = inputHeight;
        padWidth     = 0;
        padHeight    = 0;
    }
    if (mParameter->padType() == PoolPadType_SAME) {
        int padNeededWidth  = (outputWidth - 1) * strideWidth + kernelWidth - inputWidth;
        int padNeededHeight = (outputHeight - 1) * strideHeight + kernelHeight - inputHeight;
        padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    }

    const int inputPlaneStride  = inputWidth * inputHeight * 4;
    const int outputPlaneStride = outputWidth * outputHeight * 4;

    auto planeFunc = poolingInt8Max;
    if (mParameter->type() == MNN::PoolType_AVEPOOL) {
        planeFunc = poolingInt8Avg;
    }

    const int channelDiv4 = UP_DIV(input->channel(), 4);
    const int threadNumer = static_cast<CPUBackend *>(backend())->threadNumber();
    mThreadNumber         = std::min(threadNumer, channelDiv4);

    mThreadFunction = [=](int tId, const int8_t *src, int8_t *dst) {
        for (int depth = tId; depth < channelDiv4; depth += mThreadNumber) {
            planeFunc(dst + depth * outputPlaneStride, outputWidth, outputHeight, src + depth * inputPlaneStride,
                      inputWidth, inputHeight, kernelWidth, kernelHeight, strideWidth, strideHeight, padWidth,
                      padHeight);
        }
    };

    return NO_ERROR;
}

ErrorCode CPUPoolInt8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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

class CPUPoolInt8Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUPoolInt8(backend, op->main_as_Pool());
    }
};

REGISTER_CPU_OP_CREATOR(CPUPoolInt8Creator, OpType_PoolInt8);

} // namespace MNN
