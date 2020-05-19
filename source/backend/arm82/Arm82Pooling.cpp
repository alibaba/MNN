//
//  Arm82Pooling.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/arm82/Arm82Pooling.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

static void poolingMaxFp16Unit(FLOAT16 *dst, int outputWidth, int outputHeight, const FLOAT16 *src, int inputWidth,
                               int inputHeight, int kernelWidth, int kernelHeight, int strideWidth, int strideHeight,
                               int padWidth, int padHeight) {
    for (int oy = 0; oy < outputHeight; ++oy) {
        for (int ox = 0; ox < outputWidth; ++ox) {
            const int srcOriginX = ox * strideWidth - padWidth;
            const int srcOriginY = oy * strideHeight - padHeight;
            const int kxs        = std::max(0, -srcOriginX);
            const int kxe        = std::min(kernelWidth, inputWidth - srcOriginX);
            const int kys        = std::max(0, -srcOriginY);
            const int kye        = std::min(kernelHeight, inputHeight - srcOriginY);

            auto dstCurPtr = dst + (oy * outputWidth + ox) * ARMV82_CHANNEL_UNIT;

#ifdef MNN_USE_NEON
            float16x8_t curIn, curOut;
            curOut = vdupq_n_f16(float16_t(-65504.0));
#else
            // init
            FLOAT16 curOut[ARMV82_CHANNEL_UNIT];
            for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                curOut[i] = -65504.0;
            }
#endif
            for (int y = kys; y < kye; ++y) {
                for (int x = kxs; x < kxe; ++x) {
                    const int inOffset = ((srcOriginY + y) * inputWidth + srcOriginX + x) * ARMV82_CHANNEL_UNIT;
#ifdef MNN_USE_NEON
                    curIn  = vld1q_f16(src + inOffset);
                    curOut = vmaxq_f16(curIn, curOut);
#else
                    for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                        curOut[i] = std::max(curOut[i], src[inOffset + i]);
                    }
#endif
                }
            }
#ifdef MNN_USE_NEON
            vst1q_f16(dstCurPtr, curOut);
#else
            memcpy(dstCurPtr, curOut, sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);
#endif
        }
    }
}

static void poolingAvgFp16Unit(FLOAT16 *dst, int outputWidth, int outputHeight, const FLOAT16 *src, int inputWidth,
                               int inputHeight, int kernelWidth, int kernelHeight, int strideWidth, int strideHeight,
                               int padWidth, int padHeight) {
    for (int oy = 0; oy < outputHeight; ++oy) {
        for (int ox = 0; ox < outputWidth; ++ox) {
            const int srcOriginX  = ox * strideWidth - padWidth;
            const int srcOriginY  = oy * strideHeight - padHeight;
            const int kxs         = std::max(0, -srcOriginX);
            const int kxe         = std::min(kernelWidth, inputWidth - srcOriginX);
            const int kys         = std::max(0, -srcOriginY);
            const int kye         = std::min(kernelHeight, inputHeight - srcOriginY);
            const int kernelCount = (kxe - kxs) * (kye - kys);

            auto dstCurPtr = dst + (oy * outputWidth + ox) * ARMV82_CHANNEL_UNIT;

#ifdef MNN_USE_NEON
            float16x8_t curIn, curOut;
            curOut           = vdupq_n_f16(float16_t(0));
            float16x8_t size = vdupq_n_f16(float16_t(kernelCount));
#else
            // init
            FLOAT16 curOut[ARMV82_CHANNEL_UNIT];
            for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                curOut[i] = 0;
            }
#endif
            for (int y = kys; y < kye; ++y) {
                for (int x = kxs; x < kxe; ++x) {
                    const int inOffset = ((srcOriginY + y) * inputWidth + srcOriginX + x) * ARMV82_CHANNEL_UNIT;
                    const auto srcUnit = src + inOffset;
#ifdef MNN_USE_NEON
                    curIn  = vld1q_f16(srcUnit);
                    curOut = vaddq_f16(curIn, curOut);
#else
                    for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                        curOut[i] = curOut[i] + srcUnit[i];
                    }
#endif
                }
            }
#ifdef MNN_USE_NEON
            vst1q_f16(dstCurPtr, vdivq_f16(curOut, size));
#else
            for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                curOut[i] = curOut[i] / kernelCount;
            }
            memcpy(dstCurPtr, curOut, sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);
#endif
        }
    }
}

Arm82Pooling::Arm82Pooling(Backend *bn, const Pool *parameter) : Execution(bn), mParameter(parameter) {
}

ErrorCode Arm82Pooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    }else if(mParameter->padType() == PoolPadType_VALID){
        padWidth = 0;
        padHeight = 0;
    }

    const int inputPlaneStride  = inputWidth * inputHeight * ARMV82_CHANNEL_UNIT;
    const int outputPlaneStride = outputWidth * outputHeight * ARMV82_CHANNEL_UNIT;

    auto planeFunc = poolingMaxFp16Unit;
    if (mParameter->type() == MNN::PoolType_AVEPOOL) {
        planeFunc = poolingAvgFp16Unit;
    }

    const int channelDiv4 = UP_DIV(input->channel(), ARMV82_CHANNEL_UNIT);
    const int threadNumer = static_cast<Arm82Backend *>(backend())->numberThread();
    mThreadNumber         = std::min(threadNumer, channelDiv4);

    mThreadFunction = [=](int tId, const FLOAT16 *src, FLOAT16 *dst) {
        for (int depth = tId; depth < channelDiv4; depth += mThreadNumber) {
            planeFunc(dst + depth * outputPlaneStride, outputWidth, outputHeight, src + depth * inputPlaneStride,
                      inputWidth, inputHeight, kernelWidth, kernelHeight, strideWidth, strideHeight, padWidth,
                      padHeight);
        }
    };

    return NO_ERROR;
}

ErrorCode Arm82Pooling::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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

class Arm82PoolingCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new Arm82Pooling(backend, op->main_as_Pool());
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_Pooling, Arm82PoolingCreator);

} // namespace MNN
