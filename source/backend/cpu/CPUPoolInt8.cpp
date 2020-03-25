//
//  CPUPoolInt8.cpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUPoolInt8.hpp"
#include "core/Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#include "core/Concurrency.h"

#define DST_TILE 16
#define CACHE_SIZE 128

namespace MNN {

static void poolingMaxNHWCInt8(const Tensor *src, Tensor *dst, int sx, int sy, int kx, int ky, int px, int py) {
    const int inputHeight  = src->length(1);
    const int inputWidth   = src->length(2);
    const int outputHeight = dst->length(1);
    const int outputWidth  = dst->length(2);
    const int channel      = dst->length(3);
    int8_t result[CACHE_SIZE];

    const auto srcPtr = src->host<int8_t>();
    auto dstPtr       = dst->host<int8_t>();

    for (int oc = 0; oc < channel; oc += CACHE_SIZE) {
        const int realChannel = std::min(channel - oc, CACHE_SIZE);

        for (int oy = 0; oy < outputHeight; ++oy) {
            for (int ox = 0; ox < outputWidth; ++ox) {
                const int srcOriginX = ox * sx - px;
                const int srcOriginY = oy * sy - py;
                const int kxs        = std::max(0, -srcOriginX);
                const int kxe        = std::min(kx, inputWidth - srcOriginX);
                const int kys        = std::max(0, -srcOriginY);
                const int kye        = std::min(ky, inputHeight - srcOriginY);

                const int8_t *srcCurPtr = srcPtr + oc + (srcOriginX + srcOriginY * inputWidth) * channel;
                memset(result, INT8_MIN, sizeof(int8_t) * realChannel);
                for (int y = kys; y < kye; ++y) {
                    const int8_t *srcCurRowPtr = srcCurPtr + (y * inputWidth + kxs) * channel;
                    for (int x = kxs; x < kxe; ++x) {
                        const int8_t *srcCurChannlePtr = srcCurRowPtr;
                        int index                      = 0;
#ifdef MNN_USE_NEON
                        for (; index <= realChannel - 16; index += 16) {
                            int8x16_t maxValue   = vld1q_s8(result + index);
                            int8x16_t inputValue = vld1q_s8(srcCurChannlePtr);
                            srcCurChannlePtr += 16;
                            maxValue = vmaxq_s8(maxValue, inputValue);
                            vst1q_s8(result + index, maxValue);
                        }
                        for (; index <= realChannel - 8; index += 8) {
                            int8x8_t maxValue   = vld1_s8(result + index);
                            int8x8_t inputValue = vld1_s8(srcCurChannlePtr);
                            srcCurChannlePtr += 8;
                            maxValue = vmax_s8(maxValue, inputValue);
                            vst1_s8(result + index, maxValue);
                        }
#endif
                        for (; index < realChannel; ++index) {
                            result[index] = std::max(result[index], *srcCurChannlePtr++);
                        }
                        srcCurRowPtr += channel;
                    }
                }

                int8_t *dstCurPtr = dstPtr + oc + (ox + oy * outputWidth) * channel;
                memcpy(dstCurPtr, result, sizeof(int8_t) * realChannel);
            }
        }
    }
}

static void poolingAvgNHWCInt8(const Tensor *src, Tensor *dst, int sx, int sy, int kx, int ky, int px, int py) {
    const int inputHeight  = src->length(1);
    const int inputWidth   = src->length(2);
    const int outputHeight = dst->length(1);
    const int outputWidth  = dst->length(2);
    const int channel      = dst->length(3);
    int16_t result[CACHE_SIZE];

    const auto srcPtr = src->host<int8_t>();
    auto dstPtr       = dst->host<int8_t>();

    for (int oc = 0; oc < channel; oc += CACHE_SIZE) {
        const int realChannel = std::min(channel - oc, CACHE_SIZE);

        for (int oy = 0; oy < outputHeight; ++oy) {
            for (int ox = 0; ox < outputWidth; ++ox) {
                const int srcOriginX  = ox * sx - px;
                const int srcOriginY  = oy * sy - py;
                const int kxs         = std::max(0, -srcOriginX);
                const int kxe         = std::min(kx, inputWidth - srcOriginX);
                const int kys         = std::max(0, -srcOriginY);
                const int kye         = std::min(ky, inputHeight - srcOriginY);
                const int kernelCount = (kxe - kxs) * (kye - kys);

                const int8_t *srcCurPtr = srcPtr + oc + (srcOriginX + srcOriginY * inputWidth) * channel;
                memset(result, 0, sizeof(int16_t) * realChannel);
                for (int y = kys; y < kye; ++y) {
                    const int8_t *srcCurRowPtr = srcCurPtr + (y * inputWidth + kxs) * channel;
                    for (int x = kxs; x < kxe; ++x) {
                        const int8_t *srcCurChannlePtr = srcCurRowPtr;
                        int index                      = 0;
#ifdef MNN_USE_NEON
                        for (; index <= realChannel - 16; index += 16) {
                            int16x8_t accResult[2];
                            accResult[0]         = vld1q_s16(result + index);
                            accResult[1]         = vld1q_s16(result + index + 8);
                            int8x16_t inputValue = vld1q_s8(srcCurChannlePtr);
                            srcCurChannlePtr += 16;
                            accResult[0] = vaddw_s8(accResult[0], vget_low_s8(inputValue));
                            accResult[1] = vaddw_s8(accResult[1], vget_high_s8(inputValue));
                            vst1q_s16(result + index, accResult[0]);
                            vst1q_s16(result + index + 8, accResult[1]);
                        }
                        for (; index <= realChannel - 8; index += 8) {
                            int16x8_t accResult = vld1q_s16(result + index);
                            int8x8_t inputValue = vld1_s8(srcCurChannlePtr);
                            srcCurChannlePtr += 8;
                            accResult = vaddw_s8(accResult, inputValue);
                            vst1q_s16(result + index, accResult);
                        }
#endif
                        for (; index < realChannel; ++index) {
                            result[index] += *srcCurChannlePtr++;
                        }
                        srcCurRowPtr += channel;
                    }
                }

                int8_t *dstCurPtr = dstPtr + oc + (ox + oy * outputWidth) * channel;
                int index         = 0;
                for (; index < realChannel; ++index) {
                    int16_t a = result[index] > 0 ? (result[index] + kernelCount / 2) / kernelCount
                                                  : (result[index] - kernelCount / 2) / kernelCount;
                    dstCurPtr[index] = static_cast<int8_t>(a);
                }
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

    const int channel = input->channel();
    auto poolFunc     = poolingMaxNHWCInt8;
    if (mParameter->type() == MNN::PoolType_AVEPOOL) {
        poolFunc = poolingAvgNHWCInt8;
    }
    mInputTemp.reset(Tensor::createDevice<int8_t>({input->batch(), inputHeight, inputWidth, channel}));
    mOutputTemp.reset(Tensor::createDevice<int8_t>({output->batch(), outputHeight, outputWidth, channel}));

    bool allocSucc = backend()->onAcquireBuffer(mInputTemp.get(), Backend::DYNAMIC);
    allocSucc      = allocSucc && backend()->onAcquireBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    if (!allocSucc) {
        return OUT_OF_MEMORY;
    }

    mThreadFunction = [=](const Tensor *src, Tensor *dst) {
        poolFunc(src, dst, strideWidth, strideHeight, kernelWidth, kernelHeight, padWidth, padHeight);
    };

    backend()->onReleaseBuffer(mInputTemp.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUPoolInt8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    backend()->onCopyBuffer(input, mInputTemp.get());
    mThreadFunction(mInputTemp.get(), mOutputTemp.get());
    backend()->onCopyBuffer(mOutputTemp.get(), output);
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
