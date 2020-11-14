//
//  CPULRN.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPULRN.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

CPULRN::CPULRN(Backend *backend, int regionType, int localSize, float alpha, float beta)
    : Execution(backend), mRegionType(regionType), mLocalSize(localSize), mAlpha(alpha), mBeta(beta) {
    // nothing to do
}

// powfParam[0...5]: taylor expansion param (level = 5)
// powfParam[6] = pow(3/2, -betaFrac), this number have not error
static void initPowfContext(float beta, float* powfParam) {
    beta = beta - int(beta); // betaFrac
    powfParam[0] = 1;
    for (int i = 1; i < 6; ++i) {
        powfParam[i] = -powfParam[i - 1] * (beta + i - 1) / i;
    }
    powfParam[6] = powf(1.5, -beta);
}

// dst = src^(-beta), src >= 1, beta > 0
/*
 f(x) = x^(-beta), x >= 1, beta > 0
 taylor expansion: f(x) = f(1) + f'(1)(x-1) + f''(1)/2!*(x-1)^2 + ... + f'n'(1)/n!*(x-1)^n
 f'n'(1) = (-1)^n * beta * (beta + 1) * ... * (beta + n - 1)
 R(x) = h'(n+1)'(sigma)/(n+1)!*(x-1)^(n+1), min(1, x) <= sigma <= max(1, x)
 |R(x)| = (\prod_{i=0}^{n}(beta + i)/(1 + i)) * (x-1)^(n+1) / sigma^(beta+n+1)
 |R(x)| will close to 0 as n increase, when |(beta + i) / (1 + i)| < 1 and |(x-1) / sigma| < 1, that is 0 <= beta < 1, 0.5 < x < 2
 f(x) = x^(-beta), beta = betaInt + betaFrac >= 0, betaInt is integer part of beta, betaFrac is frac part of beta
 so, f(x) = x^(-betaInt-betaFrac) = (1/x)^betaInt * g(x)
 g(x) = x^(-betaFrac), 0 <= betaFrac < 1, x >= 1
 x = (3/2)^m * b, 0.8 <= b < 1.25
 g(x) = pow(3/2, -betaFrac) * h(b) = C * h(b)
 we pre compute pow(3/2, -betaFrac), make it a constant.
 h(x) = x^(-betaFrac), 0.8 <= x < 1.25, 0<= betaFrac < 1, so we can compute it by taylor expansion.
 finally, f(x) = x^(-beta) = (1/x)^betaInt * C^m * b^(-betaFrac), C = pow(3/2, -betaFrac)
*/
static void powfWithContext(float* dst, float* src, float beta, const int dataSize, const float* powfParam) {
    int countC8 = dataSize / 8;
    int betaInt = (int)beta;
    if (countC8 > 0) {
        MNNPowC8(dst, src, powfParam, betaInt, countC8);
    }
    int remain = countC8 * 8;
    const float powfConstant = powfParam[6];
    for (int i = remain; i < dataSize; ++i) {
        float result = 1, x, xInv = 1/src[i];
        // result = (1/x)^betaInt
        for (int j = 0; j < betaInt; result *= xInv, ++j);
        // result = result * ((3/2)^(-betaFrac))^m = (1/x)^betaInt * ((3/2)^(-betaFrac))^m
        for (x = src[i]; x >= 1.25; x /= 1.5, result *= powfConstant);
        // result = result * b^(-betaFrac) = f(x)
        float t = x - 1;
        float powRemain = powfParam[0] + t * (powfParam[1] + t * (powfParam[2] + t * (powfParam[3] + t * (powfParam[4] + t * powfParam[5]))));
        result *= powRemain;
        dst[i] = result;
    }
}

ErrorCode CPULRN::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // input transform space
    auto &input = inputs[0]->buffer();
    memcpy(mStorage.buffer().dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
    mStorage.buffer().dim[0].extent = 1;
    backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);

    // square space
    memcpy(mSquare.buffer().dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
    mSquare.buffer().dim[0].extent = 1;
    if (mRegionType == 1) {
        mSquare.buffer().dim[1].extent = ((CPUBackend*)backend())->threadNumber();
    }
    if (mRegionType == 1 && mLocalSize > 1) {
        mSquare.buffer().dim[3].extent += mLocalSize;
        mSquare.buffer().dim[2].extent += mLocalSize;
    }
    backend()->onAcquireBuffer(&mSquare, Backend::DYNAMIC);

    // release temp buffer space
    backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mSquare, Backend::DYNAMIC);
    return NO_ERROR;
}

void CPULRN::executeAcrossChannels(const float* srcData, float* dstData, const int width, const int height, const int channels, const float* powfParam) {
    const auto size     = width * height;
    const int threadNum = ((CPUBackend*)backend())->threadNumber();

    // calc pow
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        for (int c = (int)tId; c < channels; c += threadNum) {
            const float* inChannel = srcData + c * size;
            float* sqrtChannel = mSquare.host<float>() + c * size;
            int i            = 0;
#ifdef MNN_USE_NEON
            for (; i + 3 < size; i += 4) {
                float32x4_t v4 = vld1q_f32(inChannel + i);
                vst1q_f32(sqrtChannel + i, v4 * v4);
            }
#endif
            for (; i < size; i++) {
                float v        = inChannel[i];
                sqrtChannel[i] = v * v;
            }
        }
    }
    MNN_CONCURRENCY_END()

    // clear output
    memset(dstData, 0, size * channels * sizeof(float));
    auto outFactor = mAlpha / mLocalSize;

    // calc output
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        for (int c = (int)tId; c < channels; c += threadNum) {
            const float* inChannel = srcData + c * size;
            float* outChannel = dstData + c * size;
            auto startChanenl = std::max((int)c - mLocalSize / 2, 0);
            auto endChannel   = std::min((int)c + mLocalSize / 2, channels - 1);

            for (int lc = startChanenl; lc <= endChannel; lc++) {
                auto sqrtChannel = mSquare.host<float>() + lc * size;
                int i            = 0;
#ifdef MNN_USE_NEON
                for (; i + 3 < size; i += 4) {
                    vst1q_f32(outChannel + i, vld1q_f32(outChannel + i) + vld1q_f32(sqrtChannel + i));
                }
#endif
                for (; i < size; i++) {
                    outChannel[i] += sqrtChannel[i];
                }
            }
            for (int i = 0; i < size; i++) {
                outChannel[i] = 1.f + outFactor * outChannel[i];
            }
            powfWithContext(outChannel, outChannel, mBeta, size, powfParam);
            for (int i = 0; i < size; ++i) {
                outChannel[i] *= inChannel[i];
            }
        }
    }
    MNN_CONCURRENCY_END()
}

void CPULRN::executeWithInChannels(const float* srcData, float* dstData, const int width, const int height, const int channels, const float* powfParam) {
    const int size     = width * height;
    const int threadNum = ((CPUBackend*)backend())->threadNumber();

    // front = mLocalSize / 2 + 1 (extra dim in upper-left be used for two dim prefix square sum), behind = mLocalSize - front
    int halfLocalSize = mLocalSize / 2, padF = halfLocalSize + 1, padB = mLocalSize - padF;
    int padWidth = width + mLocalSize, padHeight = height + mLocalSize, padSize = padWidth * padHeight;

    // norm window offsets
    auto area    = mLocalSize * mLocalSize;

    // clear square and output
    memset(mSquare.host<float>(), 0, mSquare.size());
    memset(dstData, 0, size * channels * sizeof(float));

    // calc output
    auto outFactor = mAlpha / area;
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        const int mapping[7] = {-1, -padWidth, -padWidth - 1, halfLocalSize * padWidth + halfLocalSize, halfLocalSize * padWidth - halfLocalSize - 1,
            -(halfLocalSize + 1) * padWidth + halfLocalSize, -(halfLocalSize + 1) * padWidth - halfLocalSize - 1};
        for (int c = (int)tId; c < channels; c += threadNum) {
            const float* inChannel = srcData + c * size;
            float* outChannel  = dstData + c * size;
            float* sqrtChannel = mSquare.host<float>() + tId * padSize + padF * padWidth + padF;
            // We compute the two-dim prefix square sum
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    float v = inChannel[w];
                    *(sqrtChannel + w) = *(sqrtChannel + w + mapping[0]) + *(sqrtChannel + w + mapping[1]) - *(sqrtChannel + w + mapping[2]) + v * v;
                }
                sqrtChannel += width;
                inChannel += width;
                for (int pad = 0; pad < padB; ++pad) {
                    *(sqrtChannel + pad) = *(sqrtChannel + pad - 1);
                }
                sqrtChannel += padWidth - width;
            }
            for (int pad = 0, wEnd = width + padB; pad < padB; ++pad) {
                for (int w = 0; w < wEnd; ++w) {
                    *(sqrtChannel + w) = *(sqrtChannel + w - padWidth);
                }
                sqrtChannel += padWidth;
            }
            sqrtChannel = mSquare.host<float>() + tId * padSize + padF * padWidth + padF;
            // sum_of_region(h1, h2, w1, w2) = prefix_sum(h2, w2) - prefix_sum(h2, w1 - 1) - prefix_sum(h1 - 1, w2) + prefix_sum(h1 - 1, w1 - 1)
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float sum = *(sqrtChannel + w + mapping[3]) - *(sqrtChannel + w + mapping[4]) - *(sqrtChannel + w + mapping[5]) + *(sqrtChannel + w + mapping[6]);
                    outChannel[w] = 1.f + outFactor * sum;
                }
                outChannel += width;
                sqrtChannel += padWidth;
            }
            inChannel = srcData + c * size;
            outChannel  = dstData + c * size;
            powfWithContext(outChannel, outChannel, mBeta, size, powfParam);
            for (int i = 0; i < size; ++i) {
                outChannel[i] *= inChannel[i];
            }
        }
    }
    MNN_CONCURRENCY_END()

}

ErrorCode CPULRN::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    auto outputTensor = outputs[0];
    auto inputDataPtr = inputTensor->host<float>();
    auto outputDataPtr = outputTensor->host<float>();
    const int batch = outputTensor->batch();
    const int batchStride = outputTensor->stride(0);
    const int width = outputTensor->width();
    const int height = outputTensor->height();
    const int channel = outputTensor->channel();
    const int area = width * height;
    float powfParam[7];
    initPowfContext(mBeta, powfParam);
    float* tempData = mStorage.host<float>();
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto inputData  = inputDataPtr + batchIndex * batchStride;
        auto outputData = outputDataPtr + batchIndex * batchStride;
        // input transform
        MNNUnpackC4(outputData, inputData, area, channel);
        // clear square
        memset(mSquare.host<float>(), 0, mSquare.size());
        if (mRegionType == 0) {
            executeAcrossChannels(outputData, tempData, width, height, channel, powfParam);
        } else if (mRegionType == 1) {
            executeWithInChannels(outputData, tempData, width, height, channel, powfParam);
        } else {
            // not supported
        }
        // output transform
        MNNPackC4(outputData, tempData, area, channel);
    }

    return NO_ERROR;
}

class CPULRNCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto lrn = op->main_as_LRN();
        return new CPULRN(backend, lrn->regionType(), lrn->localSize(), lrn->alpha(), lrn->beta());
    }
};
REGISTER_CPU_OP_CREATOR(CPULRNCreator, OpType_LRN);

} // namespace MNN
