//
//  CPUSoftmax.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSoftmax.hpp"
#include <math.h>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

static int _softmax1(const float *srcData, float *dstData, int outside, int channel) {
    const float *srcY = srcData;
    float *dstY       = dstData;
    for (int y = 0; y < outside; ++y, srcY += channel, dstY += channel) {
        // max
        float maxValue = srcY[0];
        {
            int c = 1;
#ifdef MNN_USE_NEON
#if !(defined(__ARM_FEATURE_FMA) && defined(__aarch64__))
#define vmaxvq_f32(v)                 \
    ({                                \
        float __m = v[0];             \
        for (int i = 1; i < 4; i++) { \
            if (v[i] > __m)           \
                __m = v[i];           \
        }                             \
        __m;                          \
    })
#endif
            if (c + 3 < channel) {
                float32x4_t maxx4 = vld1q_f32(srcY + c);
                c += 4;
                for (; c + 3 < channel; c += 4) {
                    maxx4 = vmaxq_f32(maxx4, vld1q_f32(srcY + c));
                }
                float value = vmaxvq_f32(maxx4);
                if (value > maxValue)
                    maxValue = value;
            }
#endif
            for (; c < channel; ++c) {
                float value = srcY[c];
                if (value > maxValue)
                    maxValue = value;
            }
        }

        // sum
        float sumValue = 0;
#pragma clang loop vectorize(enable)
        for (int c = 0; c < channel; ++c) {
            dstY[c] = expf(srcY[c] - maxValue);
            sumValue += dstY[c];
        }

        // div
        {
            int c = 0;
#ifdef MNN_USE_NEON
            float div = 1.f / sumValue;
            for (; c + 3 < channel; c += 4) {
                vst1q_f32(dstY + c, vmulq_n_f32(vld1q_f32(dstY + c), div));
            }
#endif
            for (; c < channel; ++c) {
                dstY[c] /= sumValue;
            }
        }
    }

    return 0;
}
static int _softmaxCommon(const float *srcData, float *dstData, int inside, int outside, int channel) {
    if (inside == 1)
        return _softmax1(srcData, dstData, outside, channel);

    // malloc temp memory
    float *maxValue = (float *)malloc(sizeof(float) * inside);
    float *sumValue = (float *)malloc(sizeof(float) * inside);

    const float *srcY = srcData;
    float *dstY       = dstData;
    const int stepY   = inside * channel;
    for (int y = 0; y < outside; ++y, srcY += stepY, dstY += stepY) {
        memcpy(maxValue, srcY, sizeof(float) * inside);
        const float *src = srcY + inside;
        for (int c = 1; c < channel; ++c, src += inside) {
            for (int x = 0; x < inside; ++x) {
                if (src[x] > maxValue[x])
                    maxValue[x] = src[x];
            }
        }

        memset(sumValue, 0, sizeof(float) * inside);
        src        = srcY;
        float *dst = dstY;
        for (int c = 0; c < channel; ++c, src += inside, dst += inside) {
            for (int x = 0; x < inside; ++x) {
                dst[x] = expf(src[x] - maxValue[x]);
                sumValue[x] += dst[x];
            }
        }

        dst = dstY;
        for (int c = 0; c < channel; ++c, dst += inside) {
            for (int x = 0; x < inside; ++x) {
                dst[x] /= sumValue[x];
            }
        }
    }

    free(maxValue);
    free(sumValue);
    return 0;
}

ErrorCode CPUSoftmax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input    = inputs[0];
    int totalSize = 1;
    for (int i = 0; i < input->buffer().dimensions; ++i) {
        totalSize *= input->buffer().dim[i].extent;
    }
    mStorage.buffer().dim[0].extent = 1;
    mStorage.buffer().dim[1].extent = totalSize;
    mStorage.buffer().dim[1].flags  = 0;
    mStorage.buffer().dimensions    = 2;
    mStorage.buffer().type          = input->getType();
    backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUSoftmax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    auto inputTensor  = inputs[0];
    auto outputTensor = outputs[0];
    // TODO
    const auto dims = inputTensor->buffer().dimensions;
    MNN_ASSERT(dims >= 2);
    if (-1 == mAxis) {
        mAxis = dims - 1;
    }
    float *tempData = mStorage.host<float>();

    int areaInput = 1;
    for (int i = 2; i < inputTensor->buffer().dimensions; ++i) {
        areaInput *= inputTensor->buffer().dim[i].extent;
    }
    int inside  = 1;
    int outside = 1;
    int channel = 1;
    for (int i = 1; i < mAxis; ++i) {
        outside *= inputTensor->buffer().dim[i].extent;
    }
    channel = inputTensor->buffer().dim[mAxis].extent;
    for (int i = mAxis + 1; i < inputTensor->buffer().dimensions; ++i) {
        inside *= inputTensor->buffer().dim[i].extent;
    }

    int batchSize = outputTensor->size() / sizeof(float) / outputTensor->batch();
    for (int batchIndex = 0; batchIndex < outputTensor->batch(); ++batchIndex) {
        auto inputData  = inputTensor->host<float>() + batchIndex * batchSize;
        auto outputData = outputTensor->host<float>() + batchIndex * batchSize;
        if (1 == areaInput) {
            _softmaxCommon(inputData, outputData, inside, outside, channel);
            continue;
        }
        MNNUnpackC4(outputData, inputData, areaInput, inputTensor->channel());
        _softmaxCommon(outputData, tempData, inside, outside, channel);
        MNNPackC4(outputData, tempData, areaInput, outputTensor->channel());
    }

    return NO_ERROR;
}

CPUSoftmax::CPUSoftmax(Backend *b, int axis) : MNN::Execution(b), mAxis(axis), mStorage(2) {
    // nothing to do
}

class CPUSoftmaxCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUSoftmax(backend, op->main_as_Axis()->axis());
    }
};

REGISTER_CPU_OP_CREATOR(CPUSoftmaxCreator, OpType_Softmax);

} // namespace MNN
