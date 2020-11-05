//
//  CPUNormalize.cpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUNormalize.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {
CPUNormalize::CPUNormalize(Backend* b, const MNN::Op* op) : MNN::Execution(b) {
    auto normalize = op->main_as_Normalize();
    mAcrossSpatial = normalize->acrossSpatial();
    mChannelShared = normalize->channelShared();

    mEps = normalize->eps();
    mScale.reset(normalize->scale()->size());
    ::memcpy(mScale.get(), normalize->scale()->data(), normalize->scale()->size() * sizeof(float));
}

ErrorCode CPUNormalize::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inputTensor  = inputs[0];
    int totalSize     = 1;
    auto outputTensor = outputs[0];

    MNN_ASSERT(1 == inputTensor->batch());
    MNN_ASSERT(1 == outputTensor->batch());

    // Across channel
    int inside  = inputTensor->width() * inputTensor->height();
    int axis    = inputTensor->channel();
    int outside = 1;

    // Across Spatial
    if (mAcrossSpatial) {
        inside  = 1;
        axis    = inputTensor->width() * inputTensor->height() * inputTensor->channel();
        outside = 1;
    }
    for (int i = 1; i < inputTensor->buffer().dimensions; ++i) {
        totalSize *= inputTensor->buffer().dim[i].extent;
    }
    mSourceStorage.buffer().dim[0].extent = 1;
    mSourceStorage.buffer().dim[1].extent = totalSize;
    mSourceStorage.buffer().dim[2].extent = 1;
    mSourceStorage.buffer().dim[3].extent = 1;

    mSummer.buffer().dim[0].extent = 1;
    mSummer.buffer().dim[1].extent = inside * outside;
    mSummer.buffer().dim[2].extent = 1;
    mSummer.buffer().dim[3].extent = 1;

    backend()->onAcquireBuffer(&mSummer, Backend::DYNAMIC);
    backend()->onAcquireBuffer(&mSourceStorage, Backend::DYNAMIC);

    backend()->onReleaseBuffer(&mSummer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mSourceStorage, Backend::DYNAMIC);

    return NO_ERROR;
}

static void _normalize(const float* input, float* summer, float* output, int inside, int outside, int axis, float eps) {
    // Compute summer
    ::memset(summer, 0, inside * outside * sizeof(float));
    for (int z = 0; z < outside; ++z) {
        float* summerZ      = summer + inside * z;
        const float* inputZ = input + axis * inside * z;
        for (int y = 0; y < axis; ++y) {
            const float* inputY = inputZ + y * inside;
            for (int x = 0; x < inside; ++x) {
                summerZ[x] += inputY[x] * inputY[x];
            }
        }
    }

    // Compute scale
    for (int i = 0; i < inside * outside; ++i) {
        summer[i] = 1.0f / sqrtf(summer[i] + eps);
    }

    // Scale
    for (int z = 0; z < outside; ++z) {
        float* summerZ      = summer + inside * z;
        const float* inputZ = input + axis * inside * z;
        float* outputZ      = output + axis * inside * z;
        for (int y = 0; y < axis; ++y) {
            const float* inputY = inputZ + y * inside;
            float* outputY      = outputZ + y * inside;
            for (int x = 0; x < inside; ++x) {
                outputY[x] = inputY[x] * summerZ[x];
            }
        }
    }
}

static void _scaleChannel(const float* input, float* output, float* scale, int area, int channel) {
    for (int z = 0; z < channel; ++z) {
        float* outputZ      = output + z * area;
        const float* inputZ = input + z * area;
        float s             = scale[z];
        for (int i = 0; i < area; ++i) {
            outputZ[i] = inputZ[i] * s;
        }
    }
}

static void _scaleSingleValue(const float* input, float* output, float* scale, int area, int channel) {
    float s = scale[0];
    for (int z = 0; z < channel; ++z) {
        float* outputZ      = output + z * area;
        const float* inputZ = input + z * area;
        for (int i = 0; i < area; ++i) {
            outputZ[i] = inputZ[i] * s;
        }
    }
}
ErrorCode CPUNormalize::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(!mAcrossSpatial);
    MNN_ASSERT(!mChannelShared);
    auto inputTensor  = inputs[0];
    auto outputTensor = outputs[0];

    MNN_ASSERT(1 == inputTensor->batch());
    MNN_ASSERT(1 == outputTensor->batch());

    // Across channel
    int inside  = inputTensor->width() * inputTensor->height();
    int axis    = inputTensor->channel();
    int outside = 1;

    // Across Spatial
    if (mAcrossSpatial) {
        inside  = 1;
        axis    = inputTensor->width() * inputTensor->height() * inputTensor->channel();
        outside = 1;
    }

    int area = inputTensor->width() * inputTensor->height();

    const float* inputData = inputTensor->host<float>();
    MNNUnpackC4(mSourceStorage.host<float>(), inputData, area, inputTensor->channel());

    float* outputData = outputTensor->host<float>();
    _normalize(mSourceStorage.host<float>(), mSummer.host<float>(), mSourceStorage.host<float>(), inside, outside, axis,
               mEps);

    if (mChannelShared) {
        _scaleSingleValue(mSourceStorage.host<float>(), mSourceStorage.host<float>(), mScale.get(), area,
                          inputTensor->channel());
    } else {
        _scaleChannel(mSourceStorage.host<float>(), mSourceStorage.host<float>(), mScale.get(), area,
                      inputTensor->channel());
    }
    MNNPackC4(outputData, mSourceStorage.host<float>(), area, outputTensor->channel());

    return NO_ERROR;
}
class CPUNormalizeCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUNormalize(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUNormalizeCreator, OpType_Normalize);
} // namespace MNN
