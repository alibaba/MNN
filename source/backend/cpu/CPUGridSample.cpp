//
//  CPUGridSample.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUGridSample.hpp"
#include <math.h>
#include <string.h>
#include "core/Concurrency.h"
#include <algorithm>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include <math/Vec.hpp>
using Vec4 = MNN::Math::Vec<float, 4>;

namespace MNN {
CPUGridSample::CPUGridSample(Backend *b, SampleMode mode, BorderMode paddingMode, bool alignCorners)
        : Execution(b) {
    mMode = mode;
    mPaddingMode = paddingMode;
    mAlignCorners = alignCorners;
}

static float getPosition(float x, int range, bool alignCorners) {
    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    return ((1 + x) * (range - a) - b) / 2.0f;
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

static Vec4 sample(int h, int w, const float *buffer, int height, int width, BorderMode padMode) {
    if (h < 0 || h >= height || w < 0 || w >= width) {
        if(padMode == BorderMode_ZEROS) {
            return 0.0f;
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = CLAMP(h, 0, height - 1);
        w = CLAMP(w, 0, width - 1);
    }

    return Vec4::load(buffer + h * width * 4 + w * 4);
}

static Vec4 interpolate(float h, float w, const float *buffer, int height, int width, SampleMode mode, BorderMode padMode) {
    if (mode == SampleMode_NEAREST) {
        int nh = ::floor(h+0.5f);
        int nw = ::floor(w+0.5f);
        return sample(nh, nw, buffer, height, width, padMode);
    }
    // mode == GridSampleMode_BILINEAR
    int w0_h = ::floor(h);
    int w0_w = ::floor(w);
    int w1_h = ::ceil(h);
    int w1_w = ::ceil(w);
    auto oneV = Vec4(1.0f);

    Vec4 i00 = sample(w0_h, w0_w, buffer, height, width, padMode);
    Vec4 i01 = sample(w0_h, w1_w, buffer, height, width, padMode);
    Vec4 i10 = sample(w1_h, w0_w, buffer, height, width, padMode);
    Vec4 i11 = sample(w1_h, w1_w, buffer, height, width, padMode);
    auto f0 = Vec4((float)w1_w - w);
    auto f1 = oneV - f0;
    auto h0 = Vec4((float)w1_h - h);
    auto h1 = oneV - h0;

    Vec4 i0 = i00 * f0 + i01 * f1;
    Vec4 i1 = i10 * f0 + i11 * f1;

    return i0 * h0 + i1 * h1;
}


ErrorCode CPUGridSample::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int numberThread = static_cast<CPUBackend*>(backend())->threadNumber();
    auto outputTensor = outputs[0];
    auto outH = outputTensor->buffer().dim[2].extent;
    auto outW = outputTensor->buffer().dim[3].extent;
    mTempCordBuffer.reset(Tensor::createDevice<float>({1, outH * outW * 2}));
    auto res = backend()->onAcquireBuffer(mTempCordBuffer.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempCordBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUGridSample::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    auto gridTensor = inputs[1];
    auto outputTensor = outputs[0];

    float *inputPtr = inputTensor->host<float>();
    float *gridPtr = gridTensor->host<float>();
    auto *outputPtr = outputTensor->host<float>();
    
    auto batches = inputTensor->buffer().dim[0].extent;
    auto channels = inputTensor->buffer().dim[1].extent;
    auto channelC4 = UP_DIV(channels, 4);
    auto inH = inputTensor->buffer().dim[2].extent;
    auto inW = inputTensor->buffer().dim[3].extent;
    auto outH = outputTensor->buffer().dim[2].extent;
    auto outW = outputTensor->buffer().dim[3].extent;
    auto cordPtr = mTempCordBuffer->host<float>();
    auto threadCount = static_cast<CPUBackend*>(backend())->threadNumber();
    auto tileCount = channelC4 * outH;
    for (auto b = 0; b < batches; ++b) {
        const float *_inputPtr = inputPtr + b * inputTensor->buffer().dim[0].stride;
        const float *_gridPtr = gridPtr + b * gridTensor->buffer().dim[0].stride;
        float *_outputPtr = outputPtr + b * outputTensor->buffer().dim[0].stride;
        // Compute cord
        for (auto h = 0; h < outH; ++h) {
            auto __gridPtr = _gridPtr + h * gridTensor->buffer().dim[1].stride;
            auto cordH = cordPtr + h * outW * 2;
            for (auto w = 0; w < outW; ++w) {
                auto x = getPosition(__gridPtr[2 * w + 0], inW, mAlignCorners);
                auto y = getPosition(__gridPtr[2 * w + 1], inH, mAlignCorners);
                cordH[2 * w + 0] = x;
                cordH[2 * w + 1] = y;
            }
        }
        MNN_CONCURRENCY_BEGIN(tId, threadCount) {
            for (int index=tId; index < tileCount; index += threadCount) {
                auto c = index / outH;
                auto h = index % outH;
                auto inpC = _inputPtr + c * inW * inH * 4;
                auto outC = _outputPtr + c * outW * outH * 4;
                auto cordH = cordPtr + h * outW * 2;
                auto outH = outC + h * outW * 4;
                for (auto w = 0; w < outW; ++w) {
                    auto x = cordH[2 * w + 0];
                    auto y = cordH[2 * w + 1];
                    Vec4::save(outH + 4 * w, interpolate(y, x, inpC, inH, inW, mMode, mPaddingMode));
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUGridSampleCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto gridSampleParam = op->main_as_GridSample();
        auto mode = gridSampleParam->mode();
        auto paddingMode = gridSampleParam->paddingMode();
        auto alignCorners = gridSampleParam->alignCorners();
        return new CPUGridSample(backend, mode, paddingMode, alignCorners);
    }
};

REGISTER_CPU_OP_CREATOR(CPUGridSampleCreator, OpType_GridSample);


} // namespace MNN
