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

ErrorCode CPUGridSample::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int numberThread = static_cast<CPUBackend*>(backend())->threadNumber();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto outputTensor = outputs[0];
    auto outH = outputTensor->buffer().dim[2].extent;
    auto outW = outputTensor->buffer().dim[3].extent;
    mTempCordBuffer.reset(Tensor::createDevice<uint8_t>({1, outH * outW * 2 * core->bytes}));
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
    auto inputPtr = inputTensor->host<uint8_t>();
    auto gridPtr = gridTensor->host<uint8_t>();
    auto outputPtr = outputTensor->host<uint8_t>();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto batches = inputTensor->buffer().dim[0].extent;
    auto channels = inputTensor->buffer().dim[1].extent;
    auto channelC4 = UP_DIV(channels, core->pack);
    auto inH = inputTensor->buffer().dim[2].extent;
    auto inW = inputTensor->buffer().dim[3].extent;
    auto outH = outputTensor->buffer().dim[2].extent;
    auto outW = outputTensor->buffer().dim[3].extent;
    auto threadCount = static_cast<CPUBackend*>(backend())->threadNumber();
    auto tileCount = outH;
    auto inOffset  = batches * inH * inW * core->pack;
    auto outOffset = batches * outH * outW * core->pack;
    auto cordPtr = mTempCordBuffer->host<uint8_t>();
    for (auto b = 0; b < batches; ++b) {
        auto _inputPtr = inputPtr + b * inH * inW * core->pack * core->bytes;
        auto _gridPtr = gridPtr + b * gridTensor->buffer().dim[0].stride * core->bytes;
        auto _outputPtr = outputPtr + b * outH * outW * core->pack * core->bytes;
        core->MNNGridSampleComputeCord((float *)cordPtr, (const float *)_gridPtr, inH, inW, outH, outW, gridTensor->buffer().dim[1].stride, mAlignCorners);
        // Compute cord
        MNN_CONCURRENCY_BEGIN(tId, threadCount) {
            for (int index=tId; index < tileCount; index += threadCount) {
                auto c = index / outH;
                auto h = index % outH;
                auto inputC = _inputPtr + c * inW * inH * batches * core->pack * core->bytes;
                auto outputC = _outputPtr + c * outW * outH * batches * core->pack * core->bytes;
                auto cordH = cordPtr + h * outW * 2 * core->bytes;
                auto outputH = outputC + h * outW * core->pack * core->bytes;
                core->MNNGridSampleInterp((float *)outputH, (const float *)inputC, (const float *)cordH, inH, inW, outW, channelC4, inOffset, outOffset, (mMode == SampleMode_NEAREST), (mPaddingMode == BorderMode_ZEROS));
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
        auto core = static_cast<CPUBackend*>(backend)->functions();
        if (core->MNNGridSampleInterp == nullptr) {
            MNN_ERROR("Don't has function for CPUGridSample\n");
            return nullptr;
        }
        return new CPUGridSample(backend, mode, paddingMode, alignCorners);
    }
};

REGISTER_CPU_OP_CREATOR(CPUGridSampleCreator, OpType_GridSample);


} // namespace MNN
