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
    int outD, outH, outW;
    if (outputTensor->dimensions() == 4) {
        outH = outputTensor->buffer().dim[2].extent;
        outW = outputTensor->buffer().dim[3].extent;
        mTempCordBuffer.reset(Tensor::createDevice<uint8_t>({1, outH * outW * 2 * core->bytes}));
    } else {
        outD = outputTensor->buffer().dim[2].extent;
        outH = outputTensor->buffer().dim[3].extent;
        outW = outputTensor->buffer().dim[4].extent;
        mTempCordBuffer.reset(Tensor::createDevice<uint8_t>({1, outD * outH * outW * 3 * core->bytes}));
    }
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
    if (outputs[0]->dimensions() == 4) {
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
    } else {
        auto inD = inputTensor->buffer().dim[2].extent;
        auto inH = inputTensor->buffer().dim[3].extent;
        auto inW = inputTensor->buffer().dim[4].extent;
        auto outD = outputTensor->buffer().dim[2].extent;
        auto outH = outputTensor->buffer().dim[3].extent;
        auto outW = outputTensor->buffer().dim[4].extent;
        auto threadCount = static_cast<CPUBackend*>(backend())->threadNumber();
        auto tileCount = outD;
        auto inOffset  = batches * inH * inW * core->pack;
        auto outOffset = batches * outH * outW * core->pack;
        auto cordPtr = mTempCordBuffer->host<uint8_t>();
        for (auto b = 0; b < batches; ++b) {
            auto _inputPtr = inputPtr + b * inD * inH * inW * core->pack * core->bytes;
            auto _gridPtr = gridPtr + b * gridTensor->buffer().dim[0].stride * core->bytes;
            auto _outputPtr = outputPtr + b * outD * outH * outW * core->pack * core->bytes;
            core->MNNGridSampleComputeCord3D((float *)cordPtr, (const float *)_gridPtr, inD, inH, inW, outD, outH, outW, gridTensor->buffer().dim[1].stride, gridTensor->buffer().dim[2].stride, mAlignCorners);
            // Compute cord
            MNN_CONCURRENCY_BEGIN(tId, threadCount) {
                for (int index=tId; index < tileCount; index += threadCount) {
                    auto c = index / outD;
                    auto d = index % outD;
                    auto inputC = _inputPtr + c * inD * inW * inH * batches * core->pack * core->bytes;
                    auto outputC = _outputPtr + c * outD * outW * outH * batches * core->pack * core->bytes;
                    auto cordD = cordPtr + d * outH * outW * 3 * core->bytes;
                    auto outputD = outputC + d * outH * outW * core->pack * core->bytes;
                    for (int h = 0; h < outH; h++) {
                        auto cordH = cordD + h * outW * 3 * core->bytes;
                        auto outputH = outputD + h * outW * core->pack * core->bytes;
                        core->MNNGridSampleInterp3D((float *)outputH, (const float *)inputC, (const float *)cordH, inD, inH, inW, outW, channelC4, inOffset, outOffset, (mMode == SampleMode_NEAREST), (mPaddingMode == BorderMode_ZEROS));
                    }
                }
            }
            MNN_CONCURRENCY_END();
        }
    }
    return NO_ERROR;
}

class CPUGridSampleGrad : public CPUGridSample {
public:
    CPUGridSampleGrad(Backend *b, SampleMode mode, BorderMode paddingMode, bool alignCorners) : CPUGridSample(b, mode, paddingMode, alignCorners) {
        // Do nothing
    }

    virtual ~CPUGridSampleGrad() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        int numberThread = static_cast<CPUBackend*>(backend())->threadNumber();
        auto core = static_cast<CPUBackend*>(backend())->functions();
        auto outputTensor = inputs[0];
        int outD, outH, outW;
        if (outputTensor->dimensions() == 4) {
            outH = outputTensor->buffer().dim[2].extent;
            outW = outputTensor->buffer().dim[3].extent;
            mTempCordBuffer.reset(Tensor::createDevice<uint8_t>({1, outH * outW * 2 * core->bytes}));
        } else {
            outD = outputTensor->buffer().dim[2].extent;
            outH = outputTensor->buffer().dim[3].extent;
            outW = outputTensor->buffer().dim[4].extent;
            mTempCordBuffer.reset(Tensor::createDevice<uint8_t>({1, outD * outH * outW * 3 * core->bytes}));
        }
        auto res = backend()->onAcquireBuffer(mTempCordBuffer.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        backend()->onReleaseBuffer(mTempCordBuffer.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto inputTensor = outputs[0];
        ::memset(inputTensor->host<uint8_t>(), 0, static_cast<CPUBackend*>(backend())->getTensorSize(inputTensor, false) * static_cast<CPUBackend*>(backend())->functions()->bytes);
        auto gridTensor = inputs[1];
        auto outputTensor = inputs[0];
        auto inputPtr = inputTensor->host<uint8_t>();
        auto gridPtr = gridTensor->host<uint8_t>();
        auto outputPtr = outputTensor->host<uint8_t>();
        auto core = static_cast<CPUBackend*>(backend())->functions();
        auto batches = inputTensor->buffer().dim[0].extent;
        auto channels = inputTensor->buffer().dim[1].extent;
        auto channelC4 = UP_DIV(channels, core->pack);
        if (outputTensor->dimensions() != 4) {
            return NOT_SUPPORT;
        }
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
            for (int index=0; index < tileCount; index++) {
                auto c = index / outH;
                auto h = index % outH;
                auto inputC = _inputPtr + c * inW * inH * batches * core->pack * core->bytes;
                auto outputC = _outputPtr + c * outW * outH * batches * core->pack * core->bytes;
                auto cordH = cordPtr + h * outW * 2 * core->bytes;
                auto outputH = outputC + h * outW * core->pack * core->bytes;
                core->MNNGridSampleInterpGrad((float *)outputH, (float *)inputC, (const float *)cordH, inH, inW, outW, channelC4, inOffset, outOffset, (mMode == SampleMode_NEAREST), (mPaddingMode == BorderMode_ZEROS));
            }
        }

        return NO_ERROR;
    }
};

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
        if (gridSampleParam->backward()) {
            return new CPUGridSampleGrad(backend, mode, paddingMode, alignCorners);;
        }
        if (outputs[0]->dimensions() > 4 && core->MNNGridSampleInterp3D == nullptr) {
            MNN_ERROR("Don't support gridsampler grad for pack = %d, float bytes = %d\n", core->pack, core->bytes);
            return nullptr;
        }
        return new CPUGridSample(backend, mode, paddingMode, alignCorners);
    }
};

REGISTER_CPU_OP_CREATOR(CPUGridSampleCreator, OpType_GridSample);


} // namespace MNN
