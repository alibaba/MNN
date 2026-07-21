#include "HexagonPooling.hpp"

#include "HexagonBackend.hpp"
#include "core/Macro.h"
#include "HexagonRuntime.hpp"
#include "htp_command.h"

namespace MNN {

HexagonPooling::HexagonPooling(Backend* backend, const Pool* parameter) : HexagonExecution(backend), mParameter(parameter) {
}

ErrorCode HexagonPooling::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                     std::vector<HexagonCommand>& dst) {
    if (inputs.empty() || outputs.empty()) {
        mValid = false;
        return INPUT_DATA_ERROR;
    }
    auto input = inputs[0];
    auto output = outputs[0];

    mBytes = HexagonBackend::getBytes(output);
    const auto runtime = static_cast<const HexagonRuntime*>(backend()->getRuntime());
    mPack = runtime->info().vectorSize;
    if (mBytes != 2) {
        mValid = false;
        return NOT_SUPPORT;
    }

    auto layer = mParameter;
    mStrideX = layer->strideX();
    mStrideY = layer->strideY();
    mPadX = layer->padX();
    mPadY = layer->padY();

    mKernelX = layer->kernelX();
    mKernelY = layer->kernelY();
    if (layer->isGlobal()) {
        mKernelX = input->width();
        mKernelY = input->height();
        mStrideX = input->width();
        mStrideY = input->height();
        mPadX = 0;
        mPadY = 0;
    }

    auto originPadType = layer->padType();
    if (originPadType == PoolPadType_SAME) {
        int padNeededWidth = (output->width() - 1) * mStrideX + mKernelX - input->width();
        int padNeededHeight = (output->height() - 1) * mStrideY + mKernelY - input->height();
        mPadX = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        mPadY = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    } else if (originPadType == PoolPadType_VALID) {
        mPadX = 0;
        mPadY = 0;
    }
    mPadType = originPadType;
    if (layer->pads() != nullptr && mPadType == PoolPadType_CAFFE) {
        mPadType = PoolPadType_VALID;
    }

    mCountType = layer->countType();
    mPoolType = layer->type();

    mBatch = input->batch();
    mInputHeight = input->height();
    mInputWidth = input->width();
    mOutputHeight = output->height();
    mOutputWidth = output->width();
    mChannelBlock = UP_DIV(input->channel(), mPack);

    auto inputDev = HexagonBackend::getDevicePtr(input);
    auto outputDev = HexagonBackend::getDevicePtr(output);

    int params[] = {mBatch, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth, mChannelBlock,
                    mKernelY, mKernelX, mStrideY, mStrideX, mPadY, mPadX, mPadType, mCountType, mPoolType};
    std::vector<std::pair<int, int>> inputFds = {inputDev};
    std::vector<std::pair<int, int>> outputFds = {outputDev};

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_POOL2D_FP16, params, sizeof(params),
                     inputFds,  outputFds,  inputs, outputs);

    return NO_ERROR;
}

bool HexagonPooling::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto exe = new HexagonPooling(bn, mParameter);
    exe->mKernelX = mKernelX;
    exe->mKernelY = mKernelY;
    exe->mStrideX = mStrideX;
    exe->mStrideY = mStrideY;
    exe->mPadX = mPadX;
    exe->mPadY = mPadY;
    exe->mPadType = mPadType;
    exe->mCountType = mCountType;
    exe->mPoolType = mPoolType;
    exe->mBatch = mBatch;
    exe->mInputHeight = mInputHeight;
    exe->mInputWidth = mInputWidth;
    exe->mOutputHeight = mOutputHeight;
    exe->mOutputWidth = mOutputWidth;
    exe->mChannelBlock = mChannelBlock;
    exe->mPack = mPack;
    exe->mBytes = mBytes;
    *dst = exe;
    return true;
}

HexagonPooling* HexagonPooling::create(Backend* backend, const Op* op) {
    if (op->type() != OpType_Pooling) {
        return nullptr;
    }
    auto pool = op->main_as_Pool();
    if (nullptr == pool) {
        return nullptr;
    }
    // MaxPool with indices output is not supported currently
    return new HexagonPooling(backend, pool);
}

} // namespace MNN
