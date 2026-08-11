#include "HexagonConvolutionDepthwise.hpp"

#include "HexagonBackend.hpp"
#include "core/Macro.h"
#include "HexagonRuntime.hpp"
#include "htp_command.h"

namespace MNN {

HexagonConvolutionDepthwise::Resource::~Resource() {
    if (weight.first != nullptr) {
        allocator->free(weight);
    }
    if (bias.first != nullptr) {
        allocator->free(bias);
    }
}

HexagonConvolutionDepthwise::HexagonConvolutionDepthwise(Backend* backend, std::shared_ptr<Resource> res,
                                                         const Convolution2DCommon* common)
    : HexagonExecution(backend), mResource(std::move(res)), mCommon(common) {
    mKernelX = common->kernelX();
    mKernelY = common->kernelY();
    mStrideX = common->strideX();
    mStrideY = common->strideY();
    mDilateX = common->dilateX();
    mDilateY = common->dilateY();
    mPadX = common->padX();
    mPadY = common->padY();
    mPadMode = common->padMode();
    mRelu = common->relu() ? 1 : 0;
    mRelu6 = common->relu6() ? 1 : 0;
}

HexagonConvolutionDepthwise::HexagonConvolutionDepthwise(Backend* backend, std::shared_ptr<Resource> res)
    : HexagonExecution(backend), mResource(std::move(res)) {
}

ErrorCode HexagonConvolutionDepthwise::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                                  std::vector<HexagonCommand>& dst) {
    if (inputs.empty() || outputs.empty()) {
        mValid = false;
        return INPUT_DATA_ERROR;
    }

    mBytes = HexagonBackend::getBytes(outputs[0]);
    const auto runtime = static_cast<const HexagonRuntime*>(backend()->getRuntime());
    mPack = runtime->info().vectorSize;
    if (mBytes != 2) {
        mValid = false;
        return NOT_SUPPORT;
    }

    auto input = inputs[0];
    auto output = outputs[0];

    mBatch = input->batch();
    mInputHeight = input->height();
    mInputWidth = input->width();
    mOutputHeight = output->height();
    mOutputWidth = output->width();

    mChannel = input->channel();
    mChannelBlock = UP_DIV(mChannel, mPack);

    if (mPadMode == PadMode_SAME) {
        int padNeededWidth = (mOutputWidth - 1) * mStrideX + (mKernelX - 1) * mDilateX + 1 - mInputWidth;
        int padNeededHeight = (mOutputHeight - 1) * mStrideY + (mKernelY - 1) * mDilateY + 1 - mInputHeight;
        mPadX = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        mPadY = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    } else if (mPadMode == PadMode_VALID) {
        mPadX = 0;
        mPadY = 0;
    }

    auto inputDev = HexagonBackend::getDevicePtr(input);
    auto outputDev = HexagonBackend::getDevicePtr(output);
    auto weightDev = HexagonBackend::getDevicePtr(mResource->weight);
    auto biasDev = HexagonBackend::getDevicePtr(mResource->bias);

    int params[] = {mBatch, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth, mChannelBlock,
                    mKernelY, mKernelX, mStrideY, mStrideX, mPadY, mPadX, mDilateY, mDilateX, mRelu, mRelu6};

    std::vector<std::pair<int, int>> inputFds = {inputDev, weightDev, biasDev};
    std::vector<std::pair<int, int>> outputFds = {outputDev};

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_CONV_DEPTHWISE2D_FP16, params, sizeof(params),
                     inputFds,  outputFds,  inputs, outputs);

    return NO_ERROR;
}

bool HexagonConvolutionDepthwise::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto exe = new HexagonConvolutionDepthwise(bn, mResource);
    exe->mKernelX = mKernelX;
    exe->mKernelY = mKernelY;
    exe->mStrideX = mStrideX;
    exe->mStrideY = mStrideY;
    exe->mDilateX = mDilateX;
    exe->mDilateY = mDilateY;
    exe->mPadX = mPadX;
    exe->mPadY = mPadY;
    exe->mPadMode = mPadMode;
    exe->mRelu = mRelu;
    exe->mRelu6 = mRelu6;
    exe->mBatch = mBatch;
    exe->mInputHeight = mInputHeight;
    exe->mInputWidth = mInputWidth;
    exe->mOutputHeight = mOutputHeight;
    exe->mOutputWidth = mOutputWidth;
    exe->mChannel = mChannel;
    exe->mChannelBlock = mChannelBlock;
    exe->mPack = mPack;
    exe->mBytes = mBytes;
    *dst = exe;
    return true;
}

HexagonConvolutionDepthwise* HexagonConvolutionDepthwise::create(Backend* backend, const Op* op) {
    if (op->type() != OpType_ConvolutionDepthwise) {
        return nullptr;
    }
    const auto conv2d = op->main_as_Convolution2D();
    if (nullptr == conv2d || nullptr == conv2d->common()) {
        return nullptr;
    }
    const auto common = conv2d->common();

    if (common->group() != 1 && common->group() != common->outputCount()) {
        return nullptr;
    }

    if (common->inputCount() > 0 && common->outputCount() > 0 && common->inputCount() != common->outputCount()) {
        return nullptr;
    }

    if (common->inputCount() == 0 && common->outputCount() == 0) {
        return nullptr;
    }

    const float* originWeight = nullptr;
    int originWeightSize = 0;
    const float* originBias = nullptr;
    int originBiasSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;

    if (nullptr != conv2d->quanParameter()) {
        bool forceFloat = true;
        bool lowMemory = false;
        quanCommon = ConvolutionCommon::load(op, backend, forceFloat, lowMemory);
        if (quanCommon == nullptr) {
            return nullptr;
        }
        originWeight = quanCommon->weightFloat.get();
        originWeightSize = quanCommon->weightFloat.size();
    } else {
        if (conv2d->weight() == nullptr) {
            return nullptr;
        }
        originWeight = conv2d->weight()->data();
        originWeightSize = conv2d->weight()->size();
    }
    if (conv2d->bias() != nullptr) {
        originBias = conv2d->bias()->data();
        originBiasSize = conv2d->bias()->size();
    }

    int oc = common->outputCount();
    int ic = common->inputCount();
    if (oc == 0) {
        oc = originBiasSize;
    }
    if (ic == 0) {
        ic = oc;
    }
    if (ic != oc) {
        return nullptr;
    }
    if (originWeight == nullptr || originWeightSize <= 0) {
        return nullptr;
    }
    if (originWeightSize != oc * common->kernelX() * common->kernelY()) {
        return nullptr;
    }

    const auto runtime = static_cast<const HexagonRuntime*>(backend->getRuntime());
    int pack = runtime->info().vectorSize;

    int oc4 = UP_DIV(oc, pack);
    int kw = common->kernelX();
    int kh = common->kernelY();

    auto bufferAlloc = static_cast<HexagonBackend*>(backend)->getAllocator(2);
    std::shared_ptr<Resource> res(new Resource);
    res->allocator = bufferAlloc;
    res->weight = bufferAlloc->alloc(oc4 * kh * kw * pack * sizeof(int16_t));
    res->bias = bufferAlloc->alloc(oc4 * pack * sizeof(int16_t));

    auto weightPtr = reinterpret_cast<int16_t*>(HexagonBackend::getPtr(res->weight));
    auto biasPtr = reinterpret_cast<int16_t*>(HexagonBackend::getPtr(res->bias));

    ::memset(weightPtr, 0, oc4 * kh * kw * pack * sizeof(int16_t));
    ::memset(biasPtr, 0, oc4 * pack * sizeof(int16_t));

    std::vector<int16_t> biasHalf(oc);
    if (originBias != nullptr && originBiasSize > 0) {
        HexagonBackend::fp32ToFp16(originBias, biasHalf.data(), std::min(oc, originBiasSize));
    }
    for (int c = 0; c < oc; ++c) {
        biasPtr[c] = biasHalf[c];
    }

    std::vector<int16_t> weightHalf(originWeightSize);
    HexagonBackend::fp32ToFp16(originWeight, weightHalf.data(), originWeightSize);

    for (int c = 0; c < oc; ++c) {
        int c4Index = c / pack;
        int c4Inner = c % pack;
        for (int y = 0; y < kh; ++y) {
            for (int x = 0; x < kw; ++x) {
                int srcIndex = c * kh * kw + y * kw + x;
                int dstIndex = ((c4Index * kh + y) * kw + x) * pack + c4Inner;
                weightPtr[dstIndex] = weightHalf[srcIndex];
            }
        }
    }
    auto hexagonBackend = static_cast<HexagonBackend*>(backend);
    hexagonBackend->markHostInput(res->weight, oc4 * kh * kw * pack * (int)sizeof(int16_t));
    hexagonBackend->markHostInput(res->bias, oc4 * pack * (int)sizeof(int16_t));

    return new HexagonConvolutionDepthwise(backend, res, common);
}

} // namespace MNN
