#include "HexagonScale.hpp"
#include "HexagonRuntime.hpp"
#include "HexagonBackend.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "htp_command.h"

namespace MNN {

HexagonScale::HexagonScale(const Op* op, Backend* bn) : HexagonExecution(bn) {
    auto scale = op->main_as_Scale();
    int outputCount = scale->scaleData()->size();

    auto pack = static_cast<const HexagonRuntime*>(bn->getRuntime())->info().vectorSize;
    int cPack = UP_DIV(outputCount, pack);

    // Scale and Bias, 2 arrays, each size: cPack * pack * sizeof(int16_t)
    mScaleBias.reset(Tensor::createDevice<int16_t>({2, cPack * pack}));
    auto res = bn->onAcquireBuffer(mScaleBias.get(), Backend::STATIC);
    if (!res) {
        MNN_ERROR("Error for alloc buffer for HexagonScale\n");
        mScaleBias = nullptr;
        return;
    }

    ::memset(HexagonBackend::getPtr(mScaleBias.get()), 0, mScaleBias->size());

    // CPU to fp16
    HexagonBackend::fp32ToFp16(scale->scaleData()->data(), (int16_t*)(HexagonBackend::getPtr(mScaleBias.get())), outputCount);

    if (nullptr != scale->biasData() && nullptr != scale->biasData()->data()) {
        auto biasPtr = mScaleBias->host<int16_t>() + mScaleBias->length(1);
        HexagonBackend::fp32ToFp16(scale->biasData()->data(), biasPtr, outputCount);
        mHasBias = true;
    }
    static_cast<HexagonBackend*>(bn)->markHostInput(mScaleBias.get());

}

HexagonScale::HexagonScale(Backend* bn, const std::shared_ptr<Tensor>& scaleBias, bool hasBias)
    : HexagonExecution(bn), mScaleBias(scaleBias), mHasBias(hasBias) {
}

HexagonScale::~HexagonScale() {
    // Do nothing
}

bool HexagonScale::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (mScaleBias == nullptr) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new HexagonScale(bn, mScaleBias, mHasBias);
    return true;
}

ErrorCode HexagonScale::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   std::vector<HexagonCommand>& dst) {
    auto input = inputs[0];
    auto output = outputs[0];

    auto srcDev = HexagonBackend::getDevicePtr(input);
    auto dstDev = HexagonBackend::getDevicePtr(output);
    auto scaleDev = HexagonBackend::getDevicePtr(mScaleBias.get());

    auto pack = static_cast<const HexagonRuntime*>(backend()->getRuntime())->info().vectorSize;
    int plane = input->batch() * input->height() * input->width();
    int cPack = UP_DIV(input->channel(), pack);

    int scaleBiasOffset = scaleDev.second;
    int params[] = {plane, cPack, mHasBias ? 1 : 0};
    std::vector<std::pair<int, int>> inputFds = {srcDev, {scaleDev.first, scaleBiasOffset}};
    std::vector<std::pair<int, int>> outputFds = {dstDev};

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_SCALE, params, sizeof(params),
                     inputFds,  outputFds,  inputs, outputs);

    return NO_ERROR;
}

HexagonScale* HexagonScale::create(Backend* backend, const Op* op) {
    if (op->type() != OpType_Scale) {
        return nullptr;
    }
    return new HexagonScale(op, backend);
}

} // namespace MNN
