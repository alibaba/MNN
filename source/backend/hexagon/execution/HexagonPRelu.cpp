#include "HexagonPRelu.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "htp_command.h"
#include <climits>
#include <cstring>

namespace MNN {

HexagonPRelu::HexagonPRelu(Backend* backend, const Op* op, int pack) : HexagonExecution(backend), mPack(pack) {
    auto prelu = op->main_as_PRelu();
    if (prelu == nullptr || prelu->slope() == nullptr || prelu->slopeCount() <= 0) {
        return;
    }
    mSlopeCount = prelu->slopeCount();
    const int slopePackCount = mSlopeCount == 1 ? 1 : UP_DIV(mSlopeCount, mPack) * mPack;
    mSlope.reset(Tensor::createDevice<int16_t>({slopePackCount}));
    if (!backend->onAcquireBuffer(mSlope.get(), Backend::STATIC)) {
        mSlope = nullptr;
        return;
    }
    ::memset(HexagonBackend::getPtr(mSlope.get()), 0, mSlope->size());
    HexagonBackend::fp32ToFp16(prelu->slope()->data(), reinterpret_cast<int16_t*>(HexagonBackend::getPtr(mSlope.get())),
                               mSlopeCount);
    static_cast<HexagonBackend*>(backend)->markHostInput(mSlope.get());
}

HexagonPRelu* HexagonPRelu::create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs) {
    if (op == nullptr || op->type() != OpType_PReLU || inputs.size() != 1 || outputs.size() != 1 ||
        HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    auto runtime = static_cast<const HexagonRuntime*>(backend->getRuntime());
    int pack = runtime->info().vectorSize;
    if (pack <= 0) {
        pack = 4;
    }
    auto prelu = op->main_as_PRelu();
    if (prelu == nullptr || prelu->slope() == nullptr || prelu->slopeCount() <= 0) {
        return nullptr;
    }
    if (prelu->slopeCount() != 1) {
        auto output = outputs[0];
        auto des = TensorUtils::getDescribe(output);
        if (output == nullptr || output->dimensions() < 2 ||
            des->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
            prelu->slopeCount() != output->channel()) {
            return nullptr;
        }
    }
    return new HexagonPRelu(backend, op, pack);
}

ErrorCode HexagonPRelu::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1 || mSlope == nullptr || mSlopeCount <= 0) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    auto output = outputs[0];
    if (input == nullptr || output == nullptr ||
        input->getType().code != halide_type_float || output->getType().code != halide_type_float ||
        HexagonBackend::getBytes(input) != 2 || HexagonBackend::getBytes(output) != 2) {
        return NOT_SUPPORT;
    }

    auto hexBackend = static_cast<HexagonBackend*>(backend());
    const size_t inputSize = hexBackend->getElementSize(input);
    const size_t outputSize = hexBackend->getElementSize(output);
    if (inputSize != outputSize || outputSize > INT_MAX) {
        return NOT_SUPPORT;
    }
    int plane = 1;
    int channel = 1;
    if (output->dimensions() >= 2) {
        channel = output->channel();
        plane = output->batch();
        for (int i = 2; i < output->dimensions(); ++i) {
            plane *= output->length(i);
        }
    }

    auto srcDev = HexagonBackend::getDevicePtr(input);
    auto slopeDev = HexagonBackend::getDevicePtr(mSlope.get());
    auto dstDev = HexagonBackend::getDevicePtr(output);
    if (srcDev.first <= 0 || slopeDev.first <= 0 || dstDev.first <= 0) {
        return NOT_SUPPORT;
    }

    struct PReluParam {
        int32_t size;
        int32_t bytes;
        int32_t plane;
        int32_t channel;
        int32_t slopeCount;
        int32_t pack;
    } __attribute__((packed));

    PReluParam params;
    ::memset(&params, 0, sizeof(params));
    params.size = static_cast<int32_t>(outputSize);
    params.bytes = 2;
    params.plane = plane;
    params.channel = channel;
    params.slopeCount = mSlopeCount;
    params.pack = mPack;

    std::vector<std::pair<int, int>> inputFds = {srcDev, slopeDev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};
    dst.emplace_back();
    dst.back().build(hexBackend, DSP_OP_PRELU, &params, sizeof(params), inputFds, outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonPRelu::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (mSlope == nullptr || mSlopeCount <= 0 || bn == nullptr || op == nullptr || op->type() != OpType_PReLU) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    auto runtime = static_cast<const HexagonRuntime*>(bn->getRuntime());
    int pack = runtime->info().vectorSize;
    if (pack <= 0) {
        pack = 4;
    }
    auto exe = new HexagonPRelu(bn, op, pack);
    if (exe->mSlope == nullptr || exe->mSlopeCount <= 0) {
        delete exe;
        return false;
    }
    *dst = exe;
    return true;
}

} // namespace MNN
