#include "HexagonRelu6.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "htp_command.h"

namespace MNN {

HexagonRelu6::HexagonRelu6(Backend* backend, float minValue, float maxValue)
    : HexagonExecution(backend), mMinValue(minValue), mMaxValue(maxValue) {
}

HexagonRelu6* HexagonRelu6::create(Backend* backend, const Op* op) {
    if (op == nullptr || op->type() != OpType_ReLU6) {
        return nullptr;
    }
    auto relu6 = op->main_as_Relu6();
    if (relu6 == nullptr || HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    return new HexagonRelu6(backend, relu6->minValue(), relu6->maxValue());
}

ErrorCode HexagonRelu6::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    auto output = outputs[0];
    if (input == nullptr || output == nullptr) {
        return INPUT_DATA_ERROR;
    }
    if (input->getType().code != halide_type_float || output->getType().code != halide_type_float) {
        return NOT_SUPPORT;
    }
    if (HexagonBackend::getBytes(input) != 2 || HexagonBackend::getBytes(output) != 2) {
        return NOT_SUPPORT;
    }
    auto hexBackend = static_cast<HexagonBackend*>(backend());
    const size_t inputSize = hexBackend->getElementSize(input);
    const size_t outputSize = hexBackend->getElementSize(output);
    if (inputSize != outputSize) {
        return NOT_SUPPORT;
    }

    auto srcDev = HexagonBackend::getDevicePtr(input);
    auto dstDev = HexagonBackend::getDevicePtr(output);
    if (srcDev.first <= 0 || dstDev.first <= 0) {
        return NOT_SUPPORT;
    }

    struct Relu6Param {
        int32_t size;
        int32_t bytes;
        float minValue;
        float maxValue;
    } __attribute__((packed));

    Relu6Param params;
    params.size = static_cast<int32_t>(outputSize);
    params.bytes = 2;
    params.minValue = mMinValue;
    params.maxValue = mMaxValue;

    std::vector<std::pair<int, int>> inputFds = {srcDev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};
    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_RELU6, &params, sizeof(params),
                     inputFds, outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonRelu6::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonRelu6(bn, mMinValue, mMaxValue);
    return true;
}

} // namespace MNN
