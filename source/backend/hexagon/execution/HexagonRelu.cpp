#include "HexagonRelu.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "htp_command.h"
#include <climits>
#include <cstring>

namespace MNN {

HexagonRelu::HexagonRelu(Backend* backend, float slope) : HexagonExecution(backend), mSlope(slope) {
}

HexagonRelu* HexagonRelu::create(Backend* backend, const Op* op) {
    if (op == nullptr || op->type() != OpType_ReLU || HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    float slope = 0.0f;
    if (op->main_type() == OpParameter_Relu && op->main_as_Relu() != nullptr) {
        slope = op->main_as_Relu()->slope();
    }
    return new HexagonRelu(backend, slope);
}

ErrorCode HexagonRelu::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1) {
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
    if (inputSize != outputSize || outputSize > INT32_MAX) {
        return NOT_SUPPORT;
    }
    auto srcDev = HexagonBackend::getDevicePtr(input);
    auto dstDev = HexagonBackend::getDevicePtr(output);
    if (srcDev.first <= 0 || dstDev.first <= 0) {
        return NOT_SUPPORT;
    }

    struct ReluParam {
        int32_t size;
        int32_t bytes;
        float slope;
    } __attribute__((packed));

    ReluParam params;
    ::memset(&params, 0, sizeof(params));
    params.size = static_cast<int32_t>(outputSize);
    params.bytes = 2;
    params.slope = mSlope;

    std::vector<std::pair<int, int>> inputFds = {srcDev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};
    dst.emplace_back();
    dst.back().build(hexBackend, DSP_OP_RELU, &params, sizeof(params), inputFds, outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonRelu::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonRelu(bn, mSlope);
    return true;
}

} // namespace MNN
