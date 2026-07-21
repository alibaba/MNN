#include "HexagonSoftmax.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "htp_command.h"

namespace MNN {

HexagonSoftmax::HexagonSoftmax(Backend* backend, int axis) : HexagonExecution(backend), mAxis(axis) {
}

HexagonSoftmax* HexagonSoftmax::create(Backend* backend, const Op* op,
                                       const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs) {
    if (op == nullptr || op->type() != OpType_Softmax || inputs.size() != 1 || outputs.size() != 1) {
        return nullptr;
    }
    if (inputs[0] == nullptr || outputs[0] == nullptr) {
        return nullptr;
    }
    auto axis = op->main_as_Axis();
    if (axis == nullptr) {
        return nullptr;
    }
    if (HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    auto inputDes = TensorUtils::getDescribe(inputs[0]);
    auto outputDes = TensorUtils::getDescribe(outputs[0]);
    if (inputDes->dimensionFormat != MNN_DATA_FORMAT_NCHW ||
        outputDes->dimensionFormat != MNN_DATA_FORMAT_NCHW) {
        return nullptr;
    }
    if (inputs[0]->getType().code != halide_type_float || outputs[0]->getType().code != halide_type_float) {
        return nullptr;
    }
    if (HexagonBackend::getBytes(inputs[0]) != 2 || HexagonBackend::getBytes(outputs[0]) != 2) {
        return nullptr;
    }
    return new HexagonSoftmax(backend, axis->axis());
}

ErrorCode HexagonSoftmax::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                     std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }
    const auto input = inputs[0];
    const auto output = outputs[0];
    if (input == nullptr || output == nullptr) {
        return INPUT_DATA_ERROR;
    }
    if (input->getType().code != halide_type_float || output->getType().code != halide_type_float) {
        return NOT_SUPPORT;
    }
    if (HexagonBackend::getBytes(input) != 2 || HexagonBackend::getBytes(output) != 2) {
        return NOT_SUPPORT;
    }
    if (input->elementSize() != output->elementSize()) {
        return NOT_SUPPORT;
    }

    const int dimensions = input->dimensions();
    if (dimensions <= 0) {
        return NOT_SUPPORT;
    }
    int axis = mAxis;
    if (axis < 0) {
        axis += dimensions;
    }
    if (axis < 0 || axis >= dimensions) {
        return NOT_SUPPORT;
    }

    int outside = 1;
    int channel = input->length(axis);
    int inside = 1;
    for (int i = 0; i < axis; ++i) {
        outside *= input->length(i);
    }
    for (int i = axis + 1; i < dimensions; ++i) {
        inside *= input->length(i);
    }
    if (outside <= 0 || channel <= 0 || inside <= 0) {
        return NOT_SUPPORT;
    }

    auto srcDev = HexagonBackend::getDevicePtr(input);
    auto dstDev = HexagonBackend::getDevicePtr(output);
    if (srcDev.first <= 0 || dstDev.first <= 0) {
        return NOT_SUPPORT;
    }

    int32_t params[4] = {outside, channel, inside, 2};
    std::vector<std::pair<int, int>> inputFds = {srcDev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_SOFTMAX, params, sizeof(params),
                     inputFds, outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonSoftmax::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonSoftmax(bn, mAxis);
    return true;
}

} // namespace MNN
