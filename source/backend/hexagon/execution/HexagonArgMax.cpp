#include "HexagonArgMax.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "htp_command.h"

namespace MNN {

HexagonArgMax::HexagonArgMax(Backend* backend, int axis) : HexagonExecution(backend), mAxis(axis) {}

HexagonArgMax* HexagonArgMax::create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    if (op == nullptr || op->type() != OpType_ArgMax || inputs.size() != 1 || outputs.size() != 1) {
        return nullptr;
    }
    auto param = op->main_as_ArgMax();
    if (param == nullptr || param->topK() > 1 || param->outMaxVal() != 0) {
        return nullptr;
    }
    auto input = inputs[0];
    auto indexOutput = outputs[0];
    if (input == nullptr || indexOutput == nullptr || input->dimensions() <= 0) {
        return nullptr;
    }
    if (input->getType().code != halide_type_float || indexOutput->getType().code != halide_type_int ||
        indexOutput->getType().bits != 32) {
        return nullptr;
    }
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        return nullptr;
    }
    if (HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    int axis = param->axis();
    if (axis < 0) {
        axis += input->dimensions();
    }
    if (axis != input->dimensions() - 1) {
        // The kernel reduces the innermost dimension only; the DFlash argmax over [rows, vocab] is
        // already in that shape.
        return nullptr;
    }
    return new HexagonArgMax(backend, axis);
}

ErrorCode HexagonArgMax::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }
    const auto input = inputs[0];
    const auto indexOutput = outputs[0];
    const int dimensions = input->dimensions();
    if (dimensions <= 0 || mAxis < 0 || mAxis != dimensions - 1) {
        return NOT_SUPPORT;
    }
    const int rowSize = input->length(mAxis);
    if (rowSize <= 0 || input->elementSize() % rowSize != 0) {
        return NOT_SUPPORT;
    }
    const int rows = input->elementSize() / rowSize;
    if (indexOutput->elementSize() != rows) {
        return NOT_SUPPORT;
    }
    if (HexagonBackend::getBytes(input) != 2 || HexagonBackend::getBytes(indexOutput) != 4) {
        return NOT_SUPPORT;
    }

    auto inputDev = HexagonBackend::getDevicePtr(input);
    auto indexDev = HexagonBackend::getDevicePtr(indexOutput);

    int32_t params[2] = {rowSize, rows};
    std::vector<std::pair<int, int>> inputFds = {inputDev};
    std::vector<std::pair<int, int>> outputFds = {indexDev};

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_ARGMAX_FP16, params, sizeof(params), inputFds,
                     outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonArgMax::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    auto param = op->main_as_ArgMax();
    int axis = param == nullptr ? -1 : param->axis();
    *dst = new HexagonArgMax(bn, axis);
    return true;
}

} // namespace MNN
