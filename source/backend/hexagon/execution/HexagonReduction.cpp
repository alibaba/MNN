#include "HexagonReduction.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "htp_command.h"

namespace MNN {

enum HexagonReductionOpType {
    HEXAGON_REDUCTION_SUM = 1,
    HEXAGON_REDUCTION_MAXIMUM = 2,
    HEXAGON_REDUCTION_MEAN = 3,
};

static bool _mapReductionOp(ReductionType op, int* dspOpType) {
    switch (op) {
        case ReductionType_SUM:
            *dspOpType = HEXAGON_REDUCTION_SUM;
            return true;
        case ReductionType_MAXIMUM:
            *dspOpType = HEXAGON_REDUCTION_MAXIMUM;
            return true;
        case ReductionType_MEAN:
            *dspOpType = HEXAGON_REDUCTION_MEAN;
            return true;
        default:
            return false;
    }
}

HexagonReduction::HexagonReduction(Backend* backend, int opType, int axis, bool masked)
    : HexagonExecution(backend), mOpType(opType), mAxis(axis), mMasked(masked) {
}

HexagonReduction* HexagonReduction::create(Backend* backend, const Op* op,
                                           const std::vector<Tensor*>& inputs,
                                           const std::vector<Tensor*>& outputs) {
    if (op == nullptr || op->type() != OpType_Reduction || outputs.size() != 1) {
        return nullptr;
    }
    const bool masked = inputs.size() == 2;
    if (inputs.size() != 1 && !masked) {
        return nullptr;
    }
    auto reduction = op->main_as_ReductionParam();
    if (reduction == nullptr || reduction->dim() == nullptr || reduction->dim()->size() != 1) {
        return nullptr;
    }
    if (HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    int dspOpType = 0;
    if (!_mapReductionOp(reduction->operation(), &dspOpType)) {
        return nullptr;
    }
    if (inputs[0] == nullptr || outputs[0] == nullptr || (masked && inputs[1] == nullptr)) {
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
    if (masked && inputs[1]->getType().code != halide_type_float) {
        return nullptr;
    }
    if (HexagonBackend::getBytes(inputs[0]) != 2 || HexagonBackend::getBytes(outputs[0]) != 2) {
        return nullptr;
    }
    if (masked && (dspOpType != HEXAGON_REDUCTION_SUM || HexagonBackend::getBytes(inputs[1]) != 2)) {
        return nullptr;
    }
    return new HexagonReduction(backend, dspOpType, reduction->dim()->Get(0), masked);
}

ErrorCode HexagonReduction::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                       std::vector<HexagonCommand>& dst) {
    if ((inputs.size() != 1 && !(mMasked && inputs.size() == 2)) || outputs.size() != 1) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    auto mask = mMasked ? inputs[1] : nullptr;
    auto output = outputs[0];
    if (input == nullptr || output == nullptr || (mMasked && mask == nullptr)) {
        return INPUT_DATA_ERROR;
    }
    if (input->getType().code != halide_type_float || output->getType().code != halide_type_float) {
        return NOT_SUPPORT;
    }
    if (HexagonBackend::getBytes(input) != 2 || HexagonBackend::getBytes(output) != 2) {
        return NOT_SUPPORT;
    }
    if (mMasked && (mask->getType().code != halide_type_float || HexagonBackend::getBytes(mask) != 2)) {
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
    int reduce = input->length(axis);
    int inside = 1;
    for (int i = 0; i < axis; ++i) {
        outside *= input->length(i);
    }
    for (int i = axis + 1; i < dimensions; ++i) {
        inside *= input->length(i);
    }
    if (outside <= 0 || reduce <= 0 || inside <= 0) {
        return NOT_SUPPORT;
    }
    if (output->elementSize() != outside * inside) {
        return NOT_SUPPORT;
    }

    auto srcDev = HexagonBackend::getDevicePtr(input);
    auto dstDev = HexagonBackend::getDevicePtr(output);
    if (srcDev.first <= 0 || dstDev.first <= 0) {
        return NOT_SUPPORT;
    }
    auto maskDev = mMasked ? HexagonBackend::getDevicePtr(mask) : std::make_pair(-1, 0);
    if (mMasked && maskDev.first <= 0) {
        return NOT_SUPPORT;
    }

    int32_t params[5] = {outside, reduce, inside, mOpType, 2};
    std::vector<std::pair<int, int>> inputFds = mMasked ? std::vector<std::pair<int, int>>{srcDev, maskDev}
                                                        : std::vector<std::pair<int, int>>{srcDev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), mMasked ? DSP_OP_MASKED_REDUCTION : DSP_OP_REDUCTION,
                     params, sizeof(params),
                     inputFds, outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonReduction::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonReduction(bn, mOpType, mAxis, mMasked);
    return true;
}

} // namespace MNN
