#include "HexagonSharedGather.hpp"

#include "HexagonBackend.hpp"
#include "core/Macro.h"
#include "htp_command.h"

namespace MNN {

HexagonSharedGather::HexagonSharedGather(Backend* backend, std::shared_ptr<HexagonConvolution::Resource> res)
    : HexagonExecution(backend), mResource(std::move(res)) {
}

ErrorCode HexagonSharedGather::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                          std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }
    if (mResource == nullptr) {
        return NOT_SUPPORT;
    }

    auto indices = inputs[0];
    auto output = outputs[0];
    if (indices == nullptr || output == nullptr) {
        return INPUT_DATA_ERROR;
    }
    if (indices->getType().code != halide_type_int || output->getType().code != halide_type_float) {
        return NOT_SUPPORT;
    }

    const int selectSize = indices->elementSize();
    const int ic = output->length(output->dimensions() - 1);
    const int oc = mResource->gatherOutputChannels;
    const int bytes = HexagonBackend::getBytes(output);
    if (selectSize <= 0 || ic != mResource->gatherInputChannels || (bytes != 2 && bytes != 4)) {
        return NOT_SUPPORT;
    }

    struct SharedGatherParam {
        int32_t selectSize;
        int32_t ic;
        int32_t oc;
        int32_t bytes;
        int32_t isInt4;
    } __attribute__((packed));

    const bool useInt4 = mResource->useInt4W4A16;
    if ((!useInt4 && mResource->weight.first == nullptr) ||
        (useInt4 && mResource->int4Weight.first == nullptr && mResource->gatherInt4Weight.first == nullptr)) {
        return NOT_SUPPORT;
    }
    const bool useRawInt4Gather = useInt4 && mResource->gatherInt4Weight.first != nullptr;
    SharedGatherParam params = {selectSize, ic, oc, bytes, useRawInt4Gather ? 2 : (useInt4 ? 1 : 0)};

    auto indicesDev = HexagonBackend::getDevicePtr(indices);
    auto weightDev = useRawInt4Gather ? HexagonBackend::getDevicePtr(mResource->gatherInt4Weight)
                                      : (useInt4 ? HexagonBackend::getDevicePtr(mResource->int4Weight)
                                                 : HexagonBackend::getDevicePtr(mResource->weight));
    auto outputDev = HexagonBackend::getDevicePtr(output);
    auto hexagonBackend = static_cast<HexagonBackend*>(backend());

    std::vector<std::pair<int, int>> inputFds = {indicesDev, weightDev};
    std::vector<std::pair<int, int>> outputFds = {outputDev};

    dst.emplace_back();
    dst.back().build(hexagonBackend, DSP_OP_SHARED_GATHER, &params, sizeof(params), inputFds,
                     outputFds,  inputs, outputs);
    return NO_ERROR;
}

bool HexagonSharedGather::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonSharedGather(bn, mResource);
    return true;
}

} // namespace MNN
