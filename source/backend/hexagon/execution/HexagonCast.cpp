#include "HexagonCast.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "htp_command.h"

namespace MNN {

enum HexagonCastType {
    HEXAGON_CAST_FP16_TO_FP16 = 0,
    HEXAGON_CAST_INT32_TO_INT32 = 1,
    HEXAGON_CAST_INT32_TO_FP16 = 2,
    HEXAGON_CAST_FP16_TO_INT32 = 3,
};

static bool _mapCastType(const Tensor* input, const Tensor* output, int* castType) {
    const auto inputType = input->getType();
    const auto outputType = output->getType();
    const int inputBytes = HexagonBackend::getBytes(input);
    const int outputBytes = HexagonBackend::getBytes(output);

    if (inputType.code == halide_type_float && outputType.code == halide_type_float &&
        inputBytes == 2 && outputBytes == 2) {
        *castType = HEXAGON_CAST_FP16_TO_FP16;
        return true;
    }
    if (inputType.code == halide_type_int && outputType.code == halide_type_int &&
        inputBytes == 4 && outputBytes == 4) {
        *castType = HEXAGON_CAST_INT32_TO_INT32;
        return true;
    }
    if (inputType.code == halide_type_int && outputType.code == halide_type_float &&
        inputBytes == 4 && outputBytes == 2) {
        *castType = HEXAGON_CAST_INT32_TO_FP16;
        return true;
    }
    if (inputType.code == halide_type_float && outputType.code == halide_type_int &&
        inputBytes == 2 && outputBytes == 4) {
        *castType = HEXAGON_CAST_FP16_TO_INT32;
        return true;
    }
    return false;
}

HexagonCast::HexagonCast(Backend* backend, int castType) : HexagonExecution(backend), mCastType(castType) {
    mAllocator = static_cast<HexagonBackend*>(backend)->getAllocator(1);
}

HexagonCast* HexagonCast::create(Backend* backend, const Op* op,
                                 const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    if (op->type() != OpType_Cast) {
        return nullptr;
    }
    if (op->main_as_CastParam() == nullptr) {
        return nullptr;
    }
    if (HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    if (inputs.size() != 1 || outputs.size() != 1 || inputs[0] == nullptr || outputs[0] == nullptr) {
        return nullptr;
    }
    int castType = 0;
    if (!_mapCastType(inputs[0], outputs[0], &castType)) {
        return nullptr;
    }
    return new HexagonCast(backend, castType);
}

ErrorCode HexagonCast::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    auto output = outputs[0];
    if (input == nullptr || output == nullptr) {
        return INPUT_DATA_ERROR;
    }

    int castType = 0;
    if (!_mapCastType(input, output, &castType)) {
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

    struct CastParam {
        int32_t size;
        int32_t castType;
    } __attribute__((packed));

    CastParam params;
    params.size = (int32_t)outputSize;
    params.castType = castType;

    std::vector<std::pair<int, int>> inputFds = {srcDev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};

    dst.emplace_back();
    dst.back().build(hexBackend, DSP_OP_CAST, &params, sizeof(params),
                     inputFds, outputFds, inputs, outputs);

    mCastType = castType;
    return NO_ERROR;
}

bool HexagonCast::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonCast(bn, mCastType);
    return true;
}

} // namespace MNN
