#include "HexagonUnary.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "htp_command.h"

namespace MNN {

static bool _mapUnaryOp(int mnnOpType, int* dspOpType) {
    // Keep in sync with HtpOpsUnaryOpType in htp-ops-lib/src/dsp/unary_ops.cc
    switch (mnnOpType) {
        case UnaryOpOperation_ABS:
            *dspOpType = 1;
            return true;
        case UnaryOpOperation_NEG:
            *dspOpType = 2;
            return true;
        case UnaryOpOperation_GELU:
            *dspOpType = 3;
            return true;
        case UnaryOpOperation_SIGMOID:
            *dspOpType = 4;
            return true;
        case UnaryOpOperation_EXP:
            *dspOpType = 5;
            return true;
        case UnaryOpOperation_LOG:
            *dspOpType = 6;
            return true;
        case UnaryOpOperation_SILU:
            *dspOpType = 7;
            return true;
        case UnaryOpOperation_TANH:
            *dspOpType = 8;
            return true;
        case UnaryOpOperation_SQUARE:
            *dspOpType = 9;
            return true;
        case UnaryOpOperation_SQRT:
            *dspOpType = 10;
            return true;
        case UnaryOpOperation_RSQRT:
            *dspOpType = 11;
            return true;
        case UnaryOpOperation_EXPM1:
            *dspOpType = 12;
            return true;
        case UnaryOpOperation_COS:
            *dspOpType = 13;
            return true;
        case UnaryOpOperation_SIN:
            *dspOpType = 14;
            return true;
        default:
            return false;
    }
}

HexagonUnary::HexagonUnary(Backend* backend, int dspOpType) : HexagonExecution(backend), mDspOpType(dspOpType) {
    mAllocator = static_cast<HexagonBackend*>(backend)->getAllocator(1);
}

HexagonUnary* HexagonUnary::create(Backend* backend, const Op* op) {
    if (op->type() != OpType_UnaryOp) {
        return nullptr;
    }
    auto unary = op->main_as_UnaryOp();
    if (unary == nullptr) {
        return nullptr;
    }
    int dspOpType = 0;
    if (!_mapUnaryOp(unary->opType(), &dspOpType)) {
        return nullptr;
    }

    auto functions = HexagonRuntime::getDstFunctions();
    if (functions == nullptr) {
        return nullptr;
    }
    return new HexagonUnary(backend, dspOpType);
}

ErrorCode HexagonUnary::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }

    const auto input = inputs[0];
    const auto output = outputs[0];
    if (input == nullptr || output == nullptr) {
        return INPUT_DATA_ERROR;
    }

    if ((input->getType().code != halide_type_float && input->getType().code != halide_type_int) ||
        (output->getType().code != halide_type_float && output->getType().code != halide_type_int)) {
        return NOT_SUPPORT;
    }
    if (input->getType().code != output->getType().code) {
        return NOT_SUPPORT;
    }

    mBytes = HexagonBackend::getBytes(output);
    if (mBytes != 2 && mBytes != 4) {
        return NOT_SUPPORT;
    }
    if (mBytes == 4 && (mDspOpType != 1 && mDspOpType != 2)) {
        return NOT_SUPPORT;
    }
    if (mBytes == 2 && (input->getType().code != halide_type_float || output->getType().code != halide_type_float)) {
        return NOT_SUPPORT;
    }
    if (mBytes == 4 && (input->getType().code != halide_type_int || output->getType().code != halide_type_int)) {
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

    struct MergedUnaryParam {
        int32_t size;
        int32_t opType;
        int32_t bytes;
    } __attribute__((packed));

    MergedUnaryParam params;
    params.size = (int32_t)outputSize;
    params.opType = mDspOpType;
    params.bytes = mBytes;

    std::vector<std::pair<int, int>> inputFds = {srcDev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};

    dst.emplace_back();
    dst.back().build(hexBackend, DSP_OP_UNARY, &params, sizeof(params), inputFds, outputFds, inputs, outputs);

    return NO_ERROR;
}

bool HexagonUnary::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonUnary(bn, mDspOpType);
    return true;
}

} // namespace MNN
