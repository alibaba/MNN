#include "HexagonBinary.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "htp_command.h"
#include <cstring>

namespace MNN {

static constexpr int kBinaryAddReluOpType = 8;
static constexpr int kBinaryGreaterOpType = 9;
static constexpr int kBinaryLessOpType = 10;
static constexpr int kBinarySquaredDifferenceOpType = 11;

static bool _mapBinaryOp(int mnnOpType, int* dspOpType) {
    // Keep in sync with HtpOpsBinaryOpType in htp-ops-lib/src/dsp/blit_ops.c
    switch (mnnOpType) {
        case BinaryOpOperation_ADD:
            *dspOpType = 1;
            return true;
        case BinaryOpOperation_SUB:
            *dspOpType = 2;
            return true;
        case BinaryOpOperation_MUL:
            *dspOpType = 3;
            return true;
        case BinaryOpOperation_DIV:
        case BinaryOpOperation_REALDIV:
            *dspOpType = 4;
            return true;
        case BinaryOpOperation_MAXIMUM:
            *dspOpType = 5;
            return true;
        case BinaryOpOperation_MINIMUM:
            *dspOpType = 6;
            return true;
        case BinaryOpOperation_MUL_SILU:
            *dspOpType = 7;
            return true;
        case BinaryOpOperation_GREATER:
            *dspOpType = kBinaryGreaterOpType;
            return true;
        case BinaryOpOperation_LESS:
            *dspOpType = kBinaryLessOpType;
            return true;
        case BinaryOpOperation_SquaredDifference:
            *dspOpType = kBinarySquaredDifferenceOpType;
            return true;
        default:
            return false;
    }
}

static bool _buildBroadcastShape(const Tensor* input0, const Tensor* input1, const Tensor* output,
                                 int32_t outDims[8], int32_t in0Strides[8], int32_t in1Strides[8],
                                 int32_t* broadcastDims) {
    const int dims = output->dimensions();
    if (dims <= 0 || dims > 8) {
        return false;
    }
    if (TensorUtils::getDescribe(output)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && dims == 4) {
        return false;
    }

    int32_t in0Shape[8] = {0};
    int32_t in1Shape[8] = {0};
    int32_t in0RawStrides[8] = {0};
    int32_t in1RawStrides[8] = {0};

    int stride = 1;
    for (int i = input0->dimensions() - 1; i >= 0; --i) {
        const int dst = dims - input0->dimensions() + i;
        if (dst < 0) {
            return false;
        }
        in0Shape[dst] = input0->length(i);
        in0RawStrides[dst] = stride;
        stride *= input0->length(i);
    }
    stride = 1;
    for (int i = input1->dimensions() - 1; i >= 0; --i) {
        const int dst = dims - input1->dimensions() + i;
        if (dst < 0) {
            return false;
        }
        in1Shape[dst] = input1->length(i);
        in1RawStrides[dst] = stride;
        stride *= input1->length(i);
    }

    bool needBroadcast = false;
    for (int i = 0; i < dims; ++i) {
        outDims[i] = output->length(i);
        if (outDims[i] <= 0) {
            return false;
        }
        if (in0Shape[i] == 0) {
            in0Shape[i] = 1;
            in0RawStrides[i] = 0;
        }
        if (in1Shape[i] == 0) {
            in1Shape[i] = 1;
            in1RawStrides[i] = 0;
        }
        if (in0Shape[i] != outDims[i] && in0Shape[i] != 1) {
            return false;
        }
        if (in1Shape[i] != outDims[i] && in1Shape[i] != 1) {
            return false;
        }
        in0Strides[i] = in0Shape[i] == 1 ? 0 : in0RawStrides[i];
        in1Strides[i] = in1Shape[i] == 1 ? 0 : in1RawStrides[i];
        needBroadcast = needBroadcast || in0Strides[i] == 0 || in1Strides[i] == 0;
    }
    *broadcastDims = needBroadcast ? dims : 0;
    return true;
}

HexagonBinary::HexagonBinary(Backend* backend, int dspOpType) : HexagonExecution(backend), mDspOpType(dspOpType) {
    mAllocator = static_cast<HexagonBackend*>(backend)->getAllocator(1);
}



HexagonBinary* HexagonBinary::create(Backend* backend, const Op* op) {
    if (op->type() != OpType_BinaryOp) {
        return nullptr;
    }
    auto binary = op->main_as_BinaryOp();
    if (binary == nullptr) {
        return nullptr;
    }
    int dspOpType = 0;
    if (!_mapBinaryOp(binary->opType(), &dspOpType)) {
        return nullptr;
    }
    int activationType = binary->activationType();
    if (activationType != 0) {
        if (activationType != 1 || binary->opType() != BinaryOpOperation_ADD) {
            return nullptr;
        }
        dspOpType = kBinaryAddReluOpType;
    }

    auto functions = HexagonRuntime::getDstFunctions();
    if (functions == nullptr ) {
        return nullptr;
    }

    return new HexagonBinary(backend, dspOpType);
}

ErrorCode HexagonBinary::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }

    const auto input0 = inputs[0];
    const auto input1 = inputs[1];
    const auto output = outputs[0];
    if (input0 == nullptr || input1 == nullptr || output == nullptr) {
        return INPUT_DATA_ERROR;
    }

    if ((input0->getType().code != halide_type_float && input0->getType().code != halide_type_int) ||
        (input1->getType().code != halide_type_float && input1->getType().code != halide_type_int) ||
        (output->getType().code != halide_type_float && output->getType().code != halide_type_int)) {
        return NOT_SUPPORT;
    }

    mBytes = HexagonBackend::getBytes(outputs[0]);
    if (mBytes != 2 && mBytes != 4) {
        return NOT_SUPPORT;
    }
    const int inputBytes = HexagonBackend::getBytes(input0);
    if (inputBytes != HexagonBackend::getBytes(input1) || (inputBytes != 2 && inputBytes != 4)) {
        return NOT_SUPPORT;
    }

    auto hexBackend = static_cast<HexagonBackend*>(backend());
    mOutSize = hexBackend->getElementSize(output);
    mIn0Size = hexBackend->getElementSize(input0);
    mIn1Size = hexBackend->getElementSize(input1);
    int32_t outDims[8] = {0};
    int32_t in0Strides[8] = {0};
    int32_t in1Strides[8] = {0};
    int32_t broadcastDims = 0;
    if (!((mIn0Size == mOutSize || mIn0Size == 1) && (mIn1Size == mOutSize || mIn1Size == 1))) {
        if (!_buildBroadcastShape(input0, input1, output, outDims, in0Strides, in1Strides, &broadcastDims)) {
            return NOT_SUPPORT;
        }
    }

    auto dstDev = HexagonBackend::getDevicePtr(output);
    auto src0Dev = HexagonBackend::getDevicePtr(input0);
    auto src1Dev = HexagonBackend::getDevicePtr(input1);

        struct MergedBinaryParam {
            int32_t outSize;
            int32_t in0Size;
            int32_t in1Size;
            int32_t opType;
            int32_t bytes;
            int32_t inputBytes;
            int32_t inputIsFloat;
            int32_t outputIsFloat;
            int32_t broadcastDims;
            int32_t outDims[8];
            int32_t in0Strides[8];
            int32_t in1Strides[8];
        } __attribute__((packed));

        MergedBinaryParam params;
        ::memset(&params, 0, sizeof(params));
        params.outSize = (int)mOutSize;
        params.in0Size = (int)mIn0Size;
        params.in1Size = (int)mIn1Size;
        params.opType = mDspOpType;
        params.bytes = mBytes;
        params.inputBytes = inputBytes;
        params.inputIsFloat = input0->getType().code == halide_type_float ? 1 : 0;
        params.outputIsFloat = output->getType().code == halide_type_float ? 1 : 0;
        params.broadcastDims = broadcastDims;
        ::memcpy(params.outDims, outDims, sizeof(outDims));
        ::memcpy(params.in0Strides, in0Strides, sizeof(in0Strides));
        ::memcpy(params.in1Strides, in1Strides, sizeof(in1Strides));

        std::vector<std::pair<int, int>> inputFds = {src0Dev, src1Dev};
        std::vector<std::pair<int, int>> outputFds = {dstDev};

        dst.emplace_back();
        dst.back().build(hexBackend, DSP_OP_BINARY_ELEMENTWISE, &params, sizeof(params),
                         inputFds,  outputFds,  inputs, outputs);

    return NO_ERROR;
}

bool HexagonBinary::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonBinary(bn, mDspOpType);
    return true;
}

} // namespace MNN
