#include "HexagonSelect.hpp"

#include "HexagonBackend.hpp"
#include "MNN_generated.h"
#include "htp_command.h"

namespace MNN {

HexagonSelect::HexagonSelect(Backend* backend) : HexagonExecution(backend) {
}

HexagonSelect* HexagonSelect::create(Backend* backend, const Op* op) {
    if (op->type() != OpType_Select) {
        return nullptr;
    }
    return new HexagonSelect(backend);
}

ErrorCode HexagonSelect::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 3 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }

    auto cond = inputs[0];
    auto input1 = inputs[1];
    auto input2 = inputs[2];
    auto output = outputs[0];
    if (cond == nullptr || input1 == nullptr || input2 == nullptr || output == nullptr) {
        return INPUT_DATA_ERROR;
    }

    mCondBytes = HexagonBackend::getBytes(cond);
    if (mCondBytes != 1 && mCondBytes != 2 && mCondBytes != 4) {
        return NOT_SUPPORT;
    }

    mBytes = HexagonBackend::getBytes(output);
    if (mBytes != 1 && mBytes != 2 && mBytes != 4) {
        return NOT_SUPPORT;
    }
    if (HexagonBackend::getBytes(input1) != mBytes || HexagonBackend::getBytes(input2) != mBytes) {
        return NOT_SUPPORT;
    }

    auto hexBackend = static_cast<HexagonBackend*>(backend());
    mOutSize = hexBackend->getElementSize(output);
    mCondSize = hexBackend->getElementSize(cond);
    mIn1Size = hexBackend->getElementSize(input1);
    mIn2Size = hexBackend->getElementSize(input2);

    int32_t channelSize = output->dimensions() > 1 ? output->length(1) : 0;
    int32_t innerSize = 1;
    for (int i = 2; i < output->dimensions(); ++i) {
        innerSize *= output->length(i);
    }
    auto supportInputBroadcast = [=](size_t size) {
        return size == mOutSize || size == 1 || (channelSize > 0 && innerSize > 0 && size == (size_t)channelSize);
    };
    if (!((mCondSize == mOutSize || mCondSize == 1) &&
          supportInputBroadcast(mIn1Size) &&
          supportInputBroadcast(mIn2Size))) {
        return NOT_SUPPORT;
    }

    auto condDev = HexagonBackend::getDevicePtr(cond);
    auto in1Dev = HexagonBackend::getDevicePtr(input1);
    auto in2Dev = HexagonBackend::getDevicePtr(input2);
    auto dstDev = HexagonBackend::getDevicePtr(output);

    struct SelectParam {
        int32_t outSize;
        int32_t condSize;
        int32_t in1Size;
        int32_t in2Size;
        int32_t bytes;
        int32_t condBytes;
        int32_t channelSize;
        int32_t innerSize;
    } __attribute__((packed));

    SelectParam params;
    params.outSize = (int32_t)mOutSize;
    params.condSize = (int32_t)mCondSize;
    params.in1Size = (int32_t)mIn1Size;
    params.in2Size = (int32_t)mIn2Size;
    params.bytes = mBytes;
    params.condBytes = mCondBytes;
    params.channelSize = channelSize;
    params.innerSize = innerSize;

    std::vector<std::pair<int, int>> inputFds = {condDev, in1Dev, in2Dev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};

    dst.emplace_back();
    dst.back().build(hexBackend, DSP_OP_SELECT, &params, sizeof(params),
                     inputFds, outputFds, inputs, outputs);

    return NO_ERROR;
}

bool HexagonSelect::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonSelect(bn);
    return true;
}

} // namespace MNN
