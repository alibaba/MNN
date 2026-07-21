#include "HexagonLSTM.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "htp_command.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>

namespace MNN {

namespace {

static void packLstmHmxWeight(int16_t* dst, const int16_t* src, int gateSize, int kSize, int packedKp) {
    const int np = UP_DIV(gateSize, 32);
    const int kp = UP_DIV(kSize, 32);
    const size_t packedElems = (size_t)np * packedKp * 1024;
    ::memset(dst, 0, packedElems * sizeof(int16_t));
    for (int nt = 0; nt < np; ++nt) {
        for (int kt = 0; kt < kp; ++kt) {
            int16_t* tile = dst + ((size_t)nt * packedKp + kt) * 1024;
            const int kBegin = kt * 32;
            const int nBegin = nt * 32;
            int nRemain = gateSize - nBegin;
            if (nRemain > 32) {
                nRemain = 32;
            }
            for (int k = 0; k < 32; ++k) {
                const int rawK = kBegin + k;
                if (rawK >= kSize) {
                    continue;
                }
                const int16_t* srcRow = src + rawK;
                for (int c = 0; c < nRemain; ++c) {
                    const int dstIndex = (k / 2) * 64 + c * 2 + (k & 1);
                    tile[dstIndex] = srcRow[(nBegin + c) * kSize];
                }
            }
        }
    }
}

} // namespace

HexagonLSTM::HexagonLSTM(Backend* backend, int hiddenSize) : HexagonExecution(backend), mHiddenSize(hiddenSize) {
}

HexagonLSTM::~HexagonLSTM() {
    releasePackedWeights();
}

void HexagonLSTM::releasePackedWeights() {
    if (mPackedW != nullptr) {
        backend()->onReleaseBuffer(mPackedW.get(), Backend::STATIC);
        mPackedW.reset();
    }
    if (mPackedR != nullptr) {
        backend()->onReleaseBuffer(mPackedR.get(), Backend::STATIC);
        mPackedR.reset();
    }
    mPackedInputSize = 0;
    mPackedHiddenSize = 0;
    mPackedGateSize = 0;
    mPackedDirection = 0;
}

HexagonLSTM* HexagonLSTM::create(Backend* backend, const Op* op,
                                 const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    if (op->type() != OpType_LSTM || inputs.size() < 6 || outputs.empty()) {
        return nullptr;
    }
    if (HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    auto x = inputs[0];
    auto w = inputs[1];
    auto r = inputs[2];
    if (x == nullptr || w == nullptr || r == nullptr || x->dimensions() < 3 || w->dimensions() < 3 ||
        r->dimensions() < 3) {
        return nullptr;
    }
    if (x->getType().code != halide_type_float || w->getType().code != halide_type_float ||
        r->getType().code != halide_type_float) {
        return nullptr;
    }
    int hiddenSize = r->length(2);
    if (op->main_type() == OpParameter_Axis && op->main_as_Axis()->axis() > 0) {
        hiddenSize = op->main_as_Axis()->axis();
    }
    if (hiddenSize <= 0) {
        return nullptr;
    }
    return new HexagonLSTM(backend, hiddenSize);
}

ErrorCode HexagonLSTM::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  std::vector<HexagonCommand>& dst) {
    if (inputs.size() < 6 || outputs.empty()) {
        return NOT_SUPPORT;
    }
    auto x = inputs[0];
    auto w = inputs[1];
    auto r = inputs[2];
    auto b = inputs[3];
    auto h0 = inputs[4];
    auto c0 = inputs[5];
    if (x == nullptr || w == nullptr || r == nullptr || b == nullptr || h0 == nullptr || c0 == nullptr ||
        outputs[0] == nullptr) {
        return INPUT_DATA_ERROR;
    }
    if (x->getType().code != halide_type_float || w->getType().code != halide_type_float ||
        r->getType().code != halide_type_float || b->getType().code != halide_type_float ||
        h0->getType().code != halide_type_float || c0->getType().code != halide_type_float) {
        return NOT_SUPPORT;
    }

    const int bytes = HexagonBackend::getBytes(outputs[0]);
    if (bytes != 2 && bytes != 4) {
        return NOT_SUPPORT;
    }
    const int seqLength = x->length(0);
    const int batch = x->length(1);
    const int inputSize = x->length(2);
    const int direction = w->length(0);
    const int hiddenSize = mHiddenSize > 0 ? mHiddenSize : r->length(2);
    if (seqLength <= 0 || batch <= 0 || inputSize <= 0 || direction <= 0 || hiddenSize <= 0) {
        return NOT_SUPPORT;
    }
    if (w->length(1) < 4 * hiddenSize || w->length(2) < inputSize || r->length(1) < 4 * hiddenSize ||
        r->length(2) < hiddenSize) {
        return NOT_SUPPORT;
    }
    struct LSTMParam {
        int32_t seqLength;
        int32_t batch;
        int32_t inputSize;
        int32_t hiddenSize;
        int32_t direction;
        int32_t bytes;
        int32_t outputCount;
        int32_t xSize;
        int32_t wSize;
        int32_t rSize;
        int32_t bSize;
        int32_t h0Size;
        int32_t c0Size;
        int32_t ySize;
        int32_t yhSize;
        int32_t ycSize;
        int32_t scratchSize;
        int32_t packedWeightBytes;
    } __attribute__((packed));

    const int gateSize = 4 * hiddenSize;
    const int stateSize = batch * hiddenSize;
    const bool canUseHmxLstm = bytes == 2 && inputSize % 64 == 0 && hiddenSize % 64 == 0;
    size_t scratchBytes = 4 * (size_t)stateSize * sizeof(float);
    if (canUseHmxLstm) {
        const size_t stateBytes = (size_t)stateSize * sizeof(int16_t);
        scratchBytes = std::max(scratchBytes, 2 * stateBytes);
    }
    if (scratchBytes > (size_t)INT_MAX) {
        return NOT_SUPPORT;
    }

    LSTMParam params;
    params.seqLength = seqLength;
    params.batch = batch;
    params.inputSize = inputSize;
    params.hiddenSize = hiddenSize;
    params.direction = direction;
    params.bytes = bytes;
    params.outputCount = (int32_t)outputs.size();
    params.xSize = x->elementSize();
    params.wSize = w->elementSize();
    params.rSize = r->elementSize();
    params.bSize = b->elementSize();
    params.h0Size = h0->elementSize();
    params.c0Size = c0->elementSize();
    params.ySize = outputs[0]->elementSize();
    params.yhSize = outputs.size() > 1 ? outputs[1]->elementSize() : 0;
    params.ycSize = outputs.size() > 2 ? outputs[2]->elementSize() : 0;
    params.scratchSize = (int32_t)scratchBytes;
    params.packedWeightBytes = 0;

    const int packedKp = std::max(UP_DIV(inputSize, 32), UP_DIV(hiddenSize, 32));
    const int packedWeightBytes = (int)(UP_DIV(gateSize, 32) * packedKp * 1024 * sizeof(int16_t));
    const bool canUsePackedWeights = canUseHmxLstm;
    if (!canUsePackedWeights) {
        releasePackedWeights();
    }
    if (canUsePackedWeights &&
        (mPackedW == nullptr || mPackedR == nullptr ||
         mPackedInputSize != inputSize || mPackedHiddenSize != hiddenSize ||
         mPackedGateSize != gateSize || mPackedDirection != direction)) {
        releasePackedWeights();
        const float* wHost = w->host<float>();
        const float* rHost = r->host<float>();
        if (wHost != nullptr && rHost != nullptr) {
            const size_t totalPackedBytes = (size_t)direction * packedWeightBytes;
            std::shared_ptr<Tensor> packedW(Tensor::createDevice<int8_t>({(int)totalPackedBytes}));
            std::shared_ptr<Tensor> packedR(Tensor::createDevice<int8_t>({(int)totalPackedBytes}));
            const bool packedWAllocated = packedW != nullptr && backend()->onAcquireBuffer(packedW.get(), Backend::STATIC);
            const bool packedRAllocated = packedR != nullptr && backend()->onAcquireBuffer(packedR.get(), Backend::STATIC);
            if (packedWAllocated && packedRAllocated) {
                auto packedWPtr = reinterpret_cast<int16_t*>(HexagonBackend::getPtr(packedW.get()));
                auto packedRPtr = reinterpret_cast<int16_t*>(HexagonBackend::getPtr(packedR.get()));
                std::vector<int16_t> wHalf(w->elementSize());
                std::vector<int16_t> rHalf(r->elementSize());
                HexagonBackend::fp32ToFp16(wHost, wHalf.data(), wHalf.size());
                HexagonBackend::fp32ToFp16(rHost, rHalf.data(), rHalf.size());
                for (int d = 0; d < direction; ++d) {
                    packLstmHmxWeight(packedWPtr + ((size_t)d * packedWeightBytes / sizeof(int16_t)),
                                      wHalf.data() + (size_t)d * gateSize * inputSize, gateSize, inputSize, packedKp);
                    packLstmHmxWeight(packedRPtr + ((size_t)d * packedWeightBytes / sizeof(int16_t)),
                                      rHalf.data() + (size_t)d * gateSize * hiddenSize, gateSize, hiddenSize, packedKp);
                }
                mPackedW = packedW;
                mPackedR = packedR;
                static_cast<HexagonBackend*>(backend())->markHostInput(mPackedW.get());
                static_cast<HexagonBackend*>(backend())->markHostInput(mPackedR.get());
                mPackedInputSize = inputSize;
                mPackedHiddenSize = hiddenSize;
                mPackedGateSize = gateSize;
                mPackedDirection = direction;
            } else {
                if (packedWAllocated) {
                    backend()->onReleaseBuffer(packedW.get(), Backend::STATIC);
                }
                if (packedRAllocated) {
                    backend()->onReleaseBuffer(packedR.get(), Backend::STATIC);
                }
            }
        }
    }
    if (canUsePackedWeights && mPackedW != nullptr && mPackedR != nullptr &&
        mPackedInputSize == inputSize && mPackedHiddenSize == hiddenSize &&
        mPackedGateSize == gateSize && mPackedDirection == direction) {
        params.packedWeightBytes = packedWeightBytes;
    }

    std::vector<std::pair<int, int>> inputFds = {
        HexagonBackend::getDevicePtr(x), HexagonBackend::getDevicePtr(w), HexagonBackend::getDevicePtr(r),
        HexagonBackend::getDevicePtr(b), HexagonBackend::getDevicePtr(h0), HexagonBackend::getDevicePtr(c0)};
    std::vector<Tensor*> commandInputs = {x, w, r, b, h0, c0};
    if (params.packedWeightBytes > 0) {
        inputFds.emplace_back(HexagonBackend::getDevicePtr(mPackedW.get()));
        inputFds.emplace_back(HexagonBackend::getDevicePtr(mPackedR.get()));
        commandInputs.emplace_back(mPackedW.get());
        commandInputs.emplace_back(mPackedR.get());
    }
    std::vector<std::pair<int, int>> outputFds;
    outputFds.reserve(outputs.size() + 1);
    std::vector<Tensor*> commandOutputs;
    commandOutputs.reserve(outputs.size() + 1);
    for (auto output : outputs) {
        if (output == nullptr) {
            return INPUT_DATA_ERROR;
        }
        if (output->getType().code != halide_type_float || HexagonBackend::getBytes(output) != bytes) {
            return NOT_SUPPORT;
        }
        outputFds.emplace_back(HexagonBackend::getDevicePtr(output));
        commandOutputs.emplace_back(output);
    }
    mScratch.reset(Tensor::createDevice<int8_t>({(int)scratchBytes}));
    if (!backend()->onAcquireBuffer(mScratch.get(), Backend::DYNAMIC)) {
        mScratch.reset();
        return OUT_OF_MEMORY;
    }
    outputFds.emplace_back(HexagonBackend::getDevicePtr(mScratch.get()));
    commandOutputs.emplace_back(mScratch.get());

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_LSTM, &params, sizeof(params),
                     inputFds, outputFds, commandInputs, commandOutputs);
    backend()->onReleaseBuffer(mScratch.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

bool HexagonLSTM::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonLSTM(bn, mHiddenSize);
    return true;
}

} // namespace MNN
