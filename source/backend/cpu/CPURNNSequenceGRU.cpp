//
//  CPURNNSequenceGRU.cpp
//  MNN
//
//  Created by MNN on 2019/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPURNNSequenceGRU.hpp"
#include <math.h>
#include "CPUBackend.hpp"
#include "ConvOpt.h"
#include "Matrix.hpp"

namespace MNN {

static inline float sigmoid(float x) {
    return 1. / (1. + expf(-x));
}

// implement GRU cell function
// Ref: tensorflow/python/ops/rnn_cell_impl.py
static void runRNNStep(const float* input, const int inputLength, std::shared_ptr<Tensor>& hiddenState,
                       const int numUnits, std::shared_ptr<Tensor>& gateWeight, std::shared_ptr<Tensor>& gateBias,
                       std::shared_ptr<Tensor>& candidateWeight, std::shared_ptr<Tensor>& candidateBias,
                       std::shared_ptr<Tensor>& inputAndState, std::shared_ptr<Tensor>& gate) {
    // gate is (r_t, z_t)
    auto inputAndStatePtr = inputAndState->host<float>();
    auto hiddenStatePtr   = hiddenState->host<float>();
    ::memcpy(inputAndStatePtr, input, inputLength * sizeof(float));
    ::memcpy(inputAndStatePtr + inputLength, hiddenStatePtr, numUnits * sizeof(float));
    Math::Matrix::multi(gate.get(), inputAndState.get(), gateWeight.get());
    Math::Matrix::add(gate.get(), gate.get(), gateBias.get());
    const int gateSize = gate->elementSize();
    auto gatePtr       = gate->host<float>();
    for (int i = 0; i < gateSize; ++i) {
        gatePtr[i] = sigmoid(gatePtr[i]);
    }

    {
        // reset gate
        auto resetGatePtr = inputAndStatePtr + inputLength;
        int k             = 0;
        auto numUnitC4    = numUnits / 4;
        if (numUnitC4 > 0) {
            MNNMatrixProd(resetGatePtr, gatePtr, hiddenStatePtr, numUnitC4, 0, 0, 0, 1);
            k = numUnitC4 * 4;
        }
        for (; k < numUnits; ++k) {
            resetGatePtr[k] = gatePtr[k] * hiddenStatePtr[k];
        }
    }

    // use r_t to apply Matrix multi and add
    gate->setLength(1, numUnits);
    Math::Matrix::multi(gate.get(), inputAndState.get(), candidateWeight.get());
    Math::Matrix::add(gate.get(), gate.get(), candidateBias.get());

    for (int i = 0; i < numUnits; ++i) {
        hiddenStatePtr[i] =
            gatePtr[numUnits + i] * hiddenStatePtr[i] + (1.0 - gatePtr[numUnits + i]) * tanhf(gatePtr[i]);
    }
    // reset gate shape fot the next iteration
    gate->setLength(1, 2 * numUnits);
}

CPURNNSequenceGRU::CPURNNSequenceGRU(const Op* op, Backend* backend) : MNN::Execution(backend) {
    auto rnnParam       = op->main_as_RNNParam();
    mKeepAllOutputs     = rnnParam->keepAllOutputs();
    mIsBidirectionalRNN = rnnParam->isBidirectionalRNN();
    mNumUnits           = rnnParam->numUnits();

    auto copyData = [=](std::shared_ptr<Tensor>& tensor, const Blob* src) {
        std::vector<int> shape;
        for (int i = 0; i < src->dims()->size(); ++i) {
            shape.push_back(src->dims()->data()[i]);
        }
        tensor.reset(Tensor::createDevice<float>(shape));
        backend->onAcquireBuffer(tensor.get(), Backend::STATIC);
        ::memcpy(tensor->host<float>(), src->float32s()->data(), src->float32s()->size() * sizeof(float));
    };
    copyData(mFwGateWeight, rnnParam->fwGateWeight());
    copyData(mFwGateBias, rnnParam->fwGateBias());
    copyData(mFwCandidateWeight, rnnParam->fwCandidateWeight());
    copyData(mFwCandidateBias, rnnParam->fwCandidateBias());
    MNN_ASSERT(mFwCandidateBias->length(0) == mNumUnits);
    if (mIsBidirectionalRNN) {
        copyData(mBwGateWeight, rnnParam->bwGateWeight());
        copyData(mBwGateBias, rnnParam->bwGateBias());
        copyData(mBwCandidateWeight, rnnParam->bwCandidateWeight());
        copyData(mBwCandidateBias, rnnParam->bwCandidateBias());
    }
}

CPURNNSequenceGRU::~CPURNNSequenceGRU() {
    backend()->onReleaseBuffer(mFwGateWeight.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mFwGateBias.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mFwCandidateWeight.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mFwCandidateBias.get(), Backend::STATIC);
    if (mIsBidirectionalRNN) {
        backend()->onReleaseBuffer(mBwGateWeight.get(), Backend::STATIC);
        backend()->onReleaseBuffer(mBwGateBias.get(), Backend::STATIC);
        backend()->onReleaseBuffer(mBwCandidateWeight.get(), Backend::STATIC);
        backend()->onReleaseBuffer(mBwCandidateBias.get(), Backend::STATIC);
    }
}

ErrorCode CPURNNSequenceGRU::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input                 = inputs[0];
    const int inputLastDimSize = input->length(2);
    mHiddenState.reset(Tensor::createDevice<float>(std::vector<int>{1, mNumUnits}));
    mInputAndState.reset(Tensor::createDevice<float>(std::vector<int>{1, inputLastDimSize + mNumUnits}));
    mGate.reset(Tensor::createDevice<float>(std::vector<int>{1, 2 * mNumUnits}));

    backend()->onAcquireBuffer(mHiddenState.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mInputAndState.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mGate.get(), Backend::DYNAMIC);

    backend()->onReleaseBuffer(mHiddenState.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mInputAndState.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPURNNSequenceGRU::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // firstly set the hidden state to zero
    float* const hiddenStatePtr   = mHiddenState->host<float>();
    const int hiddenStateDataSize = mHiddenState->size();
    ::memset(hiddenStatePtr, 0, hiddenStateDataSize);

    auto input                    = inputs[0];
    auto output                   = outputs[0];
    float* const inputPtr         = input->host<float>();
    float* const outputPtr        = output->host<float>();
    const int batchSize           = input->length(0);
    const int batchStride         = input->stride(0);
    const int inputSequenceLength = input->length(1);
    const int inputCodeLength     = input->length(2);

    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < inputSequenceLength; ++i) {
            const int inputOffset = b * batchStride + i * inputCodeLength;
            runRNNStep(inputPtr + inputOffset, inputCodeLength, mHiddenState, mNumUnits, mFwGateWeight, mFwGateBias,
                       mFwCandidateWeight, mFwCandidateBias, mInputAndState, mGate);
            if (mKeepAllOutputs) {
                ::memcpy(outputPtr + b * output->stride(0) + i * mNumUnits, hiddenStatePtr, hiddenStateDataSize);
            }
        }
    }

    if (!mKeepAllOutputs) {
        ::memcpy(outputPtr, hiddenStatePtr, hiddenStateDataSize);
    }
    // backward rnn
    if (mIsBidirectionalRNN) {
        ::memset(hiddenStatePtr, 0, hiddenStateDataSize);
        auto outputBw            = outputs[1];
        float* const outputBwPtr = outputBw->host<float>();
        for (int b = 0; b < batchSize; ++b) {
            for (int i = inputSequenceLength - 1; i >= 0; i--) {
                const int inputOffset = b * batchStride + i * inputCodeLength;
                runRNNStep(inputPtr + inputOffset, inputCodeLength, mHiddenState, mNumUnits, mBwGateWeight, mBwGateBias,
                           mBwCandidateWeight, mBwCandidateBias, mInputAndState, mGate);
                if (mKeepAllOutputs) {
                    ::memcpy(outputBwPtr + b * outputBw->stride(0) + (inputSequenceLength - 1 - i) * mNumUnits,
                             hiddenStatePtr, hiddenStateDataSize);
                }
            }
        }

        if (!mKeepAllOutputs) {
            ::memcpy(outputBwPtr, hiddenStatePtr, hiddenStateDataSize);
        }
    }

    return NO_ERROR;
}

class CPURNNSequenceGRUCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPURNNSequenceGRU(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPURNNSequenceGRUCreator, OpType_RNNSequenceGRU);

} // namespace MNN
