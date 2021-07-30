//
//  CPURNNSequenceGRU.cpp
//  MNN
//
//  Created by MNN on 2019/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPURNNSequenceGRU.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/ConvOpt.h"
#include "math/Matrix.hpp"

namespace MNN {

static inline float sigmoid(float x) {
    return 1. / (1. + expf(-x));
}

static inline void ArrayProduct(float* C, float* A, float* B, const int length) {
    int numUnit4 = length >> 2;
    if (numUnit4 > 0) {
        MNNMatrixProd(C, A, B, numUnit4, 0, 0, 0, 1);
    }
    for (int i = numUnit4 << 2; i < length; i++) {
        C[i] = A[i] * B[i];
    }
    return;
}

static inline void ArrayAdd(float* C, float* A, float* B, const int length) {
    int numUnit4 = length >> 2;
    if (numUnit4 > 0) {
        MNNMatrixAdd(C, A, B, numUnit4, 0, 0, 0, 1);
    }
    for (int i = numUnit4 << 2; i < length; i++) {
        C[i] = A[i] + B[i];
    }
    return;
}

// implement GRU cell function
// Ref: tensorflow/python/ops/rnn_cell_impl.py
static void runRNNStep(const float* input, const int inputLength, const bool linearBeforeReset,
                       std::shared_ptr<Tensor>& hiddenState, const int numUnits, Tensor* gateWeight, Tensor* gateBias,
                       Tensor* candidateWeight, Tensor* candidateBias, Tensor* recurrentBias,
                       std::shared_ptr<Tensor>& inputAndState, std::shared_ptr<Tensor>& gate,
                       std::shared_ptr<Tensor>& resetHt) {
    // gate is (z_t, r_t)
    auto inputAndStatePtr = inputAndState->host<float>();
    auto hiddenStatePtr   = hiddenState->host<float>();
    ::memcpy(inputAndStatePtr, input, inputLength * sizeof(float));
    ::memcpy(inputAndStatePtr + inputLength, hiddenStatePtr, numUnits * sizeof(float));
    inputAndState->setLength(1, inputLength + numUnits);

    // to be fused
    Math::Matrix::multi(gate.get(), inputAndState.get(), gateWeight);  // [x_t, h_t-1] * [W_z; W_r]: (1, inputLength + numUnits) X (inputLength + numUnits, 2 * numUnits)
    Math::Matrix::add(gate.get(), gate.get(), gateBias);

    recurrentBias->setLength(1, 2 * numUnits);
    Math::Matrix::add(gate.get(), gate.get(), recurrentBias);
    const int gateSize = gate->elementSize();                          // (1, 2*numUnits)
    auto gatePtr       = gate->host<float>();
    for (int i = 0; i < gateSize; ++i) {
        gatePtr[i] = sigmoid(gatePtr[i]);
    }
    // reset gate
    auto rtPtr = gatePtr + numUnits;                              // r_t is the second segment
    auto resetGatePtr = inputAndStatePtr + inputLength;           // r_t: (1, numUnits)
    ArrayProduct(resetGatePtr, rtPtr, hiddenStatePtr, numUnits);  // h_t1(1, numUnits) = r_t(1, numUnits) * h_t-1_(1, numUnits)

    // deal with recurrent bias and linear_before_reset parameter
    auto recurrentBiasAddedPtr = inputAndStatePtr + inputLength + numUnits;
    auto recurrentHiddenBiasPtr = recurrentBias->host<float>() + 2 * numUnits;
    if (linearBeforeReset) {
        ArrayProduct(recurrentBiasAddedPtr, rtPtr, recurrentHiddenBiasPtr, numUnits); //2.  rt * h_t_1
        ArrayAdd(recurrentBiasAddedPtr, recurrentBiasAddedPtr, candidateBias->host<float>(), numUnits);
    } else {
        ArrayAdd(recurrentBiasAddedPtr, recurrentHiddenBiasPtr, candidateBias->host<float>(), numUnits);
    }

    Math::Matrix::multi(resetHt.get(), inputAndState.get(), candidateWeight); // [x_t, h_t1](1, inputLength + numUnits) * candidateWeight_(inputLength + numUnits, numUnits)
    // Math::Matrix::add(gate.get(), gate.get(), candidateBias.get());
    ArrayAdd(rtPtr, resetHt->host<float>(), recurrentBiasAddedPtr, numUnits); // reuse r_t memory as h_t

    for (int i = 0; i < numUnits; ++i) {
        hiddenStatePtr[i] =
            (1 - gatePtr[i]) * tanhf(rtPtr[i]) + gatePtr[i] * hiddenStatePtr[i];
    }

    inputAndState->setLength(1, inputLength + 2 * numUnits);
}

CPURNNSequenceGRU::CPURNNSequenceGRU(const Op* op, Backend* backend) : MNN::Execution(backend) {
    auto rnnParam       = op->main_as_RNNParam();
    mKeepAllOutputs     = rnnParam->keepAllOutputs();
    mIsBidirectionalRNN = rnnParam->isBidirectionalRNN();
    mNumUnits           = rnnParam->numUnits();
    mlinearBeforeReset  = rnnParam->linearBeforeReset();
    // MNN_PRINT("mKeepAllOutputs:%d, mNumUnits:%d, mlinearBeforeReset:%d", mKeepAllOutputs, mNumUnits, mlinearBeforeReset);
    // auto copyData = [=](std::shared_ptr<Tensor>& tensor, const Blob* src) {
    //     std::vector<int> shape;
    //     for (int i = 0; i < src->dims()->size(); ++i) {
    //         shape.push_back(src->dims()->data()[i]);
    //     }
    //     tensor.reset(Tensor::createDevice<float>(shape));
    //     backend->onAcquireBuffer(tensor.get(), Backend::STATIC);
    //     ::memcpy(tensor->host<float>(), src->float32s()->data(), src->float32s()->size() * sizeof(float));
    // };
    // copyData(mFwGateWeight, rnnParam->fwGateWeight());
    // copyData(mFwGateBias, rnnParam->fwGateBias());
    // copyData(mFwCandidateWeight, rnnParam->fwCandidateWeight());
    // copyData(mFwCandidateBias, rnnParam->fwCandidateBias());
    // copyData(mFwRecurrentBias, rnnParam->fwRecurrentBias());
    // MNN_ASSERT(mFwCandidateBias->length(0) == mNumUnits);
    // if (mIsBidirectionalRNN) {
    //     copyData(mBwGateWeight, rnnParam->bwGateWeight());
    //     copyData(mBwGateBias, rnnParam->bwGateBias());
    //     copyData(mBwCandidateWeight, rnnParam->bwCandidateWeight());
    //     copyData(mBwCandidateBias, rnnParam->bwCandidateBias());
    //     copyData(mBwRecurrentBias, rnnParam->bwRecurrentBias());
    // }
}

CPURNNSequenceGRU::~CPURNNSequenceGRU() {
    // backend()->onReleaseBuffer(mFwGateWeight.get(), Backend::STATIC);
    // backend()->onReleaseBuffer(mFwGateBias.get(), Backend::STATIC);
    // backend()->onReleaseBuffer(mFwCandidateWeight.get(), Backend::STATIC);
    // backend()->onReleaseBuffer(mFwCandidateBias.get(), Backend::STATIC);
    // if (mIsBidirectionalRNN) {
    //     backend()->onReleaseBuffer(mBwGateWeight.get(), Backend::STATIC);
    //     backend()->onReleaseBuffer(mBwGateBias.get(), Backend::STATIC);
    //     backend()->onReleaseBuffer(mBwCandidateWeight.get(), Backend::STATIC);
    //     backend()->onReleaseBuffer(mBwCandidateBias.get(), Backend::STATIC);
    // }

    backend()->onReleaseBuffer(mHiddenState.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mInputAndState.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mResetHt.get(), Backend::DYNAMIC);
}

ErrorCode CPURNNSequenceGRU::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 + 5 * (mIsBidirectionalRNN + 1) <= inputs.size());
    auto input                 = inputs[0];
    const int inputLastDimSize = input->length(2);
    mHiddenState.reset(Tensor::createDevice<float>(std::vector<int>{1, mNumUnits}));
    mInputAndState.reset(Tensor::createDevice<float>(std::vector<int>{1, inputLastDimSize + mNumUnits + mNumUnits}));
    mGate.reset(Tensor::createDevice<float>(std::vector<int>{1, 2 * mNumUnits}));
    mResetHt.reset(Tensor::createDevice<float>(std::vector<int>{1, mNumUnits}));

    backend()->onAcquireBuffer(mHiddenState.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mInputAndState.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mGate.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mResetHt.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPURNNSequenceGRU::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    auto inputSize = inputs.size();
    auto outputSize = outputs.size();
    const int forwardParamNumber = 5;
    MNN_ASSERT(inputSize >= 1 + forwardParamNumber * (mIsBidirectionalRNN + 1));
    auto fwGateWeight = inputs[1];
    auto fwGateBias = inputs[2];
    auto fwCandidateWeight = inputs[3];
    auto fwCandidateBias = inputs[4];
    auto fwRecurrentBias = inputs[5];

    // fwGateWeight->printShape();// mFwGateWeight
    // fwGateBias->printShape();// mFwGateBias
    // fwCandidateWeight->printShape();// mFwCandidateWeight
    // fwCandidateBias->printShape();// mFwCandidateBias
    // fwRecurrentBias->printShape();// mFwRecurrentBias

    // firstly set the hidden state to zero
    float* const hiddenStatePtr   = mHiddenState->host<float>();
    const int hiddenStateDataSize = mHiddenState->size();

    auto input                    = inputs[0];  // shape :(seq_length, batch_size, input_size)
    auto output                   = outputs[0]; // shape :(seq_length, num_directions, batch_size, hidden_size)
    float* const inputPtr         = input->host<float>();
    float* const outputPtr        = output->host<float>();

    float* outputYhPtr = mKeepAllOutputs && outputSize > 1 ? outputs[1]->host<float>() : outputs[0]->host<float>();
    const int batchSize           = input->length(1);
    const int SequenceStride      = input->stride(0);
    const int inputSequenceLength = input->length(0);
    const int inputCodeLength     = input->length(2);

    for (int b = 0; b < batchSize; ++b) { // swap order
        if (inputSize > 1 + forwardParamNumber * (mIsBidirectionalRNN + 1)) {
            auto source = inputs[inputSize - 1]->host<uint8_t>() + b * hiddenStateDataSize;
            ::memcpy(hiddenStatePtr, source, hiddenStateDataSize);
        } else {
            ::memset(hiddenStatePtr, 0, hiddenStateDataSize);
        }

        for (int i = 0; i < inputSequenceLength; ++i) {
            const int inputOffset = i * SequenceStride + b * inputCodeLength;
            runRNNStep(inputPtr + inputOffset, inputCodeLength, mlinearBeforeReset, mHiddenState, mNumUnits, fwGateWeight, fwGateBias,
                       fwCandidateWeight, fwCandidateBias, fwRecurrentBias, mInputAndState, mGate, mResetHt);
            if (mKeepAllOutputs) {
                ::memcpy(outputPtr + i * output->stride(0) + b * mNumUnits, hiddenStatePtr, hiddenStateDataSize);
            }
        }
        if ((mKeepAllOutputs && outputSize > 1) || !mKeepAllOutputs) {
            ::memcpy(outputYhPtr, hiddenStatePtr, hiddenStateDataSize);
            outputYhPtr += mNumUnits;
        }

    }

    // backward rnn
    if (mIsBidirectionalRNN) {
        float* outputYhPtr = mKeepAllOutputs && outputSize > 1 ? outputs[1]->host<float>() : outputs[0]->host<float>();
        outputYhPtr += batchSize * mNumUnits;
        // todo: modify the inputOffset
        MNN_ASSERT(11 <= inputs.size());
        auto bwGateWeight = inputs[6];
        auto bwGateBias = inputs[7];
        auto bwCandidateWeight = inputs[8];
        auto bwCandidateBias = inputs[9];
        auto bwRecurrentBias = inputs[10];

        auto outputBw            = outputs[0];
        float* const outputBwPtr = outputBw->host<float>();
        for (int b = 0; b < batchSize; ++b) {

            if (inputSize > 1 + forwardParamNumber * 2) {
                auto source = inputs[inputSize - 1]->host<uint8_t>() + (batchSize + b) * hiddenStateDataSize;
                ::memcpy(hiddenStatePtr, source, hiddenStateDataSize);
            } else {
                ::memset(hiddenStatePtr, 0, hiddenStateDataSize);
            }

            for (int i = inputSequenceLength - 1; i >= 0; i--) {
                const int inputOffset = i * SequenceStride + b * inputCodeLength;
                runRNNStep(inputPtr + inputOffset, inputCodeLength, mlinearBeforeReset, mHiddenState, mNumUnits, bwGateWeight, bwGateBias,
                           bwCandidateWeight, bwCandidateBias, bwRecurrentBias, mInputAndState, mGate, mResetHt);
                if (mKeepAllOutputs) {
                    ::memcpy(outputBwPtr + (inputSequenceLength - 1 - i) * outputBw->stride(0) + (batchSize + b) * mNumUnits,
                             hiddenStatePtr, hiddenStateDataSize);
                }
            }
            if ((mKeepAllOutputs && outputSize > 1) || !mKeepAllOutputs) {
                ::memcpy(outputYhPtr, hiddenStatePtr, hiddenStateDataSize);
                outputYhPtr += mNumUnits;
            }
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
