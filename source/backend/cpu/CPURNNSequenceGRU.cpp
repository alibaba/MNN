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
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// implement GRU cell function
// Ref: tensorflow/python/ops/rnn_cell_impl.py
void CPURNNSequenceGRU::runRNNStep(const uint8_t* input, const int inputLength, const bool linearBeforeReset,
                       std::shared_ptr<Tensor>& hiddenState, const int numUnits, Tensor* gateWeight, Tensor* gateBias,
                       Tensor* candidateWeight, Tensor* candidateBias, Tensor* recurrentBias,
                       std::shared_ptr<Tensor>& inputAndState, std::shared_ptr<Tensor>& gate,
                       std::shared_ptr<Tensor>& resetHt) {
    auto bn = static_cast<CPUBackend*>(backend());
    auto mulFunction = bn->functions()->MNNSelectBinaryFunctionForFloat(BinaryOpOperation_MUL);
    auto addFunction = bn->functions()->MNNSelectBinaryFunctionForFloat(BinaryOpOperation_ADD);
    auto subFunction = bn->functions()->MNNSelectBinaryFunctionForFloat(BinaryOpOperation_SUB);
    auto tanhFunction = bn->functions()->MNNSelectUnaryFunctionForFloat(UnaryOpOperation_TANH, bn->precisionMode());
    auto bytes = bn->functions()->bytes;
    auto sigmoidFunc = bn->functions()->MNNSelectUnaryFunctionForFloat(UnaryOpOperation_SIGMOID, bn->precisionMode());
    // gate is (z_t, r_t)
    auto inputAndStatePtr = inputAndState->host<uint8_t>();
    auto hiddenStatePtr   = hiddenState->host<uint8_t>();
    ::memcpy(inputAndStatePtr, input, inputLength * bytes);
    ::memcpy(inputAndStatePtr + inputLength * bytes, hiddenStatePtr, numUnits * bytes);
    inputAndState->setLength(1, inputLength + numUnits);

    // // [x_t, h_t-1] * [W_zr, R_zr]: (1, inputLength + numUnits) X (inputLength + numUnits, 2 * numUnits)
    mMatMulIU2U->execute(inputAndState->host<float>(), gateWeight->host<float>(), gate->host<float>(), gateBias->host<float>());

    recurrentBias->setLength(1, 2 * numUnits);
    addFunction(gate->host<float>(), gate->host<float>(), recurrentBias->host<float>(), 2*numUnits, -1);
    // (1, 2*numUnits)
    const int gateSize = gate->elementSize();
    auto gatePtr       = gate->host<uint8_t>();
    sigmoidFunc(gatePtr, gatePtr, gateSize);
    // reset gate, // r_t is the second segment
    auto rtPtr = gatePtr + numUnits * bytes;

    if (linearBeforeReset) {
        // calculate Rt (.) (Ht_1 * Rh + Rbh)
        auto recurrentHiddenBiasPtr = recurrentBias->host<uint8_t>() + 2 * numUnits * bytes;
        auto rhWeightPtr = candidateWeight->host<uint8_t>() + inputLength * numUnits * bytes;
        mMatMulU2U->execute(hiddenState->host<float>(), (float*)rhWeightPtr, resetHt->host<float>(), (float*)recurrentHiddenBiasPtr);
        mulFunction(resetHt->host<float>(), rtPtr, resetHt->host<float>(), numUnits, -1);

        // calculate Xt * Wh
        mMatMulI2U->execute((float*)input, candidateWeight->host<float>(), (float*)(inputAndStatePtr + (inputLength + numUnits) * bytes), nullptr);
        // sum 3 parts
        addFunction(resetHt->host<float>(), resetHt->host<float>(), inputAndStatePtr + (inputLength + numUnits) * bytes, numUnits, -1);
        addFunction(rtPtr, resetHt->host<float>(), candidateBias->host<float>(), numUnits, -1);

    } else {
        // r_t: (1, numUnits)
        auto resetGatePtr = inputAndStatePtr + inputLength * bytes;
        // h_t1(1, numUnits) = r_t(1, numUnits) * h_t-1_(1, numUnits)
        mulFunction(resetGatePtr, rtPtr, hiddenStatePtr, numUnits, -1);
        // deal with recurrent bias and linear_before_reset parameter
        auto recurrentBiasAddedPtr = inputAndStatePtr + (inputLength + numUnits) * bytes;
        auto recurrentHiddenBiasPtr = recurrentBias->host<float>() + 2 * numUnits * bytes;
        addFunction(recurrentBiasAddedPtr, recurrentHiddenBiasPtr, candidateBias->host<float>(), numUnits, -1);
        mMatMulI2U->execute(inputAndState->host<float>(), candidateWeight->host<float>(),  resetHt->host<float>(), nullptr);
        // reuse r_t memory as h_t'
        addFunction(rtPtr, resetHt->host<float>(), recurrentBiasAddedPtr, numUnits, -1);
    }
    // h = (1-g)*t+g*h = t + g*(h-t)
    tanhFunction(resetHt->host<float>(), rtPtr, numUnits);
    subFunction(hiddenStatePtr, hiddenStatePtr, resetHt->host<float>(), numUnits, -1);
    mulFunction(hiddenStatePtr, hiddenStatePtr, gatePtr, numUnits, -1);
    addFunction(hiddenStatePtr, hiddenStatePtr, resetHt->host<float>(), numUnits, -1);
    inputAndState->setLength(1, inputLength + 2 * numUnits);
}

CPURNNSequenceGRU::CPURNNSequenceGRU(const Op* op, Backend* backend) : MNN::Execution(backend) {
    auto rnnParam       = op->main_as_RNNParam();
    mKeepAllOutputs     = rnnParam->keepAllOutputs();
    mIsBidirectionalRNN = rnnParam->isBidirectionalRNN();
    mNumUnits           = rnnParam->numUnits();
    mlinearBeforeReset  = rnnParam->linearBeforeReset();
    mMatMulIU2U.reset(new CPUMatMul(backend, false, false, true, true));
    mMatMulU2U.reset(new CPUMatMul(backend, false, false, true, true));
    mMatMulI2U.reset(new CPUMatMul(backend, false, false, true, true));
}

CPURNNSequenceGRU::~CPURNNSequenceGRU() {
    mMatMulIU2U.reset();
    mMatMulU2U.reset();
    mMatMulI2U.reset();
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
    mInputAndState->setLength(1, inputLastDimSize + mNumUnits);
    auto code = mMatMulIU2U->onResize({mInputAndState.get(), inputs[1]}, {mGate.get()});
    if (NO_ERROR != code) {
        return code;
    }
    mInputAndState->setLength(1, inputLastDimSize + 2 * mNumUnits);

    if (mlinearBeforeReset) {
        std::shared_ptr<Tensor> rhWeight(Tensor::create<float>({mNumUnits, mNumUnits}));
        // unit, unit * unit -> unit
        code = mMatMulU2U->onResize({mHiddenState.get(), rhWeight.get()}, {mResetHt.get()});
        if (NO_ERROR != code) {
            return code;
        }
        std::shared_ptr<Tensor> XtWhTensor(Tensor::create<float>({1, mNumUnits}));
        std::shared_ptr<Tensor> inputTensor(Tensor::create<float>({1, inputLastDimSize}));
        std::shared_ptr<Tensor> wTensor(Tensor::create<float>({inputLastDimSize, mNumUnits}));
        code = mMatMulI2U->onResize({inputTensor.get(), wTensor.get()}, {XtWhTensor.get()});
    } else {
        std::shared_ptr<Tensor> A(Tensor::create<float>({1, mNumUnits + inputLastDimSize}));
        std::shared_ptr<Tensor> B(Tensor::create<float>({mNumUnits + inputLastDimSize, mNumUnits}));
        std::shared_ptr<Tensor> C(Tensor::create<float>({1, mNumUnits}));
        code = mMatMulI2U->onResize({A.get(), B.get()}, {C.get()});
    }
    if (NO_ERROR != code) {
        return code;
    }
    backend()->onReleaseBuffer(mHiddenState.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mInputAndState.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mResetHt.get(), Backend::DYNAMIC);

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
    auto cpuBn = static_cast<CPUBackend*>(backend());
    auto bytes = cpuBn->functions()->bytes;

    // fwGateWeight->printShape();// mFwGateWeight
    // fwGateBias->printShape();// mFwGateBias
    // fwCandidateWeight->printShape();// mFwCandidateWeight
    // fwCandidateBias->printShape();// mFwCandidateBias
    // fwRecurrentBias->printShape();// mFwRecurrentBias

    // firstly set the hidden state to zero
    auto const hiddenStatePtr   = mHiddenState->host<uint8_t>();
    const int hiddenStateDataSize = mHiddenState->elementSize() * bytes;

    auto input                    = inputs[0];  // shape :(seq_length, batch_size, input_size)
    auto output                   = outputs[0]; // shape :(seq_length, num_directions, batch_size, hidden_size)
    auto const inputPtr         = input->host<uint8_t>();
    auto const outputPtr        = output->host<uint8_t>();

    auto outputYhPtr = mKeepAllOutputs && outputSize > 1 ? outputs[1]->host<uint8_t>() : outputs[0]->host<uint8_t>();
    const int batchSize           = input->length(1);
    const int SequenceStride      = input->stride(0);
    const int inputSequenceLength = input->length(0);
    const int inputCodeLength     = input->length(2);
    // MNN_PRINT("inputSequenceLength:%d, batchSize:%d, inputCodeLength:%d, mNumUnits:%d, hiddenStateDataSize:%d\n", inputSequenceLength, batchSize, inputCodeLength, mNumUnits, hiddenStateDataSize);
    for (int b = 0; b < batchSize; ++b) { // swap order
        if (inputSize > 1 + forwardParamNumber * (mIsBidirectionalRNN + 1)) {
            auto source = inputs[inputSize - 1]->host<uint8_t>() + b * hiddenStateDataSize;
            ::memcpy(hiddenStatePtr, source, hiddenStateDataSize);
        } else {
            ::memset(hiddenStatePtr, 0, hiddenStateDataSize);
        }

        for (int i = 0; i < inputSequenceLength; ++i) {
            const int inputOffset = i * SequenceStride + b * inputCodeLength;
            runRNNStep(inputPtr + inputOffset * bytes, inputCodeLength, mlinearBeforeReset, mHiddenState, mNumUnits, fwGateWeight, fwGateBias,
                       fwCandidateWeight, fwCandidateBias, fwRecurrentBias, mInputAndState, mGate, mResetHt);

            if (mKeepAllOutputs) {
                ::memcpy(outputPtr + (i * output->stride(0) + b * mNumUnits) * bytes, hiddenStatePtr, hiddenStateDataSize);
            }
        }
        if ((mKeepAllOutputs && outputSize > 1) || !mKeepAllOutputs) {
            ::memcpy(outputYhPtr, hiddenStatePtr, hiddenStateDataSize);
            outputYhPtr += mNumUnits * bytes;
        }

    }

    // backward rnn
    if (mIsBidirectionalRNN) {
        auto outputYhPtr = mKeepAllOutputs && outputSize > 1 ? outputs[1]->host<uint8_t>() : outputs[0]->host<uint8_t>();
        outputYhPtr += batchSize * mNumUnits * bytes;
        // todo: modify the inputOffset
        MNN_ASSERT(11 <= inputs.size());
        auto bwGateWeight = inputs[6];
        auto bwGateBias = inputs[7];
        auto bwCandidateWeight = inputs[8];
        auto bwCandidateBias = inputs[9];
        auto bwRecurrentBias = inputs[10];

        auto outputBw            = outputs[0];
        auto const outputBwPtr = outputBw->host<uint8_t>();
        for (int b = 0; b < batchSize; ++b) {

            if (inputSize > 1 + forwardParamNumber * 2) {
                auto source = inputs[inputSize - 1]->host<uint8_t>() + (batchSize + b) * hiddenStateDataSize;
                ::memcpy(hiddenStatePtr, source, hiddenStateDataSize);
            } else {
                ::memset(hiddenStatePtr, 0, hiddenStateDataSize);
            }

            for (int i = inputSequenceLength - 1; i >= 0; i--) {
                const int inputOffset = i * SequenceStride + b * inputCodeLength;
                runRNNStep(inputPtr + inputOffset * bytes, inputCodeLength, mlinearBeforeReset, mHiddenState, mNumUnits, bwGateWeight, bwGateBias,
                           bwCandidateWeight, bwCandidateBias, bwRecurrentBias, mInputAndState, mGate, mResetHt);
                if (mKeepAllOutputs) {
                    ::memcpy(outputBwPtr + (i * outputBw->stride(0) + (batchSize + b) * mNumUnits) * bytes,
                             hiddenStatePtr, hiddenStateDataSize);
                }
            }
            if ((mKeepAllOutputs && outputSize > 1) || !mKeepAllOutputs) {
                ::memcpy(outputYhPtr, hiddenStatePtr, hiddenStateDataSize);
                outputYhPtr += mNumUnits * bytes;
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
