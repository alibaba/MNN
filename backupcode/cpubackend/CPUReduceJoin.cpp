//
//  CPUReduceJoin.cpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUReduceJoin.hpp"
#include "core/AutoStorage.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class CPUReduceJoinExecutor : public Execution {
public:
    CPUReduceJoinExecutor(Backend* bn, const MNN::Op* op) : Execution(bn) {
        auto rdjoin = op->main_as_ReduceJoin();
        mKeepDims   = rdjoin->keepDims();
        if (nullptr != rdjoin->separator()) {
            mSeperate = rdjoin->separator()->c_str();
        }
    }
    virtual ~CPUReduceJoinExecutor() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto inputTensor  = inputs[0];
        auto outputTensor = outputs[0];
        TensorUtils::clearHandleData(outputTensor);

        char* outputString = nullptr;
        auto inputSize     = inputTensor->size() / inputTensor->getType().bytes();
        auto inputStrings  = inputTensor->host<char*>();
        std::vector<int> lengths(inputSize);
        int totalLength = 0;
        for (int i = 0; i < inputSize; ++i) {
            int length = (int)::strlen(inputStrings[i]);
            lengths[i] = length;
            totalLength += length;
        }

        auto fullLength = totalLength + (inputSize - 1) * mSeperate.size() + 1;

        AutoStorage<char> outputStringBuffer((int)fullLength);
        outputString                 = outputStringBuffer.get();
        outputString[fullLength - 1] = 0;

        ::memcpy(outputString, inputStrings[0], lengths[0] * sizeof(char));

        int currentPos = lengths[0];
        for (int i = 1; i < inputSize; ++i) {
            ::memcpy(outputString + currentPos, mSeperate.c_str(), mSeperate.size() * sizeof(char));
            currentPos += mSeperate.size();

            ::memcpy(outputString + currentPos, inputStrings[i], lengths[i] * sizeof(char));
            currentPos += lengths[i];
        }
        outputTensor->host<char*>()[0] = ::strdup(outputString);
        return NO_ERROR;
    }

private:
    std::string mSeperate;
    bool mKeepDims;
};

Execution* CPUReduceJoinCreator::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                          const MNN::Op* op, Backend* backend) const {
    return new CPUReduceJoinExecutor(backend, op);
}

REGISTER_CPU_OP_CREATOR(CPUReduceJoinCreator, OpType_ReduceJoin);

} // namespace MNN
