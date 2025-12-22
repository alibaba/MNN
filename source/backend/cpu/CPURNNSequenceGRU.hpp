//
//  CPURNNSequenceGRU.hpp
//  MNN
//
//  Created by MNN on 2019/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURNNSequenceGRU_hpp
#define CPURNNSequenceGRU_hpp

#include "core/Execution.hpp"
#include "CPUMatMul.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
namespace MNN {

class CPURNNSequenceGRU : public Execution {
public:
    CPURNNSequenceGRU(const Op *op, Backend *backend);
    virtual ~CPURNNSequenceGRU();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    struct RNNFuntions {
        MNNBinaryExecute mulFunction;
        MNNBinaryExecute addFunction;
        MNNBinaryExecute subFunction;
        MNNUnaryExecute tanhFunction;
        MNNUnaryExecute sigmoidFunction;
        int bytes;
    };
private:
    void runRNNStep(const uint8_t* input, const int inputLength, const bool linearBeforeReset,
                           uint8_t* hiddenStateInput, const int numUnits, Tensor* gateWeight, Tensor* gateBias,
                           Tensor* candidateWeight, Tensor* candidateBias, Tensor* recurrentBias,
                           std::shared_ptr<Tensor>& inputAndState, std::shared_ptr<Tensor>& gate,
               std::shared_ptr<Tensor>& resetHt, uint8_t* hiddenStateOutput);
    bool mKeepAllOutputs;
    bool mIsBidirectionalRNN;
    bool mlinearBeforeReset;
    int mNumUnits;

    std::shared_ptr<Tensor> mHiddenState;
    std::shared_ptr<Tensor> mInputAndState;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mResetHt;
    
    // For inputLength + numUnit -> numUnit
    std::shared_ptr<CPUMatMul> mMatMulIU2U;
    // For numUnit -> numUnit
    std::shared_ptr<CPUMatMul> mMatMulU2U;
    // For inputLength -> numUnit
    std::shared_ptr<CPUMatMul> mMatMulI2U;
    RNNFuntions mRNNFunctions;
};

} // namespace MNN

#endif /* CPURNNSequenceGRU_hpp */
