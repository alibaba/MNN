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
namespace MNN {

class CPURNNSequenceGRU : public Execution {
public:
    CPURNNSequenceGRU(const Op *op, Backend *backend);
    virtual ~CPURNNSequenceGRU();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    void runRNNStep(const float* input, const int inputLength, const bool linearBeforeReset,
                           std::shared_ptr<Tensor>& hiddenState, const int numUnits, Tensor* gateWeight, Tensor* gateBias,
                           Tensor* candidateWeight, Tensor* candidateBias, Tensor* recurrentBias,
                           std::shared_ptr<Tensor>& inputAndState, std::shared_ptr<Tensor>& gate,
               std::shared_ptr<Tensor>& resetHt);
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
};

} // namespace MNN

#endif /* CPURNNSequenceGRU_hpp */
