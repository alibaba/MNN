//
//  CPURNNSequenceGRU.hpp
//  MNN
//
//  Created by MNN on 2019/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURNNSequenceGRU_hpp
#define CPURNNSequenceGRU_hpp

#include "Execution.hpp"

namespace MNN {

class CPURNNSequenceGRU : public Execution {
public:
    CPURNNSequenceGRU(const Op *op, Backend *backend);
    virtual ~CPURNNSequenceGRU();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mKeepAllOutputs;
    bool mIsBidirectionalRNN;
    int mNumUnits;

    std::shared_ptr<Tensor> mHiddenState;
    std::shared_ptr<Tensor> mInputAndState;
    std::shared_ptr<Tensor> mGate;
    // forward weight and bias
    std::shared_ptr<Tensor> mFwGateWeight;
    std::shared_ptr<Tensor> mFwGateBias;
    std::shared_ptr<Tensor> mFwCandidateWeight;
    std::shared_ptr<Tensor> mFwCandidateBias;
    // backward weight and bias
    std::shared_ptr<Tensor> mBwGateWeight;
    std::shared_ptr<Tensor> mBwGateBias;
    std::shared_ptr<Tensor> mBwCandidateWeight;
    std::shared_ptr<Tensor> mBwCandidateBias;
};

} // namespace MNN

#endif /* CPURNNSequenceGRU_hpp */
