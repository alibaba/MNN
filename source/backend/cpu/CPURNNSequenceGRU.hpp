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
    bool mlinearBeforeReset;
    int mNumUnits;

    std::shared_ptr<Tensor> mHiddenState;
    std::shared_ptr<Tensor> mInputAndState;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mResetHt;
};

} // namespace MNN

#endif /* CPURNNSequenceGRU_hpp */
