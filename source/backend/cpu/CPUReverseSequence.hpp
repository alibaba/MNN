//
//  CPUReverseSequence.hpp
//  MNN
//
//  Created by MNN on 2019/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

class CPUReverseSequence : public Execution {
public:
    CPUReverseSequence(Backend *b, int seqDim, int batchDim) : Execution(b) {
        mSeqDim   = seqDim;
        mBatchDim = batchDim;
        mValid    = mSeqDim != mBatchDim;
    }
    virtual ~CPUReverseSequence() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mSeqDim;
    int mBatchDim;

    int mInsideStride;
    int mOutsideSize;
    int mOutSideStride;
    int mMidSize;
    int mMidStride;
};

} // namespace MNN
