//
//  MNNRvvAttention.hpp
//  MNN
//
//  Created by MNN on 2026/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_RVV_ATTENTION_HPP
#define MNN_RVV_ATTENTION_HPP

#include "backend/cpu/CPUAttention.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

class MNNRvvAttention : public CPUAttention {
public:
    MNNRvvAttention(Backend* backend, bool kvCache);
    ~MNNRvvAttention() override;
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

protected:
    bool tryExecuteFastPath(const int8_t* query, int8_t* output, int seqLen, int kvSeqLen, int paddingLength,
                            float qScale, float attentionScale, bool lowerTriangular, bool hasSinks, bool outputC4,
                            bool directC4Output) override;
    CPUAttention* createClone(Backend* backend) const override;

private:
    bool acquireDecodeScratch(int capacity);
    void releaseDecodeScratch();

    std::shared_ptr<Tensor> mDecodeScratch;
    int mDecodeScratchCapacity = 0;
};

Execution* MNNRvvCreateAttentionExecution(Backend* backend, bool kvCache);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

#endif // MNN_RVV_ATTENTION_HPP
