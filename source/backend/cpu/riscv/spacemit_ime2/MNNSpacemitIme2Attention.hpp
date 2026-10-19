//
//  MNNSpacemitIme2Attention.hpp
//  MNN
//
//  Created by MNN on 2026/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_SPACEMIT_IME2_ATTENTION_HPP
#define MNN_SPACEMIT_IME2_ATTENTION_HPP

#include "backend/cpu/riscv/rvv/MNNRvvAttention.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

class MNNSpacemitIme2Attention : public MNNRvvAttention {
public:
    MNNSpacemitIme2Attention(Backend* backend, bool kvCache);

protected:
    bool tryExecuteFastPath(const int8_t* query, int8_t* output, int seqLen, int kvSeqLen, int paddingLength,
                            float qScale, float attentionScale, bool lowerTriangular, bool hasSinks, bool outputC4,
                            bool directC4Output) override;
    CPUAttention* createClone(Backend* backend) const override;
};

Execution* MNNSpacemitIme2CreateAttentionExecution(Backend* backend, bool kvCache);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

#endif // MNN_SPACEMIT_IME2_ATTENTION_HPP
