//
//  CPULinearAttention.hpp
//  MNN
//
//  Created by MNN on 2026/02/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPULINEARATTENTION_HPP
#define CPULINEARATTENTION_HPP

#include <functional>
#include "core/Execution.hpp"
#include "core/OpCommonUtils.hpp"
#include "CPUKVCacheManager.hpp"
#include "MNN/ErrorCode.hpp"

namespace MNN {

struct StateCache {
    std::shared_ptr<Tensor> mConvState;      // Conv1D padding state: [B, D, kernel_size - 1]
    std::shared_ptr<Tensor> mRecurrentState; // Gated Delta Rule recurrent state S: [B, H, d_k, d_v]
};

class CPULinearAttention : public Execution {
public:
    CPULinearAttention(Backend *backend, const MNN::Op* op);
    virtual ~CPULinearAttention();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void gated_delta_rule_ref(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    void gated_delta_rule_mnn(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
private:
    std::string mAttentionType;
    int mHeadKDim;
    int mHeadVDim;
    int mNumKHeads;
    int mNumVHeads;
    bool mUseQKL2Norm;
    std::shared_ptr<StateCache> mStateCache;

    // Temporary buffers for MNN-optimized path (per-Execution, DYNAMIC)
    std::shared_ptr<Tensor> mConvPadded;     // Padded conv input:  [B, D, convStateSize + L]
    std::shared_ptr<Tensor> mConvOut;        // Conv output after SiLU: [B, D, L]
    std::shared_ptr<Tensor> mTempVPred;      // Temp for v_pred: [d_v]
    std::shared_ptr<Tensor> mTempDelta;      // Temp for delta:  [d_v]
};

} // namespace MNN

#endif // CPULINEARATTENTION_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE