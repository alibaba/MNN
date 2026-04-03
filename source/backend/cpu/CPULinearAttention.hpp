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
    void short_conv(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
private:
    std::string mAttentionType;
    int mHeadKDim;
    int mHeadVDim;
    int mNumKHeads;
    int mNumVHeads;
    bool mUseQKL2Norm;
    int mBytes;  // 4 for fp32, 2 for fp16 (Arm82)
    std::shared_ptr<StateCache> mStateCache;
    KVMeta* mMeta;
    std::string mPrefixCacheDir;

    // Temporary buffers for MNN-optimized path (per-Execution, DYNAMIC)
    std::shared_ptr<Tensor> mConvPadded;         // Padded conv input:  [B, D, convStateSize + L]
    std::shared_ptr<Tensor> mConvOut;            // Conv output after SiLU: [B, D, L]
    std::shared_ptr<Tensor> mThreadLocalBuf;     // Per-thread q/k/v/vpred/delta: [threadNum, 2*d_k + 3*d_v]
    std::shared_ptr<Tensor> mDecayBuf;           // Pre-computed exp(gate): [B*L*H]
    std::shared_ptr<Tensor> mConvFp32Buf;       // fp16 path: per-thread fp32 temp for Conv1D+SiLu
};

} // namespace MNN

#endif // CPULINEARATTENTION_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE