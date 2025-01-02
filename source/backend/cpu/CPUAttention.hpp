//
//  CPUAttention.hpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPUATTENTION_HPP
#define CPUATTENTION_HPP

#include <functional>
#include "core/Execution.hpp"
#include "core/OpCommonUtils.hpp"
#include "MNN/ErrorCode.hpp"
#include "KVCacheManager.hpp"

namespace MNN {

class CPUAttention : public Execution {
public:
    CPUAttention(Backend *backend, bool kv_cache);
    virtual ~CPUAttention();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    bool mKVCache        = true;
    bool mUseGemmInt8    = false;
    int bytes = 4;
    int mThreadNum = 1;;
    int eP, lP, hP, unit; // float matmul packing
    int eP8, lP8, hP8;    // GemmInt8 packing
    int mNumHead, mKvNumHead, mHeadDim;
    std::shared_ptr<Tensor> mPackQ, mPackQKV, mSumQ;
    std::shared_ptr<KVCacheManager> mKVCacheManager = nullptr;
    std::vector<float> mMinQ, mMaxQ, mQueryScale, mQueryZeroPoint;
    template <typename T> void pack_query(Tensor* query, char* pack_q, char* sum_q, int seq_len, int h, float q_scale);
    template <typename T> void unpack_QK(float * unpack_qk_dst, char * pack_qk_src, int seq_len, int kv_seq_len);
    KVMeta* mMeta;
};

} // namespace MNN

#endif // CPUATTENTION_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
