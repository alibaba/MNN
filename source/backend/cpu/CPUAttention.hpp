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
    bool mIsPrefill      = true;
    bool mIsFirstPrefill = true;
    bool mKVCache        = true;
    int bytes = 4;
    int mThreadNum = 1;;
    int eP, lP, hP, unit;
    int mNumHead, mKvNumHead, mHeadDim;
    std::shared_ptr<Tensor> mPackQ, mPackQKV;
    std::shared_ptr<KVCacheManager> mKVCacheManager = nullptr;
};

} // namespace MNN

#endif // CPUATTENTION_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
