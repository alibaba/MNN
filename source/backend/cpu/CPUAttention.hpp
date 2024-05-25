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

namespace MNN {

class CPUAttentionImpl {
public:
    CPUAttentionImpl(Backend *backend, bool kv_cache) : mBackend(backend), mKVCache(kv_cache) {}
    ~CPUAttentionImpl() = default;
    ErrorCode onResize(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onExecute(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
private:
    void allocKVCache();
    void reallocKVCache();
    Backend* backend() { return mBackend; }
private:
    Backend* mBackend;
    bool mKVCache;
    float mScale;
    const int mExpandChunk = 64;
    int mThreadNum = 1;
    bool mIsDecode = false;
    int mPastLength = 0, mMaxLength = 0;
    std::shared_ptr<Tensor> mPastKey, mPastValue, mTempQK;
    std::shared_ptr<Tensor> mPackQ, mPackQKV;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0;
    int eP, lP, hP, bytes;
    std::function<void(int)> mFunction, mPrefill, mDecode;
};

class CPUAttention : public Execution {
public:
    CPUAttention(Backend *backend, bool kv_cache);
    CPUAttention(std::shared_ptr<CPUAttentionImpl> impl, Backend *backend);
    virtual ~CPUAttention() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    std::shared_ptr<CPUAttentionImpl> mImpl;
};
} // namespace MNN

#endif // CPUATTENTION_HPP
#endif
