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


class CPUAttention : public Execution {
public:
    CPUAttention(Backend *backend, bool kv_cache);
    virtual ~CPUAttention() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    struct Resource {
        std::shared_ptr<Tensor> mPastKey;
        std::shared_ptr<Tensor> mPastValue;
        float mScale;
        const int mExpandChunk = 64;
        int mPastLength = 0, mMaxLength = 0;
        int mNumHead = 0, mKvNumHead = 0, mHeadDim = 0, mValueH = 0;
    };
private:
    void allocKVCache();
    void reallocKVCache();
    bool mIsDecode = false;
    bool mKVCache;
    int mThreadNum = 1;
    std::shared_ptr<Resource> mResource;
    std::shared_ptr<Tensor> mTempQK, mPackQ, mPackQKV;
    int eP, lP, hP, bytes;
    std::function<void(int)> mFunction, mPrefill, mDecode;
};
} // namespace MNN

#endif // CPUATTENTION_HPP
#endif
