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
        std::shared_ptr<Tensor> mPastKey;               // numhead, [maxlen/eP, headdim, eP]
        std::shared_ptr<Tensor> mPastValue;             // numhead, [headdim/eP, maxlen, eP]
        std::shared_ptr<Tensor> mDequantKeyScale;       // numhead, [maxlen/eP, 1, eP]
        std::shared_ptr<Tensor> mDequantKeyZeroPoint;   // numhead, [maxlen/eP, 1, eP]
        int mPastLength = 0, mMaxLength = 0;
        const int mExpandChunk = 64;
        int mNumHead = 0, mKvNumHead = 0, mHeadDim = 0;
    };
private:
    void allocKVCache(int kv_seq_len, bool quantK, bool quantV);
    void reallocKVCache(int kv_seq_len, bool quantK, bool quantV);
    bool mIsPrefill = true;
    bool mIsFirstPrefill = true;
    bool mKVCache;
    int mThreadNum = 1;
    std::shared_ptr<Resource> mResource;
    std::shared_ptr<Tensor> mPackQ, mPackQKV;
    int eP, lP, hP, bytes, unit;
};
} // namespace MNN

#endif // CPUATTENTION_HPP
#endif
