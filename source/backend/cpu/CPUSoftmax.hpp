//
//  CPUSoftmax.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSoftmax_hpp
#define CPUSoftmax_hpp

#include "core/Execution.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class CPUSoftmax : public Execution {
public:
    CPUSoftmax(Backend *b, int axis);
    virtual ~CPUSoftmax() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    static Execution* create(const MNN::Op *op, Backend *backend);

private:
    int _softmaxCommon(const uint8_t* srcData, uint8_t* dstData);

    int mAxis;
    Tensor mStorage;
    bool mNeedUnpackC4;
    MemChunk mTmpInput;
    MemChunk mTmpOutput;

    int mInside;
    int mOutside;
    int mChannel;

    std::shared_ptr<QuantAttr> mInQuantAttr;
    std::shared_ptr<QuantAttr> mOutQuantAttr;

    int mLowOrInt8;
};
} // namespace MNN

#endif /* CPUSoftmax_hpp */
