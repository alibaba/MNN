//
//  CPURoPE.hpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURoPE_hpp
#define CPURoPE_hpp

#include <MNN/Tensor.hpp>
#include "backend/cpu/CPULayerNorm.hpp"
#include "core/Execution.hpp"

namespace MNN {
class CPURoPE : public Execution {
public:
    CPURoPE(const Op* op, Backend* bn);
    virtual ~CPURoPE();
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    CPURoPE(Backend* bn);
    int mRopeCutHeadDim = 0;
    std::shared_ptr<CPULayerNorm::Resource> mQNorm;
    std::shared_ptr<CPULayerNorm::Resource> mKNorm;
    MemChunk mTmpQFloat;
    MemChunk mTmpKFloat;
};

} // namespace MNN
#endif /* CPURoPE_hpp */
