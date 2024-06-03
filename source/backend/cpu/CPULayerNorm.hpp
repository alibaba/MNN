//
//  CPULayerNorm.hpp
//  MNN
//
//  Created by MNN on 2023/07/11
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPULayerNorm_hpp
#define CPULayerNorm_hpp

#include "core/Execution.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"
namespace MNN {
class CPULayerNorm : public Execution {
public:
    struct Resource {
        int mGroup = 1;
        float mEpsilon = 0.001;
        std::unique_ptr<Tensor> mGamma;
        std::unique_ptr<Tensor> mBeta;
        bool mIniGammaBeta = false;
        bool mRMSNorm = false;
        int mAxis = 0;
    };
    CPULayerNorm(std::shared_ptr<Resource> res, Backend* backend);
    virtual ~CPULayerNorm();

    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static std::shared_ptr<Resource> makeResource(const MNN::Op* op, Backend* backend);
private:
    std::shared_ptr<Resource> mResource;
    int mInnerSize = 1;
    int mOutterSize = 1;
    MemChunk mTmpInputFloat;
    MemChunk mTmpOutputFloat;
};
} // namespace MNN
#endif /* CPULayerNorm_hpp */
