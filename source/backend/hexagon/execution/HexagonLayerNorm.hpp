//
//  HexagonLayerNorm.hpp
//  MNN
//
//  Created by MNN on 2025/04/28
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef HexagonLayerNorm_hpp
#define HexagonLayerNorm_hpp

#include "core/BufferAllocator.hpp"
#include "HexagonExecution.hpp"

namespace MNN {

class HexagonLayerNorm : public HexagonExecution {
public:
    struct Resource {
        int mGroup = 1;
        float mEpsilon = 0.001;
        MemChunk mGamma;
        MemChunk mBeta;
        bool mIniGammaBeta = false;
        bool mRMSNorm = false;
        bool mBetaZero = true;
        int mAxis = 0;
        BufferAllocator* mAllocator = nullptr;

        ~Resource() {
            if (mGamma.first != nullptr) mAllocator->free(mGamma);
            if (mBeta.first != nullptr) mAllocator->free(mBeta);
        }
    };

    HexagonLayerNorm(std::shared_ptr<Resource> res, Backend* backend);
    virtual ~HexagonLayerNorm();

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonLayerNorm* create(Backend* backend, const MNN::Op* op);
    static std::shared_ptr<Resource> makeResource(Backend* backend, const MNN::Op* op);

private:
    ErrorCode onBuildCmd(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs,
                         std::vector<HexagonCommand>& dst) override;

    std::shared_ptr<Resource> mResource;
    int mInnerSize = 1;
    int mOutterSize = 1;

    BufferAllocator* mAllocator = nullptr;
};

} // namespace MNN

#endif /* HexagonLayerNorm_hpp */
