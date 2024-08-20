//
//  GroupNormBufExecution.hpp
//  MNN
//
//  Created by MNN on 2024/06/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef GroupNormBufExecution_hpp
#define GroupNormBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {

namespace OpenCL {
class GroupNormBufExecution : public CommonExecution {
public:
    GroupNormBufExecution(const MNN::Op* op, Backend *backend);
    virtual ~GroupNormBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    int getLocalSize(int size, int maxGroupSize);
private:
    OpenCLBackend *mOpenCLBackend;
    float mEpsilon{};
    int32_t mBSwish{};
    int32_t mGroup = 32;
    int32_t mBatch;
    std::unique_ptr<Tensor> mGammaTensor;
    std::unique_ptr<Tensor> mBetaTensor;
    std::shared_ptr<Tensor> mInputPlain, mOutputPlain;
    bool mHasGammaBeta = false;
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
};

} // namespace OPENCL
} // namespace MNN
#endif /* GroupNormBufExecution_hpp */
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
