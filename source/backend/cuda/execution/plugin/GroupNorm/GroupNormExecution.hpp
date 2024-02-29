//
//  GroupNormExecution.hpp
//  MNN
//
//  Created by MNN on 2023/09/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef GroupNormExecution_hpp
#define GroupNormExecution_hpp

#include "groupNormKernel.cuh"
#include <cmath>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Macro.h"
#include <MNN/MNNDefine.h>

namespace MNN {

namespace CUDA {
class GroupNormExecution : public Execution {
public:
    GroupNormExecution(const MNN::Op* op, Backend *backend);
    virtual ~GroupNormExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    size_t getWorkspaceSizeInBytes() const;
    void* mWorkSpacePtr;

    float mEpsilon{};
    int32_t mBSwish{};
    int32_t mGroup = 32;
    int32_t mBatch;
    GroupNormNHWCParams mParams;
    std::unique_ptr<Tensor> mGammaTensor;
    std::unique_ptr<Tensor> mBetaTensor;
    void *mDeviceGamma = nullptr;
    void *mDeviceBeta = nullptr;
};

} // namespace CUDA
} // namespace MNN
#endif /* GroupNormExecution_hpp */
#endif