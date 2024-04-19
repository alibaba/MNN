//
//  PoolBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef PoolBufExecution_hpp
#define PoolBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class PoolBufExecution : public CommonExecution {
public:
    PoolBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~PoolBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    int getLocalSize(int size, int maxGroupSize);

private:
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    ErrorCode SubgrouponResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
    const Pool *mPoolParams;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    PoolType mPoolType;
    PoolPadType mPadType;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mKernels{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* PoolBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
