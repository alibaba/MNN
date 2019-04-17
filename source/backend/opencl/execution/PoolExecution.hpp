//
//  PoolExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PoolExecution_hpp
#define PoolExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"
#include "core/OpenCLRunningUtils.hpp"
namespace MNN {
namespace OpenCL {

class PoolExecution : public Execution {
public:
    PoolExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~PoolExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    std::vector<uint32_t> poolLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

private:
    const Pool *mPoolParams;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    PoolType mPoolType;
    PoolPadType mPadType;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mKernels{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* PoolExecution_hpp */
