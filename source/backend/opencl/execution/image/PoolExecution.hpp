//
//  PoolExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PoolExecution_hpp
#define PoolExecution_hpp

#include "CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class PoolExecution : public CommonExecution {
public:
    PoolExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~PoolExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    std::vector<uint32_t> poolLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);
    int getLocalSize(int size, int maxGroupSize);

private:
    const Pool *mPoolParams;
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
#endif /* PoolExecution_hpp */
