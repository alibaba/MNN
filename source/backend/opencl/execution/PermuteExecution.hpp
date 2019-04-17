//
//  PermuteExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PermuteExecution_hpp
#define PermuteExecution_hpp

#include "CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class PermuteExecution : public CommonExecution {
public:
    PermuteExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~PermuteExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mDims;
    cl::Buffer *mTempInput = nullptr;
};

} // namespace OpenCL
} // namespace MNN
#endif /* PermuteExecution_hpp */
