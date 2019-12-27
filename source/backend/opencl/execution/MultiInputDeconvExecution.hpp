//
//  MultiInputDeconvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MultiInputDeconvExecution_hpp
#define MultiInputDeconvExecution_hpp

#include "backend/opencl/execution/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class MultiInputDeconvExecution : public CommonExecution {
public:
    MultiInputDeconvExecution(const MNN::Op *op, Backend *backend);
    virtual ~MultiInputDeconvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mStrides;
    std::vector<int> mPaddings;
    std::vector<int> mDilations;
    std::shared_ptr<Tensor> mFilter;
};

} // namespace OpenCL
} // namespace MNN
#endif /* MultiInputDeconvExecution_hpp */
