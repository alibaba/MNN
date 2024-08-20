//
//  MultiInputDWDeconvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MultiInputDWDeconvExecution_hpp
#define MultiInputDWDeconvExecution_hpp

#include "CommonExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class MultiInputDWDeconvExecution : public CommonExecution {
public:
    MultiInputDWDeconvExecution(const MNN::Op *op, Backend *backend);
    virtual ~MultiInputDWDeconvExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mStrides;
    std::vector<int> mPaddings;
    std::vector<int> mDilations;
    std::shared_ptr<Tensor> mFilter;
    bool isRelu = false;
    bool isRelu6 = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* MultiInputDWDeconvExecution_hpp */
