//
//  SoftmaxGradExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SoftmaxGradExecution_hpp
#define SoftmaxGradExecution_hpp

#include "backend/opencl/execution/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class SoftmaxGradExecution : public CommonExecution {
public:
    SoftmaxGradExecution(Backend *backend, int axis);
    virtual ~SoftmaxGradExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
};
}
}

#endif /* SoftmaxGradExecution_hpp */
