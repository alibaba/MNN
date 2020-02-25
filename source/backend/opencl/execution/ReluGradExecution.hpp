//
//  ReluGradExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ReluGradExecution_hpp
#define ReluGradExecution_hpp

#include <string>
#include "backend/opencl/execution/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class ReluGradExecution : public CommonExecution {
public:
    ReluGradExecution(const MNN::Op *op, Backend *backend);
    virtual ~ReluGradExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::string mKernelName;

};
}
}

#endif /* ReluGradExecution_hpp */
