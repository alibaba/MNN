//
//  UnaryExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef UnaryExecution_hpp
#define UnaryExecution_hpp

#include "Execution.hpp"

#include <vector>
#include "MNN_generated.h"
#include "core/OpenCLBackend.hpp"
#include "core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class UnaryExecution : public Execution {
public:
    UnaryExecution(const std::string &compute, Backend *backend);
    virtual ~UnaryExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const MNN::Op *mOp;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mAreadySetArg;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
};

} // namespace OpenCL
} // namespace MNN
#endif /* UnaryExecution_hpp */
