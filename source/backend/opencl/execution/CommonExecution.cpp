//
//  CommonExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

CommonExecution::CommonExecution(Backend *backend) : Execution(backend) {
}
ErrorCode CommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    for (auto &unit : mUnits) {
        auto errorCode = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel, cl::NullRange, unit.globalWorkSize,
                                                     unit.localWorkSize);
        MNN_CHECK_CL_SUCCESS(errorCode);
    }
    return NO_ERROR;
}
} // namespace OpenCL
}; // namespace MNN
