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
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int idx = 0;
#endif
    
    for (auto &unit : mUnits) {
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        auto errorCode = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel,
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize,
                                                    nullptr,
                                                    &event);
        
        int costTime = (int)runtime->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us %s%d\n",costTime, EnumNameOpType(mOp->type()), idx++);
    #else
        auto errorCode = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel,
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize);
    #endif
        MNN_CHECK_CL_SUCCESS(errorCode);
    }
    return NO_ERROR;
}
} // namespace OpenCL
}; // namespace MNN
