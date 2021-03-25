//
//  CommonExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

CommonExecution::CommonExecution(Backend *backend) : Execution(backend) {
}
ErrorCode CommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int idx = 0;
#endif
    auto res = CL_SUCCESS;
    for (auto &unit : mUnits) {
        bool lws_null = true;
        for (size_t i = 0; i < unit.globalWorkSize.dimensions(); ++i) {
            unit.globalWorkSize.get()[i] = ROUND_UP(unit.globalWorkSize.get()[i], std::max((size_t)1, unit.localWorkSize.get()[i]));
            if(unit.localWorkSize.get()[i] != 0) {
                lws_null = false;
            }
        }
        
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        if(lws_null == true) {
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel,
                                                        cl::NullRange,
                                                        unit.globalWorkSize,
                                                        cl::NullRange,
                                                        nullptr,
                                                        &event);
        } else {
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel,
                                                        cl::NullRange,
                                                        unit.globalWorkSize,
                                                        unit.localWorkSize,
                                                        nullptr,
                                                        &event);
        }
        
        int costTime = (int)runtime->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us %s%d\n",costTime, EnumNameOpType(mOp->type()), idx++);
    #else
        if(lws_null == true) {
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel,
                                                        cl::NullRange,
                                                        unit.globalWorkSize,
                                                        cl::NullRange);
        } else {
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel,
                                                        cl::NullRange,
                                                        unit.globalWorkSize,
                                                        unit.localWorkSize);
        }
    #endif
        MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));
    }
    return NO_ERROR;
}
} // namespace OpenCL
}; // namespace MNN
