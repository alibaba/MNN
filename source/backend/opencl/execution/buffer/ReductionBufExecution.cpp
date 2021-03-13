//
//  ReductionBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/ReductionBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ReductionBufExecution::ReductionBufExecution(const MNN::Op* op, Backend* backend) : CommonExecution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionBufExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto reduct = op->main_as_ReductionParam();
    if (nullptr != reduct->dim()) {
        for (int i = 0; i < reduct->dim()->size(); ++i) {
            mAxis.push_back(reduct->dim()->data()[i]);
        }
    }
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            mReductType = 0;
            break;
        case ReductionType_MAXIMUM:
            mReductType = 1;
            break;
        case ReductionType_MINIMUM:
            mReductType = 2;
            break;
        case ReductionType_PROD:
            mReductType = 3;
            break;
        case ReductionType_SUM:
            mReductType = 4;
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    mOp = op;
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReductionBufExecution init !\n");
#endif
}

ErrorCode ReductionBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
    MNN_ASSERT(mAxis.size() == 1);
    MNN_ASSERT(mAxis[0] == 1);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto input = inputs[0];
    auto output = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    //N=outside H=axis W=inside C=1
    MNN_ASSERT(inputShape[3] == 1);

    mGlobalWorkSize = {static_cast<uint32_t>(inputShape[0]), static_cast<uint32_t>(inputShape[2])};
    mLocalWorkSize = {1, 1, 1};
    
    std::set<std::string> buildOption;
    switch (mReductType) {
        case 0:
            buildOption.emplace("-DOPERATE=num+in");
            buildOption.emplace("-DGET_AVG");
            break;
        case 1:
            buildOption.emplace("-DOPERATE=max(num,in)");
            break;
        case 2:
            buildOption.emplace("-DOPERATE=min(num,in)");
            break;
        case 3:
            buildOption.emplace("-DOPERATE=num*in");
            break;
        case 4:
            buildOption.emplace("-DOPERATE=num+in");
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    mReduct1DKernel = runtime->buildKernel("reduction_buf", "reduct_buf", buildOption);

    //printf("reduce axis:%d , %d %d %d %d, useLocal:%d\n", mAxis[0], inputShape[0], inputShape[1], inputShape[2], inputShape[3], mUseLocal);

    mUnits.resize(1);
    uint32_t idx = 0;

    mReduct1DKernel.setArg(idx++, mGlobalWorkSize[0]);
    mReduct1DKernel.setArg(idx++, mGlobalWorkSize[1]);
    mReduct1DKernel.setArg(idx++, openCLBuffer(input));
    mReduct1DKernel.setArg(idx++, openCLBuffer(output));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[0]));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[1]));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[2]));

    return NO_ERROR;
}

ErrorCode ReductionBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionBufExecution onExecute !\n");
#endif

    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        runKernel2D(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                               mOpenCLBackend->getOpenCLRuntime(), &event);
        int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us Reduct1D\n",costTime);
    #else
        runKernel2D(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    #endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReductionBufExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class ReductionBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~ReductionBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                 const MNN::Op *op, Backend *backend) const override {
        if (inputs[0]->getDimensionType() == Tensor::TENSORFLOW) {
            auto openCLBackend = static_cast<OpenCLBackend *>(backend);
            auto reduct = op->main_as_ReductionParam();
            if (nullptr == reduct->dim()) {
                return NULL;
            }
            if(reduct->dim()->size() != 1) {
                return NULL;
            }
            switch (op->main_as_ReductionParam()->operation()) {
                case ReductionType_MEAN:
                    break;
                case ReductionType_MAXIMUM:
                    break;
                case ReductionType_MINIMUM:
                    break;
                case ReductionType_PROD:
                    break;
                case ReductionType_SUM:
                    break;
                default:
                    return NULL;
                    break;
            }
            return new ReductionBufExecution(op, backend);
        }
        return NULL;
    }
};

OpenCLCreatorRegister<ReductionBufCreator> __reductionBuf_op(OpType_Reduction, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
