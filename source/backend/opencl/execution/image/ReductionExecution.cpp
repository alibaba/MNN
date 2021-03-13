//
//  ReductionExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/ReductionExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ReductionExecution::ReductionExecution(const MNN::Op* op, Backend* backend) : CommonExecution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionExecution init !\n");
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
    MNN_PRINT("end ReductionExecution init !\n");
#endif
}

ErrorCode ReductionExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
    MNN_ASSERT(mAxis.size() == 1);
    MNN_ASSERT(mAxis[0] == 1);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto input = inputs[0];
    auto output = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    //N=outside H=axis W=inside C=1
    MNN_ASSERT(inputShape[3] == 1);
    if(inputShape[1] >= 256) {
        mUseLocal = true;
    }
    if(!mUseLocal) {
        mGlobalWorkSize = {static_cast<uint32_t>(inputShape[0]), static_cast<uint32_t>(inputShape[2])};
        mLocalWorkSize = {1, 1, 1};
        
        switch (mReductType) {
            case 0:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_mean", {});
                break;
            case 1:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_max", {});
                break;
            case 2:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_min", {});
                break;
            case 3:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_mul", {});
                break;
            case 4:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_sum", {});
                break;
            default:
                MNN_ASSERT(false);
                break;
        }
    } else { //useLocal
        uint32_t global_x;
        int size = inputShape[1];
        if (size >= 1024) {
            global_x = 256;
        } else if(size >= 512) {
            global_x = 128;
        } else if (size >= 256) {
            global_x = 64;
        } else if (size >= 128) {
            global_x = 32;
        } else if (size >= 64) {
            global_x = 16;
        } else if (size >= 32) {
            global_x = 8;
        }
        mGlobalWorkSize = {global_x, static_cast<uint32_t>(inputShape[0]), static_cast<uint32_t>(inputShape[2])};
        mLocalWorkSize = {global_x, 1, 1 };
        
        switch (mReductType) {
            case 0:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_mean_local", {});
                break;
            case 1:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_max_local", {});
                break;
            case 2:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_min_local", {});
                break;
            case 3:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_mul_local", {});
                break;
            case 4:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_sum_local", {});
                break;
            default:
                MNN_ASSERT(false);
                break;
        }
    }
    //printf("reduce axis:%d , %d %d %d %d, useLocal:%d\n", mAxis[0], inputShape[0], inputShape[1], inputShape[2], inputShape[3], mUseLocal);

    mUnits.resize(1);
    uint32_t idx = 0;
    if(mUseLocal) {
        mReduct1DKernel.setArg(idx++, mGlobalWorkSize[1]);
        mReduct1DKernel.setArg(idx++, mGlobalWorkSize[2]);
    } else {
        mReduct1DKernel.setArg(idx++, mGlobalWorkSize[0]);
        mReduct1DKernel.setArg(idx++, mGlobalWorkSize[1]);
    }
    mReduct1DKernel.setArg(idx++, openCLImage(input));
    mReduct1DKernel.setArg(idx++, openCLImage(output));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[0]));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[1]));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[2]));

    return NO_ERROR;
}

ErrorCode ReductionExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionExecution onExecute !\n");
#endif

    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        if(mUseLocal) {
            run3DKernelDefault(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                               mOpenCLBackend->getOpenCLRuntime(), &event);
        } else {
            runKernel2D(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                               mOpenCLBackend->getOpenCLRuntime(), &event);
        }
        int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us Reduct1D\n",costTime);
    #else
    if(mUseLocal) {
        run3DKernelDefault(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    } else {
        runKernel2D(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    }
    #endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReductionExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class ReductionCreator : public OpenCLBackend::Creator {
public:
    virtual ~ReductionCreator() = default;
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
            return new ReductionExecution(op, backend);
        }
        return NULL;
    }
};

OpenCLCreatorRegister<ReductionCreator> __reduction_op(OpType_Reduction, IMAGE);
} // namespace OpenCL
} // namespace MNN
