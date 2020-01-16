//
//  PoolGradExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include "backend/opencl/execution/PoolGradExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

PoolGradExecution::PoolGradExecution(const MNN::Op *op, Backend *backend)
: CommonExecution(backend) {
    auto pool = op->main_as_Pool();
    mType = pool->type();
    mKernels = std::vector<int>({pool->kernelY(), pool->kernelX()});
    mStrides = std::vector<int>({pool->strideY(), pool->strideX()});
}

PoolGradExecution::~PoolGradExecution() {
    // do nothing
}

ErrorCode PoolGradExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(mType == PoolType_MAXPOOL || mType == PoolType_AVEPOOL);
    mUnits.clear();
    mUnits.resize(1);

    auto shape = tensorShapeFormat(inputs[0]);
    auto poolShape = tensorShapeFormat(inputs[1]);
    uint32_t imageHeight = shape[0] * shape[1];
    uint32_t imageWidth = shape[2] * UP_DIV(shape[3], 4);

    auto runTime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    cl::Kernel kernel;
    int idx = 0;
    if (mType == PoolType_MAXPOOL) {
        kernel = runTime->buildKernel("pool_grad", "maxpool_grad", {});
        kernel.setArg(idx++, openCLImage(inputs[0]));
        kernel.setArg(idx++, openCLImage(inputs[1]));
        kernel.setArg(idx++, openCLImage(inputs[2]));
        kernel.setArg(idx++, openCLImage(outputs[0]));
    } else {
        kernel = runTime->buildKernel("pool_grad", "avepool_grad", {});
        kernel.setArg(idx++, openCLImage(inputs[2]));
        kernel.setArg(idx++, openCLImage(outputs[0]));
    }
    {
        int _shape[] = {shape[1], shape[2]};
        int _poolShape[] = {poolShape[1], poolShape[2]};
        int kernelSize[] = {mKernels[0], mKernels[1]};
        int stride[] = {mStrides[0], mStrides[1]};
        kernel.setArg(idx++, _shape);
        kernel.setArg(idx++, _poolShape);
        kernel.setArg(idx++, kernelSize);
        kernel.setArg(idx++, stride);
    }
    mUnits[0].kernel = kernel;
    mUnits[0].localWorkSize = cl::NullRange;
    mUnits[0].globalWorkSize = {imageHeight, imageWidth};

    return NO_ERROR;
}

class PoolGradCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new PoolGradExecution(op, backend);
    }
};

OpenCLCreatorRegister<PoolGradCreator> __Pool_grad_op(OpType_PoolGrad);

}
}
