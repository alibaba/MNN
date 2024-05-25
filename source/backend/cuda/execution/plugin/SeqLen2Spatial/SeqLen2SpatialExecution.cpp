//
//  SeqLen2SpatialExecution.cpp
//  MNN
//
//  Created by MNN on 2023/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "SeqLen2SpatialExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

SeqLen2SpatialExecution::SeqLen2SpatialExecution(Backend* backend) : Execution(backend) {
    // Nothing todo
}
ErrorCode SeqLen2SpatialExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Nothing todo
    return NO_ERROR;
}

ErrorCode SeqLen2SpatialExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SeqLen2SpatialExecution onExecute...");
#endif
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    int32_t const BS = inputs[1]->length(0);
    int32_t const HW = inputs[1]->length(1);
    int32_t const C = inputs[1]->length(2);
    int32_t const gridSize = BS * HW;

    bool isHalf = static_cast<CUDABackend*>(backend())->useFp16();
    void *input0 = (void *)inputs[1]->deviceId();
    void *input1 = (void *)inputs[2]->deviceId();
    void *input2 = (void *)inputs[3]->deviceId();
    void *output = (void *)outputs[0]->deviceId();

    launchSeqLen2SpatialKernel(input0, input1, input2, output, isHalf, gridSize, C);
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end SeqLen2SpatialExecution onExecute...");
#endif
    return NO_ERROR;
}


class SeqLen2SpatialCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new SeqLen2SpatialExecution(backend);
    }
};

CUDACreatorRegister<SeqLen2SpatialCreator> __SeqLen2SpatialExecution(OpType_SeqLen2Spatial);
} // namespace CUDA
} // namespace MNN
#endif