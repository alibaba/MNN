//
//  MultiInputDeconvExecution.hpp
//  MNN
//
//  Created by MNN on 2023/04/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MultiInputDeconvExecution_hpp
#define MultiInputDeconvExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "cutlass_common/CutlassDeconvCommonExecution.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

class MultiInputDeconvExecution : public CutlassDeconvCommonExecution {
public:
    MultiInputDeconvExecution(const MNN::Op* op, Backend* backend);
    virtual ~MultiInputDeconvExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;

private:
    bool mNeedWeightFill;
    KernelInfo mKernelInfo;
};


} // namespace CUDA
} // namespace MNN

#endif /* MultiInputDeconvExecution */