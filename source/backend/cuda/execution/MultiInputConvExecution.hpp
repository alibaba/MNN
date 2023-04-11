//
//  MultiInputConvExecution.hpp
//  MNN
//
//  Created by MNN on 2023/03/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MultiInputConvExecution_hpp
#define MultiInputConvExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "CutlassGemmParam.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"
#include "cutlass/CutlassCommonExecution.hpp"

namespace MNN {
namespace CUDA {

class MultiInputConvExecution : public CutlassCommonExecution {
public:
    MultiInputConvExecution(const MNN::Op* op, Backend* backend);
    virtual ~MultiInputConvExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;

private:
    bool mNeedWeightFill;
    bool mNeedBiasFill;
};


} // namespace CUDA
} // namespace MNN

#endif /* MultiInputConvExecution */