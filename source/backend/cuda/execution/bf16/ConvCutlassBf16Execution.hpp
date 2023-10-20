//
//  ConvCutlassBf16Execution.hpp
//  MNN
//
//  Created by MNN on 2023/05/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_BF16

#ifndef ConvCutlassBf16Execution_hpp
#define ConvCutlassBf16Execution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "CutlassGemmBf16Param.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
#include "../cutlass_common/CutlassConvCommonExecution.hpp"

namespace MNN {
namespace CUDA {

class ConvCutlassBf16Execution : public CutlassConvCommonExecution {
public:
    struct Resource {
        Resource(Backend* bn, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mBias;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> biasTensor;
        Backend* mBackend = nullptr;
    };
    ConvCutlassBf16Execution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvCutlassBf16Execution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;
};

} // namespace CUDA
} // namespace MNN

#endif /* ConvCutlassBf16Execution */
#endif