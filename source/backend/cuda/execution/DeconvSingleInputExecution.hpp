//
//  DeconvSingleInputExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DeconvSingleInputExecution_hpp
#define DeconvSingleInputExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "MNNCUDADefine.hpp"
#include "CutlassGemmParam.hpp"
#include "MNNCUDAFunction.cuh"
#include "cutlass_common/CutlassDeconvCommonExecution.hpp"
namespace MNN {
namespace CUDA {

extern "C"
class DeconvSingleInputExecution : public CutlassDeconvCommonExecution {
public:
    struct Resource {
        Resource(Backend* bn, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mBias;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> biasTensor;
        KernelInfo mKernelInfo;
        Backend* mBackend = nullptr;
    };
    DeconvSingleInputExecution(Backend* backend, const MNN::Op* op,  std::shared_ptr<Resource> res);
    virtual ~DeconvSingleInputExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;
};

} // namespace CUDA
} // namespace MNN

#endif /* DeconvSingleInputExecution */
