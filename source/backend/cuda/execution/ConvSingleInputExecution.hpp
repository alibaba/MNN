//
//  ConvSingleInputExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvSingleInputExecution_hpp
#define ConvSingleInputExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "TensorCoreGemm.cuh"
namespace MNN {
namespace CUDA {

struct KernelInfo {
    int groups         = 0;
    int kernelN        = 0;
    int kernelC        = 0;
    int kernelX        = 0;
    int kernelY        = 0;
    int strideX        = 0;
    int strideY        = 0;
    int dilateX        = 0;
    int dilateY        = 0;
    int activationType = 0;
};//

extern "C"
class ConvSingleInputExecution : public Execution {
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
    ConvSingleInputExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvSingleInputExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;

    const Op* mOp = nullptr;
    MatMulParam mMatMulParam;
    std::pair<void*, int> mGpuMatMulParam;

    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    std::pair<void*, int> mGpuIm2ColParam;

    __half* mIm2ColBuffer;
};

} // namespace CUDA
} // namespace MNN

#endif /* ConvSingleInputExecution */
