//
//  ConvCutlassExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvCutlassExecution_hpp
#define ConvCutlassExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "CutlassGemmParam.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

class ConvCutlassExecution : public Execution {
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
    ConvCutlassExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvCutlassExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;

    const Op* mOp = nullptr;
    CutlassGemmInfo mGemmInfo;

    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    std::pair<void*, int> mGpuIm2ColParam;

    __half* mIm2ColBuffer;

    bool mIsConv1x1S1D1P0 = false;
    bool mNeedIm2Col = true;
    std::pair<void*, int> mGpuKernelParam;
    bool mIsBlock = false;
    int mBlockNum = 1;

    GemmTensor_F16_F16_Linear_AlignTensor_Sm70 mGemmF16LnSm70;
    GemmTensor_F16_F32_Linear_AlignTensor_Sm70 mGemmF32LnSm70;
    GemmCuda_F16_F16_Linear_AlignCuda  mGemmCudaF16Ln;
    GemmCuda_F16_F32_Linear_AlignCuda  mGemmCudaF32Ln;

    GemmTensor_F16_F16_Relu_AlignTensor_Sm70 mGemmF16ReluSm70;
    GemmTensor_F16_F32_Relu_AlignTensor_Sm70 mGemmF32ReluSm70;
    GemmCuda_F16_F16_Relu_AlignCuda  mGemmCudaF16Relu;
    GemmCuda_F16_F32_Relu_AlignCuda  mGemmCudaF32Relu;

    GemmTensor_F16_F16_Relu6_AlignTensor_Sm70 mGemmF16Relu6Sm70;
    GemmTensor_F16_F32_Relu6_AlignTensor_Sm70 mGemmF32Relu6Sm70;
    GemmCuda_F16_F16_Relu6_AlignCuda  mGemmCudaF16Relu6;
    GemmCuda_F16_F32_Relu6_AlignCuda  mGemmCudaF32Relu6;

    GemmTensor_F16_F16_Linear_AlignTensor_Sm75 mGemmF16LnSm75;
    GemmTensor_F16_F32_Linear_AlignTensor_Sm75 mGemmF32LnSm75;

    GemmTensor_F16_F16_Relu_AlignTensor_Sm75 mGemmF16ReluSm75;
    GemmTensor_F16_F32_Relu_AlignTensor_Sm75 mGemmF32ReluSm75;

    GemmTensor_F16_F16_Relu6_AlignTensor_Sm75 mGemmF16Relu6Sm75;
    GemmTensor_F16_F32_Relu6_AlignTensor_Sm75 mGemmF32Relu6Sm75;

    int mGpuComputeCap = 75;
    int mActivationType = 0;
    std::shared_ptr<Tensor> workspaceTensor;
    uint8_t* mWorkspace;
};

} // namespace CUDA
} // namespace MNN

#endif /* ConvCutlassExecution */