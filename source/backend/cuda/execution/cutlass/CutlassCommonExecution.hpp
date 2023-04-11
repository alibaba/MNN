//
//  CutlassCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2023/03/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CutlassCommonExecution_hpp
#define CutlassCommonExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "../CutlassGemmParam.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

class CutlassCommonExecution : public Execution {
public:
    CutlassCommonExecution(Backend* backend);
    virtual ~CutlassCommonExecution() = default;

    ErrorCode callCutlassGemmCudaCoreFloat16(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
    ErrorCode callCutlassGemmCudaCoreFloat32(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
    ErrorCode callCutlassGemmTensorCore884(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
    ErrorCode callCutlassGemmTensorCore(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);

    ErrorCode runCutlassGemmFunc();

protected:

    Backend* mBackendPtr;
    void* mFilterAddr;
    void* mBiasAddr;
    CutlassGemmInfo mGemmInfo;
    const Op* mOp = nullptr;

    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    std::pair<void*, int> mGpuIm2ColParam;

    void* mIm2ColBuffer;

    bool mIsConv1x1S1D1P0 = false;
    bool mNeedIm2Col = true;
    std::pair<void*, int> mGpuKernelParam;
    bool mIsBlock = false;
    int mBlockNum = 1;

    GemmTensor_F16_F16_Linear_AlignTensor_Sm70 mGemmF16F16LnSm70;
    GemmTensor_F16_F32_Linear_AlignTensor_Sm70 mGemmF16F32LnSm70;
    GemmCuda_F16_F16_Linear_AlignCuda  mGemmCudaF16F16Ln;
    GemmCuda_F16_F32_Linear_AlignCuda  mGemmCudaF16F32Ln;

    GemmTensor_F16_F16_Relu_AlignTensor_Sm70 mGemmF16F16ReluSm70;
    GemmTensor_F16_F32_Relu_AlignTensor_Sm70 mGemmF16F32ReluSm70;
    GemmCuda_F16_F16_Relu_AlignCuda  mGemmCudaF16F16Relu;
    GemmCuda_F16_F32_Relu_AlignCuda  mGemmCudaF16F32Relu;

    GemmTensor_F16_F16_Relu6_AlignTensor_Sm70 mGemmF16F16Relu6Sm70;
    GemmTensor_F16_F32_Relu6_AlignTensor_Sm70 mGemmF16F32Relu6Sm70;
    GemmCuda_F16_F16_Relu6_AlignCuda  mGemmCudaF16F16Relu6;
    GemmCuda_F16_F32_Relu6_AlignCuda  mGemmCudaF16F32Relu6;

    GemmTensor_F16_F16_Linear_AlignTensor_Sm75 mGemmF16F16LnSm75;
    GemmTensor_F16_F32_Linear_AlignTensor_Sm75 mGemmF16F32LnSm75;

    GemmTensor_F16_F16_Relu_AlignTensor_Sm75 mGemmF16F16ReluSm75;
    GemmTensor_F16_F32_Relu_AlignTensor_Sm75 mGemmF16F32ReluSm75;

    GemmTensor_F16_F16_Relu6_AlignTensor_Sm75 mGemmF16F16Relu6Sm75;
    GemmTensor_F16_F32_Relu6_AlignTensor_Sm75 mGemmF16F32Relu6Sm75;

    GemmCuda_F32_F32_Relu_AlignCuda mGemmCudaF32F32Relu;
    GemmCuda_F32_F32_Relu6_AlignCuda mGemmCudaF32F32Relu6;
    GemmCuda_F32_F32_Linear_AlignCuda mGemmCudaF32F32Ln;

    int mGpuComputeCap = 75;
    int mActivationType = 0;
    bool mFp16Infer = false;
    bool mFp32Infer = false;
    bool mFp16Fp32MixInfer = false;
    int mPrecisonLevel;
    std::shared_ptr<Tensor> workspaceTensor;
    void* mWorkspace;
};

} // namespace CUDA
} // namespace MNN

#endif /* CutlassCommonExecution */