//
//  CutlassDeconvCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2023/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CutlassDeconvCommonExecution_hpp
#define CutlassDeconvCommonExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "../CutlassGemmParam.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"

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

struct Col2ImParameter {
    int padX;
    int padY;
    int dilateX;
    int dilateY;
    int strideX;
    int strideY;
    int kernelX;
    int kernelY;
    int oc;
    int ic;
    int iw;
    int ih;
    int ow;
    int oh;
    int ob;
};

class CutlassDeconvCommonExecution : public Execution {
public:
    CutlassDeconvCommonExecution(Backend* backend);
    virtual ~CutlassDeconvCommonExecution() = default;

    ErrorCode callCutlassGemmCudaCoreFloat16(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
    ErrorCode callCutlassGemmCudaCoreFloat32(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
    ErrorCode callCutlassGemmTensorCore(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);

    ErrorCode runCutlassGemmFunc();

protected:

    Backend* mBackendPtr;
    void* mFilterAddr;
    void* mBiasAddr = nullptr;
    CutlassGemmInfo mGemmInfo;
    const Op* mOp = nullptr;

    Col2ImParameter mCol2ImParamter;
    int mActivationType;
    int mGpuComputeCap;
    void* mIm2ColBuffer;
    void* mInputBuffer;
    std::shared_ptr<Tensor> workspaceTensor;
    void* mWorkspace;
    void* mZeroPtr;
    std::shared_ptr<Tensor> mZeroTensor;

    bool mFp16Infer = false;
    bool mFp32Infer = false;
    bool mFp16Fp32MixInfer = false;
    int mPrecisonLevel;

    GemmCuda_F16_F16_Linear_AlignCuda mGemmCudaF16F16Ln;
    GemmCuda_F16_F32_Linear_AlignCuda mGemmCudaF16F32Ln;
    GemmCuda_F32_F32_Linear_AlignCuda mGemmCudaF32F32Ln;

    GemmCuda_F16_F16_Relu_AlignCuda mGemmCudaF16F16Relu;
    GemmCuda_F16_F32_Relu_AlignCuda mGemmCudaF16F32Relu;
    GemmCuda_F32_F32_Relu_AlignCuda mGemmCudaF32F32Relu;

    GemmCuda_F16_F16_Relu6_AlignCuda mGemmCudaF16F16Relu6;
    GemmCuda_F16_F32_Relu6_AlignCuda mGemmCudaF16F32Relu6;
    GemmCuda_F32_F32_Relu6_AlignCuda mGemmCudaF32F32Relu6;

    GemmTensor_F16_F16_Linear_AlignCuda_Sm75 mGemmF16F16LnSm75;
    GemmTensor_F16_F32_Linear_AlignCuda_Sm75 mGemmF16F32LnSm75;

    GemmTensor_F16_F16_Relu_AlignCuda_Sm75 mGemmF16F16ReluSm75;
    GemmTensor_F16_F32_Relu_AlignCuda_Sm75 mGemmF16F32ReluSm75;

    GemmTensor_F16_F16_Relu6_AlignCuda_Sm75 mGemmF16F16Relu6Sm75;
    GemmTensor_F16_F32_Relu6_AlignCuda_Sm75 mGemmF16F32Relu6Sm75;

};

} // namespace CUDA
} // namespace MNN

#endif /* CutlassDeconvCommonExecution */