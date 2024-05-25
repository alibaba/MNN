//
//  MatMulExecution.hpp
//  MNN
//
//  Created by MNN on 2020/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MatMulExecution_hpp
#define MatMulExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "MNNCUDADefine.hpp"
#include "CutlassGemmBatchedParam.hpp"
#include "CutlassGemmParam.hpp"
#include "MNNCUDAFunction.cuh"

#ifdef ENABLE_CUDA_TUNE_PARAM
#include "cutlass_common/tune/CutlassGemmTuneCommonExecution.hpp"
#endif

namespace MNN {
namespace CUDA {
class MatMulExecution : 
    #ifdef ENABLE_CUDA_TUNE_PARAM
    public CutlassGemmTuneCommonExecution
    #else
    public Execution 
    #endif 
{
public:
    MatMulExecution(bool transposeA, bool transposeB, Backend *backend, int aS = 1, int bS = 1, int cS = 1);
    virtual ~MatMulExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    void setArguments(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    bool mTransposeA;
    bool mTransposeB;
    int mAs;
    int mBs;
    int mCs;
    Backend* mBackend = nullptr;

    std::shared_ptr<Tensor> mBiasTensor;

    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm75 mGemmBatchedF16F16LnAlign1RCSm75;
    GemmTensor_F16_F16_Linear_AlignCuda_Sm75 mGemmF16F16LnAlign1Sm75;
    GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Column_Sm75 mGemmBatchedF32F32LnAlign1RCSm75;
    GemmTensor_F32_F32_Linear_AlignCuda_Sm75 mGemmF32F32LnAlign1Sm75;
    GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Column_Sm75 mGemmBatchedF16F32LnAlign1RCSm75;
    GemmTensor_F16_F32_Linear_AlignCuda_Sm75 mGemmF16F32LnAlign1Sm75;

    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75 mGemmBatchedF16F16LnAlign8RCSm75;
    GemmTensor_F16_F16_Linear_AlignTensor_Sm75 mGemmF16F16LnAlign8Sm75;
    GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Column_Sm75 mGemmBatchedF32F32LnAlign8RCSm75;
    GemmTensor_F32_F32_Linear_AlignTensor_Sm75 mGemmF32F32LnAlign8Sm75;
    GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75 mGemmBatchedF16F32LnAlign8RCSm75;
    GemmTensor_F16_F32_Linear_AlignTensor_Sm75 mGemmF16F32LnAlign8Sm75;

    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm75 mGemmBatchedF16F16LnAlign8RRSm75;
    GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Row_Sm75 mGemmBatchedF32F32LnAlign8RRSm75;
    GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Row_Sm75 mGemmBatchedF16F32LnAlign8RRSm75;

    GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column mGemmBatchedCudaF16F16LnAlign1RC;
    GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column mGemmBatchedCudaF32F32LnAlign1RC;
    GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column mGemmBatchedCudaF16F32LnAlign1RC;

    GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Row mGemmBatchedCudaF16F16LnAlign1RR;
    GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row mGemmBatchedCudaF32F32LnAlign1RR;
    GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Row mGemmBatchedCudaF16F32LnAlign1RR;

    std::shared_ptr<Tensor> workspaceTensor;
    void* mWorkspace;
    void* mTempMatA;
    void* mTempMatB;
    void* mBiasPtr = nullptr;
    bool mNeedATempBuffer = false;
    bool mNeedBTempBuffer = false;
    bool mUseRRLayout = false;
    bool mResizeSetArgument = false;
    bool mNeedConvertMatAB = false;
    CutlassGemmInfo mGemmInfo;
    int mBatch = 1;
    int mGpuComputeCap;
    bool mIsTuned = false; 
    bool mFp16Infer = false;
    bool mFp32Infer = false;
    bool mFp16Fp32MixInfer = false;
    bool mConvertGemmSplitK = false;
    bool mLargeBatchSmallGemm = false;
};
} // namespace CUDA
} // namespace MNN

#endif
