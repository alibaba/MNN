//
//  CutlassConvCommonExecution.cu
//  MNN
//
//  Created by MNN on 2023/03/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CutlassConvCommonExecution.hpp"

namespace MNN {
namespace CUDA {

CutlassConvCommonExecution::CutlassConvCommonExecution(Backend *backend) :
#ifdef ENABLE_CUDA_TUNE_PARAM
CutlassGemmTuneCommonExecution(backend)
#else
Execution(backend)
#endif
{
    mBackendPtr = backend;
}

ErrorCode CutlassConvCommonExecution::runCutlassGemmFunc() {
    #ifdef ENABLE_CUDA_BF16
    if(mBf16Infer) {
        if(mActivationType == 1) {
            cutlass::Status status = mGemmBF16BF16ReluSm80();
            cutlass_check(status);
        } else if(mActivationType == 2) {
            cutlass::Status status = mGemmBF16BF16Relu6Sm80();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmBF16BF16LnSm80();
            cutlass_check(status);
        }
        return NO_ERROR;
    }
    #endif
    if(mFp32Infer) {
        if(mActivationType == 1) {
            cutlass::Status status = mGemmCudaF32F32Relu();
            cutlass_check(status);
        } else if(mActivationType == 2) {
            cutlass::Status status = mGemmCudaF32F32Relu6();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmCudaF32F32Ln();
            cutlass_check(status);
        }
        return NO_ERROR;
    }

    if(mGpuComputeCap < 70) {
        if(mActivationType == 1) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmCudaF16F32Relu();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmCudaF16F16Relu();
                cutlass_check(status);
            }
        } else if(mActivationType == 2) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmCudaF16F32Relu6();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmCudaF16F16Relu6();
                cutlass_check(status);
            }
        } else {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmCudaF16F32Ln();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmCudaF16F16Ln();
                cutlass_check(status);
            }
        }
    
        return NO_ERROR;
    } else if(mGpuComputeCap < 75) {
        if(mActivationType == 1) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32ReluSm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16ReluSm70();
                cutlass_check(status);
            }
        } else if(mActivationType == 2) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32Relu6Sm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16Relu6Sm70();
                cutlass_check(status);
            }
        } else {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32LnSm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16LnSm70();
                cutlass_check(status);
            }
        }
    
        return NO_ERROR;
    }

    #ifdef ENABLE_CUDA_TUNE_PARAM
    if(mIsTuned) {
        runGemmTensorCoreFloat16Infer(&mInfo);
    } 
    #endif
    if(!mIsTuned) {
        if(mActivationType == 1) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32ReluSm75();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16ReluSm75();
                cutlass_check(status);
            }
        } else if(mActivationType == 2) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32Relu6Sm75();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16Relu6Sm75();
                cutlass_check(status);
            }
        } else {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32LnSm75();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16LnSm75();
                cutlass_check(status);
            }
        }
    }
    return NO_ERROR;
}

} // namespace CUDA
} // namespace MNN
