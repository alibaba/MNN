//
//  CutlassDeconvCommonExecution.cu
//  MNN
//
//  Created by MNN on 2023/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CutlassDeconvCommonExecution.hpp"

namespace MNN {
namespace CUDA {

CutlassDeconvCommonExecution::CutlassDeconvCommonExecution(Backend *backend) : Execution(backend) {
    mBackendPtr = backend;
}

ErrorCode CutlassDeconvCommonExecution::runCutlassGemmFunc() {
    // Do Gemm Compute
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
    } else {
        if(mGpuComputeCap < 75) {
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
        } else {
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
    }
    return NO_ERROR;
}

} // namespace CUDA
} // namespace MNN
