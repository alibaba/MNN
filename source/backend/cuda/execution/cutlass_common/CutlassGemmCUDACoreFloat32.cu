//
//  CutlassGemmCUDACoreFloat32.cu
//  MNN
//
//  Created by MNN on 2023/03/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "CutlassConvCommonExecution.hpp"

namespace MNN {
namespace CUDA {
ErrorCode CutlassConvCommonExecution::callCutlassGemmCudaCoreFloat32(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    ElementInput_F32 *input_fp32_addr = mNeedIm2Col ? (ElementInput_F32 *)mIm2ColBuffer : (ElementInput_F32 *)input->deviceId();

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);
    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);// m n k

    if(mActivationType == 1) {
        // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
        // instantiated CUTLASS kernel
        typename GemmCuda_F32_F32_Relu_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {input_fp32_addr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementInput_F32 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementOutput_F32 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
        size_t workspace_size = GemmCuda_F32_F32_Relu_AlignCuda::get_workspace_size(arguments);

        if(workspace_size != 0) {
            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
            mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
        }

        // Check the problem size is supported or not 
        cutlass::Status status = mGemmCudaF32F32Relu.can_implement(arguments);
        cutlass_check(status);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = mGemmCudaF32F32Relu.initialize(arguments, (uint8_t *)mWorkspace);
        cutlass_check(status);

    } else if(mActivationType == 2) {
        // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
        // instantiated CUTLASS kernel
        typename GemmCuda_F32_F32_Relu6_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {input_fp32_addr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementInput_F32 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementOutput_F32 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
        size_t workspace_size = GemmCuda_F32_F32_Relu6_AlignCuda::get_workspace_size(arguments);

        if(workspace_size != 0) {
            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
            mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
        }

        // Check the problem size is supported or not 
        cutlass::Status status = mGemmCudaF32F32Relu6.can_implement(arguments);
        cutlass_check(status);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = mGemmCudaF32F32Relu6.initialize(arguments, (uint8_t *)mWorkspace);
        cutlass_check(status);

    } else {
        typename GemmCuda_F32_F32_Linear_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {input_fp32_addr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementInput_F32 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementOutput_F32 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
        size_t workspace_size = GemmCuda_F32_F32_Linear_AlignCuda::get_workspace_size(arguments);

        if(workspace_size != 0) {
            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
            mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
        }
        cutlass::Status status = mGemmCudaF32F32Ln.can_implement(arguments);
        cutlass_check(status);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = mGemmCudaF32F32Ln.initialize(arguments, (uint8_t *)mWorkspace);
        cutlass_check(status);
    }
    return NO_ERROR;

}

}
}
