//
//  CutlassGemmTensorCore884.cu
//  MNN
//
//  Created by MNN on 2023/03/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CutlassConvCommonExecution.hpp"

namespace MNN {
namespace CUDA {
ErrorCode CutlassConvCommonExecution::callCutlassGemmTensorCore884(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    ElementInput_F16 *inputA_ptr = mNeedIm2Col ? (ElementInput_F16 *)mIm2ColBuffer : (ElementInput_F16 *)input->deviceId();

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);
    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);// m n k

    if(mActivationType == 1) {
        if(mFp16Infer) {
            // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F16_Relu_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F16 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Sm70::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F16ReluSm70.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16ReluSm70.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F32_Relu_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Sm70::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F32ReluSm70.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32ReluSm70.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }

    } else if(mActivationType == 2) {

        if(mFp16Infer) {
            // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F16_Relu6_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F16 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Sm70::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F16Relu6Sm70.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16Relu6Sm70.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F32_Relu6_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Sm70::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F32Relu6Sm70.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32Relu6Sm70.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }

    } else {
    
        if(mFp16Infer) {
            typename GemmTensor_F16_F16_Linear_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        {(ElementInput_F16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                        {(ElementOutput_F16 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                        {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Sm70::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            cutlass::Status status = mGemmF16F16LnSm70.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16LnSm70.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            typename GemmTensor_F16_F32_Linear_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Sm70::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            cutlass::Status status = mGemmF16F32LnSm70.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32LnSm70.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }
    }
    return NO_ERROR;
}

}
}