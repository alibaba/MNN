//
//  CutlassGemmTensorCore.cu
//  MNN
//
//  Created by MNN on 2023/05/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_BF16
#include "CutlassConvCommonExecution.hpp"

namespace MNN {
namespace CUDA {
ErrorCode CutlassConvCommonExecution::callCutlassGemmBf16TensorCore(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    ElementInput_BF16 *inputA_ptr = mNeedIm2Col ? (ElementInput_BF16 *)mIm2ColBuffer : (ElementInput_BF16 *)input->deviceId();

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);
    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);// m n k
    if(mActivationType == 1) {
        // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
        // instantiated CUTLASS kernel
        typename GemmTensor_BF16_BF16_Relu_AlignTensor_Sm80::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementInput_BF16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementOutput_BF16 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_BF16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
        size_t workspace_size = GemmTensor_BF16_BF16_Relu_AlignTensor_Sm80::get_workspace_size(arguments);

        if(workspace_size != 0) {
            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
            mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
        }

        // Check the problem size is supported or not 
        cutlass::Status status = mGemmBF16BF16ReluSm80.can_implement(arguments);
        cutlass_check(status);
    
        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = mGemmBF16BF16ReluSm80.initialize(arguments, (uint8_t *)mWorkspace);
        cutlass_check(status);

    } else if(mActivationType == 2) {
            // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_BF16_BF16_Relu6_AlignTensor_Sm80::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_BF16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_BF16 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_BF16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_BF16_BF16_Relu6_AlignTensor_Sm80::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBF16BF16Relu6Sm80.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBF16BF16Relu6Sm80.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);

    } else {
    
            typename GemmTensor_BF16_BF16_Linear_AlignTensor_Sm80::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        {(ElementInput_BF16 *)mFilterAddr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                        {(ElementOutput_BF16 *)mBiasAddr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                        {(ElementOutput_BF16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_BF16_BF16_Linear_AlignTensor_Sm80::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackendPtr->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            cutlass::Status status = mGemmBF16BF16LnSm80.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBF16BF16LnSm80.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
    }
    return NO_ERROR;
}

}
}
#endif
