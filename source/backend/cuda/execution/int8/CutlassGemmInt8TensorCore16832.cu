//
//  CutlassGemmInt8TensorCore16832.cu
//  MNN
//
//  Created by MNN on 2023/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#include "ConvInt8CutlassExecution.hpp"

namespace MNN {
namespace CUDA {
ErrorCode ConvInt8CutlassExecution::callCutlassGemmInt8TensorCore16832(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int8_t *inputA_ptr = mNeedIm2Col ? (int8_t *)mIm2ColBuffer : (int8_t *)input->deviceId();

    int8_t clamp_max = int8_t(mResource->mClampMax);
    int8_t clamp_min = int8_t(mResource->mClampMin);
    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);// m n k
    if(mActivationType == 1) {
        clamp_min = int8_t(0);
    } else if(mActivationType == 2) {
        clamp_max = int8_t(6);
        clamp_min = int8_t(0);
    }

    // printf("Gemm16832 Int8 size:%d-%d-%d\n", mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);
    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel

    typename GemmInt8Tensor_Clamp_AlignTensor_Normal_Sm80::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        {(int8_t *)mResource->mWeightInt8Ptr, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                        {(int8_t *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        {(int32_t *)mResource->mBiasInt32Ptr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                        {(float *)mResource->mScaleFloatPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                        {clamp_max, clamp_min},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor
    size_t workspace_size = GemmInt8Tensor_Clamp_AlignTensor_Normal_Sm80::get_workspace_size(arguments);

    if(workspace_size != 0) {
        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
        mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
    }

    // Check the problem size is supported or not 
    // cutlass::Status status = mGemmInt8Clamp.can_implement(arguments);
    // cutlass_check(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    cutlass::Status status = mGemmInt8ClampNormalSm80.initialize(arguments, (uint8_t *)mWorkspace);
    cutlass_check(status);

    return NO_ERROR;
}

}
}
#endif