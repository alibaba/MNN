#include "MatMulExecution.hpp"

namespace MNN {
namespace CUDA {

template<typename T0, typename T1>
__global__ void PackPadFill(
    const T0* A, const T0* B,
    bool transA, bool transB,
    T1* tempA, T1* tempB, const int batch,
    const int e, const int l, const int h,
    const int ep, const int lp, const int hp,
    DivModFast d_e, DivModFast d_l, DivModFast d_h,
    DivModFast d_lp, DivModFast d_lp2
) {
    T1 zero = (T1)0.0;

    if((char *)A != (char *)tempA) {
        if(transA) { // l * e , just transpose to e * lp
            const int maxCount = batch * e * lp;
            for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                int bIndex, lpIndex, eIndex, tmp;
                d_lp.divmod(index, tmp, lpIndex);
                d_e.divmod(tmp, bIndex, eIndex);

                if(lpIndex >= l) {
                    tempA[index] = zero;
                    continue;
                }
                tempA[index] = A[bIndex * e * l + lpIndex * e + eIndex];
            }
        } else { // e * l, just pack for l
            if (l & 1 == 0) {
                const int maxCount = batch * e * (lp >> 1);
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lp2Index, eIndex, bIndex, tmp;
                    d_lp2.divmod(index, tmp, lp2Index);
                    d_e.divmod(tmp, bIndex, eIndex);

                    if(lp2Index + lp2Index >= l) {
                        tempA[index+index] = zero;
                        tempA[index+index+1] = zero;
                        continue;
                    }
                    tempA[index+index] =  A[bIndex * e * l + eIndex * l + lp2Index + lp2Index];
                    tempA[index+index+1] = A[bIndex * e * l + eIndex * l + lp2Index + lp2Index + 1];
                }
            } else {
                const int maxCount = batch * e * lp;
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, eIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_e.divmod(tmp, bIndex, eIndex);
                    if(lpIndex >= l || eIndex >= e) {
                        tempA[index] = zero;
                        continue;
                    }
                    tempA[index] = A[bIndex * e * l + eIndex * l + lpIndex];
                }
            }
        }
    }
    if((char *)B != (char *)tempB) {
        if(!transB) { // l * h 
            const int maxCount = batch * lp * h;
            if(h == hp) { // and h already packed, just pack for l -> lp * h
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hpIndex, bIndex, tmp;
                    d_h.divmod(index, tmp, hpIndex);
                    d_lp.divmod(tmp, bIndex, lpIndex);

                    if(lpIndex >= l || hpIndex >= h) {
                        tempB[index] = zero;
                        continue;
                    }
                    tempB[index] = B[bIndex * h * l + lpIndex * h + hpIndex];
                }
            } else { // and h not packed, just transpose and pack for l -> h * lp
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_h.divmod(tmp, bIndex, hIndex);

                    if(lpIndex >= l || hIndex >= h) {
                        tempB[index] = zero;
                        continue;
                    }
                    tempB[index] = B[bIndex * h * l + lpIndex * h + hIndex];
                }
            }
        } else { // h * l, just pack for l
            if(l & 1 == 0) {
                const int maxCount = batch * h * (lp >> 1);
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lp2Index, hIndex, bIndex, tmp;
                    d_lp2.divmod(index, tmp, lp2Index);
                    d_h.divmod(tmp, bIndex, hIndex);

                    if(lp2Index + lp2Index >= l) {
                        tempB[index+index] = zero;
                        tempB[index+index+1] = zero;
                        continue;
                    }
                    tempB[index+index] = B[bIndex * h * l + hIndex * l + lp2Index + lp2Index];
                    tempB[index+index+1] = B[bIndex * h * l + hIndex * l + lp2Index + lp2Index + 1];
                }
            } else {
                const int maxCount = batch * h * lp;
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_h.divmod(tmp, bIndex, hIndex);

                    if(lpIndex >= l || hIndex >= h) {
                        tempB[index] = zero;
                        continue;
                    }
                    tempB[index] = B[bIndex * h * l + hIndex * l + lpIndex];
                }
            }
        }
    }

}

MatMulExecution::MatMulExecution(bool transposeA, bool transposeB, Backend *backend) : Execution(backend) {
    mTransposeA = transposeA;
    mTransposeB = transposeB;
    mBackend = backend;
    int precisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (precisonLevel == 2);
    mFp32Infer = (precisonLevel == 1);
    mFp16Fp32MixInfer = (precisonLevel == 0);
}
MatMulExecution::~ MatMulExecution() {
    // do nothing
}

void MatMulExecution::setArguments(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();

    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    auto C = outputs[0];
    bool hAlignment = (mGemmInfo.elhPad[2] == mGemmInfo.elh[2]);

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elh[2], mGemmInfo.elhPad[1]);// m n k

    if (inputs.size() > 2) {
        mBiasPtr = (void*)inputs[2]->deviceId();
        beta = ElementComputeEpilogue(1);
    }

    if(mFp32Infer) {
        if(mUseRRLayout) {
            typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                                {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                (int64_t)(0), // batch_stride_bias
                                                {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                mBatch};                // batch_count

            size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }
            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RR.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBatchedCudaF32F32LnAlign1RR.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                                                {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                (int64_t)(0), // batch_stride_bias
                                                {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                mBatch};                // batch_count

            size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }
            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RC.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBatchedCudaF32F32LnAlign1RC.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status); 
        }
        return;
    }

    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);

    if(mGpuComputeCap < 75) {
        if(mFp16Infer) {
            if(mUseRRLayout) {
                typename GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Row::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                    {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                    (int64_t)(0), // batch_stride_bias
                    {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                    {alpha, beta},          // <- tuple of alpha and beta
                    mBatch};                // batch_count
    
                size_t workspace_size = GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Row::get_workspace_size(arguments);
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedCudaF16F16LnAlign1RR.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedCudaF16F16LnAlign1RR.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status); 
            } else {
                typename GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                    {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                    (int64_t)(0), // batch_stride_bias
                    {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                    {alpha, beta},          // <- tuple of alpha and beta
                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedCudaF16F16LnAlign1RC.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedCudaF16F16LnAlign1RC.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else {
            if(mUseRRLayout) {
                if(mNeedConvertMatAB) {
                    typename GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Row::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                        {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                        (int64_t)(0), // batch_stride_bias
                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        mBatch};                // batch_count
    
                    size_t workspace_size = GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Row::get_workspace_size(arguments);
    
                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedCudaF16F32LnAlign1RR.can_implement(arguments);
                    cutlass_check(status);
    
                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedCudaF16F32LnAlign1RR.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status);
                } else {
                    typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                        {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                        {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                        (int64_t)(0), // batch_stride_bias
                                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                        {alpha, beta},          // <- tuple of alpha and beta
                                                        mBatch};                // batch_count
    
                    size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row::get_workspace_size(arguments);
    
                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RR.can_implement(arguments);
                    cutlass_check(status);
    
                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedCudaF32F32LnAlign1RR.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status);
                }
            } else {
                if(mNeedConvertMatAB) {
                    typename GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                            (int64_t)(0), // batch_stride_bias
                                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            mBatch};                // batch_count

                    size_t workspace_size = GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedCudaF16F32LnAlign1RC.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedCudaF16F32LnAlign1RC.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status); 
                } else {
                    typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                        {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                        {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                        (int64_t)(0), // batch_stride_bias
                                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                                        {alpha, beta},          // <- tuple of alpha and beta
                                                        mBatch};                // batch_count

                    size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RC.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedCudaF32F32LnAlign1RC.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status); 
                }
            }
        }
        return;
    }

    if(mFp16Infer) {
        if(mUseRRLayout) {
            typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                (int64_t)(0), // batch_stride_bias
                {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                {alpha, beta},          // <- tuple of alpha and beta
                mBatch};                // batch_count

            size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm75::get_workspace_size(arguments);
            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }
            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBatchedF16F16LnAlign8RRSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBatchedF16F16LnAlign8RRSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status); 
        } else {
            if(hAlignment) {
                typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                    {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                    (int64_t)(0), // batch_stride_bias
                    {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                    {alpha, beta},          // <- tuple of alpha and beta
                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75::get_workspace_size(arguments);

                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedF16F16LnAlign8RCSm75.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedF16F16LnAlign8RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                    {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                    (int64_t)(0), // batch_stride_bias
                    {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                    {alpha, beta},          // <- tuple of alpha and beta
                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm75::get_workspace_size(arguments);

                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedF16F16LnAlign1RCSm75.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedF16F16LnAlign1RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
        }

    } else {
        if(mUseRRLayout) {
            if(mNeedConvertMatAB) {
                typename GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Row_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                    {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                    (int64_t)(0), // batch_stride_bias
                                    {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                    {alpha, beta},          // <- tuple of alpha and beta
                                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Row_Sm75::get_workspace_size(arguments);

                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedF16F32LnAlign8RRSm75.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedF16F32LnAlign8RRSm75.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                typename GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Row_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                    {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                                    {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                    (int64_t)(0), // batch_stride_bias
                                                    {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Row_Sm75::get_workspace_size(arguments);

                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedF32F32LnAlign8RRSm75.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedF32F32LnAlign8RRSm75.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
        } else {
            if(hAlignment) {
                if(mNeedConvertMatAB) {
                    typename GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                        {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                        (int64_t)(0), // batch_stride_bias
                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        mBatch};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75::get_workspace_size(arguments);

                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F32LnAlign8RCSm75.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F32LnAlign8RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status); 
                } else {
                    typename GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                        {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                        (int64_t)(0), // batch_stride_bias
                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        mBatch};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Column_Sm75::get_workspace_size(arguments);

                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF32F32LnAlign8RCSm75.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF32F32LnAlign8RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status); 
                }
            } else {
                if(mNeedConvertMatAB) {
                    typename GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                            (int64_t)(0), // batch_stride_bias
                                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            mBatch};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Column_Sm75::get_workspace_size(arguments);

                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F32LnAlign1RCSm75.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F32LnAlign1RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status); 
                } else {
                    typename GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                        {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                        {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]), // batch_stride_B
                                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                        (int64_t)(0), // batch_stride_bias
                                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                                        {alpha, beta},          // <- tuple of alpha and beta
                                                        mBatch};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Column_Sm75::get_workspace_size(arguments);

                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF32F32LnAlign1RCSm75.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF32F32LnAlign1RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status); 
                }
            }
        }
    }
}

ErrorCode MatMulExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);

    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    auto C = outputs[0];
    auto dimensions = C->dimensions();
    mBatch = 1;
    for (int i = 0; i < dimensions - 2; ++i) {
        mBatch *= C->length(i);
    }
    auto e = C->length(dimensions-2);
    auto h = C->length(dimensions-1);
    auto w0 = inputs[0]->length(dimensions-1);
    auto h0 = inputs[0]->length(dimensions-2);

    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }

    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[1] = UP_DIV(l, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[2] = UP_DIV(h, PACK_NUMBER) * PACK_NUMBER;

    bool lAlignment = (mGemmInfo.elhPad[1] == mGemmInfo.elh[1]);
    bool hAlignment = (mGemmInfo.elhPad[2] == mGemmInfo.elh[2]);
    bool needBTranspose = (!mTransposeB && !hAlignment);

    mUseRRLayout = (!mTransposeB && hAlignment);
    mNeedATempBuffer = (mTransposeA || !lAlignment);
    mNeedBTempBuffer = (needBTranspose || !lAlignment);
    mNeedConvertMatAB = (mNeedATempBuffer || mNeedBTempBuffer);

    //MNN_PRINT("trAtrB:%d-%d, tmpAB:%d-%d inps:%d, bwlh:%d-%d-%d-%d\n", mTransposeA, mTransposeB, mNeedATempBuffer, mNeedBTempBuffer, inputs.size(), mBatch, mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2]);

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    std::pair<void*, size_t> bufferAData, bufferBData;
    if(mNeedConvertMatAB) {
        size_t convertBytes = 2;
        if(mFp32Infer) {
            convertBytes = 4;
        }
        bufferAData = pool->alloc(convertBytes * mBatch * mGemmInfo.elh[0] * mGemmInfo.elhPad[1]);
        mTempMatA = (void*)((uint8_t*)bufferAData.first + bufferAData.second);

        bufferBData = pool->alloc(convertBytes * mBatch * mGemmInfo.elh[2] * mGemmInfo.elhPad[1]);
        mTempMatB = (void*)((uint8_t*)bufferBData.first + bufferBData.second);

        pool->free(bufferAData);
        pool->free(bufferBData);
    } else {
        mTempMatA = (void *)A->deviceId();
        mTempMatB = (void *)B->deviceId();
    }
    
    // inputSize only two, No need Bias, Fake address for mBiasPtr is ok because beta is zero.
    if(inputs.size() == 2) {
    	mBiasPtr = (void*)B->deviceId();
    }
    //printf("MatMulAB:%p-%p-%p-%p\n", A->host<void*>(), A->deviceId(), B->host<void*>(), B->deviceId());

    // Set Cutlass Param Arguments
    mResizeSetArgument = (mTempMatA != nullptr && mTempMatB != nullptr && C->deviceId() != 0);
    if(mResizeSetArgument) {
        setArguments(inputs, outputs);
    }

    return NO_ERROR;
}

ErrorCode MatMulExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    bool hAlignment = (mGemmInfo.elhPad[2] == mGemmInfo.elh[2]);

    // PreProcess for Alignment
    if(mNeedConvertMatAB) {
        DivModFast eD(mGemmInfo.elh[0]);
        DivModFast lD(mGemmInfo.elh[1]);
        DivModFast hD(mGemmInfo.elh[2]);
        DivModFast lpD((mGemmInfo.elhPad[1]));
        DivModFast lp2D((mGemmInfo.elhPad[1]/2));

        auto& prop = runtime->prop();
        int block_num = prop.multiProcessorCount;
        int block_size = prop.maxThreadsPerBlock;
        if(mFp32Infer) {
            PackPadFill<<<block_num, block_size>>>((const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deviceId(), \
                    mTransposeA, mTransposeB, (float*)mTempMatA, (float*)mTempMatB,
                    mBatch, mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2], \
                    mGemmInfo.elhPad[0], mGemmInfo.elhPad[1], mGemmInfo.elhPad[2], \
                    eD, lD, hD, lpD, lp2D);
            checkKernelErrors;        
        } else if(mFp16Fp32MixInfer) {
            PackPadFill<<<block_num, block_size>>>((const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deviceId(), \
                    mTransposeA, mTransposeB, (half*)mTempMatA, (half*)mTempMatB,
                    mBatch, mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2], \
                    mGemmInfo.elhPad[0], mGemmInfo.elhPad[1], mGemmInfo.elhPad[2], \
                    eD, lD, hD, lpD, lp2D);
            checkKernelErrors;
        } else {
            PackPadFill<<<block_num, block_size>>>((const half*)inputs[0]->deviceId(), (const half*)inputs[1]->deviceId(), \
                    mTransposeA, mTransposeB, (half*)mTempMatA, (half*)mTempMatB,
                    mBatch, mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2], \
                    mGemmInfo.elhPad[0], mGemmInfo.elhPad[1], mGemmInfo.elhPad[2],  \
                    eD, lD, hD, lpD, lp2D);
            checkKernelErrors;  
        }
    }

    if(!mResizeSetArgument) {
        // Repeat set cutlass argments if possible
        //printf("argment onexecute set\n");

        if(!mNeedConvertMatAB) {
            mTempMatA = (void *)inputs[0]->deviceId();
            mTempMatB = (void *)inputs[1]->deviceId();
        }
        setArguments(inputs, outputs);
    }


    if(mFp32Infer) {
        if(mUseRRLayout) {
            cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RR();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RC();
            cutlass_check(status);
        }
        return NO_ERROR;
    }

    if(mGpuComputeCap < 75) {
        if (mFp16Fp32MixInfer) {
            if(mUseRRLayout) {
                if(mNeedConvertMatAB) {
                    cutlass::Status status = mGemmBatchedCudaF16F32LnAlign1RR();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RR();
                    cutlass_check(status);
                }
            } else {
                if(mNeedConvertMatAB) {
                    cutlass::Status status = mGemmBatchedCudaF16F32LnAlign1RC();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RC();
                    cutlass_check(status);
                }
            }
    
        } else {
            if(mUseRRLayout) {
                cutlass::Status status = mGemmBatchedCudaF16F16LnAlign1RR();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmBatchedCudaF16F16LnAlign1RC();
                cutlass_check(status);
            }
        }
    
        return NO_ERROR;
    }

    if (mFp16Fp32MixInfer) {
        if(mUseRRLayout) {
            if(mNeedConvertMatAB) {
                cutlass::Status status = mGemmBatchedF16F32LnAlign8RRSm75();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmBatchedF32F32LnAlign8RRSm75();
                cutlass_check(status);
            }
        } else {
            if(hAlignment) {
                if(mNeedConvertMatAB) {
                    cutlass::Status status = mGemmBatchedF16F32LnAlign8RCSm75();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmBatchedF32F32LnAlign8RCSm75();
                    cutlass_check(status);
                }
            } else {
                if(mNeedConvertMatAB) {
                    cutlass::Status status = mGemmBatchedF16F32LnAlign1RCSm75();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmBatchedF32F32LnAlign1RCSm75();
                    cutlass_check(status);
                }
            }
        }

    } else {
        if(mUseRRLayout) {
            cutlass::Status status = mGemmBatchedF16F16LnAlign8RRSm75();
            cutlass_check(status);
        } else {
            if(hAlignment) {
                cutlass::Status status = mGemmBatchedF16F16LnAlign8RCSm75();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmBatchedF16F16LnAlign1RCSm75();
                cutlass_check(status);
            }
        }
    }

    return NO_ERROR;
}

class MatMulCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_MatMul();
        return new MatMulExecution(param->transposeA(), param->transposeB(), backend);
    }
};

static CUDACreatorRegister<MatMulCreator> __init(OpType_MatMul);

}
}
