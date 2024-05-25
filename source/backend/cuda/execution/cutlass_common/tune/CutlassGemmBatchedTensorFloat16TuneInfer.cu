//
//  CutlassGemmTuneCommonExecution.cu
//  MNN
//
//  Created by MNN on 2023/10/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_TUNE_PARAM

#include "CutlassGemmTuneCommonExecution.hpp"

namespace MNN {
namespace CUDA {

void CutlassGemmTuneCommonExecution::setGemmBatchedTensorCoreFloat16Argments(const GemmParamInfo* params) {

    ElementComputeEpilogue alpha = ElementComputeEpilogue(params->coefs[0]);
    ElementComputeEpilogue beta = ElementComputeEpilogue(params->coefs[1]);

    // Split K dimension into 1 partitions
    cutlass::gemm::GemmCoord problem_size(params->problemSize[0], params->problemSize[1], params->problemSize[2]);// m n k
    void* workspace_ptr = nullptr;
    //MNN_PRINT("gemmbatched: batch-%d, problem-%d %d %d, layout:%d vec:%d, best block:%s\n", params->batchSize, params->problemSize[0], params->problemSize[1], params->problemSize[2], params->layout, params->epilogueVectorize, params->prefeBlockSize.c_str());

    if(params->batchSize > 0) { // BatchGemm
        if(params->layout == 0) { // RowColumn
            if(params->epilogueVectorize) { // AlignTensor
                // BatchGemm + RowColumn + AlignTensor
                // MNN_PRINT("gemmbatched0: batch-%d, problem-%d %d %d\n", params->batchSize, params->problemSize[0], params->problemSize[1], params->problemSize[2]);
                if (params->prefeBlockSize == "_64x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_64x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RC_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_64x64x64_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x64::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_64x64x64.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RC_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_64x128x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x128x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_64x128x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RC_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_128x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_128x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RC_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_128x64x64_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x64::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_128x64x64.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RC_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_256x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_256x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_256x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RC_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_128x128x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x128x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_128x128x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RC_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
            } else {
                // BatchGemm + RowColumn + AlignCuda
                if (params->prefeBlockSize == "_64x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_64x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RC_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_64x64x64_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x64x64::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_64x64x64.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RC_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_64x128x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x128x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_64x128x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RC_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_128x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_128x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RC_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_128x64x64_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x64x64::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_128x64x64.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RC_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_256x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_256x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_256x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RC_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_128x128x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x128x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_128x128x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RC_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
            }
        } else if(params->layout == 1) { // RowRow
            if(params->epilogueVectorize) { // AlignTensor
                // BatchGemm + RowRow + AlignTensor
                if (params->prefeBlockSize == "_64x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_64x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RR_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_64x64x64_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x64x64::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_64x64x64.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RR_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_64x128x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x128x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_64x128x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RR_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_128x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_128x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RR_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_128x64x64_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x64x64::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_128x64x64.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RR_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_256x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_256x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_256x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RR_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_128x128x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x128x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_128x128x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign8RR_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
            } else {
                // BatchGemm + RowColumn + AlignCuda
                if (params->prefeBlockSize == "_64x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_64x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RR_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_64x64x64_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x64x64::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_64x64x64.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RR_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_64x128x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x128x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_64x128x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RR_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
                if (params->prefeBlockSize == "_128x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_128x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RR_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_128x64x64_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x64x64::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_128x64x64.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RR_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_256x64x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_256x64x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_256x64x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RR_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }

                if (params->prefeBlockSize == "_128x128x32_")
                {
                    typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                        {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                        (int64_t)params->batchOffset[0], // batch_stride_A
                        {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[1], // batch_stride_B
                        {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                        (int64_t)params->batchOffset[2], // batch_stride_bias
                        {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                        (int64_t)params->batchOffset[3],  // batch_stride_C
                        {alpha, beta},          // <- tuple of alpha and beta
                        params->batchSize};                // batch_count

                    size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x128x32::get_workspace_size(arguments);
                    if(workspace_size != 0 && workspace_ptr == nullptr) {
                        std::shared_ptr<Tensor> workspaceTensor;
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_128x128x32.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedF16F16TensorAlign1RR_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                    cutlass_check(status); 
                    return;
                }
            }
        } else {
            MNN_PRINT("Not support Gemm Infer now\n");
        }
    } else { // Gemm
        MNN_PRINT("Not support Gemm Infer now\n");
    }
}

void CutlassGemmTuneCommonExecution::runGemmBatchedTensorCoreFloat16Infer(const GemmParamInfo* params) {
    // MNN_PRINT("Run %d %d %d %s\n", params->batchSize, params->layout, params->epilogueVectorize, params->prefeBlockSize.c_str());
    if(params->batchSize > 0) { // BatchGemm
        if(params->layout == 0) { // RowColumn
            if(params->epilogueVectorize) { // AlignTensor
                if (params->prefeBlockSize == "_64x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_64x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_64x64x64_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_64x64x64();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_64x128x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_64x128x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_128x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x64x64_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_128x64x64();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_256x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_256x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x128x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RC_128x128x32();
                    cutlass_check(status);
                    return;
                }
            } else {
                if (params->prefeBlockSize == "_64x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_64x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_64x64x64_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_64x64x64();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_64x128x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_64x128x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_128x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x64x64_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_128x64x64();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_256x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_256x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x128x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RC_128x128x32();
                    cutlass_check(status);
                    return;
                }
            }
        } else if(params->layout == 1) {
            if(params->epilogueVectorize) { // AlignTensor
                if (params->prefeBlockSize == "_64x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_64x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_64x64x64_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_64x64x64();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_64x128x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_64x128x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_128x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x64x64_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_128x64x64();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_256x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_256x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x128x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign8RR_128x128x32();
                    cutlass_check(status);
                    return;
                }
            } else {
                if (params->prefeBlockSize == "_64x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_64x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_64x64x64_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_64x64x64();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_64x128x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_64x128x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_128x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x64x64_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_128x64x64();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_256x64x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_256x64x32();
                    cutlass_check(status);
                    return;
                }
                if (params->prefeBlockSize == "_128x128x32_") {
                    cutlass::Status status = mGemmBatchedF16F16TensorAlign1RR_128x128x32();
                    cutlass_check(status);
                    return;
                }
            }
        }
    }
    MNN_PRINT("Error Not support Gemm Infer now\n");
    return;
}

} // namespace CUDA
} // namespace MNN
#endif
