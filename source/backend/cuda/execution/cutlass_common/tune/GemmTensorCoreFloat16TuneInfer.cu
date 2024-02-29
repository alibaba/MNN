#ifdef ENABLE_CUDA_TUNE_PARAM
#include "CutlassGemmTuneCommonExecution.hpp"

namespace MNN {
namespace CUDA {

void CutlassGemmTuneCommonExecution::setGemmTensorCoreFloat16Argments(const GemmParamInfo* params) {
    MNN_ASSERT(params->batchSize == 0);

    ElementComputeEpilogue alpha = ElementComputeEpilogue(params->coefs[0]);
    ElementComputeEpilogue beta = ElementComputeEpilogue(params->coefs[1]);

    // Split K dimension into 1 partitions
    cutlass::gemm::GemmCoord problem_size(params->problemSize[0], params->problemSize[1], params->problemSize[2]);// m n k
    void* workspace_ptr = nullptr;
    // MNN_PRINT("gemm: batch-%d, problem-%d %d %d, %d %d, %s\n", params->batchSize, params->problemSize[0], params->problemSize[1], params->problemSize[2], params->epilogueType, params->precisionType, params->prefeBlockSize.c_str());

    if(params->epilogueType == 0) { // epilogueLinear
        if(params->precisionType == 2) { // InOut:FP16_FP16
            // MNN_PRINT("gemmbatched0: batch-%d, problem-%d %d %d\n", params->batchSize, params->problemSize[0], params->problemSize[1], params->problemSize[2]);
            if (params->prefeBlockSize == "_64x64x32_")
            {
                typename GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_64x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorLnAlign8RC_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x64x64_")
            {
                typename GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_64x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorLnAlign8RC_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status);
            }
            if (params->prefeBlockSize == "_64x128x32_")
            {
                typename GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_64x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorLnAlign8RC_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x32_")
            {
                typename GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_128x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorLnAlign8RC_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }

            if (params->prefeBlockSize == "_128x64x64_")
            {
                typename GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_128x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorLnAlign8RC_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }

            if (params->prefeBlockSize == "_256x64x32_")
            {
                typename GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_256x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_256x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorLnAlign8RC_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }

            if (params->prefeBlockSize == "_128x128x32_")
            {
                typename GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_128x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorLnAlign8RC_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
        } else if(params->precisionType == 0) { // InOut:FP16_FP32
            if (params->prefeBlockSize == "_64x64x32_")
            {
                typename GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_64x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorLnAlign8RC_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x64x64_")
            {
                typename GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_64x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorLnAlign8RC_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x128x32_")
            {
                typename GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_64x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorLnAlign8RC_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x32_")
            {
                typename GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_128x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorLnAlign8RC_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x64_")
            {
                typename GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_128x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorLnAlign8RC_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_256x64x32_")
            {
                typename GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_256x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_256x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorLnAlign8RC_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }

            if (params->prefeBlockSize == "_128x128x32_")
            {
                typename GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_128x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorLnAlign8RC_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
        }
    } else if(params->epilogueType == 1) { // epilogueRelu
        if(params->precisionType == 2) { // InOut:FP16_FP16
            // MNN_PRINT("gemmbatched0: batch-%d, problem-%d %d %d\n", params->batchSize, params->problemSize[0], params->problemSize[1], params->problemSize[2]);
            if (params->prefeBlockSize == "_64x64x32_")
            {
                typename GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_64x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorReluAlign8RC_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x64x64_")
            {
                typename GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_64x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorReluAlign8RC_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status);
            }
            if (params->prefeBlockSize == "_64x128x32_")
            {
                typename GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_64x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorReluAlign8RC_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x32_")
            {
                typename GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_128x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorReluAlign8RC_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x64_")
            {
                typename GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_128x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorReluAlign8RC_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_256x64x32_")
            {
                typename GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_256x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_256x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorReluAlign8RC_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }

            if (params->prefeBlockSize == "_128x128x32_")
            {
                typename GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_128x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorReluAlign8RC_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
        } else if(params->precisionType == 0) { // InOut:FP16_FP32
            if (params->prefeBlockSize == "_64x64x32_")
            {
                typename GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_64x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorReluAlign8RC_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x64x64_")
            {
                typename GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_64x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorReluAlign8RC_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x128x32_")
            {
                typename GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_64x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorReluAlign8RC_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x32_")
            {
                typename GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_128x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorReluAlign8RC_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x64_")
            {
                typename GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_128x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorReluAlign8RC_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_256x64x32_")
            {
                typename GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_256x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_256x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorReluAlign8RC_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }

            if (params->prefeBlockSize == "_128x128x32_")
            {
                typename GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_128x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorReluAlign8RC_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
        }
    } else if(params->epilogueType == 2) { // epilogueRelu6
        if(params->precisionType == 2) { // InOut:FP16_FP16
            // MNN_PRINT("gemmbatched0: batch-%d, problem-%d %d %d\n", params->batchSize, params->problemSize[0], params->problemSize[1], params->problemSize[2]);
            if (params->prefeBlockSize == "_64x64x32_")
            {
                typename GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_64x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorRelu6Align8RC_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x64x64_")
            {
                typename GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_64x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorRelu6Align8RC_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status);
            }
            if (params->prefeBlockSize == "_64x128x32_")
            {
                typename GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_64x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorRelu6Align8RC_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x32_")
            {
                typename GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_128x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorRelu6Align8RC_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x64_")
            {
                typename GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_128x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorRelu6Align8RC_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_256x64x32_")
            {
                typename GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_256x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_256x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorRelu6Align8RC_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x128x32_")
            {
                typename GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F16 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F16 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_128x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16TensorRelu6Align8RC_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
        } else if(params->precisionType == 0) { // InOut:FP16_FP32
            if (params->prefeBlockSize == "_64x64x32_")
            {
                typename GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_64x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorRelu6Align8RC_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x64x64_")
            {
                typename GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_64x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorRelu6Align8RC_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_64x128x32_")
            {
                typename GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_64x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorRelu6Align8RC_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x32_")
            {
                typename GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_128x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorRelu6Align8RC_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_128x64x64_")
            {
                typename GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x64x64::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x64x64::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_128x64x64.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorRelu6Align8RC_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
            if (params->prefeBlockSize == "_256x64x32_")
            {
                typename GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_256x64x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_256x64x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_256x64x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorRelu6Align8RC_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }

            if (params->prefeBlockSize == "_128x128x32_")
            {
                typename GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x128x32::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)params->ptrOffset[0].first, params->ptrOffset[0].second},  // Ptr + ldm
                    {(ElementInput_F16 *)params->ptrOffset[1].first, params->ptrOffset[1].second},  //  Ptr + ldm
                    {(ElementOutput_F32 *)params->ptrOffset[2].first, params->ptrOffset[2].second},  //  Ptr + ldm  if ldm = 0, vector,
                    {(ElementOutput_F32 *)params->ptrOffset[3].first, params->ptrOffset[3].second},  //  Ptr + ldm
                    {alpha, beta},          // <- tuple of alpha and beta
                    params->coefs[2]};      // splitK

                size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x128x32::get_workspace_size(arguments);
                if(workspace_size != 0 && workspace_ptr == nullptr) {
                    std::shared_ptr<Tensor> workspaceTensor;
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    static_cast<CUDABackend *>(params->backend)->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    workspace_ptr = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_128x128x32.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32TensorRelu6Align8RC_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
                cutlass_check(status); 
            }
        }
    } 
    return;
}

void CutlassGemmTuneCommonExecution::runGemmTensorCoreFloat16Infer(const GemmParamInfo* params) {
    // MNN_PRINT("Run %d %d %d %s\n", params->epilogueType, params->precisionType, params->epilogueVectorize, params->prefeBlockSize.c_str());
    if(params->epilogueType == 0) { // epilogueLinear
        if(params->precisionType == 2) { // InOut:FP16_FP16
            if (params->prefeBlockSize == "_64x64x32_") {
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_64x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x64x64_") {
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_64x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x128x32_") {
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_64x128x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x32_") {
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_128x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x64_") {
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_128x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_256x64x32_") {
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_256x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x128x32_") {
                cutlass::Status status = mGemmF16F16TensorLnAlign8RC_128x128x32();
                cutlass_check(status);
                return;
            }
        } else if(params->precisionType == 0) {
            if (params->prefeBlockSize == "_64x64x32_") {
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_64x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x64x64_") {
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_64x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x128x32_") {
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_64x128x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x32_") {
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_128x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x64_") {
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_128x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_256x64x32_") {
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_256x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x128x32_") {
                cutlass::Status status = mGemmF16F32TensorLnAlign8RC_128x128x32();
                cutlass_check(status);
                return;
            }
        }
    }

    if(params->epilogueType == 1) { // epilogueRelu
        if(params->precisionType == 2) { // InOut:FP16_FP16
            if (params->prefeBlockSize == "_64x64x32_") {
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_64x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x64x64_") {
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_64x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x128x32_") {
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_64x128x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x32_") {
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_128x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x64_") {
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_128x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_256x64x32_") {
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_256x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x128x32_") {
                cutlass::Status status = mGemmF16F16TensorReluAlign8RC_128x128x32();
                cutlass_check(status);
                return;
            }
        } else if(params->precisionType == 0) {
            if (params->prefeBlockSize == "_64x64x32_") {
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_64x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x64x64_") {
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_64x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x128x32_") {
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_64x128x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x32_") {
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_128x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x64_") {
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_128x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_256x64x32_") {
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_256x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x128x32_") {
                cutlass::Status status = mGemmF16F32TensorReluAlign8RC_128x128x32();
                cutlass_check(status);
                return;
            }
        }
    }

    if(params->epilogueType == 2) { // epilogueRelu6
        if(params->precisionType == 2) { // InOut:FP16_FP16
            if (params->prefeBlockSize == "_64x64x32_") {
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_64x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x64x64_") {
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_64x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x128x32_") {
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_64x128x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x32_") {
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_128x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x64_") {
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_128x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_256x64x32_") {
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_256x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x128x32_") {
                cutlass::Status status = mGemmF16F16TensorRelu6Align8RC_128x128x32();
                cutlass_check(status);
                return;
            }
        } else if(params->precisionType == 0) {
            if (params->prefeBlockSize == "_64x64x32_") {
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_64x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x64x64_") {
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_64x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_64x128x32_") {
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_64x128x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x32_") {
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_128x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x64x64_") {
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_128x64x64();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_256x64x32_") {
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_256x64x32();
                cutlass_check(status);
                return;
            }
            if (params->prefeBlockSize == "_128x128x32_") {
                cutlass::Status status = mGemmF16F32TensorRelu6Align8RC_128x128x32();
                cutlass_check(status);
                return;
            }
        }
    }
    MNN_PRINT("Error Not support Gemm Infer now\n");
    MNN_ASSERT(false);
    return;
}

}
}

#endif