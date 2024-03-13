#ifdef ENABLE_CUDA_TUNE_PARAM
#include "CutlassGemmTune.hpp"
namespace MNN {
namespace CUDA {
//#define MNN_CUDA_TUNE_LOG

void getGemmTensorCoreFloat16Param(GemmParamInfo* params) {
    MNN_ASSERT(params->batchSize == 0);

    auto& tunedBlockWarpShape = static_cast<CUDABackend *>(params->backend)->getCUDARuntime()->getTunedBlockWarpShape();
    std::vector<int32_t> info = {(int32_t)params->layout, (int32_t)params->epilogueVectorize, (int32_t)params->precisionType};
    std::vector<uint32_t> problem = {(uint32_t)params->batchSize, (uint32_t)params->problemSize[0], (uint32_t)params->problemSize[1], (uint32_t)params->problemSize[2]};
    auto key = std::make_pair(info, problem);

    if (tunedBlockWarpShape.find(key) != tunedBlockWarpShape.end()) {
        #ifdef MNN_CUDA_TUNE_LOG
        MNN_PRINT("getGemmTensorCoreFloat16Param Found! prefer:%s\n", tunedBlockWarpShape[key].first.c_str());
        #endif
        params->prefeBlockSize = tunedBlockWarpShape[key].first;
        return;
    }

    cudaEvent_t events[14];
    cudaError_t result;
    for (auto & event : events) {
        result = cudaEventCreate(&event);
        if (result != cudaSuccess) {
            MNN_PRINT("Failed to create CUDA event, %s.\n", cudaGetErrorString(result));
        }
    }
  
    ElementComputeEpilogue alpha = ElementComputeEpilogue(params->coefs[0]);
    ElementComputeEpilogue beta = ElementComputeEpilogue(params->coefs[1]);

    // Split K dimension into 1 partitions
    cutlass::gemm::GemmCoord problem_size(params->problemSize[0], params->problemSize[1], params->problemSize[2]);// m n k
    void* workspace_ptr = nullptr;

    #ifdef MNN_CUDA_TUNE_LOG
    MNN_PRINT("gemm: batch-%d, problem-%d %d %d. %d %d\n", params->batchSize, params->problemSize[0], params->problemSize[1], params->problemSize[2], params->epilogueType, params->precisionType);
    #endif
    const int warmup = 10;
    const int loop = 100;
    int event_index = 0;
    // us
    float costTime_64x128x32, costTime_64x64x32, costTime_64x64x64, costTime_128x64x32, costTime_128x64x64, costTime_256x64x32, costTime_128x128x32;

    // Different epilogue type(linear/relu/relu6) regard as the same
    if(params->precisionType == 2) { // InOut:FP16_FP16
        // BatchGemm + RowColumn + AlignTensor
        // MNN_PRINT("gemmbatched0: batch-%d, problem-%d %d %d\n", params->batchSize, params->problemSize[0], params->problemSize[1], params->problemSize[2]);
        {
            GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x32 gemmBatched_64x64x32;

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
            cutlass::Status status = gemmBatched_64x64x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_64x64x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_64x64x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_64x64x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x32 : %f ms\n", costTime_64x64x32);
            #endif
        }

        {
            GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x64 gemmBatched_64x64x64;

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
            cutlass::Status status = gemmBatched_64x64x64.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_64x64x64();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_64x64x64();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_64x64x64, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x64 : %f ms\n", costTime_64x64x64);
            #endif
        }

        {
            GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x128x32 gemmBatched_64x128x32;

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
            cutlass::Status status = gemmBatched_64x128x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_64x128x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_64x128x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_64x128x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x128x32 : %f ms\n", costTime_64x128x32);
            #endif
        }
        {
            GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x32 gemmBatched_128x64x32;

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
            cutlass::Status status = gemmBatched_128x64x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_128x64x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_128x64x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_128x64x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x32 : %f ms\n", costTime_128x64x32);
            #endif
        }

        {
            GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x64 gemmBatched_128x64x64;

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
            cutlass::Status status = gemmBatched_128x64x64.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_128x64x64();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_128x64x64();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_128x64x64, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x64 : %f ms\n", costTime_128x64x64);
            #endif
        }

        {
            GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_256x64x32 gemmBatched_256x64x32;

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
            cutlass::Status status = gemmBatched_256x64x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_256x64x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_256x64x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_256x64x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_256x64x32 : %f ms\n", costTime_256x64x32);
            #endif
        }

        {
            GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x128x32 gemmBatched_128x128x32;

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
            cutlass::Status status = gemmBatched_128x128x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_128x128x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_128x128x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_128x128x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x128x32 : %f ms\n", costTime_128x128x32);
            #endif
        }
    } else if(params->precisionType == 0) { // InOut:FP16_FP32
        {
            GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x32 gemmBatched_64x64x32;

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
            cutlass::Status status = gemmBatched_64x64x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_64x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_64x64x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_64x64x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_64x64x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x32 : %f ms\n", costTime_64x64x32);
            #endif
        }

        {
            GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x64 gemmBatched_64x64x64;

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
            cutlass::Status status = gemmBatched_64x64x64.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_64x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_64x64x64();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_64x64x64();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_64x64x64, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x64 : %f ms\n", costTime_64x64x64);
            #endif
        }

        {
            GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x128x32 gemmBatched_64x128x32;

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
            cutlass::Status status = gemmBatched_64x128x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_64x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_64x128x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_64x128x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_64x128x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x128x32 : %f ms\n", costTime_64x128x32);
            #endif
        }
        {
            GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x32 gemmBatched_128x64x32;

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
            cutlass::Status status = gemmBatched_128x64x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_128x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_128x64x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_128x64x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_128x64x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x32 : %f ms\n", costTime_128x64x32);
            #endif
        }

        {
            GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x64 gemmBatched_128x64x64;

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
            cutlass::Status status = gemmBatched_128x64x64.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_128x64x64.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_128x64x64();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_128x64x64();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_128x64x64, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x64 : %f ms\n", costTime_128x64x64);
            #endif
        }

        {
            GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_256x64x32 gemmBatched_256x64x32;

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
            cutlass::Status status = gemmBatched_256x64x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_256x64x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_256x64x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_256x64x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_256x64x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_256x64x32 : %f ms\n", costTime_256x64x32);
            #endif
        }

        {
            GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x128x32 gemmBatched_128x128x32;

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
            cutlass::Status status = gemmBatched_128x128x32.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemmBatched_128x128x32.initialize(arguments, (uint8_t *)workspace_ptr);
            cutlass_check(status); 

            for(int i = 0; i < warmup; i++) {
                cutlass::Status status = gemmBatched_128x128x32();
                cutlass_check(status);
            }
            cudaDeviceSynchronize();

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            for(int i = 0; i < loop; i++) {
                cutlass::Status status = gemmBatched_128x128x32();
                cutlass_check(status);
            }

            result = cudaEventRecord(events[event_index++]);
            if (result != cudaSuccess) {
                MNN_PRINT("Failed to record start event, %s.\n", cudaGetErrorString(result));
            }

            cudaDeviceSynchronize();
            cudaEventElapsedTime(&costTime_128x128x32, events[event_index-2], events[event_index-1]);
            #ifdef MNN_CUDA_TUNE_LOG
            MNN_PRINT("GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x128x32 : %f ms\n", costTime_128x128x32);
            #endif
        }
    }

    std::vector<uint32_t> times;
    std::string preferBlock;
    uint32_t time_64x64x32  = (uint32_t)((1000 * costTime_64x64x32) / loop);
    times.push_back(time_64x64x32);
    uint32_t time_64x64x64  = (uint32_t)((1000 * costTime_64x64x64) / loop);
    times.push_back(time_64x64x64);
    uint32_t time_64x128x32  = (uint32_t)((1000 * costTime_64x128x32) / loop);
    times.push_back(time_64x128x32);
    uint32_t time_128x64x32 = (uint32_t)((1000 * costTime_128x64x32) / loop);
    times.push_back(time_128x64x32);
    uint32_t time_128x64x64 = (uint32_t)((1000 * costTime_128x64x64) / loop);
    times.push_back(time_128x64x64);
    uint32_t time_256x64x32 = (uint32_t)((1000 * costTime_256x64x32) / loop);
    times.push_back(time_256x64x32);
    uint32_t time_128x128x32 = (uint32_t)((1000 * costTime_128x128x32) / loop);
    times.push_back(time_128x128x32);
    std::sort(times.begin(), times.end());

    if(time_64x64x32 == times[0]) {
        preferBlock = "_64x64x32_";
    } else if(time_64x64x64 == times[0]) {
        preferBlock = "_64x64x64_";
    } else if(time_128x64x32 == times[0]) {
        preferBlock = "_128x64x32_";
    } else if(time_128x64x64 == times[0]) {
        preferBlock = "_128x64x64_";
    } else if(time_256x64x32 == times[0]) {
        preferBlock = "_256x64x32_";
    } else if(time_128x128x32 == times[0]) {
        preferBlock = "_128x128x32_";
    } else if(time_64x128x32 == times[0]) {
        preferBlock = "_64x128x32_";
    } else {
        MNN_PRINT("param blockSize assign error, please check\n");
    }
    params->prefeBlockSize = preferBlock;

    #ifdef MNN_CUDA_TUNE_LOG
    static uint32_t total_64x128x32 = 0, total_64x64x32 = 0, total_64x64x64 = 0, total_128x64x32 = 0, total_128x64x64 = 0, total_256x64x32 = 0, total_128x128x32 = 0, total_min = 0; 
    total_64x64x32 += time_64x64x32;
    total_64x64x64 += time_64x64x64;
    total_64x128x32 += time_64x128x32;
    total_128x64x32 += time_128x64x32;
    total_128x64x64 += time_128x64x64;
    total_256x64x32 += time_256x64x32;
    total_128x128x32 += time_128x128x32;
    total_min += times[0];

    MNN_PRINT("Gemm layer time:%d %d %d %d %d %d %d, mintime:%d\ntotal time: %d %d %d %d %d %d %d, mintime:%d\n", time_64x128x32, time_64x64x32, time_64x64x64, time_128x64x32, time_128x64x64, time_256x64x32, time_128x128x32, times[0], total_64x128x32, total_64x64x32, total_64x64x64, total_128x64x32, total_128x64x64, total_256x64x32, total_128x128x32, total_min);
    #endif

    for (auto & event : events) {
        cudaEventDestroy(event);
    }
    if(tunedBlockWarpShape.find(key) == tunedBlockWarpShape.end()) {
        tunedBlockWarpShape.insert(std::make_pair(key, std::make_pair(preferBlock, times[0])));
    }
}

}
}

#endif
