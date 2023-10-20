#ifdef ENABLE_CUDA_TUNE_PARAM

#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "backend/cuda/core/CUDABackend.hpp"
#include "CutlassGemmBatchedParamTune.hpp"
#include "CutlassGemmParamTune.hpp"
#include <cuda_runtime.h>
// #define MNN_CUDA_TUNE_LOG

namespace MNN {
namespace CUDA {

struct GemmParamInfo {
    // MxNxK
    int32_t problemSize[3] = {1, 1, 1};
    // 0 -> Gemm, 1~N -> BatchGemm
    int32_t batchSize = 0;
    // [0]->A, [1]->B, [2]->bias, [3]->output
    std::pair<void *, int32_t> ptrOffset[4];
    int32_t batchOffset[4];
    // [0]->alpha, [1]->beta, [2]->splitK
    int32_t coefs[3];
    // 0 -> RowColumn, 1 -> RowRow
    int32_t layout;
    bool epilogueVectorize;
    // 0 -> Linear, 1 -> Relu, 2 -> Relu6
    int32_t epilogueType;
    // In_Out: 0 -> FP16_FP32, 1 -> FP32_FP32, 2 -> FP16_FP16
    int32_t precisionType;
    std::string prefeBlockSize;
    Backend* backend;
};

void getGemmBatchedTensorCoreFloat16Param(GemmParamInfo* params);
void getGemmTensorCoreFloat16Param(GemmParamInfo* params);

}
}

#endif