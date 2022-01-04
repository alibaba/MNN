#ifndef TENSORCORE_GEMM_CUH
#define TENSORCORE_GEMM_CUH

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include <float.h>
#define MATMULPACK 16
namespace MNN {
namespace CUDA {

struct MatMulParam {
    int elh[3];
    int elhPack[3];
    int aStride[3];
    int bStride[3];
    int cStride[3];
    int split[3];// a, b, c can split e / h in l
    float minValue = -FLT_MAX;
    float maxValue = FLT_MAX;
};
void GemmPrepareRerange(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, const float* A, __half* AP, const float* B, __half* BP);
void GemmPackedMain(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, float *c, const half *a, const half *b, const float* biasPtr);
}
}
#endif