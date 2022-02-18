#ifndef TENSORCORE_GEMM_CUH
#define TENSORCORE_GEMM_CUH

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include <float.h>
#define MATMULPACK 16
#define MATMULPACK2 (MATMULPACK * MATMULPACK)
namespace MNN {
namespace CUDA {

struct MatMulParam {
    int elh[3];
    int elhPack[3];
    int aStride[3];
    int bStride[3];
    int cStride[3];

    // Outside E, Outside L, Inside
    int aPStride[3];

    // Outside H, Outside L, Inside
    int bPStride[3];

    int batch = 1;
    float minValue = -FLT_MAX;
    float maxValue = FLT_MAX;
};
void GemmPrepareRerange(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, const void* A, __half* AP, const void* B, __half* BP, int bytes);
void GemmPackedMain(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, void *c, const half *a, const half *b, const void* biasPtr, int bytes, bool transposeA, bool transposeB);

}
}
#endif