#include "TensorCoreGemm.cuh"
namespace MNN {
namespace CUDA {

void GemmPackedFullMain(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, void *c, const half *a, const half *b, const half* biasPtr, int bytes);
void GemmPacked16x32(CUDARuntime* runtime, const MatMulParam* cpuParam, const MatMulParam* param, void *c, const half *a, const half *b, const half* biasPtr, int bytes);
}
}