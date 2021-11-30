#ifndef RASTER_CU_H
#define RASTER_CU_H
#include "core/TensorUtils.hpp"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
namespace MNN {
namespace CUDA {
    void RasterBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int bytes, CUDARuntime* runtime);
    void FuseRasterBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int fuseNum, void* sliceOffset, int bytes, CUDARuntime* runtime);
    void PackC4(uint8_t* dest, const uint8_t* src, int inside, int axis, int outside, int bytes, CUDARuntime* runtime);
    void UnpackC4(uint8_t* dest, const uint8_t* src, int inside, int axis, int outside, int bytes, CUDARuntime* runtime);
    void BlitWithIndice(uint8_t* dest, const uint8_t* src, const int32_t* dstIndices, const int32_t* srcIndices, int dstUseIndice, int srcUseIndice, int loopCount, int dstStep, int srcStep, int srcLimit, const Tensor::InsideDescribe::Region& reg, int bytes, CUDARuntime* runtime);
    void UnaryBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int bytes, CUDARuntime* runtime, int opType);
    void BinaryBlit(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, halide_type_t type, CUDARuntime* runtime, int opType);
}
}

#endif