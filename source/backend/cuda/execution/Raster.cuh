#ifndef RASTER_CU_H
#define RASTER_CU_H
#include "core/TensorUtils.hpp"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
namespace MNN {
namespace CUDA {
    void RasterBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int bytes, CUDARuntime* runtime);
    void FuseRasterBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int fuseNum, void* sliceOffset, int bytes, CUDARuntime* runtime, int offsetunit);
    void BlitWithIndice(uint8_t* dest, const uint8_t* src, const int32_t* dstIndices, const int32_t* srcIndices, int dstUseIndice, int srcUseIndice, int loopCount, int dstStep, int srcStep, int srcLimit, const Tensor::InsideDescribe::Region& reg, int bytes, CUDARuntime* runtime);
    void UnaryBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int bytes, CUDARuntime* runtime, int opType);
    void BinaryBlit(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, halide_type_t type, CUDARuntime* runtime, int opType, int activationType = 0);
    void BinaryBlitFuse(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, halide_type_t type, CUDARuntime* runtime, int opType, int fuseType = 0);

    // Offset: 8 * fuseNum, first 4 for src: limitX, limitY, limitZ, offset, second 4 for dst
    struct FuseRegion {
        int32_t size[3] = {1, 1, 1};
        int32_t srcStride[3] = {0, 0, 0};
        int32_t dstStride[3] = {0, 0, 0};
        int fuseNumber = 0;
    };
    void FuseRasterBlitFloatToHalf(uint8_t* output, const uint8_t* input, const FuseRegion* info, void* sliceOffset, CUDARuntime* runtime);
    void FuseRasterBlitHalfToFloat(uint8_t* output, const uint8_t* input, const FuseRegion* info, void* sliceOffset, CUDARuntime* runtime);
    void FuseRasterBlitFloatToFloat(uint8_t* output, const uint8_t* input, const FuseRegion* info, void* sliceOffset, CUDARuntime* runtime);
    void FuseRasterBlitCommon(uint8_t* output, const uint8_t* input, const FuseRegion* info, void* sliceOffset, CUDARuntime* runtime, int bytes);

}
}

#endif