#ifndef RASTER_CU_H
#define RASTER_CU_H
#include "core/TensorUtils.hpp"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
namespace MNN {
namespace CUDA {
    void RasterBlit(uint8_t* dest, const uint8_t* src, const Tensor::InsideDescribe::Region& reg, int bytes, CUDARuntime* runtime);
    void PackC4(uint8_t* dest, const uint8_t* src, int inside, int axis, int outside, int bytes, CUDARuntime* runtime);
    void UnpackC4(uint8_t* dest, const uint8_t* src, int inside, int axis, int outside, int bytes, CUDARuntime* runtime);
}
}

#endif