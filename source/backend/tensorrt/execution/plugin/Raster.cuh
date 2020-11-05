#ifndef RASTER_CU_H
#define RASTER_CU_H
#include <cuda_runtime_api.h>
#include "core/TensorUtils.hpp"
#include "CommonPlugin.hpp"

namespace MNN {
cudaError_t RasterBlit(nvinfer1::DataType dataType, uint8_t* dest, const uint8_t* src, const Tensor::InsideDescribe::Region& reg, int bytes,
                cudaStream_t stream);
}

#endif