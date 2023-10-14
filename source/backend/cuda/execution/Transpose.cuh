//
//  Transpose.cuh
//  MNN
//
//  Created by MNN on b'2021/12/09'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Transpose_cuh
#define Transpose_chu
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

struct PackInfo {
    int outside;
    int inside;
    int axis;
    int unit;
    int insideStride;
    int axisStride;
};
void UnpackBuffer(void* output, const void* input, const PackInfo* info, int bytes, CUDARuntime* runtime);
void PackBuffer(void* output, const void* input, const PackInfo* info, int bytes, CUDARuntime* runtime);
void PackFP16ToFP32(void* output, const void* input, const PackInfo* info, CUDARuntime* runtime);
void PackFP32ToFP16(void* output, const void* input, const PackInfo* info, CUDARuntime* runtime);
void UnpackFP16ToFP32(void* output, const void* input, const PackInfo* info, CUDARuntime* runtime);
void UnpackFP32ToFP16(void* output, const void* input, const PackInfo* info, CUDARuntime* runtime);

void FormatConvert(void* output, void* input, MNN_DATA_FORMAT srcDataFormat, MNN_DATA_FORMAT dstDataFormat, CUDARuntime* runtime, \
    const int area, const int batch, const int channel, const Tensor* srcTensor, int precision, bool srcDevice, bool dstDevice);

struct TransposeParam {
    int dims[4];
    int srcOffset;
    int srcStride;
    int dstOffset;
    int dstStride;
    int size;
    int total;
};
void Transpose(uint8_t* output, const uint8_t* input, const TransposeParam* cpuParam, const TransposeParam* gpuRegion, int bytes, CUDARuntime* runtime);

}
}

#endif
