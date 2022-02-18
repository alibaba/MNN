//
//  Transpose.cu
//  MNN
//
//  Created by MNN on b'2021/12/09'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Transpose.cuh"
#include "core/Macro.h"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"
namespace MNN {
namespace CUDA {

template<typename T0, typename T1>
__global__ void UNPACKCOMMON_4(const T0 *input, T1 *output,
    const int total, int inside, int axis, int outside,
    int insideStride, int axisStride,
    DivModFast is, DivModFast os
    ) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmpI = i >> 2;
        int yR = i & 3;
        int x, tmp, yC, z;
        is.divmod(tmpI, tmp, x);
        os.divmod(tmp, yC, z);
        int y = (yC << 2) + yR;
        int srcOffset = ((z * inside + yC * inside * outside + x) << 2) + yR;
        int dstOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        }
    }
}

template<typename T0, typename T1>
__global__ void UNPACKCOMMON(const T0 *input, T1 *output, 
    int inside, int axis, int outside, 
    int insideStride, int axisStride
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER) * PACK_NUMBER;;
    int total = axisAlign * inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmpI = i >> 4;
        int yR = i & 15;
        int x = tmpI % inside;
        int tmp = tmpI / inside;
        int yC = tmp / outside;
        int z = tmp % outside;
        int y = yC * PACK_NUMBER + yR;
        int srcOffset = PACK_NUMBER * (z * inside + yC * inside * outside + x) + yR;
        int dstOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        }
    }
}

template<typename T0, typename T1>
__global__ void PACKCOMMON_4(const T0 *input, T1 *output, 
    int inside, int axis, int outside, 
    int insideStride, int axisStride
    ) {
    int axisAlign = UP_DIV(axis, 4) * 4;;
    int total = axisAlign * inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmpI = i >> 2;
        int yR = i & 3;
        int x = tmpI % inside;
        int tmp = tmpI / inside;
        int yC = tmp / outside;
        int z = tmp % outside;
        int y = yC * 4 + yR;
        int dstOffset = 4 * (z * inside + yC * inside * outside + x) + yR;
        int srcOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        } else {
            output[dstOffset] = {0, 0, 0, 0};
        }
    }
}
template<typename T0, typename T1>
__global__ void PACKCOMMON(const T0 *input, T1 *output,
    int inside, int axis, int outside, 
    int insideStride, int axisStride
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER) * PACK_NUMBER;;
    int total = axisAlign * inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmpI = i >> 4;
        int yR = i & 15;
        int x = tmpI % inside;
        int tmp = tmpI / inside;
        int yC = tmp / outside;
        int z = tmp % outside;
        int y = yC * PACK_NUMBER + yR;
        int dstOffset = PACK_NUMBER * (z * inside + yC * inside * outside + x) + yR;
        int srcOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        } else {
            output[dstOffset] = 0.0;
        }
    }
}

void PackBuffer(void* output, const void* input, const PackInfo* info, int bytes, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    if (info->axis % 4 == 0 && info->axisStride == 1 && \
        bytes == 4 && info->insideStride == info->axis) {
        PACKCOMMON_4<<<cores, threadNumbers>>>((const int4*)input, (int4*)output,
                    info->inside, info->axis / 4, info->outside, 
                    info->insideStride / 4, info->axisStride);
        return;
    }
    switch (bytes) {
        case 4:
            PACKCOMMON<<<cores, threadNumbers>>>((const float*)input, (float*)output, 
                        info->inside, info->axis, info->outside, 
                        info->insideStride, info->axisStride);
            break;
        case 2:
            PACKCOMMON<<<cores, threadNumbers>>>((const half*)input, (half*)output, 
                        info->inside, info->axis, info->outside, 
                        info->insideStride, info->axisStride);
            break;
        case 1:
            PACKCOMMON<<<cores, threadNumbers>>>((const int8_t*)input, (int8_t*)output, 
                        info->inside, info->axis, info->outside, 
                        info->insideStride, info->axisStride);
            break;
        default:
            break;
    }
}
void UnpackBuffer(void* output, const void* input, const PackInfo* info, int bytes, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;

    if (info->axis % 4 == 0 && info->axisStride == 1 && bytes == 4 && info->insideStride == info->axis) {
        DivModFast is(info->inside);
        DivModFast os(info->outside);
        const int maxCount = info->inside * UP_DIV(info->axis / 4, 4) * 4 * info->outside;
        int block_num = runtime->blocks_num(maxCount);
        int block_size = runtime->threads_num();
        UNPACKCOMMON_4<<<block_num, block_size>>>((const int4*)input, (int4*)output, 
                        maxCount, info->inside, info->axis / 4, info->outside,
                        info->insideStride / 4, info->axisStride, is, os);
        return;
    }
    switch (bytes) {
        case 4:
            UNPACKCOMMON<<<cores, threadNumbers>>>((const float*)input, (float*)output, 
                        info->inside, info->axis, info->outside, 
                        info->insideStride, info->axisStride);
            break;
        case 2:
            UNPACKCOMMON<<<cores, threadNumbers>>>((const half*)input, (half*)output, 
                        info->inside, info->axis, info->outside, 
                        info->insideStride, info->axisStride);
            break;
        case 1:
            UNPACKCOMMON<<<cores, threadNumbers>>>((const int8_t*)input, (int8_t*)output, 
                        info->inside, info->axis, info->outside, 
                        info->insideStride, info->axisStride);
            break;
        default:
            break;
    }
}

void PackFP32ToFP16(void* output, const void* input, const PackInfo* info, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    PACKCOMMON<<<cores, threadNumbers>>>((const float*)input, (half*)output, 
                info->inside, info->axis, info->outside, 
                info->insideStride, info->axisStride);
}
void PackFP16ToFP32(void* output, const void* input, const PackInfo* info, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    PACKCOMMON<<<cores, threadNumbers>>>((const half*)input, (float*)output, 
                info->inside, info->axis, info->outside, 
                info->insideStride, info->axisStride);
}

void UnpackFP16ToFP32(void* output, const void* input, const PackInfo* info, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    UNPACKCOMMON<<<cores, threadNumbers>>>((const half*)input, (float*)output, 
                    info->inside, info->axis, info->outside, 
                    info->insideStride, info->axisStride);
}
void UnpackFP32ToFP16(void* output, const void* input, const PackInfo* info, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    UNPACKCOMMON<<<cores, threadNumbers>>>((const float*)input, (half*)output, 
                    info->inside, info->axis, info->outside, 
                    info->insideStride, info->axisStride);
}



template<typename T>
__global__ void TRANSPOSE(const T *input, T *output, const TransposeParam* param) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < param->total) {
        int x = i % param->dims[0];
        int tmp = i / param->dims[0];
        int y = tmp % param->dims[1];
        int z = tmp / param->dims[1];
        int srcOffset = param->srcStride * z + y + x * param->dims[2];
        int dstOffset = param->dstStride * z + x + y * param->dims[3];
        output[dstOffset] = input[srcOffset];
    }
}
#define LOCAL_DIM 8

template <typename T>
__global__ void TRANSPOSE_LOCAL(const T* input, T *output, const TransposeParam* param) {
    __shared__ T localM[LOCAL_DIM][LOCAL_DIM + 1];
    int num = blockIdx.z;
    for (int n = num; n < param->size; n += gridDim.z) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < param->dims[0] && y < param->dims[1]) {
            int offset                      = n * param->srcStride + x * param->dims[2] + y;
            localM[threadIdx.y][threadIdx.x] = input[offset];
        }
        __syncthreads();
        x = blockIdx.y * blockDim.y + threadIdx.x;
        y = blockIdx.x * blockDim.x + threadIdx.y;
        if (x < param->dims[1] && y < param->dims[0]) {
            int offset = n * param->dstStride + x * param->dims[3] + y;
            output[offset] = localM[threadIdx.x][threadIdx.y];
        }
    }
}

void Transpose(uint8_t* output, const uint8_t* input, const TransposeParam* cpuParam, const TransposeParam* gpuRegion, int bytes, CUDARuntime* runtime) {
    int count = cpuParam->total;
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    auto out = output + bytes * cpuParam->dstOffset;
    auto inp = input + bytes * cpuParam->srcOffset;
    if (runtime->prop().maxThreadsPerBlock >= LOCAL_DIM * LOCAL_DIM && (cpuParam->dims[0] >= LOCAL_DIM || cpuParam->dims[1] >= LOCAL_DIM)) {
        dim3 localSize(LOCAL_DIM, LOCAL_DIM, 1);
        //printf("%d, %d - %d, %d - %d\n", cpuParam->size, cpuParam->dims[0], cpuParam->dims[1], cpuParam->dims[2], cpuParam->dims[3]);
        int globalZ = ALIMIN(runtime->prop().multiProcessorCount, cpuParam->size);
        dim3 globalSize(UP_DIV(cpuParam->dims[0], LOCAL_DIM), UP_DIV(cpuParam->dims[1], LOCAL_DIM), globalZ);
        switch (bytes) {
            case 4:
                TRANSPOSE_LOCAL<<<globalSize, localSize>>>((const float *)inp, (float *)out, gpuRegion);
                break;
            case 2:
                TRANSPOSE_LOCAL<<<globalSize, localSize>>>((const half *)inp, (half *)out, gpuRegion);
                break;
            case 1:
                TRANSPOSE_LOCAL<<<globalSize, localSize>>>((const int8_t *)inp, (int8_t *)out, gpuRegion);
                break;
            default:
                break;
        }
        return;
    }
    switch (bytes) {
        case 4:
            TRANSPOSE<<<block_num, threads_num>>>((int*)inp, (int*)out, gpuRegion);
            break;
        case 2:
            TRANSPOSE<<<block_num, threads_num>>>((int16_t*)inp, (int16_t*)out, gpuRegion);
            break;
        case 1:
            TRANSPOSE<<<block_num, threads_num>>>((int8_t*)inp, (int8_t*)out, gpuRegion);
            break;
        default:
            break;
    }
}

};
};