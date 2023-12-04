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
    int insideStride, int axisStride, int axisAlign,
    DivModFast is, DivModFast cs
    ) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmp, x, y, z;
        cs.divmod(i, tmp, y);
        is.divmod(tmp, z, x);
        if (y < axis) {
            int srcOffset = (z * inside + x) * axisAlign + y;// NHWC8 , inside <-> HW, ouside <-> N
            int dstOffset = x * insideStride + y * axisStride + z * inside * axis;
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
        int tmpI = i / axisAlign;
        int y = i % axisAlign;
        int x = tmpI % inside;
        int z = tmpI / inside;

        int srcOffset = (z * inside + x) * axisAlign + y;// NHWC8 , inside <-> HW, ouside <-> N
        int dstOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        }
    }
}

__global__ void UNPACKCOMMON_REARRANGE_half_4(const double *input, double *output,
    int inside, int axis, int outside,
    int insideStride, int axisStride
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER) * PACK_NUMBER / 4;
    int insideAlign = inside / 4;
    int axisNum = axis / 16;
    int insideNum = inside / 32;

    __shared__ double sharedData[128];
    int tid = blockIdx.x;
    int localIdx = threadIdx.x;

    int tmpI = tid / axisNum;
    int y = tid % axisNum;
    int x = tmpI % insideNum;
    int z = tmpI / insideNum;

    int y_mod = localIdx % 4; // [0 ~ 3]
    int x_mod = localIdx / 4; // [0 ~ 31]
    int srcOffset = (z * inside + 32*x+x_mod) * axisAlign + 4*y+y_mod;
    sharedData[localIdx] = input[srcOffset];// [HW_32, C4_4]

    __syncthreads();

    int oy_mod = localIdx % 16; // [0 ~ 15]
    int ox_mod = localIdx / 16; // [0 ~ 7]

    // [4*ox_mod, oy_mod]
    half tmp_data[4];
    tmp_data[0] = ((half*)sharedData)[(4*ox_mod+0) * 16 + oy_mod];
    tmp_data[1] = ((half*)sharedData)[(4*ox_mod+1) * 16 + oy_mod];
    tmp_data[2] = ((half*)sharedData)[(4*ox_mod+2) * 16 + oy_mod];
    tmp_data[3] = ((half*)sharedData)[(4*ox_mod+3) * 16 + oy_mod];

    int dstOffset = (8*x+ox_mod) + (z * axis + (16*y+oy_mod)) * insideAlign;

    output[dstOffset] = ((double*)tmp_data)[0];
}

template<typename T0, typename T1>
__global__ void PACKCOMMON_4(const T0 *input, T1 *output, 
    int inside, int axis, int outside, 
    int insideStride, int axisStride,
    DivModFast is, DivModFast cs
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER/ 4) * PACK_NUMBER / 4;;
    int total = axisAlign * inside * outside;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmp, x, y, z;
        cs.divmod(i, tmp, y);
        is.divmod(tmp, z, x);
        int dstOffset = (z * inside + x) * axisAlign + y;
        int srcOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        } else {
            output[dstOffset] = {0, 0, 0, 0};
        }
    }
}

template<typename T0, typename T1>
__global__ void PACKCOMMON_half_4(const T0 *input, T1 *output,
    int inside, int axis, int outside,
    int insideStride, int axisStride,
    DivModFast is, DivModFast cs
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER/ 4) * PACK_NUMBER / 4;;
    int total = axisAlign * inside * outside;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmp, x, y, z;
        cs.divmod(i, tmp, y);
        is.divmod(tmp, z, x);
        int dstOffset = (z * inside + x) * axisAlign + y;
        int srcOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        } else {
            output[dstOffset] = {0, 0};
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
        int tmpI = i / axisAlign;
        int y = i % axisAlign;
        int x = tmpI % inside;
        int z = tmpI / inside;

        int dstOffset = (z * inside + x) * axisAlign + y;
        int srcOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        } else {
            output[dstOffset] = 0.0;
        }
    }
}

__global__ void PACKCOMMON_REARRANGE_half_4(const double *input, double *output,
    int inside, int axis, int outside,
    int insideStride, int axisStride
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER) * PACK_NUMBER / 4;

    int insideAlign = inside / 4;
    int axisNum = axis / 16;
    int insideNum = inside / 32;

    __shared__ double sharedData[128];
    int tid = blockIdx.x;
    int localIdx = threadIdx.x;

    int tmpI = tid / axisNum;
    int y = tid % axisNum;
    int x = tmpI % insideNum;
    int z = tmpI / insideNum;

    int x_mod = localIdx % 8; // [0 ~ 8]
    int y_mod = localIdx / 8; // [0 ~ 15]
    int srcOffset = (8*x+x_mod) + (z * axis + (16*y+y_mod)) * insideAlign;
    sharedData[localIdx] = input[srcOffset];// [C_16, HW4_8]

    __syncthreads();

    int oy_mod = localIdx % 4; // [0 ~ 3]
    int ox_mod = localIdx / 4; // [0 ~ 31]
    int dstOffset = (z * inside + 32*x+ox_mod) * axisAlign + 4*y+oy_mod;

    // [4*oy_mod, ox_mod]
    half tmp_data[4];
    tmp_data[0] = ((half*)sharedData)[(4*oy_mod+0) * 32 + ox_mod];
    tmp_data[1] = ((half*)sharedData)[(4*oy_mod+1) * 32 + ox_mod];
    tmp_data[2] = ((half*)sharedData)[(4*oy_mod+2) * 32 + ox_mod];
    tmp_data[3] = ((half*)sharedData)[(4*oy_mod+3) * 32 + ox_mod];

    output[dstOffset] = ((double*)tmp_data)[0];
}

void PackBuffer(void* output, const void* input, const PackInfo* info, int bytes, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    
    if (info->axis % 4 == 0 && info->axisStride == 1 && \
        info->insideStride == info->axis) {
        
        int axis_pack = UP_DIV(info->axis, PACK_NUMBER) * PACK_NUMBER / 4;
        DivModFast is(info->inside);
        DivModFast cs(axis_pack);
	if(bytes == 4) {
            PACKCOMMON_4<<<cores, threadNumbers>>>((const int4*)input, (int4*)output,
                    info->inside, info->axis / 4, info->outside, 
                    info->insideStride / 4, info->axisStride, is, cs);
            checkKernelErrors;
	    return;
	}
        if(bytes == 2) {
            PACKCOMMON_half_4<<<cores, threadNumbers>>>((const int2*)input, (int2*)output,
                    info->inside, info->axis / 4, info->outside,
                    info->insideStride / 4, info->axisStride, is, cs);
            checkKernelErrors;
            return;
        }
    }
    if (info->axis % 16 == 0 && info->inside % 32 == 0 && info->insideStride == 1 && info->axisStride == info->inside && bytes == 2) {
        int thread=128;
        int block=info->axis/16 * info->inside/32 * info->outside;
        PACKCOMMON_REARRANGE_half_4<<<block, thread>>>((const double*)input, (double*)output,
                        info->inside, info->axis, info->outside,
                        info->insideStride, info->axisStride);
        checkKernelErrors;
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

    if (info->axis % 4 == 0 && info->axisStride == 1 && info->insideStride == info->axis) {
        int axis_pack = UP_DIV(info->axis, PACK_NUMBER) * PACK_NUMBER / 4;
        DivModFast is(info->inside);
        DivModFast cs(axis_pack);
        const int maxCount = info->inside * axis_pack * info->outside;
        int block_num = runtime->blocks_num(maxCount);
        int block_size = runtime->threads_num();
        int axisAlign = UP_DIV(info->axis / 4, PACK_NUMBER / 4) * PACK_NUMBER / 4;;
        if(bytes == 4) {        
            UNPACKCOMMON_4<<<block_num, block_size>>>((const int4*)input, (int4*)output, 
                            maxCount, info->inside, info->axis / 4, info->outside,
                            info->insideStride / 4, info->axisStride, axisAlign, is, cs);
            checkKernelErrors;
            return;
        }        
        if(bytes == 2) {
            UNPACKCOMMON_4<<<block_num, block_size>>>((const int2*)input, (int2*)output,
                        maxCount, info->inside, info->axis / 4, info->outside,
                        info->insideStride / 4, info->axisStride, axisAlign, is, cs);
            checkKernelErrors;
            return;
        }    
    }
    //printf("unpack size:%d %d %d, stride:%d %d, %p %p\n", info->outside, info->axis, info->inside, info->axisStride, info->insideStride, input, output); 

    if (info->axis % 16 == 0 && info->inside % 32 == 0 && info->insideStride == 1 && info->axisStride == info->inside && bytes == 2) {
        int thread=128;
        int block=info->axis/16 * info->inside/32 * info->outside;
        UNPACKCOMMON_REARRANGE_half_4<<<block, thread>>>((const double*)input, (double*)output,
                        info->inside, info->axis, info->outside,
                        info->insideStride, info->axisStride);
	checkKernelErrors;	
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

// for the following transpose kernels:
// maxCount is num of threads i.e., num of elements of output format
// inChannelPack is num of channel pack of input format
// divOutChannelPack is Div for channel pack of output format

// copy kernel
template<typename T0, typename T1>
__global__ void NCHW_2_NCHW(const T0* input,
                            T1* output,
                            const int maxCount
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        output[index] = (T1)input[index];
    }
}

// NHWC NCHW
template<typename T0, typename T1>
__global__ void NHWC_2_NCHW(const T0* input,
                            T1* output,
                            const int maxCount,
                            const int channel, // redundant parameter
                            const int area,
                            const int inChannelPack,
                            DivModFast divOutChannelPack,
                            DivModFast divArea
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);

        int src_offset = (batch_idx * area + area_idx) * inChannelPack+ chnl_idx;
        output[index] = (T1)input[src_offset];
    }
}

// NHWC8_2_NCHW
template<typename T0, typename T1>
__global__ void NHWC8_2_NCHW(const T0* input,
                             T1* output,
                             const int maxCount,
                             const int channel, // redundant parameter
                             const int area,
                             const int inChannelPack,
                             DivModFast divOutChannelPack,
                             DivModFast divArea
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);

        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        output[index] = (T1)input[src_offset];
    }
}

// C4NHW4_2_NCHW
template<typename T0, typename T1>
__global__ void C4NHW4_2_NCHW(const T0* input,
                             T1* output,
                             const int maxCount,
                             const int channel,
                             const int area,
                             const int inChannelPack, // redundant parameter
                             DivModFast divOutChannelPack,
                             DivModFast divArea
) {
    const int batch = (maxCount / channel) / area;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);

        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int src_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        output[index] = (T1)input[src_offset];
    }
}

// NCHW NHWC
template<typename T0, typename T1>
__global__ void NCHW_2_NHWC(const T0* input,
                            T1* output,
                            const int maxCount,
                            const int channel, // redundant parameter
                            const int area,
                            const int inChannelPack,
                            DivModFast divOutChannelPack,
                            DivModFast divArea
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        int src_offset = (batch_idx * inChannelPack + chnl_idx) * area + area_idx;
        output[index] = (T1)input[src_offset];
    }
}

// NHWC8 NHWC
template<typename T0, typename T1>
__global__ void NHWC8_2_NHWC(const T0* input,
                             T1* output,
                             const int maxCount,
                             const int channel, // redundant parameter
                             const int area,
                             const int inChannelPack,
                             DivModFast divOutChannelPack,
                             DivModFast divArea
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        output[index] = (T1)input[src_offset];
    }
}

// C4NHW4 NHWC
template<typename T0, typename T1>
__global__ void C4NHW4_2_NHWC(const T0* input,
                             T1* output,
                             const int maxCount,
                             const int channel,
                             const int area,
                             const int inChannelPack, // redundant parameter
                             DivModFast divOutChannelPack,
                             DivModFast divArea
) {
    const int batch = (maxCount / channel) / area;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int src_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        output[index] = (T1)input[src_offset];
    }
}

// NHWC NHWC8
template<typename T0, typename T1>
__global__ void NHWC_2_NHWC8(const T0* input,
                             T1* output,
                             const int maxCount,
                             const int channel,
                             const int area,
                             const int inChannelPack,
                             DivModFast divOutChannelPack,
                             DivModFast divArea
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        if(chnl_idx >= channel) {
            output[index] = (T1)0.0f;
            continue;
        }

        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        output[index] = (T1)input[src_offset];
    }
}

// NCHW NHWC8
template<typename T0, typename T1>
__global__ void NCHW_2_NHWC8(const T0* input,
                             T1* output,
                             const int maxCount,
                             const int channel,
                             const int area,
                             const int inChannelPack,
                             DivModFast divOutChannelPack,
                             DivModFast divArea
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        if(chnl_idx >= channel) {
            output[index] = (T1)0.0f;
            continue;
        }

        int src_offset = (batch_idx * inChannelPack + chnl_idx) * area + area_idx;
        output[index] = (T1)input[src_offset];
    }
}

// C4NHW4 NHWC8
template<typename T0, typename T1>
__global__ void C4NHW4_2_NHWC8(const T0* input,
                              T1* output,
                              const int maxCount,
                              const int channel,
                              const int area,
                             const int inChannelPack, // redundant parameter
                              DivModFast divOutChannelPack,
                              DivModFast divArea
) {
    const int batch = (maxCount / (UP_DIV(channel, 8) * 8)) / area;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        if(chnl_idx >= channel) {
            output[index] = (T1)0.0f;
            continue;
        }

        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int src_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        output[index] = (T1)input[src_offset];
    }
}

// NHWC_2_C4NHW4
template<typename T0, typename T1>
__global__ void NHWC_2_C4NHW4(const T0* input,
                               T1* output,
                               const int maxCount,
                               const int channel,
                               const int area,
                               const int inChannelPack,
                               DivModFast divOutChannelPack,
                               DivModFast divArea
) {
    const int batch = (maxCount / (UP_DIV(channel, 4) * 4)) / area;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        // arrange threads arrodring to NHWC4 format
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        int c4_idx = chnl_idx >> 2; // chnl_idx / 4
        int cL_idx = chnl_idx & 3; // chnl_idx % 4
        int dst_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;

        if (chnl_idx >= channel) {
            output[dst_offset] = (T1)0.0f;;
            continue;
        }

        output[dst_offset] = (T1)input[src_offset];
    }
}

// NCHW C4NHW4
template<typename T0, typename T1>
__global__ void NCHW_2_C4NHW4(const T0* input,
                               T1* output,
                               const int maxCount,
                               const int channel,
                               const int area,
                               const int inChannelPack,
                               DivModFast divOutChannelPack,
                               DivModFast divArea
) {
    const int batch = (maxCount / (UP_DIV(channel, 4) * 4)) / area;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        // arrange threads arrodring to NHWC4 format
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        int c4_idx = chnl_idx >> 2; // chnl_idx / 4
        int cL_idx = chnl_idx & 3; // chnl_idx % 4
        int dst_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        int src_offset = (batch_idx * inChannelPack + chnl_idx) * area + area_idx;

        if (chnl_idx >= channel) {
            output[dst_offset] = (T1)0.0f;;
            continue;
        }

        output[dst_offset] = (T1)input[src_offset];
    }
}

// NHWC8 C4NHW4
template<typename T0, typename T1>
__global__ void NHWC8_2_C4NHW4(const T0* input,
                               T1* output,
                               const int maxCount,
                               const int channel,
                               const int area,
                               const int inChannelPack,
                               DivModFast divOutChannelPack,
                               DivModFast divArea
) {
    const int batch = (maxCount / (UP_DIV(channel, 4) * 4)) / area;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        // arrange threads arrodring to NHWC4 format
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        int c4_idx = chnl_idx >> 2; // chnl_idx / 4
        int cL_idx = chnl_idx & 3; // chnl_idx % 4
        int dst_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;;

        output[dst_offset] = (T1)input[src_offset];
    }
}

template<class T0, class T1>
static void insideFormatConvert(T0* input, T1* output, MNN_DATA_FORMAT srcDataFormat, MNN_DATA_FORMAT dstDataFormat, CUDARuntime* runtime, \
    const int area, const int batch, const int channel, const bool srcDevice, const bool dstDevice) {
    DivModFast d_oc(channel);
    DivModFast d_oc4(UP_DIV(channel, 4) * 4);
    DivModFast d_oc8(UP_DIV(channel, 8) * 8);
    DivModFast d_area(area);

    // NCHW NCHW
    // NHWC NHWC
    if ((srcDataFormat == MNN_DATA_FORMAT_NCHW && dstDataFormat == MNN_DATA_FORMAT_NCHW) || \
        (srcDataFormat == MNN_DATA_FORMAT_NHWC && dstDataFormat == MNN_DATA_FORMAT_NHWC)) {
        const int maxCount = batch * area * channel;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NCHW_2_NCHW<T0, T1><<<block_num, block_size>>>(input, output, maxCount);
        checkKernelErrors;
        return;
    }

    // NC4HW4 NC4HW4
    if (srcDataFormat == MNN_DATA_FORMAT_NC4HW4 && dstDataFormat == MNN_DATA_FORMAT_NC4HW4) {
        if(!srcDevice && dstDevice) {
            const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            C4NHW4_2_NHWC8<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 4) * 4, d_oc8, d_area);
            checkKernelErrors;
        } else if (srcDevice && !dstDevice) {
            const int maxCount = batch * area * UP_DIV(channel, 4) * 4;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            NHWC8_2_C4NHW4<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 8) * 8, d_oc4, d_area);
            checkKernelErrors;
        } else {
            const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            NCHW_2_NCHW<T0, T1><<<block_num, block_size>>>(input, output, maxCount);
            checkKernelErrors;
        }
        return;
    }

    // NHWC NCHW
    if (srcDataFormat == MNN_DATA_FORMAT_NHWC && dstDataFormat == MNN_DATA_FORMAT_NCHW) {
        const int maxCount = batch * area * channel;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NHWC_2_NCHW<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, channel, d_oc, d_area);
        checkKernelErrors;
        return;
    }

    // NC4HW4 NCHW
    if (srcDataFormat == MNN_DATA_FORMAT_NC4HW4 && dstDataFormat == MNN_DATA_FORMAT_NCHW) {
        if (!srcDevice) {
            const int maxCount = batch * area * channel;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            C4NHW4_2_NCHW<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 4) * 4, d_oc, d_area);
            checkKernelErrors;
        } else {
            const int maxCount = batch * area * channel;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            NHWC8_2_NCHW<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 8) * 8, d_oc, d_area);
            checkKernelErrors;
        }
        return;
    }

    // NCHW NHWC
    if (srcDataFormat == MNN_DATA_FORMAT_NCHW && dstDataFormat == MNN_DATA_FORMAT_NHWC) {
        const int maxCount = batch * area * channel;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NCHW_2_NHWC<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, channel, d_oc, d_area);
        checkKernelErrors;
        return;
    }

    // NC4HWC4 NHWC
    if (srcDataFormat == MNN_DATA_FORMAT_NC4HW4 && dstDataFormat == MNN_DATA_FORMAT_NHWC) {
        if (!srcDevice) {
            const int maxCount = batch * area * channel;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            C4NHW4_2_NHWC<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 4) * 4, d_oc, d_area);
            checkKernelErrors;
        } else {
            const int maxCount = batch * area * channel;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            NHWC8_2_NHWC<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 8) * 8, d_oc, d_area);
            checkKernelErrors;
        }
        return;
    }

    // NCHW NC4HW4
    if(srcDataFormat == MNN_DATA_FORMAT_NCHW && dstDataFormat == MNN_DATA_FORMAT_NC4HW4) {
        if (!dstDevice) {
            const int maxCount = batch * area * UP_DIV(channel, 4) * 4;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            NCHW_2_C4NHW4<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, channel, d_oc4, d_area);
            checkKernelErrors;
        } else {
            const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            NCHW_2_NHWC8<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, channel, d_oc8, d_area);
            checkKernelErrors;
        }
        return;
    }

    // NHWC NC4HW4
    if(srcDataFormat == MNN_DATA_FORMAT_NHWC && dstDataFormat == MNN_DATA_FORMAT_NC4HW4) {
        if (!dstDevice) {
            const int maxCount = batch * area * UP_DIV(channel, 4) * 4;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            NHWC_2_C4NHW4<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, channel, d_oc4, d_area);
            checkKernelErrors;
        } else {
            const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            NHWC_2_NHWC8<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, channel, d_oc8, d_area);
            checkKernelErrors;
        }
        return;
    }

    MNN_ERROR("CUDA backend doesn't support the format conversion.\n");
    MNN_ASSERT(false);
    return;
}

void FormatConvert(void* output, void* input, MNN_DATA_FORMAT srcDataFormat, MNN_DATA_FORMAT dstDataFormat, CUDARuntime* runtime, \
    const int area, const int batch, const int channel, const Tensor* srcTensor, int precision, bool srcDevice, bool dstDevice) {
    if(batch == 0 || area == 0 || channel == 0) {
        MNN_PRINT("Error: formatConvert size batch:%d - plane:%d - channel:%d, format:%d->%d, device:%d->%d\n", batch, area, channel, srcDataFormat, dstDataFormat, srcDevice, dstDevice);
        return;
    }

    bool isFp16 = (precision == 2) && (halide_type_float == srcTensor->getType().code);
    bool isBf16 = (precision == 3) && (halide_type_float == srcTensor->getType().code);

    // int8 case
    auto des = TensorUtils::getDescribe(srcTensor);
    if ((des->quantAttr.get() != nullptr && des->type == DataType_DT_INT8) || srcTensor->getType().bits == 8) {
        insideFormatConvert<int8_t, int8_t>((int8_t *)input, (int8_t *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
        return;
    }

    // FP case
    if(!srcDevice) {
        if(isFp16) {
            insideFormatConvert<float, half>((float *)input, (half *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
        } else if(isBf16) {
            #ifdef ENABLE_CUDA_BF16
            insideFormatConvert<float, __nv_bfloat16>((float *)input, (__nv_bfloat16 *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
            #endif
        } else {
            insideFormatConvert<float, float>((float *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
        }
    } else if(!dstDevice) {
        if(isFp16) {
            insideFormatConvert<half, float>((half *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
        } else if(isBf16) {
            #ifdef ENABLE_CUDA_BF16
            insideFormatConvert<__nv_bfloat16, float>((__nv_bfloat16 *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
            #endif
        } else {
            insideFormatConvert<float, float>((float *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
        }
    } else {
        if(isFp16) {
            insideFormatConvert<half, half>((half *)input, (half *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
        } else if(isBf16) {
            #ifdef ENABLE_CUDA_BF16
            insideFormatConvert<__nv_bfloat16, __nv_bfloat16>((__nv_bfloat16 *)input, (__nv_bfloat16 *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
            #endif
        } else {
            insideFormatConvert<float, float>((float *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel, srcDevice, dstDevice);
        }
    }
    return;
}


};
};
