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
    DivModFast is, DivModFast cs
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER/ 4) * PACK_NUMBER / 4;;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmp, x, y, z;
        cs.divmod(i, tmp, y);
        is.divmod(tmp, z, x);
        int srcOffset = (z * inside + x) * axisAlign + y;// NHWC8 , inside <-> HW, ouside <-> N
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

void PackBuffer(void* output, const void* input, const PackInfo* info, int bytes, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    
    if (info->axis % 4 == 0 && info->axisStride == 1 && \
        bytes == 4 && info->insideStride == info->axis) {
        
        int axis_pack = UP_DIV(info->axis, PACK_NUMBER) * PACK_NUMBER / 4;
        DivModFast is(info->inside);
        DivModFast cs(axis_pack);

        PACKCOMMON_4<<<cores, threadNumbers>>>((const int4*)input, (int4*)output,
                    info->inside, info->axis / 4, info->outside, 
                    info->insideStride / 4, info->axisStride, is, cs);
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
        int axis_pack = UP_DIV(info->axis, PACK_NUMBER) * PACK_NUMBER / 4;
        DivModFast is(info->inside);
        DivModFast cs(axis_pack);
        const int maxCount = info->inside * axis_pack * info->outside;
        int block_num = runtime->blocks_num(maxCount);
        int block_size = runtime->threads_num();
        UNPACKCOMMON_4<<<block_num, block_size>>>((const int4*)input, (int4*)output, 
                        maxCount, info->inside, info->axis / 4, info->outside,
                        info->insideStride / 4, info->axisStride, is, cs);
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

template<typename T0, typename T1>
__global__ void NCHW_2_NHWC8(const T0* input,
    T1* output,
    const int maxCount,
    const int channel,
    const int area,
    const int channel_pack,
    DivModFast d_ocp,
    DivModFast d_area
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnlp_idx, batch_idx;
        d_ocp.divmod(index, temp, chnlp_idx);
        d_area.divmod(temp, batch_idx, area_idx);

        if(chnlp_idx >= channel) {
            output[index] = (T1)0.0f;
            continue;
        }
        int src_offset = (batch_idx * channel + chnlp_idx) * area + area_idx;
        output[index] = (T1)input[src_offset];
    }
}

template<typename T0, typename T1>
__global__ void NCHW_2_NHWC(const T0* input,
    T1* output,
    const int maxCount,
    const int channel,
    const int area,
    const int channel_pack,
    DivModFast d_oc,
    DivModFast d_area
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        d_oc.divmod(index, temp, chnl_idx);
        d_area.divmod(temp, batch_idx, area_idx);
        
        int src_offset = (batch_idx * channel + chnl_idx) * area + area_idx;
        output[index] = (T1)input[src_offset];
    }
}

template<typename T0, typename T1>
__global__ void NHWC_2_NHWC8(const T0* input,
    T1* output,
    const int maxCount,
    const int channel,
    const int area,
    const int channel_pack,
    DivModFast d_ocp,
    DivModFast d_area
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnlp_idx, batch_idx;
        d_ocp.divmod(index, temp, chnlp_idx);
        d_area.divmod(temp, batch_idx, area_idx);

        if(chnlp_idx >= channel) {
            output[index] = (T1)0.0f;
            continue;
        }
        int src_offset = (batch_idx * area + area_idx) * channel + chnlp_idx;
        output[index] = (T1)input[src_offset];
    }
}

template<typename T0, typename T1>
__global__ void NHWC8_2_NCHW(const T0* input,
    T1* output,
    const int maxCount,
    const int channel,
    const int area,
    const int channel_pack,
    DivModFast d_oc,
    DivModFast d_area
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {

        int area_idx, temp, channel_idx, batch_idx;
        d_area.divmod(index, temp, area_idx);
        d_oc.divmod(temp, batch_idx, channel_idx);

        int src_offset = (batch_idx * area + area_idx) * channel_pack + channel_idx;
        output[index] = (T1)input[src_offset];
    }
}

template<typename T0, typename T1>
__global__ void NHWC8_2_NHWC(const T0* input,
    T1* output,
    const int maxCount,
    const int channel,
    const int area,
    const int channel_pack,
    DivModFast d_oc,
    DivModFast d_area
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {

        int area_idx, temp, channel_idx, batch_idx;
        d_oc.divmod(index, temp, channel_idx);
        d_area.divmod(temp, batch_idx, area_idx);

        int src_offset = (batch_idx * area + area_idx) * channel_pack + channel_idx;
        output[index] = (T1)input[src_offset];
    }
}

template<typename T0, typename T1>
__global__ void NCHW_2_NCHW(const T0* input,
    T1* output,
    const int maxCount
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        output[index] = (T1)input[index];
    }
}

template<typename T0, typename T1>
__global__ void C4NHW4_2_NHWC8(const T0* input,
    T1* output,
    const int maxCount,
    const int batch,
    const int area,
    const int channel_pack
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int c_idx = index % channel_pack;
        int temp = index / channel_pack;
        int hw_idx = temp % area;
        int batch_idx = temp / area;

        int c4_idx = c_idx >> 2;
        int cL_idx = c_idx & 3;
        output[index] = (T1)input[((c4_idx * batch + batch_idx) * area + hw_idx) * 4 + cL_idx];
    }
}

template<typename T0, typename T1>
__global__ void NHWC8_2_C4NHW4(const T0* input,
    T1* output,
    const int maxCount,
    const int batch,
    const int channel,
    const int area,
    const int channel_pack
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int c_idx = index % channel_pack;
        int temp = index / channel_pack;
        int hw_idx = temp % area;
        int batch_idx = temp / area;

        int channel_8 = ((channel + 7) / 8) * 8;
        int c4_idx = c_idx >> 2;
        int cL_idx = c_idx & 3;
        output[((c4_idx * batch + batch_idx) * area + hw_idx) * 4 + cL_idx] = 
            (T1)input[(batch_idx * area + hw_idx) * channel_8 + c_idx];
    }
}

template<class T0, class T1>
static void insideFormatConvert(T0* input, T1* output, MNN_DATA_FORMAT srcDataFormat, MNN_DATA_FORMAT dstDataFormat, CUDARuntime* runtime, \
    const int area, const int batch, const int channel) {
    DivModFast d_oc(channel);
    DivModFast d_ocp(UP_DIV(channel, 8) * 8);
    DivModFast d_area(area);

    if(srcDataFormat == MNN_DATA_FORMAT_NCHW && dstDataFormat == MNN_DATA_FORMAT_NC4HW4) {
        const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NCHW_2_NHWC8<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 8) * 8,
                            d_ocp, d_area);
        checkKernelErrors;
        return;
    }
    if(srcDataFormat == MNN_DATA_FORMAT_NHWC && dstDataFormat == MNN_DATA_FORMAT_NC4HW4) {
        const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NHWC_2_NHWC8<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 8) * 8,
                            d_ocp, d_area);
        checkKernelErrors;
        return;
    }
    if((srcDataFormat == MNN_DATA_FORMAT_NCHW && dstDataFormat == MNN_DATA_FORMAT_NCHW) || \
        (srcDataFormat == MNN_DATA_FORMAT_NHWC && dstDataFormat == MNN_DATA_FORMAT_NHWC)) {
        const int maxCount = batch * area * channel;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NCHW_2_NCHW<T0, T1><<<block_num, block_size>>>(input, output, maxCount);
        checkKernelErrors;
        return;
    }
    if(srcDataFormat == MNN_DATA_FORMAT_NC4HW4 && dstDataFormat == MNN_DATA_FORMAT_NCHW) {
        const int maxCount = batch * area * channel;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NHWC8_2_NCHW<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 8) * 8,
                            d_oc, d_area);
        checkKernelErrors;
        return;
    }
    if(srcDataFormat == MNN_DATA_FORMAT_NC4HW4 && dstDataFormat == MNN_DATA_FORMAT_NHWC) {
        const int maxCount = batch * area * channel;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NHWC8_2_NHWC<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 8) * 8,
                            d_oc, d_area);
        checkKernelErrors;
        return;
    }
    if(srcDataFormat == MNN_DATA_FORMAT_NCHW && dstDataFormat == MNN_DATA_FORMAT_NHWC) {
        const int maxCount = batch * area * channel;
        const int block_num = runtime->blocks_num(maxCount);
        const int block_size = runtime->threads_num();
        NCHW_2_NHWC<T0, T1><<<block_num, block_size>>>(input, output, maxCount, channel, area, UP_DIV(channel, 8) * 8,
                            d_oc, d_area);
        checkKernelErrors;
        return;
    }
    MNN_PRINT("insideFormatConvert form %d to %d, not support\n", (int)srcDataFormat, (int)dstDataFormat);
    
}

void FormatConvert(void* output, void* input, MNN_DATA_FORMAT srcDataFormat, MNN_DATA_FORMAT dstDataFormat, CUDARuntime* runtime, \
    const int area, const int batch, const int channel, const Tensor* srcTensor, bool isFp16, bool srcDevice, bool dstDevice) {

    //MNN_PRINT("FormatConvert size batch:%d - plane:%d - channel:%d, %d-%d, %d-%d\n", batch, area, channel, srcDataFormat, dstDataFormat, srcDevice, dstDevice);
    if(batch == 0 || area == 0 || channel == 0) {
        return;
    }

    if(srcTensor->getType().bits == 8) {
        if(srcDataFormat == MNN_DATA_FORMAT_NC4HW4 && dstDataFormat == MNN_DATA_FORMAT_NC4HW4) {
            if(!srcDevice && dstDevice) {
                const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
                const int block_num = runtime->blocks_num(maxCount);
                const int block_size = runtime->threads_num();
                C4NHW4_2_NHWC8<<<block_num, block_size>>>((int8_t *)input, (int8_t *)output, 
                    maxCount, batch, area, UP_DIV(channel, 8) * 8);
                checkKernelErrors;
                return;
            }
    
            if(srcDevice && !dstDevice) {
                const int maxCount = batch * area * UP_DIV(channel, 4) * 4;
                const int block_num = runtime->blocks_num(maxCount);
                const int block_size = runtime->threads_num();
                NHWC8_2_C4NHW4<<<block_num, block_size>>>((int8_t *)input, (int8_t *)output, 
                    maxCount, batch, channel, area, UP_DIV(channel, 4) * 4);
                checkKernelErrors;
                return;
            }
        }
    
        insideFormatConvert<int8_t, int8_t>((int8_t *)input, (int8_t *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel);
        return;
    }

    isFp16 = isFp16 & (halide_type_float == srcTensor->getType().code);
    if(srcDataFormat == MNN_DATA_FORMAT_NC4HW4 && dstDataFormat == MNN_DATA_FORMAT_NC4HW4) {
        if(!srcDevice && dstDevice) {
            const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            if(isFp16) {
                C4NHW4_2_NHWC8<<<block_num, block_size>>>((float *)input, (half *)output, 
                    maxCount, batch, area, UP_DIV(channel, 8) * 8);
                checkKernelErrors;
            } else {
                C4NHW4_2_NHWC8<<<block_num, block_size>>>((float *)input, (float *)output, 
                    maxCount, batch, area, UP_DIV(channel, 8) * 8);
                checkKernelErrors;
            }
            return;
        }

        if(srcDevice && !dstDevice) {
            const int maxCount = batch * area * UP_DIV(channel, 4) * 4;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            if(isFp16) {
                NHWC8_2_C4NHW4<<<block_num, block_size>>>((half *)input, (float *)output, 
                    maxCount, batch, channel, area, UP_DIV(channel, 4) * 4);
                checkKernelErrors;
            } else {
                NHWC8_2_C4NHW4<<<block_num, block_size>>>((float *)input, (float *)output, 
                    maxCount, batch, channel, area, UP_DIV(channel, 4) * 4);
                checkKernelErrors;
            }
            return;
        }

        if(srcDevice && dstDevice) {
            const int maxCount = batch * area * UP_DIV(channel, 8) * 8;
            const int block_num = runtime->blocks_num(maxCount);
            const int block_size = runtime->threads_num();
            if(isFp16) {
                NCHW_2_NCHW<half, half><<<block_num, block_size>>>((half *)input, (half *)output, maxCount);
                checkKernelErrors;
            } else {
                NCHW_2_NCHW<float, float><<<block_num, block_size>>>((float *)input, (float *)output, maxCount);
                checkKernelErrors; 
            }
            return;
        }
    }

    if(!srcDevice) {
        if(isFp16) {
            insideFormatConvert<float, half>((float *)input, (half *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel);
        } else {
            insideFormatConvert<float, float>((float *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel);
        }
    } else if(!dstDevice) {
        if(isFp16) {
            insideFormatConvert<half, float>((half *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel);
        } else {
            insideFormatConvert<float, float>((float *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel);
        }
    } else {
        if(isFp16) {
            insideFormatConvert<half, half>((half *)input, (half *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel);
        } else {
            insideFormatConvert<float, float>((float *)input, (float *)output, srcDataFormat, dstDataFormat, runtime, area, batch, channel);
        }
    }
}


};
};