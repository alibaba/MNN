#include "Raster.cuh"
#include "TensorflowOp_generated.h"
#include <cuda_fp16.h>
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

// Blit don't care offset
template <typename T>
__global__ void blitRegion(const T *inputO, T *outputO,
    int count,
    int loopCount,
    const int32_t* dstIndice, const int32_t* srcIndice,
    int dstUseIndice, int srcUseIndice,
    int dstStep, int srcStep,int srcLimit,
    int sizeZ, int sizeY, int sizeX,
    int strideZ, int strideY, int strideX,
    int dstStrideZ, int dstStrideY, int dstStrideX
    ) {
    int total = count;
    for (size_t fuseIndex = blockIdx.x * blockDim.x + threadIdx.x; fuseIndex < total; fuseIndex += blockDim.x * gridDim.x) {
        int x = fuseIndex % sizeX;
        int temp = fuseIndex / sizeX;
        int y = temp % sizeY;
        temp = temp / sizeY;
        int z = temp % sizeZ;
        int i = temp / sizeZ;
        int srcOffsetO = i * srcStep;
        if (srcUseIndice >= 0) {
            srcOffsetO = srcIndice[i] * srcStep;
        }
        int dstOffsetO = i * dstStep;
        if (dstUseIndice >= 0) {
            dstOffsetO = dstIndice[i] * dstStep;
        }
        if (srcOffsetO >= 0 && srcOffsetO < srcLimit) {
            const T* input = inputO + srcOffsetO;
            T* output = outputO + dstOffsetO;
            int srcOffset = z * strideZ + y * strideY + x * strideX;
            int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
            output[dstOffset] = input[srcOffset];
        } else {
            T* output = outputO + dstOffsetO;
            int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
            output[dstOffset] = (T)0;
        }
    }
}
void BlitWithIndice(uint8_t* output, const uint8_t* input, const int32_t* dstIndices, const int32_t* srcIndices, int dstUseIndice, int srcUseIndice, int loopCount, int dstStep, int srcStep, int srcLimit, const Tensor::InsideDescribe::Region& reg, int bytes, CUDARuntime* runtime) {
    int count = loopCount * reg.size[0]*reg.size[1]*reg.size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = ALIMIN(runtime->threads_num(), count);
    switch (bytes) {
        case 4:
            blitRegion<<<block_num, threads_num>>>((const float*)input, (float*)output, 
                count,
                loopCount,
                dstIndices, srcIndices,
                dstUseIndice, srcUseIndice,
                dstStep, srcStep, srcLimit,
                reg.size[0], reg.size[1], reg.size[2],
                reg.src.stride[0], reg.src.stride[1], reg.src.stride[2],
                reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            break;
        case 2:
            blitRegion<<<block_num, threads_num>>>((const int16_t*)input, (int16_t*)output,
                count,
                loopCount,
                dstIndices, srcIndices,
                dstUseIndice, srcUseIndice,
                dstStep, srcStep, srcLimit,
                reg.size[0], reg.size[1], reg.size[2],
                reg.src.stride[0], reg.src.stride[1], reg.src.stride[2],
                reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            break;
        case 1:
            blitRegion<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output,
                count,
                loopCount,
                dstIndices, srcIndices,
                dstUseIndice, srcUseIndice,
                dstStep, srcStep, srcLimit,
                reg.size[0], reg.size[1], reg.size[2],
                reg.src.stride[0], reg.src.stride[1], reg.src.stride[2],
                reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            break;
        default:
            break;
    }
}

#define UNARY_FUNC(Name, Func)\
template<typename T>\
__global__ void Name(const T *input, T *output,\
        int count,\
        DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,\
        int strideZ, int strideY, int strideX,\
        int dstStrideZ, int dstStrideY, int dstStrideX\
        ) { \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {\
    int ix, tmp, iy, iz;\
    sizeX.divmod(i, tmp, ix);\
    sizeY.divmod(tmp, iz, iy);\
    int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
    int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
    T x = input[srcOffset];\
    output[dstOffset] = Func;\
  }\
}\
template<typename T>\
__global__ void FLOAT##Name(const T *input, T *output,\
        int count,\
        DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,\
        int strideZ, int strideY, int strideX,\
        int dstStrideZ, int dstStrideY, int dstStrideX\
        ) { \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {\
    int ix, tmp, iy, iz;\
    sizeX.divmod(i, tmp, ix);\
    sizeY.divmod(tmp, iz, iy);\
    int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
    int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
    float x = (float)input[srcOffset];\
    output[dstOffset] = (float)(Func);\
  }\
}\
template<typename T>\
__global__ void UNARY_SINGLE##Name(const T *input, T *output,\
        int count\
        ) { \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {\
    float x = (float)input[i];\
    output[i] = (T)(Func);\
  }\
}\

template<typename T>
__global__ void UNARY_HALF2_SIGMOID(const T *input, T *output,
        int count
        ) { 
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    half2 x = input[i];
    half2 one;
    one.x = 1.0;
    one.y = 1.0f;
    output[i] = __h2div(one, __hadd2(one, h2exp(__hneg2(x))));
  }
}

template<typename T>
__global__ void blit_2_float(const T *input, T *output,
    int count,
    DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
    int strideZ, int strideY,
    int dstStrideZ, int dstStrideY
    ) { 
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int ix, tmp, iy, iz;
        sizeX.divmod(i, tmp, ix);
        sizeY.divmod(tmp, iz, iy);
        int srcOffset = iz * strideZ + iy * strideY + (ix << 1);
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + (ix << 1);
        int2 * dstF = (int2 *)(output+dstOffset);
        dstF[0] = ((int2 *)(input+srcOffset))[0];
    }
}
template<typename T>
__global__ void blit_2_half(const T *input, T *output,
    int count,
    DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
    int strideZ, int strideY,
    int dstStrideZ, int dstStrideY
    ) { 
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int ix, tmp, iy, iz;
        sizeX.divmod(i, tmp, ix);
        sizeY.divmod(tmp, iz, iy);
        int srcOffset = iz * strideZ + iy * strideY + (ix << 1);
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + (ix << 1);
        int* dstF = (int *)(output+dstOffset);
        dstF[0] = ((int *)(input+srcOffset))[0];
    }
}

struct Bytes512 {
    int4 x[4];
};

UNARY_FUNC(blit, x);
UNARY_FUNC(ABS, abs(x));
UNARY_FUNC(EXP, exp(x));
UNARY_FUNC(NEG, -x);
UNARY_FUNC(RECIPROCAL, (1.0)/x);
UNARY_FUNC(FLOOR, floor(x));
UNARY_FUNC(CEIL, ceil(x));
UNARY_FUNC(SQUARE, x*x);
UNARY_FUNC(SQRT, (T)(sqrt((float)x)));
UNARY_FUNC(RSQRT, (T)(rsqrt((float)x)));
UNARY_FUNC(LOG, (T)(log((float)x)));
UNARY_FUNC(SIN, (T)(sin((float)x)));
UNARY_FUNC(COS, (T)(cos((float)x)));
UNARY_FUNC(TAN, (T)(tan((float)x)));
UNARY_FUNC(ASIN, (T)(asin((float)x)));
UNARY_FUNC(ACOS, (T)(acos((float)x)));
UNARY_FUNC(ATAN, (T)(atan((float)x)));
UNARY_FUNC(LOG1P, log(1+x));
UNARY_FUNC(TANH, tanh(x));
UNARY_FUNC(SIGMOID, (x>87.?1.0f:(x<-87.?0.0f: 1./(1.+exp(-x)))));
UNARY_FUNC(EXPM1, exp(x)-1);
UNARY_FUNC(ATANH, atanh(x));
UNARY_FUNC(ACOSH, acosh(x));
UNARY_FUNC(COSH, cosh(x));
UNARY_FUNC(SIGN, x > 0 ? 1 : (x<0 ? -1 : 0));
UNARY_FUNC(ROUND, round(x));
UNARY_FUNC(SINH, sinh(x));
UNARY_FUNC(ASINH, asinh(x));
UNARY_FUNC(HARDSWISH, 1.0/6.0 * x * min(max(x+3.0, 0.0), 6.0));
UNARY_FUNC(ERF, erf(x));
UNARY_FUNC(ERFC, erfc(x));
UNARY_FUNC(ERFINV, erfinv(x));
UNARY_FUNC(GELU, (1.0f + tanh(0.79788458f * (0.044715f * x * x * x + x))) * x * 0.5f);
UNARY_FUNC(GELU_STANDARD, (erf(x*0.7071067932881648f)+1.f)*x*0.5);

void RasterBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int bytes, CUDARuntime* runtime) {
    int count = size[0] * size[1] * size[2];

    // MNN_PRINT("blit info size:%d-%d-%d, srcStride:%d-%d-%d, dstStride:%d-%d-%d, ptr:%p %p\n", size[0], size[1], size[2], srcStride[0], srcStride[1], srcStride[2], dstStride[0], dstStride[1], dstStride[2], input, output);
    bool isThirdSizeVector  = (size[2] % 2 == 0 && srcStride[2] == 1 && dstStride[2] == 1);
    bool isSecondSizeVector = (size[1] % 2 == 0 && srcStride[1] == 1 && dstStride[1] == 1) && (size[2] == 1 && srcStride[2] == 1 && dstStride[2] == 1);
    bool isFirstSizeVector  = (size[0] % 2 == 0 && srcStride[0] == 1 && dstStride[0] == 1) && (size[1] == 1 && srcStride[1] == 1 && dstStride[1] == 1) && (size[2] == 1 && srcStride[2] == 1 && dstStride[2] == 1);
    bool isStrideVector     = (srcStride[0] % 2 == 0 || srcStride[0] == 1) && (srcStride[1] % 2 == 0 || srcStride[1] == 1) && (srcStride[2] % 2 == 0 || srcStride[2] == 1) && \
                            (dstStride[0] % 2 == 0 || dstStride[0] == 1) && (dstStride[1] % 2 == 0 || dstStride[1] == 1) && (dstStride[2] % 2 == 0 || dstStride[2] == 1);
    bool isSizeVector = isThirdSizeVector || isSecondSizeVector || isFirstSizeVector;
    if(count > 16384 && isSizeVector && isStrideVector) {
        int32_t newSize[3], newSrcStride[3], newDstStride[3];
        newSize[0] = size[0]; 
        newSize[1] = size[1]; 
        newSize[2] = size[2]; 
        newSrcStride[0] = srcStride[0]; 
        newSrcStride[1] = srcStride[1]; 
        newSrcStride[2] = srcStride[2]; 
        newDstStride[0] = dstStride[0]; 
        newDstStride[1] = dstStride[1]; 
        newDstStride[2] = dstStride[2]; 
        if(isSecondSizeVector) {
            /*  size   : [size_0, size_1, 1]  srcStride   : [ss_0, 1, 1] dstStride   : [ds_0, 1, 1]
            --> newSize: [1, size_0, size_1]  newSrcStride: [1, ss_0, 1] newDstStride: [1, ds_0, 1]
            */
            newSize[2] = size[1];
            newSize[1] = size[0];
            newSize[0] = 1;
            newSrcStride[1] = srcStride[0];
            newSrcStride[0] = 1;
            newDstStride[1] = dstStride[0];
            newDstStride[0] = 1;
        }
        if(isFirstSizeVector) {
            /*  size   : [size_0, 1, 1]  srcStride   : [1, 1, 1] dstStride   : [1, 1, 1]
            --> newSize: [1, 1, size_0]  newSrcStride: [1, 1, 1] newDstStride: [1, 1, 1]
            */
            newSize[2] = size[0];
            newSize[0] = 1;
        }

        DivModFast new_sz(newSize[0]);
        DivModFast new_sy(newSize[1]);
        DivModFast new_sx(newSize[2]/2);

        int newCount = count / 2;
        int block_num = runtime->blocks_num(newCount);
        int threads_num = runtime->threads_num();

        // Forbid addresss misalign
        if(bytes == 4 && reinterpret_cast<std::uintptr_t>(input) % 8 == 0 && reinterpret_cast<std::uintptr_t>(output) % 8 == 0) {
            blit_2_float<<<block_num, threads_num>>>((const float*)input, (float*)output, 
                newCount,
                new_sz, new_sy, new_sx,
                newSrcStride[0], newSrcStride[1],
                newDstStride[0], newDstStride[1]);
            checkKernelErrors;
            return;
        } else if(bytes == 2 && reinterpret_cast<std::uintptr_t>(input) % 4 == 0 && reinterpret_cast<std::uintptr_t>(output) % 4 == 0) {
            blit_2_half<<<block_num, threads_num>>>((const half*)input, (half*)output, 
                newCount,
                new_sz, new_sy, new_sx,
                newSrcStride[0], newSrcStride[1],
                newDstStride[0], newDstStride[1]);
            checkKernelErrors;
            return;
        }
    }
    
    DivModFast sz(size[0]);
    DivModFast sy(size[1]);
    DivModFast sx(size[2]);
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();

    switch (bytes) {
        case 64:
            blit<<<block_num, threads_num>>>((const Bytes512*)input, (Bytes512*)output,
                count,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 32:
            blit<<<block_num, threads_num>>>((const double4*)input, (double4*)output,
                count,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 4:
    	    blit<<<block_num, threads_num>>>((const float*)input, (float*)output,
                count,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 2:
            blit<<<block_num, threads_num>>>((const int16_t*)input, (int16_t*)output,
                count,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 1:
            blit<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output,
                count,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        default:
            break;
    }
    checkKernelErrors;
}

template<typename T0, typename T1>
__global__ void fuseblit(const T0 *input, T1 *output,
    int fuseNum, int count, const int32_t* sliceOffset,
    DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
    int strideZ, int strideY, int strideX,
    int dstStrideZ, int dstStrideY, int dstStrideX
    ) {
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < count; c += blockDim.x * gridDim.x) {
        int ix, tmp, iy, tmp2, iz, j;
        sizeX.divmod(c, tmp, ix);
        sizeY.divmod(tmp, tmp2, iy);
        sizeZ.divmod(tmp2, j, iz);
        int src_offset = sliceOffset[j] + iz * strideZ + iy * strideY + ix * strideX;
        int dst_offset = sliceOffset[fuseNum+j] + iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;
        output[dst_offset] = input[src_offset];
    }
}

__global__ void fuseblit_4(const int32_t *input, int32_t *output,
    int fuseNum, int count, const int32_t* sliceOffset,
    DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
    int strideZ, int strideY,
    int dstStrideZ, int dstStrideY
    ) {
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < count; c += blockDim.x * gridDim.x) {
        int ix, tmp, iy, tmp2, iz, j;
        sizeX.divmod(c, tmp, ix);
        sizeY.divmod(tmp, tmp2, iy);
        sizeZ.divmod(tmp2, j, iz);
        int src_offset = sliceOffset[j] + iz * strideZ + iy * strideY + (ix << 2);
        int dst_offset = sliceOffset[fuseNum+j] + iz * dstStrideZ + iy * dstStrideY + (ix << 2);
        int4* srcF = (int4 *)(input + src_offset);
        int4* dstF = (int4 *)(output + dst_offset);
        dstF[0] = srcF[0];
    }
}

__global__ void fuseblit_half_4(const int16_t *input, int16_t *output,
    int fuseNum, int count, const int32_t* sliceOffset,
    DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
    int strideZ, int strideY,
    int dstStrideZ, int dstStrideY
    ) {
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < count; c += blockDim.x * gridDim.x) {
        int ix, tmp, iy, tmp2, iz, j;
        sizeX.divmod(c, tmp, ix);
        sizeY.divmod(tmp, tmp2, iy);
        sizeZ.divmod(tmp2, j, iz);
        int src_offset = sliceOffset[j] + iz * strideZ + iy * strideY + (ix << 2);
        int dst_offset = sliceOffset[fuseNum+j] + iz * dstStrideZ + iy * dstStrideY + (ix << 2);
        int2* srcF = (int2 *)(input + src_offset);
        int2* dstF = (int2 *)(output + dst_offset);
        dstF[0] = srcF[0];
    }
}

void FuseRasterBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int fuseNum, void* sliceOffset, int bytes, CUDARuntime* runtime, int unit) {
    DivModFast sz(size[0]);
    DivModFast sy(size[1]);
    int count = fuseNum * size[0] * size[1] * size[2];
    bool strideC4Support = srcStride[0] % 4 == 0 && srcStride[1] % 4 == 0 && dstStride[0] % 4 == 0 && dstStride[1] % 4 == 0;
    if(size[2] % 4 == 0 && count > 16384 && srcStride[2] == 1 && dstStride[2] == 1 && unit == 4 && strideC4Support) {
        int xL4 = size[2] / 4;
        int countC4 = fuseNum * size[0] * size[1] * xL4;
        int numBlocks = runtime->blocks_num(countC4);
        int threadsPerBlock = runtime->threads_num();
        DivModFast sx_4(xL4);

        if(bytes == 4) {
            fuseblit_4<<<numBlocks, threadsPerBlock>>>((const int32_t*)input, (int32_t*)output, 
                fuseNum, countC4, (const int32_t*)sliceOffset,
                sz, sy, sx_4,
                srcStride[0], srcStride[1],
                dstStride[0], dstStride[1]);
            return;
        } else if(bytes == 2){
            fuseblit_half_4<<<numBlocks, threadsPerBlock>>>((const int16_t*)input, (int16_t*)output, 
                fuseNum, countC4, (const int32_t*)sliceOffset,
                sz, sy, sx_4,
                srcStride[0], srcStride[1],
                dstStride[0], dstStride[1]);
            return;
        }
    }
    DivModFast sx(size[2]);
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();

    switch (bytes) {
        case 64:
            fuseblit<<<block_num, threads_num>>>((const Bytes512*)input, (Bytes512*)output, 
                fuseNum, count, (const int32_t*)sliceOffset,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 16:
            fuseblit<<<block_num, threads_num>>>((const int4*)input, (int4*)output, 
                fuseNum, count, (const int32_t*)sliceOffset,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 4:
            fuseblit<<<block_num, threads_num>>>((const float*)input, (float*)output, 
                fuseNum, count, (const int32_t*)sliceOffset,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 2:
            fuseblit<<<block_num, threads_num>>>((const int16_t*)input, (int16_t*)output,
                fuseNum, count, (const int32_t*)sliceOffset,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 1:
            fuseblit<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output,
                fuseNum, count, (const int32_t*)sliceOffset,
                sz, sy, sx,
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        default:
            break;
    }
    //printf("%s, %d-%d-%d-%d\n", cudaGetErrorString(cudaGetLastError()), numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
}

template<typename T0, typename T1>
__global__ void fuseblitLimit(const T0 *input, T1 *output,
    const FuseRegion* info, const int32_t* sliceOffset
    ) {
    int sizeZ = info->size[0];
    int sizeY = info->size[1];
    int sizeX = info->size[2];
    int strideZ = info->srcStride[0];
    int strideY = info->srcStride[1];
    int strideX = info->srcStride[2];
    int dstStrideZ = info->dstStride[0];
    int dstStrideY = info->dstStride[1];
    int dstStrideX = info->dstStride[2];
    int fuseNum = info->fuseNumber;

    int count = fuseNum*sizeZ * sizeY * sizeX;

    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < (count); c += blockDim.x * gridDim.x) {
        int j = c / (sizeZ * sizeY * sizeX);
        int i = c % (sizeZ * sizeY * sizeX);
        int ix = i % sizeX;
        int tmp = i / sizeX;
        int iy = tmp % sizeY;
        int iz = tmp / sizeY;
        const int* srcOffsetPtr = sliceOffset + 8 * j;
        const int* dstOffsetPtr = sliceOffset + 8 * j + 4;
        T0 srcValue = (T0)0;
        int src_offset = srcOffsetPtr[3] + iz * strideZ + iy * strideY + ix * strideX;
        if (srcOffsetPtr[0] > iz && srcOffsetPtr[1] > iy && srcOffsetPtr[2] > ix) {
            srcValue = input[src_offset];
        }
        int dst_offset = dstOffsetPtr[3] + iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;
        //printf("%d -> %d - %f\n", src_offset, dst_offset, srcValue);
        if (dstOffsetPtr[0] > iz && dstOffsetPtr[1] > iy && dstOffsetPtr[2] > ix) {
            output[dst_offset] = srcValue;
        }
    }
}
void FuseRasterBlitFloatToHalf(uint8_t* output, const uint8_t* input, const FuseRegion* info, void* sliceOffset, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int block_num = prop.multiProcessorCount;
    fuseblitLimit<<<block_num, threads_num>>>((const float*)input, (half*)output, 
        info, (const int32_t*)sliceOffset);
}
void FuseRasterBlitHalfToFloat(uint8_t* output, const uint8_t* input, const FuseRegion* info, void* sliceOffset, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int block_num = prop.multiProcessorCount;
    fuseblitLimit<<<block_num, threads_num>>>((const half*)input, (float*)output, 
        info, (const int32_t*)sliceOffset);
}
void FuseRasterBlitFloatToFloat(uint8_t* output, const uint8_t* input, const FuseRegion* info, void* sliceOffset, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int block_num = prop.multiProcessorCount;
    fuseblitLimit<<<block_num, threads_num>>>((const float*)input, (float*)output, 
        info, (const int32_t*)sliceOffset);
}

void FuseRasterBlitCommon(uint8_t* output, const uint8_t* input, const FuseRegion* info, void* sliceOffset, CUDARuntime* runtime, int bytes) {
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int block_num = prop.multiProcessorCount;
    switch (bytes) {
        case 4:
            fuseblitLimit<<<block_num, threads_num>>>((const float*)input, (float*)output, 
                info, (const int32_t*)sliceOffset);
            break;
        case 2:
            fuseblitLimit<<<block_num, threads_num>>>((const half*)input, (half*)output, 
                info, (const int32_t*)sliceOffset);
            break;
        case 1:
            fuseblitLimit<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output, 
                info, (const int32_t*)sliceOffset);
            break;
        default:
            break;
    }
}


void UnaryBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int bytes, CUDARuntime* runtime, int opType) {
    int count = size[0] * size[1] * size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();

    DivModFast sz(size[0]);
    DivModFast sy(size[1]);
    DivModFast sx(size[2]);
    // TODO: Support FP16
    #define COMPUTE(TYPE)\
    if (opType == MNN::UnaryOpOperation_##TYPE ) {\
        if(size[0] == 1 && size[1] == 1 && srcStride[2] == 1 && dstStride[2] == 1 && opType == MNN::UnaryOpOperation_SIGMOID && bytes==2 && count % 2 == 0) {\
             block_num = runtime->blocks_num(count/2);\
             threads_num = runtime->threads_num();\
             UNARY_HALF2_SIGMOID<<<block_num, threads_num>>>((const half2*)input, (half2*)output, count/2);\
        } else if(size[0] == 1 && size[1] == 1 && srcStride[2] == 1 && dstStride[2] == 1) {\
            if(bytes==2) {\
                UNARY_SINGLE##TYPE<<<block_num, threads_num>>>((const half*)input, (half*)output, count);\
            } else {\
                UNARY_SINGLE##TYPE<<<block_num, threads_num>>>((const float*)input, (float*)output, count);\
            }\
        } else {\
            if(bytes==2) {\
                FLOAT##TYPE<<<block_num, threads_num>>>((const half*)input, (half*)output,\
                    count, \
                    sz, sy, sx,\
                    srcStride[0], srcStride[1], srcStride[2],\
                    dstStride[0], dstStride[1], dstStride[2]);\
            } else {\
                TYPE<<<block_num, threads_num>>>((const float*)input, (float*)output,\
                    count, \
                    sz, sy, sx,\
                    srcStride[0], srcStride[1], srcStride[2],\
                    dstStride[0], dstStride[1], dstStride[2]);\
            }\
        }\
        return;\
    }\

    COMPUTE(ABS);
    COMPUTE(NEG);
    COMPUTE(FLOOR);
    COMPUTE(CEIL);
    COMPUTE(SQUARE);
    COMPUTE(SQRT);
    COMPUTE(RSQRT);
    COMPUTE(EXP);
    COMPUTE(LOG);
    COMPUTE(SIN);
    COMPUTE(COS);
    COMPUTE(TAN);
    COMPUTE(GELU);
    COMPUTE(GELU_STANDARD);
    COMPUTE(ASIN);
    COMPUTE(ACOS);
    COMPUTE(ATAN);
    COMPUTE(RECIPROCAL);
    COMPUTE(LOG1P);
    COMPUTE(TANH);
    COMPUTE(SIGMOID);
    COMPUTE(EXPM1);
    COMPUTE(ACOSH);
    COMPUTE(ATANH);
    COMPUTE(SIGN);
    COMPUTE(COSH);
    COMPUTE(ROUND);
    COMPUTE(SINH);
    COMPUTE(ASINH);
    COMPUTE(HARDSWISH);
    COMPUTE(ERF);
    COMPUTE(ERFC);
    COMPUTE(ERFINV);

    #undef COMPUTE
}

#define BINARY_FUNC(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void Binary##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY, int strideX,\
    int strideZ1, int strideY1, int strideX1,\
    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix = i % sizeX;\
        int tmp = i / sizeX;\
        int iy = tmp % sizeY;\
        int iz = tmp / sizeY;\
        int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix * strideX1;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
        TIn x = input0[srcOffset];\
        TIn y = input1[srcOffset1];\
        TOut val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset] = val;\
    }\
}\


#define BINARY_FUSEADD_FUNC(Name, Func)\
__global__ void BinaryFuseAdd##Name(\
    const float *input0, const float* input1, float *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY, int strideX,\
    int strideZ1, int strideY1, int strideX1,\
    int dstStrideZ, int dstStrideY, int dstStrideX\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix = i % sizeX;\
        int tmp = i / sizeX;\
        int iy = tmp % sizeY;\
        int iz = tmp / sizeY;\
        int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix * strideX1;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
        float x = input0[srcOffset];\
        float y = input1[srcOffset1];\
        float val = (float)(Func);\
        atomicAdd(output + dstOffset, val);\
    }\
}\


#define BINARY_FUNC_FLOATMID(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMid##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY, int strideX,\
    int strideZ1, int strideY1, int strideX1,\
    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType,\
    DivModFast d_sizeY, DivModFast d_sizeX\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix, tmp, iy, iz;\
        d_sizeX.divmod(i, tmp, ix);\
        d_sizeY.divmod(tmp, iz, iy);\
        int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix * strideX1;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
        float x = input0[srcOffset];\
        float y = input1[srcOffset1];\
        float val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset] = val;\
    }\
}\
template<typename TIn, typename TOut>\
__global__ void BinaryMid4_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY,\
    int strideZ1, int strideY1,\
    int dstStrideZ, int dstStrideY, int activationType,\
    DivModFast d_sizeY, DivModFast d_sizeX,\
    bool inp0Broadcast, bool inp1Broadcast\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix, tmp, iy, iz;\
        d_sizeX.divmod(i, tmp, ix);\
        d_sizeY.divmod(tmp, iz, iy);\
        ix = ix << 2;\
        int srcOffset = iz * strideZ + iy * strideY + ix;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix;\
        float4 xx = inp0Broadcast ? make_float4(input0[srcOffset-ix],input0[srcOffset-ix], input0[srcOffset-ix], input0[srcOffset-ix]) : ((float4 *)(input0+srcOffset))[0];\
        float4 yy = inp1Broadcast ? make_float4(input1[srcOffset1-ix],input1[srcOffset1-ix], input1[srcOffset1-ix], input1[srcOffset1-ix]) :((float4 *)(input1+srcOffset1))[0];\
        float x = xx.x;\
        float y = yy.x;\
        float val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset] = val;\
        x = xx.y;\
        y = yy.y;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset+1] = val;\
        x = xx.z;\
        y = yy.z;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset+2] = val;\
        x = xx.w;\
        y = yy.w;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset+3] = val;\
    }\
}\
template<typename TIn, typename TOut>\
__global__ void BinaryMidHalf2_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY,\
    int strideZ1, int strideY1,\
    int dstStrideZ, int dstStrideY, int activationType,\
    DivModFast d_sizeY, DivModFast d_sizeX,\
    bool inp0Broadcast, bool inp1Broadcast\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix, tmp, iy, iz;\
        d_sizeX.divmod(i, tmp, ix);\
        d_sizeY.divmod(tmp, iz, iy);\
        ix = ix << 1;\
        int srcOffset = iz * strideZ + iy * strideY + ix;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix;\
        half2 xx = inp0Broadcast ? make_half2(input0[srcOffset-ix], input0[srcOffset-ix]) : ((half2 *)(input0+srcOffset))[0];\
        half2 yy = inp1Broadcast ? make_half2(input1[srcOffset1-ix], input1[srcOffset1-ix]) : ((half2 *)(input1+srcOffset1))[0];\
        float x = xx.x;\
        float y = yy.x;\
        float val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset] = val;\
        x = xx.y;\
        y = yy.y;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset+1] = val;\
    }\
}\
template<typename TIn, typename TOut>\
__global__ void BinaryMidLinear##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ,\
    int strideZ,\
    int strideZ1,\
    int dstStrideZ,\
    int activationType\
    ) { \
    int count = sizeZ;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int iz = i;\
        int srcOffset = iz * strideZ;\
        int srcOffset1 = iz * strideZ1;\
        int dstOffset = iz * dstStrideZ;\
        float x = input0[srcOffset];\
        float y = input1[srcOffset1];\
        float val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset] = (TOut)val;\
    }\
}\

#define BINARY_FUNC_FLOATMID4(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMidLinear4_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int count_4, int activationType,\
    bool inp0Broadcast, bool inp1Broadcast\
    ) { \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count_4); i += blockDim.x * gridDim.x) {\
        int iz = i;\
        int srcOffset = iz << 2;\
        int srcOffset1 = iz << 2;\
        int dstOffset = iz << 2;\
        float4 xx = inp0Broadcast ? make_float4(input0[0], input0[0], input0[0], input0[0]) : ((float4 *)(input0+srcOffset))[0];\
        float4 yy = inp1Broadcast ? make_float4(input1[0], input1[0], input1[0], input1[0]) : ((float4 *)(input1+srcOffset1))[0];\
        float x = xx.x;\
        float y = yy.x;\
        TOut val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset] = val;\
        x = xx.y;\
        y = yy.y;\
        val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset+1] = val;\
        x = xx.z;\
        y = yy.z;\
        val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset+2] = val;\
        x = xx.w;\
        y = yy.w;\
        val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset+3] = val;\
    }\
}\
template<typename TIn, typename TOut>\
__global__ void BinaryMidLinearHalf4_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int count_4, int activationType,\
    bool inp0Broadcast, bool inp1Broadcast\
    ) { \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count_4); i += blockDim.x * gridDim.x) {\
        int iz = i;\
        int srcOffset = iz << 2;\
        int srcOffset1 = iz << 2;\
        int dstOffset = iz << 2;\
        half2 xx = inp0Broadcast ? make_half2(input0[0], input0[0]) : ((half2 *)(input0+srcOffset))[0];\
        half2 yy = inp1Broadcast ? make_half2(input1[0], input1[0]) : ((half2 *)(input1+srcOffset1))[0];\
        float x = (float)xx.x;\
        float y = (float)yy.x;\
        float val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset] = (TOut)val;\
        x = (float)xx.y;\
        y = (float)yy.y;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset+1] = (TOut)val;\
        xx = inp0Broadcast ? make_half2(input0[0], input0[0]) : ((half2 *)(input0+srcOffset))[1];\
        yy = inp1Broadcast ? make_half2(input1[0], input1[0]) : ((half2 *)(input1+srcOffset1))[1];\
        x = (float)xx.x;\
        y = (float)yy.x;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val <  0.0f ? 0.0f  : val);\
        }\
        output[dstOffset+2] = (TOut)val;\
        x = (float)xx.y;\
        y = (float)yy.y;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset+3] = (TOut)val;\
    }\
}\

#define sign(y) ((y) > 0 ? 1 : ((y) < 0 ? -1 : 0))

BINARY_FUNC(ADD, x+y);
BINARY_FUNC(SUB, x-y);
BINARY_FUNC(MUL, x*y);
BINARY_FUNC(DIV, x/y);
BINARY_FUNC(REALDIV, (float)sign(y) * x / max(abs(y), 0.0000001));
BINARY_FUNC(MINIMUM, min(x, y));
BINARY_FUNC(MAXIMUM, max(x, y));
BINARY_FUNC(GREATER, x > y ? 1 : 0);
BINARY_FUNC(LESS, x < y ? 1 : 0);
BINARY_FUNC(LESS_EQUAL, x <= y ? 1 : 0);
BINARY_FUNC(GREATER_EQUAL, x >= y ? 1 : 0);
BINARY_FUNC(EQUAL, x == y ? 1 : 0);
BINARY_FUNC(NOTEQUAL, x != y ? 1 : 0);
BINARY_FUNC(FLOORDIV, floor(x / y));
BINARY_FUNC(FLOORMOD, x - floor(x / y) * y);
BINARY_FUNC(SquaredDifference, (x-y)*(x-y));
BINARY_FUNC(POW, pow(x, y));
BINARY_FUNC(ATAN2, atan2(x, y));
BINARY_FUNC(MOD, (x % y));
BINARY_FUNC(LOGICALOR, (x || y) ? 1 : 0);

BINARY_FUSEADD_FUNC(ADD, x+y);
BINARY_FUSEADD_FUNC(SUB, x-y);
BINARY_FUSEADD_FUNC(MUL, x*y);
BINARY_FUSEADD_FUNC(DIV, x/y);
BINARY_FUSEADD_FUNC(REALDIV, (float)sign(y) * x / max(abs(y), 0.0000001));
BINARY_FUSEADD_FUNC(MINIMUM, min(x, y));
BINARY_FUSEADD_FUNC(MAXIMUM, max(x, y));
BINARY_FUSEADD_FUNC(FLOORDIV, floor(x / y));
BINARY_FUSEADD_FUNC(FLOORMOD, x - floor(x / y) * y);
BINARY_FUSEADD_FUNC(SquaredDifference, (x-y)*(x-y));
BINARY_FUSEADD_FUNC(POW, pow(x, y));
BINARY_FUSEADD_FUNC(ATAN2, atan2(x, y));

BINARY_FUNC_FLOATMID(ADD, x+y);
BINARY_FUNC_FLOATMID(SUB, x-y);
BINARY_FUNC_FLOATMID(MUL, x*y);
BINARY_FUNC_FLOATMID(DIV, x/y);
BINARY_FUNC_FLOATMID(REALDIV, (float)sign(y) * x / max(abs(y), 0.0000001));
BINARY_FUNC_FLOATMID(MINIMUM, min(x, y));
BINARY_FUNC_FLOATMID(MAXIMUM, max(x, y));
BINARY_FUNC_FLOATMID(GREATER, x > y ? 1 : 0);
BINARY_FUNC_FLOATMID(LESS, x < y ? 1 : 0);
BINARY_FUNC_FLOATMID(LESS_EQUAL, x <= y ? 1 : 0);
BINARY_FUNC_FLOATMID(GREATER_EQUAL, x >= y ? 1 : 0);
BINARY_FUNC_FLOATMID(EQUAL, x == y ? 1 : 0);
BINARY_FUNC_FLOATMID(NOTEQUAL, x != y ? 1 : 0);
BINARY_FUNC_FLOATMID(FLOORDIV, floor(x / y));
BINARY_FUNC_FLOATMID(FLOORMOD, x - floor(x / y) * y);
BINARY_FUNC_FLOATMID(SquaredDifference, (x-y)*(x-y));
BINARY_FUNC_FLOATMID(POW, pow(x, y));
BINARY_FUNC_FLOATMID(ATAN2, atan2(x, y));
BINARY_FUNC_FLOATMID(MOD, fmod(x, y));
BINARY_FUNC_FLOATMID(LOGICALOR, (x || y) ? 1 : 0);

BINARY_FUNC_FLOATMID4(ADD, x+y);
BINARY_FUNC_FLOATMID4(SUB, x-y);
BINARY_FUNC_FLOATMID4(MUL, x*y);
BINARY_FUNC_FLOATMID4(DIV, x/y);
BINARY_FUNC_FLOATMID4(REALDIV, (float)sign(y) * x / max(abs(y), 0.0000001));
BINARY_FUNC_FLOATMID4(MINIMUM, min(x, y));
BINARY_FUNC_FLOATMID4(MAXIMUM, max(x, y));
BINARY_FUNC_FLOATMID4(GREATER, x > y ? 1 : 0);
BINARY_FUNC_FLOATMID4(LESS, x < y ? 1 : 0);
BINARY_FUNC_FLOATMID4(LESS_EQUAL, x <= y ? 1 : 0);
BINARY_FUNC_FLOATMID4(GREATER_EQUAL, x >= y ? 1 : 0);
BINARY_FUNC_FLOATMID4(EQUAL, x == y ? 1 : 0);
BINARY_FUNC_FLOATMID4(NOTEQUAL, x != y ? 1 : 0);
BINARY_FUNC_FLOATMID4(FLOORDIV, floor(x / y));
BINARY_FUNC_FLOATMID4(FLOORMOD, x - floor(x / y) * y);
BINARY_FUNC_FLOATMID4(SquaredDifference, (x-y)*(x-y));
BINARY_FUNC_FLOATMID4(POW, pow(x, y));
BINARY_FUNC_FLOATMID4(ATAN2, atan2(x, y));
BINARY_FUNC_FLOATMID4(MOD, fmod(x, y));
BINARY_FUNC_FLOATMID4(LOGICALOR, (x || y) ? 1 : 0);

template<typename T>
void BinaryBlitTemplateFloat(T* output, const T* input, const T* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, int bytes, CUDARuntime* runtime, int opType, int activationType) {
    int count = size[0] * size[1] * size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    // MNN_PRINT("binary :%d %d %d, %d %d %d, %d %d %d, %d %d %d, \n", size[0], size[1], size[2], srcStride[0], srcStride[1], srcStride[2], srcStride1[0], srcStride1[1], srcStride1[2], dstStride[0], dstStride[1], dstStride[2]);
    #define COMPUTE_FLOAT(TYPE, TOut)\
        if (opType == MNN::BinaryOpOperation_##TYPE ) {\
            if (size[2] == count) {\
                if(count % 4 == 0 && count > 16384 && (srcStride[2] == 0 || srcStride[2] == 1) && (srcStride1[2] == 0 || srcStride1[2] == 1) && dstStride[2] == 1) {\
                    block_num = runtime->blocks_num(count/4);\
                    threads_num = runtime->threads_num();\
                    bool srcBroadcast = srcStride[2] == 0;\
                    bool srcBroadcast1 = srcStride1[2] == 0;\
                    if(bytes == 4) {\
                        BinaryMidLinear4_##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                            count/4, activationType, srcBroadcast, srcBroadcast1);\
                    } else {\
                        BinaryMidLinearHalf4_##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                            count/4, activationType, srcBroadcast, srcBroadcast1);\
                    }\
                } else {\
                    BinaryMidLinear##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                        size[2],\
                        srcStride[2],\
                        srcStride1[2],\
                        dstStride[2],\
                        activationType);\
                }\
            } else {\
                bool isVectorSizeZ = (size[0] == 1 || ((srcStride[2] == 0 || srcStride[0] % bytes == 0) && (srcStride1[2] == 0 || srcStride1[0] % bytes == 0) && dstStride[0] % bytes == 0));\
                bool isVectorSizeY = (size[1] == 1 || ((srcStride[2] == 0 || srcStride[1] % bytes == 0) && (srcStride1[2] == 0 || srcStride1[1] % bytes == 0) && dstStride[1] % bytes == 0));\
                bool isVector4 = size[2] % bytes == 0 && isVectorSizeZ && isVectorSizeY;\
		        if(isVector4 && count > 16384 && (srcStride[2] == 0 || srcStride[2] == 1) && (srcStride1[2] == 0 || srcStride1[2] == 1) && dstStride[2] == 1) {\
                    block_num = runtime->blocks_num(count/bytes);\
                    threads_num = runtime->threads_num();\
                    DivModFast sy(size[1]);\
                    DivModFast sx(size[2]/bytes);\
                    bool srcBroadcast = srcStride[2] == 0;\
                    bool srcBroadcast1 = srcStride1[2] == 0;\
                    if(bytes == 4) {\
                        BinaryMid4_##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                            size[0], size[1], size[2]/4,\
                            srcStride[0], srcStride[1],\
                            srcStride1[0], srcStride1[1],\
                            dstStride[0], dstStride[1], activationType, sy, sx, srcBroadcast, srcBroadcast1);\
                    } else {\
                        BinaryMidHalf2_##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                            size[0], size[1], size[2]/2,\
                            srcStride[0], srcStride[1],\
                            srcStride1[0], srcStride1[1],\
                            dstStride[0], dstStride[1], activationType, sy, sx, srcBroadcast, srcBroadcast1);\
                    }\
                } else {\
                    DivModFast sy(size[1]);\
                    DivModFast sx(size[2]);\
                    BinaryMid##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                        size[0], size[1], size[2],\
                        srcStride[0], srcStride[1], srcStride[2],\
                        srcStride1[0], srcStride1[1], srcStride1[2],\
                        dstStride[0], dstStride[1], dstStride[2], activationType, sy, sx);\
                }\
            }\
            return;\
        }\

    COMPUTE_FLOAT(ADD, T);
    COMPUTE_FLOAT(SUB, T);
    COMPUTE_FLOAT(MUL, T);
    COMPUTE_FLOAT(DIV, T);
    COMPUTE_FLOAT(REALDIV, T);
    COMPUTE_FLOAT(MINIMUM, T);
    COMPUTE_FLOAT(MAXIMUM, T);
    COMPUTE_FLOAT(GREATER, int);
    COMPUTE_FLOAT(LESS, int);
    COMPUTE_FLOAT(LESS_EQUAL, int);
    COMPUTE_FLOAT(GREATER_EQUAL, int);
    COMPUTE_FLOAT(EQUAL, int);
    COMPUTE_FLOAT(NOTEQUAL, int);
    COMPUTE_FLOAT(FLOORDIV, T);
    COMPUTE_FLOAT(FLOORMOD, T);
    COMPUTE_FLOAT(POW, T);
    COMPUTE_FLOAT(SquaredDifference, T);
    COMPUTE_FLOAT(ATAN2, T);
    COMPUTE_FLOAT(MOD, T);

    #undef COMPUTE_FLOAT
}

void BinaryBlitTemplateInt32(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, int bytes, CUDARuntime* runtime, int opType, int activationType) {
    int count = size[0] * size[1] * size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    #define COMPUTE_INT(TYPE, TOut)\
    if (opType == MNN::BinaryOpOperation_##TYPE ) {\
            Binary##TYPE<<<block_num, threads_num>>>((const int*)input, (const int*)(input1), (TOut*)output,\
                size[0], size[1], size[2],\
                srcStride[0], srcStride[1], srcStride[2],\
                srcStride1[0], srcStride1[1], srcStride1[2],\
                dstStride[0], dstStride[1], dstStride[2], activationType);\
        return;\
    }\

    COMPUTE_INT(ADD, int);
    COMPUTE_INT(SUB, int);
    COMPUTE_INT(MUL, int);
    COMPUTE_INT(DIV, int);
    COMPUTE_INT(MINIMUM, int);
    COMPUTE_INT(MAXIMUM, int);
    COMPUTE_INT(GREATER, int);
    COMPUTE_INT(LESS, int);
    COMPUTE_INT(LESS_EQUAL, int);
    COMPUTE_INT(GREATER_EQUAL, int);
    COMPUTE_INT(EQUAL, int);
    COMPUTE_INT(NOTEQUAL, int);
    COMPUTE_INT(SquaredDifference, int);
    COMPUTE_INT(MOD, int);
    COMPUTE_INT(LOGICALOR, int);
}


void BinaryBlit(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, halide_type_t type, CUDARuntime* runtime, int opType, int activationType) {
    if (type.code == halide_type_float) {
        if (type.bits == 32) {
            BinaryBlitTemplateFloat((float*)output, (float*)input, (float*)input1, size, srcStride, srcStride1, dstStride, type.bytes(), runtime, opType, activationType);
        } else if (type.bits == 16) {
            BinaryBlitTemplateFloat((half*)output, (half*)input, (half*)input1, size, srcStride, srcStride1, dstStride, type.bytes(), runtime, opType, activationType);
        } else {
            MNN_ERROR("CUDA not supoort data code:%d, data bits:%d\n", type.code, type.bits);
        }
    } else if (type.code == halide_type_int) {
        if(type.bits == 32) {
            BinaryBlitTemplateInt32(output, input, input1, size, srcStride, srcStride1, dstStride, type.bytes(), runtime, opType, activationType);
        } else {
            MNN_ERROR("CUDA not supoort data code:%d, data bits:%d\n", type.code, type.bits);
        }
    } else {
        MNN_ERROR("CUDA not supoort data code:%d, data bits:%d\n", type.code, type.bits);
    }
}


void BinaryBlitFuse(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, halide_type_t type, CUDARuntime* runtime, int opType, int fuseType) {
    int count = size[0] * size[1] * size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
#define COMPUTE_FLOAT_FUSE(TYPE, T)\
    if (opType == MNN::BinaryOpOperation_##TYPE ) {\
        BinaryFuseAdd##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (T*)output,\
            size[0], size[1], size[2],\
            srcStride[0], srcStride[1], srcStride[2],\
            srcStride1[0], srcStride1[1], srcStride1[2],\
            dstStride[0], dstStride[1], dstStride[2]);\
        return;\
    }\

    COMPUTE_FLOAT_FUSE(ADD, float);
    COMPUTE_FLOAT_FUSE(SUB, float);
    COMPUTE_FLOAT_FUSE(MUL, float);
    COMPUTE_FLOAT_FUSE(DIV, float);
    COMPUTE_FLOAT_FUSE(REALDIV, float);
    COMPUTE_FLOAT_FUSE(MINIMUM, float);
    COMPUTE_FLOAT_FUSE(MAXIMUM, float);
    COMPUTE_FLOAT_FUSE(FLOORDIV, float);
    COMPUTE_FLOAT_FUSE(FLOORMOD, float);
    COMPUTE_FLOAT_FUSE(POW, float);
    COMPUTE_FLOAT_FUSE(SquaredDifference, float);
    COMPUTE_FLOAT_FUSE(ATAN2, float);

#undef COMPUTE_FLOAT_FUSE
}


}// namespace CUDA
}// namespace MNN
