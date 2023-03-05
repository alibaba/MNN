#include "Raster.cuh"
#include "TensorflowOp_generated.h"
#include <cuda_fp16.h>
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

// Blit don't care offset
template <typename T>
__global__ void blitRegion(const T *inputO, T *outputO,
    int loopCount,
    const int32_t* dstIndice, const int32_t* srcIndice,
    int dstUseIndice, int srcUseIndice,
    int dstStep, int srcStep,int srcLimit,
    int sizeZ, int sizeY, int sizeX,
    int strideZ, int strideY, int strideX,
    int dstStrideZ, int dstStrideY, int dstStrideX
    ) {
    int total = loopCount;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
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
            for (int z=0; z<sizeZ; ++z) {
                for (int y=0; y<sizeY; ++y) {
                    for (int x=0; x<sizeX; ++x) {
                        int srcOffset = z * strideZ + y * strideY + x * strideX;
                        int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
                        output[dstOffset] = input[srcOffset];
                    }
                }
            }
        } else {
            T* output = outputO + dstOffsetO;
            for (int z=0; z<sizeZ; ++z) {
                for (int y=0; y<sizeY; ++y) {
                    for (int x=0; x<sizeX; ++x) {
                        int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
                        output[dstOffset] = (T)0;
                    }
                }
            }
        }
    }
}
void BlitWithIndice(uint8_t* output, const uint8_t* input, const int32_t* dstIndices, const int32_t* srcIndices, int dstUseIndice, int srcUseIndice, int loopCount, int dstStep, int srcStep, int srcLimit, const Tensor::InsideDescribe::Region& reg, int bytes, CUDARuntime* runtime) {
    int count = loopCount;
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    switch (bytes) {
        case 4:
            blitRegion<<<block_num, threads_num>>>((const float*)input, (float*)output, 
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

template<typename T>
__global__ void blit_2(const T *input, T *output,
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
UNARY_FUNC(SIGMOID, 1./(1.+exp(-x)));
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

    DivModFast sz(size[0]);
    DivModFast sy(size[1]);
    DivModFast sx(size[2]);

    //printf("%d-%d-%d, %d-%d-%d,-%d-%d-%d\n", size[0], size[1], size[2], srcStride[0], srcStride[1], srcStride[2], dstStride[0], dstStride[1], dstStride[2]);
    if(bytes == 4 && count > 16384 && size[2] % 2 == 0 && srcStride[2] == 1 && dstStride[2] == 1) {
        //printf("%d-%d-%d, %d-%d-%d,-%d-%d-%d\n\n", size[0], size[1], size[2], srcStride[0], srcStride[1], srcStride[2], dstStride[0], dstStride[1], dstStride[2]);
        count /= 2;
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        DivModFast sx_2((size[2]/2));

        blit_2<<<block_num, threads_num>>>((const float*)input, (float*)output, 
            count,
            sz, sy, sx_2,
            srcStride[0], srcStride[1],
            dstStride[0], dstStride[1]);
        return;
    }
    
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
        int total = sizeZ * sizeY * sizeX;\
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

#define BINARY_FUNC_FLOATMID(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMid##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY, int strideX,\
    int strideZ1, int strideY1, int strideX1,\
    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType, int bytes\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int total = sizeZ * sizeY * sizeX;\
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
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        if(bytes == 2) {\
            val = min(val, 65504.0f);\
            val = max(val, -65504.0f);\
        }\
        output[dstOffset] = val;\
    }\
}\
template<typename TIn, typename TOut>\
__global__ void BinaryMidLinear##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ,\
    int strideZ,\
    int strideZ1,\
    int dstStrideZ,\
    int activationType,\
    int bytes\
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
        if(bytes == 2) {\
            val = min(val, 65504.0f);\
            val = max(val, -65504.0f);\
        }\
        output[dstOffset] = (TOut)val;\
    }\
}\

#define BINARY_FUNC_FLOATMID4(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMidLinear4_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int count_4, int activationType\
    ) { \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count_4); i += blockDim.x * gridDim.x) {\
        int iz = i;\
        int srcOffset = iz << 2;\
        int srcOffset1 = iz << 2;\
        int dstOffset = iz << 2;\
        float4 xx = ((float4 *)(input0+srcOffset))[0];\
        float4 yy = ((float4 *)(input1+srcOffset1))[0];\
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
    int count_4, int activationType\
    ) { \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count_4); i += blockDim.x * gridDim.x) {\
        int iz = i;\
        int srcOffset = iz << 2;\
        int srcOffset1 = iz << 2;\
        int dstOffset = iz << 2;\
        half2 xx = ((half2 *)(input0+srcOffset))[0];\
        half2 yy = ((half2 *)(input1+srcOffset1))[0];\
        float x = (float)xx.x;\
        float y = (float)yy.x;\
        float val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        val = min(val, 65504.0f);\
        val = max(val, -65504.0f);\
        output[dstOffset] = (TOut)val;\
        x = (float)xx.y;\
        y = (float)yy.y;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        val = min(val, 65504.0f);\
        val = max(val, -65504.0f);\
        output[dstOffset+1] = (TOut)val;\
        xx = ((half2 *)(input0+srcOffset))[1];\
        yy = ((half2 *)(input1+srcOffset1))[1];\
        x = (float)xx.x;\
        y = (float)yy.x;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val <  0.0f ? 0.0f  : val);\
        }\
        val = min(val, 65504.0f);\
        val = max(val, -65504.0f);\
        output[dstOffset+2] = (TOut)val;\
        x = (float)xx.y;\
        y = (float)yy.y;\
        val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        val = min(val, 65504.0f);\
        val = max(val, -65504.0f);\
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
    #define COMPUTE_FLOAT(TYPE, TOut)\
        if (opType == MNN::BinaryOpOperation_##TYPE ) {\
            if (size[2] == count) {\
                if(count % 4 == 0 && count > 16384 && srcStride[2] == 1 && srcStride1[2] == 1 && dstStride[2] == 1) {\
                    block_num = runtime->blocks_num(count/4);\
                    threads_num = runtime->threads_num();\
                    if(bytes == 4) {\
                        BinaryMidLinear4_##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                            count/4, activationType);\
                    } else {\
                        BinaryMidLinearHalf4_##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                            count/4, activationType);\
                    }\
                } else {\
                    BinaryMidLinear##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                        size[2],\
                        srcStride[2],\
                        srcStride1[2],\
                        dstStride[2],\
                        activationType, bytes);\
                }\
            } else {\
                BinaryMid##TYPE<<<block_num, threads_num>>>((const T*)input, (const T*)(input1), (TOut*)output,\
                    size[0], size[1], size[2],\
                    srcStride[0], srcStride[1], srcStride[2],\
                    srcStride1[0], srcStride1[1], srcStride1[2],\
                    dstStride[0], dstStride[1], dstStride[2], activationType, bytes);\
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
        }
    } else if (type.code == halide_type_int) {
        BinaryBlitTemplateInt32(output, input, input1, size, srcStride, srcStride1, dstStride, type.bytes(), runtime, opType, activationType);
    }
}

}// namespace CUDA
}// namespace MNN
