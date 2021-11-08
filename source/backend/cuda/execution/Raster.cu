#include "Raster.cuh"
#include "TensorflowOp_generated.h"
namespace MNN {
namespace CUDA {

template <typename T>
__global__ void pack_c4(const T *input, T *output, int inside, int axis, int outside, int axisC4) {
    int total = inside * axis * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % inside;
        int tmp = i / inside;
        int y = tmp % axis;
        int z = tmp / axis;
        int y4 = y / 4;
        int yR = y % 4;
        int dstOffset = 4 * (z * axisC4 * inside + y4 * inside + x) + yR;
        output[dstOffset] = input[i];
    }
}

void PackC4(uint8_t* output, const uint8_t* input, int inside, int axis, int outside, int bytes, CUDARuntime* runtime) {
    auto packAxis = (axis + 3) / 4;
    if (axis % 4 != 0) {
        runtime->memset(output, 0, inside * packAxis * 4 * outside * bytes);
    }
    int block_num = runtime->blocks_num(inside * axis * outside);
    int threads_num = runtime->threads_num();
    switch (bytes) {
        case 4:
            pack_c4<<<block_num, threads_num>>>((const float*)input, (float*)output, inside, axis, outside, packAxis);
            break;
        case 2:
            pack_c4<<<block_num, threads_num>>>((const int16_t*)input, (int16_t*)output, inside, axis, outside, packAxis);
            break;
        case 1:
            pack_c4<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output, inside, axis, outside, packAxis);
            break;
        default:
            break;
    }
}

template <typename T>
__global__ void unpack_c4(const T *input, T *output, int inside, int axis, int outside, int axisC4) {
    int total = inside * axis * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % inside;
        int tmp = i / inside;
        int y = tmp % axis;
        int z = tmp / axis;
        int y4 = y / 4;
        int yR = y % 4;
        int srcOffset = 4 * (z * axisC4 * inside + y4 * inside + x) + yR;
        output[i] = input[srcOffset];
    }
}
void UnpackC4(uint8_t* output, const uint8_t* input, int inside, int axis, int outside, int bytes, CUDARuntime* runtime) {
    auto packAxis = (axis + 3) / 4;
    int block_num = runtime->blocks_num(inside * axis * outside);
    int threads_num = runtime->threads_num();
    switch (bytes) {
        case 4:
            unpack_c4<<<block_num, threads_num>>>((const float*)input, (float*)output, inside, axis, outside, packAxis);
            break;
        case 2:
            unpack_c4<<<block_num, threads_num>>>((const int16_t*)input, (int16_t*)output, inside, axis, outside, packAxis);
            break;
        case 1:
            unpack_c4<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output, inside, axis, outside, packAxis);
            break;
        default:
            break;
    }
}


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
        int sizeZ, int sizeY, int sizeX,\
        int strideZ, int strideY, int strideX,\
        int dstStrideZ, int dstStrideY, int dstStrideX\
        ) { \
  int count = sizeZ * sizeY * sizeX;\
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
    int total = sizeZ * sizeY * sizeX;\
    int ix = i % sizeX;\
    int tmp = i / sizeX;\
    int iy = tmp % sizeY;\
    int iz = tmp / sizeY;\
    int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
    int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
    T x = input[srcOffset];\
    output[dstOffset] = Func;\
  }\
}\

UNARY_FUNC(blit, x);
UNARY_FUNC(ABS, abs(x));
UNARY_FUNC(EXP, exp(x));
UNARY_FUNC(NEG, -x);
UNARY_FUNC(RECIPROCAL, (T)(1.0)/x);
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

void RasterBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int bytes, CUDARuntime* runtime) {
    int count = size[0] * size[1] * size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    switch (bytes) {
        case 4:
            blit<<<block_num, threads_num>>>((const float*)input, (float*)output,
                size[0], size[1], size[2],
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 2:
            blit<<<block_num, threads_num>>>((const int16_t*)input, (int16_t*)output,
                size[0], size[1], size[2],
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        case 1:
            blit<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output,
                size[0], size[1], size[2],
                srcStride[0], srcStride[1], srcStride[2],
                dstStride[0], dstStride[1], dstStride[2]);
            break;
        default:
            break;
    }
}

void UnaryBlit(uint8_t* output, const uint8_t* input, const int32_t* size, const int32_t* srcStride, const int32_t* dstStride, int bytes, CUDARuntime* runtime, int opType) {
    int count = size[0] * size[1] * size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    // TODO: Support FP16
    MNN_ASSERT(bytes==4);
    #define COMPUTE(TYPE)\
    if (opType == MNN::UnaryOpOperation_##TYPE ) {\
            TYPE<<<block_num, threads_num>>>((const float*)input, (float*)output,\
                size[0], size[1], size[2],\
                srcStride[0], srcStride[1], srcStride[2],\
                dstStride[0], dstStride[1], dstStride[2]);\
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

    #undef COMPUTE
}

#define BINARY_FUNC(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void Binary##Name(\
        const TIn *input0, const TIn* input1, TOut *output,\
        int sizeZ, int sizeY, int sizeX,\
        int strideZ, int strideY, int strideX,\
        int strideZ1, int strideY1, int strideX1,\
        int dstStrideZ, int dstStrideY, int dstStrideX\
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
    output[dstOffset] = (TOut)Func;\
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
BINARY_FUNC(MOD, x - x / y);
BINARY_FUNC(LOGICALOR, (x || y) ? 1 : 0);

void BinaryBlitTemplateFloat(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, int bytes, CUDARuntime* runtime, int opType) {
    int count = size[0] * size[1] * size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    // TODO: Support FP16
    MNN_ASSERT(bytes==4);
    #define COMPUTE_FLOAT(TYPE, TOut)\
    if (opType == MNN::BinaryOpOperation_##TYPE ) {\
            Binary##TYPE<<<block_num, threads_num>>>((const float*)input, (const float*)(input1), (TOut*)output,\
                size[0], size[1], size[2],\
                srcStride[0], srcStride[1], srcStride[2],\
                srcStride1[0], srcStride1[1], srcStride1[2],\
                dstStride[0], dstStride[1], dstStride[2]);\
        return;\
    }\

    COMPUTE_FLOAT(ADD, float);
    COMPUTE_FLOAT(SUB, float);
    COMPUTE_FLOAT(MUL, float);
    COMPUTE_FLOAT(DIV, float);
    COMPUTE_FLOAT(REALDIV, float);
    COMPUTE_FLOAT(MINIMUM, float);
    COMPUTE_FLOAT(MAXIMUM, float);
    COMPUTE_FLOAT(GREATER, int);
    COMPUTE_FLOAT(LESS, int);
    COMPUTE_FLOAT(LESS_EQUAL, int);
    COMPUTE_FLOAT(GREATER_EQUAL, int);
    COMPUTE_FLOAT(EQUAL, int);
    COMPUTE_FLOAT(NOTEQUAL, int);
    COMPUTE_FLOAT(FLOORDIV, float);
    COMPUTE_FLOAT(FLOORMOD, float);
    COMPUTE_FLOAT(POW, float);
    COMPUTE_FLOAT(SquaredDifference, float);
    COMPUTE_FLOAT(ATAN2, float);
    COMPUTE_FLOAT(MOD, float);
}

void BinaryBlitTemplateInt32(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, int bytes, CUDARuntime* runtime, int opType) {
    int count = size[0] * size[1] * size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    #define COMPUTE_INT(TYPE, TOut)\
    if (opType == MNN::BinaryOpOperation_##TYPE ) {\
            Binary##TYPE<<<block_num, threads_num>>>((const int*)input, (const int*)(input1), (TOut*)output,\
                size[0], size[1], size[2],\
                srcStride[0], srcStride[1], srcStride[2],\
                srcStride1[0], srcStride1[1], srcStride1[2],\
                dstStride[0], dstStride[1], dstStride[2]);\
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


void BinaryBlit(uint8_t* output, const uint8_t* input, const uint8_t* input1, const int32_t* size, const int32_t* srcStride, const int32_t* srcStride1, const int32_t* dstStride, halide_type_t type, CUDARuntime* runtime, int opType) {
    if (type.code == halide_type_float) {
        BinaryBlitTemplateFloat(output, input, input1, size, srcStride, srcStride1, dstStride, type.bytes(), runtime, opType);
    } else if (type.code == halide_type_int) {
        BinaryBlitTemplateInt32(output, input, input1, size, srcStride, srcStride1, dstStride, type.bytes(), runtime, opType);
    }
}


}// namespace CUDA
}// namespace MNN
