#include "Raster.cuh"
namespace MNN {
namespace CUDA {

// Blit don't care offset
template <typename T>
__global__ void blit(const T *input, T *output,
        int sizeZ, int sizeY, int sizeX,
        int strideZ, int strideY, int strideX,
        int dstStrideZ, int dstStrideY, int dstStrideX
        ) {
    int total = sizeZ * sizeY * sizeX;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % sizeX;
        int tmp = i / sizeX;
        int y = tmp % sizeY;
        int z = tmp / sizeY;
        int srcOffset = z * strideZ + y * strideY + x * strideX;
        int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
        output[dstOffset] = input[srcOffset];
    }
}

void RasterBlit(uint8_t* output, const uint8_t* input, const Tensor::InsideDescribe::Region& reg, int bytes, CUDARuntime* runtime) {
    int count = reg.size[0] * reg.size[1] * reg.size[2];
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    switch (bytes) {
        case 4:
            blit<<<block_num, threads_num>>>((const float*)input, (float*)output,
                reg.size[0], reg.size[1], reg.size[2],
                reg.src.stride[0], reg.src.stride[1], reg.src.stride[2],
                reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            break;
        case 2:
            blit<<<block_num, threads_num>>>((const int16_t*)input, (int16_t*)output,
                reg.size[0], reg.size[1], reg.size[2],
                reg.src.stride[0], reg.src.stride[1], reg.src.stride[2],
                reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            break;
        case 1:
            blit<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output,
                reg.size[0], reg.size[1], reg.size[2],
                reg.src.stride[0], reg.src.stride[1], reg.src.stride[2],
                reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            break;
        default:
            break;
    }
}

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

}
}
