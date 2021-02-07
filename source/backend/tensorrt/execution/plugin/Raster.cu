#include "Raster.cuh"
namespace MNN {

// Blit don't care offset
template <typename T>
__global__ void blit(const T* input, T* output, int sizeZ, int sizeY, int sizeX, int strideZ, int strideY, int strideX,
                     int dstStrideZ, int dstStrideY, int dstStrideX) {
    int total = sizeZ * sizeY * sizeX;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x             = i % sizeX;
        int tmp           = i / sizeX;
        int y             = tmp % sizeY;
        int z             = tmp / sizeY;
        int srcOffset     = z * strideZ + y * strideY + x * strideX;
        int dstOffset     = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
        output[dstOffset] = input[srcOffset];
    }
}

cudaError_t RasterBlit(nvinfer1::DataType dataType, uint8_t* output, const uint8_t* input, const Tensor::InsideDescribe::Region& reg, int bytes,
                cudaStream_t stream) {
    int count       = reg.size[0] * reg.size[1] * reg.size[2];

    switch (bytes) {
        case 4:
            if (dataType == nvinfer1::DataType::kFLOAT){
                blit<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
                (const float*)input, (float*)output, reg.size[0], reg.size[1], reg.size[2], reg.src.stride[0],
                reg.src.stride[1], reg.src.stride[2], reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            }
            break;
        case 2:
            if (dataType == nvinfer1::DataType::kHALF){
                blit<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
                (const __half*)input, (__half*)output, reg.size[0], reg.size[1], reg.size[2], reg.src.stride[0],
                reg.src.stride[1], reg.src.stride[2], reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            }else{
                blit<int16_t><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
                    (const int16_t*)input, (int16_t*)output, reg.size[0], reg.size[1], reg.size[2], reg.src.stride[0],
                    reg.src.stride[1], reg.src.stride[2], reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            }
            
            break;
        case 1:
            blit<int8_t><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
                (const int8_t*)input, (int8_t*)output, reg.size[0], reg.size[1], reg.size[2], reg.src.stride[0],
                reg.src.stride[1], reg.src.stride[2], reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
            break;
        default:
            break;
    }

    return cudaPeekAtLastError();
}

} // namespace MNN
