#include "ScalePlugin.hpp"
namespace MNN {

template <typename T>
__global__ void SCALE(const int n, const int channels, const int dim, const T* in, T* out,
                      const float* scaleData, const float* biasData);

template <>
__global__ void SCALE<float>(const int n, const int channels, const int dim, const float* in, float* out,
                      const float* scaleData, const float* biasData) {
    CUDA_KERNEL_LOOP(index, n) {
        int c      = (index / dim) % channels;
        out[index] = in[index] * scaleData[c] + biasData[c];
    }
}

template <>
__global__ void SCALE<__half>(const int n, const int channels, const int dim, const __half* in, __half* out,
                      const float* scaleData, const float* biasData) {
    CUDA_KERNEL_LOOP(index, n) {
        int c      = (index / dim) % channels;
        out[index] = in[index] * __float2half(scaleData[c]) + __float2half(biasData[c]);
    }
}

cudaError_t ScalePlugin::ScaleExecute(nvinfer1::DataType dataType, const int count, const int channels, const int dim, const float* bottom_data,
                                      float* top_data, const float* scale, const float* bias, cudaStream_t stream) {
    if (dataType == nvinfer1::DataType::kFLOAT){
        SCALE<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, channels, dim, bottom_data, top_data,
                                                                          scale, bias);
    }else{
        SCALE<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, channels, dim, (const __half*)bottom_data, (__half*)top_data,
                                                                          scale, bias);
    }

    return cudaPeekAtLastError();
}
}; // namespace MNN