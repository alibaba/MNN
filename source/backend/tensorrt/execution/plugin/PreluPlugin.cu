
#include "PreluPlugin.hpp"

namespace MNN {

template <typename T>
__global__ void PReLU(const int n, const int channels, const int dim, const T* in, T* out,
                      const float* slope_data, const int div_factor);

template <>
__global__ void PReLU<float>(const int n, const int channels, const int dim, const float* in, float* out,
                      const float* slope_data, const int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c      = (index / dim) % channels / div_factor;
        out[index] = (float)in[index] > 0 ? in[index] : in[index] * slope_data[c];
    }
}

template <>
__global__ void PReLU<__half>(const int n, const int channels, const int dim, const __half* in, __half* out,
                      const float* slope_data, const int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c      = (index / dim) % channels / div_factor;
        out[index] = (float)in[index] > 0 ? in[index] : in[index] * __float2half(slope_data[c]);
    }
}

cudaError_t PreluPlugin::PReLUExecute(nvinfer1::DataType dataType, const int count, const int channels, const int dim, const float* bottom_data,
                                      float* top_data, void* mDeviceKernel, const int div_factor, cudaStream_t stream) {
    if (dataType == nvinfer1::DataType::kFLOAT){
        PReLU<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
        count, channels, dim, bottom_data, top_data, static_cast<const float*>(mDeviceKernel), div_factor);
    }else{
        PReLU<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
        count, channels, dim, (const __half*)bottom_data, (__half*)top_data, static_cast<const float*>(mDeviceKernel), div_factor);
    }

    return cudaPeekAtLastError();
}

}; // namespace MNN
