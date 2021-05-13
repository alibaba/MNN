#include "LayerNormPlugin.hpp"
namespace MNN {

template <typename T>
__global__ void LayerNorm(const int outter_size_, const int inner_size_, float epsilon_, const T* in, T* out,
                      const float* gamma, const float* beta);

template <>
__global__ void LayerNorm<float>(const int outter_size_, const int inner_size_, float epsilon_, const float* in, float* out,
    const float* gamma, const float* beta) {
    CUDA_KERNEL_LOOP(i, outter_size_) {
        int inner_input_index = i * inner_size_;
        int inner_output_index = i * inner_size_;
        float sum = 0.f;
        for (int j = 0; j < inner_size_; ++j) {
            sum += in[inner_input_index + j];
        }
        float mean = sum / inner_size_;
        float square_sum = 0.f;
        for (int j = 0; j < inner_size_; ++j) {
            square_sum += (in[inner_input_index + j] - mean) * (in[inner_input_index + j] - mean);
        }
        float variable = square_sum / inner_size_;
        variable = 1.f / std::sqrt(variable + epsilon_);

        for (int j = 0; j < inner_size_; ++j) {
            out[inner_output_index + j] = (in[inner_input_index + j] - mean) * variable * gamma[j] + beta[j];
        }
    }
}

template <>
__global__ void LayerNorm<__half>(const int outter_size_, const int inner_size_, float epsilon_, const __half* in, __half* out,
    const float* gamma, const float* beta) {
    CUDA_KERNEL_LOOP(i, outter_size_) {
        int inner_input_index = i * inner_size_;
        int inner_output_index = i * inner_size_;
        float sum = 0.f;
        for (int j = 0; j < inner_size_; ++j) {
            float data = __half2float(in[inner_input_index + j]);
            sum += data;
        }
        float mean = sum / inner_size_;
        float square_sum = 0.f;
        for (int j = 0; j < inner_size_; ++j) {
            float data = __half2float(in[inner_input_index + j]);
            square_sum += (data - mean) * (data - mean);
        }
        float variable = square_sum / inner_size_;
        variable = 1.f / std::sqrt(variable + epsilon_);

        for (int j = 0; j < inner_size_; ++j) {
            float data = __half2float(in[inner_input_index + j]);
            out[inner_output_index + j] = __float2half((data - mean) * variable * gamma[j] + beta[j]);
        }
    }
}

cudaError_t LayerNormPlugin::LayerNormExecute(nvinfer1::DataType dataType, const int outter_size_, const int inner_size_, const float* bottom_data,
                                      float* top_data, const float* gamma, const float* beta, cudaStream_t stream) {
    
    if (dataType == nvinfer1::DataType::kFLOAT){
        LayerNorm<float><<<CAFFE_GET_BLOCKS(outter_size_), CUDA_NUM_THREADS>>>(outter_size_, inner_size_, mEpsilon, bottom_data, top_data,
            gamma, beta);
    }else{
        LayerNorm<__half><<<CAFFE_GET_BLOCKS(outter_size_), CUDA_NUM_THREADS>>>(outter_size_, inner_size_, mEpsilon, (const __half*)bottom_data, (__half*)top_data,
        gamma, beta);
    }

    return cudaPeekAtLastError();
}
}; // namespace MNN