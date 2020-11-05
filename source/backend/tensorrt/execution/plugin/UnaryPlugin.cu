#include "UnaryPlugin.hpp"

namespace MNN {

template <typename T>
__global__ void SIGN(const int n, const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        if(in[index] > (T)0.0000001) {
            out[index] = 1;
        } else if(in[index] < (T)(-0.0000001)) {
            out[index] = -1;
        } else {
            out[index] = 0;
        }
    }
}


cudaError_t UnaryPlugin::UnaryExecute(nvinfer1::DataType dataType, const int count, const float* bottom_data,
                                        float* top_data, cudaStream_t stream) {
    if(mType == MNN::UnaryOpOperation_SIGN) {
        if (dataType == nvinfer1::DataType::kFLOAT){
            SIGN<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data, top_data);
        }else{
            SIGN<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, (const __half*)bottom_data, (__half*)top_data);
        }
    } else {
        printf("Unary Plugin:%d not support\n", mType);
    }
    return cudaPeekAtLastError();
}

}; // namespace MNN
