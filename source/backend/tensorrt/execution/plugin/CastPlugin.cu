
#include "CastPlugin.hpp"

namespace MNN {

__global__ void cast_int_to_float(const int n, const int* in, float* out) {
    CUDA_KERNEL_LOOP(index, n) {
        int data = in[index];
        out[index] = (float)data;
    }
}

cudaError_t CastPlugin::CastInt32ToFloatExecute(nvinfer1::DataType dataType, const int count, const int* bottom_data,
                                      float* top_data, cudaStream_t stream) {

    cast_int_to_float<<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data, top_data);

    return cudaPeekAtLastError();
}

}; // namespace MNN
