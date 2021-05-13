#include "GatherPlugin.hpp"

namespace MNN {

template <typename T>
__global__ void GATHER(const int count, const int outputOutsideStride, const int inputOutsideStride, const int N, const int limit, int insideStride,
                            const T *inputPtr, const T* indicesPtr, T *outputPtr) {
    CUDA_KERNEL_LOOP(index, count) {
        int o = index / (N*insideStride);
        int o_r = index % (N*insideStride);
        int i = o_r / insideStride;
        int s = o_r % insideStride;

        int outputIdx = outputOutsideStride * o + i * insideStride + s;
        int indices = int(indicesPtr[i]);
        if (indices < 0 || indices > limit) {
            outputPtr[outputIdx] = 0.0f;
        }else{
            int inputIdx = inputOutsideStride * o + insideStride * indices + s;
            outputPtr[outputIdx] = inputPtr[inputIdx];
        }
    }
}


cudaError_t GatherPlugin::GatherExecute(nvinfer1::DataType dataType, const int count, const float* bottom_data, const float* indices,
                                        float* top_data, cudaStream_t stream) {
    if (dataType == nvinfer1::DataType::kFLOAT){
        GATHER<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, mOutputOutsideStride, mInputOutsideStride, mN, mLimit, mInsideStride, bottom_data, indices, top_data);
    }else{
        GATHER<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, mOutputOutsideStride, mInputOutsideStride, mN, mLimit, mInsideStride, (const __half*)bottom_data, (const __half*)indices, (__half*)top_data);
    }
    return cudaPeekAtLastError();
}

}; // namespace MNN
