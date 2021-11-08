#include "OneHotPlugin.hpp"
namespace MNN {

template <typename T>
__global__ void OneHotImpl(const int n, const float* depthPtr, int innerSize, const T* indices, const T* onValue,
                            const T* offValue, T* output) {
    CUDA_KERNEL_LOOP(i, n) {
        
        int depth = int(depthPtr[0]);

        for (int j = 0; j < depth; j++) {
            for (int k = 0; k < innerSize; k++) {
                int index = (int)(indices[i * innerSize + k]);
                int outputIdx = i*depth*innerSize + j*innerSize + k;
                if (index == j) {
                    output[outputIdx] = onValue[0];
                } else {
                    output[outputIdx] = offValue[0];
                }
            }
        }
    }
}

cudaError_t OneHotPlugin::OneHotExecute(nvinfer1::DataType dataType, const int count, const float* depth, int innerSize, const float* indices, const float* onValueTensor,
                                    const float* offValueTensor, float* outputTensor, cudaStream_t stream) {


    if (dataType == nvinfer1::DataType::kFLOAT){
        OneHotImpl<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, depth, innerSize, indices, onValueTensor, offValueTensor, outputTensor);
    }else{
        OneHotImpl<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, depth, innerSize, (const __half*)indices, (const __half*)onValueTensor, (const __half*)offValueTensor, (__half*)outputTensor);
    }           
    
    return cudaPeekAtLastError();
}

}; // namespace MNN