#include "GatherPlugin.hpp"

namespace MNN {

template <typename T>
__global__ void GATHER(const int count, const int outside, const int inside, const int iNum, const int oNum,
                            const T *input, const T* indice, T *output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int x = i % inside;
        int y = i / inside;
        const int o = y / oNum;
        const int n = y % oNum;

        T* outPtr = output + inside * oNum * o;
        const T* inpPtr = input + inside * iNum * o;
        outPtr[n*inside+x] = inpPtr[(int)indice[n]*inside+x];
    }
    return;
}


cudaError_t GatherPlugin::GatherExecute(nvinfer1::DataType dataType, const int count, const float* bottom_data, const float* indices,
                                        float* top_data, cudaStream_t stream) {
    if (dataType == nvinfer1::DataType::kFLOAT){
        GATHER<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, mOutside, mInside, mInpNum, mOutNum, bottom_data, indices, top_data);
    }else{
        GATHER<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, mOutside, mInside, mInpNum, mOutNum, (const __half*)bottom_data, (const __half*)indices, (__half*)top_data);
    }

    return cudaPeekAtLastError();
}

}; // namespace MNN
