#include "ScatterNdPlugin.hpp"
namespace MNN {

template <typename T>
__global__ void SetZero(const int n, T* outputPtr) {
    CUDA_KERNEL_LOOP(index, n) {
        outputPtr[index] = (T)0;
    }
}

struct Lock{
    int *mutex;
    Lock(void){
      int state = 0;
      cudaMalloc((void**) &mutex, sizeof(int));
      cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }
    ~Lock(void){
      cudaFree(mutex);
    }
    __device__ void lock(void){
      while(atomicCAS(mutex, 0, 1) != 0);
    }
    __device__ void unlock(void){
      atomicExch(mutex, 0);
    }
};

template <typename T>
__global__ void ScatterNd(const int n, const int indicesLastDim, const int accNumber, const T* indicesPtr,
                          const T* updatesPtr, T* outputPtr, const int* dimsToCount, Lock cuLock);


template <>
__global__ void ScatterNd<float>(const int n, const int indicesLastDim, const int accNumber, const float* indicesPtr,
                          const float* updatesPtr, float* outputPtr, const int* dimsToCount, Lock cuLock) {
    CUDA_KERNEL_LOOP(index, n) {
        int pos = 0;
        for (int j = 0; j < indicesLastDim; ++j) {
            auto curIndex = (int)indicesPtr[index * indicesLastDim + j];
            // MNN_ASSERT(curIndex >= 0 && curIndex < output->length(j));
            pos += curIndex * dimsToCount[j];
        }
        for (int k = 0; k < accNumber; ++k) {
            float updateValue = updatesPtr[index * accNumber + k];
            atomicAdd(outputPtr + pos + k, updateValue);
        }
    }
}

template <>
__global__ void ScatterNd<__half>(const int n, const int indicesLastDim, const int accNumber, const __half* indicesPtr,
                          const __half* updatesPtr, __half* outputPtr, const int* dimsToCount, Lock cuLock) {
    CUDA_KERNEL_LOOP(index, n) {
        int pos = 0;
        for (int j = 0; j < indicesLastDim; ++j) {
            auto curIndex = (int)indicesPtr[index * indicesLastDim + j];
            // MNN_ASSERT(curIndex >= 0 && curIndex < output->length(j));
            pos += curIndex * dimsToCount[j];
        }
        for (int k = 0; k < accNumber; ++k) {
            float updateValue = updatesPtr[index * accNumber + k];
            cuLock.lock();
            outputPtr[pos + k] += updateValue;
            cuLock.unlock();
        }
    }
}

cudaError_t ScatterNdPlugin::ScatterNdExecute(nvinfer1::DataType dataType, const int count, const int outElementSize, const int indicesLastDim,
                                       const int accNumber, const float* indice, const void* update, void* top_data,
                                       const int32_t* dimsToCount, cudaStream_t stream) {
    Lock cuLock;
    if (dataType == nvinfer1::DataType::kFLOAT){
        SetZero<float><<<CAFFE_GET_BLOCKS(outElementSize), CUDA_NUM_THREADS>>>(outElementSize, (float*)top_data);
        ScatterNd<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
        count, indicesLastDim, accNumber, (const float*)indice, (const float*)update, (float*)top_data, dimsToCount, cuLock);
    }else{
        SetZero<__half><<<CAFFE_GET_BLOCKS(outElementSize), CUDA_NUM_THREADS>>>(outElementSize, (__half*)top_data);
        ScatterNd<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
        count, indicesLastDim, accNumber, (const __half*)indice, (const __half*)update, (__half*)top_data, dimsToCount, cuLock);
    }

    return cudaPeekAtLastError();
}

}; // namespace MNN