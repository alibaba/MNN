#ifndef ReductionTemplate_cuh
#define ReductionTemplate_cuh

#include "MNNCUDAFunction.cuh"
struct ReduceParam {
    int inside;
    int axis;
    int outside;
};
template <typename T>
__global__ void SUM_NAIVE(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside
) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        float sumValue = 0.0;
        const T* basicInput = input + y * axis * inside + x;
        for (int v=0; v<axis; ++v) {
            sumValue += (float)basicInput[v * inside];
        }
        output[y * inside + x] = (T)sumValue;
    }
    return;
}

template <typename T>
__global__ void SUM_REDUCE_AXIS(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside,
    const int per_block_size,
    const int calc_multi_num
) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x -  idx_outside * inside;

    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;

    float local_src = 0.0;
    __shared__ float sumValue;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            local_src += (float)(src[(tid + i * per_block_size) * inside]);
        }
    }
    float maxRes = blockReduceSum<float>(local_src);
    if(tid == 0)
        sumValue = maxRes;
    __syncthreads();

    output[idx_outside * inside + idx_inside] = (T)sumValue;
    return;
}


template <typename T>
__global__ void MEAN_NAIVE(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside
) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        float sumValue = 0.0;
        
        const T* basicInput = input + y * axis * inside + x;
        for (int v=0; v<axis; ++v) {
            sumValue += (float)basicInput[v * inside];
        }
        output[y * inside + x] = (T)(sumValue / (float)axis);
    }
    return;
}

template <typename T>
__global__ void MEAN_REDUCE_AXIS(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside,
    const int per_block_size,
    const int calc_multi_num
) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x -  idx_outside * inside;

    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;

    float local_src = 0.0;
    __shared__ float sumValue;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            local_src += (float)(src[(tid + i * per_block_size) * inside]);
        }
    }
    float maxRes = blockReduceSum<float>(local_src);
    if(tid == 0)
        sumValue = maxRes;
    __syncthreads();

    output[idx_outside * inside + idx_inside] = (T)(sumValue / (float)axis);
    return;
}

template <typename T>
__global__ void MINIMUM(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside
) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        
        const T* basicInput = input + y * axis * inside + x;
        float res = (float)basicInput[0];
        for (int v=1; v<axis; ++v) {
            res = min((float)basicInput[v * inside], res);
        }
        output[y * inside + x] = (T)res;
    }
    return;
}

template <typename T>
__global__ void MAXIMUM(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside
) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        const T* basicInput = input + y * axis * inside + x;
        
        float res = (float)basicInput[0];
        for (int v=1; v<axis; ++v) {
            res = max((float)basicInput[v * inside], res);
        }
        output[y * inside + x] = (T)res;
    }
    return;
}

template <typename T>
__global__ void PROD(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside
) {
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        
        float sumValue = 1.0;
        const T* basicInput = input + y * axis * inside + x;
        for (int v=0; v<axis; ++v) {
            sumValue *= (float)basicInput[v * inside];
        }
        output[y * inside + x] = (T)sumValue;
    }
    return;
}

#endif