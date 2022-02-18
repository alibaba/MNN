#ifndef ReductionTemplate_cuh
#define ReductionTemplate_cuh
struct ReduceParam {
    int inside;
    int axis;
    int outside;
};
template <typename T>
__global__ void SUM(const T *input, T *output, const ReduceParam* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / param->inside;
        int x = i % param->inside;
        float sumValue = 0.0;
        int axis = param->axis;
        const T* basicInput = input + y * param->axis * param->inside + x;
        for (int v=0; v<axis; ++v) {
            sumValue += (float)basicInput[v * param->inside];
        }
        output[y * param->inside + x] = (T)sumValue;
    }
    return;
}

template <typename T>
__global__ void MEAN(const T *input, T *output, const ReduceParam* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / param->inside;
        int x = i % param->inside;
        float sumValue = 0.0;
        int axis = param->axis;
        const T* basicInput = input + y * param->axis * param->inside + x;
        for (int v=0; v<axis; ++v) {
            sumValue += (float)basicInput[v * param->inside];
        }
        output[y * param->inside + x] = (T)(sumValue / (float)param->axis);
    }
    return;
}

template <typename T>
__global__ void MINIMUM(const T *input, T *output, const ReduceParam* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / param->inside;
        int x = i % param->inside;
        int axis = param->axis;
        const T* basicInput = input + y * param->axis * param->inside + x;
        float res = (float)basicInput[0];
        for (int v=1; v<axis; ++v) {
            res = min((float)basicInput[v * param->inside], res);
        }
        output[y * param->inside + x] = (T)res;
    }
    return;
}

template <typename T>
__global__ void MAXIMUM(const T *input, T *output, const ReduceParam* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / param->inside;
        int x = i % param->inside;
        const T* basicInput = input + y * param->axis * param->inside + x;
        int axis = param->axis;
        float res = (float)basicInput[0];
        for (int v=1; v<axis; ++v) {
            res = max((float)basicInput[v * param->inside], res);
        }
        output[y * param->inside + x] = (T)res;
    }
    return;
}

template <typename T>
__global__ void PROD(const T *input, T *output, const ReduceParam* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int y = i / param->inside;
        int x = i % param->inside;
        int axis = param->axis;
        float sumValue = 1.0;
        const T* basicInput = input + y * param->axis * param->inside + x;
        for (int v=0; v<axis; ++v) {
            sumValue *= (float)basicInput[v * param->inside];
        }
        output[y * param->inside + x] = (T)sumValue;
    }
    return;
}

#endif