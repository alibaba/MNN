#include "BinaryPlugin.hpp"

namespace MNN {

template <typename T>
__global__ void ADD(const int n, const T* in0, const T* in1, T* out, int s0, int s1){
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = in0[index * s0] + in1[index * s1];
    }
}

template <typename T>
__global__ void SUB(const int n, const T* in0, const T* in1, T* out, int s0, int s1){
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = in0[index * s0] - in1[index * s1];
    }
}

template <typename T>
__global__ void MUL(const int n, const T* in0, const T* in1, T* out, int s0, int s1){
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = in0[index * s0] * in1[index * s1];
    }
}


template <typename T>
__global__ void DIV(const int n, const T* in0, const T* in1, T* out, int s0, int s1){
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = in0[index * s0] / in1[index * s1];
    }
}

template <typename T>
__global__ void SQD(const int n, const T* in0, const T* in1, T* out, int s0, int s1){
    CUDA_KERNEL_LOOP(index, n) {
        T data = in0[index * s0] - in1[index * s1];
        out[index] = data * data;
    }
}

template <typename T>
__global__ void MAXIMUM(const int n, const T* in0, const T* in1, T* out, int s0, int s1);

template <>
__global__ void MAXIMUM<float>(const int n, const float* in0, const float* in1, float* out, int s0, int s1) {
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = max(in0[index * s0], in1[index * s1]);
    }
}

template <>
__global__ void MAXIMUM<half>(const int n, const half* in0, const half* in1, half* out, int s0, int s1) {
    CUDA_KERNEL_LOOP(index, n) {
        float tmp = max(__half2float(in0[index * s0]) , __half2float(in1[index * s1]));
        out[index] = __float2half(tmp);
    }
}

template <typename T>
__global__ void MINIMUM(const int n, const T* in0, const T* in1, T* out, int s0, int s1);

template <>
__global__ void MINIMUM<float>(const int n, const float* in0, const float* in1, float* out, int s0, int s1) {
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = min(in0[index * s0], in1[index * s1]);
    }
}

template <>
__global__ void MINIMUM<half>(const int n, const half* in0, const half* in1, half* out, int s0, int s1) {
    CUDA_KERNEL_LOOP(index, n) {
        float tmp = min(__half2float(in0[index * s0]) , __half2float(in1[index * s1]));
        out[index] = __float2half(tmp);
    }
}

template <typename T>
__global__ void POW(const int n, const T* in0, const T* in1, T* out, int s0, int s1);


template <>
__global__ void POW<float>(const int n, const float* in0, const float* in1, float* out, int s0, int s1) {
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = pow(in0[index * s0], in1[index * s1]);
    }
}

template <>
__global__ void POW<half>(const int n, const half* in0, const half* in1, half* out, int s0, int s1) {
    CUDA_KERNEL_LOOP(index, n) {
        float tmp = pow(__half2float(in0[index * s0]), __half2float(in1[index * s1]));
        out[index] = __float2half(tmp);
    }
}

template <typename T>
cudaError_t binary_template(int type, const int count, const T* bottom_data0, const T* bottom_data1, T* top_data, int s0, int s1, cudaStream_t stream){
    if (type == 0) {
        ADD<T><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data0, bottom_data1, top_data, s0, s1);
    } else if (type == 1) {
        SUB<T><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data0, bottom_data1, top_data,
                                                                            s0, s1);
    } else if (type == 2) {
        MUL<T><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data0, bottom_data1, top_data,
                                                                            s0, s1);
    } else if (type == 6) {
        POW<T><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data0, bottom_data1, top_data,
                                                                            s0, s1);
    } else if (type == 3 || type == 7) {
        DIV<T><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data0, bottom_data1, top_data,
                                                                            s0, s1);
    } else if (type == 9) {
        MAXIMUM<T><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data0, bottom_data1,
                                                                                top_data, s0, s1);
    } else if (type == 8) {
        MINIMUM<T><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data0, bottom_data1,
                                                                                top_data, s0, s1);
    } else if (type == 14){
        SQD<T><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data0, bottom_data1, top_data, s0, s1);
    } else {
        printf("binary op not support:%d\n", type);
    }

    return cudaPeekAtLastError();
}

cudaError_t BinaryPlugin::BinaryExecute(nvinfer1::DataType dataType, const int count, const void *const *inputs, void **outputs, int s0, int s1, cudaStream_t stream) {
#ifdef TRT_LOG
    printf("in mType:%d\n", mType);
#endif
    if (dataType == nvinfer1::DataType::kFLOAT){
        return binary_template<float>(mType, count, (const float*)inputs[0], (const float*)inputs[1], (float*)outputs[0], s0, s1, stream);
    }else{
        return binary_template<__half>(mType, count, static_cast<const __half*>(inputs[0]), static_cast<const __half*>(inputs[1]), static_cast<__half*>(outputs[0]), s0, s1, stream);
    }
}

}; // namespace MNN