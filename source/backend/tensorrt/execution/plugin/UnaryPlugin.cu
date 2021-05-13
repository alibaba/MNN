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

template <typename T>
__device__ T evalPoly(T x, float* kErfTCoefficient, int size) {
    T poly = 0.0f;
    for (int i = 0; i < size; i++) {
        poly = poly * x + kErfTCoefficient[i];
    }
    return poly;
}

template <typename T>
__device__ T erfImpl(T x) {
    float kErfTCoefficient[7] = {
        +7.853861353153693E-5f, -8.010193625184903E-4f, +5.188327685732524E-3f,
        -2.685381193529856E-2f, +1.128358514861418E-1f, -3.761262582423300E-1f,
        +1.128379165726710E+0f,
    };
    return x * evalPoly(x * x, kErfTCoefficient, 7);
}

template <typename T>
__device__ T erfcImpl(T x) {
    // Coefficients for erfc(f32), from Cephes. tensorflow
    const double kMaxlog = 88.72283905206835;
    // erfc(x) = exp(-x^2) P(1/x^2), 1 < x < 2
    float kErfcPCoefficient[9] = {
            +2.326819970068386E-2f, -1.387039388740657E-1f, +3.687424674597105E-1f,
            -5.824733027278666E-1f, +6.210004621745983E-1f, -4.944515323274145E-1f,
            +3.404879937665872E-1f, -2.741127028184656E-1f, +5.638259427386472E-1f,
    };
    // erfc(x) = exp(-x^2) R(1/x^2), 2 <= x < kMaxlog
    float kErfcRCoefficient[8] = {
            -1.047766399936249E+1f, +1.297719955372516E+1f, -7.495518717768503E+0f,
            +2.921019019210786E+0f, -1.015265279202700E+0f, +4.218463358204948E-1f,
            -2.820767439740514E-1f, +5.641895067754075E-1f,
    };
    float absX = fabsf(x);
    float z = expf(-x * x);
    float q = 1.0 / absX;
    float y = q * q;
    float p;
    if (absX < 2.0f) {
        p = evalPoly(y, kErfcPCoefficient, 9);
    } else {
        p = evalPoly(y, kErfcRCoefficient, 8);
    }
    y = z * q * p;
    float yClamp;
    if (z < -kMaxlog) {
        yClamp = 0.0f;
    } else {
        yClamp = y;
    }
    if (x < 0) {
        return T(2.0f - yClamp);
    } else {
        return T(yClamp);
    }
}


template <typename T>
__global__ void ERF(const int n, const T* in, T* out);

template <>
__global__ void ERF<float>(const int n, const float* in, float* out) {
    CUDA_KERNEL_LOOP(index, n) {
        if(abs(in[index]) < float(1.)) {
            out[index] = erfImpl<float>(in[index]);
        } else {
            out[index] = float(1.) - erfcImpl<float>(in[index]);
        }
    }
}


template <>
__global__ void ERF<__half>(const int n, const __half* in, __half* out) {
    CUDA_KERNEL_LOOP(index, n) {
        if(abs(__half2float(in[index])) < float(1.)) {
            out[index] = __float2half(erfImpl<float>(__half2float(in[index])));
        } else {
            out[index] = __float2half(float(1.) - erfcImpl<float>(__half2float(in[index])));
        }
    }
}

template <typename T>
__global__ void HARDSWISH(const int n, const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        if(in[index] <= (T)(-3)) {
            out[index] = 0;
        } else if(in[index] >= (T)3) {
            out[index] = in[index];
        } else {
            out[index] = in[index] * (in[index] + (T)3) / (T)6;
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
    } else if(mType == MNN::UnaryOpOperation_ERF) {
        if (dataType == nvinfer1::DataType::kFLOAT){
            ERF<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data, top_data);
        }else{
            ERF<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, (const __half*)bottom_data, (__half*)top_data);
        }
    } else if (mType == MNN::UnaryOpOperation_HARDSWISH){
        if (dataType == nvinfer1::DataType::kFLOAT){
            HARDSWISH<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data, top_data);
        }else{
            HARDSWISH<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, (const __half*)bottom_data, (__half*)top_data);
        }
    } else {
        printf("Unary Plugin:%d not support\n", mType);
    }
    return cudaPeekAtLastError();
}

}; // namespace MNN
