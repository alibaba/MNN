#include "InterpPlugin.hpp"
namespace MNN {

template <typename T>
__global__ void Interp(const int n, const T height_scale, const T width_scale, const int input_height,
                       const int input_width, const int out_height, const int outputWidth, const T* input,
                       T* output) {
    CUDA_KERNEL_LOOP(index, n) {
        int x         = index % outputWidth;
        int tmp       = index / outputWidth;
        int y         = tmp % out_height;
        int z         = tmp / out_height;
        int ix        = min(max(0, (int)floor((float)((T)x * width_scale))), input_width - 1);
        int iy        = min(max(0, (int)floor((float)((T)y * height_scale))), input_height - 1);
        float inValue = input[z * input_height * input_width + ix + iy * input_width];
        output[z * outputWidth * out_height + y * outputWidth + x] = inValue;
    }
}

cudaError_t InterpPlugin::InterpExecute(nvinfer1::DataType dataType, const int count, const float mHeightScale, const float mWidthScale,
                                 const int mInputHeight, const int mInputWidth, const int mOutputHeight,
                                 const int mOutputWidth, const float* bottom_data, float* top_data,
                                 cudaStream_t stream) {
    if (dataType == nvinfer1::DataType::kFLOAT){
        Interp<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
        count, mHeightScale, mWidthScale, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
        (const float*)bottom_data, (float*)top_data);
    }else{
        Interp<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
        count, (__half)mHeightScale, (__half)mWidthScale, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
        (const __half*)bottom_data, (__half*)top_data);
    }                  
    return cudaPeekAtLastError();
}

}; // namespace MNN