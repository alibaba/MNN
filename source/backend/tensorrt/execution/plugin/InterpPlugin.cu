#include "InterpPlugin.hpp"
namespace MNN {

template <typename T>
__global__ void Interp(const int n, const float height_scale, const float width_scale, const float height_offset, const float width_offset, 
                       const int input_height, const int input_width, const int out_height, const int outputWidth, 
                       const T* input, T* output) {
    CUDA_KERNEL_LOOP(index, n) {
        int x         = index % outputWidth;
        int tmp       = index / outputWidth;
        int y         = tmp % out_height;
        int z         = tmp / out_height;
        int ix        = min(max(0, (int)floor((float)x * width_scale)), input_width - 1);
        int iy        = min(max(0, (int)floor(((float)y * height_scale))), input_height - 1);
        T inValue = input[z * input_height * input_width + ix + iy * input_width];
        output[z * outputWidth * out_height + y * outputWidth + x] = inValue;
    }
}

template<typename T>
__global__ void INTERP_BILINEAR(const int n, const int ih, const int iw, const int oh, const int ow, 
    const float scaleh, const float scalew, const float offseth, const float offsetw, const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        float fx = x*scalew+offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw-1);
        int ix_1 = min(ix_0+1, iw-1);
        float fy = y*scaleh+offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih-1);
        int iy_1 = min(iy_0+1, ih-1);

        int index_00 = z*ih*iw + iy_0*iw + ix_0;
        int index_01 = z*ih*iw + iy_0*iw + ix_1;
        int index_10 = z*ih*iw + iy_1*iw + ix_0;
        int index_11 = z*ih*iw + iy_1*iw + ix_1;

        float factor_x = fx-ix_0;
        float factor_y = fy-iy_0;
        out[z*oh*ow + y*ow + x] = (T)((1.0-factor_x)*(1.0-factor_y)*((float)in[index_00]) + factor_x*(1.0-factor_y)*((float)in[index_01]) +
                                  (1.0-factor_x)*factor_y*((float)in[index_10]) + factor_x*factor_y*((float)in[index_11]));
    }
}

cudaError_t InterpPlugin::InterpExecute(nvinfer1::DataType dataType, const int count, const float heightScale, const float widthScale,
                                        const float heightOffset, const float widthOffset, const int inputHeight, const int inputWidth, 
                                        const int outputHeight, const int outputWidth, const float* bottom_data, float* top_data,
                                        cudaStream_t stream) {
    if(mResizeType == 1) {
        if (dataType == nvinfer1::DataType::kFLOAT){
            Interp<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, heightScale, widthScale, heightOffset, widthOffset, inputHeight, inputWidth, outputHeight, outputWidth,
            (const float*)bottom_data, (float*)top_data);
        }else{
            Interp<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, heightScale, widthScale, heightOffset, widthOffset, inputHeight, inputWidth, outputHeight, outputWidth,
            (const __half*)bottom_data, (__half*)top_data);
        }
    } else if(mResizeType == 2) {
        if (dataType == nvinfer1::DataType::kFLOAT){
            INTERP_BILINEAR<float><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
                count, inputHeight, inputWidth, outputHeight, outputWidth, heightScale, widthScale, 
                heightOffset, widthOffset, (const float*)bottom_data, (float*)top_data);
        }else{
            INTERP_BILINEAR<__half><<<CAFFE_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
                count, inputHeight, inputWidth, outputHeight, outputWidth, heightScale, widthScale, 
                heightOffset, widthOffset, (const __half*)bottom_data, (__half*)top_data);
        }
    } else {
        printf("Interp Type:%d Not supported!\n", mResizeType);
    }
    return cudaPeekAtLastError();
}

}; // namespace MNN