#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,


#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void buffer_set_zero(
                    GLOBAL_SIZE_2_DIMS
                    __global OUTPUT_TYPE *output
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(x, y);
    
    output[y*global_size_dim0 + x] = (OUTPUT_TYPE)(0);
}

__kernel void image_set_zero(
                    GLOBAL_SIZE_2_DIMS
                    __write_only image2d_t output
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(x, y);

    WI_DATA(output, (int2)(x, y), (OUTPUT_TYPE_I4)(0));
}

__kernel void raster_buffer_direct(
                    GLOBAL_SIZE_3_DIMS
                    __read_only image2d_t input,
                    __private const int inputOffset,
                    __private const int combineSrcOffset,
                    __private const int inputStride0,
                    __private const int inputStride1,
                    __private const int inputStride2,
                    __private const int src_width,
                    __private const int src_height,
                    __private const int src_channel,
                    __global OUTPUT_TYPE *output,
                    __private const int outputOffset,
                    __private const int combineDstOffset,
                    __private const int outputStride0,
                    __private const int outputStride1,
                    __private const int outputStride2,
                    __private const int global_size0
                    ) {
    const int idx = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(idx, y, z);
    const int x = idx % global_size0;
    const int id = idx / global_size0;
    
    int inputIndex = inputOffset + id * combineSrcOffset + z * inputStride0 + y * inputStride1 + x * inputStride2;
    int outputIndex = outputOffset + id * combineDstOffset + z * outputStride0 + y * outputStride1 + x * outputStride2;
#ifdef INPUT_DATA_FORMAT_NHWC
    int in_c = inputIndex % src_channel; inputIndex /= src_channel;
    int in_w = inputIndex % src_width; inputIndex /= src_width;
    int in_h = inputIndex % src_height;
    int in_b = inputIndex / src_height;
#else
    int in_w = inputIndex % src_width; inputIndex /= src_width;
    int in_h = inputIndex % src_height; inputIndex /= src_height;
    int in_c = inputIndex % src_channel;
    int in_b = inputIndex / src_channel;
#endif
    int2 coord = (int2)((in_c / 4) * src_width + in_w, in_b * src_height + in_h);
    INPUT_TYPE_I4 value = RI_DATA(input, SAMPLER, coord);
    INPUT_TYPE_I* value_ptr = (INPUT_TYPE_I*)&value;
    output[outputIndex] = (OUTPUT_TYPE)value_ptr[in_c % 4];
}

__kernel void raster_image(
                    GLOBAL_SIZE_3_DIMS
                    __read_only image2d_t input,
                    __private const int inputOffset,
                    __private const int inputStride0,
                    __private const int inputStride1,
                    __private const int inputStride2,
                    __private const int inputHeight,
                    __private const int inputWidth,
                    __private const int inputChannel,
                    __write_only image2d_t output,
                    __private const int outputOffset,
                    __private const int outputStride0,
                    __private const int outputStride1,
                    __private const int outputStride2,
                    __private const int outputHeight,
                    __private const int outputWidth,
                    __private const int outputChannel
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    int inputIndex = inputOffset + (z * inputStride0 + y * inputStride1 + x * inputStride2) * 4;
    int outputIndex = outputOffset + (z * outputStride0 + y * outputStride1 + x * outputStride2) * 4;
    int inp_idx_n = inputIndex / ((inputChannel+3)/4 * inputHeight * inputWidth * 4);
    int inputIndex_left = inputIndex % ((inputChannel+3)/4 * inputHeight * inputWidth * 4);
    int inp_idx_c4 = inputIndex_left / (inputHeight * inputWidth * 4);
    inputIndex_left = inputIndex_left % (inputHeight * inputWidth * 4);
    int inp_idx_h = inputIndex_left / (inputWidth * 4);
    inputIndex_left = inputIndex_left % (inputWidth * 4);
    int inp_idx_w = inputIndex_left / 4;
    
    int out_idx_n = outputIndex / ((outputChannel+3)/4 * outputHeight * outputWidth * 4);
    int outputIndex_left = outputIndex % ((outputChannel+3)/4 * outputHeight * outputWidth * 4);
    int out_idx_c4 = outputIndex_left / (outputHeight * outputWidth * 4);
    outputIndex_left = outputIndex_left % (outputHeight * outputWidth * 4);
    int out_idx_h = outputIndex_left / (outputWidth * 4);
    outputIndex_left = outputIndex_left % (outputWidth * 4);
    int out_idx_w = outputIndex_left / 4;
    
    int inp_idx0 = inp_idx_c4*inputWidth + inp_idx_w;
    int inp_idx1 = inp_idx_n*inputHeight + inp_idx_h;
    int out_idx0 = out_idx_c4*outputWidth + out_idx_w;
    int out_idx1 = out_idx_n*outputHeight + out_idx_h;

    INPUT_TYPE_I4 out = RI_DATA(input, SAMPLER, (int2)(inp_idx0, inp_idx1));
    WI_DATA(output, (int2)(out_idx0, out_idx1), CONVERT_OUTPUT_I4(out));
}
