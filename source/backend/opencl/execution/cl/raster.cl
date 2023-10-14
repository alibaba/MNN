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
                    __global FLOAT *output
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(x, y);
    
    output[y*global_size_dim0 + x] = (FLOAT)(0.0f);
}

__kernel void image_set_zero(
                    GLOBAL_SIZE_2_DIMS
                    __write_only image2d_t output
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(x, y);

    WI_F(output, (int2)(x, y), (FLOAT4)(0.0f));
}

__kernel void raster_buffer(
                    GLOBAL_SIZE_3_DIMS
                    __global FLOAT *input,
                    __private const int inputOffset,
                    __private const int inputStride0,
                    __private const int inputStride1,
                    __private const int inputStride2,
                    __global FLOAT *output,
                    __private const int outputOffset,
                    __private const int outputStride0,
                    __private const int outputStride1,
                    __private const int outputStride2
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    int inputIndex = inputOffset + z * inputStride0 + y * inputStride1 + x * inputStride2;
    int outputIndex = outputOffset + z * outputStride0 + y * outputStride1 + x * outputStride2;
    output[outputIndex] = input[inputIndex];
}

__kernel void raster_buffer_combine(
                    GLOBAL_SIZE_3_DIMS
                    __global FLOAT *input,
                    __private const int inputOffset,
                    __private const int combineSrcOffset,
                    __private const int inputStride0,
                    __private const int inputStride1,
                    __private const int inputStride2,
                    __global FLOAT *output,
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
    output[outputIndex] = input[inputIndex];
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

    FLOAT4 out = RI_F(input, SAMPLER, (int2)(inp_idx0, inp_idx1));
    WI_F(output, (int2)(out_idx0, out_idx1), out);
}
