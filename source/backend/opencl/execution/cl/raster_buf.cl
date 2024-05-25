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
    
    output[y*global_size_dim0 + x] = (OUTPUT_TYPE)(0.0f);
}

__kernel void raster_buffer(
                    GLOBAL_SIZE_3_DIMS
                    __global INPUT_TYPE *input,
                    __private const int inputOffset,
                    __private const int inputStride0,
                    __private const int inputStride1,
                    __private const int inputStride2,
                    __global OUTPUT_TYPE *output,
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
    output[outputIndex] = (OUTPUT_TYPE)input[inputIndex];
}


__kernel void raster_nc4hw4_buffer(
                    GLOBAL_SIZE_3_DIMS
                    __global INPUT_TYPE *input,
                    __private const int inputOffset,
                    __private const int inputStride0,
                    __private const int inputStride1,
                    __private const int inputStride2,
                    __private const int inputHeight,
                    __private const int inputWidth,
                    __private const int inputChannel,
                    __global OUTPUT_TYPE *output,
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
    
    vstore4(CONVERT_OUTPUT4(vload4(0, input+inputIndex)), 0, output+outputIndex);
}

__kernel void raster_direct_buffer(
                    GLOBAL_SIZE_3_DIMS
                    __private const int size_x,
                    __global INPUT_TYPE *input,
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
                    __private const int dst_width,
                    __private const int dst_height,
                    __private const int dst_channel
                    ) {
    const int idx = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(idx, y, z);
    const int x = idx % size_x;
    const int id = idx / size_x;
    
    int inputIndex = inputOffset + id * combineSrcOffset + z * inputStride0 + y * inputStride1 + x * inputStride2;
    int outputIndex = outputOffset + id * combineDstOffset + z * outputStride0 + y * outputStride1 + x * outputStride2;
#ifdef INPUT_DATA_FORMAT_NHWC
    int in_c = inputIndex % src_channel; inputIndex /= src_channel;
    int in_w = inputIndex % src_width; inputIndex /= src_width;
    int in_h = inputIndex % src_height;
    int in_b = inputIndex / src_height;
    int src_channel4 = (src_channel + 3) / 4;
    int inputIndexC4 = (((in_b * src_channel4 + (in_c / 4)) * src_height + in_h) * src_width + in_w) * 4 + (in_c % 4);
#else
    int in_w = inputIndex % src_width; inputIndex /= src_width;
    int in_h = inputIndex % src_height; inputIndex /= src_height;
    int in_c = inputIndex % src_channel;
    int in_b = inputIndex / src_channel;
    int src_channel4 = (src_channel + 3) / 4;
    int inputIndexC4 = (((in_b * src_channel4 + (in_c / 4)) * src_height + in_h) * src_width + in_w) * 4 + (in_c % 4);
#endif
    
#ifdef OUTPUT_DATA_FORMAT_NHWC
    int out_c = outputIndex % dst_channel; outputIndex /= dst_channel;
    int out_w = outputIndex % dst_width; outputIndex /= dst_width;
    int out_h = outputIndex % dst_height;
    int out_b = outputIndex / dst_height;
    int dst_channel4 = (dst_channel + 3) / 4;
    int outputIndexC4 = (((out_b * dst_channel4 + (out_c / 4)) * dst_height + out_h) * dst_width + out_w) * 4 + (out_c % 4);
#else
    int out_w = outputIndex % dst_width; outputIndex /= dst_width;
    int out_h = outputIndex % dst_height; outputIndex /= dst_height;
    int out_c = outputIndex % dst_channel;
    int out_b = outputIndex / dst_channel;
    int dst_channel4 = (dst_channel + 3) / 4;
    int outputIndexC4 = (((out_b * dst_channel4 + (out_c / 4)) * dst_height + out_h) * dst_width + out_w) * 4 + (out_c % 4);
#endif
    
    output[outputIndexC4] = (OUTPUT_TYPE)input[inputIndexC4];
}
