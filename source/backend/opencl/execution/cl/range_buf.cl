#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void range_buf(GLOBAL_SIZE_3_DIMS
                            __global const INPUT_TYPE* input0,
                            __global const INPUT_TYPE* input2,
                            __global OUTPUT_TYPE* output,
                            __private const int width,
                            __private const int height,
                            __private const int channel,
                            __private const int channelBlock
                            ) {
    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_channel_idx);
                                
    const int batch_idx = batch_channel_idx / channelBlock;
    const int channel_idx = batch_channel_idx % channelBlock;
                                
    const int offset = ((((batch_idx * channelBlock) + channel_idx) * height + height_idx) * width + width_idx)*4;
    const int channel4 = channel_idx << 2;
    int index = (((batch_idx * channel) + channel4) * height + height_idx) * width + width_idx;
    int size = height * width;
    int4 index4 = (int4)(index, index + size, index + size * 2, index + size * 3);
    INPUT_TYPE start = input0[0];
    INPUT_TYPE step = input2[0];
    OUTPUT_TYPE4 value = (OUTPUT_TYPE4)start + CONVERT_OUTPUT4(index4) * (OUTPUT_TYPE4)step;
    vstore4(value, 0, output + offset);
}
