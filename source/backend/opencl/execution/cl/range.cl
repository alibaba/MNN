#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void range(GLOBAL_SIZE_3_DIMS
                            __read_only image2d_t input0,
                            __read_only image2d_t input2,
                            __write_only image2d_t output,
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
                                
    const int bh = batch_idx * height + height_idx;
    const int cw = channel_idx * width + width_idx;
    const int channel4 = channel_idx << 2;
    int index = (((batch_idx * channel) + channel4) * height + height_idx) * width + width_idx;
    int size = height * width;
    int4 index4 = (int4)(index, index + size, index + size * 2, index + size * 3);
    INPUT_TYPE_I start = RI_DATA(input0, SAMPLER, (int2)(0, 0)).x;
    INPUT_TYPE_I step = RI_DATA(input2, SAMPLER, (int2)(0, 0)).x;
    OUTPUT_TYPE_I4 value = (OUTPUT_TYPE_I4)start + CONVERT_OUTPUT_I4(index4) * (OUTPUT_TYPE_I4)step;
    WI_DATA(output, (int2)(cw, bh), value);
}
