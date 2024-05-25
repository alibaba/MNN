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

__kernel void cast(GLOBAL_SIZE_3_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int width,
                            __private const int height,
                            __private const int channelBlock
                            ) {
    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_channel_idx);
    
    const int batch_idx = batch_channel_idx / channelBlock;
    const int channel_idx = batch_channel_idx % channelBlock;
    
#ifdef TO_BOOL
    int4 value = convert_int4(RI_DATA(input, SAMPLER, (int2)(channel_idx * width + width_idx, batch_idx * height + height_idx)));
    value = value == (int4)0 ? (int4)0 : (int4)1;
    WI_DATA(output, (int2)(channel_idx * width + width_idx, batch_idx * height + height_idx), CONVERT_OUTPUT_I4(value));
#else
    INPUT_TYPE_I4 value = RI_DATA(input, SAMPLER, (int2)(channel_idx * width + width_idx, batch_idx * height + height_idx));
    WI_DATA(output, (int2)(channel_idx * width + width_idx, batch_idx * height + height_idx), CONVERT_OUTPUT_I4(value));
#endif
}
