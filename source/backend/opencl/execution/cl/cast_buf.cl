#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void cast_buf(GLOBAL_SIZE_3_DIMS
                            __global INPUT_TYPE* input,
                            __global OUTPUT_TYPE* output,
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
    
    const int inp_offset = ((((batch_idx * channelBlock) + channel_idx) * height + height_idx) * width + width_idx)*4;
#ifdef TO_BOOL
    int4 value = convert_int4(vload4(0, input + inp_offset));
    value = value == (int4)0 ? (int4)0 : (int4)1;
    vstore4(CONVERT_OUTPUT4(value), 0, output + inp_offset);
#else
    vstore4(CONVERT_OUTPUT4(vload4(0, input + inp_offset)), 0, output + inp_offset);
#endif
}
