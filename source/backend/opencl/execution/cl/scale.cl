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

__kernel void scale(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t scale,
#ifdef HAS_BIAS
                    __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                    __write_only image2d_t output) {

    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int width = global_size_dim1;

    const int pos = mad24(channel_block_idx, width, w);

    FLOAT4 in          = RI_F(input, SAMPLER, (int2)(pos, hb));
    FLOAT4 scale_value = RI_F(scale, SAMPLER, (int2)(channel_block_idx, 0));
#ifdef HAS_BIAS
    FLOAT4 bias_value = RI_F(bias, SAMPLER, (int2)(channel_block_idx, 0));
    FLOAT4 out        = in * scale_value + bias_value;
#else
    FLOAT4 out = in * scale_value;
#endif
    WI_F(output, (int2)(pos, hb), out);
}


