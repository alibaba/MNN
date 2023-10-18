#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
inline float4 gelu(float4 in){
    float4 value = 0.79788458f * (0.044715f * in * in * in + in);
    float4 x2 = value * value;
    float4 dst = value > (float4)5.0f ? (float4)1.0f : (value <= -(float4)5.0f ? -(float4)1.0f :
        (value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)))) / (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f))));
    return (1.0f + dst) * in * 0.5f;
}

__kernel void unary_buf(GLOBAL_SIZE_3_DIMS
                        __global const FLOAT *input,
                        __global FLOAT *output,
                        __private const int height) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);

    const int batch_idx = hb / height;
    const int height_idx = hb % height;

    const int offset = (((batch_idx*global_size_dim0+channel_block_idx)*height+height_idx)*global_size_dim1+w) * 4;
#ifdef OPENCL_INPUT_INT
    FLOAT4 in  = CONVERT_FLOAT4(convert_int4(vload4(0, input+offset)));
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 in  = vload4(0, input+offset);
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    vstore4(out, 0, output+offset);
}

