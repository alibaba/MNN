#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                                                   \
    }
inline float4 gelu(float4 in){
    float4 value = 0.79788458f * (0.044715f * in * in * in + in);
    float4 x2 = value * value;
    float4 dst = value > (float4)5.0f ? (float4)1.0f : (value <= -(float4)5.0f ? -(float4)1.0f :
        (value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)))) / (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f))));
    return (1.0f + dst) * in * 0.5f;
}

__kernel void unary_buf(GLOBAL_SIZE_2_DIMS
                        __global const INPUT_TYPE *input,
                        __global OUTPUT_TYPE *output,
                        __private const int size) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(x, y);
    const int offset = x << 2;
#ifdef PACK_LEAVE
    if(offset + 3 >= size){
        int remain = size - offset;
        float4 in;
        float* in_ptr = (float*)&in;
        for(int i = 0; i < remain; ++i){
            in_ptr[i] = (float)input[offset + i];
        }
        float4 out = OPERATOR;
        float* out_ptr = (float*)&out;
        for(int i = 0; i < remain; ++i){
            output[offset + i] = (OUTPUT_TYPE)out_ptr[i];
        }
    }else {
#endif
        float4 in = convert_float4(vload4(0, input + offset));
        float4 out = OPERATOR;
        vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
#ifdef PACK_LEAVE
    }
#endif
}

