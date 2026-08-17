#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define CLAMP(a) \
    clamp(a, (float4)(-65504.0f), (float4)(65504.0f));
#else
#define CLAMP(a) a
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
// MNN: erfinv via TensorFlow's two-branch polynomial (kept in lockstep with
// CPU's UnaryErfinv in source/backend/cpu/UnaryUtils.hpp). Avoids the OpenCL
// CPU-fallback path which is broken when the runtime is in IMAGE memtype.
inline float4 erfinv4(float4 x){
    float4 w = -log((float4)1.0f - x * x + (float4)1e-12f);
    int4 lt = isless(w, (float4)5.0f);
    float4 w_sel = select(sqrt(w) - (float4)3.0f, w - (float4)2.5f, lt);
    float4 p_lt = (float4)2.81022636e-08f;
    p_lt = (float4)3.43273939e-07f + p_lt * w_sel;
    p_lt = (float4)(-3.5233877e-06f) + p_lt * w_sel;
    p_lt = (float4)(-4.39150654e-06f) + p_lt * w_sel;
    p_lt = (float4)0.00021858087f + p_lt * w_sel;
    p_lt = (float4)(-0.00125372503f) + p_lt * w_sel;
    p_lt = (float4)(-0.00417768164f) + p_lt * w_sel;
    p_lt = (float4)0.246640727f + p_lt * w_sel;
    p_lt = (float4)1.50140941f + p_lt * w_sel;
    float4 p_ge = (float4)(-0.000200214257f);
    p_ge = (float4)0.000100950558f + p_ge * w_sel;
    p_ge = (float4)0.00134934322f + p_ge * w_sel;
    p_ge = (float4)(-0.00367342844f) + p_ge * w_sel;
    p_ge = (float4)0.00573950773f + p_ge * w_sel;
    p_ge = (float4)(-0.0076224613f) + p_ge * w_sel;
    p_ge = (float4)0.00943887047f + p_ge * w_sel;
    p_ge = (float4)1.00167406f + p_ge * w_sel;
    p_ge = (float4)2.83297682f + p_ge * w_sel;
    return select(p_ge, p_lt, lt) * x;
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
        float4 in = (float4)0;
        in.x = (float)input[offset];
        if(remain > 1) { in.y = (float)input[offset + 1]; }
        if(remain > 2) { in.z = (float)input[offset + 2]; }
        if(remain > 3) { in.w = (float)input[offset + 3]; }
        float4 out = OPERATOR;
        output[offset] = (OUTPUT_TYPE)out.x;
        if(remain > 1) { output[offset + 1] = (OUTPUT_TYPE)out.y; }
        if(remain > 2) { output[offset + 2] = (OUTPUT_TYPE)out.z; }
        if(remain > 3) { output[offset + 3] = (OUTPUT_TYPE)out.w; }
    }else {
#endif
        float4 in = convert_float4(vload4(0, input + offset));
        float4 out = CLAMP(OPERATOR);
        vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
#ifdef PACK_LEAVE
    }
#endif
}

