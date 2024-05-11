#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

__kernel void scale_buf(GLOBAL_SIZE_2_DIMS
                        __global const FLOAT* input,
                        __global const FLOAT* scale,
#ifdef BIAS
                        __global const FLOAT* bias,
#endif
                        __global FLOAT* output,
                        __private const int4 shape) {//N, H, W, C4

    const int out_w_c_idx = get_global_id(0);
    const int out_h_b_idx = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(out_w_c_idx, out_h_b_idx);

    const int out_b_idx = out_h_b_idx / shape.y;
    const int out_h_idx = out_h_b_idx % shape.y;
    const int out_c_idx = out_w_c_idx / shape.z;
    const int out_w_idx = out_w_c_idx % shape.z;
    
    const int offset = (((out_b_idx * shape.w + out_c_idx) * shape.y + out_h_idx) * shape.z + out_w_idx) * 4;
    COMPUTE_FLOAT4 in_value    = CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset));
    COMPUTE_FLOAT4 scale_value = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, scale));
#ifdef BIAS
    COMPUTE_FLOAT4 bias_value = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out_value  = in_value * scale_value + bias_value;
#else
    COMPUTE_FLOAT4 out_value  = in_value * scale_value;
#endif
    vstore4(CONVERT_FLOAT4(out_value), 0, output+offset);
}
