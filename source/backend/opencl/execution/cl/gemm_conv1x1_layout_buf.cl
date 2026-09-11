#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }
__kernel void gemm_c4nhw4_to_nhwc(GLOBAL_SIZE_DIM2
__global const FLOAT* input,
__global FLOAT* output,
__private const int bhw,
__private const int channel,
__private const int channelAlign
){
    const int x = get_global_id(0); //b/4
    const int y = get_global_id(1); //c/4
    UNIFORM_BOUNDRY_CHECK(x, y);
    const int out_b_idx = x << 2;
    const int out_c_idx = y << 2;
    const int bhw4 = bhw << 2;
    const int input_offset = y * bhw4 + out_b_idx * 4;
    FLOAT4 in0, in1, in2, in3;
    if(out_c_idx + 3 < channel && out_b_idx + 3 < bhw){
        in0 = vload4(0, input + input_offset);
        in1 = vload4(0, input + input_offset + 4);
        in2 = vload4(0, input + input_offset + 8);
        in3 = vload4(0, input + input_offset + 12);
    } else{
        if(out_c_idx + 3 < channel){
            in0 = vload4(0, input + input_offset);
            in1 = out_b_idx + 1 < bhw ? vload4(0, input + input_offset + 4) : 0;
            in2 = out_b_idx + 2 < bhw ? vload4(0, input + input_offset + 8) : 0;
            in3 = out_b_idx + 3 < bhw ? vload4(0, input + input_offset + 12) : 0;
        } else if(out_c_idx + 1 == channel){
            in0 = (FLOAT4)(input[input_offset], 0, 0, 0);
            in1 = out_b_idx + 1 < bhw ? (FLOAT4)(input[input_offset + 4], 0, 0, 0) : 0;
            in2 = out_b_idx + 2 < bhw ? (FLOAT4)(input[input_offset + 8], 0, 0, 0) : 0;
            in3 = out_b_idx + 3 < bhw ? (FLOAT4)(input[input_offset + 12], 0, 0, 0) : 0;
        } else if(out_c_idx + 2 == channel){
            in0 = (FLOAT4)(input[input_offset], input[input_offset + 1], 0, 0);
            in1 = out_b_idx + 1 < bhw ? (FLOAT4)(input[input_offset + 4], input[input_offset + 5], 0, 0) : 0;
            in2 = out_b_idx + 2 < bhw ? (FLOAT4)(input[input_offset + 8], input[input_offset + 9], 0, 0) : 0;
            in3 = out_b_idx + 3 < bhw ? (FLOAT4)(input[input_offset + 12], input[input_offset + 13], 0, 0) : 0;
        } else if(out_c_idx + 3 == channel){
            in0 = (FLOAT4)(input[input_offset], input[input_offset + 1], input[input_offset + 2], 0);
            in1 = out_b_idx + 1 < bhw ? (FLOAT4)(input[input_offset + 4], input[input_offset + 5], input[input_offset + 6], 0) : 0;
            in2 = out_b_idx + 2 < bhw ? (FLOAT4)(input[input_offset + 8], input[input_offset + 9], input[input_offset + 10], 0) : 0;
            in3 = out_b_idx + 3 < bhw ? (FLOAT4)(input[input_offset + 12], input[input_offset + 13], input[input_offset + 14], 0) : 0;
        }
    }
    int out_offset = out_b_idx * channelAlign + out_c_idx;
    vstore4(in0, 0, output + out_offset);
    vstore4(in1, 0, output + out_offset + channelAlign);
    vstore4(in2, 0, output + out_offset + channelAlign + channelAlign);
    vstore4(in3, 0, output + out_offset + channelAlign + channelAlign + channelAlign);
}

__kernel void gemm_nhwc_to_c4nhw4(GLOBAL_SIZE_DIM2
__global const FLOAT* input,
__global FLOAT* output,
__private const int bhw,
__private const int channelAlign
){
    const int x = get_global_id(0); //b/4
    const int y = get_global_id(1); //c/4
    UNIFORM_BOUNDRY_CHECK(x, y);
    const int out_b_idx = x << 2;
    const int out_c_idx = y << 2;
    const int bhw4 = bhw << 2;
    const int input_offset = out_b_idx * channelAlign + out_c_idx;
    FLOAT4 in0 = vload4(0, input + input_offset);
    FLOAT4 in1 = vload4(0, input + input_offset + channelAlign);
    FLOAT4 in2 = vload4(0, input + input_offset + channelAlign + channelAlign);
    FLOAT4 in3 = vload4(0, input + input_offset + channelAlign + channelAlign + channelAlign);
    int out_offset = y * bhw4 + out_b_idx * 4;
    vstore4(in0, 0, output + out_offset);
    if(out_b_idx + 1 >= bhw) return;
    vstore4(in1, 0, output + out_offset + 4);
    if(out_b_idx + 2 >= bhw) return;
    vstore4(in2, 0, output + out_offset + 8);
    if(out_b_idx + 3 >= bhw) return;
    vstore4(in3, 0, output + out_offset + 12);
}
