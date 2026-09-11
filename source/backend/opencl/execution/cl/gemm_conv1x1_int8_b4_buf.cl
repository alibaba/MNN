#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#ifdef USE_IMAGE1D_INPUT
#ifdef MNN_SUPPORT_FP16
#define RI_F(i, coord) read_imageh(i, coord)
#else
#define RI_F(i, coord) read_imagef(i, coord)
#endif
#define LOAD_INPUT4(offset) CONVERT_COMPUTE_FLOAT4(RI_F(input, (offset) >> 2))
#define LOAD_INPUT16(offset) (COMPUTE_FLOAT16)(LOAD_INPUT4(offset), LOAD_INPUT4((offset) + 4), LOAD_INPUT4((offset) + 8), LOAD_INPUT4((offset) + 12))
#else
#define LOAD_INPUT4(offset) CONVERT_COMPUTE_FLOAT4(vload4(0, input + (offset)))
#define LOAD_INPUT16(offset) CONVERT_COMPUTE_FLOAT16(vload16(0, input + (offset)))
#endif
__kernel void gemm_b4_c8_int8_buf(GLOBAL_SIZE_DIM2
#ifdef USE_IMAGE1D_INPUT
                        __read_only image1d_buffer_t input,
#else
                        __global const FLOAT* input,
#endif
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        __global const char *weight,
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int bhw,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int x = get_global_id(0); //b/4
    const int y  = get_global_id(1); //c/8

    UNIFORM_BOUNDRY_CHECK(x, y);
    
    const int out_b_idx = x << 2;
    const int out_c_idx = y << 1;

    COMPUTE_FLOAT8 out0 = CONVERT_COMPUTE_FLOAT8(vload8(0, bias + (out_c_idx << 2)));
    COMPUTE_FLOAT8 out1 = out0;
    COMPUTE_FLOAT8 out2 = out0;
    COMPUTE_FLOAT8 out3 = out0;
    
    const int bhw4 = bhw << 2;
    const int input_offset = out_b_idx * 4;
    int out_offset = out_c_idx * bhw4 + out_b_idx * 4;
#ifndef USE_IMAGE
    const int weight_offset = y * srcChannelAlign * 8;
#endif
    const int loop = (blockDim + 4 - 1) / 4;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif

#if INPUT_BATCH_LEAVES_NUM != 0
    if(out_b_idx + 3 >= bhw){
        for (int i = 0; i < blockNum; i++){
            COMPUTE_FLOAT16 scale, offset;
            {
                #ifdef ASYMMETRIC
                COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + (out_c_idx << 3) + i * dstChannelAlign * 2)) / coef);
                scale = (COMPUTE_FLOAT16)(scaleOffset.s02468ace, scaleOffset.s02468ace);
                offset = (COMPUTE_FLOAT16)(scaleOffset.s13579bdf, scaleOffset.s13579bdf);
                #else
                scale.s01234567 = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
                scale.s89abcdef = scale.s01234567;
                offset = 0;
                #endif
            }
            for (int j = 0; j < loop_end; j++) {
                int k = i * loop + j;
                int k2 = k << 1;
                #ifdef USE_IMAGE
                COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2, y)))) * scale + offset;
                COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 + 1, y)))) * scale + offset;
                #else
                COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(vload16(k2, weight + weight_offset)) * scale + offset;
                COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(vload16(k2 + 1, weight + weight_offset)) * scale + offset;
                #endif
                COMPUTE_FLOAT4 in0 = LOAD_INPUT4(input_offset + k * bhw4);
                #if INPUT_BATCH_LEAVES_NUM >= 2
                COMPUTE_FLOAT4 in1 = LOAD_INPUT4(input_offset + k * bhw4 + 4);
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                COMPUTE_FLOAT4 in2 = LOAD_INPUT4(input_offset + k * bhw4 + 8);
                #endif
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s0, wei0.s01234567, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s0, wei0.s01234567, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s0, wei0.s01234567, out2);
                    #endif
                }
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s1, wei0.s89abcdef, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s1, wei0.s89abcdef, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s1, wei0.s89abcdef, out2);
                    #endif
                }
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s2, wei1.s01234567, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s2, wei1.s01234567, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s2, wei1.s01234567, out2);
                    #endif
                }
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s3, wei1.s89abcdef, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s3, wei1.s89abcdef, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s3, wei1.s89abcdef, out2);
                    #endif
                }
            }
            #if INPUT_CHANNEL_LEAVES_NUM != 0
            {
                int k = i * loop + loop_end;
                int k2 = k << 1;
                COMPUTE_FLOAT4 in0 = LOAD_INPUT4(input_offset + k * bhw4);
                #if INPUT_BATCH_LEAVES_NUM >= 2
                COMPUTE_FLOAT4 in1 = LOAD_INPUT4(input_offset + k * bhw4 + 4);
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                COMPUTE_FLOAT4 in2 = LOAD_INPUT4(input_offset + k * bhw4 + 8);
                #endif
                #ifdef USE_IMAGE
                COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2, y)))) * scale + offset;
                COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 + 1, y)))) * scale + offset;
                #else
                COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(vload16(k2, weight + weight_offset)) * scale + offset;
                COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(vload16(k2 + 1, weight + weight_offset)) * scale + offset;
                #endif
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s0, wei0.s01234567, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s0, wei0.s01234567, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s0, wei0.s01234567, out2);
                    #endif
                }
                #if INPUT_CHANNEL_LEAVES_NUM >= 2
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s1, wei0.s89abcdef, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s1, wei0.s89abcdef, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s1, wei0.s89abcdef, out2);
                    #endif
                }
                #endif
                #if INPUT_CHANNEL_LEAVES_NUM >= 3
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s2, wei1.s01234567, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s2, wei1.s01234567, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s2, wei1.s01234567, out2);
                    #endif
                }
                #endif
            }
            #endif
        }
    } else {
#endif
    for (int i = 0; i < blockNum; i++){
        COMPUTE_FLOAT16 scale, offset;
        {
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + (out_c_idx << 3) + i * dstChannelAlign * 2)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset.s02468ace, scaleOffset.s02468ace);
            offset = (COMPUTE_FLOAT16)(scaleOffset.s13579bdf, scaleOffset.s13579bdf);
            #else
            scale.s01234567 = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
            scale.s89abcdef = scale.s01234567;
            offset = 0;
            #endif
        }
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            int k2 = k << 1;
            COMPUTE_FLOAT16 in = LOAD_INPUT16(input_offset + k * bhw4);
            #ifdef USE_IMAGE
            COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2, y)))) * scale + offset;
            COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 + 1, y)))) * scale + offset;
            #else
            COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(vload16(k2, weight + weight_offset)) * scale + offset;
            COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(vload16(k2 + 1, weight + weight_offset)) * scale + offset;
            #endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei0.s01234567, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s4, wei0.s01234567, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s8, wei0.s01234567, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sc, wei0.s01234567, out3);
            }
            {
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei0.s89abcdef, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s5, wei0.s89abcdef, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s9, wei0.s89abcdef, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sd, wei0.s89abcdef, out3);
            }
            {
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei1.s01234567, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s6, wei1.s01234567, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sa, wei1.s01234567, out2);
                out3 = mad((COMPUTE_FLOAT8)in.se, wei1.s01234567, out3);
            }
            {
                out0 = mad((COMPUTE_FLOAT8)in.s3, wei1.s89abcdef, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s7, wei1.s89abcdef, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sb, wei1.s89abcdef, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sf, wei1.s89abcdef, out3);
            }
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            int k2 = k << 1;
            COMPUTE_FLOAT16 in = LOAD_INPUT16(input_offset + k * bhw4);
            #ifdef USE_IMAGE
            COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2, y)))) * scale + offset;
            COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 + 1, y)))) * scale + offset;
            #else
            COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(vload16(k2, weight + weight_offset)) * scale + offset;
            COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(vload16(k2 + 1, weight + weight_offset)) * scale + offset;
            #endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei0.s01234567, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s4, wei0.s01234567, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s8, wei0.s01234567, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sc, wei0.s01234567, out3);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei0.s89abcdef, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s5, wei0.s89abcdef, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s9, wei0.s89abcdef, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sd, wei0.s89abcdef, out3);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei1.s01234567, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s6, wei1.s01234567, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sa, wei1.s01234567, out2);
                out3 = mad((COMPUTE_FLOAT8)in.se, wei1.s01234567, out3);
            }
            #endif
        }
        #endif
    }
#if INPUT_BATCH_LEAVES_NUM != 0
    }
#endif
    
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT8)0);
    out1 = fmax(out1, (COMPUTE_FLOAT8)0);
    out2 = fmax(out2, (COMPUTE_FLOAT8)0);
    out3 = fmax(out3, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out1 = clamp(out1, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out2 = clamp(out2, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out3 = clamp(out3, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif

#if INPUT_BATCH_LEAVES_NUM != 0
    if(out_b_idx + 3 >= bhw){
        #if INPUT_BATCH_LEAVES_NUM == 3
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s0123, out1.s0123)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2.s0123), 0, output+out_offset+8);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s4567, out1.s4567)), 0, output+out_offset+bhw4);
            vstore4(CONVERT_FLOAT4(out2.s4567), 0, output+out_offset+bhw4+8);
        }
        #elif INPUT_BATCH_LEAVES_NUM == 2
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s0123, out1.s0123)), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s4567, out1.s4567)), 0, output+out_offset+bhw4);
        }
        #elif INPUT_BATCH_LEAVES_NUM == 1
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output+out_offset+bhw4);
        }
        #endif
    }else{
#endif
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s0123, out1.s0123, out2.s0123, out3.s0123)), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s4567, out1.s4567, out2.s4567, out3.s4567)), 0, output+out_offset+bhw4);
        }
#if INPUT_BATCH_LEAVES_NUM != 0
    }
#endif
}