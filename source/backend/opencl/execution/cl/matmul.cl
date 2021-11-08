#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
return;                                                                                   \
}

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void matmul(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                     __read_only image2d_t input_b,
                    #ifdef BIAS
                     __read_only image2d_t input_c,
                    #endif
                     __write_only image2d_t output_c, __private const int channels,
                     __private const int channel_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOAT4 a;
    FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    #ifdef BIAS
    FLOAT4 temp = RI_F(input_c, SAMPLER, (int2)(width_blocks_idx, 0));
    FLOAT result0 = temp.x;
    FLOAT result1 = temp.y;
    FLOAT result2 = temp.z;
    FLOAT result3 = temp.w;
    #else
    FLOAT result0 = 0;
    FLOAT result1 = 0;
    FLOAT result2 = 0;
    FLOAT result3 = 0;
    #endif

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        a = RI_F(input_a, SAMPLER, (int2)(pos, height_idx));

        short remain = (pos + 1) * 4 - channels;

        b0 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4));
        b1 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 1));
        b2 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 2));
        b3 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 3));

        if (remain == 3) {
            b1 = 0;
            b2 = 0;
            b3 = 0;
        } else if (remain == 2) {
            b2 = 0;
            b3 = 0;
        } else if (remain == 1) {
            b3 = 0;
        }

        FLOAT4 btmp0 = (FLOAT4)(b0.s0, b1.s0, b2.s0, b3.s0);
        FLOAT4 btmp1 = (FLOAT4)(b0.s1, b1.s1, b2.s1, b3.s1);
        FLOAT4 btmp2 = (FLOAT4)(b0.s2, b1.s2, b2.s2, b3.s2);
        FLOAT4 btmp3 = (FLOAT4)(b0.s3, b1.s3, b2.s3, b3.s3);

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }
    WI_F(output_c, (int2)(width_blocks_idx, height_idx), (FLOAT4)(result0, result1, result2, result3));
}

__kernel void matmul_transB(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                     __read_only image2d_t input_b,
                    #ifdef BIAS
                     __read_only image2d_t input_c,
                    #endif
                     __write_only image2d_t output_c, __private const int channels,
                     __private const int channel_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOAT4 a;
    FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    #ifdef BIAS
    FLOAT4 temp = RI_F(input_c, SAMPLER, (int2)(width_blocks_idx, 0));
    FLOAT result0 = temp.x;
    FLOAT result1 = temp.y;
    FLOAT result2 = temp.z;
    FLOAT result3 = temp.w;
    #else
    FLOAT result0 = 0;
    FLOAT result1 = 0;
    FLOAT result2 = 0;
    FLOAT result3 = 0;
    #endif

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        a = RI_F(input_a, SAMPLER, (int2)(pos, height_idx));

        short remain = (pos + 1) * 4 - channels;

        b0 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4));
        b1 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 1));
        b2 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 2));
        b3 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 3));
        if (remain == 3) {
            a.y = 0;
            a.z = 0;
            a.w = 0;
        } else if (remain == 2) {
            a.z = 0;
            a.w = 0;
        } else if (remain == 1) {
            a.w = 0;
        }

        result0 += dot(a, b0);
        result1 += dot(a, b1);
        result2 += dot(a, b2);
        result3 += dot(a, b3);
    }
    WI_F(output_c, (int2)(width_blocks_idx, height_idx), (FLOAT4)(result0, result1, result2, result3));
}

    __kernel void matmul_transA(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                     __read_only image2d_t input_b,
                    #ifdef BIAS
                     __read_only image2d_t input_c,
                    #endif
                     __write_only image2d_t output_c,
                     __private const int channels,
                     __private const int channel_blocks,
                      __private const int height) {
    const int width_blocks_idx = get_global_id(0);
    const int height_blocks_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_blocks_idx);

    FLOAT4 v_zero = (FLOAT4)((FLOAT)0.0);
    #ifdef BIAS
    FLOAT4 result0 = RI_F(input_c, SAMPLER, (int2)(width_blocks_idx, 0));
    FLOAT4 result1 = result0;
    FLOAT4 result2 = result0;
    FLOAT4 result3 = result0;
    #else
    FLOAT4 result0 = 0;
    FLOAT4 result1 = 0;
    FLOAT4 result2 = 0;
    FLOAT4 result3 = 0;
    #endif
    
    for (short pos = 0; pos < channel_blocks; pos += 1) {
        FLOAT4 a0 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos));
        FLOAT4 a1 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+1));
        FLOAT4 a2 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+2));
        FLOAT4 a3 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+3));

        FLOAT4 b0 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, 4*pos));
        FLOAT4 b1 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, 4*pos+1));
        FLOAT4 b2 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, 4*pos+2));
        FLOAT4 b3 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, 4*pos+3));
        
        short remain = (pos + 1) * 4 - channels;
        a3 = ((remain >= 1) ? v_zero : a3);
        a2 = ((remain >= 2) ? v_zero : a2);
        a1 = ((remain >= 3) ? v_zero : a1);

        FLOAT4 a0_trans = (FLOAT4)(a0.x, a1.x, a2.x, a3.x);
        FLOAT4 a1_trans = (FLOAT4)(a0.y, a1.y, a2.y, a3.y);
        FLOAT4 a2_trans = (FLOAT4)(a0.z, a1.z, a2.z, a3.z);
        FLOAT4 a3_trans = (FLOAT4)(a0.w, a1.w, a2.w, a3.w);
        
        FLOAT4 b0_trans = (FLOAT4)(b0.x, b1.x, b2.x, b3.x);
        FLOAT4 b1_trans = (FLOAT4)(b0.y, b1.y, b2.y, b3.y);
        FLOAT4 b2_trans = (FLOAT4)(b0.z, b1.z, b2.z, b3.z);
        FLOAT4 b3_trans = (FLOAT4)(b0.w, b1.w, b2.w, b3.w);

        //matmul
        result0.x += dot(a0_trans, b0_trans);
        result0.y += dot(a0_trans, b1_trans);
        result0.z += dot(a0_trans, b2_trans);
        result0.w += dot(a0_trans, b3_trans);
        
        result1.x += dot(a1_trans, b0_trans);
        result1.y += dot(a1_trans, b1_trans);
        result1.z += dot(a1_trans, b2_trans);
        result1.w += dot(a1_trans, b3_trans);
        
        result2.x += dot(a2_trans, b0_trans);
        result2.y += dot(a2_trans, b1_trans);
        result2.z += dot(a2_trans, b2_trans);
        result2.w += dot(a2_trans, b3_trans);
        
        result3.x += dot(a3_trans, b0_trans);
        result3.y += dot(a3_trans, b1_trans);
        result3.z += dot(a3_trans, b2_trans);
        result3.w += dot(a3_trans, b3_trans);
    }
    WI_F(output_c, (int2)(width_blocks_idx, 4*height_blocks_idx), result0);
    if(4*height_blocks_idx+1 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, 4*height_blocks_idx+1), result1);
    if(4*height_blocks_idx+2 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, 4*height_blocks_idx+2), result2);
    if(4*height_blocks_idx+3 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, 4*height_blocks_idx+3), result3);

}

__kernel void matmul_transA_transB(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                     __read_only image2d_t input_b,
                    #ifdef BIAS
                     __read_only image2d_t input_c,
                    #endif
                     __write_only image2d_t output_c,
                     __private const int channels,
                     __private const int channel_blocks,
                      __private const int height) {
    const int width_blocks_idx = get_global_id(0);
    const int height_blocks_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_blocks_idx);

    FLOAT4 v_zero = (FLOAT4)((FLOAT)0.0);
    #ifdef BIAS
    FLOAT4 result0 = RI_F(input_c, SAMPLER, (int2)(width_blocks_idx, 0));
    FLOAT4 result1 = result0;
    FLOAT4 result2 = result0;
    FLOAT4 result3 = result0;
    #else
    FLOAT4 result0 = 0;
    FLOAT4 result1 = 0;
    FLOAT4 result2 = 0;
    FLOAT4 result3 = 0;
    #endif

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        FLOAT4 a0 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos));
        FLOAT4 a1 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+1));
        FLOAT4 a2 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+2));
        FLOAT4 a3 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+3));

        FLOAT4 b0 = RI_F(input_b, SAMPLER, (int2)(pos, 4*width_blocks_idx));
        FLOAT4 b1 = RI_F(input_b, SAMPLER, (int2)(pos, 4*width_blocks_idx+1));
        FLOAT4 b2 = RI_F(input_b, SAMPLER, (int2)(pos, 4*width_blocks_idx+2));
        FLOAT4 b3 = RI_F(input_b, SAMPLER, (int2)(pos, 4*width_blocks_idx+3));
        
        short remain = (pos + 1) * 4 - channels;
        a3 = ((remain >= 1) ? v_zero : a3);
        a2 = ((remain >= 2) ? v_zero : a2);
        a1 = ((remain >= 3) ? v_zero : a1);

        FLOAT4 a0_trans = (FLOAT4)(a0.x, a1.x, a2.x, a3.x);
        FLOAT4 a1_trans = (FLOAT4)(a0.y, a1.y, a2.y, a3.y);
        FLOAT4 a2_trans = (FLOAT4)(a0.z, a1.z, a2.z, a3.z);
        FLOAT4 a3_trans = (FLOAT4)(a0.w, a1.w, a2.w, a3.w);

        //matmul
        result0.x += dot(a0_trans, b0);
        result0.y += dot(a0_trans, b1);
        result0.z += dot(a0_trans, b2);
        result0.w += dot(a0_trans, b3);
        
        result1.x += dot(a1_trans, b0);
        result1.y += dot(a1_trans, b1);
        result1.z += dot(a1_trans, b2);
        result1.w += dot(a1_trans, b3);
        
        result2.x += dot(a2_trans, b0);
        result2.y += dot(a2_trans, b1);
        result2.z += dot(a2_trans, b2);
        result2.w += dot(a2_trans, b3);
        
        result3.x += dot(a3_trans, b0);
        result3.y += dot(a3_trans, b1);
        result3.z += dot(a3_trans, b2);
        result3.w += dot(a3_trans, b3);
    }

    WI_F(output_c, (int2)(width_blocks_idx, 4*height_blocks_idx), result0);
    if(4*height_blocks_idx+1 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, 4*height_blocks_idx+1), result1);
    if(4*height_blocks_idx+2 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, 4*height_blocks_idx+2), result2);
    if(4*height_blocks_idx+3 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, 4*height_blocks_idx+3), result3);
}
