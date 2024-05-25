#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
return;                                                                                   \
}

__kernel void matmul_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                     __global const FLOAT* input_b,
                     #ifdef BIAS
                     __global const FLOAT* input_c,
                     #endif
                     __global FLOAT* output_c, 
                     __private const int channels,
                     __private const int channel_blocks,
                     __private const int width_blocks,
                     __private const int width) {
    const int width_blocks_idx = get_global_id(0);// output W
    const int height_idx       = get_global_id(1);// output H

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    COMPUTE_FLOAT4 a;
    COMPUTE_FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    COMPUTE_FLOAT4 v_zero = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)0.0);

    #ifdef BIAS
    COMPUTE_FLOAT4 temp = CONVERT_COMPUTE_FLOAT4(vload4(width_blocks_idx, input_c));

    COMPUTE_FLOAT result0 = temp.x;
    COMPUTE_FLOAT result1 = temp.y;
    COMPUTE_FLOAT result2 = temp.z;
    COMPUTE_FLOAT result3 = temp.w;
    #else
    COMPUTE_FLOAT result0 = 0;
    COMPUTE_FLOAT result1 = 0;
    COMPUTE_FLOAT result2 = 0;
    COMPUTE_FLOAT result3 = 0;
    #endif

    const int remain = channel_blocks*4 - channels;
    for (short pos = 0; pos < channel_blocks - 1; pos += 1) {
        const int inpa_offset = height_idx * channel_blocks + pos;
        a = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));

        const int inpb_offset = (pos*4) * width_blocks + width_blocks_idx;

        b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        b1 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks, input_b));
        b2 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*2, input_b));
        b3 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*3, input_b));

        COMPUTE_FLOAT4 btmp0 = (COMPUTE_FLOAT4)(b0.s0, b1.s0, b2.s0, b3.s0);
        COMPUTE_FLOAT4 btmp1 = (COMPUTE_FLOAT4)(b0.s1, b1.s1, b2.s1, b3.s1);
        COMPUTE_FLOAT4 btmp2 = (COMPUTE_FLOAT4)(b0.s2, b1.s2, b2.s2, b3.s2);
        COMPUTE_FLOAT4 btmp3 = (COMPUTE_FLOAT4)(b0.s3, b1.s3, b2.s3, b3.s3);

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }
    
    {
        const int inpa_offset = height_idx * channel_blocks + channel_blocks - 1;
        a = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));

        const int inpb_offset = ((channel_blocks - 1)*4) * width_blocks + width_blocks_idx;

        b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        b1 = (remain >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks, input_b));
        b2 = (remain >= 2) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*2, input_b));
        b3 = (remain >= 1) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*3, input_b));
        if (remain == 3) {
            a.y = 0;
            a.z = 0;
            a.w = 0;
        } else if (remain == 2) {
            a.z = 0;
            a.w = 0;
        } else if (remain == 1) {
            a.w = 0;;
        }

        COMPUTE_FLOAT4 btmp0 = (COMPUTE_FLOAT4)(b0.s0, b1.s0, b2.s0, b3.s0);
        COMPUTE_FLOAT4 btmp1 = (COMPUTE_FLOAT4)(b0.s1, b1.s1, b2.s1, b3.s1);
        COMPUTE_FLOAT4 btmp2 = (COMPUTE_FLOAT4)(b0.s2, b1.s2, b2.s2, b3.s2);
        COMPUTE_FLOAT4 btmp3 = (COMPUTE_FLOAT4)(b0.s3, b1.s3, b2.s3, b3.s3);

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }

    const int out_offset = height_idx * width_blocks + width_blocks_idx;
    vstore4(CONVERT_FLOAT4((COMPUTE_FLOAT4)(result0, result1, result2, result3)), out_offset, output_c);
}

__kernel void matmul_transB_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                     __global const FLOAT* input_b,
                    #ifdef BIAS
                     __global const FLOAT* input_c,
                    #endif
                     __global FLOAT* output_c, 
                     __private const int channels,
                     __private const int channel_blocks,
                     __private const int width_blocks,
                     __private const int width) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    COMPUTE_FLOAT4 a;
    COMPUTE_FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    COMPUTE_FLOAT4 v_zero = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)0.0);

    #ifdef BIAS
    COMPUTE_FLOAT4 temp = CONVERT_COMPUTE_FLOAT4(vload4(width_blocks_idx, input_c));
    COMPUTE_FLOAT result0 = temp.x;
    COMPUTE_FLOAT result1 = temp.y;
    COMPUTE_FLOAT result2 = temp.z;
    COMPUTE_FLOAT result3 = temp.w;
    #else
    COMPUTE_FLOAT result0 = 0;
    COMPUTE_FLOAT result1 = 0;
    COMPUTE_FLOAT result2 = 0;
    COMPUTE_FLOAT result3 = 0;
    #endif

    const int remaina = channel_blocks*4 - channels;
    const int remainb = (width_blocks_idx+1)*4 - width;
    for (short pos = 0; pos < channel_blocks - 1; pos += 1) {
        const int inpa_offset = height_idx * channel_blocks + pos;
        a = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));

        const int inpb_offset = (width_blocks_idx*4) * channel_blocks + pos;

        b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        b1 = (remainb >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks, input_b));
        b2 = (remainb >= 2) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks*2, input_b));
        b3 = (remainb >= 1) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks*3, input_b));

        result0 += dot(a, b0);
        result1 += dot(a, b1);
        result2 += dot(a, b2);
        result3 += dot(a, b3);
    }
    
    {
        const int inpa_offset = height_idx * channel_blocks + channel_blocks - 1;
        a = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));

        const int inpb_offset = (width_blocks_idx*4) * channel_blocks + channel_blocks - 1;

        b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        b1 = (remainb >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks, input_b));
        b2 = (remainb >= 2) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks*2, input_b));
        b3 = (remainb >= 1) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks*3, input_b));

        if (remaina == 3) {
            a.y = 0;
            a.z = 0;
            a.w = 0;
        } else if (remaina == 2) {
            a.z = 0;
            a.w = 0;
        } else if (remaina == 1) {
            a.w = 0;
        }

        result0 += dot(a, b0);
        result1 += dot(a, b1);
        result2 += dot(a, b2);
        result3 += dot(a, b3);
    }
    const int out_offset = height_idx * width_blocks + width_blocks_idx;
    vstore4(CONVERT_FLOAT4((COMPUTE_FLOAT4)(result0, result1, result2, result3)), out_offset, output_c);
}


__kernel void matmul_transA_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                 __global const FLOAT* input_b,
                #ifdef BIAS
                 __global const FLOAT* input_c,
                #endif
                 __global FLOAT* output_c,
                 __private const int channels,
                 __private const int channel_blocks,
                 __private const int height,
                 __private const int height_blocks,
                 __private const int width_blocks,
                 __private const int width) {
    const int width_blocks_idx = get_global_id(0);
    const int height_blocks_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_blocks_idx);

    COMPUTE_FLOAT4 v_zero = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)0.0);
    #ifdef BIAS
    COMPUTE_FLOAT4 result0 = CONVERT_COMPUTE_FLOAT4(vload4(width_blocks_idx, input_c));
    COMPUTE_FLOAT4 result1 = result0;
    COMPUTE_FLOAT4 result2 = result0;
    COMPUTE_FLOAT4 result3 = result0;
    #else
    COMPUTE_FLOAT4 result0 = 0;
    COMPUTE_FLOAT4 result1 = 0;
    COMPUTE_FLOAT4 result2 = 0;
    COMPUTE_FLOAT4 result3 = 0;
    #endif
    
    const int remain = channel_blocks*4 - channels;
    for (short pos = 0; pos < channel_blocks - 1; pos += 1) {

        const int inpa_offset = (4*pos) * height_blocks + height_blocks_idx;
        COMPUTE_FLOAT4 a0 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));
        COMPUTE_FLOAT4 a1 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks, input_a));
        COMPUTE_FLOAT4 a2 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks*2, input_a));
        COMPUTE_FLOAT4 a3 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks*3, input_a));

        const int inpb_offset = (4*pos) * width_blocks + width_blocks_idx;
        COMPUTE_FLOAT4 b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        COMPUTE_FLOAT4 b1 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks, input_b));
        COMPUTE_FLOAT4 b2 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*2, input_b));
        COMPUTE_FLOAT4 b3 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*3, input_b));

        COMPUTE_FLOAT4 a0_trans = (COMPUTE_FLOAT4)(a0.x, a1.x, a2.x, a3.x);
        COMPUTE_FLOAT4 a1_trans = (COMPUTE_FLOAT4)(a0.y, a1.y, a2.y, a3.y);
        COMPUTE_FLOAT4 a2_trans = (COMPUTE_FLOAT4)(a0.z, a1.z, a2.z, a3.z);
        COMPUTE_FLOAT4 a3_trans = (COMPUTE_FLOAT4)(a0.w, a1.w, a2.w, a3.w);
        
        COMPUTE_FLOAT4 b0_trans = (COMPUTE_FLOAT4)(b0.x, b1.x, b2.x, b3.x);
        COMPUTE_FLOAT4 b1_trans = (COMPUTE_FLOAT4)(b0.y, b1.y, b2.y, b3.y);
        COMPUTE_FLOAT4 b2_trans = (COMPUTE_FLOAT4)(b0.z, b1.z, b2.z, b3.z);
        COMPUTE_FLOAT4 b3_trans = (COMPUTE_FLOAT4)(b0.w, b1.w, b2.w, b3.w);

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
    
    {
        const int inpa_offset = (4*(channel_blocks - 1)) * height_blocks + height_blocks_idx;
        COMPUTE_FLOAT4 a0 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));
        COMPUTE_FLOAT4 a1 = ((remain >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks, input_a)));
        COMPUTE_FLOAT4 a2 = ((remain >= 2) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks*2, input_a)));
        COMPUTE_FLOAT4 a3 = ((remain >= 1) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks*3, input_a)));

        const int inpb_offset = (4*(channel_blocks - 1)) * width_blocks + width_blocks_idx;
        COMPUTE_FLOAT4 b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        COMPUTE_FLOAT4 b1 = ((remain >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks, input_b)));
        COMPUTE_FLOAT4 b2 = ((remain >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*2, input_b)));
        COMPUTE_FLOAT4 b3 = ((remain >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*3, input_b)));

        COMPUTE_FLOAT4 a0_trans = (COMPUTE_FLOAT4)(a0.x, a1.x, a2.x, a3.x);
        COMPUTE_FLOAT4 a1_trans = (COMPUTE_FLOAT4)(a0.y, a1.y, a2.y, a3.y);
        COMPUTE_FLOAT4 a2_trans = (COMPUTE_FLOAT4)(a0.z, a1.z, a2.z, a3.z);
        COMPUTE_FLOAT4 a3_trans = (COMPUTE_FLOAT4)(a0.w, a1.w, a2.w, a3.w);
        
        COMPUTE_FLOAT4 b0_trans = (COMPUTE_FLOAT4)(b0.x, b1.x, b2.x, b3.x);
        COMPUTE_FLOAT4 b1_trans = (COMPUTE_FLOAT4)(b0.y, b1.y, b2.y, b3.y);
        COMPUTE_FLOAT4 b2_trans = (COMPUTE_FLOAT4)(b0.z, b1.z, b2.z, b3.z);
        COMPUTE_FLOAT4 b3_trans = (COMPUTE_FLOAT4)(b0.w, b1.w, b2.w, b3.w);

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
    
    const int out_offset = (4*height_blocks_idx) * width_blocks + width_blocks_idx;

    vstore4(CONVERT_FLOAT4(result0), out_offset, output_c);
    if(4*height_blocks_idx+1 >= height) return;
    vstore4(CONVERT_FLOAT4(result1), out_offset + width_blocks, output_c);
    if(4*height_blocks_idx+2 >= height) return;
    vstore4(CONVERT_FLOAT4(result2), out_offset + width_blocks*2, output_c);
    if(4*height_blocks_idx+3 >= height) return;
    vstore4(CONVERT_FLOAT4(result3), out_offset + width_blocks*3, output_c);
}

__kernel void matmul_transA_transB_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                     __global const FLOAT* input_b,
                    #ifdef BIAS
                     __global const FLOAT* input_c,
                    #endif
                     __global FLOAT* output_c,
                     __private const int channels,
                     __private const int channel_blocks,
                     __private const int height,
                     __private const int height_blocks,
                     __private const int width_blocks,
                     __private const int width) {
    const int width_blocks_idx = get_global_id(0);
    const int height_blocks_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_blocks_idx);

    COMPUTE_FLOAT4 v_zero = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)0.0);
    #ifdef BIAS
    COMPUTE_FLOAT4 result0 = CONVERT_COMPUTE_FLOAT4(vload4(width_blocks_idx, input_c));

    COMPUTE_FLOAT4 result1 = result0;
    COMPUTE_FLOAT4 result2 = result0;
    COMPUTE_FLOAT4 result3 = result0;
    #else
    COMPUTE_FLOAT4 result0 = 0;
    COMPUTE_FLOAT4 result1 = 0;
    COMPUTE_FLOAT4 result2 = 0;
    COMPUTE_FLOAT4 result3 = 0;
    #endif
    
    const int remaina = channel_blocks * 4 - channels;
    const int remainb = (width_blocks_idx + 1) * 4 - width;
    for (short pos = 0; pos < channel_blocks - 1; pos += 1) {
        const int inpa_offset = (4*pos) * height_blocks + height_blocks_idx;
        COMPUTE_FLOAT4 a0 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));
        COMPUTE_FLOAT4 a1 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks, input_a));
        COMPUTE_FLOAT4 a2 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks*2, input_a));
        COMPUTE_FLOAT4 a3 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks*3, input_a));

        const int inpb_offset = (4*width_blocks_idx) * channel_blocks + pos;
        COMPUTE_FLOAT4 b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        COMPUTE_FLOAT4 b1 = ((remainb >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks, input_b)));
        COMPUTE_FLOAT4 b2 = ((remainb >= 2) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks*2, input_b)));
        COMPUTE_FLOAT4 b3 = ((remainb >= 1) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks*3, input_b)));

        COMPUTE_FLOAT4 a0_trans = (COMPUTE_FLOAT4)(a0.x, a1.x, a2.x, a3.x);
        COMPUTE_FLOAT4 a1_trans = (COMPUTE_FLOAT4)(a0.y, a1.y, a2.y, a3.y);
        COMPUTE_FLOAT4 a2_trans = (COMPUTE_FLOAT4)(a0.z, a1.z, a2.z, a3.z);
        COMPUTE_FLOAT4 a3_trans = (COMPUTE_FLOAT4)(a0.w, a1.w, a2.w, a3.w);

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
    
    {
        const int inpa_offset = (4*(channel_blocks-1)) * height_blocks + height_blocks_idx;
        COMPUTE_FLOAT4 a0 = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));
        COMPUTE_FLOAT4 a1 = ((remaina >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks, input_a)));
        COMPUTE_FLOAT4 a2 = ((remaina >= 2) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks*2, input_a)));
        COMPUTE_FLOAT4 a3 = ((remaina >= 1) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset + height_blocks*3, input_a)));

        const int inpb_offset = (4*width_blocks_idx) * channel_blocks + channel_blocks-1;
        COMPUTE_FLOAT4 b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        COMPUTE_FLOAT4 b1 = ((remainb >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks, input_b)));
        COMPUTE_FLOAT4 b2 = ((remainb >= 2) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks*2, input_b)));
        COMPUTE_FLOAT4 b3 = ((remainb >= 1) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + channel_blocks*3, input_b)));

        COMPUTE_FLOAT4 a0_trans = (COMPUTE_FLOAT4)(a0.x, a1.x, a2.x, a3.x);
        COMPUTE_FLOAT4 a1_trans = (COMPUTE_FLOAT4)(a0.y, a1.y, a2.y, a3.y);
        COMPUTE_FLOAT4 a2_trans = (COMPUTE_FLOAT4)(a0.z, a1.z, a2.z, a3.z);
        COMPUTE_FLOAT4 a3_trans = (COMPUTE_FLOAT4)(a0.w, a1.w, a2.w, a3.w);

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

    const int out_offset = (4*height_blocks_idx) * width_blocks + width_blocks_idx;

    vstore4(CONVERT_FLOAT4(result0), out_offset, output_c);
    if(4*height_blocks_idx+1 >= height) return;
    vstore4(CONVERT_FLOAT4(result1), out_offset + width_blocks, output_c);
    if(4*height_blocks_idx+2 >= height) return;
    vstore4(CONVERT_FLOAT4(result2), out_offset + width_blocks*2, output_c);
    if(4*height_blocks_idx+3 >= height) return;
    vstore4(CONVERT_FLOAT4(result3), out_offset + width_blocks*3, output_c);
}
