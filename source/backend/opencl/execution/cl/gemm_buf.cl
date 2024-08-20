#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

__kernel void gemm_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output,
                        __private const int width,//UP_DIV(wUnit*hUnit,4)
                        __private const int height,//dstChannelC4
                        __private const int srcChannelC4,
                        __private const int alpha2) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int pos_x = pos.x % width;
    const int pos_y = pos.x / width;
    const int pos_z = pos.y;

    COMPUTE_FLOAT16 o = (COMPUTE_FLOAT16)0;
    
    int kenerlY   = mad24(pos_z, height, pos_y);

    for (int k = 0; k < srcChannelC4; ++k) {        
        //NHWC  [1, 1, alpha2*height, srcChannelC4*4] x 4
        //index:[0, 0, pos_z*width+pos_y,    index+0]
        //int inp1_offset = (((k * (alpha2*height) + kenerlY) * (srcChannelC4*4) + index)*4 + 0)*4;
        
        COMPUTE_FLOAT16 k_v16 = CONVERT_COMPUTE_FLOAT16(vload16(kenerlY*(srcChannelC4) + k, input1));
        
        //NC4HW4 [alpha*alpha, srcChannelC4, width, 4] x 4
        //index: [pos_z,       k,            pos_x, 0]
        
        COMPUTE_FLOAT16 s = CONVERT_COMPUTE_FLOAT16(vload16(((pos_z*srcChannelC4 + k) * width + pos_x), input0));

        o = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s0, (COMPUTE_FLOAT4)s.s4, (COMPUTE_FLOAT4)s.s8, (COMPUTE_FLOAT4)s.sc), (COMPUTE_FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o);
        o = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s1, (COMPUTE_FLOAT4)s.s5, (COMPUTE_FLOAT4)s.s9, (COMPUTE_FLOAT4)s.sd), (COMPUTE_FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o);
        o = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s2, (COMPUTE_FLOAT4)s.s6, (COMPUTE_FLOAT4)s.sa, (COMPUTE_FLOAT4)s.se), (COMPUTE_FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o);
        o = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s3, (COMPUTE_FLOAT4)s.s7, (COMPUTE_FLOAT4)s.sb, (COMPUTE_FLOAT4)s.sf), (COMPUTE_FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o);
    }
    
    //index: [pos_y,  pos_z,  0, pos_x]
    int out_offset = (((pos_y * alpha2 + pos_z) * 4 + 0) * width + pos_x) * 4;

    vstore4(CONVERT_FLOAT4(o.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(o.s4567), 0, output+out_offset+4*width);
    vstore4(CONVERT_FLOAT4(o.s89ab), 0, output+out_offset+8*width);
    vstore4(CONVERT_FLOAT4(o.scdef), 0, output+out_offset+12*width);
}



__kernel void gemm_buf2(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output,
                        __private const int width,//UP_DIV(wUnit*hUnit,8)
                        __private const int height,//dstChannelC4
                        __private const int srcChannelC4,
                        __private const int alpha2) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int width_block = (width+1) >> 1;
    const int pos_x = (pos.x % width_block) << 1;
    const int pos_y = pos.x / width_block;
    const int pos_z = pos.y;

    COMPUTE_FLOAT16 o0 = (COMPUTE_FLOAT16)0;
    COMPUTE_FLOAT16 o1 = (COMPUTE_FLOAT16)0;

    const int kenerlY   = mad24(pos_z, height, pos_y);
    const int kernel_base = mul24(kenerlY, srcChannelC4);
    const int inp_base = (pos_z*srcChannelC4 + 0) * width + pos_x;
    
    for (int k = 0; k < srcChannelC4; ++k) {
        //NHWC  [1, 1, alpha2*height, srcChannelC4*4] x 4
        //index:[0, 0, pos_z*width+pos_y,    index+0]
        //int inp1_offset = (((k * (alpha2*height) + kenerlY) * (srcChannelC4*4) + index)*4 + 0)*4;
        
        COMPUTE_FLOAT16 k_v16 = CONVERT_COMPUTE_FLOAT16(vload16(kernel_base + k, input1));
        
        //NC4HW4 [alpha*alpha, srcChannelC4, width, 4] x 4
        //index: [pos_z,       k,            pos_x, 0]
        
        const int inp_offset = mad24(k, width, inp_base);
        COMPUTE_FLOAT16 s = CONVERT_COMPUTE_FLOAT16(vload16(inp_offset, input0));

        o0 = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s0, (COMPUTE_FLOAT4)s.s4, (COMPUTE_FLOAT4)s.s8, (COMPUTE_FLOAT4)s.sc), (COMPUTE_FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o0);
        o0 = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s1, (COMPUTE_FLOAT4)s.s5, (COMPUTE_FLOAT4)s.s9, (COMPUTE_FLOAT4)s.sd), (COMPUTE_FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o0);
        o0 = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s2, (COMPUTE_FLOAT4)s.s6, (COMPUTE_FLOAT4)s.sa, (COMPUTE_FLOAT4)s.se), (COMPUTE_FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o0);
        o0 = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s3, (COMPUTE_FLOAT4)s.s7, (COMPUTE_FLOAT4)s.sb, (COMPUTE_FLOAT4)s.sf), (COMPUTE_FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o0);
        
        s = CONVERT_COMPUTE_FLOAT16(vload16(inp_offset + 1, input0));
        o1 = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s0, (COMPUTE_FLOAT4)s.s4, (COMPUTE_FLOAT4)s.s8, (COMPUTE_FLOAT4)s.sc), (COMPUTE_FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o1);
        o1 = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s1, (COMPUTE_FLOAT4)s.s5, (COMPUTE_FLOAT4)s.s9, (COMPUTE_FLOAT4)s.sd), (COMPUTE_FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o1);
        o1 = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s2, (COMPUTE_FLOAT4)s.s6, (COMPUTE_FLOAT4)s.sa, (COMPUTE_FLOAT4)s.se), (COMPUTE_FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o1);
        o1 = mad((COMPUTE_FLOAT16)((COMPUTE_FLOAT4)s.s3, (COMPUTE_FLOAT4)s.s7, (COMPUTE_FLOAT4)s.sb, (COMPUTE_FLOAT4)s.sf), (COMPUTE_FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o1);
    }

    //index: [pos_y,  pos_z,  0, pos_x]
    int out_offset = (((pos_y * alpha2 + pos_z) * 4 + 0) * width + pos_x) * 4;

    vstore4(CONVERT_FLOAT4(o0.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(o0.s4567), 0, output+out_offset+4*width);
    vstore4(CONVERT_FLOAT4(o0.s89ab), 0, output+out_offset+8*width);
    vstore4(CONVERT_FLOAT4(o0.scdef), 0, output+out_offset+12*width);
    
    if(pos_x + 1 >= width) return;
    vstore4(CONVERT_FLOAT4(o1.s0123), 1, output+out_offset);
    vstore4(CONVERT_FLOAT4(o1.s4567), 1, output+out_offset+4*width);
    vstore4(CONVERT_FLOAT4(o1.s89ab), 1, output+out_offset+8*width);
    vstore4(CONVERT_FLOAT4(o1.scdef), 1, output+out_offset+12*width);
}

// [B, K/4, area, 4] -> [alignK, alignM] (M = B * area)
__kernel void transpose_pad(GLOBAL_SIZE_DIM2
                        const int alignM,
                        const int alignK,
                        const int M,
                        const int K,
                        const int area,
                        __global const FLOAT* input,
                        __global FLOAT* output
                        ) {
#ifdef AREA_EQUAL_1
    const int idx_m4 = get_global_id(0); // idx M
    const int idx_k4 = get_global_id(1); // idx K
    UNIFORM_BOUNDRY_CHECK(idx_m4, idx_k4);

    const int idx_m = idx_m4 << 2;
    const int idx_k = idx_k4 << 2;
    const int K_4 = (K + 3) >> 2;
    const int in_offset_base  = (idx_m * K_4 + idx_k4) * 4;
    const int out_offset_base = idx_k * alignM + idx_m;
    
    FLOAT4 m0k4 = (idx_k4 >= K_4 || idx_m + 0 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base);
    FLOAT4 m1k4 = (idx_k4 >= K_4 || idx_m + 1 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + (K_4 << 2));
    FLOAT4 m2k4 = (idx_k4 >= K_4 || idx_m + 2 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + (K_4 << 2) * 2);
    FLOAT4 m3k4 = (idx_k4 >= K_4 || idx_m + 3 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + (K_4 << 2) * 3);
    
    vstore4((FLOAT4)(m0k4.x, m1k4.x, m2k4.x, m3k4.x), 0, output + out_offset_base);
    vstore4((FLOAT4)(m0k4.y, m1k4.y, m2k4.y, m3k4.y), 0, output + out_offset_base + alignM);
    vstore4((FLOAT4)(m0k4.z, m1k4.z, m2k4.z, m3k4.z), 0, output + out_offset_base + alignM + alignM);
    vstore4((FLOAT4)(m0k4.w, m1k4.w, m2k4.w, m3k4.w), 0, output + out_offset_base + alignM + alignM + alignM);
#elif defined BATCH_EQUAL_1

    const int idx_m4 = get_global_id(0); // idx M
    const int idx_k4 = get_global_id(1); // idx K
    UNIFORM_BOUNDRY_CHECK(idx_m4, idx_k4);

    const int idx_m = idx_m4 << 2;
    const int idx_k = idx_k4 << 2;
    const int K_4 = (K + 3) >> 2;
    const int in_offset_base  = (idx_k4 * area + idx_m) * 4;
    const int out_offset_base = idx_k * alignM + idx_m;

    FLOAT4 m0k4 = (idx_k4 >= K_4 || idx_m + 0 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base);
    FLOAT4 m1k4 = (idx_k4 >= K_4 || idx_m + 1 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 4);
    FLOAT4 m2k4 = (idx_k4 >= K_4 || idx_m + 2 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 8);
    FLOAT4 m3k4 = (idx_k4 >= K_4 || idx_m + 3 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 12);

    vstore4((FLOAT4)(m0k4.x, m1k4.x, m2k4.x, m3k4.x), 0, output + out_offset_base);
    vstore4((FLOAT4)(m0k4.y, m1k4.y, m2k4.y, m3k4.y), 0, output + out_offset_base + alignM);
    vstore4((FLOAT4)(m0k4.z, m1k4.z, m2k4.z, m3k4.z), 0, output + out_offset_base + alignM + alignM);
    vstore4((FLOAT4)(m0k4.w, m1k4.w, m2k4.w, m3k4.w), 0, output + out_offset_base + alignM + alignM + alignM);

#else

    const int idx_m = get_global_id(0); // idx M
    const int idx_k4 = get_global_id(1); // idx K
    UNIFORM_BOUNDRY_CHECK(idx_m, idx_k4);
    
    const int K_4 = (K + 3) >> 2;
    const int idx_k = idx_k4 << 2;
    const int out_offset_base = idx_k * alignM + idx_m;
    
    if(idx_k4 >= K_4 || idx_m >= M) {
        output[out_offset_base] = (FLOAT)0;
        output[out_offset_base + alignM] = (FLOAT)0;
        output[out_offset_base + alignM + alignM] = (FLOAT)0;
        output[out_offset_base + alignM + alignM + alignM] = (FLOAT)0;
        return;
    }
    const int idx_b = idx_m / area;
    const int idx_area = idx_m % area;
    
    const int in_offset_base  = ((idx_b * K_4 + idx_k4) * area + idx_area) * 4;
    FLOAT4 data = vload4(0, input + in_offset_base);
    
    output[out_offset_base] = data.x;
    output[out_offset_base + alignM] = data.y;
    output[out_offset_base + alignM + alignM] = data.z;
    output[out_offset_base + alignM + alignM + alignM] = data.w;
#endif
}

// [alignM, alignN] -> [B, N/4, area, 4] (M = B * area)
__kernel void transpose_bias(GLOBAL_SIZE_DIM2
                        const int alignM,
                        const int alignN,
                        const int M,
                        const int N,
                        const int area,
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output
                        ) {
#ifdef AREA_EQUAL_1
    const int idx_m = get_global_id(0); // idx M
    const int idx_n_16 = get_global_id(1); // idx N
    UNIFORM_BOUNDRY_CHECK(idx_m, idx_n_16);

    const int N_4 = (N + 3) >> 2;
    const int N_16 = (N + 15) >> 4;
    const int N_left = N & 15;
    bool canVec16 = (N_left == 0 || (N_left != 0 && idx_n_16 < N_16 - 1));
    if(canVec16) {
        FLOAT16 res0 = vload16(0, input0 + idx_m * alignN + (idx_n_16 << 4));
        FLOAT16 res1 = vload16(0, input1 + (idx_n_16 << 4));
        FLOAT16 res = res0 + res1;
        #ifdef RELU
            res = fmax(res, (FLOAT16)0);
        #endif
        #ifdef RELU6
            res = clamp(res, (FLOAT16)0, (FLOAT16)6);
        #endif
        vstore16(res, 0, output + ((idx_m * N_4 + (idx_n_16 << 2)) << 2));
    } else {

        FLOAT4 res0 = vload4(0, input0 + idx_m * alignN + (idx_n_16 << 4));
        FLOAT4 res1 = vload4(0, input1 + (idx_n_16 << 4));
        FLOAT4 res = res0 + res1;
        #ifdef RELU
            res = fmax(res, (FLOAT4)0);
        #endif
        #ifdef RELU6
            res = clamp(res, (FLOAT4)0, (FLOAT4)6);
        #endif
        vstore4(res, 0, output + ((idx_m * N_4 + (idx_n_16 << 2)) << 2));
        
        if(idx_n_16 * 4 + 1 >= N_4) return;
        res0 = vload4(0, input0 + idx_m * alignN + (idx_n_16 << 4) + 4);
        res1 = vload4(0, input1 + (idx_n_16 << 4) + 4);
        res = res0 + res1;
        #ifdef RELU
            res = fmax(res, (FLOAT4)0);
        #endif
        #ifdef RELU6
            res = clamp(res, (FLOAT4)0, (FLOAT4)6);
        #endif
        vstore4(res, 0, output + ((idx_m * N_4 + (idx_n_16 << 2)) << 2) + 4);
        
        if(idx_n_16 * 4 + 2 >= N_4) return;
        res0 = vload4(0, input0 + idx_m * alignN + (idx_n_16 << 4) + 8);
        res1 = vload4(0, input1 + (idx_n_16 << 4) + 8);
        res = res0 + res1;
        #ifdef RELU
            res = fmax(res, (FLOAT4)0);
        #endif
        #ifdef RELU6
            res = clamp(res, (FLOAT4)0, (FLOAT4)6);
        #endif
        vstore4(res, 0, output + ((idx_m * N_4 + (idx_n_16 << 2)) << 2) + 8);
        
        if(idx_n_16 * 4 + 3 >= N_4) return;
        res0 = vload4(0, input0 + idx_m * alignN + (idx_n_16 << 4) + 12);
        res1 = vload4(0, input1 + (idx_n_16 << 4) + 12);
        res = res0 + res1;
        #ifdef RELU
            res = fmax(res, (FLOAT4)0);
        #endif
        #ifdef RELU6
            res = clamp(res, (FLOAT4)0, (FLOAT4)6);
        #endif
        vstore4(res, 0, output + ((idx_m * N_4 + (idx_n_16 << 2)) << 2) + 12);
    }
#else
    const int idx_m = get_global_id(0); // idx M
    const int idx_n_16 = get_global_id(1); // idx N
    UNIFORM_BOUNDRY_CHECK(idx_m, idx_n_16);
    
    const int N_4 = (N + 3) >> 2;

    const int idx_b = idx_m / area;
    const int idx_area = idx_m % area;
    
    const int inp_base_offset = idx_m * alignN + (idx_n_16 << 4);
    const int out_base_offset = ((idx_b * N_4 + idx_n_16 * 4) * area + idx_area) * 4;
    
    FLOAT4 res0 = vload4(0, input0 + inp_base_offset);
    FLOAT4 res1 = vload4(0, input1 + (idx_n_16 << 4));
    FLOAT4 res = res0 + res1;
    #ifdef RELU
        res = fmax(res, (FLOAT4)0);
    #endif
    #ifdef RELU6
        res = clamp(res, (FLOAT4)0, (FLOAT4)6);
    #endif
    vstore4(res, 0, output + out_base_offset);
    
    if(idx_n_16 * 4 + 1 >= N_4) return;
    res0 = vload4(0, input0 + inp_base_offset + 4);
    res1 = vload4(0, input1 + (idx_n_16 << 4) + 4);
    res = res0 + res1;
    #ifdef RELU
        res = fmax(res, (FLOAT4)0);
    #endif
    #ifdef RELU6
        res = clamp(res, (FLOAT4)0, (FLOAT4)6);
    #endif
    vstore4(res, 0, output + out_base_offset + area * 4);
    
    if(idx_n_16 * 4 + 2 >= N_4) return;
    res0 = vload4(0, input0 + inp_base_offset + 8);
    res1 = vload4(0, input1 + (idx_n_16 << 4) + 8);
    res = res0 + res1;
    #ifdef RELU
        res = fmax(res, (FLOAT4)0);
    #endif
    #ifdef RELU6
        res = clamp(res, (FLOAT4)0, (FLOAT4)6);
    #endif
    vstore4(res, 0, output + out_base_offset + area * 8);
    
    if(idx_n_16 * 4 + 3 >= N_4) return;
    res0 = vload4(0, input0 + inp_base_offset + 12);
    res1 = vload4(0, input1 + (idx_n_16 << 4) + 12);
    res = res0 + res1;
    #ifdef RELU
        res = fmax(res, (FLOAT4)0);
    #endif
    #ifdef RELU6
        res = clamp(res, (FLOAT4)0, (FLOAT4)6);
    #endif
    vstore4(res, 0, output + out_base_offset + area * 12);
#endif
}
