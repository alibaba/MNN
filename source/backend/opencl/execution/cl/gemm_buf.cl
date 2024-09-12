#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

// [K/4, M, 4] -> [alignK, alignM]
__kernel void transpose_pad(GLOBAL_SIZE_DIM2
                        const int alignM,
                        const int alignK,
                        const int M,
                        const int K,
                        const int area,
                        __global const FLOAT* input,
                        __global FLOAT* output
                        ) {
    const int idx_m4 = get_global_id(0); // idx M
    const int idx_k4 = get_global_id(1); // idx K
    UNIFORM_BOUNDRY_CHECK(idx_m4, idx_k4);

    const int idx_m = idx_m4 << 2;
    const int idx_k = idx_k4 << 2;
    const int K_4 = (K + 3) >> 2;
    const int in_offset_base  = (idx_k4 * M + idx_m) * 4;
    const int out_offset_base = idx_k * alignM + idx_m;
    
    FLOAT4 m0k4 = (idx_k4 >= K_4 || idx_m + 0 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base);
    FLOAT4 m1k4 = (idx_k4 >= K_4 || idx_m + 1 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 4);
    FLOAT4 m2k4 = (idx_k4 >= K_4 || idx_m + 2 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 8);
    FLOAT4 m3k4 = (idx_k4 >= K_4 || idx_m + 3 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 12);
    
    vstore4((FLOAT4)(m0k4.x, m1k4.x, m2k4.x, m3k4.x), 0, output + out_offset_base);
    vstore4((FLOAT4)(m0k4.y, m1k4.y, m2k4.y, m3k4.y), 0, output + out_offset_base + alignM);
    vstore4((FLOAT4)(m0k4.z, m1k4.z, m2k4.z, m3k4.z), 0, output + out_offset_base + alignM + alignM);
    vstore4((FLOAT4)(m0k4.w, m1k4.w, m2k4.w, m3k4.w), 0, output + out_offset_base + alignM + alignM + alignM);
}

#ifndef M_VEC
#define M_VEC 1
#endif

// [alignM, alignN] -> [N/4, B, area, N4] (M = B * area)
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
    int idx_m = get_global_id(0); // idx M
    int idx_n4 = get_global_id(1); // idx N
    UNIFORM_BOUNDRY_CHECK(idx_m, idx_n4);

    const int idx_n = idx_n4 << 2;

    idx_m = idx_m * M_VEC;
    FLOAT4 res1 = vload4(0, input1 + idx_n);
    #pragma unroll
    for(int i = 0; i < M_VEC; i++) {
        FLOAT4 res0 = vload4(0, input0 + (idx_m + i) * alignN + idx_n);
        FLOAT4 res = res0 + res1;
        #ifdef RELU
        res = fmax(res, (FLOAT4)0);
        #endif
        #ifdef RELU6
        res = clamp(res, (FLOAT4)0, (FLOAT4)6);
        #endif
        vstore4(res, 0, output + ((idx_n4 * M + idx_m + i) << 2));
    }
}
