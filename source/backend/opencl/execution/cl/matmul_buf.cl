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
                     __private const int M,
                     __private const int N,
                     __private const int K) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); // N M

    DEAL_NON_UNIFORM_DIM2(pos.x, pos.y);
    const int idn = pos.x << 2;
    const int idm = pos.y << 2;
    
    COMPUTE_FLOAT4 out[4];

    #ifdef BIAS
    COMPUTE_FLOAT4 bias = CONVERT_COMPUTE_FLOAT4(vload4(0, input_c + idn));
    #pragma unroll
    for(int i = 0; i < 4; ++i){
        out[i] = bias;
    }
    #else
    #pragma unroll
    for(int i = 0; i < 4; ++i){
        out[i] = (COMPUTE_FLOAT4)0;
    }
    #endif

    const int K4 = (K + 3)/4;
    #ifdef K_LEAVE
    const int loop_end = max(K4 - 1, 0);
    const int remain = K - loop_end*4;
    #else
    const int loop_end = K4;
    #endif
    
    #ifdef TRANSPOSE_A
    __global const FLOAT* input_a_offset = input_a + idm; // K x M
    #else
    __global const FLOAT* input_a_offset = input_a + idm * K; // M x K
    #endif
    
    #ifdef TRANSPOSE_B
    __global const FLOAT* input_b_offset = input_b + idn * K; // N x K
    #else
    __global const FLOAT* input_b_offset = input_b + idn; // K x N
    #endif
    
    for (int k = 0; k < loop_end; ++k) {
        int kindex = k << 2;
        COMPUTE_FLOAT4 A[4]; // m4 x k4
        COMPUTE_FLOAT4 B[4]; // k4 x n4
        #ifdef M_LEAVE
        if(idm + 3 >= M){
            #ifdef TRANSPOSE_A
                #if M_LEAVE_NUM == 3
                {
                    COMPUTE_FLOAT3 tmp0 = CONVERT_COMPUTE_FLOAT3(vload3(0, input_a_offset + kindex * M));
                    COMPUTE_FLOAT3 tmp1 = CONVERT_COMPUTE_FLOAT3(vload3(0, input_a_offset + (kindex + 1) * M));
                    COMPUTE_FLOAT3 tmp2 = CONVERT_COMPUTE_FLOAT3(vload3(0, input_a_offset + (kindex + 2) * M));
                    COMPUTE_FLOAT3 tmp3 = CONVERT_COMPUTE_FLOAT3(vload3(0, input_a_offset + (kindex + 3) * M));
            
                    A[0] = (COMPUTE_FLOAT4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x);
                    A[1] = (COMPUTE_FLOAT4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y);
                    A[2] = (COMPUTE_FLOAT4)(tmp0.z, tmp1.z, tmp2.z, tmp3.z);
                    A[3] = (COMPUTE_FLOAT4)0;
                }
                #elif M_LEAVE_NUM == 2
                {
                    COMPUTE_FLOAT2 tmp0 = CONVERT_COMPUTE_FLOAT2(vload2(0, input_a_offset + kindex * M));
                    COMPUTE_FLOAT2 tmp1 = CONVERT_COMPUTE_FLOAT2(vload2(0, input_a_offset + (kindex + 1) * M));
                    COMPUTE_FLOAT2 tmp2 = CONVERT_COMPUTE_FLOAT2(vload2(0, input_a_offset + (kindex + 2) * M));
                    COMPUTE_FLOAT2 tmp3 = CONVERT_COMPUTE_FLOAT2(vload2(0, input_a_offset + (kindex + 3) * M));
                    
                    A[0] = (COMPUTE_FLOAT4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x);
                    A[1] = (COMPUTE_FLOAT4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y);
                    A[2] = (COMPUTE_FLOAT4)0;
                    A[3] = (COMPUTE_FLOAT4)0;
                }
                #elif M_LEAVE_NUM == 1
                {
                    A[0] = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)input_a_offset[kindex * M], (COMPUTE_FLOAT)input_a_offset[(kindex + 1) * M], (COMPUTE_FLOAT)input_a_offset[(kindex + 2) * M], (COMPUTE_FLOAT)input_a_offset[(kindex + 3) * M]);
                    A[1] = (COMPUTE_FLOAT4)0;
                    A[2] = (COMPUTE_FLOAT4)0;
                    A[3] = (COMPUTE_FLOAT4)0;
                }
                #endif
            #else
                #if M_LEAVE_NUM == 3
                    A[0] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex));
                    A[1] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex + K));
                    A[2] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex + 2 * K));
                    A[3] = (COMPUTE_FLOAT4)0;
                #elif M_LEAVE_NUM == 2
                    A[0] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex));
                    A[1] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex + K));
                    A[2] = (COMPUTE_FLOAT4)0;
                    A[3] = (COMPUTE_FLOAT4)0;
                #elif M_LEAVE_NUM == 1
                    A[0] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex));
                    A[1] = (COMPUTE_FLOAT4)0;
                    A[2] = (COMPUTE_FLOAT4)0;
                    A[3] = (COMPUTE_FLOAT4)0;
                #endif
            #endif
        } else
        #endif
        {
            #ifdef TRANSPOSE_A
            {
                COMPUTE_FLOAT4 tmp0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex * M));
                COMPUTE_FLOAT4 tmp1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + (kindex + 1) * M));
                COMPUTE_FLOAT4 tmp2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + (kindex + 2) * M));
                COMPUTE_FLOAT4 tmp3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + (kindex + 3) * M));
            
                A[0] = (COMPUTE_FLOAT4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x);
                A[1] = (COMPUTE_FLOAT4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y);
                A[2] = (COMPUTE_FLOAT4)(tmp0.z, tmp1.z, tmp2.z, tmp3.z);
                A[3] = (COMPUTE_FLOAT4)(tmp0.w, tmp1.w, tmp2.w, tmp3.w);
            }
            #else
            A[0] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex));
            A[1] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex + K));
            A[2] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex + 2 * K));
            A[3] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + kindex + 3 * K));
            #endif
        }
    
        #ifdef N_LEAVE
        if(idn + 3 >= N){
            #ifdef TRANSPOSE_B
                #if N_LEAVE_NUM == 3
                {
                    COMPUTE_FLOAT4 tmp0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex));
                    COMPUTE_FLOAT4 tmp1 = idn + 1 >= N ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex + K));
                    COMPUTE_FLOAT4 tmp2 = idn + 2 >= N ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex + 2 * K));
        
                    B[0] = (COMPUTE_FLOAT4)(tmp0.x, tmp1.x, tmp2.x, 0);
                    B[1] = (COMPUTE_FLOAT4)(tmp0.y, tmp1.y, tmp2.y, 0);
                    B[2] = (COMPUTE_FLOAT4)(tmp0.z, tmp1.z, tmp2.z, 0);
                    B[3] = (COMPUTE_FLOAT4)(tmp0.w, tmp1.w, tmp2.w, 0);
                }
                #elif N_LEAVE_NUM == 2
                {
                    COMPUTE_FLOAT4 tmp0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex));
                    COMPUTE_FLOAT4 tmp1 = idn + 1 >= N ? (COMPUTE_FLOAT4)0 : CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex + K));
        
                    B[0] = (COMPUTE_FLOAT4)(tmp0.x, tmp1.x, 0, 0);
                    B[1] = (COMPUTE_FLOAT4)(tmp0.y, tmp1.y, 0, 0);
                    B[2] = (COMPUTE_FLOAT4)(tmp0.z, tmp1.z, 0, 0);
                    B[3] = (COMPUTE_FLOAT4)(tmp0.w, tmp1.w, 0, 0);
                }
                #elif N_LEAVE_NUM == 1
                {
                    COMPUTE_FLOAT4 tmp0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex));
        
                    B[0] = (COMPUTE_FLOAT4)(tmp0.x, 0, 0, 0);
                    B[1] = (COMPUTE_FLOAT4)(tmp0.y, 0, 0, 0);
                    B[2] = (COMPUTE_FLOAT4)(tmp0.z, 0, 0, 0);
                    B[3] = (COMPUTE_FLOAT4)(tmp0.w, 0, 0, 0);
                }
                #endif
            #else
                #if N_LEAVE_NUM == 3
                {
                    B[0] = (COMPUTE_FLOAT4)(CONVERT_COMPUTE_FLOAT3(vload3(0, input_b_offset + kindex * N)), 0);
                    B[1] = (COMPUTE_FLOAT4)(CONVERT_COMPUTE_FLOAT3(vload3(0, input_b_offset + (kindex + 1) * N)), 0);
                    B[2] = (COMPUTE_FLOAT4)(CONVERT_COMPUTE_FLOAT3(vload3(0, input_b_offset + (kindex + 2) * N)), 0);
                    B[3] = (COMPUTE_FLOAT4)(CONVERT_COMPUTE_FLOAT3(vload3(0, input_b_offset + (kindex + 3) * N)), 0);
                }
                #elif N_LEAVE_NUM == 2
                {
                    B[0] = (COMPUTE_FLOAT4)(CONVERT_COMPUTE_FLOAT2(vload2(0, input_b_offset + kindex * N)), 0, 0);
                    B[1] = (COMPUTE_FLOAT4)(CONVERT_COMPUTE_FLOAT2(vload2(0, input_b_offset + (kindex + 1) * N)), 0, 0);
                    B[2] = (COMPUTE_FLOAT4)(CONVERT_COMPUTE_FLOAT2(vload2(0, input_b_offset + (kindex + 2) * N)), 0, 0);
                    B[3] = (COMPUTE_FLOAT4)(CONVERT_COMPUTE_FLOAT2(vload2(0, input_b_offset + (kindex + 3) * N)), 0, 0);
                }
                #elif N_LEAVE_NUM == 1
                {
                    B[0] = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)input_b_offset[kindex * N], 0, 0, 0);
                    B[1] = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)input_b_offset[(kindex + 1) * N], 0, 0, 0);
                    B[2] = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)input_b_offset[(kindex + 2) * N], 0, 0, 0);
                    B[3] = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)input_b_offset[(kindex + 3) * N], 0, 0, 0);
                }
                #endif
            #endif
        } else
        #endif
        {
            #ifdef TRANSPOSE_B
            {
                COMPUTE_FLOAT4 tmp0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex));
                COMPUTE_FLOAT4 tmp1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex + K));
                COMPUTE_FLOAT4 tmp2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex + 2 * K));
                COMPUTE_FLOAT4 tmp3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex + 3 * K));
            
                B[0] = (COMPUTE_FLOAT4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x);
                B[1] = (COMPUTE_FLOAT4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y);
                B[2] = (COMPUTE_FLOAT4)(tmp0.z, tmp1.z, tmp2.z, tmp3.z);
                B[3] = (COMPUTE_FLOAT4)(tmp0.w, tmp1.w, tmp2.w, tmp3.w);
            }
            #else
            B[0] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + kindex * N));
            B[1] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + (kindex + 1) * N));
            B[2] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + (kindex + 2) * N));
            B[3] = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + (kindex + 3) * N));
            #endif
        }
        
        #pragma unroll
        for (int vec_m = 0; vec_m < 4; ++vec_m){
            out[vec_m] = mad((COMPUTE_FLOAT4)A[vec_m].x, B[0], out[vec_m]);
            out[vec_m] = mad((COMPUTE_FLOAT4)A[vec_m].y, B[1], out[vec_m]);
            out[vec_m] = mad((COMPUTE_FLOAT4)A[vec_m].z, B[2], out[vec_m]);
            out[vec_m] = mad((COMPUTE_FLOAT4)A[vec_m].w, B[3], out[vec_m]);
        }
    }
    #ifdef K_LEAVE
     for (int k = loop_end << 2; k < K; ++k){
        COMPUTE_FLOAT4 A; // m4
        COMPUTE_FLOAT4 B; // n4
        #ifdef M_LEAVE
        if(idm + 3 >= M){
            #ifdef TRANSPOSE_A
                #if M_LEAVE_NUM == 3
                A.s012 = CONVERT_COMPUTE_FLOAT3(vload3(0, input_a_offset + k * M));
                #elif M_LEAVE_NUM == 2
                A.s01 = CONVERT_COMPUTE_FLOAT2(vload2(0, input_a_offset + k * M));
                #elif M_LEAVE_NUM == 1
                A.s0 = (COMPUTE_FLOAT)input_a_offset[k * M];
                #endif
            #else
                A.x = (COMPUTE_FLOAT)input_a_offset[k];
                #if M_LEAVE_NUM >= 2
                A.y = (COMPUTE_FLOAT)input_a_offset[k + K];
                #endif
                #if M_LEAVE_NUM >= 3
                A.z = (COMPUTE_FLOAT)input_a_offset[k + 2 * K];
                #endif
            #endif
        } else
        #endif
        {
            #ifdef TRANSPOSE_A
            A = CONVERT_COMPUTE_FLOAT4(vload4(0, input_a_offset + k * M));
            #else
            A.x = (COMPUTE_FLOAT)input_a_offset[k];
            A.y = (COMPUTE_FLOAT)input_a_offset[k + K];
            A.z = (COMPUTE_FLOAT)input_a_offset[k + 2 * K];
            A.w = (COMPUTE_FLOAT)input_a_offset[k + 3 * K];
            #endif
        }
        
        #ifdef N_LEAVE
        if(idn + 3 >= N){
            #ifdef TRANSPOSE_B
                B.x = (COMPUTE_FLOAT)input_b_offset[k];
                #if N_LEAVE_NUM >= 2
                B.y = (COMPUTE_FLOAT)input_b_offset[k + K];
                #endif
                #if N_LEAVE_NUM >= 3
                B.z = (COMPUTE_FLOAT)input_b_offset[k + 2 * K];
                #endif
            #else
                #if N_LEAVE_NUM == 3
                B.s012 = CONVERT_COMPUTE_FLOAT3(vload3(0, input_b_offset + k * N));
                #elif N_LEAVE_NUM == 2
                B.s01 = CONVERT_COMPUTE_FLOAT2(vload2(0, input_b_offset + k * N));
                #elif N_LEAVE_NUM == 1
                B.s0 = (COMPUTE_FLOAT)input_b_offset[k * N];
                #endif
            #endif
        } else
        #endif
        {
            #ifdef TRANSPOSE_B
            B.x = (COMPUTE_FLOAT)input_b_offset[k];
            B.y = (COMPUTE_FLOAT)input_b_offset[k + K];
            B.z = (COMPUTE_FLOAT)input_b_offset[k + 2 * K];
            B.w = (COMPUTE_FLOAT)input_b_offset[k + 3 * K];
            #else
            B = CONVERT_COMPUTE_FLOAT4(vload4(0, input_b_offset + k * N));
            #endif
        }
        out[0] = mad((COMPUTE_FLOAT4)A.x, B, out[0]);
        out[1] = mad((COMPUTE_FLOAT4)A.y, B, out[1]);
        out[2] = mad((COMPUTE_FLOAT4)A.z, B, out[2]);
        out[3] = mad((COMPUTE_FLOAT4)A.w, B, out[3]);
    }
    #endif
    
    
    const int out_offset = idm * N + idn;
    #ifdef M_LEAVE
    if(idm + 3 >= M){
        #ifdef N_LEAVE
        if(idn + 3 >= N){
            for (int vec_m = 0; vec_m < M - idm; ++vec_m){
                COMPUTE_FLOAT *out_ptr = (COMPUTE_FLOAT*)&out[vec_m];
                for(int vec_n = 0; vec_n < N - idn; ++vec_n){
                    output_c[out_offset + vec_m * N + vec_n] = out_ptr[vec_n];
                }
            }
        } else {
        #endif
            for (int vec_m = 0; vec_m < M - idm; ++vec_m){
                vstore4(CONVERT_FLOAT4(out[vec_m]), 0, output_c + out_offset + vec_m * N);
            }
            
        #ifdef N_LEAVE
        }
        #endif
    } else{
    #endif
        #ifdef N_LEAVE
        if(idn + 3 >= N){
            #pragma unroll
            for (int vec_m = 0; vec_m < 4; ++vec_m){
                COMPUTE_FLOAT *out_ptr = (COMPUTE_FLOAT*)&out[vec_m];
                for(int vec_n = 0; vec_n < N - idn; ++vec_n){
                    output_c[out_offset + vec_m * N + vec_n] = out_ptr[vec_n];
                }
            }
        } else {
        #endif
            #pragma unroll
            for (int vec_m = 0; vec_m < 4; ++vec_m){
                vstore4(CONVERT_FLOAT4(out[vec_m]), 0, output_c + out_offset + vec_m * N);
            }
        #ifdef N_LEAVE
        }
        #endif
    #ifdef M_LEAVE
    }
    #endif
}
