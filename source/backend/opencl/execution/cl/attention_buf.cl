#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }


__kernel void matmul_qk_div_mask(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input0, // query [1 query_seq_len/4 head_num head_dim 4]
                              __global const FLOAT *input1, // key [1 key_seq_len/4 head_num head_dim 4]
                              __global FLOAT *output, // prefill [1 head_num query_seq_len/4 key_seq_len 4]   decode[1 head_num key_seq_len/4 4]
                              __global FLOAT *past_key, // [1 key_seq_len/4 head_num head_dim 4]
                              __global const int* mask, // [1 1 query_seq_len key_seq_len 4]
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int key_seq_len,
                              __private const int head_num,
                              __private const int head_dim) {

    const int x = get_global_id(0); // query_seq_len / 4 for prefill   1 for decode
    const int y = get_global_id(1); // head_num
    const int z = get_global_id(2); // key_seq_len / 4
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int offset = head_num * head_dim * 4;
    const int offset_head = y * head_dim * 4;
    __global const FLOAT *A_offset = input0 + x * offset + offset_head;
    __global FLOAT *Pastkey_offset = past_key + z * offset + offset_head;
    const int z4 = z << 2;
    COMPUTE_FLOAT4 Vscale = (COMPUTE_FLOAT4)scale;
#ifdef OPENCL_PREFILL_ATTENTION
    __global const FLOAT *B_offset = input1 + z * offset + offset_head;
    const int x4 = x << 2;
    const int query_seq_len4 = (query_seq_len + 3) / 4;
    const int output_offset = y * query_seq_len4 * key_seq_len * 4;
    COMPUTE_FLOAT4 out0 = 0;
    COMPUTE_FLOAT4 out1 = 0;
    COMPUTE_FLOAT4 out2 = 0;
    COMPUTE_FLOAT4 out3 = 0;
    
    const int head_dim4 = (head_dim + 3) / 4;
#ifdef HEADDIM_LEAVE
    for(int i = 0; i < head_dim4 - 1; ++i){
        COMPUTE_FLOAT16 A = CONVERT_COMPUTE_FLOAT16(vload16(i, A_offset));
        COMPUTE_FLOAT16 B = CONVERT_COMPUTE_FLOAT16(vload16(i, B_offset));
        
        out0 = mad(A.s0123, (COMPUTE_FLOAT4)B.s0, out0);
        out1 = mad(A.s0123, (COMPUTE_FLOAT4)B.s1, out1);
        out2 = mad(A.s0123, (COMPUTE_FLOAT4)B.s2, out2);
        out3 = mad(A.s0123, (COMPUTE_FLOAT4)B.s3, out3);
        
        out0 = mad(A.s4567, (COMPUTE_FLOAT4)B.s4, out0);
        out1 = mad(A.s4567, (COMPUTE_FLOAT4)B.s5, out1);
        out2 = mad(A.s4567, (COMPUTE_FLOAT4)B.s6, out2);
        out3 = mad(A.s4567, (COMPUTE_FLOAT4)B.s7, out3);
        
        out0 = mad(A.s89ab, (COMPUTE_FLOAT4)B.s8, out0);
        out1 = mad(A.s89ab, (COMPUTE_FLOAT4)B.s9, out1);
        out2 = mad(A.s89ab, (COMPUTE_FLOAT4)B.sa, out2);
        out3 = mad(A.s89ab, (COMPUTE_FLOAT4)B.sb, out3);
        
        out0 = mad(A.scdef, (COMPUTE_FLOAT4)B.sc, out0);
        out1 = mad(A.scdef, (COMPUTE_FLOAT4)B.sd, out1);
        out2 = mad(A.scdef, (COMPUTE_FLOAT4)B.se, out2);
        out3 = mad(A.scdef, (COMPUTE_FLOAT4)B.sf, out3);
        
        vstore16(CONVERT_FLOAT16(B), i, Pastkey_offset);
    }
    for(int i = (head_dim4 - 1) * 4; i < head_dim; ++i){
        COMPUTE_FLOAT4 A = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset));
        COMPUTE_FLOAT4 B = CONVERT_COMPUTE_FLOAT4(vload4(i, B_offset));
        
        out0 = mad(A, (COMPUTE_FLOAT4)B.s0, out0);
        out1 = mad(A, (COMPUTE_FLOAT4)B.s1, out1);
        out2 = mad(A, (COMPUTE_FLOAT4)B.s2, out2);
        out3 = mad(A, (COMPUTE_FLOAT4)B.s3, out3);
        
        vstore4(CONVERT_FLOAT4(B), i, Pastkey_offset);
    }
#else
    for(int i = 0; i < head_dim4; ++i){
        COMPUTE_FLOAT16 A = CONVERT_COMPUTE_FLOAT16(vload16(i, A_offset));
        COMPUTE_FLOAT16 B = CONVERT_COMPUTE_FLOAT16(vload16(i, B_offset));
        
        out0 = mad(A.s0123, (COMPUTE_FLOAT4)B.s0, out0);
        out1 = mad(A.s0123, (COMPUTE_FLOAT4)B.s1, out1);
        out2 = mad(A.s0123, (COMPUTE_FLOAT4)B.s2, out2);
        out3 = mad(A.s0123, (COMPUTE_FLOAT4)B.s3, out3);
        
        out0 = mad(A.s4567, (COMPUTE_FLOAT4)B.s4, out0);
        out1 = mad(A.s4567, (COMPUTE_FLOAT4)B.s5, out1);
        out2 = mad(A.s4567, (COMPUTE_FLOAT4)B.s6, out2);
        out3 = mad(A.s4567, (COMPUTE_FLOAT4)B.s7, out3);
        
        out0 = mad(A.s89ab, (COMPUTE_FLOAT4)B.s8, out0);
        out1 = mad(A.s89ab, (COMPUTE_FLOAT4)B.s9, out1);
        out2 = mad(A.s89ab, (COMPUTE_FLOAT4)B.sa, out2);
        out3 = mad(A.s89ab, (COMPUTE_FLOAT4)B.sb, out3);
        
        out0 = mad(A.scdef, (COMPUTE_FLOAT4)B.sc, out0);
        out1 = mad(A.scdef, (COMPUTE_FLOAT4)B.sd, out1);
        out2 = mad(A.scdef, (COMPUTE_FLOAT4)B.se, out2);
        out3 = mad(A.scdef, (COMPUTE_FLOAT4)B.sf, out3);
    
        vstore16(CONVERT_FLOAT16(B), i, Pastkey_offset);
    }
#endif
    
    out0 *= Vscale;
    out1 *= Vscale;
    out2 *= Vscale;
    out3 *= Vscale;
    
    out0.s0 = mask[((x4 + 0) * key_seq_len + (z4 + 0)) * 4] == 0 ? -FLT_MAX : out0.s0;
    out1.s0 = mask[((x4 + 0) * key_seq_len + (z4 + 1)) * 4] == 0 ? -FLT_MAX : out1.s0;
    out2.s0 = mask[((x4 + 0) * key_seq_len + (z4 + 2)) * 4] == 0 ? -FLT_MAX : out2.s0;
    out3.s0 = mask[((x4 + 0) * key_seq_len + (z4 + 3)) * 4] == 0 ? -FLT_MAX : out3.s0;
    
    out0.s1 = mask[((x4 + 1) * key_seq_len + (z4 + 0)) * 4] == 0 ? -FLT_MAX : out0.s1;
    out1.s1 = mask[((x4 + 1) * key_seq_len + (z4 + 1)) * 4] == 0 ? -FLT_MAX : out1.s1;
    out2.s1 = mask[((x4 + 1) * key_seq_len + (z4 + 2)) * 4] == 0 ? -FLT_MAX : out2.s1;
    out3.s1 = mask[((x4 + 1) * key_seq_len + (z4 + 3)) * 4] == 0 ? -FLT_MAX : out3.s1;
    
    out0.s2 = mask[((x4 + 2) * key_seq_len + (z4 + 0)) * 4] == 0 ? -FLT_MAX : out0.s2;
    out1.s2 = mask[((x4 + 2) * key_seq_len + (z4 + 1)) * 4] == 0 ? -FLT_MAX : out1.s2;
    out2.s2 = mask[((x4 + 2) * key_seq_len + (z4 + 2)) * 4] == 0 ? -FLT_MAX : out2.s2;
    out3.s2 = mask[((x4 + 2) * key_seq_len + (z4 + 3)) * 4] == 0 ? -FLT_MAX : out3.s2;
    
    out0.s3 = mask[((x4 + 3) * key_seq_len + (z4 + 0)) * 4] == 0 ? -FLT_MAX : out0.s3;
    out1.s3 = mask[((x4 + 3) * key_seq_len + (z4 + 1)) * 4] == 0 ? -FLT_MAX : out1.s3;
    out2.s3 = mask[((x4 + 3) * key_seq_len + (z4 + 2)) * 4] == 0 ? -FLT_MAX : out2.s3;
    out3.s3 = mask[((x4 + 3) * key_seq_len + (z4 + 3)) * 4] == 0 ? -FLT_MAX : out3.s3;

    vstore4(CONVERT_FLOAT4(out0), 0, output + output_offset + x * key_seq_len * 4 + z4 * 4);
    if(z4 + 1 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out1), 0, output + output_offset + x * key_seq_len * 4 + (z4 + 1) * 4);
    if(z4 + 2 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out2), 0, output + output_offset + x * key_seq_len * 4 + (z4 + 2) * 4);
    if(z4 + 3 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out3), 0, output + output_offset + x * key_seq_len * 4 + (z4 + 3) * 4);
#else
    __global const FLOAT *B_offset = input1 + offset_head;
    const int key_seq_len4 = (key_seq_len + 3) / 4;
    COMPUTE_FLOAT4 out = 0;
    const int head_dim4 = (head_dim + 3) / 4;
    
#ifdef HEADDIM_LEAVE
    for(int i = 0; i < head_dim4 - 1; ++i){
        COMPUTE_FLOAT16 A = CONVERT_COMPUTE_FLOAT16(vload16(i, A_offset));
        COMPUTE_FLOAT16 B = CONVERT_COMPUTE_FLOAT16(vload16(i, Pastkey_offset));
        
        out = mad((COMPUTE_FLOAT4)A.s0, B.s0123, out);
        out = mad((COMPUTE_FLOAT4)A.s4, B.s4567, out);
        out = mad((COMPUTE_FLOAT4)A.s8, B.s89ab, out);
        out = mad((COMPUTE_FLOAT4)A.sc, B.scdef, out);
    }
    for(int i = (head_dim4 - 1) * 4; i < head_dim; ++i){
        COMPUTE_FLOAT4 A = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset));
        COMPUTE_FLOAT4 B = CONVERT_COMPUTE_FLOAT4(vload4(i, Pastkey_offset));
        
        out = mad((COMPUTE_FLOAT4)A.s0, B, out);
    }
#else
    for(int i = 0; i < head_dim4; ++i){
        COMPUTE_FLOAT16 A = CONVERT_COMPUTE_FLOAT16(vload16(i, A_offset));
        COMPUTE_FLOAT16 B = CONVERT_COMPUTE_FLOAT16(vload16(i, Pastkey_offset));
    
        out = mad((COMPUTE_FLOAT4)A.s0, B.s0123, out);
        out = mad((COMPUTE_FLOAT4)A.s4, B.s4567, out);
        out = mad((COMPUTE_FLOAT4)A.s8, B.s89ab, out);
        out = mad((COMPUTE_FLOAT4)A.sc, B.scdef, out);
    }
#endif
    if(z == key_seq_len4 - 1){
        int remain = key_seq_len - z * 4 - 1;
        Pastkey_offset += remain;
        COMPUTE_FLOAT tmp = 0;
        for(int i = 0; i < head_dim; ++i){
            COMPUTE_FLOAT A = A_offset[i*4];
            COMPUTE_FLOAT B = B_offset[i*4];
            Pastkey_offset[i * 4] = B;
            tmp += A * B;
        }
        COMPUTE_FLOAT *out_ptr = (COMPUTE_FLOAT*)&out;
        out_ptr[remain] = tmp;
    }
    out *= Vscale;
    vstore4(CONVERT_FLOAT4(out), 0, output + y * key_seq_len4 * 4 + z4);
#endif
}

__kernel void matmul_qkv(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input0, // qk prefill [1 head_num qk_seq_len/4 value_seq_len 4]   decode[1 head_num value_seq_len/4 4]
                              __global const FLOAT *input1, // [1 value_seq_len/4 head_num head_dim 4]
                              __global FLOAT *output, // [1 qk_seq_len head_num*head_dim 1 4]
                              __global FLOAT *past_value, // [1 value_seq_len/4 head_num head_dim 4]
                              __private const int qk_seq_len,
                              __private const int value_seq_len,
                              __private const int head_num,
                              __private const int head_dim) {

    const int x = get_global_id(0); // prefill qk_seq_len / 4   decode 1
    const int y = get_global_id(1); // head_num
    const int z = get_global_id(2); // head_dim
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
#ifdef OPENCL_PREFILL_ATTENTION
    const int offset = head_num * head_dim * 4;
    const int offset_head = y * head_dim * 4 + z * 4;
    const int value_seq_len4 = (value_seq_len + 3) / 4;
    const int qk_seq_len4 = (qk_seq_len + 3) / 4;
    __global const FLOAT *A_offset = input0 + (y * qk_seq_len4 + x) * value_seq_len * 4;
    __global const FLOAT *B_offset = input1 + offset_head;
    __global FLOAT *Pastvalue_offset = past_value + offset_head;
    COMPUTE_FLOAT4 out = 0;
    
    for(int i = 0; i < value_seq_len4 - 1; ++i){
        int index = i << 2;
        COMPUTE_FLOAT4 A0 = CONVERT_COMPUTE_FLOAT4(vload4(index, A_offset));
        COMPUTE_FLOAT4 A1 = CONVERT_COMPUTE_FLOAT4(vload4(index + 1, A_offset));
        COMPUTE_FLOAT4 A2 = CONVERT_COMPUTE_FLOAT4(vload4(index + 2, A_offset));
        COMPUTE_FLOAT4 A3 = CONVERT_COMPUTE_FLOAT4(vload4(index + 3, A_offset));
        COMPUTE_FLOAT4 B = CONVERT_COMPUTE_FLOAT4(vload4(0, B_offset + i * offset));
        
        out = mad(A0, (COMPUTE_FLOAT4)B.s0, out);
        out = mad(A1, (COMPUTE_FLOAT4)B.s1, out);
        out = mad(A2, (COMPUTE_FLOAT4)B.s2, out);
        out = mad(A3, (COMPUTE_FLOAT4)B.s3, out);

        vstore4(CONVERT_FLOAT4(B), 0, Pastvalue_offset + i * offset);
    }
    
    COMPUTE_FLOAT4 B = CONVERT_COMPUTE_FLOAT4(vload4(0, B_offset + (value_seq_len4 - 1) * offset));
    vstore4(CONVERT_FLOAT4(B), 0, Pastvalue_offset + (value_seq_len4 - 1) * offset);
    COMPUTE_FLOAT *B_ptr = (COMPUTE_FLOAT*)&B;
    for(int i = (value_seq_len4 - 1) * 4, j = 0; i < value_seq_len; ++i, ++j){
        COMPUTE_FLOAT4 A0 = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset));
        out = mad(A0, (COMPUTE_FLOAT4)B_ptr[j], out);
    }
    vstore4(CONVERT_FLOAT4(out), 0, output + x * offset + (y * head_dim + z) * 4);
#else
    const int z4 = z << 2;
    const int value_seq_len4 = (value_seq_len + 3) / 4;
    const int offset = head_num * head_dim * 4;
    const int offset_head = y * head_dim * 4 + z4 * 4;
    const int loop = (value_seq_len + 2) / 4;
    __global const FLOAT *A_offset = input0 + y * value_seq_len4 * 4;
    __global const FLOAT *B_offset = input1 + offset_head;
    __global FLOAT *Pastvalue_offset = past_value + offset_head;
    COMPUTE_FLOAT4 out = 0;
    
    for(int i = 0; i < loop - 1; i++){
        COMPUTE_FLOAT4 A = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset));
        COMPUTE_FLOAT16 B = CONVERT_COMPUTE_FLOAT16(vload16(0, Pastvalue_offset + i * offset));
        
        out.s0 += dot(A, B.s0123);
        out.s1 += dot(A, B.s4567);
        out.s2 += dot(A, B.s89ab);
        out.s3 += dot(A, B.scdef);
    }
    int start = (loop - 1) < 0 ? 0 : (loop - 1);
    COMPUTE_FLOAT16 B_Vec = CONVERT_COMPUTE_FLOAT16(vload16(0, Pastvalue_offset + start * offset));
    COMPUTE_FLOAT *B_ptr = (COMPUTE_FLOAT *)&B_Vec;
    for(int i = start * 4; i < value_seq_len - 1; ++i){
        COMPUTE_FLOAT A = A_offset[i];
        
        int index = i % 4;
        out.s0 += A * B_ptr[index];
        out.s1 += A * B_ptr[index+4];
        out.s2 += A * B_ptr[index+8];
        out.s3 += A * B_ptr[index+12];
    }
    COMPUTE_FLOAT A = A_offset[value_seq_len - 1];
    COMPUTE_FLOAT B0 = B_offset[0];
    COMPUTE_FLOAT B1 = B_offset[4];
    COMPUTE_FLOAT B2 = B_offset[8];
    COMPUTE_FLOAT B3 = B_offset[12];
    out.s0 += A * B0;
    out.s1 += A * B1;
    out.s2 += A * B2;
    out.s3 += A * B3;
    int index = ((value_seq_len - 1) >> 2) * offset + ((value_seq_len - 1) % 4);
    
#ifdef HEADDIM_LEAVE
    Pastvalue_offset[index] = B0;
    output[(y * head_dim + z4) * 4] = out.s0;
    if(z4 + 1 >= head_dim) return;
    Pastvalue_offset[index + 4] = B1;
    output[(y * head_dim + z4 + 1) * 4] = out.s1;
    if(z4 + 2 >= head_dim) return;
    Pastvalue_offset[index + 8] = B2;
    output[(y * head_dim + z4 + 2) * 4] = out.s2;
    if(z4 + 3 >= head_dim) return;
    Pastvalue_offset[index + 12] = B3;
    output[(y * head_dim + z4 + 3) * 4] = out.s3;
#else
    Pastvalue_offset[index] = B0;
    Pastvalue_offset[index + 4] = B1;
    Pastvalue_offset[index + 8] = B2;
    Pastvalue_offset[index + 12] = B3;
    
    output[(y * head_dim + z4) * 4] = out.s0;
    output[(y * head_dim + z4 + 1) * 4] = out.s1;
    output[(y * head_dim + z4 + 2) * 4] = out.s2;
    output[(y * head_dim + z4 + 3) * 4] = out.s3;
#endif
    
#endif
}

