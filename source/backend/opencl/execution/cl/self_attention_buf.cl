#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define DEAL_HEAD_DIM_NOT_ALIGN \
    if(hd * 4 + 3 >= head_dim) {\
        temp_0.w = (FLOAT)0;\
        temp_1.w = (FLOAT)0;\
        temp_2.w = (FLOAT)0;\
        temp_3.w = (FLOAT)0;\
    }\
    if(hd * 4 + 2 >= head_dim) {\
        temp_0.z = (FLOAT)0;\
        temp_1.z = (FLOAT)0;\
        temp_2.z = (FLOAT)0;\
        temp_3.z = (FLOAT)0;\
    }\
    if(hd * 4 + 1 >= head_dim) {\
        temp_0.y = (FLOAT)0;\
        temp_1.y = (FLOAT)0;\
        temp_2.y = (FLOAT)0;\
        temp_3.y = (FLOAT)0;\
    }

#define DEAL_SEQ_LEN_NOT_ALIGN \
    if(4 * sl + 3 >= seq_len) {\
        temp_3 = (FLOAT4)0;\
    }\
    if(4 * sl + 2 >= seq_len) {\
        temp_2 = (FLOAT4)0;\
    }\
    if(4 * sl + 1 >= seq_len) {\
        temp_1 = (FLOAT4)0;\
    }

__kernel void split_transpose_qkv(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input, // [Batch, seqLen/4, mNumHead * 3 * mHeadDim, 4]
                              __global FLOAT *output_q, // [Batch * mNumHead, head_dim_pack_k, seq_len_pack_mn / qSeqSplitNum]
                              __global FLOAT *output_k, // [Batch * mNumHead, head_dim_pack_k, seq_len_pack_mn]
                              __global FLOAT *output_v, // [Batch * mNumHead, ROUND_UP(seqLen, tile), head_dim_pack_mn]
                              __private const int seq_len_pack_mn,
                              __private const int seq_len_piece,
                              __private const int head_dim_pack_mn,
                              __private const int head_dim_pack_k,
                              __private const int seq_len,
                              __private const int head_num,
                              __private const int head_dim,
                              __private const int seq_index
) {
    const int sl = get_global_id(0); // seqLen_4
    const int hd = get_global_id(1); // mHeadDim_4
    const int z = get_global_id(2); // Batch * mNumHead
    DEAL_NON_UNIFORM_DIM3(sl, hd, z);
    
    const int b = z / head_num;
    const int hn = z % head_num;
    
    const int seq_len_4 = (seq_len + 3) / 4;
    const int offset_q = ((b * head_num + hn) * head_dim_pack_k + 4 * hd) * seq_len_piece + 4 * sl;

    if(seq_index > 0) {
        // fill output_q only
        if(sl * 4 >= seq_len || hd * 4 >= head_dim) {
            if(hd * 4 < head_dim_pack_k) {
                if(sl * 4 < seq_len_piece) {
                    vstore4((FLOAT4)0, 0, output_q + offset_q);
                    vstore4((FLOAT4)0, 0, output_q + offset_q + seq_len_piece);
                    vstore4((FLOAT4)0, 0, output_q + offset_q + seq_len_piece + seq_len_piece);
                    vstore4((FLOAT4)0, 0, output_q + offset_q + seq_len_piece + seq_len_piece + seq_len_piece);
                }
            }
            return;
        }
        
        const int offset_inp = (((b * seq_len_4 + seq_index * seq_len_piece / 4 + sl) * head_num + hn) * 3 * head_dim + 4 * hd) * 4;
        
        if(sl * 4 < seq_len_piece) {
            FLOAT4 temp_0 = vload4(0, input + offset_inp);
            FLOAT4 temp_1 = vload4(0, input + offset_inp + 4);
            FLOAT4 temp_2 = vload4(0, input + offset_inp + 8);
            FLOAT4 temp_3 = vload4(0, input + offset_inp + 12);
            #ifdef HEADDIM_LEAVE
            DEAL_HEAD_DIM_NOT_ALIGN
            #endif
            #ifdef SEQLEN_LEAVE
            DEAL_SEQ_LEN_NOT_ALIGN
            #endif
            vstore4(temp_0, 0, output_q + offset_q);
            vstore4(temp_1, 0, output_q + offset_q + seq_len_piece);
            vstore4(temp_2, 0, output_q + offset_q + seq_len_piece + seq_len_piece);
            vstore4(temp_3, 0, output_q + offset_q + seq_len_piece + seq_len_piece + seq_len_piece);
        }
        return;
    }
    const int offset_k = ((b * head_num + hn) * head_dim_pack_k + 4 * hd) * seq_len_pack_mn + 4 * sl;

    const int offset_v = ((b * head_num + hn) * seq_len_pack_mn + 4 * sl) * head_dim_pack_mn + 4 * hd;
    if(sl * 4 >= seq_len || hd * 4 >= head_dim) {
        if(hd * 4 < head_dim_pack_k) {
            if(sl * 4 < seq_len_piece) {
                vstore4((FLOAT4)0, 0, output_q + offset_q);
                vstore4((FLOAT4)0, 0, output_q + offset_q + seq_len_piece);
                vstore4((FLOAT4)0, 0, output_q + offset_q + seq_len_piece + seq_len_piece);
                vstore4((FLOAT4)0, 0, output_q + offset_q + seq_len_piece + seq_len_piece + seq_len_piece);
            }
            vstore4((FLOAT4)0, 0, output_k + offset_k);
            vstore4((FLOAT4)0, 0, output_k + offset_k + seq_len_pack_mn);
            vstore4((FLOAT4)0, 0, output_k + offset_k + seq_len_pack_mn + seq_len_pack_mn);
            vstore4((FLOAT4)0, 0, output_k + offset_k + seq_len_pack_mn + seq_len_pack_mn + seq_len_pack_mn);
        }
        vstore4((FLOAT4)0, 0, output_v + offset_v);
        vstore4((FLOAT4)0, 0, output_v + offset_v + head_dim_pack_mn);
        vstore4((FLOAT4)0, 0, output_v + offset_v + head_dim_pack_mn + head_dim_pack_mn);
        vstore4((FLOAT4)0, 0, output_v + offset_v + head_dim_pack_mn + head_dim_pack_mn + head_dim_pack_mn);
        
        return;
    }
    

    const int offset_inp = (((b * seq_len_4 + sl) * head_num + hn) * 3 * head_dim + 4 * hd) * 4;
    
    if(sl * 4 < seq_len_piece) {
        FLOAT4 temp_0 = vload4(0, input + offset_inp);
        FLOAT4 temp_1 = vload4(0, input + offset_inp + 4);
        FLOAT4 temp_2 = vload4(0, input + offset_inp + 8);
        FLOAT4 temp_3 = vload4(0, input + offset_inp + 12);
        #ifdef HEADDIM_LEAVE
        DEAL_HEAD_DIM_NOT_ALIGN
        #endif
        #ifdef SEQLEN_LEAVE
        DEAL_SEQ_LEN_NOT_ALIGN
        #endif
        vstore4(temp_0, 0, output_q + offset_q);
        vstore4(temp_1, 0, output_q + offset_q + seq_len_piece);
        vstore4(temp_2, 0, output_q + offset_q + seq_len_piece + seq_len_piece);
        vstore4(temp_3, 0, output_q + offset_q + seq_len_piece + seq_len_piece + seq_len_piece);
    }
    
    {
        // K
        FLOAT4 temp_0 = vload4(0, input + offset_inp + 4*head_dim);
        FLOAT4 temp_1 = vload4(0, input + offset_inp + 4*head_dim + 4);
        FLOAT4 temp_2 = vload4(0, input + offset_inp + 4*head_dim + 8);
        FLOAT4 temp_3 = vload4(0, input + offset_inp + 4*head_dim + 12);
        #ifdef HEADDIM_LEAVE
        DEAL_HEAD_DIM_NOT_ALIGN
        #endif
        #ifdef SEQLEN_LEAVE
        DEAL_SEQ_LEN_NOT_ALIGN
        #endif
        
        vstore4(temp_0, 0, output_k + offset_k);
        vstore4(temp_1, 0, output_k + offset_k + seq_len_pack_mn);
        vstore4(temp_2, 0, output_k + offset_k + seq_len_pack_mn + seq_len_pack_mn);
        vstore4(temp_3, 0, output_k + offset_k + seq_len_pack_mn + seq_len_pack_mn + seq_len_pack_mn);
        
        // V
        temp_0 = vload4(0, input + offset_inp + 8 * head_dim);
        temp_1 = vload4(0, input + offset_inp + 8 * head_dim + 4);
        temp_2 = vload4(0, input + offset_inp + 8 * head_dim + 8);
        temp_3 = vload4(0, input + offset_inp + 8 * head_dim + 12);
        #ifdef HEADDIM_LEAVE
        DEAL_HEAD_DIM_NOT_ALIGN
        #endif
        #ifdef SEQLEN_LEAVE
        DEAL_SEQ_LEN_NOT_ALIGN
        #endif
        
        vstore4((FLOAT4){temp_0.x, temp_1.x, temp_2.x, temp_3.x}, 0, output_v + offset_v);
        vstore4((FLOAT4){temp_0.y, temp_1.y, temp_2.y, temp_3.y}, 0, output_v + offset_v + head_dim_pack_mn);
        vstore4((FLOAT4){temp_0.z, temp_1.z, temp_2.z, temp_3.z}, 0, output_v + offset_v + head_dim_pack_mn + head_dim_pack_mn);
        vstore4((FLOAT4){temp_0.w, temp_1.w, temp_2.w, temp_3.w}, 0, output_v + offset_v + head_dim_pack_mn + head_dim_pack_mn + head_dim_pack_mn);
    }
}


#ifndef SOFTMAX_LOCAL_SIZE
    #define SOFTMAX_LOCAL_SIZE 512
#endif

// [outside, axis, inside] -> reduce: inside
__kernel void softmax_inside(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT *input, // [batch * mNumHead, ROUND_UP(seqLen, tile), ROUND_UP(seqLen, tile)]
                            __global FLOAT *output,
                            __private const int inside_len,
                            __private const int4 shape // [batch * mNumHead, ROUND_UP(seqLen, tile), ROUND_UP(seqLen, tile)]
                            ) {
    const int inside = get_global_id(0);
    const int axis = get_global_id(1);
    const int outside = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(inside, axis, outside);

    const int offset = (outside * shape.y + axis) * shape.z + 0;

    int lid = get_local_id(0);
    float local sum[SOFTMAX_LOCAL_SIZE];

    /*Compute Max */
    float maxValue = (float)(-FLT_MAX);
    // clip to seq_len
    for (int i=lid; i<inside_len; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, (float)input[offset+ i]);
    }
    sum[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i >>= 1){
        if (lid < i)
            sum[lid] = fmax(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = sum[0];

    /*Compute Exp Sum*/
    float sumValue = 0;
    for (int i=lid; i<inside_len; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp((float)input[offset+ i] - maxValue);
    }
    sum[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i >>= 1){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum[0];

    #ifdef OUTPUT_TRANSPOSE
    const int out_offset = (outside * shape.z + 0) * shape.y + axis;
    #endif
    /*Compute Result */
    for (int i=lid; i<shape.z; i+=SOFTMAX_LOCAL_SIZE) {
        float value = exp((float)input[offset+ i] - maxValue) / sumValue;
        #ifdef OUTPUT_TRANSPOSE
        output[out_offset+ i*shape.y] = value;
        #else
        output[offset+ i] = value;
        #endif
    }
}

// [N X Y4 4] -> [N Y X]
__kernel void trans_3d_buf(__global const FLOAT* input,
                        __global FLOAT* output,
                        __private const int batch,
                        __private const int width,
                        __private const int height
) {
    int b = get_global_id(2);
    
    const int w = get_global_id(0) << 3;
    const int h = get_global_id(1) << 3;
    
    const int inp_offset = (b * width + w) * height + h;
    const int out_offset = (b * height + h) * width + w;

    FLOAT8 value_0 = vload8(0, input+inp_offset);
    FLOAT8 value_1 = vload8(0, input+inp_offset + height);
    FLOAT8 value_2 = vload8(0, input+inp_offset + height + height);
    FLOAT8 value_3 = vload8(0, input+inp_offset + height + height + height);
    FLOAT8 value_4 = vload8(0, input+inp_offset + (height << 2));
    FLOAT8 value_5 = vload8(0, input+inp_offset + height * 5);
    FLOAT8 value_6 = vload8(0, input+inp_offset + height * 6);
    FLOAT8 value_7 = vload8(0, input+inp_offset + height * 7);
    
    vstore8((FLOAT8){value_0.s0, value_1.s0, value_2.s0, value_3.s0, value_4.s0, value_5.s0, value_6.s0, value_7.s0}, 0, output + out_offset);
    vstore8((FLOAT8){value_0.s1, value_1.s1, value_2.s1, value_3.s1, value_4.s1, value_5.s1, value_6.s1, value_7.s1}, 0, output + out_offset + width);
    vstore8((FLOAT8){value_0.s2, value_1.s2, value_2.s2, value_3.s2, value_4.s2, value_5.s2, value_6.s2, value_7.s2}, 0, output + out_offset + width + width);
    vstore8((FLOAT8){value_0.s3, value_1.s3, value_2.s3, value_3.s3, value_4.s3, value_5.s3, value_6.s3, value_7.s3}, 0, output + out_offset + width + width + width);
    vstore8((FLOAT8){value_0.s4, value_1.s4, value_2.s4, value_3.s4, value_4.s4, value_5.s4, value_6.s4, value_7.s4}, 0, output + out_offset + (width << 2));
    vstore8((FLOAT8){value_0.s5, value_1.s5, value_2.s5, value_3.s5, value_4.s5, value_5.s5, value_6.s5, value_7.s5}, 0, output + out_offset + width * 5);
    vstore8((FLOAT8){value_0.s6, value_1.s6, value_2.s6, value_3.s6, value_4.s6, value_5.s6, value_6.s6, value_7.s6}, 0, output + out_offset + width * 6);
    vstore8((FLOAT8){value_0.s7, value_1.s7, value_2.s7, value_3.s7, value_4.s7, value_5.s7, value_6.s7, value_7.s7}, 0, output + out_offset + width * 7);
}

__kernel void clip_transpose_qkv(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input, // [Batch * mNumHead, ROUND_UP(mHeadDim, tile), ROUND_UP(seqLen, tile)]
                              __global FLOAT *output, // [Batch, seqLen/4, mNumHead * mHeadDim, 4]
                              __private const int tile,
                              __private const int seq_len,
                              __private const int seq_len_piece,
                              __private const int head_num,
                              __private const int head_dim,
                              __private const int seq_index
) {
    
    const int sl = get_global_id(0); // seqLen_Piece_4
    const int hd = get_global_id(1); // mHeadDim_4
    const int z = get_global_id(2); // Batch * mNumHead
    DEAL_NON_UNIFORM_DIM3(sl, hd, z);
    
    const int b = z / head_num;
    const int hn = z % head_num;
    
    const int seq_len_4 = (seq_len + 3) / 4;
    
    if(seq_index * seq_len_piece / 4 + sl >= seq_len_4) {
        return;
    }
    const int seq_len_pack = seq_len_piece;//((seq_len + tile - 1) / tile) * tile;
    const int head_dim_pack = ((head_dim + tile - 1) / tile) * tile;
    
    const int offset_inp = ((b * head_num + hn) * head_dim_pack + 4 * hd) * seq_len_pack + 4 * sl;
    
    const int offset_out = (((b * seq_len_4 + seq_index * seq_len_piece / 4 + sl) * head_num + hn) * head_dim + 4 * hd) * 4;
    
    // Q
    FLOAT4 temp_0 = vload4(0, input + offset_inp);
    FLOAT4 temp_1 = vload4(0, input + offset_inp + seq_len_pack);
    FLOAT4 temp_2 = vload4(0, input + offset_inp + 2 * seq_len_pack);
    FLOAT4 temp_3 = vload4(0, input + offset_inp + 3 * seq_len_pack);
    
    vstore4(temp_0, 0, output + offset_out);
    if(4 * hd + 1 > head_dim) return;
    vstore4(temp_1, 0, output + offset_out + 4);
    if(4 * hd + 2 > head_dim) return;
    vstore4(temp_2, 0, output + offset_out + 8);
    if(4 * hd + 3 > head_dim) return;
    vstore4(temp_3, 0, output + offset_out + 12);

}
