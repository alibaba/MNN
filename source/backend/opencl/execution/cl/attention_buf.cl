#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define DEAL_OUTER_SEQLEN_NOT_ALIGN(length) \
    if(4 * sl + 3 >= length) {\
        temp_3 = (FLOAT4)0;\
    }\
    if(4 * sl + 2 >= length) {\
        temp_2 = (FLOAT4)0;\
    }\
    if(4 * sl + 1 >= length) {\
        temp_1 = (FLOAT4)0;\
    }

#define DEAL_INNER_HEADDIM_NOT_ALIGN(length) \
    if(hd * 4 + 3 >= length) {\
        temp_0.w = (FLOAT)0;\
        temp_1.w = (FLOAT)0;\
        temp_2.w = (FLOAT)0;\
        temp_3.w = (FLOAT)0;\
    }\
    if(hd * 4 + 2 >= length) {\
        temp_0.z = (FLOAT)0;\
        temp_1.z = (FLOAT)0;\
        temp_2.z = (FLOAT)0;\
        temp_3.z = (FLOAT)0;\
    }\
    if(hd * 4 + 1 >= length) {\
        temp_0.y = (FLOAT)0;\
        temp_1.y = (FLOAT)0;\
        temp_2.y = (FLOAT)0;\
        temp_3.y = (FLOAT)0;\
    }



__kernel void rearrange_qkv(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input_q, //[batch, seqLenQ/4, headNum, headDim, seqLenQ_4]
                              __global const FLOAT *input_k, // [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4]
                              __global const FLOAT *input_v, // [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4]
                              __global FLOAT *output_q, // [batch*headNum, ROUND_UP(headDim, mTileHDK), ROUND_UP(seqLenQ, mTileQ)]
                              __global FLOAT *output_k, // [batch*headNum/group, ROUND_UP(headDim, mTileHDK), ROUND_UP(seqLenKV, mTileKV)]
                              __global FLOAT *output_v, // [batch*headNum/group, ROUND_UP(seqLenKV, mTileKV), ROUND_UP(headDim, mTileHDN)]
                              __global FLOAT *past_k, // [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4]
                              __global FLOAT *past_v, // [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4]
                              __private const int4 tile, // [mTileQ, mTileKV, mTileHDK, mTileHDN]
                              __private const int4 shape,// [seqLenQ, seqLenKV, headNum, headDim]
                              __private const int4 param // [group, batch]
) {
    const int sl = get_global_id(0); // seqLen/4 : max(seqLenPackQ/4, seqLenPackKV/4)
    const int hd = get_global_id(1); // headDim/4 : max(headDimPackQK/4, headDimPackV/4)
    const int z = get_global_id(2); // batch * headNum
    DEAL_NON_UNIFORM_DIM3(sl, hd, z);
    
    const int seqLenQ = shape.x;
    const int seqLenKV = shape.y;
    const int headNum = shape.z;
    const int headDim = shape.w;
    const int group = param.x;
    const int batch = param.y;

    const int b = z % batch;
    const int hn = z / batch;
    
    const int seqLenQ_4 = (seqLenQ + 3) / 4;
    //const int in_offset_q = (((b * seqLenQ_4 + sl) * headNum + hn) * headDim + 4 * hd) * 4;
    const int in_offset_q = (((b * seqLenQ + sl * 4) * headNum + hn) * headDim + 4 * hd);

    const int seqLenPackQ = ((seqLenQ + tile.x - 1) / tile.x) * tile.x;
    const int headDimPackQK = ((headDim + tile.z - 1) / tile.z) * tile.z;
    const int out_offset_q = (((b * headNum + hn) * headDimPackQK + hd * 4) * seqLenPackQ + sl * 4);
    
    if(sl * 4 < seqLenPackQ && hd * 4 < headDimPackQK) {
        if(sl * 4 >= seqLenQ || hd * 4 >= headDim) {
            vstore4((FLOAT4)0, 0, output_q + out_offset_q);
            vstore4((FLOAT4)0, 0, output_q + out_offset_q + seqLenPackQ);
            vstore4((FLOAT4)0, 0, output_q + out_offset_q + 2 * seqLenPackQ);
            vstore4((FLOAT4)0, 0, output_q + out_offset_q + 3 * seqLenPackQ);
        } else {
            FLOAT4 temp_0 = vload4(0, input_q + in_offset_q);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenQ) ? (FLOAT4)0 : vload4(0, input_q + in_offset_q + headNum*headDim);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenQ) ? (FLOAT4)0 : vload4(0, input_q + in_offset_q + 2*headNum*headDim);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenQ) ? (FLOAT4)0 : vload4(0, input_q + in_offset_q + 3*headNum*headDim);
            #ifdef HEADDIM_LEAVE
            DEAL_INNER_HEADDIM_NOT_ALIGN(headDim)
            #endif
            #ifdef SEQLEN_LEAVE
            DEAL_OUTER_SEQLEN_NOT_ALIGN(seqLenQ)
            #endif
            vstore4((FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0), 0, output_q + out_offset_q);
            vstore4((FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1), 0, output_q + out_offset_q + seqLenPackQ);
            vstore4((FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2), 0, output_q + out_offset_q + 2 * seqLenPackQ);
            vstore4((FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3), 0, output_q + out_offset_q + 3 * seqLenPackQ);
        }
    }
        
    if(hn >= headNum / group) {
        return;
    }
    

    const int seqLenPackKV = ((seqLenKV + tile.y - 1) / tile.y) * tile.y;
    const int headDimPackV = ((headDim + tile.w - 1) / tile.w) * tile.w;
    const int seqLenKV_4 = (seqLenKV + 3) / 4;
    const int in_offset_kv = (((b * seqLenKV + sl*4) * headNum/group + hn) * headDim + 4 * hd);
    
    if(sl * 4 < seqLenPackKV && hd * 4 < headDimPackQK) {
        const int out_offset_k = (((b * headNum/group + hn) * headDimPackQK + hd * 4) * seqLenPackKV + sl * 4);

        if(sl * 4 >= seqLenKV || hd * 4 >= headDim) {
            vstore4((FLOAT4)0, 0, output_k + out_offset_k);
            vstore4((FLOAT4)0, 0, output_k + out_offset_k + seqLenPackKV);
            vstore4((FLOAT4)0, 0, output_k + out_offset_k + 2 * seqLenPackKV);
            vstore4((FLOAT4)0, 0, output_k + out_offset_k + 3 * seqLenPackKV);
        } else {
            FLOAT4 temp_0 = vload4(0, input_k + in_offset_kv);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_k + in_offset_kv + headNum*headDim/group);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_k + in_offset_kv + 2*headNum*headDim/group);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_k + in_offset_kv + 3*headNum*headDim/group);
            #ifdef HEADDIM_LEAVE
            DEAL_INNER_HEADDIM_NOT_ALIGN(headDim)
            #endif
            #ifdef SEQLEN_LEAVE
            DEAL_OUTER_SEQLEN_NOT_ALIGN(seqLenKV)
            #endif
            vstore4((FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0), 0, output_k + out_offset_k);
            vstore4((FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1), 0, output_k + out_offset_k + seqLenPackKV);
            vstore4((FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2), 0, output_k + out_offset_k + 2 * seqLenPackKV);
            vstore4((FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3), 0, output_k + out_offset_k + 3 * seqLenPackKV);
            
            // pastK
            vstore4(temp_0, 0, past_k + in_offset_kv);
            if(sl * 4 + 1 < seqLenKV) {
                vstore4(temp_1, 0, past_k + in_offset_kv + headNum*headDim/group);
            }
            if(sl * 4 + 2 < seqLenKV) {
                vstore4(temp_2, 0, past_k + in_offset_kv + 2*headNum*headDim/group);
            }
            if(sl * 4 + 3 < seqLenKV) {
                vstore4(temp_3, 0, past_k + in_offset_kv + 3*headNum*headDim/group);
            }
        }
        
    }
    
    if(sl * 4 < seqLenPackKV && hd * 4 < headDimPackV) {
        const int out_offset_v = (((b * headNum/group + hn) * seqLenPackKV + sl * 4) * headDimPackV + hd * 4);

        if(sl * 4 >= seqLenKV || hd * 4 >= headDim) {
            vstore4((FLOAT4)0, 0, output_v + out_offset_v);
            vstore4((FLOAT4)0, 0, output_v + out_offset_v + headDimPackV);
            vstore4((FLOAT4)0, 0, output_v + out_offset_v + 2 * headDimPackV);
            vstore4((FLOAT4)0, 0, output_v + out_offset_v + 3 * headDimPackV);
        } else {
            FLOAT4 temp_0 = vload4(0, input_v + in_offset_kv);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_v + in_offset_kv + headNum*headDim/group);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_v + in_offset_kv + 2*headNum*headDim/group);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_v + in_offset_kv + 3*headNum*headDim/group);
            #ifdef HEADDIM_LEAVE
            DEAL_INNER_HEADDIM_NOT_ALIGN(headDim)
            #endif
            #ifdef SEQLEN_LEAVE
            DEAL_OUTER_SEQLEN_NOT_ALIGN(seqLenKV)
            #endif
            vstore4(temp_0, 0, output_v + out_offset_v);
            vstore4(temp_1, 0, output_v + out_offset_v + headDimPackV);
            vstore4(temp_2, 0, output_v + out_offset_v + 2 * headDimPackV);
            vstore4(temp_3, 0, output_v + out_offset_v + 3 * headDimPackV);
            
            // pastV
            vstore4(temp_0, 0, past_v + in_offset_kv);
            if(sl * 4 + 1 < seqLenKV) {
                vstore4(temp_1, 0, past_v + in_offset_kv + headNum*headDim/group);
            }
            if(sl * 4 + 2 < seqLenKV) {
                vstore4(temp_2, 0, past_v + in_offset_kv + 2*headNum*headDim/group);
            }
            if(sl * 4 + 3 < seqLenKV) {
                vstore4(temp_3, 0, past_v + in_offset_kv + 3*headNum*headDim/group);
            }
        }
        
    }
}

#ifndef MASK_DTYPE
#define MASK_DTYPE FLOAT
#define MASK_DTYPE4 FLOAT4
#endif
__kernel void rearrange_mask(GLOBAL_SIZE_3_DIMS
        __global const MASK_DTYPE *input_mask, // [batch, 1, seqLenQ, seqLenKV, 4]
        __global MASK_DTYPE *output_mask, // [batch, ROUND_UP(seqLenQ, mTileQ), ROUND_UP(seqLenKV, mTileKV)]
        const int4 shape // [seqLenQ, seqLenKV, mTileQ, mTileKV]
) {
    const int sl = get_global_id(0); // seqLen_4
    const int sl_kv = get_global_id(1); // seqLenKV_4
    const int b = get_global_id(2); // Batch
    DEAL_NON_UNIFORM_DIM3(sl, sl_kv, b);
        
    const int seq_len_pack = ((shape.x + shape.z - 1) / shape.z) * shape.z;
    const int seq_len_kv_pack = ((shape.y + shape.w - 1) / shape.w) * shape.w;

    int in_offset = ((b * shape.x + sl * 4) * shape.y + sl_kv * 4);
    int out_offset = (b * seq_len_pack + sl * 4) * seq_len_kv_pack + sl_kv * 4;

    if(sl * 4 >= shape.x || sl_kv * 4 >= shape.y) {
        vstore4((MASK_DTYPE4)0, 0, output_mask + out_offset);
        vstore4((MASK_DTYPE4)0, 0, output_mask + out_offset + seq_len_kv_pack);
        vstore4((MASK_DTYPE4)0, 0, output_mask + out_offset + seq_len_kv_pack * 2);
        vstore4((MASK_DTYPE4)0, 0, output_mask + out_offset + seq_len_kv_pack * 3);
    } else {
        int y_down_align4 = (shape.y / 4 * 4);
        MASK_DTYPE4 temp_0, temp_1, temp_2, temp_3;
        
        if(sl_kv * 4 < y_down_align4) {
            temp_0 = vload4(0, input_mask + in_offset);
            temp_1 = (sl * 4 + 1 >= shape.x) ? (MASK_DTYPE4)0 : vload4(0, input_mask + in_offset + shape.y);
            temp_2 = (sl * 4 + 2 >= shape.x) ? (MASK_DTYPE4)0 : vload4(0, input_mask + in_offset + shape.y * 2);
            temp_3 = (sl * 4 + 3 >= shape.x) ? (MASK_DTYPE4)0 : vload4(0, input_mask + in_offset + shape.y * 3);
        } else if(sl_kv * 4 + 1 == shape.y){
            temp_0 = (MASK_DTYPE4)(input_mask[in_offset], 0, 0, 0);
            temp_1 = (sl * 4 + 1 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y], 0, 0, 0);//vload4(0, input_mask + in_offset + shape.y);
            temp_2 = (sl * 4 + 2 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*2], 0, 0, 0);//vload4(0, input_mask + in_offset + shape.y * 2);
            temp_3 = (sl * 4 + 3 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*3], 0, 0, 0);//vload4(0, input_mask + in_offset + shape.y * 3);
        } else if(sl_kv * 4 + 2 == shape.y){
            temp_0 = (MASK_DTYPE4)(input_mask[in_offset], input_mask[in_offset+1], 0, 0);
            temp_1 = (sl * 4 + 1 >= shape.x) ? (MASK_DTYPE4)0 : (FLOAT4)(input_mask[in_offset + shape.y], input_mask[in_offset + shape.y + 1], 0, 0);//vload4(0, input_mask + in_offset + shape.y);
            temp_2 = (sl * 4 + 2 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*2], input_mask[in_offset + shape.y*2 + 1], 0, 0);//vload4(0, input_mask + in_offset + shape.y * 2);
            temp_3 = (sl * 4 + 3 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*3], input_mask[in_offset + shape.y*3 + 1], 0, 0);//vload4(0, input_mask + in_offset + shape.y * 3);
        } else if(sl_kv * 4 + 3 == shape.y){
            temp_0 = (MASK_DTYPE4)(input_mask[in_offset], input_mask[in_offset+1], input_mask[in_offset+2], 0);
            temp_1 = (sl * 4 + 1 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y], input_mask[in_offset + shape.y + 1], input_mask[in_offset + shape.y + 2], 0);//vload4(0, input_mask + in_offset + shape.y);
            temp_2 = (sl * 4 + 2 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*2], input_mask[in_offset + shape.y*2 + 1], input_mask[in_offset + shape.y*2 + 2], 0);//vload4(0, input_mask + in_offset + shape.y * 2);
            temp_3 = (sl * 4 + 3 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*3], input_mask[in_offset + shape.y*3 + 1], input_mask[in_offset + shape.y*3 + 2], 0);//vload4(0, input_mask + in_offset + shape.y * 3);
        }

        vstore4(temp_0, 0, output_mask + out_offset);
        vstore4(temp_1, 0, output_mask + out_offset + seq_len_kv_pack);
        vstore4(temp_2, 0, output_mask + out_offset + 2 * seq_len_kv_pack);
        vstore4(temp_3, 0, output_mask + out_offset + 3 * seq_len_kv_pack);
    }

}

__kernel void qkv_transpose_output(GLOBAL_SIZE_3_DIMS
          __global const FLOAT *input, // [Batch * mNumHead, ROUND_UP(mHeadDim, mTileHDN), ROUND_UP(seqLen, mTileQ)]
          __global FLOAT *output, // [Batch, seqLen/4, mNumHead， mHeadDim, 4]
          __private const int tile_q,
          __private const int tile_hdn,
          __private const int seq_len,
          __private const int head_num,
          __private const int head_dim
) {
    
    const int sl = get_global_id(0); // seqLen_4
    const int hd = get_global_id(1); // mHeadDim_4
    const int z = get_global_id(2); // Batch * mNumHead
    DEAL_NON_UNIFORM_DIM3(sl, hd, z);
    
    const int b = z / head_num;
    const int hn = z % head_num;
        
    const int seq_len_pack = ((seq_len + tile_q - 1) / tile_q) * tile_q;
    const int head_dim_pack = ((head_dim + tile_hdn - 1) / tile_hdn) * tile_hdn;
    
    const int offset_inp = ((b * head_num + hn) * head_dim_pack + 4 * hd) * seq_len_pack + 4 * sl;
    
    const int offset_out = (((b * seq_len + sl*4) * head_num + hn) * head_dim + 4 * hd);
    
    // Q
    FLOAT4 temp_0 = vload4(0, input + offset_inp);
    FLOAT4 temp_1 = vload4(0, input + offset_inp + seq_len_pack);
    FLOAT4 temp_2 = vload4(0, input + offset_inp + 2 * seq_len_pack);
    FLOAT4 temp_3 = vload4(0, input + offset_inp + 3 * seq_len_pack);
    
    vstore4((FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0), 0, output + offset_out);
    if(4 * sl + 1 >= seq_len) return;
    vstore4((FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1), 0, output + offset_out + head_num*head_dim);
    if(4 * sl + 2 >= seq_len) return;
    vstore4((FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2), 0, output + offset_out + 2*head_num*head_dim);
    if(4 * sl + 3 >= seq_len) return;
    vstore4((FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3), 0, output + offset_out + 3*head_num*head_dim);

}

#ifndef NUMHEAD_GROUP_SIZE
#define NUMHEAD_GROUP_SIZE 1
#endif

__kernel void matmul_qk_div_mask(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input0, // query [1 query_seq_len head_num head_dim]
                              __global const FLOAT *input1, // key [1 key_seq_len head_num head_dim]
                              __global FLOAT *output, // prefill [1 head_num query_seq_len key_seq_len]   decode[1 head_num key_seq_len/4 4]
                              __global FLOAT *past_key, // [1 max_length head_num head_dim]
                              #ifdef ADD_MASK
                              __global const FLOAT* mask,
                              #else
                              __global const int* mask, // [1 1 query_seq_len key_seq_len]
                              #endif
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int key_seq_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // key_seq_len
    const int y = get_global_id(1); // query_seq_len for prefill   1 for decode
    const int z = get_global_id(2); // head_num
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    int x4 = x << 2;
    int y4 = y << 2;
    int zin = z / NUMHEAD_GROUP_SIZE;
    __global const FLOAT *A_offset = input0 + (y4 * head_num + z) * head_dim;
    __global FLOAT *Pastkey_offset = past_key + (x4 * kv_head_num + zin) * head_dim;
    int strideA = head_num * head_dim;
    int strideB = kv_head_num * head_dim;
#ifdef OPENCL_PREFILL_ATTENTION
    __global const FLOAT *B_offset = input1 + (x4 * kv_head_num + zin) * head_dim;
    int output_offset = (z * query_seq_len + y4) * key_seq_len + x4;
    float4 out0 = 0;
    float4 out1 = 0;
    float4 out2 = 0;
    float4 out3 = 0;
    
    bool A1_enable = y4 + 1 < query_seq_len;
    bool A2_enable = y4 + 2 < query_seq_len;
    bool A3_enable = y4 + 3 < query_seq_len;
    
    bool B1_enable = x4 + 1 < key_seq_len;
    bool B2_enable = x4 + 2 < key_seq_len;
    bool B3_enable = x4 + 3 < key_seq_len;
    
    const int head_dim4 = (head_dim + 3) / 4;
    #ifdef HEADDIM_LEAVE
    for(int i = 0; i < head_dim4 - 1; ++i){
        float4 A0 = convert_float4(vload4(i, A_offset));
        float4 A1 = A1_enable ? convert_float4(vload4(i, A_offset + strideA)) : (float4)0;
        float4 A2 = A2_enable ? convert_float4(vload4(i, A_offset + strideA + strideA)) : (float4)0;
        float4 A3 = A3_enable ? convert_float4(vload4(i, A_offset + strideA + strideA + strideA)) : (float4)0;
        float4 B0 = convert_float4(vload4(i, B_offset));
        float4 B1 = B1_enable ? convert_float4(vload4(i, B_offset + strideB)) : (float4)0;
        float4 B2 = B2_enable ? convert_float4(vload4(i, B_offset + strideB + strideB)) : (float4)0;
        float4 B3 = B3_enable ? convert_float4(vload4(i, B_offset + strideB + strideB + strideB)) : (float4)0;
        
        out0.x += dot(A0, B0);
        out0.y += dot(A0, B1);
        out0.z += dot(A0, B2);
        out0.w += dot(A0, B3);
        
        out1.x += dot(A1, B0);
        out1.y += dot(A1, B1);
        out1.z += dot(A1, B2);
        out1.w += dot(A1, B3);
        
        out2.x += dot(A2, B0);
        out2.y += dot(A2, B1);
        out2.z += dot(A2, B2);
        out2.w += dot(A2, B3);
        
        out3.x += dot(A3, B0);
        out3.y += dot(A3, B1);
        out3.z += dot(A3, B2);
        out3.w += dot(A3, B3);
        
        vstore4(CONVERT_FLOAT4(B0), i, Pastkey_offset);
        vstore4(CONVERT_FLOAT4(B1), i, Pastkey_offset + strideB);
        vstore4(CONVERT_FLOAT4(B2), i, Pastkey_offset + strideB + strideB);
        vstore4(CONVERT_FLOAT4(B3), i, Pastkey_offset + strideB + strideB + strideB);
    }
    for(int i = (head_dim4 - 1) * 4; i < head_dim; ++i){
        float A0 = A_offset[i];
        float A1 = A1_enable ? A_offset[i + strideA] : 0;
        float A2 = A2_enable ? A_offset[i + strideA + strideA] : 0;
        float A3 = A3_enable ? A_offset[i + strideA + strideA + strideA] : 0;
        float B0 = B_offset[i];
        float B1 = B1_enable ? B_offset[i + strideB] : 0;
        float B2 = B2_enable ? B_offset[i + strideB + strideB] : 0;
        float B3 = B3_enable ? B_offset[i + strideB + strideB + strideB] : 0;
        
        out0.x += A0 * B0;
        out0.y += A0 * B1;
        out0.z += A0 * B2;
        out0.w += A0 * B3;
        
        out1.x += A1 * B0;
        out1.y += A1 * B1;
        out1.z += A1 * B2;
        out1.w += A1 * B3
        
        out2.x += A2 * B0;
        out2.y += A2 * B1;
        out2.z += A2 * B2;
        out2.w += A2 * B3;
        
        out3.x += A3 * B0;
        out3.y += A3 * B1;
        out3.z += A3 * B2;
        out3.w += A3 * B3;
        
        Pastkey_offset[i] = (FLOAT)B0;
        Pastkey_offset[i + strideB] = (FLOAT)B1;
        Pastkey_offset[i + strideB + strideB] = (FLOAT)B2;
        Pastkey_offset[i + strideB + strideB + strideB] = (FLOAT)B3;
    }
    #else
    for(int i = 0; i < head_dim4; ++i){
        float4 A0 = convert_float4(vload4(i, A_offset));
        float4 A1 = A1_enable ? convert_float4(vload4(i, A_offset + strideA)) : (float4)0;
        float4 A2 = A2_enable ? convert_float4(vload4(i, A_offset + strideA + strideA)) : (float4)0;
        float4 A3 = A3_enable ? convert_float4(vload4(i, A_offset + strideA + strideA + strideA)) : (float4)0;
        float4 B0 = convert_float4(vload4(i, B_offset));
        float4 B1 = B1_enable ? convert_float4(vload4(i, B_offset + strideB)) : (float4)0;
        float4 B2 = B2_enable ? convert_float4(vload4(i, B_offset + strideB + strideB)) : (float4)0;
        float4 B3 = B3_enable ? convert_float4(vload4(i, B_offset + strideB + strideB + strideB)) : (float4)0;
        
        out0.x += dot(A0, B0);
        out0.y += dot(A0, B1);
        out0.z += dot(A0, B2);
        out0.w += dot(A0, B3);
        
        out1.x += dot(A1, B0);
        out1.y += dot(A1, B1);
        out1.z += dot(A1, B2);
        out1.w += dot(A1, B3);
        
        out2.x += dot(A2, B0);
        out2.y += dot(A2, B1);
        out2.z += dot(A2, B2);
        out2.w += dot(A2, B3);
        
        out3.x += dot(A3, B0);
        out3.y += dot(A3, B1);
        out3.z += dot(A3, B2);
        out3.w += dot(A3, B3);
        
        vstore4(CONVERT_FLOAT4(B0), i, Pastkey_offset);
        vstore4(CONVERT_FLOAT4(B1), i, Pastkey_offset + strideB);
        vstore4(CONVERT_FLOAT4(B2), i, Pastkey_offset + strideB + strideB);
        vstore4(CONVERT_FLOAT4(B3), i, Pastkey_offset + strideB + strideB + strideB);
    }
    #endif
    out0 *= (float4)scale;
    out1 *= (float4)scale;
    out2 *= (float4)scale;
    out3 *= (float4)scale;
    float4 mask0 = convert_float4(vload4(0, mask + y4 * key_seq_len + x4));
    float4 mask1 = convert_float4(vload4(0, mask + (y4 + 1) * key_seq_len + x4));
    float4 mask2 = convert_float4(vload4(0, mask + (y4 + 2) * key_seq_len + x4));
    float4 mask3 = convert_float4(vload4(0, mask + (y4 + 3) * key_seq_len + x4));
    #ifdef ADD_MASK
    out0 += mask0;
    out1 += mask1;
    out2 += mask2;
    out3 += mask3;
    #else
    out0 = (mask0 == (float4)0) ? (float4)(-FLT_MAX) : out0;
    out1 = (mask1 == (float4)0) ? (float4)(-FLT_MAX) : out1;
    out2 = (mask2 == (float4)0) ? (float4)(-FLT_MAX) : out2;
    out3 = (mask3 == (float4)0) ? (float4)(-FLT_MAX) : out3;
    #endif
    if(B3_enable){
        vstore4(CONVERT_FLOAT4(out0), 0, output + output_offset);
        if(!A1_enable) return;
        output_offset += key_seq_len;
        vstore4(CONVERT_FLOAT4(out1), 0, output + output_offset);
        if(!A2_enable) return;
        output_offset += key_seq_len;
        vstore4(CONVERT_FLOAT4(out2), 0, output + output_offset);
        if(!A3_enable) return;
        output_offset += key_seq_len;
        vstore4(CONVERT_FLOAT4(out3), 0, output + output_offset);
    } else if(B2_enable){
        vstore3(CONVERT_FLOAT3((float3)(out0.x, out0.y, out0.z)), 0, output + output_offset);
        if(!A1_enable) return;
        output_offset += key_seq_len;
        vstore3(CONVERT_FLOAT3((float3)(out1.x, out1.y, out1.z)), 0, output + output_offset);
        if(!A2_enable) return;
        output_offset += key_seq_len;
        vstore3(CONVERT_FLOAT3((float3)(out2.x, out2.y, out2.z)), 0, output + output_offset);
        if(!A3_enable) return;
        output_offset += key_seq_len;
        vstore3(CONVERT_FLOAT3((float3)(out3.x, out3.y, out3.z)), 0, output + output_offset);
    } else if(B1_enable){
        vstore2(CONVERT_FLOAT2((float2)(out0.x, out0.y)), 0, output + output_offset);
        if(!A1_enable) return;
        output_offset += key_seq_len;
        vstore2(CONVERT_FLOAT2((float2)(out1.x, out1.y)), 0, output + output_offset);
        if(!A2_enable) return;
        output_offset += key_seq_len;
        vstore2(CONVERT_FLOAT2((float2)(out2.x, out2.y)), 0, output + output_offset);
        if(!A3_enable) return;
        output_offset += key_seq_len;
        vstore2(CONVERT_FLOAT2((float2)(out3.x, out3.y)), 0, output + output_offset);
    } else {
        output[output_offset] = out0.x;
        if(!A1_enable) return;
        output[output_offset + key_seq_len] = out1.x;
        if(!A2_enable) return;
        output[output_offset + key_seq_len + key_seq_len] = out2.x;
        if(!A3_enable) return;
        output[output_offset + key_seq_len + key_seq_len + key_seq_len] = out3.x;
    }
#else
    float4 out = 0;
    const int head_dim4 = (head_dim + 3) / 4;
    int key_seq_len4 = (key_seq_len + 3) / 4;
    #ifdef HEADDIM_LEAVE
    for(int i = 0; i < head_dim4 - 1; ++i){
        float4 A = convert_float4(vload4(i, A_offset));
        float4 B0 = convert_float4(vload4(i, Pastkey_offset));
        float4 B1 = convert_float4(vload4(i, Pastkey_offset + strideB));
        float4 B2 = convert_float4(vload4(i, Pastkey_offset + strideB + strideB));
        float4 B3 = convert_float4(vload4(i, Pastkey_offset + strideB + strideB + strideB));
    
        out.x += dot(A, B0);
        out.y += dot(A, B1);
        out.z += dot(A, B2);
        out.w += dot(A, B3);
    }
    for(int i = (head_dim4 - 1) * 4; i < head_dim; ++i){
        float A = A_offset[i];
        float B0 = Pastkey_offset[i];
        float B1 = Pastkey_offset[i + strideB];
        float B2 = Pastkey_offset[i + strideB + strideB];
        float B3 = Pastkey_offset[i + strideB + strideB];
        out.x += A * B0;
        out.y += A * B1;
        out.z += A * B2;
        out.w += A * B3;
    }
    #else
    for(int i = 0; i < head_dim4; ++i){
        float4 A = convert_float4(vload4(i, A_offset));
        float4 B0 = convert_float4(vload4(i, Pastkey_offset));
        float4 B1 = convert_float4(vload4(i, Pastkey_offset + strideB));
        float4 B2 = convert_float4(vload4(i, Pastkey_offset + strideB + strideB));
        float4 B3 = convert_float4(vload4(i, Pastkey_offset + strideB + strideB + strideB));
    
        out.x += dot(A, B0);
        out.y += dot(A, B1);
        out.z += dot(A, B2);
        out.w += dot(A, B3);
    }
    #endif
    int remain = key_seq_len - x4;
    if(x == key_seq_len4 - 1){
        __global const FLOAT *B_offset = input1 + zin * head_dim;
        Pastkey_offset += (remain - 1) * strideB;
        float tmp = 0;
        #ifdef HEADDIM_LEAVE
        for(int i = 0; i < head_dim4 - 1; ++i){
            float4 A = convert_float4(vload4(i, A_offset));
            float4 B = convert_float4(vload4(i, B_offset));
        
            tmp += dot(A, B);
            vstore4(CONVERT_FLOAT4(B), i, Pastkey_offset);
        }
        for(int i = (head_dim4 - 1) * 4; i < head_dim; ++i){
            float A = A_offset[i];
            float B = B_offset[i];
            tmp += A * B;
            Pastkey_offset[i] = B;
        }
        #else
        for(int i = 0; i < head_dim4; ++i){
            float4 A = convert_float4(vload4(i, A_offset));
            float4 B = convert_float4(vload4(i, B_offset));
        
            tmp += dot(A, B);
            vstore4(CONVERT_FLOAT4(B), i, Pastkey_offset);
        }
        #endif
        float *out_ptr = (float*)&out;
        out_ptr[remain - 1] = tmp;
    }
    out *= (float4)scale;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out), 0, output + z * key_seq_len + x4);
    } else if (remain >= 3){
        vstore3(CONVERT_FLOAT3((float3)(out.x, out.y, out.z)), 0, output + z * key_seq_len + x4);
    } else if (remain >= 2){
        vstore2(CONVERT_FLOAT2((float2)(out.x, out.y)), 0, output + z * key_seq_len + x4);
    } else {
        output[z * key_seq_len + x4] = out.x;
    }
#endif
}

__kernel void matmul_qkv(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input0, // qk prefill [1 head_num qk_seq_len value_seq_len]   decode[1 head_num value_seq_len]
                              __global const FLOAT *input1, // [1 value_seq_len head_num head_dim]
                              __global FLOAT *output, // [1 qk_seq_len head_num head_dim]
                              __global FLOAT *past_value, // [1 value_seq_len head_num head_dim]
                              __private const int qk_seq_len,
                              __private const int value_seq_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim << 2
    const int y = get_global_id(1); // head_num
    const int z = get_global_id(2); // prefill qk_seq_len decode 1
    
    const int x4 = x << 2;
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int yin = y / NUMHEAD_GROUP_SIZE;
#ifdef OPENCL_PREFILL_ATTENTION
    int z4 = z << 2;
    int value_seq_len4 = (value_seq_len + 3) / 4;
    int loop_end = max(value_seq_len4 - 1, 0);
    const int stride = kv_head_num * head_dim;
    __global const FLOAT *A_offset = input0 + (y * qk_seq_len + z4) * value_seq_len;
    __global const FLOAT *B_offset = input1 + yin * head_dim + x4;
    __global FLOAT *Pastvalue_offset = past_value + yin * head_dim + x4;
    COMPUTE_FLOAT4 out0 = 0;
    COMPUTE_FLOAT4 out1 = 0;
    COMPUTE_FLOAT4 out2 = 0;
    COMPUTE_FLOAT4 out3 = 0;
    
    for(int i = 0; i < loop_end; ++i){
        int index = i << 2;
        COMPUTE_FLOAT4 A0 = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset));
        COMPUTE_FLOAT4 A1 = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset + value_seq_len));
        COMPUTE_FLOAT4 A2 = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset + value_seq_len + value_seq_len));
        COMPUTE_FLOAT4 A3 = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset + value_seq_len + value_seq_len + value_seq_len));
        COMPUTE_FLOAT4 B0 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_offset + (index + 0) * stride));
        COMPUTE_FLOAT4 B1 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_offset + (index + 1) * stride));
        COMPUTE_FLOAT4 B2 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_offset + (index + 2) * stride));
        COMPUTE_FLOAT4 B3 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_offset + (index + 3) * stride));
        
        out0 = mad(B0, (COMPUTE_FLOAT4)A0.x, out0);
        out0 = mad(B1, (COMPUTE_FLOAT4)A0.y, out0);
        out0 = mad(B2, (COMPUTE_FLOAT4)A0.z, out0);
        out0 = mad(B3, (COMPUTE_FLOAT4)A0.w, out0);
        
        out1 = mad(B0, (COMPUTE_FLOAT4)A1.x, out1);
        out1 = mad(B1, (COMPUTE_FLOAT4)A1.y, out1);
        out1 = mad(B2, (COMPUTE_FLOAT4)A1.z, out1);
        out1 = mad(B3, (COMPUTE_FLOAT4)A1.w, out1);
        
        out2 = mad(B0, (COMPUTE_FLOAT4)A2.x, out2);
        out2 = mad(B1, (COMPUTE_FLOAT4)A2.y, out2);
        out2 = mad(B2, (COMPUTE_FLOAT4)A2.z, out2);
        out2 = mad(B3, (COMPUTE_FLOAT4)A2.w, out2);
        
        out3 = mad(B0, (COMPUTE_FLOAT4)A3.x, out3);
        out3 = mad(B1, (COMPUTE_FLOAT4)A3.y, out3);
        out3 = mad(B2, (COMPUTE_FLOAT4)A3.z, out3);
        out3 = mad(B3, (COMPUTE_FLOAT4)A3.w, out3);
        vstore4(CONVERT_FLOAT4(B0), 0, Pastvalue_offset + (index + 0) * stride);
        vstore4(CONVERT_FLOAT4(B1), 0, Pastvalue_offset + (index + 1) * stride);
        vstore4(CONVERT_FLOAT4(B2), 0, Pastvalue_offset + (index + 2) * stride);
        vstore4(CONVERT_FLOAT4(B3), 0, Pastvalue_offset + (index + 3) * stride);
    }
    for(int i = loop_end << 2; i < value_seq_len; ++i){
        COMPUTE_FLOAT A0 = A_offset[i];
        COMPUTE_FLOAT A1 = A_offset[i + value_seq_len];
        COMPUTE_FLOAT A2 = A_offset[i + value_seq_len + value_seq_len];
        COMPUTE_FLOAT A3 = A_offset[i + value_seq_len + value_seq_len + value_seq_len];
        COMPUTE_FLOAT4 B = CONVERT_COMPUTE_FLOAT4(vload4(0, B_offset + i * stride));
        
        out0 = mad(B, (COMPUTE_FLOAT4)A0, out0);
        out1 = mad(B, (COMPUTE_FLOAT4)A1, out1);
        out2 = mad(B, (COMPUTE_FLOAT4)A2, out2);
        out3 = mad(B, (COMPUTE_FLOAT4)A3, out3);
        vstore4(CONVERT_FLOAT4(B), 0, Pastvalue_offset + i * stride);
    }
    
    #ifdef HEADDIM_LEAVE
    int remain = head_dim - x4;
    int output_offset = (z4 * head_num + y) * head_dim + x4;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out0), 0, output + output_offset);
    } else if(remain == 3){
        vstore3(CONVERT_FLOAT3((COMPUTE_FLOAT3)(out0.x, out0.y, out0.z)), 0, output + output_offset);
    } else if(remain == 2){
        vstore2(CONVERT_FLOAT2((COMPUTE_FLOAT3)(out0.x, out0.y)), 0, output + output_offset);
    } else{
        output[output_offset] = out0.x;
    }
    if(z4 + 1 >= qk_seq_len) return;
    output_offset += head_num * head_dim;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out1), 0, output + output_offset);
    } else if(remain == 3){
        vstore3(CONVERT_FLOAT3((COMPUTE_FLOAT3)(out1.x, out1.y, out1.z)), 0, output + output_offset);
    } else if(remain == 2){
        vstore2(CONVERT_FLOAT2((COMPUTE_FLOAT3)(out1.x, out1.y)), 0, output + output_offset);
    } else{
        output[output_offset] = out1.x;
    }
    if(z4 + 2 >= qk_seq_len) return;
    output_offset += head_num * head_dim;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out2), 0, output + output_offset);
    } else if(remain == 3){
        vstore3(CONVERT_FLOAT3((COMPUTE_FLOAT3)(out2.x, out2.y, out2.z)), 0, output + output_offset);
    } else if(remain == 2){
        vstore2(CONVERT_FLOAT2((COMPUTE_FLOAT3)(out2.x, out2.y)), 0, output + output_offset);
    } else{
        output[output_offset] = out2.x;
    }
    if(z4 + 3 >= qk_seq_len) return;
    output_offset += head_num * head_dim;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out3), 0, output + output_offset);
    } else if(remain == 3){
        vstore3(CONVERT_FLOAT3((COMPUTE_FLOAT3)(out3.x, out3.y, out3.z)), 0, output + output_offset);
    } else if(remain == 2){
        vstore2(CONVERT_FLOAT2((COMPUTE_FLOAT3)(out3.x, out3.y)), 0, output + output_offset);
    } else{
        output[(x * head_num + y) * head_dim + z4] = out3.x;
    }
    #else
    int output_offset = (z4 * head_num + y) * head_dim + x4;
    vstore4(CONVERT_FLOAT4(out0), 0, output + output_offset);
    if(z4 + 1 >= qk_seq_len) return;
    output_offset += head_num * head_dim;
    vstore4(CONVERT_FLOAT4(out1), 0, output + output_offset);
    if(z4 + 2 >= qk_seq_len) return;
    output_offset += head_num * head_dim;
    vstore4(CONVERT_FLOAT4(out2), 0, output + output_offset);
    if(z4 + 3 >= qk_seq_len) return;
    output_offset += head_num * head_dim;
    vstore4(CONVERT_FLOAT4(out3), 0, output + output_offset);
    #endif

#else
    int value_seq_len4 = (value_seq_len - 1 + 3) / 4;
    int loop_end = max(value_seq_len4 - 1, 0);
    const int stride = kv_head_num * head_dim;
    __global const FLOAT *A_offset = input0 + y * value_seq_len;
    __global const FLOAT *B_offset = input1 + yin * head_dim + x4;
    __global FLOAT *Pastvalue_offset = past_value + yin * head_dim + x4;
    COMPUTE_FLOAT4 out = 0;
    
    for(int i = 0; i < loop_end; i++){
        int index = i << 2;
        COMPUTE_FLOAT4 A = CONVERT_COMPUTE_FLOAT4(vload4(i, A_offset));
        COMPUTE_FLOAT4 B0 = CONVERT_COMPUTE_FLOAT4(vload4(0, Pastvalue_offset + (index + 0) * stride));
        COMPUTE_FLOAT4 B1 = CONVERT_COMPUTE_FLOAT4(vload4(0, Pastvalue_offset + (index + 1) * stride));
        COMPUTE_FLOAT4 B2 = CONVERT_COMPUTE_FLOAT4(vload4(0, Pastvalue_offset + (index + 2) * stride));
        COMPUTE_FLOAT4 B3 = CONVERT_COMPUTE_FLOAT4(vload4(0, Pastvalue_offset + (index + 3) * stride));
        
        out = mad(B0, (COMPUTE_FLOAT4)A.x, out);
        out = mad(B1, (COMPUTE_FLOAT4)A.y, out);
        out = mad(B2, (COMPUTE_FLOAT4)A.z, out);
        out = mad(B3, (COMPUTE_FLOAT4)A.w, out);
    }
    for(int i = loop_end << 2; i < value_seq_len - 1; i++){
        COMPUTE_FLOAT A = A_offset[i];
        COMPUTE_FLOAT4 B = CONVERT_COMPUTE_FLOAT4(vload4(0, Pastvalue_offset + i * stride));
        
        out = mad(B, (COMPUTE_FLOAT4)A, out);
    }
    COMPUTE_FLOAT A = A_offset[value_seq_len - 1];
    COMPUTE_FLOAT4 B = CONVERT_COMPUTE_FLOAT4(vload4(0, B_offset));
    out = mad(B, (COMPUTE_FLOAT4)A, out);
    
    #ifdef HEADDIM_LEAVE
    int remain = head_dim - x4;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out), 0, output + y * head_dim + x4);
        vstore4(CONVERT_FLOAT4(B), 0, Pastvalue_offset + (value_seq_len - 1) * stride);
    } else if(remain == 3){
        vstore3(CONVERT_FLOAT3((COMPUTE_FLOAT3)(out.x, out.y, out.z)), 0, output + y * head_dim + x4);
        vstore3(CONVERT_FLOAT4((COMPUTE_FLOAT3)(B.x, B.y, B.z)), 0, Pastvalue_offset + (value_seq_len - 1) * stride);
    } else if(remain == 2){
        vstore2(CONVERT_FLOAT2((COMPUTE_FLOAT3)(out.x, out.y)), 0, output + y * head_dim + x4);
        vstore2(CONVERT_FLOAT4((COMPUTE_FLOAT3)(B.x, B.y)), 0, Pastvalue_offset + (value_seq_len - 1) * stride);
    } else{
        output[(x * head_num + y) * head_dim + x4] = out.x;
        Pastvalue_offset[(value_seq_len - 1) * stride] = B.x;
    }
    #else
    vstore4(CONVERT_FLOAT4(B), 0, Pastvalue_offset + (value_seq_len - 1) * stride);
    vstore4(CONVERT_FLOAT4(out), 0, output + y * head_dim + x4);
    #endif
    
#endif
}

