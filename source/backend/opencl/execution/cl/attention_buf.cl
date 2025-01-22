#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define GLOBAL_SIZE_2_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
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
                              __global FLOAT *past_k, // [batch, headNum/group, headDim, seqLenKV_4]
                              __global FLOAT *past_v, // [batch, headNum/group, seqLenKV_4, headDim]
                              __private const int4 tile, // [mTileQ, mTileKV, mTileHDK, mTileHDN]
                              __private const int4 shape,// [seqLenQ, seqLenKV, headNum, headDim]
                              __private const int4 param, // [group, batch, max_len, past_len]
                              __private const int maxLenKV
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
    const int past_offset_k = (((b * headNum/group + hn) * headDim + hd * 4) * maxLenKV + sl*4);
    const int past_offset_v = (((b * headNum/group + hn) * maxLenKV + sl*4) * headDim + 4 * hd);
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
            FLOAT4 key0 = (FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0);
            FLOAT4 key1 = (FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1);
            FLOAT4 key2 = (FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2);
            FLOAT4 key3 = (FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3);
            vstore4(key0, 0, output_k + out_offset_k);
            vstore4(key1, 0, output_k + out_offset_k + seqLenPackKV);
            vstore4(key2, 0, output_k + out_offset_k + 2 * seqLenPackKV);
            vstore4(key3, 0, output_k + out_offset_k + 3 * seqLenPackKV);
            
            // pastK
            vstore4(key0, 0, past_k + past_offset_k);
            vstore4(key1, 0, past_k + past_offset_k + maxLenKV);
            vstore4(key2, 0, past_k + past_offset_k + 2*maxLenKV);
            vstore4(key3, 0, past_k + past_offset_k + 3*maxLenKV);
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
            vstore4(temp_0, 0, past_v + past_offset_v);
            vstore4(temp_1, 0, past_v + past_offset_v + headDim);
            vstore4(temp_2, 0, past_v + past_offset_v + 2*headDim);
            vstore4(temp_3, 0, past_v + past_offset_v + 3*headDim);
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

__kernel void rearrange_q(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [1 query_seq_len head_num head_dim]
                              __global FLOAT *query_tmp, // [1 head_num head_dim key_seq_len4]
                              __private const int seq_len,
                              __private const int head_dim,
                              __private const int head_num) {
                                  
    const int x = get_global_id(0); // query_seq_len
    const int y = get_global_id(1); // head_dim
    const int z = get_global_id(2); // head_num
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int x4 = x << 2;
    const int y4 = y << 2;
    const int seq_len4 = (seq_len + 3) / 4 * 4;;
    const int stride = head_num * head_dim;
    int query_offset = (x4 * head_num + z) * head_dim + y4;
    FLOAT4 query_vec0 = vload4(0, query + query_offset); query_offset += stride;
    FLOAT4 query_vec1 = (x4 + 1 >= seq_len) ? (FLOAT4)0 : vload4(0, query + query_offset); query_offset += stride;
    FLOAT4 query_vec2 = (x4 + 2 >= seq_len) ? (FLOAT4)0 : vload4(0, query + query_offset); query_offset += stride;
    FLOAT4 query_vec3 = (x4 + 3 >= seq_len) ? (FLOAT4)0 : vload4(0, query + query_offset);
    
    const int queryout_offset = (z * head_dim + y4) * seq_len4 + x4;
    vstore4((FLOAT4)(query_vec0.s0, query_vec1.s0, query_vec2.s0, query_vec3.s0), 0, query_tmp + queryout_offset);
    vstore4((FLOAT4)(query_vec0.s1, query_vec1.s1, query_vec2.s1, query_vec3.s1), 0, query_tmp + queryout_offset + seq_len4);
    vstore4((FLOAT4)(query_vec0.s2, query_vec1.s2, query_vec2.s2, query_vec3.s2), 0, query_tmp + queryout_offset + seq_len4 + seq_len4);
    vstore4((FLOAT4)(query_vec0.s3, query_vec1.s3, query_vec2.s3, query_vec3.s3), 0, query_tmp + queryout_offset + seq_len4 + seq_len4 + seq_len4);
}

__kernel void rearrange_k(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *key, // [1 key_seq_len kv_head_num head_dim]
                              __global FLOAT *past_key, // [1 kv_head_num head_dim max_length]
                              __private const int past_len, // prefill = 0, decode = past_key len
                              __private const int max_len,
                              __private const int seq_len,
                              __private const int kv_head_num,
                              __private const int head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // seq_len decode = 1
    const int y = get_global_id(1); // head_dim
    const int z = get_global_id(2); // kv_head_num
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int y4 = y << 2;
    
#ifdef OPENCL_PREFILL_ATTENTION
    const int x4 = x << 2;
    const int stride = kv_head_num * head_dim;
    int key_offset = (x4 * kv_head_num + z) * head_dim + y4;
    FLOAT4 key_vec0 = vload4(0, key + key_offset); key_offset += stride;
    FLOAT4 key_vec1 = (x4 + 1 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset); key_offset += stride;
    FLOAT4 key_vec2 = (x4 + 2 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset); key_offset += stride;
    FLOAT4 key_vec3 = (x4 + 3 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset);
    const int output_offset = (z * head_dim + y4) * max_len + past_len + x4;
    vstore4((FLOAT4)(key_vec0.s0, key_vec1.s0, key_vec2.s0, key_vec3.s0), 0, past_key + output_offset);
    vstore4((FLOAT4)(key_vec0.s1, key_vec1.s1, key_vec2.s1, key_vec3.s1), 0, past_key + output_offset + max_len);
    vstore4((FLOAT4)(key_vec0.s2, key_vec1.s2, key_vec2.s2, key_vec3.s2), 0, past_key + output_offset + max_len + max_len);
    vstore4((FLOAT4)(key_vec0.s3, key_vec1.s3, key_vec2.s3, key_vec3.s3), 0, past_key + output_offset + max_len + max_len + max_len);
#else
    FLOAT4 key_vec = vload4(0, key + z * head_dim + y4);
    const int output_offset = (z * head_dim + y4) * max_len + past_len - 1;
    past_key[output_offset] = key_vec.s0;
    past_key[output_offset + max_len] = key_vec.s1;
    past_key[output_offset + max_len + max_len] = key_vec.s2;
    past_key[output_offset + max_len + max_len + max_len] = key_vec.s3;
#endif
}

__kernel void rearrange_v(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *value, // [1 value_seq_len kv_head_num head_dim]
                              __global FLOAT *past_value, // [1 kv_head_num max_length head_dim]
                              __private const int past_len,
                              __private const int max_len,
                              __private const int seq_len,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // seq_len decode = 1
    const int z = get_global_id(2); // kv_head_num
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int x4 = x << 2;
    
#ifdef OPENCL_PREFILL_ATTENTION
    const int y4 = y << 2;
    const int stride = kv_head_num * head_dim;
    int value_offset = (y4 * kv_head_num + z) * head_dim + x4;
    FLOAT4 value_vec0 = vload4(0, value + value_offset); value_offset += stride;
    FLOAT4 value_vec1 = (y4 + 1 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset); value_offset += stride;
    FLOAT4 value_vec2 = (y4 + 2 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset); value_offset += stride;
    FLOAT4 value_vec3 = (y4 + 3 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset);
    const int output_offset = (z * max_len + past_len + y4) * head_dim + x4;
    vstore4(value_vec0, 0, past_value + output_offset);
    vstore4(value_vec1, 0, past_value + output_offset + head_dim);
    vstore4(value_vec2, 0, past_value + output_offset + head_dim + head_dim);
    vstore4(value_vec3, 0, past_value + output_offset + head_dim + head_dim + head_dim);
#else
    FLOAT4 value_vec = vload4(0, value + z * head_dim + x4);
    const int output_offset = (z * max_len + past_len - 1) * head_dim + x4;
    vstore4(value_vec, 0, past_value + output_offset);
#endif
}

__kernel void matmul_qk_div_mask_prefill(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [1 head_num head_dim query_seq_len]
                              __global const FLOAT *past_key, // [1 head_num head_dim max_length]
                              #ifdef ADD_MASK
                              __global const FLOAT* mask,
                              #else
                              __global const int* mask, // [1 1 query_seq_len key_seq_len]
                              #endif
                              __global FLOAT *qk, // [1 head_num key_seq_len query_seq_len]
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int key_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // query_seq_len
    const int y = get_global_id(1); // key_seq_len
    const int z = get_global_id(2); // head_num
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    const int x4 = x << 2;
    const int y4 = y << 2;
    
    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;;
    const int query_offset = z * head_dim * query_seq_len4 + x4;
    const int past_offset = (z / NUMHEAD_GROUP_SIZE) * head_dim * max_len + y4;
    float4 out0 = 0, out1 = 0, out2 = 0, out3 = 0;
    
    for(int i = 0; i < head_dim / 4; ++i){
        int i4 = i << 2;
        float4 query_vec0 = convert_float4(vload4(0, query + query_offset + i4 * query_seq_len4));
        float4 query_vec1 = convert_float4(vload4(0, query + query_offset + (i4 + 1) * query_seq_len4));
        float4 query_vec2 = convert_float4(vload4(0, query + query_offset + (i4 + 2) * query_seq_len4));
        float4 query_vec3 = convert_float4(vload4(0, query + query_offset + (i4 + 3) * query_seq_len4));
        
        float4 past_vec0 = convert_float4(vload4(0, past_key + past_offset + i4 * max_len));
        float4 past_vec1 = convert_float4(vload4(0, past_key + past_offset + (i4 + 1) * max_len));
        float4 past_vec2 = convert_float4(vload4(0, past_key + past_offset + (i4 + 2) * max_len));
        float4 past_vec3 = convert_float4(vload4(0, past_key + past_offset + (i4 + 3) * max_len));

        out0 = mad((float4)past_vec0.s0, query_vec0, out0);
        out0 = mad((float4)past_vec1.s0, query_vec1, out0);
        out0 = mad((float4)past_vec2.s0, query_vec2, out0);
        out0 = mad((float4)past_vec3.s0, query_vec3, out0);
        
        out1 = mad((float4)past_vec0.s1, query_vec0, out1);
        out1 = mad((float4)past_vec1.s1, query_vec1, out1);
        out1 = mad((float4)past_vec2.s1, query_vec2, out1);
        out1 = mad((float4)past_vec3.s1, query_vec3, out1);
        
        out2 = mad((float4)past_vec0.s2, query_vec0, out2);
        out2 = mad((float4)past_vec1.s2, query_vec1, out2);
        out2 = mad((float4)past_vec2.s2, query_vec2, out2);
        out2 = mad((float4)past_vec3.s2, query_vec3, out2);
        
        out3 = mad((float4)past_vec0.s3, query_vec0, out3);
        out3 = mad((float4)past_vec1.s3, query_vec1, out3);
        out3 = mad((float4)past_vec2.s3, query_vec2, out3);
        out3 = mad((float4)past_vec3.s3, query_vec3, out3);
    }
    out0 *= (float4)scale;
    out1 *= (float4)scale;
    out2 *= (float4)scale;
    out3 *= (float4)scale;
    {
        int mask_offset = x4 * key_seq_len + y4;
        float4 mask_tmp0 = convert_float4(vload4(0, mask + mask_offset)); mask_offset += key_seq_len;
        float4 mask_tmp1 = (x4 + 1 >= query_seq_len) ? (float4)0 : convert_float4(vload4(0, mask + mask_offset)); mask_offset += key_seq_len;
        float4 mask_tmp2 = (x4 + 2 >= query_seq_len) ? (float4)0 : convert_float4(vload4(0, mask + mask_offset)); mask_offset += key_seq_len;
        float4 mask_tmp3 = (x4 + 3 >= query_seq_len) ? (float4)0 : convert_float4(vload4(0, mask + mask_offset));
        float4 mask0 = (float4)(mask_tmp0.s0, mask_tmp1.s0, mask_tmp2.s0, mask_tmp3.s0);
        float4 mask1 = (float4)(mask_tmp0.s1, mask_tmp1.s1, mask_tmp2.s1, mask_tmp3.s1);
        float4 mask2 = (float4)(mask_tmp0.s2, mask_tmp1.s2, mask_tmp2.s2, mask_tmp3.s2);
        float4 mask3 = (float4)(mask_tmp0.s3, mask_tmp1.s3, mask_tmp2.s3, mask_tmp3.s3);
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
    }
    
    const int qk_offset = (z * key_seq_len + y4) * query_seq_len4 + x4;
    vstore4(CONVERT_FLOAT4(out0), 0, qk + qk_offset);
    if(y4 + 1 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out1), 0, qk + qk_offset + query_seq_len4);
    if(y4 + 2 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out2), 0, qk + qk_offset + query_seq_len4 + query_seq_len4);
    if(y4 + 3 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out3), 0, qk + qk_offset + query_seq_len4 + query_seq_len4 + query_seq_len4);
}

__kernel void matmul_qk_decode(GLOBAL_SIZE_2_DIMS
                              __global const FLOAT *query, // key [1 head_num head_dim]
                              __global const FLOAT *past_key, // [1 head_num head_dim max_length]
                              __global FLOAT *qk, // [1 head_num key_seq_len 1]
                              __private const float scale,
                              __private const int seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // key_seq_len
    const int y = get_global_id(1); // head_num
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int x4 = x << 2;
    
    const int query_offset = y * head_dim;
    const int past_offset = (y / NUMHEAD_GROUP_SIZE) * head_dim * max_len + x4;
    float4 out0 = 0;
    
    for(int i = 0; i < head_dim / 4; ++i){
        int i4 = i << 2;
        float4 query_vec = convert_float4(vload4(0, query + query_offset + i4));
        
        float4 past_vec0 = convert_float4(vload4(0, past_key + past_offset + i4 * max_len));
        float4 past_vec1 = convert_float4(vload4(0, past_key + past_offset + (i4 + 1) * max_len));
        float4 past_vec2 = convert_float4(vload4(0, past_key + past_offset + (i4 + 2) * max_len));
        float4 past_vec3 = convert_float4(vload4(0, past_key + past_offset + (i4 + 3) * max_len));
        
        out0 = mad((float4)query_vec.s0, past_vec0, out0);
        out0 = mad((float4)query_vec.s1, past_vec1, out0);
        out0 = mad((float4)query_vec.s2, past_vec2, out0);
        out0 = mad((float4)query_vec.s3, past_vec3, out0);
    }
    out0 *= (float4)scale;
    const int qk_offset = y * seq_len + x4;
    if(x4 + 3 < seq_len){
        vstore4(CONVERT_FLOAT4(out0), 0, qk + qk_offset);
    }else {
        int remain = seq_len - x4;
        if(remain == 3){
            vstore3(CONVERT_FLOAT3((float3)(out0.s012)), 0, qk + qk_offset);
        } else if(remain == 2){
            vstore2(CONVERT_FLOAT2((float2)(out0.s01)), 0, qk + qk_offset);
        }else if(remain == 1){
            qk[qk_offset] = out0.s0;
        }
    }
}

__kernel void matmul_qkv_prefill(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *qk, // qk prefill [1 head_num qk_seq_len value_seq_len]
                              __global const FLOAT *past_value, // [1 head_num max_len head_dim]
                              __global FLOAT *output, // [1 value_seq_len head_num head_dim]
                              __private const int qk_seq_len,
                              __private const int value_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // qk_seq_len
    const int z = get_global_id(2); // head_num
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    const int x8 = x << 3;
    const int y4 = y << 2;
    
    const int qk_seq_len4 = (qk_seq_len + 3) / 4 * 4;
    const int qk_offset = z * value_seq_len * qk_seq_len4 + y4;
    const int past_offset = ((z / NUMHEAD_GROUP_SIZE) * max_len) * head_dim + x8;
    const int loop_end = max(value_seq_len / 4 - 1, 0);
    COMPUTE_FLOAT8 out0 = 0, out1 = 0, out2 = 0, out3 = 0;
    
    for(int i = 0; i < loop_end; ++i){
        int i4 = i << 2;
        COMPUTE_FLOAT4 qk_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i4 * qk_seq_len4));
        COMPUTE_FLOAT4 qk_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 1) * qk_seq_len4));
        COMPUTE_FLOAT4 qk_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 2) * qk_seq_len4));
        COMPUTE_FLOAT4 qk_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 3) * qk_seq_len4));
        
        COMPUTE_FLOAT8 past_vec0 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i4 * head_dim));
        COMPUTE_FLOAT8 past_vec1 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 1) * head_dim));
        COMPUTE_FLOAT8 past_vec2 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 2) * head_dim));
        COMPUTE_FLOAT8 past_vec3 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 3) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT8)qk_vec0.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec1.s0, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec2.s0, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec3.s0, past_vec3, out0);
        
        out1 = mad((COMPUTE_FLOAT8)qk_vec0.s1, past_vec0, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec1.s1, past_vec1, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec2.s1, past_vec2, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec3.s1, past_vec3, out1);
        
        out2 = mad((COMPUTE_FLOAT8)qk_vec0.s2, past_vec0, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec1.s2, past_vec1, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec2.s2, past_vec2, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec3.s2, past_vec3, out2);
        
        out3 = mad((COMPUTE_FLOAT8)qk_vec0.s3, past_vec0, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec1.s3, past_vec1, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec2.s3, past_vec2, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec3.s3, past_vec3, out3);
    }
    for(int i = (loop_end << 2); i < value_seq_len; ++i){
        COMPUTE_FLOAT4 qk_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i * qk_seq_len4));
        COMPUTE_FLOAT8 past_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i * head_dim));
        
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s0, past_vec, out0);
        out1 = mad((COMPUTE_FLOAT8)qk_vec.s1, past_vec, out1);
        out2 = mad((COMPUTE_FLOAT8)qk_vec.s2, past_vec, out2);
        out3 = mad((COMPUTE_FLOAT8)qk_vec.s3, past_vec, out3);
    }
    
    const int output_offset = (y4 * head_num + z) * head_dim + x8;
    const int stride = head_num * head_dim;
    vstore8(CONVERT_FLOAT8(out0), 0, output + output_offset);
    if(y4 + 1 >= qk_seq_len) return;
    vstore8(CONVERT_FLOAT8(out1), 0, output + output_offset + stride);
    if(y4 + 2 >= qk_seq_len) return;
    vstore8(CONVERT_FLOAT8(out2), 0, output + output_offset + stride + stride);
    if(y4 + 3 >= qk_seq_len) return;
    vstore8(CONVERT_FLOAT8(out3), 0, output + output_offset + stride + stride + stride);
}


__kernel void matmul_qkv_decode_b8(GLOBAL_SIZE_2_DIMS
                              __global const FLOAT *qk, // qk [1 head_num qk_seq_len 1]
                              __global const FLOAT *past_value, // [1 head_num max_len head_dim]
                              __global FLOAT *output, // [1 1 head_num head_dim]
                              __private const int qk_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // head_num
    
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int x8 = x << 3;
    
    const int qk_offset = y * qk_seq_len;
    const int past_offset = ((y / NUMHEAD_GROUP_SIZE) * max_len) * head_dim + x8;
    COMPUTE_FLOAT8 out0 = 0;
    #ifdef LOOP_UNROLL_4
    const int loop_end = max((qk_seq_len + 3) / 4 - 1, 0);
    for(int i = 0; i < loop_end; ++i){
        int i4 = i << 2;
        COMPUTE_FLOAT4 qk_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i4));
        
        COMPUTE_FLOAT8 past_vec0 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i4 * head_dim));
        COMPUTE_FLOAT8 past_vec1 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 1) * head_dim));
        COMPUTE_FLOAT8 past_vec2 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 2) * head_dim));
        COMPUTE_FLOAT8 past_vec3 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 3) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s1, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s2, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s3, past_vec3, out0);
    }
    for(int i = (loop_end << 2); i < qk_seq_len; ++i){
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT8 past_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i * head_dim));
        out0 = mad((COMPUTE_FLOAT8)qk_vec, past_vec, out0);
    }
    #elif (defined LOOP_UNROLL_8)
    const int loop_end = max((qk_seq_len + 7) / 8 - 1, 0);
    for(int i = 0; i < loop_end; ++i){
        int i8 = i << 3;
        COMPUTE_FLOAT8 qk_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, qk + qk_offset + i8));
        
        COMPUTE_FLOAT8 past_vec0 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i8 * head_dim));
        COMPUTE_FLOAT8 past_vec1 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 1) * head_dim));
        COMPUTE_FLOAT8 past_vec2 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 2) * head_dim));
        COMPUTE_FLOAT8 past_vec3 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 3) * head_dim));
        COMPUTE_FLOAT8 past_vec4 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 4) * head_dim));
        COMPUTE_FLOAT8 past_vec5 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 5) * head_dim));
        COMPUTE_FLOAT8 past_vec6 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 6) * head_dim));
        COMPUTE_FLOAT8 past_vec7 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 7) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s1, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s2, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s3, past_vec3, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s4, past_vec4, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s5, past_vec5, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s6, past_vec6, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s7, past_vec7, out0);
    }
    for(int i = (loop_end << 3); i < qk_seq_len; ++i){
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT8 past_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i * head_dim));
        out0 = mad((COMPUTE_FLOAT8)qk_vec, past_vec, out0);
    }
    #endif
    
    const int output_offset = y * head_dim + x8;
    vstore8(CONVERT_FLOAT8(out0), 0, output + output_offset);
}

__kernel void matmul_qkv_decode_b4(GLOBAL_SIZE_2_DIMS
                              __global const FLOAT *qk, // qk [1 head_num qk_seq_len 1]
                              __global const FLOAT *past_value, // [1 head_num max_len head_dim]
                              __global FLOAT *output, // [1 1 head_num head_dim]
                              __private const int qk_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // head_num
    
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int x4 = x << 2;
    
    const int qk_offset = y * qk_seq_len;
    const int past_offset = ((y / NUMHEAD_GROUP_SIZE) * max_len) * head_dim + x4;
    COMPUTE_FLOAT4 out0 = 0;
    #ifdef LOOP_UNROLL_4
    const int loop_end = max((qk_seq_len + 3) / 4 - 1, 0);
    for(int i = 0; i < loop_end; ++i){
        int i4 = i << 2;
        COMPUTE_FLOAT4 qk_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i4));
        
        COMPUTE_FLOAT4 past_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + i4 * head_dim));
        COMPUTE_FLOAT4 past_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i4 + 1) * head_dim));
        COMPUTE_FLOAT4 past_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i4 + 2) * head_dim));
        COMPUTE_FLOAT4 past_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i4 + 3) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s1, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s2, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s3, past_vec3, out0);
    }
    for(int i = (loop_end << 2); i < qk_seq_len; ++i){
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT4 past_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + i * head_dim));
        out0 = mad((COMPUTE_FLOAT4)qk_vec, past_vec, out0);
    }
    #elif (defined LOOP_UNROLL_8)
    const int loop_end = max((qk_seq_len + 7) / 8 - 1, 0);
    for(int i = 0; i < loop_end; ++i){
        int i8 = i << 3;
        COMPUTE_FLOAT8 qk_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, qk + qk_offset + i8));
        
        COMPUTE_FLOAT4 past_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + i8 * head_dim));
        COMPUTE_FLOAT4 past_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 1) * head_dim));
        COMPUTE_FLOAT4 past_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 2) * head_dim));
        COMPUTE_FLOAT4 past_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 3) * head_dim));
        COMPUTE_FLOAT4 past_vec4 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 4) * head_dim));
        COMPUTE_FLOAT4 past_vec5 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 5) * head_dim));
        COMPUTE_FLOAT4 past_vec6 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 6) * head_dim));
        COMPUTE_FLOAT4 past_vec7 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 7) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s1, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s2, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s3, past_vec3, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s4, past_vec4, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s5, past_vec5, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s6, past_vec6, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s7, past_vec7, out0);
    }
    for(int i = (loop_end << 3); i < qk_seq_len; ++i){
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT4 past_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + i * head_dim));
        out0 = mad((COMPUTE_FLOAT4)qk_vec, past_vec, out0);
    }
    #endif
    
    const int output_offset = y * head_dim + x4;
    vstore4(CONVERT_FLOAT4(out0), 0, output + output_offset);
}

