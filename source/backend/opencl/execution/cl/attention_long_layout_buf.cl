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


#ifdef VALUE_C4
static inline FLOAT load_c4_value(__global const FLOAT* value,
                                  const int seq_storage,
                                  const int token,
                                  const int channel) {
    return value[((channel >> 2) * seq_storage + token) * 4 + (channel & 3)];
}

static inline FLOAT4 load_c4_value4(__global const FLOAT* value,
                                    const int seq_storage,
                                    const int token,
                                    const int channel,
                                    const int head_dim_offset,
                                    const int head_dim) {
    return (FLOAT4)(
        load_c4_value(value, seq_storage, token, channel),
        (head_dim_offset + 1 >= head_dim) ? (FLOAT)0 : load_c4_value(value, seq_storage, token, channel + 1),
        (head_dim_offset + 2 >= head_dim) ? (FLOAT)0 : load_c4_value(value, seq_storage, token, channel + 2),
        (head_dim_offset + 3 >= head_dim) ? (FLOAT)0 : load_c4_value(value, seq_storage, token, channel + 3));
}
#endif

#ifdef ATTENTION_C4
static inline void store_attention_c4_4(__global FLOAT* output, const FLOAT4 value, const int seq_storage,
                                        const int token, const int channel, const int count) {
    if (((channel & 3) == 0) && count == 4) {
        const int offset = ((channel >> 2) * seq_storage + token) * 4;
        vstore4(value, 0, output + offset);
        return;
    }
    int c = channel;
    output[((c >> 2) * seq_storage + token) * 4 + (c & 3)] = value.x;
    if (count > 1) {
        c = channel + 1;
        output[((c >> 2) * seq_storage + token) * 4 + (c & 3)] = value.y;
    }
    if (count > 2) {
        c = channel + 2;
        output[((c >> 2) * seq_storage + token) * 4 + (c & 3)] = value.z;
    }
    if (count > 3) {
        c = channel + 3;
        output[((c >> 2) * seq_storage + token) * 4 + (c & 3)] = value.w;
    }
}

static inline void store_attention_c4_8(__global FLOAT* output, const FLOAT8 value, const int seq_storage,
                                        const int token, const int channel, const int count) {
    const int low_count = min(count, 4);
    store_attention_c4_4(output, value.lo, seq_storage, token, channel, low_count);
    if (count > 4) {
        store_attention_c4_4(output, value.hi, seq_storage, token, channel + 4, count - 4);
    }
}
#endif

// Store the first `count` (<=4) components of a FLOAT4 to contiguous addresses without vector subscript.
static inline void store_scalar4(__global FLOAT* output, const int base, const FLOAT4 value, const int count) {
    output[base] = value.x;
    if (count > 1) {
        output[base + 1] = value.y;
    }
    if (count > 2) {
        output[base + 2] = value.z;
    }
    if (count > 3) {
        output[base + 3] = value.w;
    }
}

// Store the first `count` (<=8) components of a FLOAT8 to contiguous addresses without vector subscript.
static inline void store_scalar8(__global FLOAT* output, const int base, const FLOAT8 value, const int count) {
    output[base] = value.s0;
    if (count > 1) {
        output[base + 1] = value.s1;
    }
    if (count > 2) {
        output[base + 2] = value.s2;
    }
    if (count > 3) {
        output[base + 3] = value.s3;
    }
    if (count > 4) {
        output[base + 4] = value.s4;
    }
    if (count > 5) {
        output[base + 5] = value.s5;
    }
    if (count > 6) {
        output[base + 6] = value.s6;
    }
    if (count > 7) {
        output[base + 7] = value.s7;
    }
}

// Load the first `count` (1..4) components from contiguous addresses, zeroing the rest. Used on
// the dim axis of the kv rearrange kernels, where head_dim is a runtime argument rather than a
// macro, so the tail cannot be handled with #if the way the fused kernels do it.
static inline FLOAT4 load_scalar4(__global const FLOAT* input, const int count) {
    FLOAT4 value = (FLOAT4)0;
    value.x = input[0];
    if (count > 1) {
        value.y = input[1];
    }
    if (count > 2) {
        value.z = input[2];
    }
    if (count > 3) {
        value.w = input[3];
    }
    return value;
}

// The dim axis of the plain-buffer q/k/v inputs carries no padding and heads sit back to back, so a
// vload4 on a partial tail group reads into the next head, and past the end of the buffer on the last
// one. Only the tail group takes the scalar path, so the aligned groups keep their vector load. The
// NC4HW4 value path already clamps this way inside load_c4_value4.
static inline FLOAT4 load_dim4(__global const FLOAT* input, const int count) {
    return (count >= 4) ? vload4(0, input) : load_scalar4(input, count);
}

__kernel void rearrange_qkv(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input_q, //[batch, seqLenQ/4, headNum, headDim, seqLenQ_4]
                              __global const FLOAT *input_k, // [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4]
                              __global const FLOAT *input_v, // [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4]
                              __global FLOAT *output_q, // [batch*headNum, ROUND_UP(headDim, mTileHDK), ROUND_UP(seqLenQ, mTileQ)]
                              __global FLOAT *output_k, // [batch*headNum/group, ROUND_UP(headDim, mTileHDK), ROUND_UP(seqLenKV, mTileKV)]
                              __global FLOAT *output_v, // [batch*headNum/group, ROUND_UP(seqLenKV, mTileKV), ROUND_UP(headDim, mTileHDN)]
                              #ifdef SAVE_KV
                              __global FLOAT *past_k, // [batch, headNum/group, headDim, seqLenKV_4]
                              __global FLOAT *past_v, // [batch, headNum/group, seqLenKV_4, headDim]
                              #endif
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
            const int dim_count_q = headDim - 4 * hd;
            FLOAT4 temp_0 = load_dim4(input_q + in_offset_q, dim_count_q);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenQ) ? (FLOAT4)0 : load_dim4(input_q + in_offset_q + headNum*headDim, dim_count_q);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenQ) ? (FLOAT4)0 : load_dim4(input_q + in_offset_q + 2*headNum*headDim, dim_count_q);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenQ) ? (FLOAT4)0 : load_dim4(input_q + in_offset_q + 3*headNum*headDim, dim_count_q);
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
            const int dim_count_k = headDim - 4 * hd;
            FLOAT4 temp_0 = load_dim4(input_k + in_offset_kv, dim_count_k);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenKV) ? (FLOAT4)0 : load_dim4(input_k + in_offset_kv + headNum*headDim/group, dim_count_k);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenKV) ? (FLOAT4)0 : load_dim4(input_k + in_offset_kv + 2*headNum*headDim/group, dim_count_k);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenKV) ? (FLOAT4)0 : load_dim4(input_k + in_offset_kv + 3*headNum*headDim/group, dim_count_k);
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
            
            // pastK. output_k above may write all four dim rows because headDimPackQK pads them, but
            // past_k is packed to headDim with the kv heads back to back: a tail group writing four rows
            // there lands on the next head's leading rows.
            #ifdef SAVE_KV
            vstore4(key0, 0, past_k + past_offset_k);
            if(dim_count_k > 1) {
                vstore4(key1, 0, past_k + past_offset_k + maxLenKV);
            }
            if(dim_count_k > 2) {
                vstore4(key2, 0, past_k + past_offset_k + 2*maxLenKV);
            }
            if(dim_count_k > 3) {
                vstore4(key3, 0, past_k + past_offset_k + 3*maxLenKV);
            }
            #endif
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
            const int dim_count_v = headDim - 4 * hd;
            #ifdef VALUE_C4
            const int value_seq_storage = batch * seqLenKV;
            const int value_channel = hn * headDim + 4 * hd;
            const int value_token = b * seqLenKV + sl * 4;
            FLOAT4 temp_0 = load_c4_value4(input_v, value_seq_storage, value_token, value_channel, 4 * hd, headDim);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenKV) ? (FLOAT4)0 :
                load_c4_value4(input_v, value_seq_storage, value_token + 1, value_channel, 4 * hd, headDim);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenKV) ? (FLOAT4)0 :
                load_c4_value4(input_v, value_seq_storage, value_token + 2, value_channel, 4 * hd, headDim);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenKV) ? (FLOAT4)0 :
                load_c4_value4(input_v, value_seq_storage, value_token + 3, value_channel, 4 * hd, headDim);
            #else
            FLOAT4 temp_0 = load_dim4(input_v + in_offset_kv, dim_count_v);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenKV) ? (FLOAT4)0 : load_dim4(input_v + in_offset_kv + headNum*headDim/group, dim_count_v);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenKV) ? (FLOAT4)0 : load_dim4(input_v + in_offset_kv + 2*headNum*headDim/group, dim_count_v);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenKV) ? (FLOAT4)0 : load_dim4(input_v + in_offset_kv + 3*headNum*headDim/group, dim_count_v);
            #endif
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
            
            // pastV. output_v above may write four dims because headDimPackV pads them, but past_v is
            // packed to headDim with the tokens back to back: a tail group writing four dims there lands
            // on the next token's leading dims.
            #ifdef SAVE_KV
            if(dim_count_v >= 4) {
                vstore4(temp_0, 0, past_v + past_offset_v);
                vstore4(temp_1, 0, past_v + past_offset_v + headDim);
                vstore4(temp_2, 0, past_v + past_offset_v + 2*headDim);
                vstore4(temp_3, 0, past_v + past_offset_v + 3*headDim);
            } else {
                store_scalar4(past_v, past_offset_v, temp_0, dim_count_v);
                store_scalar4(past_v, past_offset_v + headDim, temp_1, dim_count_v);
                store_scalar4(past_v, past_offset_v + 2*headDim, temp_2, dim_count_v);
                store_scalar4(past_v, past_offset_v + 3*headDim, temp_3, dim_count_v);
            }
            #endif
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
          __global FLOAT *output, // [Batch, seqLen/4, mNumHead， mHeadDim, 4]  (or NC4HW4 when ATTENTION_C4)
          __private const int tile_q,
          __private const int tile_hdn,
          __private const int seq_len,
          __private const int head_num,
          __private const int head_dim,
          __private const int batch
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

    // Q
    FLOAT4 temp_0 = vload4(0, input + offset_inp);
    FLOAT4 temp_1 = vload4(0, input + offset_inp + seq_len_pack);
    FLOAT4 temp_2 = vload4(0, input + offset_inp + 2 * seq_len_pack);
    FLOAT4 temp_3 = vload4(0, input + offset_inp + 3 * seq_len_pack);

#ifdef ATTENTION_C4
    // output is NC4HW4: [(head_num*head_dim)/4, batch*seq_len, 4].
    const int channel = hn * head_dim + 4 * hd;
    const int channel_count = min(4, head_dim - 4 * hd);
    const int seq_storage = seq_len * batch;
    int token = b * seq_len + sl * 4;
    store_attention_c4_4(output, (FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0), seq_storage, token,
                         channel, channel_count);
    if(4 * sl + 1 >= seq_len) return;
    store_attention_c4_4(output, (FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1), seq_storage, ++token,
                         channel, channel_count);
    if(4 * sl + 2 >= seq_len) return;
    store_attention_c4_4(output, (FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2), seq_storage, ++token,
                         channel, channel_count);
    if(4 * sl + 3 >= seq_len) return;
    store_attention_c4_4(output, (FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3), seq_storage, ++token,
                         channel, channel_count);
#else
    const int offset_out = (((b * seq_len + sl*4) * head_num + hn) * head_dim + 4 * hd);
    // A head_dim that is not a multiple of 4 leaves the last dim group partial, and heads sit back
    // to back in the output: a full vstore4 there overwrites the next head's leading dims, and runs
    // off the buffer on the last head. The ATTENTION_C4 branch above already clamps with
    // channel_count; do the same here. Only the tail column takes the else, and the branch is
    // uniform across it, so the aligned case keeps its vector stores.
    const int dim_count = head_dim - 4 * hd;
    const FLOAT4 out_0 = (FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0);
    const FLOAT4 out_1 = (FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1);
    const FLOAT4 out_2 = (FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2);
    const FLOAT4 out_3 = (FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3);
    if(dim_count >= 4) {
        vstore4(out_0, 0, output + offset_out);
        if(4 * sl + 1 >= seq_len) return;
        vstore4(out_1, 0, output + offset_out + head_num*head_dim);
        if(4 * sl + 2 >= seq_len) return;
        vstore4(out_2, 0, output + offset_out + 2*head_num*head_dim);
        if(4 * sl + 3 >= seq_len) return;
        vstore4(out_3, 0, output + offset_out + 3*head_num*head_dim);
    } else {
        store_scalar4(output, offset_out, out_0, dim_count);
        if(4 * sl + 1 >= seq_len) return;
        store_scalar4(output, offset_out + head_num*head_dim, out_1, dim_count);
        if(4 * sl + 2 >= seq_len) return;
        store_scalar4(output, offset_out + 2*head_num*head_dim, out_2, dim_count);
        if(4 * sl + 3 >= seq_len) return;
        store_scalar4(output, offset_out + 3*head_num*head_dim, out_3, dim_count);
    }
#endif

}
