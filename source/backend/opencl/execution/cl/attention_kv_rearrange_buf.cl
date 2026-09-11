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

#ifndef NUMHEAD_GROUP_SIZE
#define NUMHEAD_GROUP_SIZE 1
#endif

__kernel void rearrange_k(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *key, // [batch key_seq_len kv_head_num head_dim]
                              __global FLOAT *past_key, // [batch kv_head_num head_dim max_length]
                              __private const int past_len, // 0 unless this is a chunked prefill
                              __private const int max_len,
                              __private const int seq_len,
                              __private const int kv_head_num,
                              __private const int head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // seq_len decode = 1
    const int y = get_global_id(1); // head_dim
    int z = get_global_id(2); //
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int b = z / kv_head_num;
    z = z % kv_head_num;
    const int y4 = y << 2;
    
    // The dim axis is walked four at a time, but head_dim need not be a multiple of four. Each of
    // the four components below belongs to a different dim, and past_key strides dims by max_len,
    // so an unguarded tail both reads into the next head and writes whole rows that do not exist.
    const int dims = min(4, head_dim - y4);
    const int x4 = x << 2;
    const int stride = kv_head_num * head_dim;
    int key_offset = ((b * seq_len + x4) * kv_head_num + z) * head_dim + y4;
    FLOAT4 key_vec0, key_vec1, key_vec2, key_vec3;
    if (4 == dims) {
        key_vec0 = vload4(0, key + key_offset); key_offset += stride;
        key_vec1 = (x4 + 1 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset); key_offset += stride;
        key_vec2 = (x4 + 2 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset); key_offset += stride;
        key_vec3 = (x4 + 3 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset);
    } else {
        key_vec0 = load_scalar4(key + key_offset, dims); key_offset += stride;
        key_vec1 = (x4 + 1 >= seq_len) ? (FLOAT4)0 : load_scalar4(key + key_offset, dims); key_offset += stride;
        key_vec2 = (x4 + 2 >= seq_len) ? (FLOAT4)0 : load_scalar4(key + key_offset, dims); key_offset += stride;
        key_vec3 = (x4 + 3 >= seq_len) ? (FLOAT4)0 : load_scalar4(key + key_offset, dims);
    }
    const int output_offset = ((b * kv_head_num + z) * head_dim + y4) * max_len + past_len + x4;
    vstore4((FLOAT4)(key_vec0.s0, key_vec1.s0, key_vec2.s0, key_vec3.s0), 0, past_key + output_offset);
    if (dims > 1) {
        vstore4((FLOAT4)(key_vec0.s1, key_vec1.s1, key_vec2.s1, key_vec3.s1), 0, past_key + output_offset + max_len);
    }
    if (dims > 2) {
        vstore4((FLOAT4)(key_vec0.s2, key_vec1.s2, key_vec2.s2, key_vec3.s2), 0,
                past_key + output_offset + max_len + max_len);
    }
    if (dims > 3) {
        vstore4((FLOAT4)(key_vec0.s3, key_vec1.s3, key_vec2.s3, key_vec3.s3), 0,
                past_key + output_offset + max_len + max_len + max_len);
    }
}

__kernel void rearrange_v(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *value, // [batch value_seq_len kv_head_num head_dim]
                              __global FLOAT *past_value, // [batch kv_head_num max_length head_dim]
                              __private const int past_len,
                              __private const int max_len,
                              __private const int seq_len,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // seq_len
    int z = get_global_id(2); // kv_head_num
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int b = z / kv_head_num;
    z = z % kv_head_num;
    const int x4 = x << 2;
    // head_dim need not be a multiple of four; the trailing group is short.
    const int dims = min(4, head_dim - x4);

    const int y4 = y << 2;
    const int stride = kv_head_num * head_dim;
    #ifdef VALUE_C4
    const int value_seq_storage = (global_size_dim2 / kv_head_num) * seq_len;
    const int value_channel = z * head_dim + x4;
    const int value_token = b * seq_len + y4;
    FLOAT4 value_vec0 = load_c4_value4(value, value_seq_storage, value_token, value_channel, x4, head_dim);
    FLOAT4 value_vec1 = (y4 + 1 >= seq_len) ? (FLOAT4)0 :
        load_c4_value4(value, value_seq_storage, value_token + 1, value_channel, x4, head_dim);
    FLOAT4 value_vec2 = (y4 + 2 >= seq_len) ? (FLOAT4)0 :
        load_c4_value4(value, value_seq_storage, value_token + 2, value_channel, x4, head_dim);
    FLOAT4 value_vec3 = (y4 + 3 >= seq_len) ? (FLOAT4)0 :
        load_c4_value4(value, value_seq_storage, value_token + 3, value_channel, x4, head_dim);
    #else
    int value_offset = ((b * seq_len + y4) * kv_head_num + z) * head_dim + x4;
    FLOAT4 value_vec0, value_vec1, value_vec2, value_vec3;
    if (4 == dims) {
        value_vec0 = vload4(0, value + value_offset); value_offset += stride;
        value_vec1 = (y4 + 1 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset); value_offset += stride;
        value_vec2 = (y4 + 2 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset); value_offset += stride;
        value_vec3 = (y4 + 3 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset);
    } else {
        value_vec0 = load_scalar4(value + value_offset, dims); value_offset += stride;
        value_vec1 = (y4 + 1 >= seq_len) ? (FLOAT4)0 : load_scalar4(value + value_offset, dims); value_offset += stride;
        value_vec2 = (y4 + 2 >= seq_len) ? (FLOAT4)0 : load_scalar4(value + value_offset, dims); value_offset += stride;
        value_vec3 = (y4 + 3 >= seq_len) ? (FLOAT4)0 : load_scalar4(value + value_offset, dims);
    }
    #endif
    // past_value is [.., maxLen, head_dim], so here the four components are contiguous dims of one
    // token and the tail spills into the next token's leading dims.
    const int output_offset = ((b * kv_head_num + z) * max_len + past_len + y4) * head_dim + x4;
    if (4 == dims) {
        vstore4(value_vec0, 0, past_value + output_offset);
        vstore4(value_vec1, 0, past_value + output_offset + head_dim);
        vstore4(value_vec2, 0, past_value + output_offset + head_dim + head_dim);
        vstore4(value_vec3, 0, past_value + output_offset + head_dim + head_dim + head_dim);
    } else {
        store_scalar4(past_value, output_offset, value_vec0, dims);
        store_scalar4(past_value, output_offset + head_dim, value_vec1, dims);
        store_scalar4(past_value, output_offset + head_dim + head_dim, value_vec2, dims);
        store_scalar4(past_value, output_offset + head_dim + head_dim + head_dim, value_vec3, dims);
    }
}
