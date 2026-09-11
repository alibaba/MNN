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

// "Masked out" sentinel. Must stay exactly representable in half (the score tile is
// COMPUTE_FLOAT), so that the FA_NEG_TEST probe is an exact compare in both precisions.
#define FA_NEG_INF (-60000.0f)
#define FA_NEG_TEST (-50000.0f)
#ifndef FD_HEAD_DIM
#define FD_HEAD_DIM 128
#endif
#ifndef FD_WG_SIZE
#define FD_WG_SIZE 64
#endif
// kv entries per workgroup. Independent of FD_WG_SIZE: both phases walk their axis grid-stride,
// so the group size is free to follow the wave rather than the tiling. It used to be pinned to
// headDim/2 -- the value the PV phase needed -- which made the kv split a function of headDim.
#ifndef FD_CHUNK
#define FD_CHUNK 64
#endif
// PV works on dim quads, so the trailing quad is short when headDim is not 4-aligned.
#define FD_DIM_QUADS (((FD_HEAD_DIM) + 3) >> 2)
__kernel void flash_decode_reduce(
                              __global const float *partial_o, // [headNum, numChunk, headDim]
                              __global const float *partial_ml, // [headNum, numChunk, 2]
                              __global FLOAT *output, // [1, 1, headNum, headDim]
                              __private const int head_num,
                              __private const int num_chunk) {
    if (1 == num_chunk) {
        // flash_decode_partial took the fused path and wrote the normalized row itself. This launch
        // cannot be skipped from the host -- see the comment there -- so make it a no-op.
        return;
    }
    const int d4 = get_global_id(0) << 2;
    const int hn = get_global_id(1);
    if (d4 >= FD_HEAD_DIM || hn >= head_num) {
        return;
    }
    // The trailing dim group is partial unless headDim is a multiple of 4, and `d4 < FD_HEAD_DIM`
    // alone does not catch it. A whole float4 there reads past partial_o, and on the output side
    // it would overwrite the leading dims of the next head.
    const int dcount = min(4, FD_HEAD_DIM - d4);
    float m = FA_NEG_INF;
    for (int c = 0; c < num_chunk; ++c) {
        m = fmax(m, partial_ml[(hn * num_chunk + c) * 2]);
    }
    float l = 0.0f;
    float4 acc = 0.0f;
    for (int c = 0; c < num_chunk; ++c) {
        const int base = hn * num_chunk + c;
        const float mc = partial_ml[base * 2];
        const float w = (mc <= FA_NEG_TEST) ? 0.0f : native_exp(mc - m);
        l += partial_ml[base * 2 + 1] * w;
        const int off = base * FD_HEAD_DIM + d4;
#if (FD_HEAD_DIM % 4) == 0
        acc = mad((float4)w, vload4(0, partial_o + off), acc);
#else
        // Scalar loads, kept behind a compile-time branch so the aligned case has no run-time test
        // inside the chunk loop. This covers every group, not just the short one: dcount is 4 for
        // all but the last. The lanes past dcount stay zero and are never stored.
        float4 po = (float4)0.0f;
        po.x = partial_o[off];
        if (dcount > 1) {
            po.y = partial_o[off + 1];
        }
        if (dcount > 2) {
            po.z = partial_o[off + 2];
        }
        if (dcount > 3) {
            po.w = partial_o[off + 3];
        }
        acc = mad((float4)w, po, acc);
#endif
    }
    acc *= (l > 0.0f) ? (1.0f / l) : 0.0f;
#ifdef ATTENTION_C4
    store_attention_c4_4(output, CONVERT_FLOAT4(acc), 1, 0, hn * FD_HEAD_DIM + d4, dcount);
#else
    if (4 == dcount) {
        vstore4(CONVERT_FLOAT4(acc), 0, output + hn * FD_HEAD_DIM + d4);
    } else {
        store_scalar4(output, hn * FD_HEAD_DIM + d4, CONVERT_FLOAT4(acc), dcount);
    }
#endif
}
