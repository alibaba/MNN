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
// "Masked out" sentinel. Must stay exactly representable in half (the score tile is
// COMPUTE_FLOAT), so that the FA_NEG_TEST probe is an exact compare in both precisions.
#define FA_NEG_INF (-60000.0f)
#define FA_NEG_TEST (-50000.0f)
// Flash-decoding (seqLen == 1). The three-stage decode path runs its P*V matmul on
// ceil(headDim/8) * headNum work-items -- 256 for headDim 128 / 16 heads -- while
// streaming the whole V cache, which leaves most of the GPU idle. These two kernels
// split the kv axis across workgroups instead: each one softmaxes its own chunk and
// emits an unnormalized (m, l, O), and the reduce kernel combines them.
// Partials are fp32 (the host allocates a 4-byte-per-element scratch tensor).
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

// Store one finished dim quad of a decode row. The fused single-chunk path below completes a quad
// at a time, so it cannot use the float4 store that the reduce kernel does; store_attention_c4_4
// already handles a count below 4 and a channel that is not 4-aligned.
static inline void store_decode_quad(__global FLOAT* output, const int channel, const float4 value,
                                     const int count) {
#ifdef ATTENTION_C4
    store_attention_c4_4(output, CONVERT_FLOAT4(value), 1, 0, channel, count);
#else
    if (count == 4) {
        vstore4(CONVERT_FLOAT4(value), 0, output + channel);
    } else {
        output[channel] = (FLOAT)value.x;
        if (count > 1) {
            output[channel + 1] = (FLOAT)value.y;
        }
        if (count > 2) {
            output[channel + 2] = (FLOAT)value.z;
        }
    }
#endif
}

__kernel void flash_decode_partial(
                              __global const FLOAT *query, // [1, 1, headNum, headDim]
                              __global const FLOAT *past_key, // [1, kvHeadNum, headDim, maxLen]
                              __global const FLOAT *past_value, // [1, kvHeadNum, maxLen, headDim]
                              __global float *partial_o, // [headNum, numChunk, headDim]
                              __global float *partial_ml, // [headNum, numChunk, 2]
                              __private const float scale,
                              __private const int kv_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int num_chunk,
                              __global FLOAT *output, // [1, 1, headNum, headDim], fused path only
                              __global const FLOAT *key, // [1, 1, kvHeadNum, headDim], this step only
                              __global const FLOAT *value, // same layout, see the C4 note below
                              // Writable aliases keep the read-only cache pointers qualified const.
                              __global FLOAT *past_key_out,
                              __global FLOAT *past_value_out) {
    const int lid = get_local_id(0);
    const int chunk = get_group_id(0);
    const int hn = get_global_id(1);
    const int kv_hn = hn / NUMHEAD_GROUP_SIZE;
    const int k0 = chunk * FD_CHUNK;

    __local float lsd[FD_CHUNK];
    __local FLOAT lqd[FD_HEAD_DIM];

    for (int d = lid; d < FD_HEAD_DIM; d += FD_WG_SIZE) {
        lqd[d] = query[hn * FD_HEAD_DIM + d];
    }
    // The last chunk appends the current KV row before reading it. The input is contiguous for the
    // single-token, single-batch decode shape, including NC4HW4 where the sequence stride is one.
    // Other chunks only need to synchronize local query loading.
    if (chunk == num_chunk - 1) {
        const int past_len = kv_seq_len - 1;
        const int kv_base = kv_hn * FD_HEAD_DIM;
        __global FLOAT* vrow = past_value_out + (kv_hn * max_len + past_len) * FD_HEAD_DIM;
        for (int d = lid; d < FD_HEAD_DIM; d += FD_WG_SIZE) {
            past_key_out[(kv_base + d) * max_len + past_len] = key[kv_base + d];
            vrow[d] = value[kv_base + d];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    } else {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#if (FD_CHUNK == 64) && defined(FD_QK_PAIR)
    // Adjacent KV entries are contiguous in K, so one lane can process a pair with vector loads.
    float2 qk2 = (float2)FA_NEG_INF;
    for (int u = lid; u < FD_CHUNK / 2; u += FD_WG_SIZE) {
        const int kg = k0 + (u << 1);
        if (kg + 2 <= kv_seq_len) {
            __global const FLOAT* kp = past_key + (kv_hn * FD_HEAD_DIM) * max_len + kg;
            float2 acc0 = 0.0f, acc1 = 0.0f;
            for (int d2 = 0; d2 < FD_HEAD_DIM; d2 += 2) {
                const float2 q2 = convert_float2(vload2(0, lqd + d2));
                acc0 = mad((float2)q2.x, convert_float2(vload2(0, kp + (d2 + 0) * max_len)), acc0);
                acc1 = mad((float2)q2.y, convert_float2(vload2(0, kp + (d2 + 1) * max_len)), acc1);
            }
            // The two chains split the dims by residue mod 2, each holding the whole 2-kv vector.
            qk2 = (acc0 + acc1) * scale;
        } else if (kg < kv_seq_len) {
            // Trailing pair: only one valid row, and the float2 load would read the next kv row
            // past kv_seq_len, so do it scalar. At most one lane per workgroup takes this path;
            // lanes whose pair is wholly past kv_seq_len must keep their FA_NEG_INF scores.
            __global const FLOAT* kr = past_key + (kv_hn * FD_HEAD_DIM) * max_len + kg;
            float acc = 0.0f;
            for (int d = 0; d < FD_HEAD_DIM; ++d) {
                acc = mad((float)lqd[d], (float)kr[d * max_len], acc);
            }
            qk2.x = acc * scale;
        }
    }
    // The 32 pair lanes cover all 64 slots of lsd; slots past kv_seq_len hold FA_NEG_INF scores,
    // which become 0 in the prob pass below.
    if (lid < FD_CHUNK / 2) {
        vstore2(qk2, lid, lsd);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Every work-item reduces the whole chunk redundantly: all lanes read the same address on
    // each step, so it is a broadcast and needs no barrier, unlike a tree reduction.
    float m = FA_NEG_INF;
    float4 m4 = (float4)FA_NEG_INF;
    for (int i = 0; i < FD_CHUNK; i += 4) {
        m4 = fmax(m4, vload4(0, lsd + i));
    }
    m = fmax(fmax(m4.x, m4.y), fmax(m4.z, m4.w));
    float2 p2 = (float2)0.0f;
    p2.x = (qk2.x <= FA_NEG_TEST) ? 0.0f : native_exp(qk2.x - m);
    p2.y = (qk2.y <= FA_NEG_TEST) ? 0.0f : native_exp(qk2.y - m);
    if (lid < FD_CHUNK / 2) {
        vstore2(p2, lid, lsd);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Same broadcast shape as the max; the summation order differs from a scalar loop, so l is
    // not bit-identical -- it is an fp32 accumulator that the reduce kernel divides by.
    float l = 0.0f;
    float4 l4 = (float4)0.0f;
    for (int i = 0; i < FD_CHUNK; i += 4) {
        l4 += vload4(0, lsd + i);
    }
    l = (l4.x + l4.y) + (l4.z + l4.w);
#else
    // K is [.., headDim, maxLen], so consecutive kv of one dim are contiguous and lanes working on
    // neighbouring kv coalesce.
    for (int kk = lid; kk < FD_CHUNK; kk += FD_WG_SIZE) {
        const int kg = k0 + kk;
        float s = FA_NEG_INF;
        if (kg < kv_seq_len) {
            __global const FLOAT* kp = past_key + (kv_hn * FD_HEAD_DIM) * max_len + kg;
            float acc = 0.0f;
            for (int d = 0; d < FD_HEAD_DIM; ++d) {
                acc = mad((float)lqd[d], (float)kp[d * max_len], acc);
            }
            s = acc * scale;
        }
        lsd[kk] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Every work-item reduces the chunk through broadcast local reads; vector loads shorten the
    // serial dependency chain without a tree-reduction barrier.
    float m = FA_NEG_INF;
#if (FD_CHUNK % 4) == 0
    float4 m4 = (float4)FA_NEG_INF;
    for (int i = 0; i < FD_CHUNK; i += 4) {
        m4 = fmax(m4, vload4(0, lsd + i));
    }
    m = fmax(fmax(m4.x, m4.y), fmax(m4.z, m4.w));
#else
    for (int i = 0; i < FD_CHUNK; ++i) {
        m = fmax(m, lsd[i]);
    }
#endif
    // Everyone is done reading the scores before they are overwritten with probabilities. The
    // score is re-read from local rather than kept private, because a lane no longer owns exactly
    // one kv.
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = lid; kk < FD_CHUNK; kk += FD_WG_SIZE) {
        const float s = lsd[kk];
        lsd[kk] = (s <= FA_NEG_TEST) ? 0.0f : native_exp(s - m);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Same shape. The summation order differs from the scalar loop, so l is not bit-identical --
    // it is an fp32 accumulator that the reduce kernel divides by.
    float l = 0.0f;
#if (FD_CHUNK % 4) == 0
    float4 l4 = (float4)0.0f;
    for (int i = 0; i < FD_CHUNK; i += 4) {
        l4 += vload4(0, lsd + i);
    }
    l = (l4.x + l4.y) + (l4.z + l4.w);
#else
    for (int i = 0; i < FD_CHUNK; ++i) {
        l += lsd[i];
    }
#endif
#endif

    // Unnormalized O for this chunk, one dim quad per unit.
    const int kv_valid = min(FD_CHUNK, kv_seq_len - k0);
    const int base = (hn * num_chunk + chunk);
    // A single chunk already spans the whole kv axis, so (m, l) are global and this lane's acc4
    // is the finished O for its dim quad -- there is nothing for the reduce kernel to merge.
    // Dividing by l here instead saves reduce's round trip for partial_o / partial_ml through
    // global memory, which is a large share of the op at short kv. reduce is still launched --
    // onExecute enqueues a fixed kernel list, and under the record queue it is built once -- but
    // it early-returns.
    const bool fused = (1 == num_chunk);
    const float rl = (l > 0.0f) ? (1.0f / l) : 0.0f;
#if (FD_HEAD_DIM % 4) == 0
    for (int u = lid; u < FD_DIM_QUADS; u += FD_WG_SIZE) {
        const int d4 = u << 2;
        __global const FLOAT* vp = past_value + (kv_hn * max_len) * FD_HEAD_DIM + d4;
        float4 acc4 = 0.0f;
        for (int i = 0; i < kv_valid; ++i) {
            acc4 = mad((float4)lsd[i], convert_float4(vload4(0, vp + (k0 + i) * FD_HEAD_DIM)), acc4);
        }
        if (fused) {
            store_decode_quad(output, hn * FD_HEAD_DIM + d4, acc4 * rl, 4);
        } else {
            vstore4(acc4, 0, partial_o + base * FD_HEAD_DIM + d4);
        }
    }
#else
    // headDim not 4-aligned: the trailing quad is short, so fall back to scalar-safe dim pairs.
    for (int u = lid; u < FD_DIM_QUADS * 2; u += FD_WG_SIZE) {
        const int d2 = u << 1;
        __global const FLOAT* vp = past_value + (kv_hn * max_len) * FD_HEAD_DIM + d2;
        const int dims = min(2, FD_HEAD_DIM - d2);
        float2 acc2 = 0.0f;
        for (int i = 0; i < kv_valid; ++i) {
            __global const FLOAT* vrow = vp + (k0 + i) * FD_HEAD_DIM;
            float2 vv = 0.0f;
            vv.x = (float)vrow[0];
            if (dims > 1) {
                vv.y = (float)vrow[1];
            }
            acc2 = mad((float2)lsd[i], vv, acc2);
        }
        if (fused) {
            store_decode_quad(output, hn * FD_HEAD_DIM + d2, (float4)(acc2.x, acc2.y, 0.0f, 0.0f), dims);
        } else {
            partial_o[base * FD_HEAD_DIM + d2] = acc2.x;
            if (dims > 1) {
                partial_o[base * FD_HEAD_DIM + d2 + 1] = acc2.y;
            }
        }
    }
#endif
    if (0 == lid && !fused) {
        partial_ml[base * 2] = m;
        partial_ml[base * 2 + 1] = l;
    }
}
