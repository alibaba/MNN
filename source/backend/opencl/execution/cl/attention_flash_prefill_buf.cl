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
// ---------------------------------------------------------------------------
// Fused flash attention (prefill): qk + mask + softmax + qkv in one kernel, so the
// O(seqLen * kvLen) qk and softmax buffers never have to exist at all.
// A workgroup owns FA_TILE_Q query rows of one (batch, head) and walks kv in FA_TILE_KV
// chunks, keeping the online-softmax state and the whole O tile alive across the loop.
// The tile macros are interdependent; flashPrefillEligible() derives and documents them.
#ifndef FA_TILE_Q
#define FA_TILE_Q 16
#endif
#ifndef FA_TILE_KV
#define FA_TILE_KV 64
#endif
#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM 128
#endif
#ifndef FA_WG_SIZE
#define FA_WG_SIZE 64
#endif
// Work-items per query row in the softmax phase, and kv entries each one owns. Both
// divisions must be exact; the host picks the triple so that they are.
#define FA_TPR (FA_WG_SIZE / FA_TILE_Q)
#define FA_KV_PER_THREAD (FA_TILE_KV / FA_TPR)
// Dim counts rounded up to the vector widths the phases load with. lq is declared at the
// padded width because phase 0 writes four dim rows at a time; the pad rows get zeros and
// phase 1 never reads them. Phase 3 owns 8 dims per work-item, so its trailing group is
// partial when headDim is not a multiple of 8.
#define FA_HEAD_DIM_4 ((((FA_HEAD_DIM) + 3) >> 2) << 2)
#define FA_DIM_GROUPS (((FA_HEAD_DIM) + 7) >> 3)
// Phase 3 units, i.e. how many lanes actually carry an O tile. The work group can be larger
// (it also has to be a multiple of FA_TILE_Q for phase 2), and when it is not, the surplus
// lanes have to sit phase 3 out. Whether that guard exists at all is decided here rather than
// at run time: wrapping the phase-3 loop in a run-time branch costs ~0.9% on pp2048 even when
// the branch is always taken, because it changes how the O accumulators get allocated.
#define FA_P3_UNITS ((FA_TILE_Q >> 2) * FA_DIM_GROUPS)
// Storing one output row of the O tile. Only the trailing dim group can be partial, so the
// aligned case keeps its single vstore8 and pays nothing for the general one.
#if (FA_HEAD_DIM % 8) == 0
#define STORE_FA_ROW(out, off, val, n) vstore8(CONVERT_FLOAT8(val), 0, (out) + (off))
#else
#define STORE_FA_ROW(out, off, val, n) store_scalar8((out), (off), CONVERT_FLOAT8(val), (n))
#endif

// "Masked out" sentinel. Must stay exactly representable in half (the score tile is
// COMPUTE_FLOAT), so that the FA_NEG_TEST probe is an exact compare in both precisions.
#define FA_NEG_INF (-60000.0f)
#define FA_NEG_TEST (-50000.0f)

__kernel void flash_attention_prefill(
                              __global const FLOAT *query, // [batch, seqLen, headNum, headDim]
                              __global const FLOAT *past_key, // [batch, kvHeadNum, headDim, maxLen]
                              __global const FLOAT *past_value, // [batch, kvHeadNum, maxLen, headDim]
                              #ifdef SET_MASK
                              __global const int* mask, // [maskBatch, maskQLen, maskKvLen]
                              #else
                              __global const FLOAT* mask,
                              #endif
                              __global FLOAT *output, // [batch, seqLen, headNum, headDim]
                              __private const float scale,
                              __private const int seq_len,
                              __private const int kv_seq_len,
                              __private const int mask_kv_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int batch,
                              // Appended, not inserted: the record path patches args 7 and 9 by index.
                              __private const int mask_q_len,
                              // maskQLen * maskKvLen when the plane is per batch, 0 when it is shared.
                              __private const int mask_batch_stride) {
    const int lid = get_local_id(0);
    const int q_start = get_group_id(0) * FA_TILE_Q;
    const int z = get_global_id(1); // batch * headNum
    const int b = z / head_num;
    const int hn = z % head_num;
    const int kv_hn = hn / NUMHEAD_GROUP_SIZE;

    __local FLOAT lq[FA_HEAD_DIM_4 * FA_TILE_Q]; // Q tile, [headDim rounded to 4][FA_TILE_Q]
    __local COMPUTE_FLOAT ls[FA_TILE_KV * FA_TILE_Q]; // S then P, [FA_TILE_KV][FA_TILE_Q]
    __local float lred[FA_TILE_Q * FA_TPR];    // per-row partials of max / sum
    __local float ll[FA_TILE_Q];               // running row sum
    __local float la[FA_TILE_Q];               // this block's rescale factor

    // Q stays local for the whole kv loop, transposed to [dim][row] so phase 1 can
    // grab 4 rows per vload4.
    const int head_dim_4 = FA_HEAD_DIM_4 >> 2;
    for (int idx = lid; idx < FA_TILE_Q * head_dim_4; idx += FA_WG_SIZE) {
        const int r = idx / head_dim_4;
        const int d4 = (idx - r * head_dim_4) << 2;
        const int q = q_start + r;
        const int q_off = ((b * seq_len + q) * head_num + hn) * FA_HEAD_DIM + d4;
#if (FA_HEAD_DIM % 4) == 0
        FLOAT4 qv = (q < seq_len) ? vload4(0, query + q_off) : (FLOAT4)0;
#else
        // Scalar loads for every group, not just the last one: this branch is compiled in for the
        // whole loop. Only the trailing group is short, and a vload4 there would cross into the
        // next head; the lanes past the end stay zero, which is what lq's pad rows want anyway.
        FLOAT4 qv = (FLOAT4)0;
        if (q < seq_len) {
            const int dcount = min(4, FA_HEAD_DIM - d4);
            qv.x = query[q_off];
            if (dcount > 1) {
                qv.y = query[q_off + 1];
            }
            if (dcount > 2) {
                qv.z = query[q_off + 2];
            }
            if (dcount > 3) {
                qv.w = query[q_off + 3];
            }
        }
#endif
        lq[d4 * FA_TILE_Q + r] = qv.x;
        lq[(d4 + 1) * FA_TILE_Q + r] = qv.y;
        lq[(d4 + 2) * FA_TILE_Q + r] = qv.z;
        lq[(d4 + 3) * FA_TILE_Q + r] = qv.w;
    }
    for (int r = lid; r < FA_TILE_Q; r += FA_WG_SIZE) {
        ll[r] = 0.0f;
    }

    // phase 3 layout: work-item owns rows [r0, r0+4) and dims [d8, d8+8). Surplus lanes get
    // r0 clamped to 0 rather than left past the end of la / ll.
#if FA_WG_SIZE > FA_P3_UNITS
    const bool p3_active = lid < FA_P3_UNITS;
    const int r0 = p3_active ? ((lid / FA_DIM_GROUPS) << 2) : 0;
    const int d8 = p3_active ? ((lid % FA_DIM_GROUPS) << 3) : 0;
#else
    const bool p3_active = true;
    const int r0 = (lid / FA_DIM_GROUPS) << 2;
    const int d8 = (lid % FA_DIM_GROUPS) << 3;
#endif
#if (FA_HEAD_DIM % 8) == 0
    const int p3_dims = 8;
#else
    const int p3_dims = min(8, FA_HEAD_DIM - d8);
#endif
    COMPUTE_FLOAT8 out0 = 0, out1 = 0, out2 = 0, out3 = 0;

    // phase 1 layout: work-item owns rows [qr0, qr0+4) and kv [kloc, kloc+4).
    const int num_kv_quad = FA_TILE_KV >> 2;
    const int p1_units = (FA_TILE_Q >> 2) * num_kv_quad;
    // phase 2 layout: FA_TPR work-items per row, FA_KV_PER_THREAD kv entries each.
    const int p2_row = lid % FA_TILE_Q;
    const int p2_sub = lid / FA_TILE_Q;
    const int p2_beg = p2_sub * FA_KV_PER_THREAD;
    // Running row max, kept private: the FA_TPR work-items of a row all reduce the same
    // lred partials, so they derive an identical m_new and never need to exchange it.
    float m_prev = FA_NEG_INF;
    const int key_base = ((b * kv_head_num + kv_hn) * FA_HEAD_DIM) * max_len;
    const int value_base = ((b * kv_head_num + kv_hn) * max_len) * FA_HEAD_DIM + d8;

#if defined(ADD_MASK) || defined(SET_MASK)
    const int kv_end = kv_seq_len;
#else
    // Causal mask: every kv past the last query row of this tile is masked out for
    // all of the tile's rows, so those blocks can be skipped outright.
    const int kv_end = min(kv_seq_len, kv_seq_len - seq_len + q_start + FA_TILE_Q);
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_start = 0; k_start < kv_end; k_start += FA_TILE_KV) {
        // ---- phase 1: S = scale * Q.K^T ----
        for (int u = lid; u < p1_units; u += FA_WG_SIZE) {
            const int kloc = (u % num_kv_quad) << 2;
            const int qr0 = (u / num_kv_quad) << 2;
            // Clamp, don't branch: phase 2 overwrites out-of-range scores anyway.
            const int kread = min(k_start + kloc, max_len - 4);
            __local const FLOAT* qp = lq + qr0;
            __global const FLOAT* kp = past_key + key_base + kread;
            COMPUTE_FLOAT4 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            for (int d = 0; d < FA_HEAD_DIM; ++d) {
                COMPUTE_FLOAT4 qv = CONVERT_COMPUTE_FLOAT4(vload4(0, qp + d * FA_TILE_Q));
                COMPUTE_FLOAT4 kv = CONVERT_COMPUTE_FLOAT4(vload4(0, kp + d * max_len));
                acc0 = mad((COMPUTE_FLOAT4)kv.x, qv, acc0);
                acc1 = mad((COMPUTE_FLOAT4)kv.y, qv, acc1);
                acc2 = mad((COMPUTE_FLOAT4)kv.z, qv, acc2);
                acc3 = mad((COMPUTE_FLOAT4)kv.w, qv, acc3);
            }
            const COMPUTE_FLOAT scale_c = (COMPUTE_FLOAT)scale;
            __local COMPUTE_FLOAT* sp = ls + kloc * FA_TILE_Q + qr0;
            vstore4(acc0 * scale_c, 0, sp);
            vstore4(acc1 * scale_c, 0, sp + FA_TILE_Q);
            vstore4(acc2 * scale_c, 0, sp + 2 * FA_TILE_Q);
            vstore4(acc3 * scale_c, 0, sp + 3 * FA_TILE_Q);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- phase 2: mask + online softmax, FA_TPR work-items per row ----
        {
            const int q = q_start + p2_row;
            const int kv_valid = kv_seq_len - k_start;
            #if defined(ADD_MASK) || defined(SET_MASK)
            const int mask_clp = k_start + mask_kv_len - kv_seq_len;
            const int mask_row = b * mask_batch_stride + q * mask_kv_len;
            // A plane shorter than the query leaves its trailing rows unmasked, the same way a
            // plane shorter than the kv axis leaves the history columns unmasked. Hoisted out of
            // the kv loop: it only depends on the row.
            const bool mask_on = q < mask_q_len;
            #else
            const int kv_valid_offset = kv_seq_len - seq_len;
            #endif
            float pmax = FA_NEG_INF;
            for (int j = 0; j < FA_KV_PER_THREAD; ++j) {
                const int kk = p2_beg + j;
                float s = (float)ls[kk * FA_TILE_Q + p2_row];
                if (kk >= kv_valid || q >= seq_len) {
                    s = FA_NEG_INF;
                } else {
                    #ifdef ADD_MASK
                    const int kc = mask_clp + kk;
                    if (mask_on && kc >= 0 && kc < mask_kv_len) {
                        s += (float)mask[mask_row + kc];
                    }
                    #elif defined(SET_MASK)
                    const int kc = mask_clp + kk;
                    if (mask_on && (!(kc >= 0 && kc < mask_kv_len) || mask[mask_row + kc] == 0)) {
                        s = FA_NEG_INF;
                    }
                    #else
                    if (k_start + kk > kv_valid_offset + q) {
                        s = FA_NEG_INF;
                    }
                    #endif
                    // Normalize -inf / -FLT_MAX / NaN onto the sentinel.
                    s = fmax(s, FA_NEG_INF);
                }
                ls[kk * FA_TILE_Q + p2_row] = (COMPUTE_FLOAT)s;
                pmax = fmax(pmax, s);
            }
            lred[p2_sub * FA_TILE_Q + p2_row] = pmax;
            barrier(CLK_LOCAL_MEM_FENCE);

            float m_blk = FA_NEG_INF;
            for (int t = 0; t < FA_TPR; ++t) {
                m_blk = fmax(m_blk, lred[t * FA_TILE_Q + p2_row]);
            }
            const float m_new = fmax(m_prev, m_blk);
            const float alpha = native_exp(m_prev - m_new);
            float psum = 0.0f;
            for (int j = 0; j < FA_KV_PER_THREAD; ++j) {
                const int kk = p2_beg + j;
                const float s = (float)ls[kk * FA_TILE_Q + p2_row];
                const float p = (s <= FA_NEG_TEST) ? 0.0f : native_exp(s - m_new);
                ls[kk * FA_TILE_Q + p2_row] = (COMPUTE_FLOAT)p;
                psum += p;
            }
            m_prev = m_new;
            // All reads of lred are done, so it can be reused now.
            barrier(CLK_LOCAL_MEM_FENCE);
            lred[p2_sub * FA_TILE_Q + p2_row] = psum;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (0 == p2_sub) {
                float l_blk = 0.0f;
                for (int t = 0; t < FA_TPR; ++t) {
                    l_blk += lred[t * FA_TILE_Q + p2_row];
                }
                ll[p2_row] = ll[p2_row] * alpha + l_blk;
                la[p2_row] = alpha;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // ---- phase 3: O = O * alpha + P.V ----
        // Guarded, but the barrier below stays outside: it has to be reached by every lane.
        if (p3_active) {
            out0 *= (COMPUTE_FLOAT)la[r0];
            out1 *= (COMPUTE_FLOAT)la[r0 + 1];
            out2 *= (COMPUTE_FLOAT)la[r0 + 2];
            out3 *= (COMPUTE_FLOAT)la[r0 + 3];
            // kv_end, not kv_seq_len: P is zero past it for every row of this tile.
            const int kv_valid = min(FA_TILE_KV, kv_end - k_start);
            __global const FLOAT* vp = past_value + value_base + k_start * FA_HEAD_DIM;
            for (int kk = 0; kk < kv_valid; ++kk) {
                COMPUTE_FLOAT4 p = vload4(0, ls + kk * FA_TILE_Q + r0);
#if (FA_HEAD_DIM % 8) == 0
                COMPUTE_FLOAT8 vv = CONVERT_COMPUTE_FLOAT8(vload8(0, vp + kk * FA_HEAD_DIM));
#else
                // V is [.. ][maxLen][headDim] with no padding on the dim axis, so the trailing
                // group cannot be read as a whole vload8. Compile-time branch, so this covers
                // every group and not just the short one -- p3_dims is 8 for all but the last.
                // The aligned case above keeps its vector load in this hot loop.
                COMPUTE_FLOAT8 vv = 0;
                {
                    __global const FLOAT* vrow = vp + kk * FA_HEAD_DIM;
                    vv.s0 = (COMPUTE_FLOAT)vrow[0];
                    if (p3_dims > 1) { vv.s1 = (COMPUTE_FLOAT)vrow[1]; }
                    if (p3_dims > 2) { vv.s2 = (COMPUTE_FLOAT)vrow[2]; }
                    if (p3_dims > 3) { vv.s3 = (COMPUTE_FLOAT)vrow[3]; }
                    if (p3_dims > 4) { vv.s4 = (COMPUTE_FLOAT)vrow[4]; }
                    if (p3_dims > 5) { vv.s5 = (COMPUTE_FLOAT)vrow[5]; }
                    if (p3_dims > 6) { vv.s6 = (COMPUTE_FLOAT)vrow[6]; }
                    if (p3_dims > 7) { vv.s7 = (COMPUTE_FLOAT)vrow[7]; }
                }
#endif
                out0 = mad((COMPUTE_FLOAT8)p.x, vv, out0);
                out1 = mad((COMPUTE_FLOAT8)p.y, vv, out1);
                out2 = mad((COMPUTE_FLOAT8)p.z, vv, out2);
                out3 = mad((COMPUTE_FLOAT8)p.w, vv, out3);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---- normalize and store ----
    // Past the last barrier, so the surplus lanes can leave here.
    if (!p3_active) {
        return;
    }
    const float l0 = ll[r0], l1 = ll[r0 + 1], l2 = ll[r0 + 2], l3 = ll[r0 + 3];
    out0 *= (COMPUTE_FLOAT)((l0 > 0.0f) ? (1.0f / l0) : 0.0f);
    out1 *= (COMPUTE_FLOAT)((l1 > 0.0f) ? (1.0f / l1) : 0.0f);
    out2 *= (COMPUTE_FLOAT)((l2 > 0.0f) ? (1.0f / l2) : 0.0f);
    out3 *= (COMPUTE_FLOAT)((l3 > 0.0f) ? (1.0f / l3) : 0.0f);

    const int q0 = q_start + r0;
#ifdef ATTENTION_C4
    const int channel = hn * FA_HEAD_DIM + d8;
    const int seq_storage = seq_len * batch;
    int token = b * seq_len + q0;
    if (q0 >= seq_len) return;
    store_attention_c4_8(output, CONVERT_FLOAT8(out0), seq_storage, token, channel, p3_dims);
    if (q0 + 1 >= seq_len) return;
    store_attention_c4_8(output, CONVERT_FLOAT8(out1), seq_storage, ++token, channel, p3_dims);
    if (q0 + 2 >= seq_len) return;
    store_attention_c4_8(output, CONVERT_FLOAT8(out2), seq_storage, ++token, channel, p3_dims);
    if (q0 + 3 >= seq_len) return;
    store_attention_c4_8(output, CONVERT_FLOAT8(out3), seq_storage, ++token, channel, p3_dims);
#else
    const int stride = head_num * FA_HEAD_DIM;
    const int offset = ((b * seq_len + q0) * head_num + hn) * FA_HEAD_DIM + d8;
    // A whole vstore8 on the trailing dim group would land on the next head's leading dims.
    if (q0 >= seq_len) return;
    STORE_FA_ROW(output, offset, out0, p3_dims);
    if (q0 + 1 >= seq_len) return;
    STORE_FA_ROW(output, offset + stride, out1, p3_dims);
    if (q0 + 2 >= seq_len) return;
    STORE_FA_ROW(output, offset + 2 * stride, out2, p3_dims);
    if (q0 + 3 >= seq_len) return;
    STORE_FA_ROW(output, offset + 3 * stride, out3, p3_dims);
#endif
}
