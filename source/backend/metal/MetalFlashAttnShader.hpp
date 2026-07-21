//
//  MetalFlashAttnShader.hpp
//  MNN
//
//  Fused prefill flash-attention Metal kernel.
//  Follows llama.cpp's kernel_flash_attn_ext_impl online-softmax layout,
//  adapted to MNN tensor layouts.
//
//  Preprocessor macros expected at pipeline compile time:
//    ftype           : half or float
//    HEAD_DIM        : 64, 128, or 256
//    GROUP_SIZE      : GQA group (num_heads / num_kv_heads)
//    HAS_MASK        : defined when ADD-type fp16 mask input is present at buffer(8)
//    ATTENTION_C4    : defined when output tensor is NC4HW4 [H*D/4, B*seq_q, 4]
//    QUANT_K         : K is int8 with per-token scale/bias at k_scales[kv_token * 2]
//                      (buffer(9) when defined)
//    QUANT_V         : V is int8 with per-token scale/bias at v_scales[kv_token * 2]
//                      (buffer(10) when defined)
//
//  Fixed tile constants:
//    Q_TILE  = 16   Q rows per threadgroup
//    KV_TILE = 32   K/V cols per streaming block (= 1 simdgroup width)
//    NSG     = 4    simdgroups per threadgroup (128 threads)
//  Each SG:
//    - QK output: 1 * 8 kv cols, spanning all 16 Q rows -> 2 stacked 8x8 accumulators
//    - Softmax: 4 Q rows, each row 32 cols = 1 lane per col
//    - PV output: NO_PER_SG d slices, each spanning all 16 Q rows -> 2 stacked 8x8 accumulators
//
//  Threadgroup memory (D=128): ~15 KB total, under 32 KB limit.
//    sq : 16 * 128 * 2 =  4 KB
//    sf : 16 *  32 * 4 =  2 KB
//    ss : 16 *  32 * 2 =  1 KB
//    so : 16 * 128 * 4 =  8 KB
//    (+ 512B sK/sV scratch when QUANT_K/QUANT_V)
//

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

const char* gPrefillFlashAttn = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

struct FaParam {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
    int mask_batch;
    int mask_head_num;
    int mask_q_len;
    int mask_k_len;
    float v_scale;
    float k_scale;
};

static inline long fa_mask_offset(constant FaParam& param, int b, int hn, int q, int k) {
    int mask_b = param.mask_batch <= 1 ? 0 : b;
    int mask_h = param.mask_head_num <= 1 ? 0 : hn;
    int mask_q = param.mask_q_len <= 1 ? 0 : min(q, param.mask_q_len - 1);
    int mask_k_start = max(param.key_seq_len - param.mask_k_len, 0);
    int local_k = param.mask_k_len <= 1 ? 0 : clamp(k - mask_k_start, 0, param.mask_k_len - 1);
    return ((long(mask_b) * param.mask_head_num + mask_h) * param.mask_q_len + mask_q) * (long)param.mask_k_len + local_k;
}

#define Q_TILE      16
#define KV_TILE     32
#define NSG         4
#define NQ_PER_SG   (Q_TILE / NSG)                    // 4
#define NO_PER_SG   ((HEAD_DIM / 8) / NSG)            // 2 (D=64) or 4 (D=128)

kernel void prefill_flash_attn(
    const device ftype* Q       [[buffer(0)]],
    device ftype* O             [[buffer(1)]],
#ifdef QUANT_K
    const device char* K        [[buffer(2)]],
#else
    const device ftype* K       [[buffer(2)]],
#endif
#ifdef QUANT_V
    const device char* V        [[buffer(3)]],
#else
    const device ftype* V       [[buffer(3)]],
#endif
    constant FaParam& param     [[buffer(4)]],
    constant int& seq_idx       [[buffer(5)]],
    constant int& kv_start_arg  [[buffer(6)]],
    constant int& kv_len_arg    [[buffer(7)]],
#ifdef HAS_MASK
    const device ftype* Mask    [[buffer(8)]],
#endif
#ifdef QUANT_K
    const device ftype* k_scales [[buffer(9)]],
#endif
#ifdef QUANT_V
    const device ftype* v_scales [[buffer(10)]],
#endif
    uint3 tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int block_q = int(tgpig.x);
    const int hb      = int(tgpig.y);
    const int b       = hb / param.head_num;
    const int h       = hb % param.head_num;
    const int kh      = h / param.group;

    const int seq_q            = param.query_seq_len;
    const int seq_k            = param.key_seq_len;
    const int kv_valid_offset  = seq_k - seq_q;
    const int q_row_base       = block_q * Q_TILE + seq_idx * param.q_seq_piece_len;
    const int kv_heads         = param.head_num / param.group;

    threadgroup half  sq[Q_TILE * HEAD_DIM];
    threadgroup float sf[Q_TILE * KV_TILE];
    threadgroup half  ss[Q_TILE * KV_TILE];
    threadgroup float so[Q_TILE * HEAD_DIM];
#ifdef QUANT_K
    // Per-SG 8x8 scratch for dequanting K int8 -> fp16 before simdgroup_load.
    threadgroup half sK[NSG * 8 * 8];
#endif
#ifdef QUANT_V
    threadgroup half sV[NSG * 8 * 8];
#endif

    const int tid = int(sgitg) * 32 + int(tiisg);

    // Zero O accumulator
    for (int i = tid; i < Q_TILE * HEAD_DIM; i += NSG * 32) {
        so[i] = 0.0f;
    }

    // Load Q tile
    for (int i = tid; i < Q_TILE * HEAD_DIM; i += NSG * 32) {
        int row   = i / HEAD_DIM;
        int col   = i % HEAD_DIM;
        int q_row = q_row_base + row;
        if (q_row < seq_q) {
            long q_off = ((long)(b * seq_q + q_row) * param.head_num + h) * param.head_dim + col;
            sq[i] = half(Q[q_off]);
        } else {
            sq[i] = half(0.0f);
        }
    }

    float M[NQ_PER_SG];
    float S[NQ_PER_SG];
    for (int j = 0; j < NQ_PER_SG; j++) {
        M[j] = -INFINITY;
        S[j] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int kv_end = kv_start_arg + kv_len_arg;
    for (int kv_block = kv_start_arg; kv_block < kv_end; kv_block += KV_TILE) {
        // Whole-tile causal early-exit
        int max_attend_k = q_row_base + Q_TILE - 1 + kv_valid_offset;
        if (kv_block > max_attend_k) {
            break;
        }

        // ============ (1) QK: Q * K^T -> sf ============
        // Each SG owns 8 kv cols, spans all 16 Q rows -> 2 stacked 8x8 accumulators.
        {
            const int col_base = int(sgitg) * 8;
            simdgroup_float8x8 mQK_top = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 mQK_bot = make_filled_simdgroup_matrix<float, 8>(0.0f);

            for (int k_step = 0; k_step < HEAD_DIM; k_step += 8) {
                simdgroup_half8x8 mQ_top;
                simdgroup_half8x8 mQ_bot;
                simdgroup_load(mQ_top, sq + k_step, HEAD_DIM);
                simdgroup_load(mQ_bot, sq + 8 * HEAD_DIM + k_step, HEAD_DIM);

                const int K_row_stride = param.batch * kv_heads * param.head_dim;
                simdgroup_half8x8 mK;
#ifdef QUANT_K
                // int8 K: dequant per-row (per-kv-token) into small tg scratch.
                // 8 rows x 8 cols = 64 halves per SG; first 8 lanes each handle one row.
                threadgroup half* my_sK = sK + int(sgitg) * 64;
                const device char* K_char = K
                    + (((kv_block + col_base) * param.batch + b) * kv_heads + kh) * param.head_dim
                    + k_step;
                if (int(tiisg) < 8) {
                    int row = int(tiisg);
                    int kv_tok = kv_block + col_base + row;
                    float ks = k_scales[kv_tok * 2];
                    float kb = k_scales[kv_tok * 2 + 1];
                    for (int c = 0; c < 8; c++) {
                        my_sK[row * 8 + c] = half(float(K_char[row * K_row_stride + c]) * ks + kb);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(mK, my_sK, 8, ulong2(0, 0), true);
#else
                const device ftype* K_ptr = K
                    + (((kv_block + col_base) * param.batch + b) * kv_heads + kh) * param.head_dim
                    + k_step;
                simdgroup_load(mK, K_ptr, K_row_stride, ulong2(0, 0), true);
#endif

                simdgroup_multiply_accumulate(mQK_top, mQ_top, mK, mQK_top);
                simdgroup_multiply_accumulate(mQK_bot, mQ_bot, mK, mQK_bot);
            }
            simdgroup_store(mQK_top, sf + col_base, KV_TILE);
            simdgroup_store(mQK_bot, sf + 8 * KV_TILE + col_base, KV_TILE);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============ (2) Online softmax + rescale so ============
        for (int j = 0; j < NQ_PER_SG; j++) {
            int q_row = int(sgitg) * NQ_PER_SG + j;
            int q_abs = q_row_base + q_row;

            float s = sf[q_row * KV_TILE + int(tiisg)] * param.scale;

            int kv_col_abs = kv_block + int(tiisg);
            bool in_bounds = (q_abs < seq_q)
                             && (kv_col_abs < seq_k)
                             && (kv_col_abs <= q_abs + kv_valid_offset);
            if (!in_bounds) {
                s = -INFINITY;
            }

#ifdef HAS_MASK
            if (in_bounds) {
                long mask_off = fa_mask_offset(param, b, h, q_abs, kv_col_abs);
                s += float(Mask[mask_off]);
            }
#endif

            float M_new = simd_max(fmax(M[j], s));
            float ms = (M[j] == -INFINITY) ? 0.0f : exp(M[j] - M_new);
            float vs = (s    == -INFINITY) ? 0.0f : exp(s    - M_new);
            S[j] = S[j] * ms + simd_sum(vs);
            M[j] = M_new;

            ss[q_row * KV_TILE + int(tiisg)] = half(vs);

            for (int d = int(tiisg); d < HEAD_DIM; d += 32) {
                so[q_row * HEAD_DIM + d] *= ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============ (3) P * V -> so ============
        {
            const int d_stride  = HEAD_DIM / NSG;
            const int d_base_sg = int(sgitg) * d_stride;
            for (int cc = 0; cc < NO_PER_SG; cc++) {
                int d_base = d_base_sg + cc * 8;

                simdgroup_float8x8 mO_top;
                simdgroup_float8x8 mO_bot;
                simdgroup_load(mO_top, so + d_base, HEAD_DIM);
                simdgroup_load(mO_bot, so + 8 * HEAD_DIM + d_base, HEAD_DIM);

                for (int k_step = 0; k_step < KV_TILE; k_step += 8) {
                    simdgroup_half8x8 mP_top;
                    simdgroup_half8x8 mP_bot;
                    simdgroup_load(mP_top, ss + k_step, KV_TILE);
                    simdgroup_load(mP_bot, ss + 8 * KV_TILE + k_step, KV_TILE);

                    const int V_row_stride = param.max_kv_len;
                    simdgroup_half8x8 mV;
#ifdef QUANT_V
                    // int8 V: dequant per-column (per-kv-token) into small tg scratch.
                    // 8 rows (d) x 8 cols (kv positions); first 8 lanes each handle one col.
                    threadgroup half* my_sV = sV + int(sgitg) * 64;
                    const device char* V_char = V
                        + ((b * kv_heads + kh) * param.head_dim + d_base) * param.max_kv_len
                        + kv_block + k_step;
                    if (int(tiisg) < 8) {
                        int c = int(tiisg);
                        int kv_tok = kv_block + k_step + c;
                        float vs = v_scales[kv_tok * 2];
                        float vb = v_scales[kv_tok * 2 + 1];
                        for (int r = 0; r < 8; r++) {
                            my_sV[r * 8 + c] = half(float(V_char[r * V_row_stride + c]) * vs + vb);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(mV, my_sV, 8, ulong2(0, 0), true);
#else
                    const device ftype* V_ptr = V
                        + ((b * kv_heads + kh) * param.head_dim + d_base) * param.max_kv_len
                        + kv_block + k_step;
                    simdgroup_load(mV, V_ptr, V_row_stride, ulong2(0, 0), true);
#endif

                    simdgroup_multiply_accumulate(mO_top, mP_top, mV, mO_top);
                    simdgroup_multiply_accumulate(mO_bot, mP_bot, mV, mO_bot);
                }

                simdgroup_store(mO_top, so + d_base, HEAD_DIM);
                simdgroup_store(mO_bot, so + 8 * HEAD_DIM + d_base, HEAD_DIM);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ============ Epilogue: normalize + write O ============
    for (int j = 0; j < NQ_PER_SG; j++) {
        int q_row = int(sgitg) * NQ_PER_SG + j;
        int q_abs = q_row_base + q_row;
        if (q_abs >= seq_q) continue;

        float inv_S = (S[j] > 0.0f) ? (1.0f / S[j]) : 0.0f;
        for (int d = int(tiisg); d < HEAD_DIM; d += 32) {
            float v = so[q_row * HEAD_DIM + d] * inv_S;
#ifdef ATTENTION_C4
            long o_off = (long)(h * (param.head_dim / 4) + (d / 4)) * 4 * param.batch * seq_q
                       + (long)(b * seq_q + q_abs) * 4
                       + (d & 3);
#else
            long o_off = ((long)(b * seq_q + q_abs) * param.head_num + h) * param.head_dim + d;
#endif
            O[o_off] = ftype(v);
        }
    }
}
)metal";

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
