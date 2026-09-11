//
//  MetalFlashAttnShader.hpp
//  MNN
//
//  Fused prefill flash-attention Metal kernel.
//  Uses an online-softmax layout adapted to MNN tensor layouts.
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

                    const int V_row_stride = param.batch * kv_heads * param.head_dim;
                    simdgroup_half8x8 mV;
#ifdef QUANT_V
                    // int8 V: dequant per-row (per-kv-token) into small tg scratch.
                    // 8 rows (kv positions) x 8 cols (d); first 8 lanes each handle one row.
                    threadgroup half* my_sV = sV + int(sgitg) * 64;
                    const device char* V_char = V
                        + (((kv_block + k_step) * param.batch + b) * kv_heads + kh) * param.head_dim
                        + d_base;
                    if (int(tiisg) < 8) {
                        int row = int(tiisg);
                        int kv_tok = kv_block + k_step + row;
                        float vs = v_scales[kv_tok * 2];
                        float vb = v_scales[kv_tok * 2 + 1];
                        for (int c = 0; c < 8; c++) {
                            my_sV[row * 8 + c] = half(float(V_char[row * V_row_stride + c]) * vs + vb);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(mV, my_sV, 8);
#else
                    const device ftype* V_ptr = V
                        + (((kv_block + k_step) * param.batch + b) * kv_heads + kh) * param.head_dim
                        + d_base;
                    simdgroup_load(mV, V_ptr, V_row_stride);
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

//  ---------------------------------------------------------------------------
//  prefill_flash_attn_tc -- fused prefill attention on the Metal tensor API.
//
//  Why the tensor API: an 8x8 simdgroup_matrix variant of this fused structure
//  (prefill_flash_attn_v2, since removed) was numerically correct but lost
//  to the three-stage path, because 8x8 simdgroup_matrix delivers about
//  half the FLOP efficiency of matmul2d for these shapes (see
//  skills/metal-optimize/kernel-dev-and-optimize.md 2.3.3). This version keeps the same
//  fused structure -- S and O resident in registers, scores never written to
//  global memory -- but does the matmuls with matmul2d.
//
//  matmul2d input cooperative tensors are only allowed at single-simdgroup
//  scope, so each simdgroup runs its own 16x32x16 matmul and every operand is a
//  cooperative tensor. That is what lets the QK destination become the PV left
//  operand in registers.
//
//  Per-lane element layout of those cooperative tensors is dumped and documented
//  in skills/metal-optimize/kernel-dev-and-optimize.md; the two facts this kernel leans on
//  are that a lane owns exactly two M rows (fm and fm+8) and that row-mates
//  differ only in lane bits 0 and 3.
//
//  Macros: ftype, HEAD_DIM (64/128/256), GROUP_SIZE, ATTENTION_C4, HAS_MASK,
//  FATC_QK_K32 (QK accumulate width 32, requires HEAD_DIM % 32 == 0).
//  Tiles: 64 queries/threadgroup over 4 simdgroups (16 rows each), 32 kv/step.
//  Zero threadgroup memory: Q/K/V go straight from device to registers.
//  ---------------------------------------------------------------------------
const char* gPrefillFlashAttnTc = R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

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

static inline long fatc_mask_offset(constant FaParam& param, int b, int hn, int q, int k) {
    int mask_b = param.mask_batch <= 1 ? 0 : b;
    int mask_h = param.mask_head_num <= 1 ? 0 : hn;
    int mask_q = param.mask_q_len <= 1 ? 0 : min(q, param.mask_q_len - 1);
    int mask_k_start = max(param.key_seq_len - param.mask_k_len, 0);
    int local_k = param.mask_k_len <= 1 ? 0 : clamp(k - mask_k_start, 0, param.mask_k_len - 1);
    return ((long(mask_b) * param.mask_head_num + mask_h) * param.mask_q_len + mask_q) * (long)param.mask_k_len + local_k;
}

#define FATC_NSG   4
#define FATC_BQ    64                   // queries per threadgroup (16 per simdgroup)
// kv columns per step (= matmul2d N for QK). Do not widen this to buy scalar
// moves: the QK destination capacity tracks N, so BK=64 doubles s_acc and pushes
// the lane past an occupancy cliff, which costs more than the moves it saves.
// Widening K is the free direction (capacity is K-independent) -- see FATC_QK_K.
#define FATC_BK    32
#define FATC_TK    (FATC_BK / 16)      // kv frags
#define FATC_TD    (HEAD_DIM / 16)      // head_dim frags
#define FATC_NEG   (-1.0e30f)

// QK destination capacity per lane, and the layout it implies. Element bits
// 0..3 index within the 32-wide tile as
//   m = fm + ((i>>2)&1)*8
//   n = fn + (i&3) + ((i>>3)&1)*16
#define FATC_SCAP  (FATC_BK / 2)

// FATC_DSPLIT: split head_dim across the dispatch grid's z dimension, so a
// threadgroup owns 64 q rows x FATC_DDIM output dims instead of all HEAD_DIM.
// The persistent O accumulator is HEAD_DIM/32 destination CTs of 16 floats each,
// i.e. 128 float registers per lane at HEAD_DIM=256 -- past Apple's ~128-reg
// full-occupancy budget, which is why the naive 256 generalization collapses
// under register pressure. Splitting z-ways divides that by
// FATC_DSPLIT. The cost is that QK is recomputed per z slice (1.5x total FLOPs
// at DSPLIT=2), but it needs no barrier and no cross-simdgroup traffic, unlike
// every staging variant tried here.
#ifndef FATC_DSPLIT
#define FATC_DSPLIT 1
#endif
#define FATC_DDIM (HEAD_DIM / FATC_DSPLIT)
#if FATC_DSPLIT > 1 && !defined(FATC_O_CT)
#error "FATC_DSPLIT requires FATC_O_CT (only the persistent-CT PV path is d-sliced)"
#endif

// 32-wide O tiles, one destination cooperative tensor each. The per-tile code
// (declare / zero / softmax rescale / PV accumulate / epilogue unpack) is
// identical, so it is written once and expanded through FATC_O_FOREACH; a
// cooperative tensor is not default-constructible, so an array is not an option.
#define FATC_TO (FATC_DDIM / 32)
#if FATC_TO == 1
#define FATC_O_FOREACH(F) F(0, ct_o0)
#elif FATC_TO == 2
#define FATC_O_FOREACH(F) F(0, ct_o0) F(1, ct_o1)
#elif FATC_TO == 4
#define FATC_O_FOREACH(F) F(0, ct_o0) F(1, ct_o1) F(2, ct_o2) F(3, ct_o3)
#elif FATC_TO == 8
#define FATC_O_FOREACH(F) F(0, ct_o0) F(1, ct_o1) F(2, ct_o2) F(3, ct_o3) \
                           F(4, ct_o4) F(5, ct_o5) F(6, ct_o6) F(7, ct_o7)
#else
#error "prefill_flash_attn_tc supports HEAD_DIM 64 / 128 / 256"
#endif

// QK accumulate width. K=32 (FATC_QK_K32) halves the head_dim loop: each
// matmul2d covers two consecutive 16-K frags. The K=32 coop layout is the
// K=16 layout replicated along K with a +16 offset in the upper half:
//   A: K = fn + (i&3) + ((i>>3)&1)*16, M = fm + ((i>>2)&1)*8        (cap 16)
//   B: K = fn + (i&3) + ((i>>4)&1)*16, N = fm + ((i>>2)&1)*8 + ((i>>3)&1)*16 (cap 32)
//   D: bit-identical to K=16 (cap 16) -- softmax / PV handoff untouched.
#ifdef FATC_QK_K32
#define FATC_QK_K 32
#else
#define FATC_QK_K 16
#endif
#define FATC_NH    (FATC_QK_K / 16)        // 16-K halves per QK call
#define FATC_TDK   (HEAD_DIM / FATC_QK_K)  // QK accumulate steps

// PV accumulate width. K=32 (FATC_PV_K32) consumes the whole BK=32 kv tile in
// one matmul2d, so the PV call count drops from FATC_TK*(FATC_TD/2) to
// FATC_TD/2 (8 -> 4 at HEAD_DIM=128). The PV slot (A=float B=half,
// tb=false) K=32 again just adds a +16 K level on top of the K=16 layout:
//   A: K = kv frag as in K=16, so element i is exactly s_acc[i]        (cap 16)
//   B: N = fn + f*16 + j, K = fm + g*8 + ((i>>4)&1)*16                 (cap 32)
//   D: bit-identical to K=16 (cap 16).
#ifdef FATC_PV_K32
#define FATC_PV_K 32
#else
#define FATC_PV_K 16
#endif
#define FATC_PV_NH (FATC_PV_K / 16)        // 16-K halves per PV call
#if defined(FATC_PV_K32) && !defined(FATC_O_CT)
#error "FATC_PV_K32 requires FATC_O_CT (only the persistent-CT PV path is wide-K)"
#endif

// FATC_KV_DEV_TENSOR: hand K/V to matmul2d as `tensor` handle slices pointing
// straight at device memory (strided tensor_inline over the kv cache), instead
// of hand-filling the right input cooperative tensor. The matmul then issues the
// operand loads itself, so the per-lane B fill -- BK*HEAD_DIM/32 = 128 half
// moves per lane per kv tile for K plus another 128 for V, the single largest
// scalar cost in this kernel and one that no BK/K widening can reduce --
// disappears, and nothing is staged, so there is no barrier and no need to widen
// the kv loop to the threadgroup's max q row.
//
// Staging K/V through threadgroup smem first was tried twice and lost both
// times: once still hand-packing every ct_b out of smem, once as a staged
// tensor-handle form that dropped the packing but kept paying stage + barrier +
// a kv loop widened to the threadgroup's max q row. Reading in place is what
// makes the idea pay.
#ifdef FATC_KV_DEV_TENSOR
#ifndef FATC_O_CT
#error "FATC_KV_DEV_TENSOR requires FATC_O_CT (only the persistent-CT PV path is generalized)"
#endif
#endif

// QK uses transpose_b (K stored [kv][d], supplied as [n=kv][k=d]); PV uses
// non-transposed B now that the V cache is row-major [kv][d] as well, so both
// device reads stay contiguous along d (4 consecutive halves per lane).
#define FATC_DESC_QK matmul2d_descriptor(16, FATC_BK, FATC_QK_K, false, true, true, \
                                       matmul2d_descriptor::mode::multiply_accumulate)
#define FATC_DESC_PV matmul2d_descriptor(16, 32, FATC_PV_K, false, false, true, \
                                       matmul2d_descriptor::mode::multiply_accumulate)

kernel void prefill_flash_attn_tc(
    const device ftype* Q       [[buffer(0)]],
    device ftype* O             [[buffer(1)]],
    const device ftype* K       [[buffer(2)]],
    const device ftype* V       [[buffer(3)]],
    constant FaParam& param     [[buffer(4)]],
    constant int& seq_idx       [[buffer(5)]],
#ifdef HAS_MASK
    const device ftype* Mask    [[buffer(8)]],
#endif
    uint3 tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]])
{
    const int hb = int(tgpig.y);
    const int b  = hb / param.head_num;
    const int h  = hb % param.head_num;
    const int kh = h / param.group;

    const int seq_q = param.query_seq_len;
    const int seq_k = param.key_seq_len;
    const int kv_valid_offset = seq_k - seq_q;
    const int kv_heads = param.head_num / param.group;
    // Each simdgroup owns 16 consecutive query rows of this threadgroup's 64.
#ifdef FATC_QREV
    // Causal cost per threadgroup grows with the q-tile index (its kv range is
    // q_row_base+16 long), but the grid issues x-major, so the cheapest tiles
    // launch first and the longest ones last -- the tail runs a few long
    // threadgroups with the machine mostly idle. Reversing the mapping issues
    // the longest tiles first and leaves cheap ones to fill the tail. This is
    // a permutation of the same tile set, so every q row is still covered once.
    const int q_tiles = (param.q_seq_piece_len + FATC_BQ - 1) / FATC_BQ;
    const int q_tile  = q_tiles - 1 - int(tgpig.x);
#else
    const int q_tile  = int(tgpig.x);
#endif
    const int q_row_base = q_tile * FATC_BQ + seq_idx * param.q_seq_piece_len
                         + int(sgitg) * 16;
#if FATC_DSPLIT > 1
    // This threadgroup produces only [q_row_base, +16) x [d_base, +FATC_DDIM).
    // QK spans the full head_dim regardless, so the z slices duplicate it.
    const int d_base = int(tgpig.z) * FATC_DDIM;
#else
    constexpr int d_base = 0;
#endif

    // Per-lane base offsets inside a cooperative tensor tile (kernel-dev-and-optimize.md §1.10):
    // fn indexes 4 consecutive elements of the fast axis, fm the slow axis.
    const ushort qid = tiisg >> 2;
    const ushort fm  = (qid & 4) | ((tiisg >> 1) & 3);
    const ushort fn  = ((qid & 2) | (tiisg & 1)) * 4;

    matmul2d<FATC_DESC_QK, metal::execution_simdgroup> mm;
    matmul2d<FATC_DESC_PV, metal::execution_simdgroup> mm_pv;

#ifdef FATC_KV_DEV_TENSOR
    // Handles span all seq_k rows of this kv head, so slice() also does the
    // tail-row edge checking that the staged path had to do by hand.
    const int kv_row_stride_t = param.batch * kv_heads * param.head_dim;
    const long kv_head_off = ((long)b * kv_heads + kh) * param.head_dim;
    const array<int, 2> kv_strides = {1, kv_row_stride_t};
    // Element type must be non-const: tensor_ops only accepts unqualified
    // element types. The handles are still only ever read.
    auto t_k = tensor<device ftype, dextents<int32_t, 2>, tensor_inline>(
        (device ftype*)(K + kv_head_off), dextents<int32_t, 2>(param.head_dim, seq_k), kv_strides);
    auto t_v = tensor<device ftype, dextents<int32_t, 2>, tensor_inline>(
        (device ftype*)(V + kv_head_off), dextents<int32_t, 2>(param.head_dim, seq_k), kv_strides);
#endif

#ifdef FATC_O_CT
    // O accumulator lives in a persistent cooperative_tensor destination per
    // 32-D tile: values stay in registers across the
    // entire kv-loop instead of round-tripping through a float array.
    // Grab reference input CTs once so we can spell out the destination CT
    // type via decltype(). These variables are shadowed inside the QK/PV
    // loops so their register footprint is transient.
    auto ct_a_ref = mm_pv.get_left_input_cooperative_tensor<float, ftype, float>();
#ifdef FATC_KV_DEV_TENSOR
    // The destination CT type depends on both operand types, so the reference B
    // must be the same tensor-slice type that run() will be handed below.
    auto ct_b_ref = t_v.slice(0, 0);
#else
    auto ct_b_ref = mm_pv.get_right_input_cooperative_tensor<float, ftype, float>();
#endif
    (void)ct_a_ref; (void)ct_b_ref;
#define FATC_DECL_O(IDX, CT) \
    auto CT = mm_pv.get_destination_cooperative_tensor<decltype(ct_a_ref), decltype(ct_b_ref), float>();
    FATC_O_FOREACH(FATC_DECL_O)
#undef FATC_DECL_O
    // Zero the persistent destination CTs.
#define FATC_ZERO_O(IDX, CT) CT[i] = 0.0f;
    for (ushort i = 0; i < 16; i++) {
        FATC_O_FOREACH(FATC_ZERO_O)
    }
#undef FATC_ZERO_O
#else
    // O accumulator: HEAD_DIM/32 destination tiles of 16x32, 16 floats each.
    float o_acc[FATC_TO][16];
    for (int t = 0; t < FATC_TO; t++) {
        for (int i = 0; i < 16; i++) {
            o_acc[t][i] = 0.0f;
        }
    }
#endif
    // A lane owns exactly two query rows (fm and fm+8), so the softmax state is
    // two running maxima and two running sums.
    float run_max[2] = {FATC_NEG, FATC_NEG};
    float run_sum[2] = {0.0f, 0.0f};

    // exp2-based softmax wants logits in log2 units; fold scale*log2(e) into Q.
    const float qscale = param.scale * M_LOG2E_F;

    // Hoisted addressing: all offsets below are 32-bit and loop-invariant except
    // for the tile index, and every gather reads 4 consecutive halves.
    const int q_row_stride = param.head_num * param.head_dim;
    const int k_row_stride = param.batch * kv_heads * param.head_dim;
    const int v_row_stride = param.batch * kv_heads * param.head_dim;
    const device ftype* q_lane = Q + ((long)b * seq_q * param.head_num + h) * param.head_dim
                                  + (long)(q_row_base + int(fm)) * q_row_stride + int(fn);
    const device ftype* k_head = K + ((long)b * kv_heads + kh) * param.head_dim + int(fn);
    const device ftype* v_head = V + ((long)b * kv_heads + kh) * param.head_dim + int(fn);

    // Tile-level causal bounds: blocks fully above the diagonal never run, and
    // blocks fully below it skip masking entirely.
    const int kv_lim = min(seq_k, q_row_base + 16 + kv_valid_offset);
    const int kv_no_mask_end = q_row_base + kv_valid_offset;
    const int q_valid_rows = seq_q - q_row_base;   // rows >= this are padding

#if defined(FATC_Q_CT)
    // Q is invariant across the kv loop, so build the QK left-input CTs here and
    // reuse them for every tile. faTcQReg only got Q as far as a register array;
    // each tile still repacked it into a fresh input CT. The CTs hold the same
    // HEAD_DIM/2 halves per lane that q_reg did, so this replaces that storage
    // rather than adding to it.
    // One left-input CT per QK K-step. Same storage the q_reg array used, so the
    // list only has to be spelled out; FATC_Q_CT implies the wide-K QK, hence
    // FATC_TDK = HEAD_DIM/32 in {2, 4, 8}.
#if FATC_TDK == 2
#define FATC_Q_FOREACH(F) F(0, ct_q0) F(1, ct_q1)
#elif FATC_TDK == 4
#define FATC_Q_FOREACH(F) F(0, ct_q0) F(1, ct_q1) F(2, ct_q2) F(3, ct_q3)
#elif FATC_TDK == 8
#define FATC_Q_FOREACH(F) F(0, ct_q0) F(1, ct_q1) F(2, ct_q2) F(3, ct_q3) \
                           F(4, ct_q4) F(5, ct_q5) F(6, ct_q6) F(7, ct_q7)
#else
#error "FATC_Q_CT expects FATC_TDK in {2, 4, 8}"
#endif
#define FATC_FILL_Q(dd, ctq)                                                                      \
    {                                                                                              \
        for (ushort h = 0; h < FATC_NH; h++) {                                                    \
            const device ftype* qp = q_lane + ((dd) * FATC_NH + int(h)) * 16;                     \
            for (ushort g = 0; g < 2; g++) {                                                       \
                const bool row_ok = (int(fm) + g * 8) < q_valid_rows;                              \
                ftype4 qv = row_ok ? *(const device ftype4*)(qp + g * 8 * q_row_stride) : ftype4(0);\
                for (ushort j = 0; j < 4; j++) {                                                   \
                    ctq[h * 8 + g * 4 + j] = ftype(float(qv[j]) * qscale);                         \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }
#define FATC_DECL_Q(IDX, CT) auto CT = mm.get_left_input_cooperative_tensor<ftype, ftype, float>();
    FATC_Q_FOREACH(FATC_DECL_Q)
#undef FATC_DECL_Q
    FATC_Q_FOREACH(FATC_FILL_Q)
#undef FATC_FILL_Q
#elif defined(FATC_Q_REG)
    // Load and pre-scale Q for this q-tile once; reuse across every kv0 tile.
    // Footprint per lane is FATC_TD*8 halves = HEAD_DIM/2 bytes: 128 B at
    // HEAD_DIM=128, 256 B at HEAD_DIM=256. Watch occupancy if HEAD_DIM grows.
    ftype q_reg[FATC_TD][8];
    for (int dd = 0; dd < FATC_TD; dd++) {
        const device ftype* qp = q_lane + dd * 16;
        for (ushort g = 0; g < 2; g++) {
            const bool row_ok = (int(fm) + g * 8) < q_valid_rows;
            ftype4 qv = row_ok ? *(const device ftype4*)(qp + g * 8 * q_row_stride) : ftype4(0);
            for (ushort j = 0; j < 4; j++) {
                q_reg[dd][g * 4 + j] = ftype(float(qv[j]) * qscale);
            }
        }
    }
#endif

    for (int kv0 = 0; kv0 < kv_lim; kv0 += FATC_BK) {
        // ---- QK: S[16 q][FATC_BK kv], accumulated over head_dim ----
        float s_acc[FATC_SCAP];
        for (int i = 0; i < FATC_SCAP; i++) {
            s_acc[i] = 0.0f;
        }
        for (int dd = 0; dd < FATC_TDK; dd++) {
#ifdef FATC_Q_CT
#if FATC_TDK == 2
            thread auto& ct_a = (dd == 0) ? ct_q0 : ct_q1;
#elif FATC_TDK == 4
            thread auto& ct_a = (dd == 0) ? ct_q0 : (dd == 1) ? ct_q1 : (dd == 2) ? ct_q2 : ct_q3;
#else
            thread auto& ct_a = (dd == 0) ? ct_q0 : (dd == 1) ? ct_q1 : (dd == 2) ? ct_q2
                              : (dd == 3) ? ct_q3 : (dd == 4) ? ct_q4 : (dd == 5) ? ct_q5
                              : (dd == 6) ? ct_q6 : ct_q7;
#endif
#else
            auto ct_a = mm.get_left_input_cooperative_tensor<ftype, ftype, float>();
#endif
#ifdef FATC_KV_DEV_TENSOR
            // B operand is the K tile itself: origin (k = dd*QK_K, n = this
            // tile's first kv row), extents from the descriptor. No per-lane fill.
            auto ct_b = t_k.slice(dd * FATC_QK_K, kv0);
#else
            auto ct_b = mm.get_right_input_cooperative_tensor<ftype, ftype, float>();
#endif
#ifdef FATC_Q_CT
            auto ct_c = mm.get_destination_cooperative_tensor<decltype(ct_q0), decltype(ct_b), float>();
#else
            auto ct_c = mm.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
#endif

#if defined(FATC_Q_CT)
            // A was filled once before the kv loop.
#elif defined(FATC_Q_REG)
            // K=32 extends the K=16 A layout along K: element i sits at
            // (k = fn + (i&3) + ((i>>3)&1)*16, m = fm + ((i>>2)&1)*8), i.e.
            // 16-K frag (i>>3) of this step.
            for (ushort i = 0; i < FATC_QK_K / 2; i++) {
                ct_a[i] = q_reg[dd * FATC_NH + (i >> 3)][i & 7];
            }
#else
            for (ushort h = 0; h < FATC_NH; h++) {
                // A[m=q][k=d]: k = fn + (j&3) + h*16, m = fm + (j>>2)*8
                const device ftype* qp = q_lane + (dd * FATC_NH + h) * 16;
                for (ushort g = 0; g < 2; g++) {
                    const bool row_ok = (int(fm) + g * 8) < q_valid_rows;
                    ftype4 qv = row_ok ? *(const device ftype4*)(qp + g * 8 * q_row_stride) : ftype4(0);
                    for (ushort j = 0; j < 4; j++) {
                        ct_a[h * 8 + g * 4 + j] = ftype(float(qv[j]) * qscale);
                    }
                }
            }
#endif
#if defined(FATC_KV_DEV_TENSOR)
            // no B fill: matmul2d reads the K tile directly
#else
            // B stored [n=kv][k=d]: k = fn + (j&3) + h*16, n = fm + g*8 + f*16
            for (ushort h = 0; h < FATC_NH; h++) {
                const device ftype* kp = k_head + (dd * FATC_NH + h) * 16;
                for (ushort f = 0; f < 2; f++) {
                for (ushort g = 0; g < 2; g++) {
                        const int kv = kv0 + int(fm) + g * 8 + f * 16;
                        ftype4 kvv = (kv < seq_k) ? *(const device ftype4*)(kp + kv * k_row_stride) : ftype4(0);
                    for (ushort j = 0; j < 4; j++) {
                            ct_b[h * 16 + f * 8 + g * 4 + j] = kvv[j];
                        }
                    }
                }
            }
#endif
            // D capacity is K-independent (it tracks N), so the accumulate
            // carry-in fill is the same for both QK widths.
            for (ushort i = 0; i < FATC_SCAP; i++) {
                ct_c[i] = s_acc[i];
            }
            mm.run(ct_a, ct_b, ct_c);
            for (ushort i = 0; i < FATC_SCAP; i++) {
                s_acc[i] = ct_c[i];
            }
        }

        // ---- mask, in registers. In a destination tile element i sits at
        // (m = fm + ((i>>2)&1)*8, n = fn + (i&3) + ((i>>3)&1)*16).
        const bool tile_all_valid = (kv0 + FATC_BK - 1) <= kv_no_mask_end && (kv0 + FATC_BK) <= seq_k;
        if (!tile_all_valid) {
        for (ushort i = 0; i < FATC_SCAP; i++) {
                const int qr = q_row_base + int(fm) + ((i >> 2) & 1) * 8;
                const int kv = kv0 + int(fn) + (i & 3) + ((i >> 3) & 1) * 16;
                const bool ok = (kv < seq_k) && (kv <= qr + kv_valid_offset);
                s_acc[i] = ok ? s_acc[i] : FATC_NEG;
            }
#ifdef HAS_MASK
        for (ushort i = 0; i < FATC_SCAP; i++) {
                const int qr = q_row_base + int(fm) + ((i >> 2) & 1) * 8;
                const int kv = kv0 + int(fn) + (i & 3) + ((i >> 3) & 1) * 16;
                if (s_acc[i] > FATC_NEG && kv < seq_k && qr < seq_q) {
                    s_acc[i] += M_LOG2E_F * float(Mask[fatc_mask_offset(param, b, h, qr, kv)]);
                }
            }
#endif
        }

        // ---- online softmax; element i belongs to row half (i>>2)&1 ----
        float m_new[2] = {run_max[0], run_max[1]};
        for (ushort i = 0; i < FATC_SCAP; i++) {
            const ushort r = (i >> 2) & 1;
            m_new[r] = max(m_new[r], s_acc[i]);
        }
        float tile_sum[2] = {0.0f, 0.0f};
        float factor[2];
    for (int r = 0; r < 2; r++) {
            // Row-mates are the lanes differing in bits 0 and 3.
            m_new[r] = max(m_new[r], simd_shuffle_xor(m_new[r], 1u));
            m_new[r] = max(m_new[r], simd_shuffle_xor(m_new[r], 8u));
        }
        for (ushort i = 0; i < FATC_SCAP; i++) {
            const ushort r = (i >> 2) & 1;
            const float p = exp2(s_acc[i] - m_new[r]);
            s_acc[i] = p;
            tile_sum[r] += p;
        }
    for (int r = 0; r < 2; r++) {
            tile_sum[r] += simd_shuffle_xor(tile_sum[r], 1u);
            tile_sum[r] += simd_shuffle_xor(tile_sum[r], 8u);
            factor[r] = (run_max[r] == FATC_NEG) ? 0.0f : exp2(run_max[r] - m_new[r]);
            run_sum[r] = run_sum[r] * factor[r] + tile_sum[r];
            run_max[r] = m_new[r];
        }
#ifdef FATC_O_CT
        // Rescale each persistent O destination CT by the online-softmax factor.
#define FATC_RESCALE_O(IDX, CT) CT[i] *= f;
        for (ushort i = 0; i < 16; i++) {
            const float f = factor[(i >> 2) & 1];
            FATC_O_FOREACH(FATC_RESCALE_O)
        }
#undef FATC_RESCALE_O
#else
    for (int t = 0; t < FATC_TO; t++) {
        for (ushort i = 0; i < 16; i++) {
                o_acc[t][i] *= factor[(i >> 2) & 1];
            }
        }
#endif

#ifdef FATC_O_CT
        // Persistent-CT PV: run() accumulates into ct_oX directly, so the
        // per-iteration ct_c pack/unpack goes away and O never leaves regs.
        // FATC_PV_NH kv frags per call, so FATC_TK/FATC_PV_NH steps.
        for (int ik = 0; ik < FATC_TK / FATC_PV_NH; ik++) {
            auto ct_a = mm_pv.get_left_input_cooperative_tensor<float, ftype, float>();
            for (ushort j = 0; j < 8 * FATC_PV_NH; j++) {
                ct_a[j] = s_acc[ik * 8 * FATC_PV_NH + j];
            }
#ifdef FATC_KV_DEV_TENSOR
            // V straight from the tile: origin (n = T_IDX*32 along head_dim,
            // k = this call's kv frag base).
            // run() takes non-const lvalue refs, so the slice must be named.
#define FATC_PV_ONE(T_IDX, CT_O) \
            { \
                auto ct_b = t_v.slice(d_base + (T_IDX) * 32, kv0 + ik * FATC_PV_K); \
                mm_pv.run(ct_a, ct_b, CT_O); \
            }
#else
#define FATC_PV_ONE(T_IDX, CT_O) \
            { \
                auto ct_b = mm_pv.get_right_input_cooperative_tensor<float, ftype, float>(); \
                for (ushort hh = 0; hh < FATC_PV_NH; hh++) { \
                for (ushort f = 0; f < 2; f++) { \
                    for (ushort g = 0; g < 2; g++) { \
                        const int kv = kv0 + (ik * FATC_PV_NH + hh) * 16 + int(fm) + g * 8; \
                        ftype4 vv = *(const device ftype4*)(v_head + (long)kv * v_row_stride + d_base + (T_IDX) * 32 + f * 16); \
                        for (ushort j = 0; j < 4; j++) { \
                            const ushort i = hh * 16 + f * 8 + g * 4 + j; \
                            ct_b[i] = vv[j]; \
                        } \
                    } \
                } \
                } \
                mm_pv.run(ct_a, ct_b, CT_O); \
            }
#endif
            FATC_O_FOREACH(FATC_PV_ONE)
#undef FATC_PV_ONE
        }
#else
        // ---- PV: O[16 q][32 d] += P[16 q][16 kv] * V ----
        // P needs no shuffling: the QK destination element carrying (m, kv) is
        // exactly the one the PV left operand wants at (m, k=kv).
        for (int ik = 0; ik < FATC_TK; ik++) {
    for (int t = 0; t < FATC_TD / 2; t++) {
                auto ct_a = mm_pv.get_left_input_cooperative_tensor<float, ftype, float>();
                auto ct_b = mm_pv.get_right_input_cooperative_tensor<float, ftype, float>();
                auto ct_c = mm_pv.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

                for (ushort j = 0; j < 8; j++) {
                    ct_a[j] = s_acc[ik * 8 + (j >> 2) * 4 + (j & 3)];
                }
                // B stored [k=kv][n=d]: k = kv0 + ik*16 + fm + g*8,
                // n = d = t*32 + fn + f*16 + (j&3) (4 consecutive)
            for (ushort f = 0; f < 2; f++) {
            for (ushort g = 0; g < 2; g++) {
                        const int kv = kv0 + ik * 16 + int(fm) + g * 8;
                        ftype4 vv = *(const device ftype4*)(v_head + (long)kv * v_row_stride + t * 32 + f * 16);
                for (ushort j = 0; j < 4; j++) {
                            const ushort i = f * 8 + g * 4 + j;
                            ct_b[i] = vv[j];
                            ct_c[i] = o_acc[t][i];
                        }
                    }
                }
                mm_pv.run(ct_a, ct_b, ct_c);
        for (ushort i = 0; i < 16; i++) {
                    o_acc[t][i] = ct_c[i];
                }
            }
        }
#endif
    }

    // ---- epilogue: normalize per row, then write out ----
    float inv_s[2];
    for (int r = 0; r < 2; r++) {
        inv_s[r] = (run_sum[r] > 0.0f) ? (1.0f / run_sum[r]) : 0.0f;
    }
#ifdef FATC_O_CT
    // Unpack the persistent destination CTs into a scalar array once, so the
    // existing store loop below stays generic.
    float o_acc[FATC_TO][16];
#define FATC_UNPACK_O(IDX, CT) o_acc[IDX][i] = CT[i];
    for (ushort i = 0; i < 16; i++) {
        FATC_O_FOREACH(FATC_UNPACK_O)
    }
#undef FATC_UNPACK_O
#endif
    for (int t = 0; t < FATC_TO; t++) {
        for (ushort i = 0; i < 16; i++) {
            const ushort r = (i >> 2) & 1;
            const int qr = q_row_base + int(fm) + r * 8;
            if (qr >= seq_q) {
                continue;
            }
            const int d = d_base + t * 32 + int(fn) + (i & 3) + (i >> 3) * 16;
            const float v = o_acc[t][i] * inv_s[r];
#ifdef ATTENTION_C4
            long o_off = (long)(h * (param.head_dim / 4) + (d / 4)) * 4 * param.batch * seq_q
                       + (long)(b * seq_q + qr) * 4
                       + (d & 3);
#else
            long o_off = ((long)(b * seq_q + qr) * param.head_num + h) * param.head_dim + d;
#endif
            O[o_off] = ftype(v);
        }
    }
}
)metal";

// --------------------------------------------------------------------------------
// prefill_flash_attn_sg -- M4 fused prefill (no tensor-coop / TC).
//
// Tiling on non-TC devices: BQ=32, BK=16 (D>=128) / 32 (D<128),
// WM=4. Q lives in smem; K/V stream through one reused smem tile in [kv, d]
// (row-major, matching the current KV cache). QK uses simdgroup_load(K,
// transpose=true); online softmax + O stay in 8x8 float fragments so S never
// touches global memory (each lane holds 2 consecutive
// cols of one row, row-reduce = local op + shuffle_xor 1 and 8).
//
// vs the previous STEEL-like attempt: K/V are now loaded
// coalesced along head_dim (consecutive threads = consecutive d), not along
// kv; V matches the row-major cache; K and V sit in separate smem tiles so
// each KV block is one barrier, not three. No KV-split.
//
// Macros: ftype, HEAD_DIM (64/128), HAS_MASK, ATTENTION_C4.
// Grid: (ceil(seq_q/32), B*H, 1), threadgroup 128 (4 simdgroups).
// --------------------------------------------------------------------------------
const char* gPrefillFlashAttnSg = R"metal(
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

static inline long fasg_mask_offset(constant FaParam& param, int b, int hn, int q, int k) {
    int mask_b = param.mask_batch <= 1 ? 0 : b;
    int mask_h = param.mask_head_num <= 1 ? 0 : hn;
    int mask_q = param.mask_q_len <= 1 ? 0 : min(q, param.mask_q_len - 1);
    int mask_k_start = max(param.key_seq_len - param.mask_k_len, 0);
    int local_k = param.mask_k_len <= 1 ? 0 : clamp(k - mask_k_start, 0, param.mask_k_len - 1);
    return (((long)(mask_b) * param.mask_head_num + mask_h) * param.mask_q_len + mask_q) * (long)param.mask_k_len + local_k;
}

static inline short2 fasg_frag_coord(ushort lane) {
    const short qid = short(lane / 4);
    const short fm = (qid & 4) + short((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
    return short2(fn, fm);
}

static inline thread float2& fasg_elems(thread simdgroup_float8x8& m) {
    return reinterpret_cast<thread float2&>(m.thread_elements());
}
static inline thread half2& fasg_elems_h(thread simdgroup_half8x8& m) {
    return reinterpret_cast<thread half2&>(m.thread_elements());
}

// BQ is FASG_NSG * 8: 8 q rows per simdgroup times the simdgroup count.
// FASG_NSG8 widens the q tile to 64 rows by adding simdgroups (256 threads),
// which halves how many times the device K/V stream is re-read without touching
// per-lane register pressure.
#ifdef FASG_NSG8
#define FASG_NSG    8
#else
#define FASG_NSG    4
#endif
#define FASG_PAD    8
#define FASG_BQ     (FASG_NSG * 8)
#if HEAD_DIM < 128
#define FASG_BK     32
#else
#define FASG_BK     16
#endif
#define FASG_LDQ    (HEAD_DIM + FASG_PAD)
#define FASG_LDV    (HEAD_DIM + FASG_PAD)
#define FASG_NK     (FASG_BK / 8)
#define FASG_ND     (HEAD_DIM / 8)

// FASG_VB: how many V fragments FASG_LOADBATCH issues before any MMA consumes
// one. The host only sets LOADBATCH when FASG_VB divides FASG_ND.
#ifndef FASG_VB
#define FASG_VB     4
#endif

// Stage one kv tile into smem, coalesced along head_dim. Rows past seq_k are
// zero-filled; the softmax masks them out regardless.
static inline void fasg_stage_kv(threadgroup half* Ks, threadgroup half* Vs,
                                 const device ftype* k_head, const device ftype* v_head,
                                 int kv_block, int seq_k, int k_row_stride, int tid) {
    for (int i = tid; i < FASG_BK * (HEAD_DIM / 4); i += FASG_NSG * 32) {
        int kv_local = i / (HEAD_DIM / 4);
        int d = (i % (HEAD_DIM / 4)) * 4;
        int kv_abs = kv_block + kv_local;
        half4 k4 = half4(0.0h);
        half4 v4 = half4(0.0h);
        if (kv_abs < seq_k) {
            const long row = (long)kv_abs * k_row_stride + d;
            k4 = half4(*(const device ftype4*)(k_head + row));
            v4 = half4(*(const device ftype4*)(v_head + row));
        }
        *((threadgroup half4*)&Ks[kv_local * FASG_LDV + d]) = k4;
        *((threadgroup half4*)&Vs[kv_local * FASG_LDV + d]) = v4;
    }
}

kernel void prefill_flash_attn_sg(
    const device ftype* Q     [[buffer(0)]],
    device ftype* O           [[buffer(1)]],
    const device ftype* K     [[buffer(2)]],
    const device ftype* V     [[buffer(3)]],
    constant FaParam& param   [[buffer(4)]],
    constant int& seq_idx     [[buffer(5)]],
    constant int& kv_start_arg [[buffer(6)]],
    constant int& kv_len_arg  [[buffer(7)]],
#ifdef HAS_MASK
    const device ftype* Mask  [[buffer(8)]],
#endif
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int block_q = int(tgpig.x);
    const int hb      = int(tgpig.y);
    const int b       = hb / param.head_num;
    const int h       = hb % param.head_num;
    const int kh      = h / param.group;
    const int seq_q   = param.query_seq_len;
    const int seq_k   = param.key_seq_len;
    const int kv_heads = param.head_num / param.group;
    const int kv_valid_offset = seq_k - seq_q;
    const int q_row_base = block_q * FASG_BQ + seq_idx * param.q_seq_piece_len;
    const int tid = int(sgitg) * 32 + int(tiisg);
    const int q_row_sg = int(sgitg) * 8;
    const short2 fcoord = fasg_frag_coord(tiisg);
    const int fn = int(fcoord.x);
    const int fm = int(fcoord.y);

    threadgroup half Qs[FASG_BQ * FASG_LDQ];
    threadgroup half Ks[FASG_BK * FASG_LDV];
    threadgroup half Vs[FASG_BK * FASG_LDV];

    const float qscale = param.scale * 1.4426950408889634f;
    const int k_row_stride = param.batch * kv_heads * HEAD_DIM;

    // Cooperative Q load, coalesced along head_dim. Scale folded in (log2e).
    for (int i = tid; i < FASG_BQ * (HEAD_DIM / 4); i += FASG_NSG * 32) {
        int row = i / (HEAD_DIM / 4);
        int col = (i % (HEAD_DIM / 4)) * 4;
        int q_abs = q_row_base + row;
        half4 v = half4(0.0h);
        if (q_abs < seq_q) {
            long q_off = ((long)(b * seq_q + q_abs) * param.head_num + h) * param.head_dim + col;
            float4 qv = float4(*(const device ftype4*)(Q + q_off));
            v = half4(qv * qscale);
        }
        *((threadgroup half4*)&Qs[row * FASG_LDQ + col]) = v;
    }

    simdgroup_float8x8 mO[FASG_ND];
    for (int i = 0; i < FASG_ND; i++) {
        mO[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    float M = -INFINITY;
    float S = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef FASG_Q_REG
    // Q is a kv-loop invariant; keep its ND fragments in registers so the QK
    // loop issues one smem load per MMA instead of two.
    simdgroup_half8x8 mQreg[FASG_ND];
    for (int dd = 0; dd < FASG_ND; dd++) {
        simdgroup_load(mQreg[dd], Qs + q_row_sg * FASG_LDQ + dd * 8, FASG_LDQ);
    }
#endif

    const int kv_lo = kv_start_arg;
    const int kv_hi = kv_start_arg + kv_len_arg;
    const device ftype* k_head = K + ((long)b * kv_heads + kh) * HEAD_DIM;
    const device ftype* v_head = V + ((long)b * kv_heads + kh) * HEAD_DIM;
    // Last kv column any q row in this tile can attend to.
    const int max_attend_k = q_row_base + FASG_BQ - 1 + kv_valid_offset;

    for (int kv_block = kv_lo; kv_block < kv_hi; kv_block += FASG_BK) {
        if (kv_block > max_attend_k) {
            break;
        }

        fasg_stage_kv(Ks, Vs, k_head, v_head, kv_block, seq_k, k_row_stride, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mS[FASG_NK];
        for (int kf = 0; kf < FASG_NK; kf++) {
            mS[kf] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
#ifdef FASG_LOADBATCH
        // Issue every K fragment for this head_dim slab before any MMA consumes
        // one, so each load's latency is covered by the preceding MMAs instead of
        // stalling the MMA that needs it. Every index must fold to a compile-time
        // constant: a dynamically indexed simdgroup-matrix array spills out of
        // registers and costs ~2.8x, so the batch loops stay fully unrolled.
#pragma clang loop unroll(full)
        for (int dd = 0; dd < FASG_ND; dd++) {
            simdgroup_half8x8 mKb[FASG_NK];
#pragma clang loop unroll(full)
            for (int kf = 0; kf < FASG_NK; kf++) {
                simdgroup_load(mKb[kf], Ks + kf * 8 * FASG_LDV + dd * 8, FASG_LDV, ulong2(0, 0), true);
            }
#pragma clang loop unroll(full)
            for (int kf = 0; kf < FASG_NK; kf++) {
#ifdef FASG_Q_REG
                simdgroup_multiply_accumulate(mS[kf], mQreg[dd], mKb[kf], mS[kf]);
#else
                simdgroup_half8x8 mQl;
                simdgroup_load(mQl, Qs + q_row_sg * FASG_LDQ + dd * 8, FASG_LDQ);
                simdgroup_multiply_accumulate(mS[kf], mQl, mKb[kf], mS[kf]);
#endif
            }
        }
#else
        for (int dd = 0; dd < FASG_ND; dd++) {
#ifdef FASG_Q_REG
            simdgroup_half8x8 mQ = mQreg[dd];
#else
            simdgroup_half8x8 mQ;
            simdgroup_load(mQ, Qs + q_row_sg * FASG_LDQ + dd * 8, FASG_LDQ);
#endif
            for (int kf = 0; kf < FASG_NK; kf++) {
                simdgroup_half8x8 mK;
                simdgroup_load(mK, Ks + kf * 8 * FASG_LDV + dd * 8, FASG_LDV, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mS[kf], mQ, mK, mS[kf]);
            }
        }
#endif

        // P stays half so PV runs on the same half MMA path as QK; only the
        // O accumulator is float.
        simdgroup_half8x8 mP[FASG_NK];
        {
            const int q_abs = q_row_base + q_row_sg + fm;
            float ev[FASG_NK][2];
            float rmax = M;
            for (int kf = 0; kf < FASG_NK; kf++) {
                float2 sp = fasg_elems(mS[kf]);
                ev[kf][0] = sp.x;
                ev[kf][1] = sp.y;
                for (int j = 0; j < 2; j++) {
                    int kv_col = kv_block + kf * 8 + fn + j;
                    bool in_bounds = (q_abs < seq_q) && (kv_col < seq_k) &&
                                     (kv_col <= q_abs + kv_valid_offset);
                    if (!in_bounds) {
                        ev[kf][j] = -INFINITY;
                    }
#ifdef HAS_MASK
                    else {
                        ev[kf][j] += float(Mask[fasg_mask_offset(param, b, h, q_abs, kv_col)]) *
                                     1.4426950408889634f;
                    }
#endif
                }
                rmax = fmax(rmax, fmax(ev[kf][0], ev[kf][1]));
            }
            rmax = fmax(rmax, simd_shuffle_xor(rmax, 1));
            rmax = fmax(rmax, simd_shuffle_xor(rmax, 8));

            const float factor = (M == -INFINITY) ? 0.0f : fast::exp2(M - rmax);
            M = rmax;
            for (int dd = 0; dd < FASG_ND; dd++) {
                float2 ov = fasg_elems(mO[dd]);
                ov *= factor;
                fasg_elems(mO[dd]) = ov;
            }

            float rsum = 0.0f;
            for (int kf = 0; kf < FASG_NK; kf++) {
                float vx = (ev[kf][0] == -INFINITY) ? 0.0f : fast::exp2(ev[kf][0] - rmax);
                float vy = (ev[kf][1] == -INFINITY) ? 0.0f : fast::exp2(ev[kf][1] - rmax);
                rsum += vx + vy;
                fasg_elems_h(mP[kf]) = half2(half(vx), half(vy));
            }
            rsum += simd_shuffle_xor(rsum, 1);
            rsum += simd_shuffle_xor(rsum, 8);
            S = S * factor + rsum;
        }

#ifdef FASG_LOADBATCH
#pragma clang loop unroll(full)
        for (int kf = 0; kf < FASG_NK; kf++) {
#pragma clang loop unroll(full)
            for (int d0 = 0; d0 < FASG_ND; d0 += FASG_VB) {
                simdgroup_half8x8 mVb[FASG_VB];
#pragma clang loop unroll(full)
                for (int j = 0; j < FASG_VB; j++) {
                    simdgroup_load(mVb[j], Vs + kf * 8 * FASG_LDV + (d0 + j) * 8, FASG_LDV);
                }
#pragma clang loop unroll(full)
                for (int j = 0; j < FASG_VB; j++) {
                    simdgroup_multiply_accumulate(mO[d0 + j], mP[kf], mVb[j], mO[d0 + j]);
                }
            }
        }
#else
        for (int kf = 0; kf < FASG_NK; kf++) {
            for (int dd = 0; dd < FASG_ND; dd++) {
                simdgroup_half8x8 mVh;
                simdgroup_load(mVh, Vs + kf * 8 * FASG_LDV + dd * 8, FASG_LDV);
                simdgroup_multiply_accumulate(mO[dd], mP[kf], mVh, mO[dd]);
            }
        }
#endif
        // Keeps the next iteration's staging from clobbering the tile the MMAs
        // above just read.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int q_abs = q_row_base + q_row_sg + fm;
    if (q_abs >= seq_q) {
        return;
    }
    const float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    for (int dd = 0; dd < FASG_ND; dd++) {
        float2 op = fasg_elems(mO[dd]);
        int d0 = dd * 8 + fn;
        int d1 = d0 + 1;
#ifdef ATTENTION_C4
        long o0 = (long)(h * (param.head_dim / 4) + (d0 / 4)) * 4 * param.batch * seq_q
                + (long)(b * seq_q + q_abs) * 4 + (d0 & 3);
        long o1 = (long)(h * (param.head_dim / 4) + (d1 / 4)) * 4 * param.batch * seq_q
                + (long)(b * seq_q + q_abs) * 4 + (d1 & 3);
#else
        long o0 = ((long)(b * seq_q + q_abs) * param.head_num + h) * param.head_dim + d0;
        long o1 = ((long)(b * seq_q + q_abs) * param.head_num + h) * param.head_dim + d1;
#endif
        O[o0] = ftype(op.x * inv_S);
        O[o1] = ftype(op.y * inv_S);
    }
}
)metal";

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
