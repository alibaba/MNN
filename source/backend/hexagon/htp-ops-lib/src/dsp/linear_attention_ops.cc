// LinearAttention (OpType_LinearAttention) for the packed LLM layout, batch 1.
//
//   gated_delta_rule (Qwen3.5): depthwise causal conv1d + SiLU over qkv, optional q/k L2 norm, then per
//                               head  S = decay * S + k (beta (v - decay S^T k))^T,  o = S^T q.
//   short_conv (LFM2):          y = C * conv1d(B * x), no activation.
//
// Layouts (all fp16 tensors are NC4HW4 with pack 64, so one (token, 64-channel block) row is one vector):
//   qkv  [L, D]            row (t, cb) at ((cb * L) + t) * 64
//   a/b  [L, Hv]           element (t, h) at (((h / 64) * L) + t) * 64 + h % 64
//   out  gdr: [L * Hv, dv] row (t * Hv + h, cb) at ((cb * L * Hv) + t * Hv + h) * 64;  short: [L, H] like qkv
//   convW [Dc, 1, K]       plain fp16, channel-major
// Internal fp32 data (conv history, conv output, recurrent state) is kept in the lane order produced by
// la_widen(); la_narrow() is its exact inverse, and all reductions / broadcasts are order independent,
// so that permutation never reaches an output.

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_math.h"
#include "dsp/hvx_utils.h"
#include "dsp/pwl.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "htp_command.h"
#include "htp_gated_rms_norm_params.h"

extern "C" {

#define LA_MAX_K              8
#define LA_MAX_VEC            8  // head dims up to 256 (8 fp32 vectors)
// Per-worker phase-B scratch, in vectors, carved from the host workspace after the conv output (worker
// stacks are small and keeping ~90 live vectors on them corrupted the result).
#define LA_SCRATCH_VECS       (32 + 7 * LA_MAX_VEC)
// Chunked prefill (gated_delta_rule): chunk length, and the per-worker scalar matrices that follow the
// vector scratch in DDR (scalar loads from VTCM are slow).
#define LA_CHUNK              32
#define LA_CHUNK_SCALAR_BYTES (80 * 1024)
// T, QK, DEC [C][C], eight [C] vectors (the last one the conv tap staging), then the fused-conv Q / K rows
// [2][C][dk] for the Gram's scalar reads.
static_assert((3 * LA_CHUNK * LA_CHUNK + 8 * LA_CHUNK + 2 * LA_CHUNK * 32 * LA_MAX_VEC) * 4 <= LA_CHUNK_SCALAR_BYTES,
              "chunk scalar scratch");
static_assert(LA_SCRATCH_VECS * 128 + LA_CHUNK_SCALAR_BYTES == DSP_LINEAR_ATTN_SCRATCH_BYTES,
              "host and DSP scratch sizes differ");

typedef struct {
  const DSPLinearAttnParam *p;
  const __fp16             *qkv;
  const __fp16             *in1;          // gate (or raw a when folded)
  const __fp16             *in2;          // beta (or raw b when folded)
  const __fp16             *conv_w;
  const float              *gate_const;   // [coef[Hv], bias[Hv]] when folded
  __fp16                   *out;
  float                    *conv_state;   // [blocks][K-1][64]
  float                    *rnn_state;    // [Hv][dk][dv]
  float                    *conv_out;     // [L][D] (gated_delta_rule), then per-worker scratch
  HVX_Vector               *scratch;      // LA_SCRATCH_VECS per worker
  int                       fuse_conv;    // chunked prefill computes each head's conv rows itself (Hk == Hv)
  int                       fuse_decode;  // decode (L == 1, Hk == Hv): each head's worker runs its own conv
} la_args_t;

typedef struct {
  const la_args_t    *a;
  int                 first, step, count;  // first is also the worker slot
  worker_synctoken_t *sync;
} la_task_t;

static inline HVX_Vector la_zero() {
  return Q6_V_vzero();
}

static inline HVX_Vector la_splat(float f) {
  union {
    float    f;
    uint32_t u;
  } c;

  c.f = f;
  return Q6_V_vsplat_R(c.u);
}

// 64 fp16 -> two fp32 vectors. Lane order is whatever the widening multiply produces; la_narrow undoes it.
static inline void la_widen(HVX_Vector hf, HVX_Vector *lo, HVX_Vector *hi) {
  HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(hf, Q6_Vh_vsplat_R(0x3C00));
  *lo              = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
  *hi              = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
}

static inline HVX_Vector la_narrow(HVX_Vector lo, HVX_Vector hi) {
  return Q6_Vhf_equals_Wqf32(
    Q6_W_vcombine_VV(Q6_Vqf32_vadd_VsfVsf(hi, la_zero()), Q6_Vqf32_vadd_VsfVsf(lo, la_zero())));
}

static inline HVX_Vector la_mul(HVX_Vector x, HVX_Vector y) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, y));
}

static inline HVX_Vector la_add(HVX_Vector x, HVX_Vector y) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, y));
}

static inline HVX_Vector la_sub(HVX_Vector x, HVX_Vector y) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x, y));
}

// 1 / d for d in [1, 2]: minimax linear seed (24 - 8 d) / 17 (relative error <= 1/17), then Newton
// r = r (2 - d r) twice (error 0.059 -> 3.5e-3 -> 1.2e-5). hvx_my_inv_vqf32_vsf has no other user and produced NaN.
static inline HVX_Vector la_recip_1_2(HVX_Vector d) {
  const HVX_Vector two = la_splat(2.0f);
  HVX_Vector       r   = la_sub(la_splat(24.0f / 17.0f), la_mul(d, la_splat(8.0f / 17.0f)));
  for (int i = 0; i < 2; ++i) {
    r = la_mul(r, la_sub(two, la_mul(d, r)));
  }
  return r;
}

// x * sigmoid(x). hvx_my_exp2_vsf is only used on non-positive arguments (as in the softmax kernels):
// e = 2^(-|x| log2 e) in (0, 1], sigmoid = 1 / (1 + e) for x >= 0 and e / (1 + e) otherwise.
static inline HVX_Vector la_silu(HVX_Vector x) {
  const HVX_Vector ax  = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fffffff));
  HVX_Vector       t   = la_mul(ax, la_splat(-1.44269504f));
  t                    = Q6_Vsf_vmax_VsfVsf(t, la_splat(-64.0f));
  const HVX_Vector e   = hvx_my_exp2_vsf(t);
  const HVX_Vector r   = la_recip_1_2(la_add(e, la_splat(1.0f)));
  const HVX_Vector sig = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(Q6_V_vzero(), x), la_mul(e, r), r);
  return la_mul(x, sig);
}

// Sum of the 32 fp32 lanes, left in every lane.
static inline HVX_Vector la_hsum(HVX_Vector v) {
  for (int r = 64; r >= 4; r >>= 1) {
    v = la_add(v, Q6_V_vror_VR(v, r));
  }
  return v;
}

// vdelta control replicating 32-bit lane j: ctrl[k] = (k & 0x7c) ^ 4j (see matmul_q4block_gemv_i8.c).
// The base table is file-scope: 128-byte aligned locals on worker stacks are not reliable.
_Alignas(128) static const uint8_t kLaLaneBase[128] = {
#define LA_G4(x) (x), (x), (x), (x)
#define LA_G32(x)                                                                                                 \
  LA_G4((x)), LA_G4((x) + 4), LA_G4((x) + 8), LA_G4((x) + 12), LA_G4((x) + 16), LA_G4((x) + 20), LA_G4((x) + 24), \
    LA_G4((x) + 28)
  LA_G32(0), LA_G32(32), LA_G32(64), LA_G32(96)
#undef LA_G32
#undef LA_G4
};

static void la_build_bcast(HVX_Vector *ctrl) {
  const HVX_Vector v_base = vmem(kLaLaneBase);
  for (int j = 0; j < 32; ++j) {
    ctrl[j] = Q6_V_vxor_VV(v_base, Q6_Vb_vsplat_R(4 * j));
  }
}

// Compiler barrier between scalar float and vmem() (HVX_Vector *) accesses to the same scratch: they are
// different types, so without it the compiler may reorder them (strict aliasing).
#define LA_FENCE() __asm__ __volatile__("" ::: "memory")

// Per-stage timing of the chunked prefill (diagnostic, off): profile[229..243]. Each worker owns one head at a
// time and runs its stages strictly in sequence, so per-worker stage times add up to that worker's busy time with
// no cross-stage overlap to misattribute -- unlike the GEMM, where a phase's measured time moves when another
// phase does. Summed over workers and divided by the worker count, a slot is that stage's share of the op.
// Keep in lockstep with execute_command.cc and dsp_op_name.h.
#ifndef HTP_LA_PHASE_PROFILE
#  define HTP_LA_PHASE_PROFILE 0
#endif
enum {
  LA_PH_STATE_IO = 0,  // recurrent state DDR <-> VTCM, once per head
  LA_PH_CONV,          // fused short conv (or the copy from the unfused conv output) into Q/K/V rows
  LA_PH_TRANSPOSE,     // K -> K^T for the Gram rows
  LA_PH_GATES,         // read g/beta, softplus/sigmoid, prefix sum, exp
  LA_PH_GRAM,          // K K^T and Q K^T rows
  LA_PH_NORMS,         // |q|^2 reductions and the 1/sqrt scales
  LA_PH_DECAY,         // decay-weighted, masked attn rows
  LA_PH_FWDSUB,        // forward substitution (I + A)^-1
  LA_PH_NORMALIZE,     // scale K, Q rows
  LA_PH_TRI_IN,        // VN = T (beta V), KCD = T (beta e^g K)
  LA_PH_ROWS_STATE,    // v_new = VN - KCD S, out = Q S
  LA_PH_TRI_OUT,       // out += QK v_new
  LA_PH_STATE_UPD,     // S = e^gl S + (K w)^T v_new
  LA_PH_OUT_STORE,     // narrow to fp16, scatter to the packed output
  LA_PH_HEAD_TOTAL,    // whole la_delta_head_chunked call, to check the stages add up
  LA_PH_COUNT
};
#if HTP_LA_PHASE_PROFILE
#  include "HAP_perf.h"
unsigned long long g_la_phase_ticks[32][LA_PH_COUNT];
#  define LA_T0()            HAP_perf_get_qtimer_count()
#  define LA_ADD(wid, ph, t) (g_la_phase_ticks[(wid) & 31][(ph)] += HAP_perf_get_qtimer_count() - (t))
#else
#  define LA_T0()            0ULL
#  define LA_ADD(wid, ph, t) ((void) (wid), (void) (t))
#endif

// Lane j holds j (memory order), for the triangular masks of the chunked path.
_Alignas(128) static const int32_t kLaIota[32] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };

// e^x for x <= 0 (clamped at 2^-126) with the vector exp2.
static inline HVX_Vector la_exp_nonpos(HVX_Vector x) {
  HVX_Vector t = la_mul(x, la_splat(1.44269504f));
  t            = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(t, la_splat(-126.0f)), Q6_V_vzero());
  return hvx_my_exp2_vsf(t);
}

// Lane 0 of an fp32 vector as a scalar, without a store / reload.
static inline float la_lane0(HVX_Vector v) {
  union {
    int32_t i;
    float   f;
  } c;

  c.i = Q6_R_vextract_VR(v, 0);
  return c.f;
}

// The four taps of 64 channels from channel-major [64][4] fp16 weights (four vectors): two rounds of halfword
// deals split (k0, k2) from (k1, k3) and then the pairs, leaving tap k for channels 0..63 in order.
static inline void la_taps_k4(const __fp16 *w, HVX_Vector *lo, HVX_Vector *hi) {
  const HVX_UVector   *src = (const HVX_UVector *) w;
  const HVX_VectorPair w01 = Q6_W_vdeal_VVR(src[1], src[0], -2);
  const HVX_VectorPair w23 = Q6_W_vdeal_VVR(src[3], src[2], -2);
  const HVX_VectorPair e   = Q6_W_vdeal_VVR(Q6_V_lo_W(w23), Q6_V_lo_W(w01), -2);  // k0, k2
  const HVX_VectorPair o   = Q6_W_vdeal_VVR(Q6_V_hi_W(w23), Q6_V_hi_W(w01), -2);  // k1, k3
  la_widen(Q6_V_lo_W(e), &lo[0], &hi[0]);
  la_widen(Q6_V_lo_W(o), &lo[1], &hi[1]);
  la_widen(Q6_V_hi_W(e), &lo[2], &hi[2]);
  la_widen(Q6_V_hi_W(o), &lo[3], &hi[3]);
}

// ---- Phase A: causal depthwise conv1d over 64-channel blocks -------------------------------------------
// Tokens [t0, t1) of channel block cb, continuing from (and updating) its conv history. gated_delta_rule
// writes SiLU(conv) for token t to dst + (t - t0) * dst_stride; short_conv writes the output tensor.
// wpre: the block's taps already widened by la_conv_taps ([K][lo, hi]), or NULL to gather them here
// (phase A, whose shallow frame keeps the aligned stack staging reliable).
// KC > 0 fixes the kernel size at compile time, so the tap / history arrays are fully unrolled into
// registers (a runtime K keeps them in memory and reloads the whole history every token).
extern "C++" {
template <int KC>
static __attribute__((noinline)) void la_conv_rows_t(const la_args_t *a, int cb, int t0, int t1, float *dst_base,
                                                     size_t dst_stride, const HVX_Vector *wpre) {
  const DSPLinearAttnParam *p   = a->p;
  const int                 L   = p->seqLen;
  const int                 K   = KC > 0 ? KC : p->kernelSize;
  const int                 S   = K - 1;
  const bool                gdr = p->type == DSP_LINEAR_ATTN_GATED_DELTA;
  const int                 D   = p->convDim;
  const int                 nb  = gdr ? D / 64 : D / 192;  // short_conv: channel blocks of B (= of C, x)
  const int                 c0  = cb * 64;
  // The block's rows are contiguous: one background fetch for this range and the following one, so the
  // loop below streams instead of taking a DDR miss per token.
  // A single token skips it: a new l2fetch replaces the thread's outstanding one (the decode state prefetch).
  if (t1 - t0 > 1) {
    const int ahead = t1 + (t1 - t0) < L ? t1 + (t1 - t0) : L;
    l2fetch(a->qkv + ((size_t) cb * L + t0) * 64, 128, 128, ahead - t0, 0);
  }
  // The per-channel taps as vectors (weights are channel-major [C][K]).
  HVX_Vector w_lo[LA_MAX_K], w_hi[LA_MAX_K];
  if (wpre) {
    for (int k = 0; k < K; ++k) {
      w_lo[k] = wpre[2 * k];
      w_hi[k] = wpre[2 * k + 1];
    }
  } else if (K == 4) {
    la_taps_k4(a->conv_w + (size_t) c0 * 4, w_lo, w_hi);
  } else {
    _Alignas(128) __fp16 tmp[64];
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < 64; ++i) {
        tmp[i] = a->conv_w[(size_t) (c0 + i) * K + k];
      }
      LA_FENCE();
      la_widen(vmem(tmp), &w_lo[k], &w_hi[k]);
    }
  }
  HVX_Vector h_lo[LA_MAX_K], h_hi[LA_MAX_K];
  float     *state = a->conv_state + (size_t) cb * S * 64;
  for (int r = 0; r < S; ++r) {
    h_lo[r] = vmem(state + r * 64);
    h_hi[r] = vmem(state + r * 64 + 32);
  }
  for (int t = t0; t < t1; ++t) {
    HVX_Vector x_lo, x_hi;
    la_widen(vmem(a->qkv + ((size_t) cb * L + t) * 64), &x_lo, &x_hi);
    if (!gdr) {
      HVX_Vector y_lo, y_hi;  // B * x
      la_widen(vmem(a->qkv + ((size_t) (2 * nb + cb) * L + t) * 64), &y_lo, &y_hi);
      x_lo = la_mul(x_lo, y_lo);
      x_hi = la_mul(x_hi, y_hi);
    }
    HVX_Vector acc_lo = Q6_Vqf32_vmpy_VsfVsf(x_lo, w_lo[S]);
    HVX_Vector acc_hi = Q6_Vqf32_vmpy_VsfVsf(x_hi, w_hi[S]);
    for (int r = 0; r < S; ++r) {
      acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_Vqf32_vmpy_VsfVsf(h_lo[r], w_lo[r]));
      acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_Vqf32_vmpy_VsfVsf(h_hi[r], w_hi[r]));
    }
    for (int r = 0; r + 1 < S; ++r) {
      h_lo[r] = h_lo[r + 1];
      h_hi[r] = h_hi[r + 1];
    }
    if (S > 0) {
      h_lo[S - 1] = x_lo;
      h_hi[S - 1] = x_hi;
    }
    HVX_Vector y_lo = Q6_Vsf_equals_Vqf32(acc_lo);
    HVX_Vector y_hi = Q6_Vsf_equals_Vqf32(acc_hi);
    if (gdr) {
      float *dst     = dst_base + (size_t) (t - t0) * dst_stride;
      vmem(dst)      = la_silu(y_lo);
      vmem(dst + 32) = la_silu(y_hi);
    } else {
      HVX_Vector c_lo, c_hi;
      la_widen(vmem(a->qkv + ((size_t) (nb + cb) * L + t) * 64), &c_lo, &c_hi);
      vmem(a->out + ((size_t) cb * L + t) * 64) = la_narrow(la_mul(c_lo, y_lo), la_mul(c_hi, y_hi));
    }
  }
  for (int r = 0; r < S; ++r) {
    vmem(state + r * 64)      = h_lo[r];
    vmem(state + r * 64 + 32) = h_hi[r];
  }
}
}  // extern "C++"

static void la_conv_rows(const la_args_t *a, int cb, int t0, int t1, float *dst_base, size_t dst_stride,
                         const HVX_Vector *wpre) {
  switch (a->p->kernelSize) {
    case 2:
      la_conv_rows_t<2>(a, cb, t0, t1, dst_base, dst_stride, wpre);
      break;
    case 3:
      la_conv_rows_t<3>(a, cb, t0, t1, dst_base, dst_stride, wpre);
      break;
    case 4:
      la_conv_rows_t<4>(a, cb, t0, t1, dst_base, dst_stride, wpre);
      break;
    default:
      la_conv_rows_t<0>(a, cb, t0, t1, dst_base, dst_stride, wpre);
      break;
  }
}

// Widened taps of channel block cb into out[K][lo, hi]; wtmp is 128-byte aligned staging for 64 fp16.
static void la_conv_taps(const la_args_t *a, int cb, __fp16 *wtmp, HVX_Vector *out) {
  const int K = a->p->kernelSize;
  if (K == 4) {
    HVX_Vector lo[4], hi[4];
    la_taps_k4(a->conv_w + (size_t) cb * 64 * 4, lo, hi);
    for (int k = 0; k < 4; ++k) {
      out[2 * k]     = lo[k];
      out[2 * k + 1] = hi[k];
    }
    return;
  }
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < 64; ++i) {
      wtmp[i] = a->conv_w[(size_t) (cb * 64 + i) * K + k];
    }
    LA_FENCE();
    HVX_Vector lo, hi;
    la_widen(vmem(wtmp), &lo, &hi);
    out[2 * k]     = lo;
    out[2 * k + 1] = hi;
  }
}

static void la_conv_worker(void *data, int worker_index) {
  (void) worker_index;
  const la_task_t          *t = (const la_task_t *) data;
  const DSPLinearAttnParam *p = t->a->p;
  for (int cb = t->first; cb < t->count; cb += t->step) {
    float *dst = t->a->conv_out ? t->a->conv_out + (size_t) cb * 64 : NULL;  // NULL for short_conv
    la_conv_rows(t->a, cb, 0, p->seqLen, dst, (size_t) p->convDim, NULL);
  }
  worker_pool_synctoken_jobdone(t->sync);
}

// logf is not exported to the skel. Natural log for y > 0: y = m 2^e with m in [sqrt(1/2), sqrt(2)), then
// log m = 2 atanh((m - 1) / (m + 1)) by its odd series (|s| <= 0.172, truncation < 1e-8).
static inline float la_logf(float y) {
  union {
    float    f;
    uint32_t u;
  } c;

  c.f     = y;
  int e   = (int) ((c.u >> 23) & 0xff) - 127;
  c.u     = (c.u & 0x007fffffu) | 0x3f800000u;  // m in [1, 2)
  float m = c.f;
  if (m > 1.41421356f) {
    m *= 0.5f;
    e += 1;
  }
  const float s  = (m - 1.0f) / (m + 1.0f);
  const float s2 = s * s;
  const float r  = s * (2.0f + s2 * (0.6666667f + s2 * (0.4f + s2 * (0.2857143f + s2 * 0.2222222f))));
  return r + (float) e * 0.69314718f;
}

// ---- Phase B: gated delta rule, one head per job --------------------------------------------------------
// One token of one head, S [nvk * 32][NV * 32]. Row by row (each k / q lane broadcast once per row, S read
// contiguously): S^T k and S^T q from the previous state, then
//   delta = beta (v - decay S^T k),  o = decay S^T q + (k . q) delta,  S = decay S + k delta^T.
extern "C++" {
template <int NV>
static void la_delta_step(float *S, int nvk, const HVX_Vector *k, const HVX_Vector *q, const HVX_Vector *v,
                          const HVX_Vector *bcast, HVX_Vector vdecay, HVX_Vector vbeta, HVX_Vector vkq,
                          HVX_Vector *delta, HVX_Vector *o) {
  HVX_Vector aK[NV], aQ[NV];
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    aK[j] = aQ[j] = Q6_V_vzero();
  }
  for (int i = 0; i < nvk; ++i) {
    const HVX_Vector ki = k[i], qi = q[i];
    for (int ln = 0; ln < 32; ++ln) {
      const HVX_Vector kb = Q6_V_vdelta_VV(ki, bcast[ln]);
      const HVX_Vector qb = Q6_V_vdelta_VV(qi, bcast[ln]);
      const float     *sr = S + (size_t) (i * 32 + ln) * NV * 32;
#pragma unroll
      for (int j = 0; j < NV; ++j) {
        const HVX_Vector sv = vmem(sr + j * 32);
        aK[j]               = Q6_Vqf32_vadd_Vqf32Vqf32(aK[j], Q6_Vqf32_vmpy_VsfVsf(sv, kb));
        aQ[j]               = Q6_Vqf32_vadd_Vqf32Vqf32(aQ[j], Q6_Vqf32_vmpy_VsfVsf(sv, qb));
      }
    }
  }
  HVX_Vector dl[NV];
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    dl[j]    = la_mul(vbeta, la_sub(v[j], la_mul(vdecay, Q6_Vsf_equals_Vqf32(aK[j]))));
    delta[j] = dl[j];
    o[j]     = la_add(la_mul(vdecay, Q6_Vsf_equals_Vqf32(aQ[j])), la_mul(vkq, dl[j]));
  }
  for (int i = 0; i < nvk; ++i) {
    const HVX_Vector ki = k[i];
    for (int ln = 0; ln < 32; ++ln) {
      const HVX_Vector kb = Q6_V_vdelta_VV(ki, bcast[ln]);
      float           *sr = S + (size_t) (i * 32 + ln) * NV * 32;
#pragma unroll
      for (int j = 0; j < NV; ++j) {
        vmem(sr + j * 32) = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(vmem(sr + j * 32), vdecay), Q6_Vqf32_vmpy_VsfVsf(kb, dl[j])));
      }
    }
  }
}
}  // extern "C++"

static inline float la_read_gate(const __fp16 *src, int L, int t, int h) {
  return (float) src[((size_t) (h / 64) * L + t) * 64 + h % 64];
}

// frow: fused decode, this head's conv output [q dk][k dk][v dv] (L == 1). resident: vtcm_state already holds
// the head's state and the caller writes it back.
static void la_delta_head(const la_args_t *a, int h, HVX_Vector *scr, float *vtcm_state, const float *frow,
                          int resident) {
  const DSPLinearAttnParam *p      = a->p;
  const int                 L      = p->seqLen;
  const int                 D      = p->convDim;
  const int                 Hk     = p->numKHeads;
  const int                 Hv     = p->numVHeads;
  const int                 dk     = p->headKDim;
  const int                 dv     = p->headVDim;
  const int                 nvk    = dk / 32;
  const int                 nvv    = dv / 32;
  const int                 kh     = h / (Hv / Hk);
  const float               qScale = 1.0f / sqrtf((float) dk);
  float                    *S_ddr  = a->rnn_state + (size_t) h * dk * dv;
  // The state is re-read twice and rewritten once per token; from DDR/L2 that made the recurrence
  // memory bound (~10 us per token-head), so the head's state lives in this worker's VTCM slot
  // for the whole sequence.
  float                    *S      = vtcm_state ? vtcm_state : S_ddr;
  if (vtcm_state && !resident) {
    for (int i = 0; i < dk * dv; i += 32) {
      vmem(S + i) = vmem(S_ddr + i);
    }
  }
  const HVX_Vector *bcast = scr;
  HVX_Vector       *q     = scr + 32;
  HVX_Vector       *k     = q + LA_MAX_VEC;
  HVX_Vector       *v     = k + LA_MAX_VEC;
  HVX_Vector       *delta = v + LA_MAX_VEC;
  HVX_Vector       *o     = delta + LA_MAX_VEC;

  const int qo = frow ? 0 : kh * dk, ko = frow ? dk : Hk * dk + kh * dk, vo = frow ? 2 * dk : 2 * Hk * dk + h * dv;
  for (int t = 0; t < L; ++t) {
    const float *row = frow ? frow : a->conv_out + (size_t) t * D;
    HVX_Vector   sq = Q6_V_vzero(), sk = Q6_V_vzero(), sqk = Q6_V_vzero();
    for (int i = 0; i < nvk; ++i) {
      const HVX_Vector qi = vmem(row + qo + i * 32);
      const HVX_Vector ki = vmem(row + ko + i * 32);
      q[i]                = qi;
      k[i]                = ki;
      sq                  = Q6_Vqf32_vadd_Vqf32Vqf32(sq, Q6_Vqf32_vmpy_VsfVsf(qi, qi));
      sk                  = Q6_Vqf32_vadd_Vqf32Vqf32(sk, Q6_Vqf32_vmpy_VsfVsf(ki, ki));
      sqk                 = Q6_Vqf32_vadd_Vqf32Vqf32(sqk, Q6_Vqf32_vmpy_VsfVsf(qi, ki));
    }
    for (int i = 0; i < nvv; ++i) {
      v[i] = vmem(row + vo + i * 32);
    }
    const float ssq  = la_lane0(la_hsum(Q6_Vsf_equals_Vqf32(sq)));
    const float ssk  = la_lane0(la_hsum(Q6_Vsf_equals_Vqf32(sk)));
    const float sdot = la_lane0(la_hsum(Q6_Vsf_equals_Vqf32(sqk)));
    float       qs = qScale, ks = 1.0f;
    if (p->useL2Norm) {
      qs = qScale / sqrtf(ssq + 1e-6f);
      ks = 1.0f / sqrtf(ssk + 1e-6f);
    }
    const float      kq  = sdot * qs * ks;
    const HVX_Vector vqs = la_splat(qs), vks = la_splat(ks);
    for (int i = 0; i < nvk; ++i) {
      q[i] = la_mul(q[i], vqs);
      k[i] = la_mul(k[i], vks);
    }

    float g = la_read_gate(a->in1, L, t, h);
    float b = la_read_gate(a->in2, L, t, h);
    if (p->gateFold) {
      const float x  = g + a->gate_const[Hv + h];
      const float sp = x > 20.0f ? x : la_logf(1.0f + expf(x));
      g              = a->gate_const[h] * sp;
      b              = 1.0f / (1.0f + expf(-b));
    }
    // Same clamp as the chunked path applies via vmin/vmax: the gate coefficient is -exp(A_log) and so is
    // negative by construction, but nothing validates that, and an unclamped decay > 1 would make the
    // recurrent state diverge here while the chunked path stayed bounded.
    if (g > 0.0f) {
      g = 0.0f;
    } else if (g < -88.0f) {
      g = -88.0f;
    }
    const HVX_Vector vdecay = la_splat(expf(g));

    const HVX_Vector vbeta = la_splat(b), vkq = la_splat(kq);
    switch (nvv) {
      case 2:
        la_delta_step<2>(S, nvk, k, q, v, bcast, vdecay, vbeta, vkq, delta, o);
        break;
      case 4:
        la_delta_step<4>(S, nvk, k, q, v, bcast, vdecay, vbeta, vkq, delta, o);
        break;
      case 6:
        la_delta_step<6>(S, nvk, k, q, v, bcast, vdecay, vbeta, vkq, delta, o);
        break;
      default:
        la_delta_step<8>(S, nvk, k, q, v, bcast, vdecay, vbeta, vkq, delta, o);
        break;
    }
    const size_t plane = (size_t) L * Hv;
    for (int jb = 0; jb < dv / 64; ++jb) {
      vmem(a->out + ((size_t) jb * plane + (size_t) t * Hv + h) * 64) = la_narrow(o[2 * jb], o[2 * jb + 1]);
    }
  }
  if (vtcm_state && !resident) {
    for (int i = 0; i < dk * dv; i += 32) {
      vmem(S_ddr + i) = vmem(S + i);
    }
  }
}

// ---- Phase B, chunked prefill (torch_chunk_gated_delta_rule, as the OpenCL / Metal backends) -------------
// Per chunk of LA_CHUNK tokens: in-chunk Gram matrices, a 32x32 triangular solve, and three products against
// the state S [dk][dv] (kept in VTCM). S is touched once per chunk instead of three times per token, and
// the products run with 4 rows x JB column vectors of independent accumulators.

extern "C++" {
// dst[p][j0 .. j0+JB) = (sub ? sub[p] - X[p] S : X[p] S) for the np (<= 4) rows starting at X.
// X rows (stride dk), S [dk][dv]; d indexes X lanes in the same (la_widen) order as S rows.
template <int JB>
static void la_rows_times_state(const float *X, int np, const float *S, int dk, int dv, int j0, const HVX_Vector *bcast,
                                float *dst, const float *sub) {
  HVX_Vector a0[JB], a1[JB], a2[JB], a3[JB];
#pragma unroll
  for (int jj = 0; jj < JB; ++jj) {
    a0[jj] = a1[jj] = a2[jj] = a3[jj] = Q6_V_vzero();
  }
  const float *x0 = X;
  const float *x1 = X + (np > 1 ? 1 : 0) * dk;
  const float *x2 = X + (np > 2 ? 2 : 0) * dk;
  const float *x3 = X + (np > 3 ? 3 : 0) * dk;
  for (int vi = 0; vi < dk / 32; ++vi) {
    const HVX_Vector xv0 = vmem(x0 + vi * 32), xv1 = vmem(x1 + vi * 32);
    const HVX_Vector xv2 = vmem(x2 + vi * 32), xv3 = vmem(x3 + vi * 32);
    for (int ln = 0; ln < 32; ++ln) {
      const HVX_Vector b0 = Q6_V_vdelta_VV(xv0, bcast[ln]), b1 = Q6_V_vdelta_VV(xv1, bcast[ln]);
      const HVX_Vector b2 = Q6_V_vdelta_VV(xv2, bcast[ln]), b3 = Q6_V_vdelta_VV(xv3, bcast[ln]);
      const float     *sr = S + (size_t) (vi * 32 + ln) * dv + j0 * 32;
#pragma unroll
      for (int jj = 0; jj < JB; ++jj) {
        const HVX_Vector sv = vmem(sr + jj * 32);
        a0[jj]              = Q6_Vqf32_vadd_Vqf32Vqf32(a0[jj], Q6_Vqf32_vmpy_VsfVsf(sv, b0));
        a1[jj]              = Q6_Vqf32_vadd_Vqf32Vqf32(a1[jj], Q6_Vqf32_vmpy_VsfVsf(sv, b1));
        a2[jj]              = Q6_Vqf32_vadd_Vqf32Vqf32(a2[jj], Q6_Vqf32_vmpy_VsfVsf(sv, b2));
        a3[jj]              = Q6_Vqf32_vadd_Vqf32Vqf32(a3[jj], Q6_Vqf32_vmpy_VsfVsf(sv, b3));
      }
    }
  }
  // Explicit per-row stores: taking the accumulators' addresses forced them to the stack and miscompiled.
#define LA_STORE_ROW(r, acc)                                                       \
  if (np > (r)) {                                                                  \
    const HVX_Vector v = Q6_Vsf_equals_Vqf32(acc);                                 \
    float           *o = dst + (size_t) (r) * dv + off;                            \
    vmem(o)            = sub ? la_sub(vmem(sub + (size_t) (r) * dv + off), v) : v; \
  }
#pragma unroll
  for (int jj = 0; jj < JB; ++jj) {
    const int off = (j0 + jj) * 32;
    LA_STORE_ROW(0, a0[jj]);
    LA_STORE_ROW(1, a1[jj]);
    LA_STORE_ROW(2, a2[jj]);
    LA_STORE_ROW(3, a3[jj]);
  }
#undef LA_STORE_ROW
}

// S[d0 .. d0+4)[j0 .. j0+JB) = decay S + sum_p (K[p] w[p])[d] VN[p].
template <int JB>
static void la_state_update(float *S, int dk, int dv, int d0, int j0, const float *K, const float *w, int n,
                            const float *VN, HVX_Vector vdecay, const HVX_Vector *bcast) {
  HVX_Vector a0[JB], a1[JB], a2[JB], a3[JB];
#pragma unroll
  for (int jj = 0; jj < JB; ++jj) {
    const int off = (j0 + jj) * 32;
    a0[jj]        = Q6_Vqf32_vmpy_VsfVsf(vmem(S + (size_t) (d0 + 0) * dv + off), vdecay);
    a1[jj]        = Q6_Vqf32_vmpy_VsfVsf(vmem(S + (size_t) (d0 + 1) * dv + off), vdecay);
    a2[jj]        = Q6_Vqf32_vmpy_VsfVsf(vmem(S + (size_t) (d0 + 2) * dv + off), vdecay);
    a3[jj]        = Q6_Vqf32_vmpy_VsfVsf(vmem(S + (size_t) (d0 + 3) * dv + off), vdecay);
  }
  const int vi = d0 / 32, ln = d0 % 32;
  for (int p = 0; p < n; ++p) {
    const HVX_Vector kw = la_mul(vmem(K + (size_t) p * dk + vi * 32), la_splat(w[p]));
    const HVX_Vector b0 = Q6_V_vdelta_VV(kw, bcast[ln]), b1 = Q6_V_vdelta_VV(kw, bcast[ln + 1]);
    const HVX_Vector b2 = Q6_V_vdelta_VV(kw, bcast[ln + 2]), b3 = Q6_V_vdelta_VV(kw, bcast[ln + 3]);
    const float     *vr = VN + (size_t) p * dv + j0 * 32;
#pragma unroll
    for (int jj = 0; jj < JB; ++jj) {
      const HVX_Vector vn = vmem(vr + jj * 32);
      a0[jj]              = Q6_Vqf32_vadd_Vqf32Vqf32(a0[jj], Q6_Vqf32_vmpy_VsfVsf(vn, b0));
      a1[jj]              = Q6_Vqf32_vadd_Vqf32Vqf32(a1[jj], Q6_Vqf32_vmpy_VsfVsf(vn, b1));
      a2[jj]              = Q6_Vqf32_vadd_Vqf32Vqf32(a2[jj], Q6_Vqf32_vmpy_VsfVsf(vn, b2));
      a3[jj]              = Q6_Vqf32_vadd_Vqf32Vqf32(a3[jj], Q6_Vqf32_vmpy_VsfVsf(vn, b3));
    }
  }
#pragma unroll
  for (int jj = 0; jj < JB; ++jj) {
    const int off                          = (j0 + jj) * 32;
    vmem(S + (size_t) (d0 + 0) * dv + off) = Q6_Vsf_equals_Vqf32(a0[jj]);
    vmem(S + (size_t) (d0 + 1) * dv + off) = Q6_Vsf_equals_Vqf32(a1[jj]);
    vmem(S + (size_t) (d0 + 2) * dv + off) = Q6_Vsf_equals_Vqf32(a2[jj]);
    vmem(S + (size_t) (d0 + 3) * dv + off) = Q6_Vsf_equals_Vqf32(a3[jj]);
  }
}

}  // extern "C++"

// Y[p] (+)= sum_{r <= p} M[p][r] f[r] R[r]   (lower-triangular LA_CHUNK x LA_CHUNK scalar M, rows of NV vectors).
extern "C++" {
template <int NV>
static void la_tri_rows_t(const float *M, const float *f, const float *R, int rs, int n, float *Y, int accumulate) {
  for (int p = 0; p < n; ++p) {
    HVX_Vector acc[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      acc[i] = accumulate ? Q6_Vqf32_vadd_VsfVsf(vmem(Y + (size_t) p * rs + i * 32), Q6_V_vzero()) : Q6_V_vzero();
    }
    for (int r = 0; r <= p; ++r) {
      const HVX_Vector m = la_splat(M[p * LA_CHUNK + r] * (f ? f[r] : 1.0f));
#pragma unroll
      for (int i = 0; i < NV; ++i) {
        acc[i] = Q6_Vqf32_vadd_Vqf32Vqf32(acc[i], Q6_Vqf32_vmpy_VsfVsf(vmem(R + (size_t) r * rs + i * 32), m));
      }
    }
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      vmem(Y + (size_t) p * rs + i * 32) = Q6_Vsf_equals_Vqf32(acc[i]);
    }
  }
}
}  // extern "C++"

static void la_tri_rows(const float *M, const float *f, const float *R, int rs, int nv, int n, float *Y,
                        int accumulate) {
  switch (nv) {
    case 2:
      la_tri_rows_t<2>(M, f, R, rs, n, Y, accumulate);
      break;
    case 4:
      la_tri_rows_t<4>(M, f, R, rs, n, Y, accumulate);
      break;
    case 6:
      la_tri_rows_t<6>(M, f, R, rs, n, Y, accumulate);
      break;
    default:
      la_tri_rows_t<8>(M, f, R, rs, n, Y, accumulate);
      break;
  }
}

// dst[r][:] = X_r K^T for np (<= 4) rows X_r = src + r * stride (DDR, scalar reads), Kt [dk][LA_CHUNK].
static void la_gram_rows(const float *src, int stride, int np, const float *Kt, int dk, float *dst) {
  const float *x0 = src;
  const float *x1 = src + (np > 1 ? 1 : 0) * stride;
  const float *x2 = src + (np > 2 ? 2 : 0) * stride;
  const float *x3 = src + (np > 3 ? 3 : 0) * stride;
  HVX_Vector   a0 = Q6_V_vzero(), a1 = Q6_V_vzero(), a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
  for (int d = 0; d < dk; ++d) {
    const HVX_Vector kt = vmem(Kt + d * LA_CHUNK);
    a0                  = Q6_Vqf32_vadd_Vqf32Vqf32(a0, Q6_Vqf32_vmpy_VsfVsf(kt, la_splat(x0[d])));
    a1                  = Q6_Vqf32_vadd_Vqf32Vqf32(a1, Q6_Vqf32_vmpy_VsfVsf(kt, la_splat(x1[d])));
    a2                  = Q6_Vqf32_vadd_Vqf32Vqf32(a2, Q6_Vqf32_vmpy_VsfVsf(kt, la_splat(x2[d])));
    a3                  = Q6_Vqf32_vadd_Vqf32Vqf32(a3, Q6_Vqf32_vmpy_VsfVsf(kt, la_splat(x3[d])));
  }
  vmem(dst) = Q6_Vsf_equals_Vqf32(a0);
  if (np > 1) {
    vmem(dst + LA_CHUNK) = Q6_Vsf_equals_Vqf32(a1);
  }
  if (np > 2) {
    vmem(dst + 2 * LA_CHUNK) = Q6_Vsf_equals_Vqf32(a2);
  }
  if (np > 3) {
    vmem(dst + 3 * LA_CHUNK) = Q6_Vsf_equals_Vqf32(a3);
  }
}

// 32 x 32 fp32 transpose: dst[c][r] = src[r * stride + c]. Five rounds of word zips of rows (r, r + 16)
// into rows (2r, 2r + 1); each round rotates the 10-bit (row, column) address left by one bit.
static void la_transpose32(const float *src, int stride, float *b0, float *b1, float *dst) {
  const float *in = src;
  int          is = stride;
  for (int round = 0; round < 5; ++round) {
    float *out = round == 4 ? dst : ((round & 1) ? b1 : b0);
    for (int r = 0; r < 16; ++r) {
      const HVX_VectorPair z       = Q6_W_vshuff_VVR(vmem(in + (r + 16) * is), vmem(in + r * is), -4);
      vmem(out + (2 * r) * 32)     = Q6_V_lo_W(z);
      vmem(out + (2 * r + 1) * 32) = Q6_V_hi_W(z);
    }
    in = out;
    is = 32;
  }
}

static size_t la_chunk_vtcm_floats(int dk, int dv) {
  return (size_t) dk * dv + 4 * (size_t) LA_CHUNK * dk + 2 * (size_t) LA_CHUNK * dv + 2 * LA_CHUNK * 32 +
         (size_t) (2 * dk + dv) / 64 * 2 * LA_MAX_K * 32;
}

static void la_delta_head_chunked(const la_args_t *a, int h, const HVX_Vector *bcast, float *vt, float *sc, int wid) {
  const unsigned long long  t_head = LA_T0();
  unsigned long long        t_ph   = LA_T0();
  const DSPLinearAttnParam *p      = a->p;
  const int                 L      = p->seqLen;
  const int                 D      = p->convDim;
  const int                 Hk     = p->numKHeads;
  const int                 Hv     = p->numVHeads;
  const int                 dk     = p->headKDim;
  const int                 dv     = p->headVDim;
  const int                 nvk    = dk / 32;
  const int                 nvv    = dv / 32;
  const int                 kh     = h / (Hv / Hk);
  const float               qScale = 1.0f / sqrtf((float) dk);
  const int                 C      = LA_CHUNK;
  float                    *S_ddr  = a->rnn_state + (size_t) h * dk * dv;
  float                    *S      = vt;                     // [dk][dv]
  float                    *Kc     = S + (size_t) dk * dv;   // [C][dk] keys (raw, then L2-normalized)
  float                    *Qc     = Kc + (size_t) C * dk;   // [C][dk] queries (raw, then normalized * scale * e^g)
  float                    *KCD    = Qc + (size_t) C * dk;   // [C][dk] T (beta e^g K)
  float                    *Vc     = KCD + (size_t) C * dk;  // [C][dv] values, then the chunk output
  float                    *VN     = Vc + (size_t) C * dv;   // [C][dv] T (beta V), then v_new
  float                    *Kt     = VN + (size_t) C * dv;   // [dk][C] raw keys transposed (lane = token)
  float                    *TB0    = Kt + (size_t) dk * C;   // [32][32] x 2 transpose ping-pong
  float                    *TB1    = TB0 + C * 32;
  HVX_Vector *Wc      = (HVX_Vector *) (TB1 + C * 32);       // fused conv: widened taps, [q blocks, k blocks, v blocks]
  const int   wstride = 2 * p->kernelSize;                   // vectors per block
  float      *T       = sc;                                  // [C][C] raw K K^T, then the corrected triangular matrix
  float      *QK      = T + C * C;                           // [C][C] raw Q K^T, then normalized * decay
  float      *DEC     = QK + C * C;                          // [C][C] exp(gc_i - gc_j), j <= i
  float      *gc      = DEC + C * C;                         // cumulative log decay in the chunk
  float      *beta    = gc + C;
  float      *eg      = beta + C;                            // exp(gc)
  float      *w       = eg + C;                              // exp(gc_last - gc)
  float      *nk      = w + C;
  float      *nq      = nk + C;
  float      *qq      = nq + C;
  __fp16     *wtmp    = (__fp16 *) (qq + C);  // conv tap staging (64 fp16)
  float      *Kst     = qq + 2 * C;           // fused conv: [C][dk] raw K rows in DDR (scalar reads from VTCM are slow)
  float      *Qst     = Kst + (size_t) C * dk;  // fused conv: [C][dk] raw Q rows
  const int   qOff = kh * dk, kOff = Hk * dk + kh * dk, vOff = 2 * Hk * dk + h * dv;

  for (int i = 0; i < dk * dv; i += 32) {
    vmem(S + i) = vmem(S_ddr + i);
  }
  LA_ADD(wid, LA_PH_STATE_IO, t_ph);
  t_ph = LA_T0();
  if (a->fuse_conv) {
    for (int b = 0; b < dk / 64; ++b) {
      la_conv_taps(a, (qOff >> 6) + b, wtmp, Wc + b * wstride);
      la_conv_taps(a, (kOff >> 6) + b, wtmp, Wc + (dk / 64 + b) * wstride);
    }
    for (int b = 0; b < dv / 64; ++b) {
      la_conv_taps(a, (vOff >> 6) + b, wtmp, Wc + (2 * (dk / 64) + b) * wstride);
    }
  }
  for (int c0 = 0; c0 < L; c0 += C) {
    const int n = L - c0 < C ? L - c0 : C;
    if (!a->fuse_conv && c0 + C < L) {  // the next chunk's rows: three strided column windows of the conv output
      const int    nn   = L - c0 - C < C ? L - c0 - C : C;
      const float *next = a->conv_out + (size_t) (c0 + C) * D;
      l2fetch(next + qOff, D * 4, dk * 4, nn, 0);
      l2fetch(next + kOff, D * 4, dk * 4, nn, 0);
      l2fetch(next + vOff, D * 4, dv * 4, nn, 0);
    }
    // Rows of this chunk into VTCM: computed here from qkv when fused, else copied from the conv output.
    if (a->fuse_conv) {
      const HVX_Vector *wq = Wc, *wk = Wc + (dk / 64) * wstride, *wv = Wc + 2 * (dk / 64) * wstride;
      for (int b = 0; b < dk / 64; ++b) {
        la_conv_rows(a, (qOff >> 6) + b, c0, c0 + n, Qc + b * 64, dk, wq + b * wstride);
        la_conv_rows(a, (kOff >> 6) + b, c0, c0 + n, Kc + b * 64, dk, wk + b * wstride);
      }
      for (int b = 0; b < dv / 64; ++b) {
        la_conv_rows(a, (vOff >> 6) + b, c0, c0 + n, Vc + b * 64, dv, wv + b * wstride);
      }
      for (int i = 0; i < n * dk; i += 32) {
        vmem(Kst + i) = vmem(Kc + i);
        vmem(Qst + i) = vmem(Qc + i);
      }
      LA_FENCE();
    }
    for (int t = 0; !a->fuse_conv && t < n; ++t) {
      const float *src = a->conv_out + (size_t) (c0 + t) * D;
      for (int i = 0; i < nvk; ++i) {
        vmem(Qc + (size_t) t * dk + i * 32) = vmem(src + qOff + i * 32);
        vmem(Kc + (size_t) t * dk + i * 32) = vmem(src + kOff + i * 32);
      }
      for (int i = 0; i < nvv; ++i) {
        vmem(Vc + (size_t) t * dv + i * 32) = vmem(src + vOff + i * 32);
      }
    }
    LA_ADD(wid, LA_PH_CONV, t_ph);
    t_ph = LA_T0();
    for (int t = n; t < C; ++t) {  // a short last chunk: finite keys past n for the transpose
      for (int i = 0; i < nvk; ++i) {
        vmem(Kc + (size_t) t * dk + i * 32) = Q6_V_vzero();
      }
    }
    for (int vi = 0; vi < nvk; ++vi) {
      la_transpose32(Kc + vi * 32, dk, TB0, TB1, Kt + (size_t) vi * 32 * C);
    }
    LA_FENCE();
    LA_ADD(wid, LA_PH_TRANSPOSE, t_ph);
    t_ph = LA_T0();
    // Gates for the whole chunk as vectors (lane = token). g is clamped to [-88, 0] (as the OpenCL chunk
    // path) so every exp argument below is <= 0.
    for (int t = 0; t < C; ++t) {
      gc[t]   = t < n ? la_read_gate(a->in1, L, c0 + t, h) : 0.0f;
      beta[t] = t < n ? la_read_gate(a->in2, L, c0 + t, h) : 0.0f;
    }
    LA_FENCE();
    {
      HVX_Vector g = vmem(gc), b = vmem(beta);
      if (p->gateFold) {
        // g = coef * softplus(g + bias), softplus(x) = max(x, 0) + log1p(e^-|x|); log1p(y) = 2 atanh(y / (2 + y)).
        const HVX_Vector absMask = Q6_V_vsplat_R(0x7fffffff);
        const HVX_Vector x       = la_add(g, la_splat(a->gate_const[Hv + h]));
        const HVX_Vector y       = la_exp_nonpos(Q6_V_vxor_VV(Q6_V_vand_VV(x, absMask), Q6_V_vsplat_R(0x80000000)));
        const HVX_Vector s =
          la_mul(la_mul(y, la_splat(0.5f)), la_recip_1_2(la_add(la_mul(y, la_splat(0.5f)), la_splat(1.0f))));
        const HVX_Vector s2   = la_mul(s, s);
        HVX_Vector       poly = la_splat(1.0f / 13.0f);
        poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 11.0f));
        poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 9.0f));
        poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 7.0f));
        poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 5.0f));
        poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 3.0f));
        poly                  = la_add(la_mul(poly, s2), la_splat(1.0f));
        const HVX_Vector sp   = la_add(Q6_Vsf_vmax_VsfVsf(x, Q6_V_vzero()), la_mul(la_mul(s, la_splat(2.0f)), poly));
        g                     = la_mul(sp, la_splat(a->gate_const[h]));
        // beta = sigmoid(b) = 1 / (1 + e) for b >= 0 and e / (1 + e) otherwise, e = e^-|b|.
        const HVX_Vector e    = la_exp_nonpos(Q6_V_vxor_VV(Q6_V_vand_VV(b, absMask), Q6_V_vsplat_R(0x80000000)));
        const HVX_Vector r    = la_recip_1_2(la_add(e, la_splat(1.0f)));
        b                     = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(Q6_V_vzero(), b), la_mul(e, r), r);
      }
      vmem(gc)   = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(g, la_splat(-88.0f)), Q6_V_vzero());
      vmem(beta) = b;
    }
    LA_FENCE();
    float run = 0.0f;
    for (int t = 0; t < C; ++t) {
      run += t < n ? gc[t] : 0.0f;
      gc[t] = run;
    }
    LA_FENCE();
    {
      const HVX_Vector gcv = vmem(gc);
      vmem(eg)             = la_exp_nonpos(gcv);
      vmem(w)              = la_exp_nonpos(la_sub(la_splat(gc[n - 1]), gcv));
    }
    LA_FENCE();
    LA_ADD(wid, LA_PH_GATES, t_ph);
    t_ph = LA_T0();
    // Raw Gram rows K_i K^T and Q_i K^T as whole vectors (lane = j), against the transposed keys.
    for (int p0 = 0; p0 < n; p0 += 4) {
      const int np = n - p0 < 4 ? n - p0 : 4;
      if (a->fuse_conv) {
        la_gram_rows(Kst + (size_t) p0 * dk, dk, np, Kt, dk, T + p0 * C);
        la_gram_rows(Qst + (size_t) p0 * dk, dk, np, Kt, dk, QK + p0 * C);
      } else {
        la_gram_rows(a->conv_out + (size_t) (c0 + p0) * D + kOff, D, np, Kt, dk, T + p0 * C);
        la_gram_rows(a->conv_out + (size_t) (c0 + p0) * D + qOff, D, np, Kt, dk, QK + p0 * C);
      }
    }
    LA_FENCE();
    LA_ADD(wid, LA_PH_GRAM, t_ph);
    t_ph = LA_T0();
    for (int i0 = 0; i0 < n; i0 += 4) {  // |q_i|^2, four independent reductions at a time
      const float *q0 = Qc + (size_t) i0 * dk;
      const float *q1 = Qc + (size_t) (i0 + 1 < n ? i0 + 1 : i0) * dk;
      const float *q2 = Qc + (size_t) (i0 + 2 < n ? i0 + 2 : i0) * dk;
      const float *q3 = Qc + (size_t) (i0 + 3 < n ? i0 + 3 : i0) * dk;
      HVX_Vector   s0 = Q6_V_vzero(), s1 = Q6_V_vzero(), s2 = Q6_V_vzero(), s3 = Q6_V_vzero();
      for (int v = 0; v < nvk; ++v) {
        const HVX_Vector x0 = vmem(q0 + v * 32), x1 = vmem(q1 + v * 32);
        const HVX_Vector x2 = vmem(q2 + v * 32), x3 = vmem(q3 + v * 32);
        s0 = Q6_Vqf32_vadd_Vqf32Vqf32(s0, Q6_Vqf32_vmpy_VsfVsf(x0, x0));
        s1 = Q6_Vqf32_vadd_Vqf32Vqf32(s1, Q6_Vqf32_vmpy_VsfVsf(x1, x1));
        s2 = Q6_Vqf32_vadd_Vqf32Vqf32(s2, Q6_Vqf32_vmpy_VsfVsf(x2, x2));
        s3 = Q6_Vqf32_vadd_Vqf32Vqf32(s3, Q6_Vqf32_vmpy_VsfVsf(x3, x3));
      }
      s0 = Q6_Vsf_equals_Vqf32(s0);
      s1 = Q6_Vsf_equals_Vqf32(s1);
      s2 = Q6_Vsf_equals_Vqf32(s2);
      s3 = Q6_Vsf_equals_Vqf32(s3);
      for (int sh = 64; sh >= 4; sh >>= 1) {
        s0 = la_add(s0, Q6_V_vror_VR(s0, sh));
        s1 = la_add(s1, Q6_V_vror_VR(s1, sh));
        s2 = la_add(s2, Q6_V_vror_VR(s2, sh));
        s3 = la_add(s3, Q6_V_vror_VR(s3, sh));
      }
      qq[i0]     = la_lane0(s0);
      qq[i0 + 1] = la_lane0(s1);
      qq[i0 + 2] = la_lane0(s2);
      qq[i0 + 3] = la_lane0(s3);
    }
    for (int i = 0; i < C; ++i) {
      nk[i] = i < n && p->useL2Norm ? 1.0f / sqrtf(T[i * C + i] + 1e-6f) : 1.0f;
      nq[i] = i < n && p->useL2Norm ? qScale / sqrtf(qq[i] + 1e-6f) : qScale;
    }
    LA_FENCE();
    LA_ADD(wid, LA_PH_NORMS, t_ph);
    t_ph = LA_T0();
    // Per row i, as vectors over j: decay exp(gc_i - gc_j) (argument <= 0 for j <= i; lanes j > i are
    // clamped to 1 and unused), attn = -(beta_i nk_i k_i . nk_j k_j) decay masked to j < i, and the
    // scaled Q K^T. Then the Neumann / forward-substitution correction + I.
    {
      const HVX_Vector gcv  = vmem(gc);
      const HVX_Vector nkv  = vmem(nk);
      const HVX_Vector iota = vmem(kLaIota);
      for (int i = 0; i < n; ++i) {
        const HVX_Vector dec = la_exp_nonpos(la_sub(la_splat(gc[i]), gcv));
        const HVX_Vector kd  = la_mul(nkv, dec);
        const HVX_Vector a   = la_mul(la_mul(vmem(T + i * C), kd), la_splat(-beta[i] * nk[i]));
        vmem(T + i * C)      = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(i), iota), a, Q6_V_vzero());
        vmem(QK + i * C)     = la_mul(la_mul(vmem(QK + i * C), kd), la_splat(nq[i]));
      }
    }
    LA_FENCE();
    LA_ADD(wid, LA_PH_DECAY, t_ph);
    t_ph = LA_T0();
    // Forward substitution on whole rows: T_i = A_i + sum_{m < i} A_i[m] T_m (T_m already corrected, its
    // lanes >= m are zero), i.e. the HF Neumann row update, with one vector FMA per (i, m).
    for (int i = 1; i < n; ++i) {
      HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(vmem(T + i * C), Q6_V_vzero());
      for (int m = 1; m < i; ++m) {
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(vmem(T + m * C), la_splat(T[i * C + m])));
      }
      vmem(T + i * C) = Q6_Vsf_equals_Vqf32(acc);
    }
    LA_FENCE();
    for (int i = 0; i < n; ++i) {
      T[i * C + i] = 1.0f;
    }
    LA_ADD(wid, LA_PH_FWDSUB, t_ph);
    t_ph = LA_T0();
    // Normalize the rows: K by its L2 norm, Q by norm * scale * e^g (the inter-chunk term).
    for (int t = 0; t < n; ++t) {
      const HVX_Vector vk = la_splat(nk[t]), vq = la_splat(nq[t] * eg[t]);
      for (int i = 0; i < nvk; ++i) {
        vmem(Kc + (size_t) t * dk + i * 32) = la_mul(vmem(Kc + (size_t) t * dk + i * 32), vk);
        vmem(Qc + (size_t) t * dk + i * 32) = la_mul(vmem(Qc + (size_t) t * dk + i * 32), vq);
      }
    }
    LA_ADD(wid, LA_PH_NORMALIZE, t_ph);
    t_ph = LA_T0();
    // VN = T (beta V);  KCD = T (beta e^g K).
    la_tri_rows(T, beta, Vc, dv, nvv, n, VN, 0);
    for (int t = 0; t < n; ++t) {
      eg[t] *= beta[t];  // beta e^g, only used for KCD from here on
    }
    la_tri_rows(T, eg, Kc, dk, nvk, n, KCD, 0);
    LA_ADD(wid, LA_PH_TRI_IN, t_ph);
    t_ph = LA_T0();
    // v_new = VN - KCD S;  out = (Q e^g) S + QK v_new (into Vc);  S = e^gl S + (K w)^T v_new.
    for (int p0 = 0; p0 < n; p0 += 4) {
      const int np = n - p0 < 4 ? n - p0 : 4;
      for (int j0 = 0; j0 < nvv; j0 += 4) {
        if (nvv - j0 >= 4) {
          la_rows_times_state<4>(KCD + (size_t) p0 * dk, np, S, dk, dv, j0, bcast, VN + (size_t) p0 * dv,
                                 VN + (size_t) p0 * dv);
          la_rows_times_state<4>(Qc + (size_t) p0 * dk, np, S, dk, dv, j0, bcast, Vc + (size_t) p0 * dv, NULL);
        } else {
          la_rows_times_state<2>(KCD + (size_t) p0 * dk, np, S, dk, dv, j0, bcast, VN + (size_t) p0 * dv,
                                 VN + (size_t) p0 * dv);
          la_rows_times_state<2>(Qc + (size_t) p0 * dk, np, S, dk, dv, j0, bcast, Vc + (size_t) p0 * dv, NULL);
        }
      }
    }
    LA_ADD(wid, LA_PH_ROWS_STATE, t_ph);
    t_ph = LA_T0();
    la_tri_rows(QK, NULL, VN, dv, nvv, n, Vc, 1);
    LA_ADD(wid, LA_PH_TRI_OUT, t_ph);
    t_ph                    = LA_T0();
    const HVX_Vector vglast = la_splat(expf(gc[n - 1]));
    for (int d0 = 0; d0 < dk; d0 += 4) {
      for (int j0 = 0; j0 < nvv; j0 += 4) {
        if (nvv - j0 >= 4) {
          la_state_update<4>(S, dk, dv, d0, j0, Kc, w, n, VN, vglast, bcast);
        } else {
          la_state_update<2>(S, dk, dv, d0, j0, Kc, w, n, VN, vglast, bcast);
        }
      }
    }
    LA_ADD(wid, LA_PH_STATE_UPD, t_ph);
    t_ph               = LA_T0();
    const size_t plane = (size_t) L * Hv;
    for (int t = 0; t < n; ++t) {
      for (int jb = 0; jb < dv / 64; ++jb) {
        vmem(a->out + ((size_t) jb * plane + (size_t) (c0 + t) * Hv + h) * 64) =
          la_narrow(vmem(Vc + (size_t) t * dv + jb * 64), vmem(Vc + (size_t) t * dv + jb * 64 + 32));
      }
    }
    LA_ADD(wid, LA_PH_OUT_STORE, t_ph);
    t_ph = LA_T0();
  }
  for (int i = 0; i < dk * dv; i += 32) {
    vmem(S_ddr + i) = vmem(S + i);
  }
  LA_ADD(wid, LA_PH_STATE_IO, t_ph);
  LA_ADD(wid, LA_PH_HEAD_TOTAL, t_head);
}

// ---- Chunked prefill with the matrix stages on HMX ------------------------------------------------------
//
// The HVX chunked path above runs five matrix-shaped stages per chunk and head in fp32 (rows x state,
// Gram, state update, the two triangular products): 58 of its 88 ms on Qwen3.5-0.8B. Here they become
// fp16 HMX products, and the schedule is inverted to make that pay: instead of every worker owning a head
// for all of its chunks, this thread walks the chunks for a batch of heads, the workers run each HVX phase
// with one head per job, and this thread issues every head's products between the phases. Measured unit
// costs on v81 decide that shape: an HMX output tile is 6-23 ns, an HMX -> HVX drain about 16 ns, but a
// worker barrier is 4.9 us -- so the products are nearly free and the barriers are what must be few.
// Two barriers per chunk, for all heads at once.
//
// Indexing. Every row here is in memory-position order (the order la_widen leaves channels in), and every
// product contracts over memory positions on both sides, so no permutation is ever undone: the HVX path
// already relies on this (rows_times_state broadcasts lane m of X against row m of S). A tile vector is
// la_narrow(row 2i, row 2i + 1), which interleaves the two rows exactly as an HMX tile wants them, and
// la_widen is its inverse for reading products back.
//
// Changes from the HVX path, all exact rewrites:
//  - K and Q are L2-normalised before the Gram product (the HVX path takes |k| from the Gram diagonal),
//    so the fp16 Gram values stay within [-1, 1] instead of reaching |k|^2.
//  - KCD S = T (beta e^g Kn) S is computed as T (diag(beta e^g) (Kn S)), and (Q e^g) S as
//    diag(e^g) (Qn S), so Kn S and Qn S share the Gram's operand tiles and run in the same HMX round.
//  - VN = T (beta V) - KCD S becomes T Z with Z = beta (V - e^g (Kn S)): one product instead of two.
// The recurrent state S stays fp32 and is only ever converted to fp16 as an operand; the forward
// substitution, the gates and the decay stay fp32 on HVX.

#define LA_HMX_MAX_BLK 4  // head dims up to 128 (4 blocks of 32)

typedef struct {
  int         h, n, c0;
  // VTCM, fp32 rows (memory-position order)
  float      *S;    // [dk][dv] master state
  float      *Kn;   // [C][dk] keys, L2-normalised
  float      *Qn;   // [C][dk] queries, L2-normalised and scaled
  float      *Vc;   // [C][dv] values, then the chunk output
  float      *Knt;  // [dk][C] Kn transposed
  float      *TB0, *TB1;
  float      *Zr;   // [C][dv] Z rows
  HVX_Vector *Wc;   // fused conv taps
  // VTCM, fp16 HMX operand tiles
  __fp16     *tKn, *tQn;  // A, [kb]
  __fp16     *tKnt;       // W, [kb]
  __fp16     *tS;         // W, [cb][kb]
  __fp16     *tT, *tQK;   // A, one tile each
  __fp16     *tZ;         // W, [cb]
  __fp16     *tKwT;       // A, [rb]
  // VTCM, fp16 HMX output tiles
  __fp16     *oT, *oQK;                 // one tile each
  __fp16     *oKnS, *oQnS, *oVN, *oO2;  // [cb]
  __fp16     *oSu;                      // [rb][cb]
  // DDR scratch, scalars (fast scalar reads through L2)
  float      *T, *QK, *gc, *beta, *eg, *w, *nk, *nq, *qq, *kk;
  __fp16     *wtmp;
} la_hmx_head_t;

typedef struct {
  const la_args_t    *a;
  la_hmx_head_t      *heads;
  int                 nheads;  // in this batch
  int                 phase;
  int                 nw;
  worker_synctoken_t *sync;
} la_hmx_task_t;

enum { LA_HMX_PH_INIT = 0, LA_HMX_PH_B = 1, LA_HMX_PH_CA = 2 };

static size_t la_hmx_align(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

// Per-head VTCM bytes; tiles are kept 2 KB aligned.
static size_t la_hmx_head_vtcm_bytes(int dk, int dv) {
  const int C = LA_CHUNK, nkb = dk / 32, nvb = dv / 32;
  size_t    f32   = (size_t) dk * dv + 2 * (size_t) C * dk + 2 * (size_t) C * dv + (size_t) dk * C + 2 * C * 32;
  size_t    wc    = (size_t) (2 * dk + dv) / 64 * 2 * LA_MAX_K * 32;          // floats
  size_t    tiles = (size_t) (2 * nkb + nkb + nvb * nkb + 1 + 1 + nvb + nkb)  // operands
                    + (size_t) (1 + 1 + 4 * nvb + nkb * nvb);                 // outputs
  return la_hmx_align((f32 + wc) * sizeof(float), 2048) + tiles * HMX_FP16_TILE_SIZE;
}

static void la_hmx_head_init(la_hmx_head_t *H, const la_args_t *a, int h, int slot, uint8_t *vtcm) {
  const DSPLinearAttnParam *p  = a->p;
  const int                 dk = p->headKDim, dv = p->headVDim, C = LA_CHUNK, nkb = dk / 32, nvb = dv / 32;
  memset(H, 0, sizeof(*H));
  H->h     = h;
  float *f = (float *) vtcm;
  H->S     = f;
  f += (size_t) dk * dv;
  H->Kn = f;
  f += (size_t) C * dk;
  H->Qn = f;
  f += (size_t) C * dk;
  H->Vc = f;
  f += (size_t) C * dv;
  H->Zr = f;
  f += (size_t) C * dv;
  H->Knt = f;
  f += (size_t) dk * C;
  H->TB0 = f;
  f += C * 32;
  H->TB1 = f;
  f += C * 32;
  H->Wc = (HVX_Vector *) f;
  f += (size_t) (2 * dk + dv) / 64 * 2 * LA_MAX_K * 32;
  __fp16 *t = (__fp16 *) (vtcm + la_hmx_align((size_t) ((uint8_t *) f - vtcm), 2048));
#define LA_HMX_TAKE(ptr, ntiles)                   \
  do {                                             \
    (ptr) = t;                                     \
    t += (size_t) (ntiles) * HMX_FP16_TILE_N_ELMS; \
  } while (0)
  LA_HMX_TAKE(H->tKn, nkb);
  LA_HMX_TAKE(H->tQn, nkb);
  LA_HMX_TAKE(H->tKnt, nkb);
  LA_HMX_TAKE(H->tS, nvb * nkb);
  LA_HMX_TAKE(H->tT, 1);
  LA_HMX_TAKE(H->tQK, 1);
  LA_HMX_TAKE(H->tZ, nvb);
  LA_HMX_TAKE(H->tKwT, nkb);
  LA_HMX_TAKE(H->oT, 1);
  LA_HMX_TAKE(H->oQK, 1);
  LA_HMX_TAKE(H->oKnS, nvb);
  LA_HMX_TAKE(H->oQnS, nvb);
  LA_HMX_TAKE(H->oVN, nvb);
  LA_HMX_TAKE(H->oO2, nvb);
  LA_HMX_TAKE(H->oSu, nkb * nvb);
#undef LA_HMX_TAKE
  float *sc =
    (float *) ((uint8_t *) a->scratch + (size_t) slot * DSP_LINEAR_ATTN_SCRATCH_BYTES + LA_SCRATCH_VECS * 128);
  H->T = sc;
  sc += C * C;
  H->QK = sc;
  sc += C * C;
  H->gc = sc;
  sc += C;
  H->beta = sc;
  sc += C;
  H->eg = sc;
  sc += C;
  H->w = sc;
  sc += C;
  H->nk = sc;
  sc += C;
  H->nq = sc;
  sc += C;
  H->qq = sc;
  sc += C;
  H->kk = sc;
  sc += C;
  H->wtmp = (__fp16 *) sc;
}

// 32 rows x (32 * nblk) fp32 (row stride `stride` floats) -> nblk tiles, block b at tiles + b * tile_stride.
static void la_hmx_rows_to_tiles(const float *src, int stride, int nblk, __fp16 *tiles, size_t tile_stride) {
  for (int b = 0; b < nblk; ++b) {
    HVX_Vector *dst = (HVX_Vector *) (tiles + (size_t) b * tile_stride);
    for (int i = 0; i < 16; ++i) {
      dst[i] =
        la_narrow(vmem(src + (size_t) (2 * i) * stride + b * 32), vmem(src + (size_t) (2 * i + 1) * stride + b * 32));
    }
  }
}

// Same, each row scaled lane-wise by `colscale` (32 lanes), for (K w)^T.
static void la_hmx_rows_to_tiles_scaled(const float *src, int stride, const float *colscale, __fp16 *tile) {
  const HVX_Vector cs  = vmem(colscale);
  HVX_Vector      *dst = (HVX_Vector *) tile;
  for (int i = 0; i < 16; ++i) {
    dst[i] = la_narrow(la_mul(vmem(src + (size_t) (2 * i) * stride), cs),
                       la_mul(vmem(src + (size_t) (2 * i + 1) * stride), cs));
  }
}

// One output tile back into 32 rows of fp32 at column block offset `col` (floats), optionally accumulating.
static void la_hmx_tile_to_rows(const __fp16 *tile, float *dst, int stride, int col) {
  const HVX_Vector *src = (const HVX_Vector *) tile;
  for (int i = 0; i < 16; ++i) {
    HVX_Vector lo, hi;
    la_widen(src[i], &lo, &hi);
    vmem(dst + (size_t) (2 * i) * stride + col)     = lo;
    vmem(dst + (size_t) (2 * i + 1) * stride + col) = hi;
  }
}

// |row|^2 for C rows of `nvec` vectors each (stride floats), four independent reductions at a time.
static void la_hmx_row_sumsq(const float *X, int stride, int nvec, float *out) {
  for (int i0 = 0; i0 < LA_CHUNK; i0 += 4) {
    HVX_Vector   s0 = Q6_V_vzero(), s1 = Q6_V_vzero(), s2 = Q6_V_vzero(), s3 = Q6_V_vzero();
    const float *x0 = X + (size_t) i0 * stride, *x1 = x0 + stride, *x2 = x1 + stride, *x3 = x2 + stride;
    for (int v = 0; v < nvec; ++v) {
      const HVX_Vector a0 = vmem(x0 + v * 32), a1 = vmem(x1 + v * 32), a2 = vmem(x2 + v * 32), a3 = vmem(x3 + v * 32);
      s0 = Q6_Vqf32_vadd_Vqf32Vqf32(s0, Q6_Vqf32_vmpy_VsfVsf(a0, a0));
      s1 = Q6_Vqf32_vadd_Vqf32Vqf32(s1, Q6_Vqf32_vmpy_VsfVsf(a1, a1));
      s2 = Q6_Vqf32_vadd_Vqf32Vqf32(s2, Q6_Vqf32_vmpy_VsfVsf(a2, a2));
      s3 = Q6_Vqf32_vadd_Vqf32Vqf32(s3, Q6_Vqf32_vmpy_VsfVsf(a3, a3));
    }
    out[i0]     = la_lane0(la_hsum(Q6_Vsf_equals_Vqf32(s0)));
    out[i0 + 1] = la_lane0(la_hsum(Q6_Vsf_equals_Vqf32(s1)));
    out[i0 + 2] = la_lane0(la_hsum(Q6_Vsf_equals_Vqf32(s2)));
    out[i0 + 3] = la_lane0(la_hsum(Q6_Vsf_equals_Vqf32(s3)));
  }
}

// Phase A: this chunk's rows, their L2 normalisation, the gates, and every operand tile HMX round 1 needs.
static void la_hmx_phase_a(const la_args_t *a, la_hmx_head_t *H, int c) {
  const DSPLinearAttnParam *p = a->p;
  const int   L = p->seqLen, D = p->convDim, Hk = p->numKHeads, Hv = p->numVHeads, dk = p->headKDim, dv = p->headVDim;
  const int   nvk = dk / 32, nvv = dv / 32, nkb = dk / 32, nvb = dv / 32, C = LA_CHUNK, h = H->h;
  const int   kh     = h / (Hv / Hk);
  const float qScale = 1.0f / sqrtf((float) dk);
  const int   qOff = kh * dk, kOff = Hk * dk + kh * dk, vOff = 2 * Hk * dk + h * dv;
  const int   c0 = c * C, n = L - c0 < C ? L - c0 : C;
  H->c0 = c0;
  H->n  = n;
  if (!a->fuse_conv && c0 + C < L) {
    const int    nn   = L - c0 - C < C ? L - c0 - C : C;
    const float *next = a->conv_out + (size_t) (c0 + C) * D;
    l2fetch(next + qOff, D * 4, dk * 4, nn, 0);
    l2fetch(next + kOff, D * 4, dk * 4, nn, 0);
    l2fetch(next + vOff, D * 4, dv * 4, nn, 0);
  }
  if (a->fuse_conv) {
    const int         wstride = 2 * p->kernelSize;
    const HVX_Vector *wq = H->Wc, *wk = H->Wc + (dk / 64) * wstride, *wv = H->Wc + 2 * (dk / 64) * wstride;
    for (int b = 0; b < dk / 64; ++b) {
      la_conv_rows(a, (qOff >> 6) + b, c0, c0 + n, H->Qn + b * 64, dk, wq + b * wstride);
      la_conv_rows(a, (kOff >> 6) + b, c0, c0 + n, H->Kn + b * 64, dk, wk + b * wstride);
    }
    for (int b = 0; b < dv / 64; ++b) {
      la_conv_rows(a, (vOff >> 6) + b, c0, c0 + n, H->Vc + b * 64, dv, wv + b * wstride);
    }
  } else {
    for (int t = 0; t < n; ++t) {
      const float *src = a->conv_out + (size_t) (c0 + t) * D;
      for (int i = 0; i < nvk; ++i) {
        vmem(H->Qn + (size_t) t * dk + i * 32) = vmem(src + qOff + i * 32);
        vmem(H->Kn + (size_t) t * dk + i * 32) = vmem(src + kOff + i * 32);
      }
      for (int i = 0; i < nvv; ++i) {
        vmem(H->Vc + (size_t) t * dv + i * 32) = vmem(src + vOff + i * 32);
      }
    }
  }
  // A short last chunk: zero rows past n, so every product below sees exact zeros there.
  for (int t = n; t < C; ++t) {
    for (int i = 0; i < nvk; ++i) {
      vmem(H->Qn + (size_t) t * dk + i * 32) = Q6_V_vzero();
      vmem(H->Kn + (size_t) t * dk + i * 32) = Q6_V_vzero();
    }
    for (int i = 0; i < nvv; ++i) {
      vmem(H->Vc + (size_t) t * dv + i * 32) = Q6_V_vzero();
    }
  }
  LA_FENCE();
  la_hmx_row_sumsq(H->Kn, dk, nvk, H->kk);
  la_hmx_row_sumsq(H->Qn, dk, nvk, H->qq);
  for (int i = 0; i < C; ++i) {
    H->nk[i] = i < n ? 1.0f / sqrtf(H->kk[i] + 1e-6f) : 1.0f;
    H->nq[i] = i < n ? qScale / sqrtf(H->qq[i] + 1e-6f) : qScale;
  }
  LA_FENCE();
  for (int t = 0; t < n; ++t) {
    const HVX_Vector vk = la_splat(H->nk[t]), vq = la_splat(H->nq[t]);
    for (int i = 0; i < nvk; ++i) {
      vmem(H->Kn + (size_t) t * dk + i * 32) = la_mul(vmem(H->Kn + (size_t) t * dk + i * 32), vk);
      vmem(H->Qn + (size_t) t * dk + i * 32) = la_mul(vmem(H->Qn + (size_t) t * dk + i * 32), vq);
    }
  }
  for (int vi = 0; vi < nvk; ++vi) {
    la_transpose32(H->Kn + vi * 32, dk, H->TB0, H->TB1, H->Knt + (size_t) vi * 32 * C);
  }
  // Gates, exactly as the HVX path.
  float *gc = H->gc, *beta = H->beta;
  for (int t = 0; t < C; ++t) {
    gc[t]   = t < n ? la_read_gate(a->in1, L, c0 + t, h) : 0.0f;
    beta[t] = t < n ? la_read_gate(a->in2, L, c0 + t, h) : 0.0f;
  }
  LA_FENCE();
  {
    HVX_Vector g = vmem(gc), b = vmem(beta);
    if (p->gateFold) {
      const HVX_Vector absMask = Q6_V_vsplat_R(0x7fffffff);
      const HVX_Vector x       = la_add(g, la_splat(a->gate_const[Hv + h]));
      const HVX_Vector y       = la_exp_nonpos(Q6_V_vxor_VV(Q6_V_vand_VV(x, absMask), Q6_V_vsplat_R(0x80000000)));
      const HVX_Vector sx =
        la_mul(la_mul(y, la_splat(0.5f)), la_recip_1_2(la_add(la_mul(y, la_splat(0.5f)), la_splat(1.0f))));
      const HVX_Vector s2   = la_mul(sx, sx);
      HVX_Vector       poly = la_splat(1.0f / 13.0f);
      poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 11.0f));
      poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 9.0f));
      poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 7.0f));
      poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 5.0f));
      poly                  = la_add(la_mul(poly, s2), la_splat(1.0f / 3.0f));
      poly                  = la_add(la_mul(poly, s2), la_splat(1.0f));
      const HVX_Vector sp   = la_add(Q6_Vsf_vmax_VsfVsf(x, Q6_V_vzero()), la_mul(la_mul(sx, la_splat(2.0f)), poly));
      g                     = la_mul(sp, la_splat(a->gate_const[h]));
      const HVX_Vector e    = la_exp_nonpos(Q6_V_vxor_VV(Q6_V_vand_VV(b, absMask), Q6_V_vsplat_R(0x80000000)));
      const HVX_Vector r    = la_recip_1_2(la_add(e, la_splat(1.0f)));
      b                     = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(Q6_V_vzero(), b), la_mul(e, r), r);
    }
    vmem(gc)   = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(g, la_splat(-88.0f)), Q6_V_vzero());
    vmem(beta) = b;
  }
  LA_FENCE();
  float run = 0.0f;
  for (int t = 0; t < C; ++t) {
    run += t < n ? gc[t] : 0.0f;
    gc[t] = run;
    if (t >= n) {
      beta[t] = 0.0f;  // the folded sigmoid maps a padded 0 to 0.5; padded tokens must not contribute
    }
  }
  LA_FENCE();
  {
    const HVX_Vector gcv = vmem(gc);
    vmem(H->eg)          = la_exp_nonpos(gcv);
    vmem(H->w)           = la_exp_nonpos(la_sub(la_splat(gc[n - 1]), gcv));
  }
  LA_FENCE();
  // Operand tiles for round 1: Kn and Qn as A, Kn^T and S as W ([cb][kb]).
  la_hmx_rows_to_tiles(H->Kn, dk, nkb, H->tKn, HMX_FP16_TILE_N_ELMS);
  la_hmx_rows_to_tiles(H->Qn, dk, nkb, H->tQn, HMX_FP16_TILE_N_ELMS);
  for (int kb = 0; kb < nkb; ++kb) {
    la_hmx_rows_to_tiles(H->Knt + (size_t) kb * 32 * C, C, 1, H->tKnt + (size_t) kb * HMX_FP16_TILE_N_ELMS, 0);
    la_hmx_rows_to_tiles(H->S + (size_t) kb * 32 * dv, dv, nvb, H->tS + (size_t) kb * HMX_FP16_TILE_N_ELMS,
                         (size_t) nkb * HMX_FP16_TILE_N_ELMS);
  }
}

// Phase B: decay, forward substitution and Z, then the operand tiles of HMX round 2.
static void la_hmx_phase_b(const la_args_t *a, la_hmx_head_t *H) {
  const DSPLinearAttnParam *p = a->p;
  const int dk = p->headKDim, dv = p->headVDim, nvv = dv / 32, nkb = dk / 32, nvb = dv / 32, C = LA_CHUNK;
  const int n = H->n;
  float    *T = H->T, *QK = H->QK;
  la_hmx_tile_to_rows(H->oT, T, C, 0);
  la_hmx_tile_to_rows(H->oQK, QK, C, 0);
  LA_FENCE();
  {
    const HVX_Vector gcv  = vmem(H->gc);
    const HVX_Vector iota = vmem(kLaIota);
    for (int i = 0; i < n; ++i) {
      const HVX_Vector dec = la_exp_nonpos(la_sub(la_splat(H->gc[i]), gcv));
      const HVX_Vector av  = la_mul(la_mul(vmem(T + i * C), dec), la_splat(-H->beta[i]));
      vmem(T + i * C)      = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(i), iota), av, Q6_V_vzero());
      // The HVX path only ever reads the lower triangle (with the diagonal) of QK; a full product needs
      // the rest to be zero.
      vmem(QK + i * C) =
        Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(i + 1), iota), la_mul(vmem(QK + i * C), dec), Q6_V_vzero());
    }
    for (int i = n; i < C; ++i) {
      vmem(T + i * C)  = Q6_V_vzero();
      vmem(QK + i * C) = Q6_V_vzero();
    }
  }
  LA_FENCE();
  for (int i = 1; i < n; ++i) {
    HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(vmem(T + i * C), Q6_V_vzero());
    for (int m = 1; m < i; ++m) {
      acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(vmem(T + m * C), la_splat(T[i * C + m])));
    }
    vmem(T + i * C) = Q6_Vsf_equals_Vqf32(acc);
  }
  LA_FENCE();
  for (int i = 0; i < n; ++i) {
    T[i * C + i] = 1.0f;
  }
  LA_FENCE();
  la_hmx_rows_to_tiles(T, C, 1, H->tT, 0);
  la_hmx_rows_to_tiles(QK, C, 1, H->tQK, 0);
  // Z = beta (V - e^g (Kn S)).
  for (int cb = 0; cb < nvb; ++cb) {
    la_hmx_tile_to_rows(H->oKnS + (size_t) cb * HMX_FP16_TILE_N_ELMS, H->Zr, dv, cb * 32);
  }
  for (int t = 0; t < C; ++t) {
    const HVX_Vector vb = la_splat(t < n ? H->beta[t] : 0.0f), ve = la_splat(H->eg[t]);
    for (int v = 0; v < nvv; ++v) {
      float *z = H->Zr + (size_t) t * dv + v * 32;
      vmem(z)  = la_mul(la_sub(vmem(H->Vc + (size_t) t * dv + v * 32), la_mul(vmem(z), ve)), vb);
    }
  }
  la_hmx_rows_to_tiles(H->Zr, dv, nvb, H->tZ, HMX_FP16_TILE_N_ELMS);
  // (Kn w)^T as A, one tile per 32-row block of dk; lanes of a Kn^T row are tokens.
  for (int rb = 0; rb < nkb; ++rb) {
    la_hmx_rows_to_tiles_scaled(H->Knt + (size_t) rb * 32 * C, C, H->w, H->tKwT + (size_t) rb * HMX_FP16_TILE_N_ELMS);
  }
}

// Phase C: this chunk's output and the state update.
static void la_hmx_phase_c(const la_args_t *a, la_hmx_head_t *H) {
  const DSPLinearAttnParam *p = a->p;
  const int                 L = p->seqLen, Hv = p->numVHeads, dk = p->headKDim, dv = p->headVDim, nvv = dv / 32;
  const int                 nkb = dk / 32, nvb = dv / 32, n = H->n, c0 = H->c0, h = H->h;
  for (int cb = 0; cb < nvb; ++cb) {
    la_hmx_tile_to_rows(H->oQnS + (size_t) cb * HMX_FP16_TILE_N_ELMS, H->Vc, dv, cb * 32);
    la_hmx_tile_to_rows(H->oO2 + (size_t) cb * HMX_FP16_TILE_N_ELMS, H->Zr, dv, cb * 32);
  }
  LA_FENCE();
  for (int t = 0; t < n; ++t) {
    const HVX_Vector ve = la_splat(H->eg[t]);
    for (int v = 0; v < nvv; ++v) {
      float *o = H->Vc + (size_t) t * dv + v * 32;
      vmem(o)  = la_add(la_mul(vmem(o), ve), vmem(H->Zr + (size_t) t * dv + v * 32));
    }
  }
  const size_t plane = (size_t) L * Hv;
  for (int t = 0; t < n; ++t) {
    for (int jb = 0; jb < dv / 64; ++jb) {
      vmem(a->out + ((size_t) jb * plane + (size_t) (c0 + t) * Hv + h) * 64) =
        la_narrow(vmem(H->Vc + (size_t) t * dv + jb * 64), vmem(H->Vc + (size_t) t * dv + jb * 64 + 32));
    }
  }
  // S = e^gl S + (Kn w)^T VN, fp32.
  const HVX_Vector vgl = la_splat(expf(H->gc[n - 1]));
  for (int rb = 0; rb < nkb; ++rb) {
    for (int cb = 0; cb < nvb; ++cb) {
      const HVX_Vector *src = (const HVX_Vector *) (H->oSu + (size_t) (rb * nvb + cb) * HMX_FP16_TILE_N_ELMS);
      for (int i = 0; i < 16; ++i) {
        HVX_Vector lo, hi;
        la_widen(src[i], &lo, &hi);
        float *s0 = H->S + (size_t) (rb * 32 + 2 * i) * dv + cb * 32;
        float *s1 = s0 + dv;
        vmem(s0)  = la_add(la_mul(vmem(s0), vgl), lo);
        vmem(s1)  = la_add(la_mul(vmem(s1), vgl), hi);
      }
    }
  }
}

static void la_hmx_head_begin(const la_args_t *a, la_hmx_head_t *H) {
  const DSPLinearAttnParam *p  = a->p;
  const int                 dk = p->headKDim, dv = p->headVDim, Hk = p->numKHeads, Hv = p->numVHeads, h = H->h;
  const float              *S_ddr = a->rnn_state + (size_t) h * dk * dv;
  for (int i = 0; i < dk * dv; i += 32) {
    vmem(H->S + i) = vmem(S_ddr + i);
  }
  if (a->fuse_conv) {
    const int kh = h / (Hv / Hk), wstride = 2 * p->kernelSize;
    const int qOff = kh * dk, kOff = Hk * dk + kh * dk, vOff = 2 * Hk * dk + h * dv;
    for (int b = 0; b < dk / 64; ++b) {
      la_conv_taps(a, (qOff >> 6) + b, H->wtmp, H->Wc + b * wstride);
      la_conv_taps(a, (kOff >> 6) + b, H->wtmp, H->Wc + (dk / 64 + b) * wstride);
    }
    for (int b = 0; b < dv / 64; ++b) {
      la_conv_taps(a, (vOff >> 6) + b, H->wtmp, H->Wc + (2 * (dk / 64) + b) * wstride);
    }
  }
}

static void la_hmx_head_end(const la_args_t *a, la_hmx_head_t *H) {
  const int dk = a->p->headKDim, dv = a->p->headVDim;
  float    *S_ddr = a->rnn_state + (size_t) H->h * dk * dv;
  for (int i = 0; i < dk * dv; i += 32) {
    vmem(S_ddr + i) = vmem(H->S + i);
  }
}

typedef struct {
  const la_args_t    *a;
  la_hmx_head_t      *heads;
  int                 nheads, phase, chunk, nchunks, first, step;
  worker_synctoken_t *sync;
} la_hmx_job_t;

static void la_hmx_worker(void *data, int worker_index) {
  (void) worker_index;
  const la_hmx_job_t *j = (const la_hmx_job_t *) data;
  for (int i = j->first; i < j->nheads; i += j->step) {
    la_hmx_head_t *H = &j->heads[i];
    if (j->phase == LA_HMX_PH_INIT) {
      la_hmx_head_begin(j->a, H);
      la_hmx_phase_a(j->a, H, 0);
    } else if (j->phase == LA_HMX_PH_B) {
      la_hmx_phase_b(j->a, H);
    } else {
      la_hmx_phase_c(j->a, H);
      if (j->chunk + 1 < j->nchunks) {
        la_hmx_phase_a(j->a, H, j->chunk + 1);
      } else {
        la_hmx_head_end(j->a, H);
      }
    }
  }
  worker_pool_synctoken_jobdone(j->sync);
}

static void la_hmx_run(const la_args_t *a, la_hmx_head_t *heads, int nheads, int phase, int chunk, int nchunks) {
  int nw = (int) g_max_num_workers;
  if (nw < 1) {
    nw = 1;
  }
  if (nw > nheads) {
    nw = nheads;
  }
  if (nw > DSP_LINEAR_ATTN_MAX_WORKERS) {
    nw = DSP_LINEAR_ATTN_MAX_WORKERS;
  }
  worker_synctoken_t sync;
  worker_pool_synctoken_init(&sync, nw);
  la_hmx_job_t jobs[DSP_LINEAR_ATTN_MAX_WORKERS];
  for (int w = 0; w < nw; ++w) {
    jobs[w] = (la_hmx_job_t) { a, heads, nheads, phase, chunk, nchunks, w, nw, &sync };
    worker_pool_job_t job;
    job.fptr = la_hmx_worker;
    job.dptr = &jobs[w];
    if (worker_pool_submit(NULL, job) != 0) {
      la_hmx_worker(&jobs[w], 0);
    }
  }
  worker_pool_synctoken_wait(&sync);
}

// Make the products of a round visible before HVX touches them: an HVX load of an HMX output waits for
// that write (16 ns measured), and HMX completes in order, so one read per tile is enough.
static void la_hmx_drain(const __fp16 *tiles, int ntiles) {
  HVX_Vector sink = Q6_V_vzero();
  for (int t = 0; t < ntiles; ++t) {
    sink = Q6_V_vor_VV(sink, vmem(tiles + (size_t) t * HMX_FP16_TILE_N_ELMS));
  }
  __asm__ __volatile__("" : : "v"(sink) : "memory");
}

static void la_hmx_round1(la_hmx_head_t *H, int nkb, int nvb) {
  hmx_load_tiles_fp16(H->tKn, H->tKnt, (size_t) nkb);
  hmx_consume_accumulator_fp16(H->oT);
  hmx_load_tiles_fp16(H->tQn, H->tKnt, (size_t) nkb);
  hmx_consume_accumulator_fp16(H->oQK);
  for (int cb = 0; cb < nvb; ++cb) {
    const __fp16 *ws = H->tS + (size_t) cb * nkb * HMX_FP16_TILE_N_ELMS;
    hmx_load_tiles_fp16(H->tKn, ws, (size_t) nkb);
    hmx_consume_accumulator_fp16(H->oKnS + (size_t) cb * HMX_FP16_TILE_N_ELMS);
    hmx_load_tiles_fp16(H->tQn, ws, (size_t) nkb);
    hmx_consume_accumulator_fp16(H->oQnS + (size_t) cb * HMX_FP16_TILE_N_ELMS);
  }
}

static void la_hmx_round2(la_hmx_head_t *H, int nkb, int nvb) {
  for (int cb = 0; cb < nvb; ++cb) {
    hmx_load_tiles_fp16(H->tT, H->tZ + (size_t) cb * HMX_FP16_TILE_N_ELMS, 1);
    hmx_consume_accumulator_fp16(H->oVN + (size_t) cb * HMX_FP16_TILE_N_ELMS);
  }
  // VN is the W operand of both products below; its tile layout is already the W layout.
  la_hmx_drain(H->oVN, nvb);
  for (int cb = 0; cb < nvb; ++cb) {
    const __fp16 *vn = H->oVN + (size_t) cb * HMX_FP16_TILE_N_ELMS;
    hmx_load_tiles_fp16(H->tQK, vn, 1);
    hmx_consume_accumulator_fp16(H->oO2 + (size_t) cb * HMX_FP16_TILE_N_ELMS);
    for (int rb = 0; rb < nkb; ++rb) {
      hmx_load_tiles_fp16(H->tKwT + (size_t) rb * HMX_FP16_TILE_N_ELMS, vn, 1);
      hmx_consume_accumulator_fp16(H->oSu + (size_t) (rb * nvb + cb) * HMX_FP16_TILE_N_ELMS);
    }
  }
}

static int la_hmx_supported(const la_args_t *a) {
  const DSPLinearAttnParam *p = a->p;
  if (p->type != DSP_LINEAR_ATTN_GATED_DELTA || p->seqLen <= 1 || !p->useL2Norm) {
    return 0;
  }
  if (p->headKDim % 64 || p->headVDim % 64 || p->headKDim > 32 * LA_HMX_MAX_BLK || p->headVDim > 32 * LA_HMX_MAX_BLK) {
    return 0;
  }
  const size_t reserve = 256 * 1024 + 4096;  // reserved areas at the top of VTCM, and the output scales
  const size_t total   = (size_t) vtcm_manager_get_vtcm_size();
  return total > reserve && (total - reserve) >= la_hmx_head_vtcm_bytes(p->headKDim, p->headVDim);
}

static void la_hmx_prefill(const la_args_t *a) {
  const DSPLinearAttnParam *p  = a->p;
  const int                 dk = p->headKDim, dv = p->headVDim, Hv = p->numVHeads, nkb = dk / 32, nvb = dv / 32;
  const int                 nchunks  = (p->seqLen + LA_CHUNK - 1) / LA_CHUNK;
  const size_t              per_head = la_hmx_head_vtcm_bytes(dk, dv);
  uint8_t                  *base     = (uint8_t *) vtcm_manager_get_vtcm_base();
  // vtcm_seq_alloc-style bump from the base; the reserved areas live at the top, so stay 256 KB clear.
  const size_t              avail    = (size_t) vtcm_manager_get_vtcm_size() - 256 * 1024 - 4096;
  int                       batch    = (int) (avail / per_head);
  if (batch > Hv) {
    batch = Hv;
  }
  if (batch > DSP_LINEAR_ATTN_MAX_WORKERS) {
    batch = DSP_LINEAR_ATTN_MAX_WORKERS;  // one scratch slot per head in the batch
  }
  uint8_t      *scales = base;
  uint8_t      *heads0 = base + 4096;
  la_hmx_head_t heads[DSP_LINEAR_ATTN_MAX_WORKERS];

  hmx_manager_enable_execution();
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_unit_acquire();
  hmx_set_output_scales(scales);
  for (int hb = 0; hb < Hv; hb += batch) {
    const int nb = Hv - hb < batch ? Hv - hb : batch;
    for (int i = 0; i < nb; ++i) {
      la_hmx_head_init(&heads[i], a, hb + i, i, heads0 + (size_t) i * per_head);
    }
    la_hmx_run(a, heads, nb, LA_HMX_PH_INIT, 0, nchunks);
    for (int c = 0; c < nchunks; ++c) {
      for (int i = 0; i < nb; ++i) {
        la_hmx_round1(&heads[i], nkb, nvb);
      }
      for (int i = 0; i < nb; ++i) {
        la_hmx_drain(heads[i].oT, 2 + 2 * nvb);  // oT, oQK, oKnS, oQnS are contiguous
      }
      la_hmx_run(a, heads, nb, LA_HMX_PH_B, c, nchunks);
      for (int i = 0; i < nb; ++i) {
        la_hmx_round2(&heads[i], nkb, nvb);
      }
      for (int i = 0; i < nb; ++i) {
        la_hmx_drain(heads[i].oVN, 2 * nvb + nkb * nvb);  // oVN, oO2, oSu are contiguous
      }
      la_hmx_run(a, heads, nb, LA_HMX_PH_CA, c, nchunks);
    }
  }
  hmx_unit_release();
  hmx_manager_disable_execution();
}

// Decode with one value head per key head: each worker runs the conv of its own heads, and their states move
// through three VTCM buffers by this thread's DMA, so the write-back of one head and the fetch of the next overlap
// the conv and the recurrence. The DMA goes through L2 (no bypass): prefill and the restart zeroing write the
// state through the cache.
static void la_conv_head(const la_args_t *a, int h, float *frow, HVX_Vector *taps, __fp16 *wtmp) {
  const DSPLinearAttnParam *p  = a->p;
  const int                 dk = p->headKDim, dv = p->headVDim, Hk = p->numKHeads;
  const int                 qb = h * dk / 64, kb = (Hk * dk + h * dk) / 64, vb = (2 * Hk * dk + h * dv) / 64;
  for (int b = 0; b < dk / 64; ++b) {
    la_conv_taps(a, qb + b, wtmp, taps);
    la_conv_rows(a, qb + b, 0, 1, frow + b * 64, 0, taps);
    la_conv_taps(a, kb + b, wtmp, taps);
    la_conv_rows(a, kb + b, 0, 1, frow + dk + b * 64, 0, taps);
  }
  for (int b = 0; b < dv / 64; ++b) {
    la_conv_taps(a, vb + b, wtmp, taps);
    la_conv_rows(a, vb + b, 0, 1, frow + 2 * dk + b * 64, 0, taps);
  }
  LA_FENCE();
}

static void la_dma_desc(dma_desc_1d_t *d, const void *src, void *dst, size_t bytes, dma_desc_1d_t *next) {
  memset(d, 0, sizeof(*d));
  d->next    = (uint32_t) next;
  d->length  = (uint32_t) bytes;
  d->type    = DMA_DESC_TYPE_1D;
  d->ordered = 1;
  d->dstate  = DMA_DESC_DSTATE_PENDING;
  d->src     = (uint32_t) src;
  d->dst     = (uint32_t) dst;
}

static void la_decode_fused(const la_task_t *t, HVX_Vector *scr, float *vt) {
  const la_args_t          *a  = t->a;
  const DSPLinearAttnParam *p  = a->p;
  const int                 dk = p->headKDim, dv = p->headVDim;
  const size_t              sfl    = (size_t) dk * dv;
  float                    *buf[3] = { vt, vt + sfl, vt + 2 * sfl };
  float                    *frow   = (float *) (scr + LA_SCRATCH_VECS);  // the chunk scalar scratch is free in decode
  HVX_Vector               *taps   = (HVX_Vector *) (frow + 2 * dk + dv);
  __fp16                   *wtmp   = (__fp16 *) (taps + 2 * LA_MAX_K);
  dma_desc_1d_t            *desc   = (dma_desc_1d_t *) (wtmp + 64);  // [out, in], 64-byte aligned
  if (t->first >= t->count) {
    return;
  }
  la_dma_desc(&desc[1], a->rnn_state + (size_t) t->first * sfl, buf[0], sfl * sizeof(float), NULL);
  dmstart(&desc[1]);
  la_conv_head(a, t->first, frow, taps, wtmp);
  dma_wait_for_idle();
  int idx = 0;
  for (int h = t->first; h < t->count; h += t->step, ++idx) {
    const int      b = idx % 3, bn = (idx + 1) % 3, bp = (idx + 2) % 3;
    const int      prev = h - t->step, next = h + t->step;
    dma_desc_1d_t *head = NULL;
    if (next < t->count) {
      la_dma_desc(&desc[1], a->rnn_state + (size_t) next * sfl, buf[bn], sfl * sizeof(float), NULL);
      head = &desc[1];
    }
    if (idx > 0) {
      la_dma_desc(&desc[0], buf[bp], a->rnn_state + (size_t) prev * sfl, sfl * sizeof(float), head);
      head = &desc[0];
    }
    if (head) {
      dmstart(head);
    }
    if (idx > 0) {
      la_conv_head(a, h, frow, taps, wtmp);
    }
    la_delta_head(a, h, scr, buf[b], frow, 1);
    dma_wait_for_idle();
    if (next >= t->count) {
      la_dma_desc(&desc[0], buf[b], a->rnn_state + (size_t) h * sfl, sfl * sizeof(float), NULL);
      dmstart(&desc[0]);
      dma_wait_for_idle();
    }
  }
}

static void la_delta_worker(void *data, int worker_index) {
  (void) worker_index;
  const la_task_t *t   = (const la_task_t *) data;
  HVX_Vector      *scr = (HVX_Vector *) ((uint8_t *) t->a->scratch + (size_t) t->first * DSP_LINEAR_ATTN_SCRATCH_BYTES);
  la_build_bcast(scr);
  const DSPLinearAttnParam *p           = t->a->p;
  const size_t              chunk_bytes = la_chunk_vtcm_floats(p->headKDim, p->headVDim) * sizeof(float);
  if (t->a->fuse_conv || (p->seqLen > 1 && (p->headKDim % 4) == 0 &&
                          (size_t) (t->first + 1) * chunk_bytes <= vtcm_manager_get_vtcm_size())) {
    float *vt = (float *) ((uint8_t *) vtcm_manager_get_vtcm_base() + (size_t) t->first * chunk_bytes);
    float *sc = (float *) (scr + LA_SCRATCH_VECS);
    for (int h = t->first; h < t->count; h += t->step) {
      la_delta_head_chunked(t->a, h, scr, vt, sc, t->first);
    }
    worker_pool_synctoken_jobdone(t->sync);
    return;
  }
  const size_t state_bytes = (size_t) p->headKDim * p->headVDim * sizeof(float);
  float       *vtcm_state  = NULL;
  if ((size_t) (t->first + 1) * state_bytes <= vtcm_manager_get_vtcm_size()) {
    vtcm_state = (float *) ((uint8_t *) vtcm_manager_get_vtcm_base() + (size_t) t->first * state_bytes);
  }
  if (t->a->fuse_decode) {
    la_decode_fused(t, scr, (float *) ((uint8_t *) vtcm_manager_get_vtcm_base() + (size_t) t->first * 3 * state_bytes));
    worker_pool_synctoken_jobdone(t->sync);
    return;
  }
  for (int h = t->first; h < t->count; h += t->step) {
    la_delta_head(t->a, h, scr, vtcm_state, NULL, 0);
  }
  worker_pool_synctoken_jobdone(t->sync);
}

static int la_num_workers(int count) {
  int nw = (int) g_max_num_workers;
  if (nw < 1) {
    nw = 1;
  }
  if (nw > count) {
    nw = count;
  }
  if (nw > DSP_LINEAR_ATTN_MAX_WORKERS) {
    nw = DSP_LINEAR_ATTN_MAX_WORKERS;
  }
  return nw;
}

static void la_run_parallel(const la_args_t *a, int count, void (*fn)(void *, int)) {
  const int          nw = la_num_workers(count);
  worker_synctoken_t sync;
  worker_pool_synctoken_init(&sync, nw);
  la_task_t tasks[DSP_LINEAR_ATTN_MAX_WORKERS];
  for (int w = 0; w < nw; ++w) {
    tasks[w] = (la_task_t) { a, w, nw, count, &sync };
    worker_pool_job_t job;
    job.fptr = fn;
    job.dptr = &tasks[w];
    if (worker_pool_submit(NULL, job) != 0) {
      fn(&tasks[w], 0);
    }
  }
  worker_pool_synctoken_wait(&sync);
}

// HTP_LA_HMX = 0 builds the HVX chunked path only, for A/B against it.
#ifndef HTP_LA_HMX
#  define HTP_LA_HMX 1
#endif
// Self-check (diagnostic, off): run the HVX chunked path as ground truth, then the HMX path from the same
// state, and record the worst relative error of the output and of the final state over all calls:
//   [229] max|d out| / max|out| * 1e6    [230] max|d S| / max|S| * 1e6
//   [231] max|d out| * 1e6               [232] calls compared
#ifndef HTP_LA_HMX_SELFCHECK
#  define HTP_LA_HMX_SELFCHECK 0
#endif
#if HTP_LA_HMX_SELFCHECK
#  include <stdlib.h>
unsigned long long g_la_sc[4];

static void la_hmx_selfcheck(const la_args_t *a) {
  const DSPLinearAttnParam *p      = a->p;
  const size_t              outN   = (size_t) p->seqLen * p->numVHeads * p->headVDim;
  const size_t              sN     = (size_t) p->numVHeads * p->headKDim * p->headVDim;
  const size_t              convN  = (size_t) (p->convDim / 64) * (p->kernelSize - 1) * 64;
  __fp16                   *refOut = (__fp16 *) memalign(128, outN * sizeof(__fp16));
  float                    *refS   = (float *) memalign(128, sN * sizeof(float));
  float                    *s0     = (float *) memalign(128, sN * sizeof(float));
  float                    *cv0    = (float *) memalign(128, convN * sizeof(float) + 128);
  if (!refOut || !refS || !s0 || !cv0) {
    free(refOut);
    free(refS);
    free(s0);
    free(cv0);
    la_hmx_prefill(a);
    return;
  }
  memcpy(s0, a->rnn_state, sN * sizeof(float));
  memcpy(cv0, a->conv_state, convN * sizeof(float));
  la_run_parallel(a, p->numVHeads, la_delta_worker);
  memcpy(refOut, a->out, outN * sizeof(__fp16));
  memcpy(refS, a->rnn_state, sN * sizeof(float));
  memcpy(a->rnn_state, s0, sN * sizeof(float));
  memcpy(a->conv_state, cv0, convN * sizeof(float));
  la_hmx_prefill(a);
  float dOut = 0, mOut = 0, dS = 0, mS = 0;
  for (size_t i = 0; i < outN; ++i) {
    const float r = (float) refOut[i], d = fabsf((float) a->out[i] - r);
    dOut = d > dOut ? d : dOut;
    mOut = fabsf(r) > mOut ? fabsf(r) : mOut;
  }
  for (size_t i = 0; i < sN; ++i) {
    const float r = refS[i], d = fabsf(a->rnn_state[i] - r);
    dS = d > dS ? d : dS;
    mS = fabsf(r) > mS ? fabsf(r) : mS;
  }
  const unsigned long long e0 = (unsigned long long) (mOut > 0 ? dOut / mOut * 1e6f : 0);
  const unsigned long long e1 = (unsigned long long) (mS > 0 ? dS / mS * 1e6f : 0);
  const unsigned long long e2 = (unsigned long long) (dOut * 1e6f);
  g_la_sc[0]                  = e0 > g_la_sc[0] ? e0 : g_la_sc[0];
  g_la_sc[1]                  = e1 > g_la_sc[1] ? e1 : g_la_sc[1];
  g_la_sc[2]                  = e2 > g_la_sc[2] ? e2 : g_la_sc[2];
  g_la_sc[3] += 1;
  free(refOut);
  free(refS);
  free(s0);
  free(cv0);
}
#endif

// Unit costs for redesigning the chunked prefill around HMX (diagnostic, off). Measures, on the calling
// thread after the op's own work:
//   [229] us for LA_MB_HMX_ITERS small products of 1 k-tile (load + consume), the cost of a C x C
//         triangular product or a rank-C state update per output tile
//   [230] the same with 4 k-tiles (a 128-deep contraction: rows x state, Gram)
//   [231] us for LA_MB_DISPATCH_ITERS worker dispatches of one empty job per worker, i.e. the price of
//         one HVX phase barrier in a phase-by-phase schedule
//   [232] number of op calls measured, to divide by
//   [233] us for LA_MB_HMX_ITERS products whose output HVX reads right away -- the HMX-to-HVX drain a
//         phase boundary pays when one head's result is consumed immediately
//   [234] us for LA_MB_HMX_ITERS / 16 rounds of 16 products into 16 tiles followed by one HVX read of
//         all of them, i.e. the same drain amortised over a batch
// The HMX operands are finite fp16 ones: denormal or garbage input slows the multiply down and would
// make the number meaningless (see the HMX split experiment in matmul_q4fp16.c).
#ifndef HTP_LA_MICROBENCH
#  define HTP_LA_MICROBENCH 0
#endif
#if HTP_LA_MICROBENCH
#  include "HAP_perf.h"
#  define LA_MB_HMX_ITERS      4096
#  define LA_MB_DISPATCH_ITERS 256
unsigned long long g_la_mb_us[6];

static void la_mb_noop_worker(void *data, int worker_index) {
  (void) worker_index;
  worker_pool_synctoken_jobdone((worker_synctoken_t *) data);
}

static void la_microbench(void) {
  uint8_t         *base   = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16          *act    = (__fp16 *) base;                  // 32 tiles
  __fp16          *wgt    = act + 32 * HMX_FP16_TILE_N_ELMS;  // 32 tiles
  __fp16          *out    = wgt + 32 * HMX_FP16_TILE_N_ELMS;  // 16 tiles
  uint8_t         *scales = (uint8_t *) (out + 16 * HMX_FP16_TILE_N_ELMS);
  const HVX_Vector one    = Q6_Vh_vsplat_R(0x3C00);
  for (int i = 0; i < 64 * HMX_FP16_TILE_N_ELMS; i += 64) {
    vmem(act + i) = one;
  }
#  define LA_MB_A(it) (act + (size_t) (((it) * 4) & 31) * HMX_FP16_TILE_N_ELMS)
#  define LA_MB_W(it) (wgt + (size_t) (((it) * 4 + 7) & 31) * HMX_FP16_TILE_N_ELMS)
#  define LA_MB_O(it) (out + (size_t) ((it) & 15) * HMX_FP16_TILE_N_ELMS)
  hmx_manager_enable_execution();
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_unit_acquire();
  hmx_set_output_scales(scales);
  for (int kt = 1; kt <= 4; kt += 3) {
    const unsigned long long t0 = HAP_perf_get_time_us();
    for (int it = 0; it < LA_MB_HMX_ITERS; ++it) {
      hmx_load_tiles_fp16(LA_MB_A(it), LA_MB_W(it), (size_t) kt);
      hmx_consume_accumulator_fp16(LA_MB_O(it));
    }
    g_la_mb_us[kt == 1 ? 0 : 1] += HAP_perf_get_time_us() - t0;
  }
  {
    HVX_Vector               sink = Q6_V_vzero();
    const unsigned long long t0   = HAP_perf_get_time_us();
    for (int it = 0; it < LA_MB_HMX_ITERS; ++it) {
      hmx_load_tiles_fp16(LA_MB_A(it), LA_MB_W(it), 1);
      hmx_consume_accumulator_fp16(LA_MB_O(it));
      sink = Q6_Vqf16_vadd_VhfVhf(sink, vmem(LA_MB_O(it)));
    }
    g_la_mb_us[4] += HAP_perf_get_time_us() - t0;
    vmem(out) = sink;
  }
  {
    __fp16                  *outs = out;
    HVX_Vector               sink = Q6_V_vzero();
    const unsigned long long t0   = HAP_perf_get_time_us();
    for (int it = 0; it < LA_MB_HMX_ITERS / 16; ++it) {
      for (int o = 0; o < 16; ++o) {
        hmx_load_tiles_fp16(LA_MB_A(it * 16 + o), LA_MB_W(it * 16 + o), 1);
        hmx_consume_accumulator_fp16(outs + (size_t) o * HMX_FP16_TILE_N_ELMS);
      }
      for (int o = 0; o < 16; ++o) {
        sink = Q6_Vqf16_vadd_VhfVhf(sink, vmem(outs + (size_t) o * HMX_FP16_TILE_N_ELMS));
      }
    }
    g_la_mb_us[5] += HAP_perf_get_time_us() - t0;
    vmem(out) = sink;
  }
  hmx_unit_release();
  hmx_manager_disable_execution();

  const int                nw = (int) g_max_num_workers > 0 ? (int) g_max_num_workers : 1;
  const unsigned long long t1 = HAP_perf_get_time_us();
  for (int it = 0; it < LA_MB_DISPATCH_ITERS; ++it) {
    worker_synctoken_t sync;
    worker_pool_synctoken_init(&sync, nw);
    for (int w = 0; w < nw; ++w) {
      worker_pool_job_t job;
      job.fptr = la_mb_noop_worker;
      job.dptr = &sync;
      if (worker_pool_submit(NULL, job) != 0) {
        la_mb_noop_worker(&sync, 0);
      }
    }
    worker_pool_synctoken_wait(&sync);
  }
  g_la_mb_us[2] += HAP_perf_get_time_us() - t1;
  g_la_mb_us[3] += 1;
}
#endif

AEEResult htp_ops_linear_attention(const DSPLinearAttnParam *p, const uint8_t *qkv, const uint8_t *in1,
                                   const uint8_t *in2, const uint8_t *conv_w, const uint8_t *gate_const, uint8_t *out,
                                   uint8_t *conv_state, uint8_t *rnn_state, uint8_t *conv_out) {
  if (p == NULL || qkv == NULL || conv_w == NULL || out == NULL || conv_state == NULL || p->seqLen <= 0 ||
      p->kernelSize < 1 || p->kernelSize > LA_MAX_K || (p->convDim % 64) != 0) {
    return AEE_EBADPARM;
  }
  const bool gdr = p->type == DSP_LINEAR_ATTN_GATED_DELTA;
  if (gdr) {
    if (in1 == NULL || in2 == NULL || rnn_state == NULL || conv_out == NULL || (p->gateFold && gate_const == NULL) ||
        p->numKHeads <= 0 || p->numVHeads % p->numKHeads != 0 || (p->headKDim % 64) != 0 || (p->headVDim % 64) != 0 ||
        p->headKDim > 32 * LA_MAX_VEC || p->headVDim > 32 * LA_MAX_VEC ||
        p->convDim != 2 * p->numKHeads * p->headKDim + p->numVHeads * p->headVDim) {
      return AEE_EBADPARM;
    }
  } else if (p->type != DSP_LINEAR_ATTN_SHORT_CONV || (p->convDim % 192) != 0) {
    return AEE_EBADPARM;
  }
  la_args_t a = { p,
                  (const __fp16 *) qkv,
                  (const __fp16 *) in1,
                  (const __fp16 *) in2,
                  (const __fp16 *) conv_w,
                  (const float *) gate_const,
                  (__fp16 *) out,
                  (float *) conv_state,
                  (float *) rnn_state,
                  (float *) conv_out,
                  gdr ? (HVX_Vector *) ((float *) conv_out + (size_t) p->seqLen * p->convDim) : NULL,
                  0,
                  0 };
  // With one value head per key head every conv channel block belongs to exactly one head, so the chunked
  // prefill can run the conv itself and the fp32 conv output never goes through DDR.
  a.fuse_conv =
    gdr && p->seqLen > 1 && p->numKHeads == p->numVHeads && (p->headKDim % 4) == 0 &&
    (size_t) la_num_workers(p->numVHeads) * la_chunk_vtcm_floats(p->headKDim, p->headVDim) * sizeof(float) <=
      vtcm_manager_get_vtcm_size();
  a.fuse_decode = gdr && p->seqLen == 1 && p->numKHeads == p->numVHeads &&
                  (size_t) la_num_workers(p->numVHeads) * 3 * p->headKDim * p->headVDim * sizeof(float) <=
                    vtcm_manager_get_vtcm_size();
  if (!a.fuse_conv && !a.fuse_decode) {
    la_run_parallel(&a, gdr ? p->convDim / 64 : p->convDim / 192, la_conv_worker);
  }
  if (gdr) {
#if HTP_LA_HMX
    if (la_hmx_supported(&a)) {
#  if HTP_LA_HMX_SELFCHECK
      la_hmx_selfcheck(&a);
#  else
      la_hmx_prefill(&a);
#  endif
    } else
#endif
    {
      la_run_parallel(&a, p->numVHeads, la_delta_worker);
    }
  }
#if HTP_LA_MICROBENCH
  if (gdr && p->seqLen > 1) {
    la_microbench();
  }
#endif
  return AEE_SUCCESS;
}

// ---- GatedRMSNorm: out = (x * rsqrt(mean(x^2) + eps) * gamma + beta) * silu(z) ----
// The gated output norm that follows every linear-attention layer. One pass replaces the LayerNorm + raster
// repack + SILU + MUL decomposition: x row h = b * heads + head is read from [outside][inside], z and out are
// addressed in [batch][heads * inside], so the repack is folded into the indexing. The norm is fp32 between
// the fp16 load and store; silu(z) is the fp16 PWL the standalone SILU uses.
//
// Work unit = one head over LA_GRN_BC consecutive tokens. For a fixed (head, channel block) z / out are one
// contiguous run over the tokens and x is a 2D run with a heads * 128 B stride, so each of the unit's 2 * nb
// streams is a single l2fetch. A thread keeps one l2fetch in flight (a new one cancels the old), so they are
// staged one segment ahead: z of the next channel block while this one is stored, x of the next unit during
// the second half of each store pass. Without this the kernel is DDR-latency bound (5.6 GB/s).
typedef struct {
  const __fp16      *x, *z;
  const float       *gamma, *beta;
  __fp16            *out;
  int                batch, heads, nb, outside, chunks, units, nw;
  float              epsilon, invInside;
  worker_synctoken_t sync;
} la_grn_args_t;

// The pack is shared with the host geometry gate: see htp_gated_rms_norm_params.h.
#define LA_GRN_PACK HTP_GATED_RMS_NORM_PACK
#define LA_GRN_BC   64

static inline const __fp16 *la_grn_x(const la_grn_args_t *g, int head, int b, int cb) {
  return g->x + ((size_t) cb * g->outside + (size_t) b * g->heads + head) * LA_GRN_PACK;
}

static inline size_t la_grn_zo(const la_grn_args_t *g, int head, int b, int cb) {
  return ((size_t) (head * g->nb + cb) * g->batch + b) * LA_GRN_PACK;
}

static inline void la_grn_fetch_x(const la_grn_args_t *g, int u, int cb) {
  if (u < g->units) {
    const int head = u / g->chunks, b0 = (u % g->chunks) * LA_GRN_BC;
    const int n = g->batch - b0 < LA_GRN_BC ? g->batch - b0 : LA_GRN_BC;
    l2fetch(la_grn_x(g, head, b0, cb), g->heads * LA_GRN_PACK * 2, LA_GRN_PACK * 2, n, 0);
  }
}

static inline void la_grn_fetch_z(const la_grn_args_t *g, int u, int cb) {
  const int head = u / g->chunks, b0 = (u % g->chunks) * LA_GRN_BC;
  const int n = g->batch - b0 < LA_GRN_BC ? g->batch - b0 : LA_GRN_BC;
  l2fetch(g->z + la_grn_zo(g, head, b0, cb), LA_GRN_PACK * 2, LA_GRN_PACK * 2, n, 0);
}

static void la_grn_unit(const la_grn_args_t *g, int u, int next) {
  const int head = u / g->chunks, b0 = (u % g->chunks) * LA_GRN_BC;
  const int n  = g->batch - b0 < LA_GRN_BC ? g->batch - b0 : LA_GRN_BC;
  const int nb = g->nb;
  float     inv[LA_GRN_BC];
  la_grn_fetch_z(g, u, 0);
  for (int i = 0; i < n; ++i) {
    HVX_Vector sq = la_zero();
    for (int cb = 0; cb < nb; ++cb) {
      const HVX_VectorPair xs = Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *) la_grn_x(g, head, b0 + i, cb));
      sq                      = Q6_Vsf_vadd_VsfVsf(sq, Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(xs), Q6_V_lo_W(xs)));
      sq                      = Q6_Vsf_vadd_VsfVsf(sq, Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(xs), Q6_V_hi_W(xs)));
    }

    union {
      HVX_Vector v;
      float      f[32];
    } r;

    r.v    = la_hsum(sq);
    inv[i] = 1.0f / __builtin_sqrtf(r.f[0] * g->invInside + g->epsilon);
  }
  for (int cb = 0; cb < nb; ++cb) {
    if (cb + 1 < nb) {
      la_grn_fetch_z(g, u, cb + 1);
    }
    HVX_Vector g0 = la_splat(1.0f), g1 = g0, e0 = la_zero(), e1 = e0;
    if (g->gamma) {
      const HVX_Vector    *pg = (const HVX_Vector *) (g->gamma + cb * LA_GRN_PACK);
      const HVX_VectorPair gd = Q6_W_vdeal_VVR(pg[1], pg[0], -4);
      g0                      = Q6_V_lo_W(gd);
      g1                      = Q6_V_hi_W(gd);
    }
    if (g->beta) {
      const HVX_Vector    *pb = (const HVX_Vector *) (g->beta + cb * LA_GRN_PACK);
      const HVX_VectorPair bd = Q6_W_vdeal_VVR(pb[1], pb[0], -4);
      e0                      = Q6_V_lo_W(bd);
      e1                      = Q6_V_hi_W(bd);
    }
    const size_t zo = la_grn_zo(g, head, b0, cb);
    for (int i = 0; i < n; ++i) {
      if (i == n / 2) {
        la_grn_fetch_x(g, next, cb);
      }
      const HVX_Vector     vi = la_splat(inv[i]);
      const HVX_VectorPair xs = Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *) la_grn_x(g, head, b0 + i, cb));
      // silu(z) in fp16 with the same PWL table the SILU / MUL_SILU kernels use, then widened.
      const HVX_VectorPair zs =
        Q6_Wsf_vcvt_Vhf(htp_ops_silu_pwl_fp16_vec(*(const HVX_Vector *) (g->z + zo + (size_t) i * LA_GRN_PACK)));
      HVX_Vector o0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(xs), vi), g0), e0);
      HVX_Vector o1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(xs), vi), g1), e1);
      o0            = Q6_Vsf_vmpy_VsfVsf(o0, Q6_V_lo_W(zs));
      o1            = Q6_Vsf_vmpy_VsfVsf(o1, Q6_V_hi_W(zs));
      *(HVX_Vector *) (g->out + zo + (size_t) i * LA_GRN_PACK) = Q6_Vhf_vcvt_VsfVsf(o0, o1);
    }
  }
}

typedef struct {
  const la_grn_args_t *g;
  int                  first;
} la_grn_task_t;

static void la_grn_worker(void *data, int worker_index) {
  (void) worker_index;
  const la_grn_task_t *t = (const la_grn_task_t *) data;
  const la_grn_args_t *g = t->g;
  la_grn_fetch_x(g, t->first, 0);
  for (int u = t->first; u < g->units; u += g->nw) {
    la_grn_unit(g, u, u + g->nw);
  }
  worker_pool_synctoken_jobdone((worker_synctoken_t *) &g->sync);
}

AEEResult htp_ops_gated_rms_norm(const DSPGatedRMSNormParam *p, const uint8_t *x, const uint8_t *z,
                                 const uint8_t *gamma, const uint8_t *beta, uint8_t *out) {
  if (p->batch <= 0 || p->heads <= 0 || p->inside <= 0 || (p->inside % LA_GRN_PACK) != 0 || !x || !z || !out) {
    return AEE_EBADPARM;
  }
  la_grn_args_t g;
  memset(&g, 0, sizeof(g));
  g.x                = (const __fp16 *) x;
  g.z                = (const __fp16 *) z;
  g.gamma            = (const float *) gamma;
  g.beta             = (const float *) beta;
  g.out              = (__fp16 *) out;
  g.batch            = p->batch;
  g.heads            = p->heads;
  g.nb               = p->inside / LA_GRN_PACK;
  g.outside          = p->batch * p->heads;
  g.chunks           = (p->batch + LA_GRN_BC - 1) / LA_GRN_BC;
  g.units            = g.chunks * p->heads;
  g.epsilon          = p->epsilon;
  g.invInside        = 1.0f / (float) p->inside;
  // Decode-sized calls (a few KB) finish before a worker dispatch would.
  const size_t bytes = (size_t) g.outside * p->inside * sizeof(__fp16);
  int          nw    = (int) g_max_num_workers < g.units ? (int) g_max_num_workers : g.units;
  if (bytes < 32768) {
    nw = 1;
  }
  g.nw = nw > 0 ? nw : 1;
  la_grn_task_t tasks[DSP_LINEAR_ATTN_MAX_WORKERS];
  if (g.nw > DSP_LINEAR_ATTN_MAX_WORKERS) {
    g.nw = DSP_LINEAR_ATTN_MAX_WORKERS;
  }
  if (g.nw == 1) {
    la_grn_fetch_x(&g, 0, 0);
    for (int u = 0; u < g.units; ++u) {
      la_grn_unit(&g, u, u + 1);
    }
    return AEE_SUCCESS;
  }
  worker_pool_synctoken_init(&g.sync, g.nw);
  for (int w = 0; w < g.nw; ++w) {
    tasks[w] = (la_grn_task_t) { &g, w };
    worker_pool_job_t job;
    job.fptr = la_grn_worker;
    job.dptr = &tasks[w];
    if (worker_pool_submit(NULL, job) != 0) {
      la_grn_worker(&tasks[w], 0);
    }
  }
  worker_pool_synctoken_wait(&g.sync);
  return AEE_SUCCESS;
}

}  // extern "C"
