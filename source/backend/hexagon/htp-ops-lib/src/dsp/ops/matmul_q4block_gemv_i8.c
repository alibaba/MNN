// Decode GEMV (M=1) for block-quant int4 weights using HVX integer vrmpy.
//
// Rationale (see hexagon_gemm_opt_strategy.md): for M=1 decode the HMX 64-row
// tile is 98% wasted and the fp16 dequant is pure overhead. This kernel keeps
// the whole matmul in the integer domain: activation is dynamically quantized
// to symmetric int8 (per-token), weights are symmetric int4 ([-8,7]), and the
// dot product is done with Q6_Vw_vrmpyacc_VwVbVb (int8xint8 -> int32). The
// combined activation*weight scale is applied once per block in fp32. No HMX,
// no fp16 dequant.
//
// ---------------------------------------------------------------------------
// WEIGHT LAYOUT CONTRACT (produced host-side by reorderInt4WeightForVrmpyGemv)
// ---------------------------------------------------------------------------
// Dimensions: K (=ic), N (=oc). kp=K/32, np=N/32, icP=kp, ocP=np.
// Weight blob `b_wt` = icP*ocP int4 tiles of 512 bytes, tile (ocTile y, kTile x)
// at byte offset (y*icP + x)*512.
// Inside a 512B tile: 8 groups g in [0,8), each 64 bytes, group g covers the 4
// k-values k = x*32 + 4g + {0,1,2,3} for all 32 output channels of the tile.
// Byte at (g*64 + ocIn*2 + p), p in {0,1}, holds (value+8, i.e. [0,15]):
//   low  nibble = quant(oc, k = x*32 + 4g + 2p)
//   high nibble = quant(oc, k = x*32 + 4g + 2p + 1)
// On the DSP a 128B load (= 2 groups) is nibble-expanded + byte-interleaved so
// that Q6_W_vshuff gives, per group, an int8 vector [oc:[w4g,w4g+1,w4g+2,w4g+3]]
// which is exactly what vrmpy consumes (4 k-values contiguous per 32-bit lane).
//
// SCALE LAYOUT CONTRACT:
// `b_scale` = fp32, per oc-tile contiguous, entry stride E = 32 symmetric / 64 asymmetric:
//   sw(y, blk, ocIn)    at b_scale[(y*nblk + blk)*E + ocIn]
//   qbias(y, blk, ocIn) at b_scale[(y*nblk + blk)*E + 32 + ocIn]   (asymmetric only)
// nblk = scale_block_num = K/blocksize. Keeping qbias in the same entry means the one scale DMA per
// chunk still covers it. Asymmetric dequant is w = q*scale + qbias, so the dot product gains the
// term sum_b qbias[oc][b] * (sum of block b's activations) -- see build_block_act_splats.
//
// OUTPUT: fp16, linear by output channel (for M=1 the NC4HW4 pack-64 layout
// degenerates to linear), c[oc].
//
// Execution: one ordered DMA chain (scales, then weight chunks) streams into VTCM while the issuing
// thread quantizes the activation; workers take oc-tiles round-robin and start on a chunk as soon as
// its descriptor is done. The same kernel serves FusedLinear decode (several members per launch).

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "dsp/dma_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_utils.h"
#include "dsp/ops.h"
#include "dsp/pwl.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

// -------- helpers ----------------------------------------------------------

// fp16 -> signed int16, round-to-nearest. On v81+ the native Q6_Vh_vcvt_Vhf maps
// to a single instruction; on v79 and earlier the compiler lowers it to a runtime
// helper (__qf_convert_hf_to_h_rne) that is not present in the DSP skel, so use
// Q6_Vh_equals_Vhf there (available on all HVX arches; same fp16->int16 convert).
static inline HVX_Vector vhf_to_h_round(HVX_Vector s) {
#if defined(HTP_OPS_SKEL_ARCH) && (HTP_OPS_SKEL_ARCH >= 0x81)
  return Q6_Vh_vcvt_Vhf(s);
#else
  return Q6_Vh_equals_Vhf(s);
#endif
}

// Nibble-expand 128 packed int4 bytes into two signed-int8 vrmpy weight vectors
// (one per 4-k group). Byte i (=oc*2+p): low nibble -> out[2i], high -> out[2i+1],
// each minus the symmetric 8 offset. Verified on-device against a scalar ref.
static inline HVX_VectorPair unpack_vrmpy_weight_128(HVX_Vector v_int4) {
  const HVX_Vector v_mask_lo = Q6_Vb_vsplat_R(0x0f);
  const HVX_Vector v_eight   = Q6_Vb_vsplat_R(0x08);
  HVX_Vector       v_lo      = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(v_int4, v_mask_lo), v_eight);
  HVX_Vector       v_hi      = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(v_int4, 4), v_eight);
  return Q6_W_vshuff_VVR(v_hi, v_lo, -1);  // wv.lo = group 2l, wv.hi = group 2l+1
}

// int32 -> fp32 via the magic-number trick (no direct HVX cvt intrinsic).
// Valid for |x| < 2^22. The accumulator covers one scale block: weights are [-8, 7] after the symmetric
// offset and the activation is symmetric int8, so the bound is blocksize * 127 * 8, which needs
// blocksize <= 4128. hqq's 64 / 128 are far below that, but the margin is thinner than it looks --
// blocksize 4096 would already reach 4161536 of the 4194304 budget.
static inline HVX_Vector sf_from_w(HVX_Vector v_w) {
  const HVX_Vector v_magic_bits = Q6_V_vsplat_R(0x4B400000);           // 1.5 * 2^23 = 12582912.0
  const HVX_Vector v_neg_magic  = Q6_V_vsplat_R(0xCB400000);           // -12582912.0 (as fp32)
  HVX_Vector       vbits        = Q6_Vw_vadd_VwVw(v_w, v_magic_bits);  // reinterpret as fp32 = 12582912 + x
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(vbits, v_neg_magic));
}

// Convert 32 fp32 lanes (oc order) -> 32 fp16 in the low 32 lanes (oc order).
// hvx_my_wsf_to_vhf(v,v) interleaves to [v0,v0,v1,v1,...]; vdeal deinterleaves
// so the low 32 lanes become [v0,v1,...,v31] in oc order (verified on-device).
static inline HVX_Vector sf32_to_hf_low(HVX_Vector v_sf) {
  return Q6_Vh_vdeal_Vh(hvx_my_wsf_to_vhf(v_sf, v_sf));
}

typedef struct {
  int8_t *qa;  // K int8 (per-token symmetric quantized activation)
  float   sa;  // per-token scale = absmax/127
} act_quant_t;

// Broadcast every 4-byte group of qa across a whole vector: a_splat[g] = splat(qa[4g..4g+3]).
// Done with the vdelta permute network instead of a scalar load + vsplat per group: qa was just
// written by HVX into VTCM, and reading it back with 224+ scalar loads stalled ~36 ns per group,
// which was 8 of the 15 us a small (N=128) decode GEMV cost.
// vdelta applies, per stage, out[k] = (ctrl[k] & off) ? in[k ^ off] : in[k]; because the source of
// every lane differs from k only in the bits set in ctrl[k], ctrl[k] = (k & 0x7c) ^ 4j routes byte
// 4j + (k & 3) to lane k for every stage order, i.e. replicates word j. qa is padded to 128 bytes
// by vtcm_seq_alloc, so the last (possibly half-used) chunk is safe to load whole.
// kLaneGroupBase[k] = k & 0x7c: vdelta control base for replicating one 32-bit lane.
_Alignas(128) static const uint8_t kLaneGroupBase[128] = {
#define MNN_LG4(x) (x), (x), (x), (x)
#define MNN_LG32(x)                                                                                          \
  MNN_LG4((x)), MNN_LG4((x) + 4), MNN_LG4((x) + 8), MNN_LG4((x) + 12), MNN_LG4((x) + 16), MNN_LG4((x) + 20), \
    MNN_LG4((x) + 24), MNN_LG4((x) + 28)
  MNN_LG32(0), MNN_LG32(32), MNN_LG32(64), MNN_LG32(96)
#undef MNN_LG32
#undef MNN_LG4
};

static void build_act_splats(const int8_t *qa, int n_groups, HVX_Vector *a_splat) {
  const HVX_Vector v_base = vmem(kLaneGroupBase);
  for (int g0 = 0; g0 < n_groups; g0 += 32) {
    const HVX_Vector v_qa = vmem(qa + g0 * 4);
    const int        n    = n_groups - g0 < 32 ? n_groups - g0 : 32;
    for (int j = 0; j < n; ++j) {
      const HVX_Vector v_ctrl = Q6_V_vxor_VV(v_base, Q6_Vb_vsplat_R(4 * j));
      a_splat[g0 + j]         = Q6_V_vdelta_VV(v_qa, v_ctrl);
    }
  }
}

// Per-token symmetric int8 quantization of the single activation row (K fp16).
static void quantize_activation_row(const __fp16 *a, int K, int8_t *qa_out, float *sa_out) {
  const int        nv        = K / 64;  // K is a multiple of 64 (blocksize) for supported cases
  const HVX_Vector v_absmask = Q6_Vh_vsplat_R(0x7fff);
  HVX_Vector       v_max     = Q6_V_vzero();
  for (int i = 0; i < nv; ++i) {
    HVX_Vector v = vmemu(a + i * 64);
    v_max        = Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vand_VV(v, v_absmask));
  }
  // Horizontal max: a vror tree leaves the max in every lane, so only one scalar is read back
  // (an HVX store feeding scalar loads stalls; 64 of them cost ~1 us per GEMV).
  for (int r = 64; r >= 2; r >>= 1) {
    v_max = Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vror_VR(v_max, r));
  }
  _Alignas(128) __fp16 tmp[64];
  vmem(tmp)    = v_max;
  float absmax = (float) tmp[0];
  if (absmax <= 0.0f) {
    absmax = 1.0f;
  }
  const float sa  = absmax / 127.0f;
  const float inv = 127.0f / absmax;
  *sa_out         = sa;

  const __fp16 inv_hf        = (__fp16) inv;
  const HVX_Vector v_inv     = Q6_Vh_vsplat_R(fp16_to_bits(&inv_hf));
  const HVX_Vector v_clamp_n = Q6_Vb_vsplat_R((int8_t) -127);
  int              k         = 0;
  for (; k + 128 <= K; k += 128) {
    HVX_Vector v0    = vmemu(a + k);
    HVX_Vector v1    = vmemu(a + k + 64);
    HVX_Vector s0    = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, v_inv));
    HVX_Vector s1    = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v1, v_inv));
    HVX_Vector i0    = vhf_to_h_round(s0);              // round-to-nearest fp16 -> int16
    HVX_Vector i1    = vhf_to_h_round(s1);
    HVX_Vector i8    = Q6_Vb_vpack_VhVh_sat(i1, i0);    // int16x2 -> int8 saturate (i0 low, i1 high)
    i8               = Q6_Vb_vmax_VbVb(i8, v_clamp_n);  // clamp -128 -> -127
    vmem(qa_out + k) = i8;
  }
  // tail: one 64-fp16 chunk (K multiple of 64)
  if (k < K) {
    HVX_Vector v0 = vmemu(a + k);
    HVX_Vector s0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, v_inv));
    HVX_Vector i0 = vhf_to_h_round(s0);
    HVX_Vector i8 = Q6_Vb_vpack_VhVh_sat(Q6_V_vzero(), i0);
    i8            = Q6_Vb_vmax_VbVb(i8, v_clamp_n);
    vstu_variable(qa_out + k, K - k, i8);
  }
}

// Per-scale-block activation sums, pre-splatted across 32 fp32 lanes so every oc-tile can just
// vmem them (mirrors how a_splat is prepared once and reused).
//
// The asymmetric correction is out[oc] += sum_b qbias[oc][b] * (sum of a[k] for k in block b).
// With a[k] == sa * qa[k] the per-token scale factors out of that term exactly as it does out of
// the main term, so both share the single final multiply by sa and only the integer sums matter
// here. M == 1, so this is one value per block, computed once and reused across every oc-tile.
//
// Three deliberate choices, all measured. The naive version -- HVX store to a stack array followed
// immediately by gpb scalar loads of it, once per 128 activations -- cost 1020 ms of the 1537 ms
// asymmetric decode regression (66% of it), because an HVX store feeding a scalar load of the same
// address stalls hard, and worse when that address is in DDR rather than VTCM:
//   1) `scratch` is VTCM, not stack.
//   2) A vror tree reduction folds each block's gpb lanes onto lane j*gpb, so exactly ONE scalar is
//      read back per block instead of gpb of them.
//   3) The splat happens here, once, so compute_oc_tile_i8() does no scalar work at all -- it used
//      to do a scalar load + splat per (oc-tile, block), and lm_head has 4748 oc-tiles.
//
// scratch needs 32 + nblk int32: one vector of staging, plus the generic path's accumulators.
static void build_block_act_splats(const int8_t *qa, int K, int nblk, int32_t *scratch, HVX_Vector *qasum_splat) {
  const int        bs     = K / nblk;  // activations per block, a multiple of 32
  const HVX_Vector v_ones = Q6_Vb_vsplat_R(0x01);
  // Fast path: a 128-activation chunk holds a whole number of blocks, so block boundaries line up
  // with the vrmpy groups and the tree reduction lands each block's sum on a known lane.
  if (bs <= 128 && (128 % bs) == 0) {
    const HVX_Vector v_base = vmem(kLaneGroupBase);
    const int gpb = bs / 4;    // 4-wide vrmpy groups per block (8..32)
    const int bpc = 128 / bs;  // blocks per chunk
    int       b   = 0;
    for (int k = 0; k < K && b < nblk; k += 128) {
      HVX_Vector g = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), vmem(qa + k), v_ones);
      // After this lane i holds sum(lane i .. i+gpb-1) (rotating), so lane j*gpb is block j's sum.
      for (int r = 4; r < 4 * gpb; r <<= 1) {
        g = Q6_Vw_vadd_VwVw(g, Q6_V_vror_VR(g, r));
      }
      // Broadcast lane j*gpb with vdelta (see build_act_splats) instead of a scalar read-back.
      for (int j = 0; j < bpc && b < nblk; ++j, ++b) {
        const HVX_Vector v_ctrl = Q6_V_vxor_VV(v_base, Q6_Vb_vsplat_R(4 * j * gpb));
        qasum_splat[b]          = sf_from_w(Q6_V_vdelta_VV(g, v_ctrl));  // exact: |sum| <= 128*127
      }
    }
    return;
  }
  // Generic path (block size does not divide a chunk): fold group sums into per-block accumulators.
  int32_t *sums = scratch + 32;
  for (int b = 0; b < nblk; ++b) {
    sums[b] = 0;
  }
  for (int k = 0; k < K; k += 128) {
    const int remain = K - k;
    const int n      = remain < 128 ? remain : 128;  // K % 64 == 0, so n is 64 or 128
    // qa lives in VTCM with a_splat allocated straight after, so the 128B read is in bounds even
    // when only 64 activations remain; the extra group sums are simply not folded in.
    vmem(scratch)    = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), vmem(qa + k), v_ones);
    for (int j = 0; j < n / 4; ++j) {
      sums[(k + 4 * j) / bs] += scratch[j];
    }
  }
  for (int b = 0; b < nblk; ++b) {
    const float f  = (float) sums[b];
    qasum_splat[b] = Q6_V_vsplat_R(*(const int32_t *) &f);
  }
}

// Compute one output-channel tile (32 oc): out_f[oc] = sum_blk sw*acc, *sa.
// Weights are int4 (vrmpy-friendly layout): k-tile kt = 512 bytes = 4 loads of
// 128 int4; each load nibble-expands into two 4-k groups. Activation is
// pre-splatted per 4-k group into `a_splat` (reused across oc-tiles). Keeping the
// weight int4 halves the DDR stream (the dominant cost); the on-DSP unpack is a
// few HVX ops that overlap the DMA.
static inline HVX_Vector compute_oc_tile_i8(const uint8_t    *wtile,        // 512*kp int4 for this oc-tile
                                            const HVX_Vector *a_splat,      // K/4 pre-splatted qa words
                                            const float      *sw,           // nblk*32*(asym?2:1) fp32
                                            const HVX_Vector *qasum_splat,  // nblk splats, NULL if symmetric
                                            int kp, int nblk, float sa, int asym) {
  const int  ktpb   = kp / nblk;                                        // k-tiles per scale block (block=64 -> 2)
  // Asymmetric entries are 32 fp32 scale followed by 32 fp32 qbias.
  const int  entry  = 32;  // 128 B per block: 32 fp32 scale, or 32 fp16 (scale, qbias) pairs when asymmetric
  HVX_Vector out_qf = Q6_V_vzero();                                     // accumulate sum_blk (sw*acc) in qf32
  for (int b = 0; b < nblk; ++b) {
    HVX_Vector acc = Q6_V_vzero();                                      // 32 int32 lanes (one per oc)
    const int  kt0 = b * ktpb;
    for (int ktl = 0; ktl < ktpb; ++ktl) {
      const int         kt  = kt0 + ktl;
      const uint8_t    *wt  = wtile + (size_t) kt * 512;  // 4 loads * 128 int4
      const HVX_Vector *asp = a_splat + kt * 8;
      for (int l = 0; l < 4; ++l) {                       // 4 loads -> 8 groups of 4 k = 32 k
        HVX_VectorPair wv = unpack_vrmpy_weight_128(vmem(wt + l * 128));
        acc               = Q6_Vw_vrmpyacc_VwVbVb(acc, Q6_V_lo_W(wv), asp[2 * l]);
        acc               = Q6_Vw_vrmpyacc_VwVbVb(acc, Q6_V_hi_W(wv), asp[2 * l + 1]);
      }
    }
    HVX_Vector acc_sf = sf_from_w(acc);                 // int32 -> fp32
    HVX_Vector sw_sf  = vmem(sw + (size_t) b * entry);  // 32 fp32 (symmetric)
    HVX_Vector qb_sf  = Q6_V_vzero();
    if (asym) {  // even halves -> fp32 scale, odd halves -> fp32 qbias, both in oc order
      const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(sw_sf, Q6_Vh_vsplat_R(0x3C00));
      sw_sf                  = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w));
      qb_sf                  = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w));
    }
    HVX_Vector prod   = Q6_Vqf32_vmpy_VsfVsf(acc_sf, sw_sf);
    if (asym) {
      // + qbias[oc][b] * qasum[b]; the shared sa is applied once, below. Both operands are plain
      // vector loads -- no scalar work here, which matters because this runs per (oc-tile, block).
      prod             = Q6_Vqf32_vadd_Vqf32Vqf32(prod, Q6_Vqf32_vmpy_VsfVsf(qb_sf, qasum_splat[b]));
    }
    out_qf = Q6_Vqf32_vadd_Vqf32Vqf32(out_qf, prod);
  }
  HVX_Vector out_sf = Q6_Vsf_equals_Vqf32(out_qf);
  HVX_Vector v_sa   = Q6_V_vsplat_R(*(const uint32_t *) &sa);
  out_sf            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(out_sf, v_sa));
  return out_sf;  // 32 fp32 lanes, oc order
}

// Output routing over "members" laid out back to back in VTCM by global oc-tile.
//  plain GEMV: one member, one output, unit = tile.
//  split   (FusedLinear QKV):    member p's tiles go to outputs[p]; unit = tile. One launch
//                                instead of one per projection.
//  gate_up (FusedLinear gate/up): members gate, up; unit u = tiles u and tile_begin[1] + u, stored as
//                                up * silu(gate) -- the same fp16 values and the same
//                                htp_ops_binary_mul_silu arithmetic as the separate MUL_SILU op.
// A unit is usable once its DMA descriptor (desc_first[p] + desc_stride[p] * (local tile / chunk))
// is done; for gate_up that is the up chunk, which the ordered chain places after its gate chunk.
typedef struct {
  int                  nmem;
  int                  gate_up;
  int                  tile_begin[GEMV_I8_MAX_SEGMENTS + 1];  // tile_begin[nmem] == total tiles
  uint8_t             *out[GEMV_I8_MAX_SEGMENTS];
  const uint8_t       *bias[GEMV_I8_MAX_SEGMENTS];            // fp16 per member, NULL = none
  // DMA gating (wdesc NULL: the caller waited for the data already).
  const dma_desc_1d_t *wdesc;
  int                  desc_first[GEMV_I8_MAX_SEGMENTS];
  int                  desc_stride[GEMV_I8_MAX_SEGMENTS];
  int                  chunk_tiles;
} gemv_i8_layout_t;

typedef struct {
  const gemv_i8_layout_t *lay;
  const uint8_t          *vtcm_weight;  // int4 vrmpy layout; global tile `tile_base` at offset 0
  const HVX_Vector       *a_splat;      // K/4 pre-splatted qa words
  const float            *vtcm_sw;      // scales of tile `tile_base` at offset 0, nblk*(asym?64:32) per tile
  const HVX_Vector       *qasum_splat;  // nblk pre-splatted block activation sums, NULL when symmetric
  int                     kp, nblk;
  float                   sa;
  int                     asym;
  int                     tile_base;
  int                     u_start, u_end, u_step;  // output units handled by this worker
  worker_synctoken_t     *sync;
} gemv_i8_task_t;

static inline void wait_dma_desc_done(const dma_desc_1d_t *desc) {
  const volatile uint32_t *ctrl_word = (volatile const uint32_t *) &desc->dstate_order_bypass_type_length;
  while (((*ctrl_word >> 31) & 0x1) == 0) {
    asm volatile("nop");
  }
}

// Global tile -> fp16 (+bias of member p, local tile l) in the low 32 lanes.
static inline HVX_Vector gemv_i8_tile_hf(const gemv_i8_task_t *s, int tile, int p, int l) {
  const int      local = tile - s->tile_base;
  const uint8_t *wtile = s->vtcm_weight + (size_t) local * 512 * s->kp;
  const float   *sw    = s->vtcm_sw + (size_t) local * s->nblk * 32;
  HVX_Vector     out_hf =
    sf32_to_hf_low(compute_oc_tile_i8(wtile, s->a_splat, sw, s->qasum_splat, s->kp, s->nblk, s->sa, s->asym));
  const uint8_t *bias = s->lay->bias[p];
  if (bias) {
    HVX_Vector v_bias = vmemu((const __fp16 *) bias + (size_t) l * 32);
    out_hf            = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(out_hf, v_bias));
  }
  return out_hf;
}

static inline void gemv_i8_wait_tile(const gemv_i8_layout_t *lay, int p, int l) {
  if (lay->wdesc != NULL) {
    wait_dma_desc_done(&lay->wdesc[lay->desc_first[p] + lay->desc_stride[p] * (l / lay->chunk_tiles)]);
  }
}

// Each worker stores disjoint 64B output regions (cache-coherent CPU stores).
static void gemv_i8_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  const gemv_i8_task_t   *s   = (const gemv_i8_task_t *) data;
  const gemv_i8_layout_t *lay = s->lay;
  for (int u = s->u_start; u < s->u_end; u += s->u_step) {
    HVX_Vector v;
    int        p = 0;
    int        l = u;
    if (lay->gate_up) {
      gemv_i8_wait_tile(lay, 1, u);  // the up chunk; the ordered chain finished the gate chunk first
      const HVX_Vector gate = gemv_i8_tile_hf(s, u, 0, u);
      const HVX_Vector up   = gemv_i8_tile_hf(s, lay->tile_begin[1] + u, 1, u);
      // MUL_SILU convention: out = in0 * silu(in1), in0 = up, in1 = gate.
      v                     = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(up, htp_ops_silu_pwl_fp16_vec(gate)));
    } else {
      while (u >= lay->tile_begin[p + 1]) {
        ++p;
      }
      l = u - lay->tile_begin[p];
      gemv_i8_wait_tile(lay, p, l);
      v = gemv_i8_tile_hf(s, u, p, l);
    }
    vstu_variable(lay->out[p] + (size_t) l * 32 * sizeof(__fp16), 32 * sizeof(__fp16), v);
  }
  worker_pool_synctoken_jobdone(s->sync);
}

typedef struct {
  const uint8_t *weight;  // vrmpy int4 tiles followed by fp32 scales (the GEMV_I8 blob)
  int            np;      // oc-tiles of this member
} gemv_i8_member_t;

static int gemv_i8_run(gemv_i8_layout_t *lay, const gemv_i8_member_t *mem, const uint8_t *a, int K, int scale_block_num,
                       int scale_asymmetric) {
  if (scale_block_num <= 0 || (K % 64) != 0 || lay->nmem < 1 || lay->nmem > GEMV_I8_MAX_SEGMENTS) {
    return -1;
  }
  const int kp   = K / 32;
  const int nblk = scale_block_num;
  const int asym = scale_asymmetric ? 1 : 0;
  if ((kp % nblk) != 0) {
    return -1;  // supports block sizes multiple of 32 that divide K, K%64==0
  }
  int np = 0;
  for (int p = 0; p < lay->nmem; ++p) {
    if (mem[p].weight == NULL || mem[p].np <= 0 || lay->out[p] == NULL) {
      return -1;
    }
    lay->tile_begin[p] = np;
    np += mem[p].np;
  }
  lay->tile_begin[lay->nmem] = np;
  if (lay->gate_up && (lay->nmem != 2 || mem[0].np != mem[1].np)) {
    return -1;
  }

  const size_t weight_tile_bytes = (size_t) 512 * kp;  // per oc-tile (int4 vrmpy layout)
  // Asymmetric carries a qbias next to every scale, so this plane doubles.
  const size_t sw_bytes_per_oc   = (size_t) nblk * 128;  // see compute_oc_tile_i8
  const int    n_groups          = K / 4;              // 4-k vrmpy groups
  // Per-block activation sums (asymmetric only), shared by every worker: nblk pre-splatted vectors
  // plus one vector of reduction staging (see build_block_act_splats).
  const size_t qasum_bytes       = asym ? (size_t) nblk * sizeof(HVX_Vector) : 0;
  const size_t qscratch_bytes    = asym ? (((size_t) (32 + nblk) * sizeof(int32_t) + 127) & ~(size_t) 127) : 0;
  int          nworkers          = (int) g_max_num_workers;
  if (nworkers < 1) {
    nworkers = 1;
  }

  // Check if the whole weight fits in VTCM (fast path: single DMA chain, max overlap).
  const size_t vtcm_total = vtcm_manager_get_vtcm_size();
  const size_t need       = (size_t) K + (size_t) n_groups * sizeof(HVX_Vector) + qasum_bytes + qscratch_bytes +
                            (size_t) np * sw_bytes_per_oc + (size_t) np * weight_tile_bytes + 4096;
  if (need > vtcm_total) {
    if (lay->nmem != 1) {
      return -1;  // fused projections are only emitted when they fit (host checks the VTCM size)
    }
    const uint8_t *b_wt    = mem[0].weight;
    const uint8_t *b_scale = b_wt + (size_t) np * weight_tile_bytes;
    // Chunked path (e.g. lm_head): two VTCM buffers, so the DMA of chunk c + 1 streams while the workers
    // compute chunk c, instead of each chunk's DMA and compute running back to back.
    const size_t fixed_bytes =
      (size_t) K + (size_t) n_groups * sizeof(HVX_Vector) + qasum_bytes + qscratch_bytes + 4096;
    const size_t per_np = weight_tile_bytes + sw_bytes_per_oc;
    if (vtcm_total <= fixed_bytes || per_np == 0) {
      return -1;
    }
    int np_chunk = (int) ((vtcm_total - fixed_bytes) / (2 * per_np));
    if (np_chunk <= 0) {
      return -1;
    }
    if (np_chunk > np) {
      np_chunk = np;
    }

    // VTCM layout: qa / a_splat / qasum_splat / qscratch / {sw, weight} x 2
    uint8_t    *vtcm        = (uint8_t *) vtcm_manager_get_vtcm_base();
    int8_t     *vtcm_qa     = (int8_t *) vtcm_seq_alloc(&vtcm, (size_t) K * sizeof(int8_t));
    HVX_Vector *vtcm_asplat = (HVX_Vector *) vtcm_seq_alloc(&vtcm, (size_t) n_groups * sizeof(HVX_Vector));
    HVX_Vector *vtcm_qasum  = asym ? (HVX_Vector *) vtcm_seq_alloc(&vtcm, qasum_bytes) : NULL;
    int32_t    *vtcm_qscr   = asym ? (int32_t *) vtcm_seq_alloc(&vtcm, qscratch_bytes) : NULL;
    float      *vtcm_sw[2];
    uint8_t    *vtcm_weight[2];
    for (int b = 0; b < 2; ++b) {
      vtcm_sw[b]     = (float *) vtcm_seq_alloc(&vtcm, (size_t) np_chunk * sw_bytes_per_oc);
      vtcm_weight[b] = (uint8_t *) vtcm_seq_alloc(&vtcm, (size_t) np_chunk * weight_tile_bytes);
    }

    _Alignas(64) dma_desc_1d_t wdesc[2];
    _Alignas(64) dma_desc_1d_t sdesc[2];
    // Weight + scale of chunk [oy_base, oy_base + count) into buffer b.
#define GEMV_I8_ISSUE_CHUNK(b, oy_base, count)                                     \
  do {                                                                             \
    memset(&wdesc[b], 0, sizeof(wdesc[b]));                                        \
    memset(&sdesc[b], 0, sizeof(sdesc[b]));                                        \
    wdesc[b].length = (uint32_t) ((size_t) (count) * weight_tile_bytes);           \
    wdesc[b].src    = (uint32_t) (b_wt + (size_t) (oy_base) * weight_tile_bytes);  \
    wdesc[b].dst    = (uint32_t) vtcm_weight[b];                                   \
    wdesc[b].next   = (uint32_t) &sdesc[b];                                        \
    sdesc[b].length = (uint32_t) ((size_t) (count) * sw_bytes_per_oc);             \
    sdesc[b].src    = (uint32_t) (b_scale + (size_t) (oy_base) * sw_bytes_per_oc); \
    sdesc[b].dst    = (uint32_t) vtcm_sw[b];                                       \
    wdesc[b].type = sdesc[b].type = DMA_DESC_TYPE_1D;                              \
    wdesc[b].src_bypass = sdesc[b].src_bypass = 1;                                 \
    wdesc[b].dst_bypass = sdesc[b].dst_bypass = 1;                                 \
    wdesc[b].ordered = sdesc[b].ordered = 1;                                       \
    wdesc[b].dstate = sdesc[b].dstate = DMA_DESC_DSTATE_PENDING;                   \
    dmstart(&wdesc[b]);                                                            \
  } while (0)

    // The first chunk's DMA overlaps the activation quantization.
    GEMV_I8_ISSUE_CHUNK(0, 0, np_chunk);
    float sa = 1.0f;
    quantize_activation_row((const __fp16 *) a, K, vtcm_qa, &sa);
    build_act_splats(vtcm_qa, n_groups, vtcm_asplat);
    if (asym) {
      build_block_act_splats(vtcm_qa, K, nblk, vtcm_qscr, vtcm_qasum);
    }
    dma_wait_for_idle();
    lay->wdesc = NULL;

    // One worker fewer than HVX contexts, as in the single-chain path: this thread's DMA must keep running.
    const int nwmax = nworkers > 1 ? nworkers - 1 : 1;
    int       buf   = 0;
    for (int oy_base = 0; oy_base < np; oy_base += np_chunk, buf ^= 1) {
      const int oy_end   = oy_base + np_chunk < np ? oy_base + np_chunk : np;
      const int chunk_np = oy_end - oy_base;
      // Prefetch the next chunk into the other buffer (its previous chunk finished last iteration).
      if (oy_end < np) {
        const int next_np = oy_end + np_chunk < np ? np_chunk : np - oy_end;
        GEMV_I8_ISSUE_CHUNK(buf ^ 1, oy_end, next_np);
      }
      const int          nw  = nwmax > chunk_np ? chunk_np : nwmax;
      const int          per = (chunk_np + nw - 1) / nw;
      worker_synctoken_t sync;
      worker_pool_synctoken_init(&sync, nw);
      gemv_i8_task_t tasks[nw];
      for (int w = 0; w < nw; ++w) {
        int s0 = oy_base + w * per;
        int s1 = s0 + per;
        if (s1 > oy_end) {
          s1 = oy_end;
        }
        tasks[w] = (gemv_i8_task_t) { lay,  vtcm_weight[buf], vtcm_asplat, vtcm_sw[buf], vtcm_qasum, kp,   nblk, sa,
                                      asym, oy_base,          s0,          s1,           1,          &sync };
        worker_pool_job_t job;
        job.fptr = gemv_i8_worker_loop;
        job.dptr = &tasks[w];
        if (worker_pool_submit(NULL, job) != 0) {
          gemv_i8_worker_loop(&tasks[w], 0);
        }
      }
      // dmwait before the semaphore (see the single-chain path below).
      dma_wait_for_idle();
      worker_pool_synctoken_wait(&sync);
    }
#undef GEMV_I8_ISSUE_CHUNK
    return 0;
  }

  uint8_t    *vtcm        = (uint8_t *) vtcm_manager_get_vtcm_base();
  int8_t     *vtcm_qa     = (int8_t *) vtcm_seq_alloc(&vtcm, (size_t) K * sizeof(int8_t));
  HVX_Vector *vtcm_asplat = (HVX_Vector *) vtcm_seq_alloc(&vtcm, (size_t) n_groups * sizeof(HVX_Vector));
  HVX_Vector *vtcm_qasum  = asym ? (HVX_Vector *) vtcm_seq_alloc(&vtcm, qasum_bytes) : NULL;
  int32_t    *vtcm_qscr   = asym ? (int32_t *) vtcm_seq_alloc(&vtcm, qscratch_bytes) : NULL;
  float      *vtcm_sw     = (float *) vtcm_seq_alloc(&vtcm, (size_t) np * sw_bytes_per_oc);
  uint8_t    *vtcm_weight = (uint8_t *) vtcm_seq_alloc(&vtcm, (size_t) np * weight_tile_bytes);

  const int nunits = lay->gate_up ? mem[0].np : np;
  // One worker fewer than HVX contexts: the workers spin on the descriptors, and the issuing thread's user
  // DMA only advances while that thread is scheduled. With a spinner on every hardware thread (v79: 6 workers
  // + the issuer on 6 threads) it was intermittently starved and single decode GEMVs stalled for tens of ms.
  const int nwmax  = nworkers > 1 ? nworkers - 1 : 1;
  const int nw     = nwmax > nunits ? nunits : nwmax;

  // 1. One ordered DMA chain: every member's scales first, then the weights in oc-tile chunks
  // (gate_up: gate chunk c, then up chunk c). Workers start on a chunk as soon as its descriptor
  // is done, so compute overlaps the weight stream instead of following it. Units are dealt
  // round-robin, so every chunk is spread over all workers and only the last chunk's units remain
  // after the stream ends. Descriptors live on the stack (DDR) -- the proven done-bit polling.
  enum { kMaxChunksPerMember = 8, kMaxDescs = GEMV_I8_MAX_SEGMENTS * (kMaxChunksPerMember + 1) };

  int maxMemberTiles = 0;
  for (int p = 0; p < lay->nmem; ++p) {
    maxMemberTiles = mem[p].np > maxMemberTiles ? mem[p].np : maxMemberTiles;
  }
  int chunk_tiles = nw;
  if (chunk_tiles * kMaxChunksPerMember < maxMemberTiles) {
    chunk_tiles = (maxMemberTiles + kMaxChunksPerMember - 1) / kMaxChunksPerMember;
  }
  _Alignas(64) dma_desc_1d_t desc[kMaxDescs];
  int                        ndesc = 0;
  for (int p = 0; p < lay->nmem; ++p) {  // scales
    dma_desc_1d_t *d = &desc[ndesc++];
    memset(d, 0, sizeof(*d));
    d->length = (uint32_t) ((size_t) mem[p].np * sw_bytes_per_oc);
    d->src    = (uint32_t) (mem[p].weight + (size_t) mem[p].np * weight_tile_bytes);
    d->dst    = (uint32_t) ((uint8_t *) vtcm_sw + (size_t) lay->tile_begin[p] * sw_bytes_per_oc);
  }
  const int interleave = lay->gate_up ? 2 : 1;
  for (int p = 0; p < lay->nmem; ++p) {
    lay->desc_first[p]  = lay->gate_up ? lay->nmem + p : ndesc;
    lay->desc_stride[p] = interleave;
    if (!lay->gate_up) {
      for (int t0 = 0; t0 < mem[p].np; t0 += chunk_tiles) {
        const int      t1 = t0 + chunk_tiles < mem[p].np ? t0 + chunk_tiles : mem[p].np;
        dma_desc_1d_t *d  = &desc[ndesc++];
        memset(d, 0, sizeof(*d));
        d->length = (uint32_t) ((size_t) (t1 - t0) * weight_tile_bytes);
        d->src    = (uint32_t) (mem[p].weight + (size_t) t0 * weight_tile_bytes);
        d->dst    = (uint32_t) (vtcm_weight + (size_t) (lay->tile_begin[p] + t0) * weight_tile_bytes);
      }
    }
  }
  if (lay->gate_up) {
    for (int t0 = 0; t0 < mem[0].np; t0 += chunk_tiles) {
      const int t1 = t0 + chunk_tiles < mem[0].np ? t0 + chunk_tiles : mem[0].np;
      for (int p = 0; p < 2; ++p) {
        dma_desc_1d_t *d = &desc[ndesc++];
        memset(d, 0, sizeof(*d));
        d->length = (uint32_t) ((size_t) (t1 - t0) * weight_tile_bytes);
        d->src    = (uint32_t) (mem[p].weight + (size_t) t0 * weight_tile_bytes);
        d->dst    = (uint32_t) (vtcm_weight + (size_t) (lay->tile_begin[p] + t0) * weight_tile_bytes);
      }
    }
  }
  for (int i = 0; i < ndesc; ++i) {
    desc[i].next       = i + 1 < ndesc ? (uint32_t) &desc[i + 1] : 0;
    desc[i].type       = DMA_DESC_TYPE_1D;
    desc[i].src_bypass = 1;
    desc[i].dst_bypass = 1;
    desc[i].ordered    = 1;
    desc[i].dstate     = DMA_DESC_DSTATE_PENDING;
  }
  lay->wdesc       = desc;
  lay->chunk_tiles = chunk_tiles;
  dmstart(&desc[0]);

  // 2. quantize activation to symmetric int8 + pre-splat (overlaps the DMA above).
  float sa = 1.0f;
  quantize_activation_row((const __fp16 *) a, K, vtcm_qa, &sa);
  build_act_splats(vtcm_qa, n_groups, vtcm_asplat);
  if (asym) {
    build_block_act_splats(vtcm_qa, K, nblk, vtcm_qscr, vtcm_qasum);
  }

  // 3. compute, round-robin over output units, each gated on its chunk's descriptor.
  worker_synctoken_t sync;
  worker_pool_synctoken_init(&sync, nw);
  gemv_i8_task_t tasks[nw];
  for (int w = 0; w < nw; ++w) {
    tasks[w] = (gemv_i8_task_t) { lay, vtcm_weight, vtcm_asplat, vtcm_sw, vtcm_qasum, kp, nblk,
                                  sa,  asym,        0,           w,       nunits,     nw, &sync };
    worker_pool_job_t job;
    job.fptr = gemv_i8_worker_loop;
    job.dptr = &tasks[w];
    if (worker_pool_submit(NULL, job) != 0) {
      gemv_i8_worker_loop(&tasks[w], 0);  // pool full: run inline
    }
  }
  // The issuing thread must stay on dmwait until the chain drains -- the same order as
  // conv1x1_w8a16_sym_per_channel.cc. Blocking on the semaphore first while the workers spin on
  // dstate hung the cDSP (the user DMA belongs to this thread) and took the device down.
  dma_wait_for_idle();
  worker_pool_synctoken_wait(&sync);
  return 0;
}

int hmx_matmulq4block_gemv_i8(uint8_t *c, const uint8_t *a, const uint8_t *b_wt, const uint8_t *b_scale,
                              const uint8_t *bias, int K, int N, int scale_block_num, int scale_asymmetric) {
  if ((N % 32) != 0 || (K % 32) != 0 || b_scale != b_wt + (size_t) (K / 32) * (N / 32) * 512) {
    return -1;
  }
  gemv_i8_layout_t lay;
  memset(&lay, 0, sizeof(lay));
  lay.nmem                   = 1;
  lay.out[0]                 = c;
  lay.bias[0]                = bias;
  const gemv_i8_member_t mem = { b_wt, N / 32 };
  return gemv_i8_run(&lay, &mem, a, K, scale_block_num, scale_asymmetric);
}

int hmx_matmulq4block_gemv_i8_fused(uint8_t *const *outs, const uint8_t *const *weights, const uint8_t *const *biases,
                                    const int32_t *n, int nmem, int gate_up, const uint8_t *a, int K,
                                    int scale_block_num, int scale_asymmetric) {
  if (nmem < 1 || nmem > GEMV_I8_MAX_SEGMENTS || (gate_up && nmem != 2)) {
    return -1;
  }
  gemv_i8_layout_t lay;
  memset(&lay, 0, sizeof(lay));
  lay.nmem    = nmem;
  lay.gate_up = gate_up ? 1 : 0;
  gemv_i8_member_t mem[GEMV_I8_MAX_SEGMENTS];
  for (int p = 0; p < nmem; ++p) {
    if ((n[p] % 32) != 0) {
      return -1;
    }
    mem[p].weight = weights[p];
    mem[p].np     = n[p] / 32;
    lay.bias[p]   = biases[p];
    lay.out[p]    = gate_up ? outs[0] : outs[p];
  }
  return gemv_i8_run(&lay, mem, a, K, scale_block_num, scale_asymmetric);
}
