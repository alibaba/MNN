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
// `b_scale` = fp32, per oc-tile contiguous: sw(y, blk, ocIn) at
//   b_scale[(y*nblk + blk)*32 + ocIn].  nblk = scale_block_num = K/blocksize.
//
// OUTPUT: fp16, linear by output channel (for M=1 the NC4HW4 pack-64 layout
// degenerates to linear), c[oc].
//
// v1: single-threaded, correctness-first. Parallelization over oc-tiles and
// weight DMA double-buffering are follow-ups (see strategy doc Phase 1 step 6).

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "dsp/dma_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_utils.h"
#include "dsp/ops.h"
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
// Valid for |x| < 2^22; our accumulator is bounded by ~blocksize*127*7 << 2^22.
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

// Per-token symmetric int8 quantization of the single activation row (K fp16).
static void quantize_activation_row(const __fp16 *a, int K, int8_t *qa_out, float *sa_out) {
  const int        nv        = K / 64;  // K is a multiple of 64 (blocksize) for supported cases
  const HVX_Vector v_absmask = Q6_Vh_vsplat_R(0x7fff);
  HVX_Vector       v_max     = Q6_V_vzero();
  for (int i = 0; i < nv; ++i) {
    HVX_Vector v = vmemu(a + i * 64);
    v_max        = Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vand_VV(v, v_absmask));
  }
  // horizontal max over the 64 fp16 lanes (once per matmul, scalar is fine)
  _Alignas(128) __fp16 tmp[64];
  vmem(tmp)    = v_max;
  float absmax = 0.0f;
  for (int i = 0; i < 64; ++i) {
    float f = (float) tmp[i];
    if (f > absmax) {
      absmax = f;
    }
  }
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

// Compute one output-channel tile (32 oc): out_f[oc] = sum_blk sw*acc, *sa.
// Weights are int4 (vrmpy-friendly layout): k-tile kt = 512 bytes = 4 loads of
// 128 int4; each load nibble-expands into two 4-k groups. Activation is
// pre-splatted per 4-k group into `a_splat` (reused across oc-tiles). Keeping the
// weight int4 halves the DDR stream (the dominant cost); the on-DSP unpack is a
// few HVX ops that overlap the DMA.
static inline HVX_Vector compute_oc_tile_i8(const uint8_t    *wtile,    // 512*kp int4 for this oc-tile
                                            const HVX_Vector *a_splat,  // K/4 pre-splatted qa words
                                            const float      *sw,       // nblk*32 fp32
                                            int kp, int nblk, float sa) {
  const int  ktpb   = kp / nblk;                                        // k-tiles per scale block (block=64 -> 2)
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
    HVX_Vector acc_sf = sf_from_w(acc);              // int32 -> fp32
    HVX_Vector sw_sf  = vmem(sw + (size_t) b * 32);  // 32 fp32
    HVX_Vector prod   = Q6_Vqf32_vmpy_VsfVsf(acc_sf, sw_sf);
    out_qf            = Q6_Vqf32_vadd_Vqf32Vqf32(out_qf, prod);
  }
  HVX_Vector out_sf = Q6_Vsf_equals_Vqf32(out_qf);
  HVX_Vector v_sa   = Q6_V_vsplat_R(*(const uint32_t *) &sa);
  out_sf            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(out_sf, v_sa));
  return out_sf;  // 32 fp32 lanes, oc order
}

typedef struct {
  uint8_t            *c;
  const uint8_t      *vtcm_weight;  // whole weight (int4 vrmpy layout), tile oy at oy*512*kp
  const HVX_Vector   *a_splat;      // K/4 pre-splatted qa words
  const float        *vtcm_sw;      // whole scales, oc-tile oy at oy*nblk*32
  const uint8_t      *bias;
  int                 kp, nblk;
  float               sa;
  int                 oy_start, oy_end;  // this worker's global oc-tile range
  worker_synctoken_t *sync;
} gemv_i8_task_t;

// Each worker computes a disjoint range of oc-tiles and stores directly to c
// (non-overlapping 64B regions, cache-coherent CPU stores).
static void gemv_i8_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  gemv_i8_task_t *s           = (gemv_i8_task_t *) data;
  const size_t    wtile_bytes = (size_t) 512 * s->kp;  // int4 vrmpy tiles
  for (int oy = s->oy_start; oy < s->oy_end; ++oy) {
    const uint8_t *wtile  = s->vtcm_weight + (size_t) oy * wtile_bytes;
    const float   *sw     = s->vtcm_sw + (size_t) oy * s->nblk * 32;
    HVX_Vector     out_sf = compute_oc_tile_i8(wtile, s->a_splat, sw, s->kp, s->nblk, s->sa);
    HVX_Vector     out_hf = sf32_to_hf_low(out_sf);
    if (s->bias) {
      HVX_Vector v_bias = vmemu((const __fp16 *) s->bias + (size_t) oy * 32);
      out_hf            = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(out_hf, v_bias));
    }
    vstu_variable((uint8_t *) s->c + (size_t) oy * 32 * sizeof(__fp16), 32 * sizeof(__fp16), out_hf);
  }
  worker_pool_synctoken_jobdone(s->sync);
}

int hmx_matmulq4block_gemv_i8(uint8_t *c, const uint8_t *a, const uint8_t *b_wt, const uint8_t *b_scale,
                              const uint8_t *bias, int K, int N, int scale_block_num) {
  if (scale_block_num <= 0 || (K % 32) != 0 || (N % 32) != 0) {
    return -1;
  }
  const int kp   = K / 32;
  const int np   = N / 32;
  const int nblk = scale_block_num;
  if ((kp % nblk) != 0 || (K % 64) != 0) {
    return -1;  // supports block sizes multiple of 32 that divide K, K%64==0
  }

  const size_t weight_tile_bytes = (size_t) 512 * kp;  // per oc-tile (int4 vrmpy layout)
  const size_t sw_bytes_per_oc   = (size_t) nblk * 32 * sizeof(float);
  const int    n_groups          = K / 4;              // 4-k vrmpy groups

  // Check if the whole weight fits in VTCM (fast path: single DMA, max overlap).
  const size_t vtcm_total = vtcm_manager_get_vtcm_size();
  const size_t need = (size_t) K + (size_t) n_groups * sizeof(HVX_Vector) + (size_t) np * sw_bytes_per_oc +
                      (size_t) np * weight_tile_bytes + 4096;
  if (need > vtcm_total) {
    // Fall through to chunked path: iterate over oc-tile groups.
    const size_t fixed_bytes = (size_t) K + (size_t) n_groups * sizeof(HVX_Vector) + 4096;
    const size_t per_np      = weight_tile_bytes + sw_bytes_per_oc;
    if (vtcm_total <= fixed_bytes || per_np == 0) {
      return -1;
    }
    int np_chunk = (int) ((vtcm_total - fixed_bytes) / per_np);
    if (np_chunk <= 0) {
      return -1;
    }
    if (np_chunk > np) {
      np_chunk = np;
    }

    // VTCM layout: qa / a_splat / chunk_sw / chunk_weight
    uint8_t    *vtcm        = (uint8_t *) vtcm_manager_get_vtcm_base();
    int8_t     *vtcm_qa     = (int8_t *) vtcm_seq_alloc(&vtcm, (size_t) K * sizeof(int8_t));
    HVX_Vector *vtcm_asplat = (HVX_Vector *) vtcm_seq_alloc(&vtcm, (size_t) n_groups * sizeof(HVX_Vector));
    float      *vtcm_sw     = (float *) vtcm_seq_alloc(&vtcm, (size_t) np_chunk * sw_bytes_per_oc);
    uint8_t    *vtcm_weight = (uint8_t *) vtcm_seq_alloc(&vtcm, (size_t) np_chunk * weight_tile_bytes);

    // Pipeline overlap: start first chunk's DMA before quantizing activation,
    // so the DMA streams in parallel with the HVX quantize+splat work.
    _Alignas(64) dma_desc_1d_t wdesc;
    _Alignas(64) dma_desc_1d_t sdesc;
    const int                  first_chunk_np = np_chunk > np ? np : np_chunk;
    memset(&wdesc, 0, sizeof(wdesc));
    wdesc.length     = (uint32_t) ((size_t) first_chunk_np * weight_tile_bytes);
    wdesc.type       = DMA_DESC_TYPE_1D;
    wdesc.src_bypass = 1;
    wdesc.dst_bypass = 1;
    wdesc.ordered    = 1;
    wdesc.dstate     = DMA_DESC_DSTATE_PENDING;
    wdesc.src        = (uint32_t) b_wt;
    wdesc.dst        = (uint32_t) vtcm_weight;
    memset(&sdesc, 0, sizeof(sdesc));
    sdesc.length     = (uint32_t) ((size_t) first_chunk_np * sw_bytes_per_oc);
    sdesc.type       = DMA_DESC_TYPE_1D;
    sdesc.src_bypass = 1;
    sdesc.dst_bypass = 1;
    sdesc.ordered    = 1;
    sdesc.dstate     = DMA_DESC_DSTATE_PENDING;
    sdesc.src        = (uint32_t) b_scale;
    sdesc.dst        = (uint32_t) vtcm_sw;
    sdesc.next       = 0;
    wdesc.next       = (uint32_t) &sdesc;
    dmstart(&wdesc);

    // Quantize activation (overlaps first chunk DMA — independent HVX work).
    float sa = 1.0f;
    quantize_activation_row((const __fp16 *) a, K, vtcm_qa, &sa);
    for (int gg = 0; gg < n_groups; ++gg) {
      vtcm_asplat[gg] = Q6_V_vsplat_R(*(const uint32_t *) (vtcm_qa + gg * 4));
    }

    // Wait for first chunk DMA, then iterate over all chunks.
    dma_wait_for_idle();

    for (int oy_base = 0; oy_base < np; oy_base += np_chunk) {
      int oy_end = oy_base + np_chunk;
      if (oy_end > np) {
        oy_end = np;
      }
      const int chunk_np = oy_end - oy_base;

      // For chunks after the first, DMA this chunk's weight + scale.
      if (oy_base > 0) {
        memset(&wdesc, 0, sizeof(wdesc));
        wdesc.length     = (uint32_t) ((size_t) chunk_np * weight_tile_bytes);
        wdesc.type       = DMA_DESC_TYPE_1D;
        wdesc.src_bypass = 1;
        wdesc.dst_bypass = 1;
        wdesc.ordered    = 1;
        wdesc.dstate     = DMA_DESC_DSTATE_PENDING;
        wdesc.src        = (uint32_t) (b_wt + (size_t) oy_base * weight_tile_bytes);
        wdesc.dst        = (uint32_t) vtcm_weight;
        memset(&sdesc, 0, sizeof(sdesc));
        sdesc.length     = (uint32_t) ((size_t) chunk_np * sw_bytes_per_oc);
        sdesc.type       = DMA_DESC_TYPE_1D;
        sdesc.src_bypass = 1;
        sdesc.dst_bypass = 1;
        sdesc.ordered    = 1;
        sdesc.dstate     = DMA_DESC_DSTATE_PENDING;
        sdesc.src        = (uint32_t) (b_scale + (size_t) oy_base * sw_bytes_per_oc);
        sdesc.dst        = (uint32_t) vtcm_sw;
        sdesc.next       = 0;
        wdesc.next       = (uint32_t) &sdesc;
        dmstart(&wdesc);
        dma_wait_for_idle();
      }

      // Parallel compute over this chunk's oc-tiles.
      int nworkers = (int) g_max_num_workers;
      if (nworkers < 1) {
        nworkers = 1;
      }
      int                nw  = nworkers > chunk_np ? chunk_np : nworkers;
      const int          per = (chunk_np + nw - 1) / nw;
      worker_synctoken_t sync;
      worker_pool_synctoken_init(&sync, nw);
      gemv_i8_task_t tasks[nw];
      for (int w = 0; w < nw; ++w) {
        int s0 = w * per;
        int s1 = s0 + per;
        if (s1 > chunk_np) {
          s1 = chunk_np;
        }
        tasks[w] = (gemv_i8_task_t) { c + (size_t) oy_base * 32 * sizeof(__fp16),
                                      vtcm_weight,
                                      vtcm_asplat,
                                      vtcm_sw,
                                      bias ? bias + (size_t) oy_base * 32 * sizeof(__fp16) : NULL,
                                      kp,
                                      nblk,
                                      sa,
                                      s0,
                                      s1,
                                      &sync };
        worker_pool_job_t job;
        job.fptr = gemv_i8_worker_loop;
        job.dptr = &tasks[w];
        if (worker_pool_submit(NULL, job) != 0) {
          gemv_i8_worker_loop(&tasks[w], 0);
        }
      }
      worker_pool_synctoken_wait(&sync);
    }
    return 0;
  }

  uint8_t    *vtcm        = (uint8_t *) vtcm_manager_get_vtcm_base();
  int8_t     *vtcm_qa     = (int8_t *) vtcm_seq_alloc(&vtcm, (size_t) K * sizeof(int8_t));
  HVX_Vector *vtcm_asplat = (HVX_Vector *) vtcm_seq_alloc(&vtcm, (size_t) n_groups * sizeof(HVX_Vector));
  float      *vtcm_sw     = (float *) vtcm_seq_alloc(&vtcm, (size_t) np * sw_bytes_per_oc);
  uint8_t    *vtcm_weight = (uint8_t *) vtcm_seq_alloc(&vtcm, (size_t) np * weight_tile_bytes);

  // DMA descriptors on stack (DDR) — matches the fp16 kernel's proven done-bit
  // polling (the DMA engine updates the dstate bit coherently there).
  _Alignas(64) dma_desc_1d_t sdesc;
  _Alignas(64) dma_desc_1d_t wdesc;

  // 1. Issue the whole weight + all scales as one DMA (two chained descriptors),
  // started before quant so it streams on the HW DMA engine while the main thread
  // quantizes below (independent inputs/engines) — the main pipeline overlap.
  memset(&wdesc, 0, sizeof(wdesc));
  wdesc.length     = (uint32_t) ((size_t) np * weight_tile_bytes);
  wdesc.type       = DMA_DESC_TYPE_1D;
  wdesc.src_bypass = 1;
  wdesc.dst_bypass = 1;
  wdesc.ordered    = 1;
  wdesc.dstate     = DMA_DESC_DSTATE_PENDING;
  wdesc.src        = (uint32_t) b_wt;
  wdesc.dst        = (uint32_t) vtcm_weight;
  memset(&sdesc, 0, sizeof(sdesc));
  sdesc.length     = (uint32_t) ((size_t) np * sw_bytes_per_oc);
  sdesc.type       = DMA_DESC_TYPE_1D;
  sdesc.src_bypass = 1;
  sdesc.dst_bypass = 1;
  sdesc.ordered    = 1;
  sdesc.dstate     = DMA_DESC_DSTATE_PENDING;
  sdesc.src        = (uint32_t) b_scale;
  sdesc.dst        = (uint32_t) vtcm_sw;
  sdesc.next       = 0;
  wdesc.next       = (uint32_t) &sdesc;
  dmstart(&wdesc);

  // 2. quantize activation to symmetric int8 + pre-splat (overlaps the weight DMA
  // above — independent inputs/engines, this is the main pipeline overlap).
  float sa = 1.0f;
  quantize_activation_row((const __fp16 *) a, K, vtcm_qa, &sa);
  for (int gg = 0; gg < n_groups; ++gg) {
    vtcm_asplat[gg] = Q6_V_vsplat_R(*(const uint32_t *) (vtcm_qa + gg * 4));
  }

  // 3. wait for the whole weight DMA, then compute (parallel across oc-tiles).
  dma_wait_for_idle();
  int nworkers = (int) g_max_num_workers;
  if (nworkers < 1) {
    nworkers = 1;
  }
  int                nw  = nworkers > np ? np : nworkers;
  const int          per = (np + nw - 1) / nw;
  worker_synctoken_t sync;
  worker_pool_synctoken_init(&sync, nw);
  gemv_i8_task_t tasks[nw];
  for (int w = 0; w < nw; ++w) {
    int s0 = w * per;
    int s1 = s0 + per;
    if (s1 > np) {
      s1 = np;
    }
    tasks[w] = (gemv_i8_task_t) { c, vtcm_weight, vtcm_asplat, vtcm_sw, bias, kp, nblk, sa, s0, s1, &sync };
    worker_pool_job_t job;
    job.fptr = gemv_i8_worker_loop;
    job.dptr = &tasks[w];
    if (worker_pool_submit(NULL, job) != 0) {
      gemv_i8_worker_loop(&tasks[w], 0);  // pool full: run inline
    }
  }
  worker_pool_synctoken_wait(&sync);
  return 0;
}
