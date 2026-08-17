// Decode GEMV (M=1) for block-quant int8 symmetric weights using HVX integer vrmpy.
//
// Specialized M=1 W8A16 fast path: activation is quantized to symmetric int8
// once per token and stays resident in VTCM across all output-channel tiles;
// weights stay in the integer domain (int8 x int8 -> int32 via Q6_Vw_vrmpyacc)
// and are streamed from DDR by DMA (whole blob when it fits, else double-buffered
// oc-tile chunks). The combined activation*weight scale is applied once per
// 64-element block in fp32. No HMX, no fp16 dequant.
//
// ---------------------------------------------------------------------------
// WEIGHT LAYOUT CONTRACT (consumes the existing HMX int8Weight blob, produced
// host-side by reorderInt8SymWeightForHmx — no second layout is allocated)
// ---------------------------------------------------------------------------
// Dimensions: K (=ic), N (=oc). kp=K/32, np=N/32.
// Weight blob b_wt = kp*np int8 tiles of 1024 bytes, tile (oy, kx) at byte
// offset (oy*kp + kx)*1024.
// Inside a 1024B tile: 8 groups g in [0,8), each 128 bytes, group g covers the
// 4 k-values k = kx*32 + 4g + {0,1,2,3} for all 32 output channels of the tile.
// The HMX interleave stores byte (g*128 + ocIn*4 + p) = quant(ocIn, k = kx*32 +
// 4g + perm[p]) with perm = {0,2,1,3} (the pair-interleave swaps the middle two
// bytes of each 32-bit lane). A 128B load is one group; per 32-bit lane ocIn the
// four int8 weights are [w(k0),w(k2),w(k1),w(k3)]. Because the dot product is
// commutative, the kernel instead splats the activation in the same permuted
// order (swap bytes 1 and 2 of each 4-byte activation word when building
// a_splat), so vrmpy still computes sum_k w(k)*a(k) exactly.
//
// SCALE LAYOUT CONTRACT (separate small fp32 blob, host-side scales-only reorder):
// b_scale = fp32, per oc-tile contiguous: sw(y, blk, ocIn) at
// b_scale[(y*nblk + blk)*32 + ocIn].  nblk = scale_block_num = K/blocksize (64).
//
// OUTPUT: fp16, linear by output channel (for M=1 the NC4HW4 pack-64 layout
// degenerates to linear), c[oc].
//
// Guard (host side): M==1, K%64==0, N%32==0, block size 64, INT8 symmetric,
// kernel 1x1, no relu. Anything else stays on hmx_matmul_w8a16_block_fp16.

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

// -------- helpers (mirror matmul_q4block_gemv_i8.c) -------------------------

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

// int32 -> fp32 via the magic-number trick (no direct HVX cvt intrinsic).
// Valid for |x| < 2^22; our per-block accumulator is bounded by
// 64*127*127 << 2^22.
static inline HVX_Vector sf_from_w(HVX_Vector v_w) {
  const HVX_Vector v_magic_bits = Q6_V_vsplat_R(0x4B400000);           // 1.5 * 2^23 = 12582912.0
  const HVX_Vector v_neg_magic  = Q6_V_vsplat_R(0xCB400000);           // -12582912.0 (as fp32)
  HVX_Vector       vbits        = Q6_Vw_vadd_VwVw(v_w, v_magic_bits);  // reinterpret as fp32 = 12582912 + x
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(vbits, v_neg_magic));
}

// Convert 32 fp32 lanes (oc order) -> 32 fp16 in the low 32 lanes (oc order).
static inline HVX_Vector sf32_to_hf_low(HVX_Vector v_sf) {
  return Q6_Vh_vdeal_Vh(hvx_my_wsf_to_vhf(v_sf, v_sf));
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

  const __fp16     inv_hf    = (__fp16) inv;
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

// Splat one 4-k activation group as a 32-bit word in the HMX weight order
// [a(k0), a(k2), a(k1), a(k3)] (swap bytes 1 and 2 of the linear word). vrmpy
// then pairs each weight w(k) with its own a(k) (sum order is irrelevant).
static inline HVX_Vector splat_group_permuted(const int8_t *qa, int gg) {
  uint32_t w = *(const uint32_t *) (qa + (size_t) gg * 4);
  w          = (w & 0xFF0000FFu) | ((w & 0x00FF0000u) >> 8) | ((w & 0x0000FF00u) << 8);
  return Q6_V_vsplat_R(w);
}

// Compute one output-channel tile (32 oc): out_f[oc] = sa * sum_blk sw*acc.
// Weights are int8 (vrmpy-friendly layout): k-tile kt = 1024 bytes = 8 loads of
// 128 int8; each load is one 4-k group, consumed directly by vrmpy. Activation
// is pre-splatted per 4-k group into a_splat (reused across oc-tiles).
static inline HVX_Vector compute_oc_tile_i8(const uint8_t    *wtile,    // 1024*kp int8 for this oc-tile
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
      const uint8_t    *wt  = wtile + (size_t) kt * 1024;
      const HVX_Vector *asp = a_splat + kt * 8;
      for (int l = 0; l < 8; ++l) {  // 8 loads -> 8 groups of 4 k = 32 k
        acc = Q6_Vw_vrmpyacc_VwVbVb(acc, vmem(wt + l * 128), asp[l]);
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
  const uint8_t      *vtcm_weight;  // whole chunk weight (int8 vrmpy layout), tile oy at oy*1024*kp
  const HVX_Vector   *a_splat;      // K/4 pre-splatted qa words
  const float        *vtcm_sw;      // chunk scales, oc-tile oy at oy*nblk*32
  const uint8_t      *bias;
  int                 kp, nblk;
  float               sa;
  int                 oy_start, oy_end;  // this worker's oc-tile range within the chunk
  worker_synctoken_t *sync;
} w8a16_gemv_task_t;

// Each worker computes a disjoint range of oc-tiles and stores directly to c
// (non-overlapping 64B regions, cache-coherent CPU stores).
static void w8a16_gemv_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  w8a16_gemv_task_t *s           = (w8a16_gemv_task_t *) data;
  const size_t       wtile_bytes = (size_t) 1024 * s->kp;
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

// Parallel compute over [oy_base, oy_end) oc-tiles of one resident chunk.
static void compute_chunk_parallel(uint8_t *c, const uint8_t *vtcm_weight, const float *vtcm_sw,
                                   const HVX_Vector *a_splat, const uint8_t *bias, int kp, int nblk, float sa,
                                   int oy_base, int oy_end, int chunk_np) {
  int nworkers = (int) g_max_num_workers;
  if (nworkers < 1) {
    nworkers = 1;
  }
  int                nw  = nworkers > chunk_np ? chunk_np : nworkers;
  const int          per = (chunk_np + nw - 1) / nw;
  worker_synctoken_t sync;
  worker_pool_synctoken_init(&sync, nw);
  w8a16_gemv_task_t tasks[nw];
  for (int w = 0; w < nw; ++w) {
    int s0 = w * per;
    int s1 = s0 + per;
    if (s1 > chunk_np) {
      s1 = chunk_np;
    }
    tasks[w] = (w8a16_gemv_task_t){ c + (size_t) oy_base * 32 * sizeof(__fp16),
                                    vtcm_weight,
                                    a_splat,
                                    vtcm_sw,
                                    bias ? bias + (size_t) oy_base * 32 * sizeof(__fp16) : NULL,
                                    kp,
                                    nblk,
                                    sa,
                                    s0,
                                    s1,
                                    &sync };
    worker_pool_job_t job;
    job.fptr = w8a16_gemv_worker_loop;
    job.dptr = &tasks[w];
    if (worker_pool_submit(NULL, job) != 0) {
      w8a16_gemv_worker_loop(&tasks[w], 0);  // pool full: run inline
    }
  }
  worker_pool_synctoken_wait(&sync);
}

// Issue one weight+scale DMA chunk into vtcm_weight/vtcm_sw.
static void start_chunk_dma(dma_desc_1d_t *wdesc, dma_desc_1d_t *sdesc, const uint8_t *b_wt, const uint8_t *b_scale,
                            uint8_t *vtcm_weight, float *vtcm_sw, size_t wtile_bytes, size_t sw_bytes_per_oc,
                            int oy_base, int chunk_np) {
  memset(wdesc, 0, sizeof(*wdesc));
  wdesc->length     = (uint32_t) ((size_t) chunk_np * wtile_bytes);
  wdesc->type       = DMA_DESC_TYPE_1D;
  wdesc->src_bypass = 1;
  wdesc->dst_bypass = 1;
  wdesc->ordered    = 1;
  wdesc->dstate     = DMA_DESC_DSTATE_PENDING;
  wdesc->src        = (uint32_t) (b_wt + (size_t) oy_base * wtile_bytes);
  wdesc->dst        = (uint32_t) vtcm_weight;
  memset(sdesc, 0, sizeof(*sdesc));
  sdesc->length     = (uint32_t) ((size_t) chunk_np * sw_bytes_per_oc);
  sdesc->type       = DMA_DESC_TYPE_1D;
  sdesc->src_bypass = 1;
  sdesc->dst_bypass = 1;
  sdesc->ordered    = 1;
  sdesc->dstate     = DMA_DESC_DSTATE_PENDING;
  sdesc->src        = (uint32_t) (b_scale + (size_t) oy_base * sw_bytes_per_oc);
  sdesc->dst        = (uint32_t) vtcm_sw;
  sdesc->next       = 0;
  wdesc->next       = (uint32_t) sdesc;
  dmstart(wdesc);
}

int hmx_matmulw8a16block_gemv_i8(uint8_t *c, const uint8_t *a, const uint8_t *b_wt, const uint8_t *b_scale,
                                 const uint8_t *bias, int K, int N, int scale_block_num) {
  if (scale_block_num <= 0 || (K % 64) != 0 || (N % 32) != 0) {
    return -1;  // supported: block size 64, K%64==0, N%32==0
  }
  const int kp   = K / 32;
  const int np   = N / 32;
  const int nblk = scale_block_num;
  if ((kp % nblk) != 0) {
    return -1;
  }

  const size_t weight_tile_bytes = (size_t) 1024 * kp;  // per oc-tile (int8 vrmpy layout)
  const size_t sw_bytes_per_oc   = (size_t) nblk * 32 * sizeof(float);
  const int    n_groups          = K / 4;               // 4-k vrmpy groups

  // Whole-blob fast path when it fits in VTCM (single DMA, max overlap).
  const size_t vtcm_total = vtcm_manager_get_vtcm_size();
  const size_t need       = (size_t) K + (size_t) n_groups * sizeof(HVX_Vector) + (size_t) np * sw_bytes_per_oc +
                      (size_t) np * weight_tile_bytes + 4096;
  if (need > vtcm_total) {
    // Chunked path with double-buffered weight DMA: chunk i+1 streams while
    // chunk i computes. Fixed activation/scale workspace plus two chunk copies
    // must fit; vtcm_seq_alloc does no bounds checking, so size conservatively.
    const size_t fixed_bytes = (size_t) K + (size_t) n_groups * sizeof(HVX_Vector) + 4096;
    const size_t per_np      = weight_tile_bytes + sw_bytes_per_oc;
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

    // VTCM layout: qa / a_splat / sw[2] / weight[2]
    uint8_t    *vtcm        = (uint8_t *) vtcm_manager_get_vtcm_base();
    int8_t     *vtcm_qa     = (int8_t *) vtcm_seq_alloc(&vtcm, (size_t) K * sizeof(int8_t));
    HVX_Vector *vtcm_asplat = (HVX_Vector *) vtcm_seq_alloc(&vtcm, (size_t) n_groups * sizeof(HVX_Vector));
    float      *vtcm_sw[2]  = { (float *) vtcm_seq_alloc(&vtcm, (size_t) np_chunk * sw_bytes_per_oc),
                                (float *) vtcm_seq_alloc(&vtcm, (size_t) np_chunk * sw_bytes_per_oc) };
    uint8_t    *vtcm_wt[2]  = { (uint8_t *) vtcm_seq_alloc(&vtcm, (size_t) np_chunk * weight_tile_bytes),
                                (uint8_t *) vtcm_seq_alloc(&vtcm, (size_t) np_chunk * weight_tile_bytes) };

    _Alignas(64) dma_desc_1d_t wdesc[2];
    _Alignas(64) dma_desc_1d_t sdesc[2];

    const int nchunks = (np + np_chunk - 1) / np_chunk;

    // First chunk DMA overlaps activation quantization below.
    start_chunk_dma(&wdesc[0], &sdesc[0], b_wt, b_scale, vtcm_wt[0], vtcm_sw[0], weight_tile_bytes, sw_bytes_per_oc, 0,
                    np_chunk > np ? np : np_chunk);
    float sa = 1.0f;
    quantize_activation_row((const __fp16 *) a, K, vtcm_qa, &sa);
    for (int gg = 0; gg < n_groups; ++gg) {
      vtcm_asplat[gg] = splat_group_permuted(vtcm_qa, gg);
    }
    dma_wait_for_idle();

    for (int chunk = 0; chunk < nchunks; ++chunk) {
      const int oy_base = chunk * np_chunk;
      int       oy_end  = oy_base + np_chunk;
      if (oy_end > np) {
        oy_end = np;
      }
      const int chunk_np = oy_end - oy_base;
      const int buf      = chunk & 1;
      // Prefetch the next chunk into the other buffer before computing this one.
      if (chunk + 1 < nchunks) {
        const int next_oy_base = (chunk + 1) * np_chunk;
        int       next_oy_end  = next_oy_base + np_chunk;
        if (next_oy_end > np) {
          next_oy_end = np;
        }
        start_chunk_dma(&wdesc[1 - buf], &sdesc[1 - buf], b_wt, b_scale, vtcm_wt[1 - buf], vtcm_sw[1 - buf],
                        weight_tile_bytes, sw_bytes_per_oc, next_oy_base, next_oy_end - next_oy_base);
      }
      compute_chunk_parallel(c, vtcm_wt[buf], vtcm_sw[buf], vtcm_asplat, bias, kp, nblk, sa, oy_base, oy_end, chunk_np);
      if (chunk + 1 < nchunks) {
        dma_wait_for_idle();
      }
    }
    return 0;
  }

  uint8_t    *vtcm        = (uint8_t *) vtcm_manager_get_vtcm_base();
  int8_t     *vtcm_qa     = (int8_t *) vtcm_seq_alloc(&vtcm, (size_t) K * sizeof(int8_t));
  HVX_Vector *vtcm_asplat = (HVX_Vector *) vtcm_seq_alloc(&vtcm, (size_t) n_groups * sizeof(HVX_Vector));
  float      *vtcm_sw     = (float *) vtcm_seq_alloc(&vtcm, (size_t) np * sw_bytes_per_oc);
  uint8_t    *vtcm_weight = (uint8_t *) vtcm_seq_alloc(&vtcm, (size_t) np * weight_tile_bytes);

  // DMA descriptors on stack (DDR) — matches the fp16 kernel's proven done-bit
  // polling pattern (the DMA engine updates the dstate bit coherently there).
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

  // 2. quantize activation to symmetric int8 + pre-splat (overlaps the weight DMA).
  float sa = 1.0f;
  quantize_activation_row((const __fp16 *) a, K, vtcm_qa, &sa);
  for (int gg = 0; gg < n_groups; ++gg) {
    vtcm_asplat[gg] = splat_group_permuted(vtcm_qa, gg);
  }

  // 3. wait for the whole weight DMA, then compute (parallel across oc-tiles).
  dma_wait_for_idle();
  compute_chunk_parallel(c, vtcm_weight, vtcm_sw, vtcm_asplat, bias, kp, nblk, sa, 0, np, np);
  return 0;
}
