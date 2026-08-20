#ifndef HTP_OPS_DSP_VRMPY_TO_HMX_H
#define HTP_OPS_DSP_VRMPY_TO_HMX_H

// Path A (perf/hexagon/hexagon_phase2_plan.md Step 2): on-DSP converters that let
// the prefill GEMM read the single vrmpy-layout weight (reorderInt4WeightForVrmpyGemv)
// instead of a duplicate HMX-layout weight. One 512B int4 tile is reordered into the
// HMX layout that dequant_q4_tile_scaled / the vlut16 path consumes, and the vrmpy
// fp32 block scales are repacked into the two HMX fp16 scale regions the prefill
// kernels expect. All three are byte/numerically verified — see
// source/backend/hexagon/tests/test_vrmpy_to_hmx_convert.cpp (host, byte-exact) and
// the device self-test in matmul_q4fp16.c (HVX == scalar).

#include <stdint.h>

#include "dsp/hvx_convert.h"  // HVX types + hvx_my_wsf_to_vhf

// vrmpy int4 tile (512B) -> HMX int4 tile (512B), byte-identical to
// reorderInt4WeightForHmx / htp_ops_weight_reorder_int4_block.
//
// vrmpy tile: 8 groups g of 64B; byte g*64+2*oc+p holds lo=w(oc,4g+2p),
// hi=w(oc,4g+2p+1). A 128B load = groups (2m,2m+1) -> local vectors (2m,2m+1),
// then the verified HMX tail (8x Q6_Vb_vshuff_Vb + 4x nibble pack).
static inline void __attribute__((unused)) vrmpy_tile_to_hmx_int4_512(uint8_t *dst, const uint8_t *src) {
  const HVX_Vector     mask0f    = Q6_Vb_vsplat_R(0x0f);
  const HVX_VectorPred q_first64 = Q6_Q_vsetq_R(64);
  const HVX_Vector    *src_vec   = (const HVX_Vector *) src;
  HVX_Vector           local[8];
  for (int i = 0; i < 4; ++i) {
    HVX_Vector     v   = src_vec[i];                // groups (2i, 2i+1)
    HVX_Vector     vd  = Q6_Vb_vdeal_Vb(v);         // p0 bytes -> low 64, p1 -> high 64
    HVX_Vector     vlo = Q6_V_vand_VV(vd, mask0f);  // low nibbles (even k)
    HVX_Vector     vhi = Q6_Vub_vlsr_VubR(vd, 4);   // high nibbles (odd k)
    HVX_VectorPair P   = Q6_W_vshuff_VVR(vhi, vlo, -1);
    HVX_Vector     Plo = Q6_V_lo_W(P);
    HVX_Vector     Phi = Q6_V_hi_W(P);
    local[2 * i]       = Q6_V_vmux_QVV(q_first64, Plo, Q6_V_vror_VR(Phi, 64));
    local[2 * i + 1]   = Q6_V_vmux_QVV(q_first64, Q6_V_vror_VR(Plo, 64), Phi);
  }
  for (int q = 0; q < 8; ++q) {
    local[q] = Q6_Vb_vshuff_Vb(local[q]);
  }
  HVX_Vector *dst_vec = (HVX_Vector *) dst;
  for (int q = 0; q < 4; ++q) {
    HVX_Vector v_shifted = Q6_Vh_vasl_VhR(local[2 * q + 1], 4);
    dst_vec[q]           = Q6_V_vor_VV(local[2 * q], v_shifted);
  }
}

// vrmpy fp32 block scales -> HMX fp16 scale, per oc-tile.
//
// vrmpy scale layout: entry (y, b) starts at sw + (y*nblk + b)*VRMPY_SCALE_ENTRY_FLOATS and holds
// 32 fp32 alpha(o=y*32+ocIn, b) in oc order, NOT duplicated. When the layer is asymmetric the entry
// is twice as wide and carries 32 fp32 qbias(o, b) right after the scale, so one DMA covers both.
// sw_oy_tile points at the oc-tile base (sw + y*nblk*entry_floats).
// hvx_my_wsf_to_vhf(v1, v0) yields 64 fp16 with out[2i]=f16(v0[i]), out[2i+1]=f16(v1[i]).
#define VRMPY_SCALE_ENTRY_FLOATS(asym) ((asym) ? 64 : 32)
#define VRMPY_QBIAS_OFFSET_FLOATS      32

// dup region (hmx_matmulq4fp16 / dequant_q4_tile_scaled vBlockScale):
// 64 lanes, lane 2*ocIn = lane 2*ocIn+1 = f16(alpha(o, block)).
static inline HVX_Vector __attribute__((unused)) vrmpy_scale_block_dup_hf(const float *sw_oy_tile, int block,
                                                                          int entry_floats) {
  HVX_Vector sf = *(const HVX_Vector *) (sw_oy_tile + (size_t) block * entry_floats);
  return hvx_my_wsf_to_vhf(sf, sf);
}

// Same shape, reading the qbias half of an asymmetric entry. Only valid when entry_floats == 64.
static inline HVX_Vector __attribute__((unused)) vrmpy_qbias_block_dup_hf(const float *sw_oy_tile, int block,
                                                                          int entry_floats) {
  HVX_Vector sf = *(const HVX_Vector *) (sw_oy_tile + (size_t) block * entry_floats + VRMPY_QBIAS_OFFSET_FLOATS);
  return hvx_my_wsf_to_vhf(sf, sf);
}

// packed region (mle32 accumulate): 64 lanes, lane 2*ocIn = f16(alpha(o,2p)),
// lane 2*ocIn+1 = f16(alpha(o,2p+1)); odd tail (2p+1 >= nblk) -> 0.
static inline HVX_Vector __attribute__((unused)) vrmpy_scale_pair_packed_hf(const float *sw_oy_tile, int nblk, int pair,
                                                                            int entry_floats) {
  HVX_Vector sf0 = *(const HVX_Vector *) (sw_oy_tile + (size_t) (2 * pair) * entry_floats);
  HVX_Vector sf1 =
    (2 * pair + 1 < nblk) ? *(const HVX_Vector *) (sw_oy_tile + (size_t) (2 * pair + 1) * entry_floats) : Q6_V_vzero();
  return hvx_my_wsf_to_vhf(sf1, sf0);
}

#endif  // HTP_OPS_DSP_VRMPY_TO_HMX_H
