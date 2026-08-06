#ifndef MNN_HEXAGON_DSP_PWL_H
#define MNN_HEXAGON_DSP_PWL_H

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <stdint.h>

#ifndef HTP_OPS_PWL_COMPANDED16
#  define HTP_OPS_PWL_COMPANDED16 1
#endif

#ifndef HTP_OPS_PWL_LEARNED8
#  define HTP_OPS_PWL_LEARNED8 1
#endif

#if HTP_OPS_PWL_LEARNED8
// A hardware-constrained search selects eight magnitude intervals. A vlut16
// maps compressed FP16 exponent/mantissa states to segments before the two
// coefficient lookups.
extern const uint32_t htp_ops_silu_learned_index_lut[32];
extern const uint32_t htp_ops_silu_learned_slope[32];
extern const uint32_t htp_ops_silu_learned_bias[32];
#elif HTP_OPS_PWL_COMPANDED16
// A single 16-entry bank covers [0, 8] with segment widths 0.25, 0.5,
// and 1.0 over [0, 2], [2, 4], and [4, 8], respectively.
extern const uint32_t htp_ops_silu_slope[32];
extern const uint32_t htp_ops_silu_bias[32];
#else
// SiLU uses 0.25-wide FP16 PWL segments over [0, 8]. Two 16-entry
// vlut16 banks cover [0, 4] and [4, 8]. Keep a single copy of the tables in
// pwl.cc while inlining the vector arithmetic into each operator kernel.
extern const uint32_t htp_ops_silu_slope_lo[32];
extern const uint32_t htp_ops_silu_bias_lo[32];
extern const uint32_t htp_ops_silu_slope_hi[32];
extern const uint32_t htp_ops_silu_bias_hi[32];
#endif

static inline HVX_Vector htp_ops_pwl_lookup16(HVX_Vector index, const uint32_t *table) {
  index                     = Q6_V_vand_VV(index, Q6_Vh_vsplat_R(0x000f));
  // vlut16 consumes both bytes of each input halfword. Duplicate the index so
  // either result vector retains all 64 FP16 lanes.
  HVX_Vector     byte_index = Q6_V_vor_VV(index, Q6_Vw_vasl_VwR(index, 8));
  HVX_VectorPair table_pair = Q6_Wh_vlut16_VbVhR_nomatch(byte_index, *((const HVX_Vector *) table), 0);
  return Q6_V_lo_W(table_pair);
}

static inline HVX_Vector htp_ops_pwl_index16(HVX_Vector x, uint16_t scale_bits) {
  HVX_Vector       scaled      = Q6_Vhf_vmpy_VhfVhf(x, Q6_Vh_vsplat_R(scale_bits));
  // FP16 may round a value just below 16 to exactly 16. Clamp before masking
  // so the last segment cannot wrap back to segment zero.
  const HVX_Vector max_index_v = Q6_Vh_vsplat_R(0x4b80);                              // 15.0
  scaled                       = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(scaled, max_index_v), max_index_v, scaled);
  scaled                       = Q6_Vhf_vadd_VhfVhf(scaled, Q6_Vh_vsplat_R(0x4c00));  // +16.0
  return Q6_Vuh_vlsr_VuhR(scaled, 6);
}

#if HTP_OPS_PWL_COMPANDED16
static inline HVX_Vector htp_ops_pwl_companded_index16(HVX_Vector x) {
  // Below 2, keep the 0.25-wide uniform index. In [2, 8), FP16 values
  // have exponents 16 or 17; exponent bit 0 and the top two mantissa bits
  // directly encode the eight wider segments:
  //   [2, 4): 8 + top2(mantissa)
  //   [4, 8): 12 + top2(mantissa)
  const HVX_Vector two_v      = Q6_Vh_vsplat_R(0x4000);
  const HVX_Vector index_low  = htp_ops_pwl_index16(x, 0x4400);
  HVX_Vector       index_wide = Q6_Vuh_vlsr_VuhR(x, 8);
  index_wide                  = Q6_V_vand_VV(index_wide, Q6_Vh_vsplat_R(0x0007));
  index_wide                  = Q6_Vh_vadd_VhVh(index_wide, Q6_Vh_vsplat_R(8));
  const HVX_VectorPred wide   = Q6_Q_not_Q(Q6_Q_vcmp_gt_VhfVhf(two_v, x));
  return Q6_V_vmux_QVV(wide, index_wide, index_low);
}
#endif

#if HTP_OPS_PWL_LEARNED8
static inline HVX_Vector htp_ops_pwl_learned_index8(HVX_Vector abs_v) {
  const HVX_Vector     zero_v    = Q6_V_vzero();
  const HVX_Vector     bit_state = Q6_Vuh_vlsr_VuhR(abs_v, 8);
  const HVX_Vector     raw_state = Q6_Vh_vsub_VhVh(bit_state, Q6_Vh_vsplat_R(48));
  const HVX_VectorPred has_state = Q6_Q_vcmp_gt_VhVh(bit_state, Q6_Vh_vsplat_R(48));
  const HVX_Vector     clamped   = Q6_V_vmux_QVV(has_state, raw_state, zero_v);

  const HVX_Vector     narrow_state = Q6_Vuh_vlsr_VuhR(clamped, 1);
  const HVX_Vector     wide_state   = Q6_Vh_vsub_VhVh(clamped, Q6_Vh_vsplat_R(8));
  const HVX_VectorPred wide         = Q6_Q_vcmp_gt_VhVh(clamped, Q6_Vh_vsplat_R(15));
  const HVX_Vector     state        = Q6_V_vmux_QVV(wide, wide_state, narrow_state);
  return htp_ops_pwl_lookup16(state, htp_ops_silu_learned_index_lut);
}
#endif

static inline HVX_Vector htp_ops_pwl_eval(HVX_Vector x, HVX_Vector slope, HVX_Vector bias) {
  HVX_Vector product = Q6_Vqf16_vmpy_VhfVhf(x, slope);
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(product, bias));
}

static inline HVX_Vector htp_ops_silu_pwl_fp16_vec(HVX_Vector v) {
  const HVX_Vector zero_v  = Q6_V_vzero();
  const HVX_Vector eight_v = Q6_Vh_vsplat_R(0x4800);
#if HTP_OPS_PWL_LEARNED8
  const HVX_VectorPred negative = Q6_Q_vcmp_gt_VhfVhf(zero_v, v);
  const HVX_Vector     abs_v    = Q6_V_vand_VV(v, Q6_Vh_vsplat_R(0x7fff));
  const HVX_Vector     index    = htp_ops_pwl_learned_index8(abs_v);
  const HVX_Vector     slope    = htp_ops_pwl_lookup16(index, htp_ops_silu_learned_slope);
  const HVX_Vector     bias     = htp_ops_pwl_lookup16(index, htp_ops_silu_learned_bias);

  const HVX_Vector     positive_y = htp_ops_pwl_eval(abs_v, slope, bias);
  const HVX_Vector     negative_y = Q6_Vhf_vsub_VhfVhf(positive_y, abs_v);
  const HVX_Vector     result     = Q6_V_vmux_QVV(negative, negative_y, positive_y);
  const HVX_VectorPred saturated  = Q6_Q_not_Q(Q6_Q_vcmp_gt_VhfVhf(eight_v, abs_v));
  const HVX_Vector     limit      = Q6_V_vmux_QVV(negative, zero_v, abs_v);
  return Q6_V_vmux_QVV(saturated, limit, result);
#else
#  if !HTP_OPS_PWL_COMPANDED16
  const HVX_Vector four_v = Q6_Vh_vsplat_R(0x4400);
#  endif

  const HVX_VectorPred negative = Q6_Q_vcmp_gt_VhfVhf(zero_v, v);
  const HVX_Vector     abs_v    = Q6_V_vmux_QVV(negative, Q6_Vhf_vsub_VhfVhf(zero_v, v), v);
#  if HTP_OPS_PWL_COMPANDED16
  const HVX_Vector index = htp_ops_pwl_companded_index16(abs_v);
  const HVX_Vector slope = htp_ops_pwl_lookup16(index, htp_ops_silu_slope);
  const HVX_Vector bias  = htp_ops_pwl_lookup16(index, htp_ops_silu_bias);
#  else
  const HVX_VectorPred high_bank = Q6_Q_not_Q(Q6_Q_vcmp_gt_VhfVhf(four_v, abs_v));
  const HVX_Vector     local_x   = Q6_V_vmux_QVV(high_bank, Q6_Vhf_vsub_VhfVhf(abs_v, four_v), abs_v);
  const HVX_Vector     index     = htp_ops_pwl_index16(local_x, 0x4400);

  HVX_Vector slope = htp_ops_pwl_lookup16(index, htp_ops_silu_slope_lo);
  HVX_Vector bias  = htp_ops_pwl_lookup16(index, htp_ops_silu_bias_lo);
  slope            = Q6_V_vmux_QVV(high_bank, htp_ops_pwl_lookup16(index, htp_ops_silu_slope_hi), slope);
  bias             = Q6_V_vmux_QVV(high_bank, htp_ops_pwl_lookup16(index, htp_ops_silu_bias_hi), bias);
#  endif

  const HVX_Vector positive_y = htp_ops_pwl_eval(abs_v, slope, bias);
  const HVX_Vector negative_y = Q6_Vhf_vsub_VhfVhf(positive_y, abs_v);  // SiLU(-x) = SiLU(x) - x
  HVX_Vector       result     = Q6_V_vmux_QVV(negative, negative_y, positive_y);

  const HVX_VectorPred saturated = Q6_Q_not_Q(Q6_Q_vcmp_gt_VhfVhf(eight_v, abs_v));
  const HVX_Vector     limit     = Q6_V_vmux_QVV(negative, zero_v, abs_v);
  return Q6_V_vmux_QVV(saturated, limit, result);
#endif
}

#endif  // MNN_HEXAGON_DSP_PWL_H
