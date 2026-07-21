#pragma once

#include <stdint.h>

#include "dsp/hvx_utils.h"

// adapted from library qhmath_hvx
// NOTE: This function is intended to use in safe softmax (x <= 0, overflow handing is omitted)
static HVX_INLINE_ALWAYS HVX_Vector hvx_my_exp2_vhf(HVX_Vector x_v) {
  const uint16_t e5_qf16 = 0x5082;  // ori: 0.000153534, actual: 0.000153661
  const uint16_t e4_hf   = 0x157d;  // ori: 0.00133989
  const uint16_t e3_hf   = 0x20ed;  // ori: 0.00961844
  const uint16_t e2_hf   = 0x2b1b;  // ori: 0.0555033
  const uint16_t e1_hf   = 0x33b0;  // ori: 0.240226
  const uint16_t e0_hf   = 0x398c;  // ori: 0.693147

  const HVX_Vector zero_v    = Q6_V_vzero();
  const HVX_Vector half_hf_v = Q6_Vh_vsplat_R(0x3800);  // fp16: 0.5

  HVX_Vector f_v, k_v, y_v, x_qf16_v;
  HVX_Vector e5_qf16_v, e4_hf_v, e3_hf_v, e2_hf_v, e1_hf_v, e0_hf_v, one_hf_v;

  // round to int
  //    k = (int) x;
  //    f = (float) k;
  //        k = Q6_R_convert_sf2w_R(x); // f = floorf(x + 0.5);
  //        f = Q6_R_convert_w2sf_R(k); // k = (int) f;
#if __HVX_ARCH__ >= 73
  // NOTE(hzx): Q6_Vh_equals_Vhf exhibits round-to-zero behavior
  // subtract 0.5 since we assume the inputs are negative
  HVX_Vector x_minus_half_v = Q6_Vqf16_vsub_VhfVhf(x_v, half_hf_v);
  x_minus_half_v            = Q6_Vhf_equals_Vqf16(x_minus_half_v);

  k_v = Q6_Vh_equals_Vhf(x_minus_half_v);
  f_v = Q6_Vhf_equals_Vh(k_v);
#else
  HVX_Vector x_plus_half_v = Q6_Vqf16_vadd_VhfVhf(x_v, half_hf_v);
  x_plus_half_v            = Q6_Vhf_equals_Vqf16(x_plus_half_v);

  // NOTE(hzx): Q6_Vh_vfloor_VhfVhf is defined in hvx_utils.h
  // The comment says that it's deprecated, but I think it has less instructions than
  // that of the combination of qhmath_hvx_vhf_floor_vhf and qhmath_hvx_vh_truncate_vhf
  // (which is used in qhmath_hvx_exp_vhf in qhmath_hvx_vector.h)
  k_v = Q6_Vh_vfloor_VhfVhf(x_plus_half_v, &f_v);
  // x_qf16_v = Q6_Vqf16_vadd_VhfVhf(x_v, zero_v);
  // x_qf16_v = Q6_Vqf16_vsub_Vqf16Vhf(x_qf16_v, f_v); // x = x - f;

  // NOTE(hzx): This is weird. why don't we add 0.5 before doing floor?
  // //    if (x > 0.5)
  // x_v = Q6_Vhf_equals_Vqf16(x_qf16_v);
  // QLarger = Q6_Q_vcmp_gt_VhfVhf(x_v, half_hf);
  //     // k += 1;
  // k_v =  Q6_Vh_condacc_QVhVh(QLarger, k_v, one_h);
  //     // x -= 1.0;
  // temp_qf16_v = Q6_Vqf16_vsub_Vqf16Vhf(x_qf16_v, one_hf);
  // x_qf16_v =  Q6_V_vmux_QVV(QLarger, temp_qf16_v, x_qf16_v);
#endif

  x_qf16_v = Q6_Vqf16_vsub_VhfVhf(x_v, f_v);  // x = x - f;

  //    y = E4 + E5 * x;
  e5_qf16_v = Q6_Vh_vsplat_R(e5_qf16);
  y_v       = Q6_Vqf16_vmpy_Vqf16Vqf16(e5_qf16_v, x_qf16_v);
  e4_hf_v   = Q6_Vh_vsplat_R(e4_hf);
  y_v       = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e4_hf_v);

  //    y = E3 + y * x;
  e3_hf_v = Q6_Vh_vsplat_R(e3_hf);
  y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e3_hf_v);

  //    y = E2 + y * x;
  e2_hf_v = Q6_Vh_vsplat_R(e2_hf);
  y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e2_hf_v);

  //    y = E1 + y * x;
  e1_hf_v = Q6_Vh_vsplat_R(e1_hf);
  y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e1_hf_v);

  //    y = E0 + y * x;
  e0_hf_v = Q6_Vh_vsplat_R(e0_hf);
  y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e0_hf_v);

  //    y = y * x + 1.0;
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  one_hf_v = Q6_Vh_vsplat_R(0x3c00);  // 0x3c00 --> fp16 1.0
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, one_hf_v);

  // insert exponents
  //        y = ldexpf(y, k);
  //    y_v += k_v; // qf32
  // modify exponent
  y_v = Q6_Vhf_equals_Vqf16(y_v);

  // add k_v to the exponent of y_v
  HVX_Vector y_v_exponent = Q6_Vh_vasl_VhR(y_v, 1);              // shift away sign bit
  y_v_exponent            = Q6_Vuh_vlsr_VuhR(y_v_exponent, 11);  // shift back by sign bit + 10-bit mantissa
  y_v_exponent            = Q6_Vh_vadd_VhVh(k_v, y_v_exponent);

  // exponent cannot be negative; if underflow detected, result is set to zero
  HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VhVh(zero_v, y_v_exponent);

  // NOTE(hzx): We don't expect exp to cause overflow in safe softmax, so overflow detection is omitted
  // // max IEEE hf exponent; if overflow detected, result is set to infinity
  // HVX_Vector exp_max_v = Q6_Vh_vsplat_R(0x1e);
  // // float16's 65504.0 is 0x7C00
  // HVX_Vector inf_v = Q6_Vh_vsplat_R(0x7C00);
  // HVX_VectorPred qy_v_overflow_exponent = Q6_Q_vcmp_gt_VhVh(y_v_exponent, exp_max_v);

  // update exponent
  y_v = Q6_Vh_vaslacc_VhVhR(y_v, k_v, 10);

  // clip to min/max values
  y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);
  // y_v = Q6_V_vmux_QVV(qy_v_overflow_exponent, inf_v, y_v);
  return y_v;
}

// adapted from libs/qhl_hvx/src/qhmath_hvx/qhmath_hvx_log2_ahf.c
static HVX_INLINE_ALWAYS HVX_Vector hvx_my_log2_vqf16_vhf(HVX_Vector x_v) {
  const uint16_t sqrt_half_hf = 0x39a8;  // 0.707107 sqrt(2)/2
  const uint16_t log2e_m1_hf  = 0x3715;  // 0.442695 log2(e)-1

  const uint16_t e9_hf = 0x2C81;         //  0.0703768
  const uint16_t e8_hf = 0xAF5F;         // -0.115146
  const uint16_t e7_hf = 0x2F79;         //  0.11677
  const uint16_t e6_hf = 0xAFF3;         // -0.124201
  const uint16_t e5_hf = 0x308F;         //  0.142493
  const uint16_t e4_hf = 0xB155;         //  -0.166681
  const uint16_t e3_hf = 0x3266;         //  0.200007
  const uint16_t e2_hf = 0xB400;         //  -0.25
  const uint16_t e1_hf = 0x3555;         //  0.333333
  const uint16_t e0_hf = 0xB800;         //  -0.5

  const HVX_Vector zero_v = Q6_V_vzero();

  HVX_Vector e_v, y_v, x_qf16_v, z_qf16_v, tmp_v;
  HVX_Vector e9_hf_v, e8_hf_v, e7_hf_v, e6_hf_v, e5_hf_v, e4_hf_v, e3_hf_v, e2_hf_v, e1_hf_v, e0_hf_v;

  // frexp(x): obtain exponent of x
  e_v = Q6_Vuh_vlsr_VuhR(x_v, 10);                 // x >> 10
  e_v = Q6_V_vand_VV(e_v, Q6_Vh_vsplat_R(0x1f));   // only keep 5-bits exponent
  e_v = Q6_Vh_vsub_VhVh(e_v, Q6_Vh_vsplat_R(14));  // minus offset 14

  // calculate frac part
  x_v = Q6_V_vand_VV(x_v, Q6_Vh_vsplat_R(0x83ff));
  x_v = Q6_V_vor_VV(x_v, Q6_Vh_vsplat_R(0x3800));

  // if (sqrt_half > x) {
  //    e = e - 1;
  //    x = 2.0 * x;
  // }
  HVX_VectorPred q = Q6_Q_vcmp_gt_VhfVhf(Q6_Vh_vsplat_R(sqrt_half_hf), x_v);

  HVX_Vector tmp_e_v = Q6_Vh_vsub_VhVh(e_v, Q6_Vh_vsplat_R(1));  // e = e - 1
  e_v                = Q6_V_vmux_QVV(q, tmp_e_v, e_v);

  HVX_Vector tmp_x_v = Q6_Vqf16_vmpy_VhfVhf(x_v, Q6_Vh_vsplat_R(0x4000));  // 2.0 (fp16)
  x_qf16_v           = Q6_Vqf16_vadd_VhfVhf(x_v, zero_v);
  x_qf16_v           = Q6_V_vmux_QVV(q, tmp_x_v, x_qf16_v);

  // compute log(1+x) via polynomial approximation
  // NOTE(hzx): computing log via taylor expansion needs more terms than exp does
  //   to reach similar precision
  //    x = x - 1.0;
  x_qf16_v = Q6_Vqf16_vsub_Vqf16Vhf(x_qf16_v, Q6_Vh_vsplat_R(0x3c00));  // 1.0 (fp16)
  //    z = x * x;
  z_qf16_v = Q6_Vqf16_vmpy_Vqf16Vqf16(x_qf16_v, x_qf16_v);
  x_v      = Q6_Vhf_equals_Vqf16(x_qf16_v);
  //    y = E8 + E9 * x;
  e9_hf_v  = Q6_Vh_vsplat_R(e9_hf);
  y_v      = Q6_Vqf16_vmpy_VhfVhf(e9_hf_v, x_v);
  e8_hf_v  = Q6_Vh_vsplat_R(e8_hf);
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e8_hf_v);
  //    y = E7 + y * x;
  e7_hf_v  = Q6_Vh_vsplat_R(e7_hf);
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e7_hf_v);
  //    y = E6 + y * x;
  e6_hf_v  = Q6_Vh_vsplat_R(e6_hf);
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e6_hf_v);
  //    y = E5 + y * x;
  e5_hf_v  = Q6_Vh_vsplat_R(e5_hf);
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e5_hf_v);
  //    y = E4 + y * x;
  e4_hf_v  = Q6_Vh_vsplat_R(e4_hf);
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e4_hf_v);
  //    y = E3 + y * x;
  e3_hf_v  = Q6_Vh_vsplat_R(e3_hf);
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e3_hf_v);
  //    y = E2 + y * x;
  e2_hf_v  = Q6_Vh_vsplat_R(e2_hf);
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e2_hf_v);
  //    y = E1 + y * x;
  e1_hf_v  = Q6_Vh_vsplat_R(e1_hf);
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v      = Q6_Vqf16_vadd_Vqf16Vhf(y_v, e1_hf_v);
  //    y = y * x * z;
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
  y_v      = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, z_qf16_v);
  //    y += E0 * z;
  e0_hf_v  = Q6_Vh_vsplat_R(e0_hf);
  tmp_v    = Q6_Vqf16_vadd_VhfVhf(e0_hf_v, zero_v);
  tmp_v    = Q6_Vqf16_vmpy_Vqf16Vqf16(tmp_v, z_qf16_v);
  y_v      = Q6_Vqf16_vadd_Vqf16Vqf16(y_v, tmp_v);

  // z = (x + y) * log2(e) + (float) e
  // NOTE(hzx): the original code here accumulate y*(log2(e)-1), x*(log2(e)-1), y, x
  // separately. I assume separate accumulation preserves better precision.
  HVX_Vector log2e_m1_hf_v   = Q6_Vh_vsplat_R(log2e_m1_hf);
  HVX_Vector log2e_m1_qf16_v = Q6_Vqf16_vadd_VhfVhf(log2e_m1_hf_v, zero_v);

  z_qf16_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, log2e_m1_qf16_v);  // z = y * (log2(e)-1);
  tmp_v    = Q6_Vqf16_vmpy_Vqf16Vqf16(x_qf16_v, log2e_m1_qf16_v);
  z_qf16_v = Q6_Vqf16_vadd_Vqf16Vqf16(z_qf16_v, tmp_v);       // z += x * (log2(e)-1);
  z_qf16_v = Q6_Vqf16_vadd_Vqf16Vqf16(z_qf16_v, y_v);         // z += y;
  z_qf16_v = Q6_Vqf16_vadd_Vqf16Vqf16(z_qf16_v, x_qf16_v);    // z += x;

  HVX_Vector qf16e_v = vqf16_from_int(e_v);
  z_qf16_v           = Q6_Vqf16_vadd_Vqf16Vqf16(z_qf16_v, qf16e_v);  // z += (float) e;
  return z_qf16_v;
}

// adapted from libs/qhl_hvx/src/qhmath_hvx/qhmath_hvx_inv_ahf.c
static HVX_INLINE_ALWAYS HVX_Vector hvx_my_inv_vhf(HVX_Vector x_v) {
  /*
   * Polynomial coefficients packed in specific format (adding zeros on some places)
   * in order to easier manipulate with them later using VLUT instructions
   */
  static const float c0_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.8807721943716516,
    3.6618209528616856,
    3.4657742282097708,
    3.2853461610022414,
    3.1229570908314015,
    2.976379865829892,
    2.8438614274889833,
    2.723793061029549,
    2.613859154046634,
    2.5119508509784287,
    2.4167270706641473,
    2.3286721812015188,
    2.2462659531748064,
    2.1692490555028736,
    2.0981551828382417,
    2.0319234960945,
  };
  static const float c1_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -5.646783581176797,
    -5.027704168781284,
    -4.5037889029173535,
    -4.0470997487793445,
    -3.6569569537789364,
    -3.3217563552211695,
    -3.03258650196419,
    -2.781935505534812,
    -2.5619261358961922,
    -2.3660577978107398,
    -2.190083163030879,
    -2.033405493468989,
    -1.8920413948588666,
    -1.7645298754188785,
    -1.6507730169513504,
    -1.5482028127706613,
  };
  static const float c2_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.6511964773849632,
    3.0676375988553106,
    2.6008750952258324,
    2.215514199159397,
    1.9030391013295935,
    1.6474963735373633,
    1.4371447652517673,
    1.2627141904289978,
    1.11593649827749,
    0.9904415490260164,
    0.882033772823834,
    0.7891019704346331,
    0.7082630629776306,
    0.6378888508693012,
    0.5772121720355701,
    0.524261196551401,
  };
  static const float c3_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.8851851956149304,
    -0.7018008948429424,
    -0.5631686602024177,
    -0.4547647803673564,
    -0.37133287830029976,
    -0.3063883382130307,
    -0.255378412302572,
    -0.2149126167280633,
    -0.18226975346347984,
    -0.15546600267845986,
    -0.13320337246909697,
    -0.11482846255803722,
    -0.0994184164975366,
    -0.08647114157420362,
    -0.07568254923048714,
    -0.06657033258736733,
  };

  HVX_DV         c0_coeff_dv, c1_coeff_dv, c2_coeff_dv, c3_coeff_dv, output_dv;
  HVX_VectorPair c0_coeff_vp, c1_coeff_vp, c2_coeff_vp, c3_coeff_vp;

  /*
   * Splat scale factor in order to be used later for finding indexes of coefficients.
   * Scale factor is represented in IEEE 16-bit floating-point format and it is
   * calculated using the following formula:
   *    scale_factor = (convert_sf_to_hf) (16.0 / (b0 - a0))
   * NOTE: Calculated value is slightly decreased in order to avoid out of bound
   *       indexes during VLUT lookup.
   */
  HVX_Vector scale_v = Q6_Vh_vsplat_R(0x4bfb);

  /* Vector of ones used as mpy neutral element in conversions from hf vector to qf32 vector pair */
  HVX_Vector one_v_hf = Q6_Vh_vsplat_R(0x3c00);

  /*
   * Vector of zeroes used as neutral element in hf to qf16 conversions.
   * NOTE: Some of conversions (i.e conversion of scale factor and coefficients)
   *       can be avoided in real-time, but this is not done in order to don't
   *       sacrify code readibility in expense of insignificant performance improvement.
   */
  HVX_Vector zero_v_hf = Q6_V_vzero();

  /* Set sign = 0, exp = 30, mant = 0 */
  HVX_Vector exp = Q6_Vh_vsplat_R(0x7800);

  /* Set mask for sign and exponent */
  HVX_Vector signexp_mask = Q6_Vh_vsplat_R(0xFC00);

  /* Mask for extracting only 4 bits of mantissa */
  HVX_Vector mask_idx1_v = Q6_Vh_vsplat_R(0x000F);
  HVX_Vector mask_idx2_v = Q6_V_vsplat_R(0x00001010);

  /* 16.0 in IEEE 16-bit floating-point representation */
  HVX_Vector const16_0_v_hf = Q6_Vh_vsplat_R(0x4c00);

  /*
   * Prepare vector of input_min values, that is used later in shifting input range.
   * input_min is low boundary of specified input range.
   */
  HVX_Vector input_min_v_hf = Q6_Vh_vsplat_R(0x3c00);

  /* Convert scale factor from hf to q16. Use the same vector for both formats */
  scale_v = Q6_Vqf16_vadd_VhfVhf(scale_v, zero_v_hf);

  /* Load coefficients */
  HVX_Vector c0_coeff_v = *((HVX_Vector *) (c0_coeffs));
  HVX_Vector c1_coeff_v = *((HVX_Vector *) (c1_coeffs));
  HVX_Vector c2_coeff_v = *((HVX_Vector *) (c2_coeffs));
  HVX_Vector c3_coeff_v = *((HVX_Vector *) (c3_coeffs));

  /* Convert coefficients from hf to qf32 format. Use the same vector for both representations */
  c0_coeff_v = Q6_Vqf32_vadd_VsfVsf(c0_coeff_v, zero_v_hf);
  c1_coeff_v = Q6_Vqf32_vadd_VsfVsf(c1_coeff_v, zero_v_hf);
  c2_coeff_v = Q6_Vqf32_vadd_VsfVsf(c2_coeff_v, zero_v_hf);
  c3_coeff_v = Q6_Vqf32_vadd_VsfVsf(c3_coeff_v, zero_v_hf);

  /* Split 32-bit coefficients to lower and upper part in order to obtain them later with VLUT16. */
  c0_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
  c1_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
  c2_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
  c3_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c3_coeff_v);

  /* Calculate normalization factor */
  HVX_Vector norm_factor = Q6_V_vand_VV(x_v, signexp_mask);
  norm_factor            = Q6_Vh_vsub_VhVh(exp, norm_factor);

  /* Normalize input */
  x_v = Q6_Vqf16_vmpy_VhfVhf(x_v, norm_factor);

  /* Convert normalization factor to qf32 */
  HVX_VectorPair norm_factor_qf32 = Q6_Wqf32_vmpy_VhfVhf(norm_factor, one_v_hf);

  /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
  HVX_Vector tmp_v              = Q6_Vh_vdeal_Vh(x_v);
  HVX_Vector input_shifted_v_hf = Q6_Vqf16_vsub_Vqf16Vhf(tmp_v, input_min_v_hf);

  /*
     * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
     * in order to get corresponding coefficient indexes
     */
  HVX_Vector input_scaled_v = Q6_Vqf16_vmpy_Vqf16Vqf16(input_shifted_v_hf, scale_v);

  /*
     * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
     * to [16.0,32.0) in order to convert float indexes to integer values.
     * Float values, represented in IEEE 754, in range [16.0,32.0] have the
     * same exponent, which means 4 MSB of mantissa carry information about
     * integer index.
     */
  /* Use the same input_scaled_v vector for hf and qf16 representation */
  input_scaled_v = Q6_Vqf16_vadd_Vqf16Vhf(input_scaled_v, const16_0_v_hf);

  /* Convert back from qf16 to hf in order to extract integer index  */
  tmp_v = Q6_Vhf_equals_Vqf16(input_scaled_v);

  /* Only 4 MSB bits of mantissa represent segment index */
  HVX_Vector idx1_v = Q6_Vuh_vlsr_VuhR(tmp_v, 6);

  /* Ensure only 4 MSB bits of mantissa are used as indexes */
  idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
  idx1_v = Q6_Vb_vshuff_Vb(idx1_v);
  idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);

  HVX_Vector idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

  /* Obtain the polynomial coefficients from lookup table */
  c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
  c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
  c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
  c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
  c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
  c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
  c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
  c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);

  /* Convert input from hf vector to qf32 vector pair for Horner's method*/
  HVX_VectorPair input_vp_qf32 = Q6_Wqf32_vmpy_Vqf16Vhf(x_v, one_v_hf);

  /* Perform evaluation of polynomial using Horner's method */
  output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c3_coeff_vp), Q6_V_lo_W(input_vp_qf32));
  output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c2_coeff_vp));
  output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
  output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c1_coeff_vp));
  output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
  output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c0_coeff_vp));

  output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(c3_coeff_vp), Q6_V_hi_W(input_vp_qf32));
  output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c2_coeff_vp));
  output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
  output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c1_coeff_vp));
  output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
  output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c0_coeff_vp));

  /* Multiply result by same normalization factor applied to input earlier */
  output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(norm_factor_qf32));
  output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(norm_factor_qf32));

  /* Convert from qf32 to hf */
  return Q6_Vhf_equals_Wqf32(output_dv.VV);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_my_exp2_vhf_vqf16(HVX_Vector x) {
  return hvx_my_exp2_vhf(Q6_Vhf_equals_Vqf16(x));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_my_log2_vqf16(HVX_Vector x) {
  return hvx_my_log2_vqf16_vhf(Q6_Vhf_equals_Vqf16(x));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_my_exp2_vsf(HVX_Vector x_v) {
  const uint32_t e5_sf = 0x3920FDDE;  // ori: 0.000153534
  const uint32_t e4_sf = 0x3AAF9F29;  // ori: 0.00133989
  const uint32_t e3_sf = 0x3C1D96A6;  // ori: 0.00961844
  const uint32_t e2_sf = 0x3D635774;  // ori: 0.0555033
  const uint32_t e1_sf = 0x3E75FDEE;  // ori: 0.240226
  const uint32_t e0_sf = 0x3F317218;  // ori: 0.693147

  const HVX_Vector zero_v    = Q6_V_vzero();
  const HVX_Vector half_sf_v = Q6_V_vsplat_R(0x3F000000);  // fp32: 0.5
  const HVX_Vector one_sf_v  = Q6_V_vsplat_R(0x3F800000);  // fp32: 1.0

  HVX_Vector f_v, k_v, y_v, x_qf32_v;
  HVX_Vector e5_sf_v, e4_sf_v, e3_sf_v, e2_sf_v, e1_sf_v, e0_sf_v;

  // round to int
  //    k = (int) x;
  //    f = (float) k;
  //        k = Q6_R_convert_sf2w_R(x); // f = floorf(x + 0.5);
  //        f = Q6_R_convert_w2sf_R(k); // k = (int) f;
#if __HVX_ARCH__ >= 73
  // NOTE(hzx): Q6_Vh_equals_Vhf exhibits round-to-zero behavior
  // subtract 0.5 since we assume the inputs are negative
  HVX_Vector x_minus_half_v = Q6_Vqf32_vsub_VsfVsf(x_v, half_sf_v);
  x_minus_half_v            = Q6_Vsf_equals_Vqf32(x_minus_half_v);

  k_v = Q6_Vw_equals_Vsf(x_minus_half_v);
  f_v = Q6_Vsf_equals_Vw(k_v);
#else
  HVX_Vector x_plus_half_v = Q6_Vqf32_vadd_VsfVsf(x_v, half_sf_v);
  x_plus_half_v            = Q6_Vsf_equals_Vqf32(x_plus_half_v);

  k_v = Q6_Vw_vfloor_VsfVsf(x_plus_half_v, &f_v);
#endif

  x_qf32_v = Q6_Vqf32_vsub_VsfVsf(x_v, f_v);  // x = x - f;
  x_v      = Q6_Vsf_equals_Vqf32(x_qf32_v);

  //    y = E4 + E5 * x;
  e5_sf_v = Q6_V_vsplat_R(e5_sf);
  y_v     = Q6_Vqf32_vmpy_VsfVsf(e5_sf_v, x_v);
  e4_sf_v = Q6_V_vsplat_R(e4_sf);
  y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, e4_sf_v);

  //    y = E3 + y * x;
  e3_sf_v = Q6_V_vsplat_R(e3_sf);
  y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, e3_sf_v);

  //    y = E2 + y * x;
  e2_sf_v = Q6_V_vsplat_R(e2_sf);
  y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, e2_sf_v);

  //    y = E1 + y * x;
  e1_sf_v = Q6_V_vsplat_R(e1_sf);
  y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, e1_sf_v);

  //    y = E0 + y * x;
  e0_sf_v = Q6_V_vsplat_R(e0_sf);
  y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, e0_sf_v);

  //    y = y * x + 1.0;
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, one_sf_v);

  // insert exponents
  //        y = ldexpf(y, k);
  //    y_v += k_v; // qf32
  // modify exponent
  y_v = Q6_Vsf_equals_Vqf32(y_v);

  // add k_v to the exponent of y_v
  HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);              // shift away sign bit
  y_v_exponent            = Q6_Vuw_vlsr_VuwR(y_v_exponent, 24);  // shift back by sign bit + 23-bit mantissa
  y_v_exponent            = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

  // exponent cannot be negative; if underflow detected, result is set to zero
  HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

  // update exponent
  y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, 23);

  y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);
  return y_v;
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_my_exp2_vsf_vqf32(HVX_Vector x) {
  return hvx_my_exp2_vsf(Q6_Vsf_equals_Vqf32(x));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_my_exp2_vqf32(HVX_Vector x) {
  return Q6_Vqf32_vadd_VsfVsf(hvx_my_exp2_vsf_vqf32(x), Q6_V_vzero());
}

// adapted from libs/qhl_hvx/src/qhmath_hvx/qhmath_hvx_inv_af.c
static HVX_INLINE_ALWAYS HVX_Vector hvx_my_inv_vqf32_vsf(HVX_Vector x_v) {
  static const float c0_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.882601794814435,
    3.6625422144222575,
    3.464451548227971,
    3.2869700047974098,
    3.126105117815294,
    2.9797652947122333,
    2.846287833147896,
    2.7247270166228237,
    2.614282526778659,
    2.5119448279766914,
    2.4168240690138916,
    2.3287715099556494,
    2.2470044371606255,
    2.1705097010458525,
    2.0993232550771013,
    2.032425103348979,
  };
  static const float c1_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -5.65213466274883,
    -5.029649818173625,
    -4.500359068222728,
    -4.051125252469975,
    -3.6643282495304743,
    -3.3293252513210945,
    -3.0377500909629918,
    -2.78384542029156,
    -2.562751394984757,
    -2.3660481944625364,
    -2.1902579830702398,
    -2.033579850063907,
    -1.8932880190031018,
    -1.7665817851802996,
    -1.6526109646324616,
    -1.5489652830974667,
  };
  static const float c2_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.6564123863772062,
    3.0693863078484034,
    2.5979108429264546,
    2.2188401136904137,
    1.90879196515026,
    1.6531365145318937,
    1.4408072849395228,
    1.2640160009581791,
    1.1164726565567085,
    0.9904366133906549,
    0.8821387892416702,
    0.7892039810345458,
    0.7089644931002874,
    0.6390020714403465,
    0.5781761255999769,
    0.5246475096790261,
  };
  static const float c3_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.8868796162009371,
    -0.7023245532864408,
    -0.5623148115716742,
    -0.45568061400557225,
    -0.3728293181808119,
    -0.30778916969628956,
    -0.25624427383670373,
    -0.21520836864975557,
    -0.18238585316003267,
    -0.1554651987039696,
    -0.133224398745864,
    -0.11484835534787588,
    -0.09954996553138899,
    -0.08667244996867919,
    -0.07585106425203664,
    -0.06663557250850614,
  };

  HVX_DV         c0_coeff_dv, c1_coeff_dv, c2_coeff_dv, c3_coeff_dv;
  HVX_VectorPair c0_coeff_vp, c1_coeff_vp, c2_coeff_vp, c3_coeff_vp;

  /*
   * Splat scale factor in order to be used later for finding indexes of coefficients.
   */
  HVX_Vector scale_v = Q6_V_vsplat_R(0x417ffffe);

  /*
   * Vector of zeroes used as neutral element in sf to qf32 conversions.
   * NOTE: Some of conversions (i.e conversion of scale factor and coefficients)
   *       can be avoided in real-time, but this is not done in order to don't
   *       sacrify code readibility in expense of insignificant performance improvement.
   */
  const HVX_Vector zero_v_sf = Q6_V_vzero();

  /* Set sign = 0, exp = 254, mant = 0 */
  const HVX_Vector exp = Q6_V_vsplat_R(0x7F000000);

  /* Set mask for sign and exponent */
  const HVX_Vector signexp_mask = Q6_V_vsplat_R(0xFF800000);

  /* Mask for extracting only 4 bits of mantissa */
  const HVX_Vector mask_idx1_v = Q6_V_vsplat_R(0x0000000F);
  const HVX_Vector mask_idx2_v = Q6_V_vsplat_R(0x00000010);

  /* 16.0 in IEEE 32-bit floating-point representation */
  const HVX_Vector const16_0_v_sf = Q6_V_vsplat_R(0x41800000);

  /*
   * Prepare vector of input_min values, that is used later in shifting input range.
   * input_min is low boundary of specified input range.
   */
  const HVX_Vector input_min_v_f = Q6_V_vsplat_R(0x3f800000);

  /* Convert scale factor from sf to q32. Use the same vector for both formats */
  scale_v = Q6_Vqf32_vadd_VsfVsf(scale_v, zero_v_sf);

  /* Load coefficients */
  HVX_Vector c0_coeff_v = *((HVX_Vector *) (c0_coeffs));
  HVX_Vector c1_coeff_v = *((HVX_Vector *) (c1_coeffs));
  HVX_Vector c2_coeff_v = *((HVX_Vector *) (c2_coeffs));
  HVX_Vector c3_coeff_v = *((HVX_Vector *) (c3_coeffs));

  /* Convert coefficients from sf to qf32 format. Use the same vector for both representations */
  c0_coeff_v = Q6_Vqf32_vadd_VsfVsf(c0_coeff_v, zero_v_sf);
  c1_coeff_v = Q6_Vqf32_vadd_VsfVsf(c1_coeff_v, zero_v_sf);
  c2_coeff_v = Q6_Vqf32_vadd_VsfVsf(c2_coeff_v, zero_v_sf);
  c3_coeff_v = Q6_Vqf32_vadd_VsfVsf(c3_coeff_v, zero_v_sf);

  /* Split 32-bit coefficients to lower and upper part in order to obtain them later with VLUT16. */
  c0_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
  c1_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
  c2_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
  c3_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c3_coeff_v);

  /* Calculate normalization factor */
  HVX_Vector norm_factor;
  norm_factor = Q6_V_vand_VV(x_v, signexp_mask);
  norm_factor = Q6_Vw_vsub_VwVw(exp, norm_factor);

  /* Normalize input */
  x_v = Q6_Vqf32_vmpy_VsfVsf(x_v, norm_factor);

  /* Convert normalization factor to qf32 */
  norm_factor = Q6_Vqf32_vadd_VsfVsf(norm_factor, zero_v_sf);

  /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
  HVX_Vector input_shifted_v_qf32 = Q6_Vqf32_vsub_Vqf32Vsf(x_v, input_min_v_f);

  /*
   * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
   * in order to get corresponding coefficient indexes
   */
  HVX_Vector input_scaled_v_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v_qf32, scale_v);

  /*
   * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
   * to [16.0,32.0) in order to convert float indexes to integer values.
   * Float values, represented in IEEE 754, in range [16.0,32.0] have the
   * same exponent, which means 4 MSB of mantissa carry information about
   * integer index.
   */
  input_scaled_v_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v_qf32, const16_0_v_sf);

  /* Convert back from qf32 to sf in order to extract integer index */
  HVX_Vector tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v_qf32);

  /* Only 4 MSB bits of mantissa represent segment index */
  HVX_Vector idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);

  idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
  idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);

  HVX_Vector idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

  /* Obtain the polynomial coefficients from lookup table */
  c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
  c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
  c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
  c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
  c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
  c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
  c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
  c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);

  /* Perform evaluation of polynomial using Horner's method */
  HVX_Vector output_v;
  output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c3_coeff_vp), x_v);
  output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c2_coeff_vp));
  output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, x_v);
  output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c1_coeff_vp));
  output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, x_v);
  output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c0_coeff_vp));

  /* Multiply result by same normalization factor applied to input earlier */
  output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, norm_factor);

  return output_v;
  // return Q6_Vsf_equals_Vqf32(output_v);
}
