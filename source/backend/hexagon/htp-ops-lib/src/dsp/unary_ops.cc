#include <AEEStdErr.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <stdint.h>

#include "dsp/hvx_math.h"
#include "dsp/worker_pool.h"

extern "C" {

typedef enum {
  HTP_OPS_UNARY_ABS     = 1,
  HTP_OPS_UNARY_NEG     = 2,
  HTP_OPS_UNARY_GELU    = 3,
  HTP_OPS_UNARY_SIGMOID = 4,
  HTP_OPS_UNARY_EXP     = 5,
  HTP_OPS_UNARY_LOG     = 6,
  HTP_OPS_UNARY_SILU    = 7,
  HTP_OPS_UNARY_TANH    = 8,
  HTP_OPS_UNARY_SQUARE  = 9,
  HTP_OPS_UNARY_SQRT    = 10,
  HTP_OPS_UNARY_RSQRT   = 11,
  HTP_OPS_UNARY_EXPM1   = 12,
  HTP_OPS_UNARY_COS     = 13,
  HTP_OPS_UNARY_SIN     = 14,
} HtpOpsUnaryOpType;

#define HTP_OPS_UNARY_MT_MIN_FP16_ELEMS    2048
#define HTP_OPS_UNARY_MT_FP16_GRAIN_ELEMS  (128 / (int) sizeof(__fp16))
#define HTP_OPS_UNARY_MT_MIN_INT32_ELEMS   1024
#define HTP_OPS_UNARY_MT_INT32_GRAIN_ELEMS 128
#define HTP_OPS_UNARY_L2FETCH_VECS         8

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  int                size;
  int                grain;
  int                opType;
  int                bytes;
  __fp16            *fp16_dst;
  const __fp16      *fp16_src;
  int32_t           *int32_dst;
  const int32_t     *int32_src;
} HtpOpsUnaryTaskState;

static inline float htp_ops_unary_fast_logf(float x) {
  if (!(x > 0.0f)) {
    return -65504.0f;
  }

  union {
    float    f;
    uint32_t u;
  } v;

  v.f     = x;
  int e   = (int) ((v.u >> 23) & 0xff) - 127;
  v.u     = (v.u & 0x007fffffu) | 0x3f800000u;
  float m = v.f;
  if (m > 1.41421356237f) {
    m *= 0.5f;
    e += 1;
  }
  const float t    = (m - 1.0f) / (m + 1.0f);
  const float t2   = t * t;
  const float t3   = t * t2;
  const float t5   = t3 * t2;
  const float t7   = t5 * t2;
  const float ln_m = 2.0f * (t + t3 * (1.0f / 3.0f) + t5 * (1.0f / 5.0f) + t7 * (1.0f / 7.0f));
  return ln_m + (float) e * 0.69314718056f;
}

static inline float htp_ops_unary_fast_rsqrtf(float x) {
  union {
    float    f;
    uint32_t u;
  } v = { .f = x };

  const float half_x = 0.5f * x;
  v.u                = 0x5f3759dfu - (v.u >> 1);
  float y            = v.f;
  y                  = y * (1.5f - half_x * y * y);
  y                  = y * (1.5f - half_x * y * y);
  return y;
}

static inline float htp_ops_unary_reduce_angle(float x) {
  const float inv_two_pi = 0.15915494309189535f;
  const float two_pi     = 6.28318530717958648f;
  const int   k          = (int) (x * inv_two_pi + (x >= 0.0f ? 0.5f : -0.5f));
  x -= (float) k * two_pi;
  if (x > 3.14159265358979324f) {
    x -= two_pi;
  } else if (x < -3.14159265358979324f) {
    x += two_pi;
  }
  return x;
}

static inline float htp_ops_unary_fast_sinf(float x) {
  x                   = htp_ops_unary_reduce_angle(x);
  const float pi      = 3.14159265358979324f;
  const float half_pi = 1.57079632679489662f;
  if (x > half_pi) {
    x = pi - x;
  } else if (x < -half_pi) {
    x = -pi - x;
  }
  const float x2 = x * x;
  return x * (1.0f + x2 * (-0.1666666716f + x2 * (0.0083333477f + x2 * (-0.0001984090f))));
}

static inline float htp_ops_unary_fast_cosf(float x) {
  x                   = htp_ops_unary_reduce_angle(x);
  float       sign    = 1.0f;
  const float pi      = 3.14159265358979324f;
  const float half_pi = 1.57079632679489662f;
  if (x > half_pi) {
    x    = pi - x;
    sign = -1.0f;
  } else if (x < -half_pi) {
    x    = -pi - x;
    sign = -1.0f;
  }
  const float x2 = x * x;
  return sign * (1.0f + x2 * (-0.5f + x2 * (0.0416666418f + x2 * (-0.0013888397f))));
}

static inline HVX_Vector htp_ops_unary_fast_rsqrt_vsf(HVX_Vector x) {
  const HVX_Vector half         = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, Q6_V_vsplat_R(0x3f000000)));
  HVX_Vector       bits         = Q6_Vuw_vlsr_VuwR(x, 1);
  HVX_Vector       y            = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5f3759df), bits);
  const HVX_Vector onePointFive = Q6_V_vsplat_R(0x3fc00000);
  HVX_Vector       yy           = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, y));
  HVX_Vector       corr         = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(half, yy));
  corr                          = Q6_Vsf_vsub_VsfVsf(onePointFive, corr);
  y                             = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, corr));
  yy                            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, y));
  corr                          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(half, yy));
  corr                          = Q6_Vsf_vsub_VsfVsf(onePointFive, corr);
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, corr));
}

static inline _Float16 htp_ops_unary_apply_fp16(_Float16 x, int opType) {
  switch (opType) {
    case HTP_OPS_UNARY_ABS:
      return x < (_Float16) 0 ? -x : x;
    case HTP_OPS_UNARY_NEG:
      return -x;
    case HTP_OPS_UNARY_GELU:
      {
        float x_f    = (float) x;
        float x3     = x_f * x_f * x_f;
        float inner  = 0.79788456f * (x_f + 0.044715f * x3);
        float gelu_f = 0.5f * x_f * (1.0f + tanhf(inner));
        return (_Float16) gelu_f;
      }
    case HTP_OPS_UNARY_SIGMOID:
      {
        float x_f = (float) x;
        return (_Float16) (1.0f / (1.0f + expf(-x_f)));
      }
    case HTP_OPS_UNARY_EXP:
      return (_Float16) expf((float) x);
    case HTP_OPS_UNARY_LOG:
      return (_Float16) htp_ops_unary_fast_logf((float) x);
    case HTP_OPS_UNARY_SILU:
      {
        float x_f = (float) x;
        return (_Float16) (x_f / (1.0f + expf(-x_f)));
      }
    case HTP_OPS_UNARY_TANH:
      return (_Float16) tanhf((float) x);
    case HTP_OPS_UNARY_SQUARE:
      return x * x;
    case HTP_OPS_UNARY_SQRT:
      return (_Float16) __builtin_sqrtf((float) x);
    case HTP_OPS_UNARY_RSQRT:
      return (_Float16) htp_ops_unary_fast_rsqrtf((float) x);
    case HTP_OPS_UNARY_EXPM1:
      return (_Float16) (expf((float) x) - 1.0f);
    case HTP_OPS_UNARY_COS:
      return (_Float16) htp_ops_unary_fast_cosf((float) x);
    case HTP_OPS_UNARY_SIN:
      return (_Float16) htp_ops_unary_fast_sinf((float) x);
    default:
      return x;
  }
}

static inline int32_t htp_ops_unary_apply_int32(int32_t x, int opType) {
  switch (opType) {
    case HTP_OPS_UNARY_ABS:
      return x < 0 ? -x : x;
    case HTP_OPS_UNARY_NEG:
      return -x;
    default:
      return x;
  }
}

static inline HVX_Vector htp_ops_unary_sigmoid_fp16_vec(HVX_Vector v, HVX_Vector zero_v, HVX_Vector one_v) {
  HVX_VectorPred q_v_lt_0 = Q6_Q_vcmp_gt_VhfVhf(zero_v, v);
  HVX_Vector     neg_v    = Q6_Vhf_vsub_VhfVhf(zero_v, v);
  HVX_Vector     ax       = Q6_V_vmux_QVV(q_v_lt_0, neg_v, v);

  static const uint16_t slope_bits[32] = { 0x33f5, 0x33b7, 0x3343, 0x32a4, 0x31eb, 0x3128, 0x3067, 0x2f62,
                                           0x2e1b, 0x2cfd, 0x2c0a, 0x2a7b, 0x292c, 0x281a, 0x267d, 0x251c,
                                           0x2404, 0x224d, 0x20ef, 0x1fb8, 0x1e08, 0x1cb6, 0x1b5a, 0x19bc,
                                           0x1879, 0x16f9, 0x156f, 0x143c, 0x1299, 0x1124, 0x1001, 0x0e3d };
  static const uint16_t bias_bits[32]  = { 0x3800, 0x3804, 0x3812, 0x3830, 0x385e, 0x389b, 0x38e4, 0x3933,
                                           0x3985, 0x39d5, 0x3a22, 0x3a68, 0x3aa7, 0x3ade, 0x3b0e, 0x3b38,
                                           0x3b5b, 0x3b78, 0x3b91, 0x3ba5, 0x3bb6, 0x3bc4, 0x3bcf, 0x3bd9,
                                           0x3be0, 0x3be6, 0x3beb, 0x3bef, 0x3bf3, 0x3bf5, 0x3bf7, 0x3bf9 };
  static const uint16_t edge_bits[31]  = { 0x3400, 0x3800, 0x3a00, 0x3c00, 0x3d00, 0x3e00, 0x3f00, 0x4000,
                                           0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380, 0x4400,
                                           0x4440, 0x4480, 0x44c0, 0x4500, 0x4540, 0x4580, 0x45c0, 0x4600,
                                           0x4640, 0x4680, 0x46c0, 0x4700, 0x4740, 0x4780, 0x47c0 };

  HVX_Vector y =
    Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(ax, Q6_Vh_vsplat_R(slope_bits[0])), Q6_Vh_vsplat_R(bias_bits[0]));
  for (int i = 1; i < 32; ++i) {
    HVX_Vector candidate =
      Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(ax, Q6_Vh_vsplat_R(slope_bits[i])), Q6_Vh_vsplat_R(bias_bits[i]));
    HVX_VectorPred q_ge_edge = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(edge_bits[i - 1]));
    y                        = Q6_V_vmux_QVV(q_ge_edge, candidate, y);
  }

  HVX_VectorPred q_ge_8 = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(0x4800));
  y                     = Q6_V_vmux_QVV(q_ge_8, one_v, y);
  HVX_Vector y_neg      = Q6_Vhf_vsub_VhfVhf(one_v, y);
  return Q6_V_vmux_QVV(q_v_lt_0, y_neg, y);
}

static inline HVX_Vector htp_ops_unary_sigmoid_fp16_fast_vec(HVX_Vector v, HVX_Vector zero_v, HVX_Vector one_v) {
  HVX_VectorPred q_v_lt_0 = Q6_Q_vcmp_gt_VhfVhf(zero_v, v);
  HVX_Vector     neg_v    = Q6_Vhf_vsub_VhfVhf(zero_v, v);
  HVX_Vector     ax       = Q6_V_vmux_QVV(q_v_lt_0, neg_v, v);

  static const uint16_t slope_bits[16] = { 0x33d6, 0x32f3, 0x3189, 0x300c, 0x2d8c, 0x2b47, 0x28a3, 0x25cd,
                                           0x232b, 0x2066, 0x1d5f, 0x1a8b, 0x17f5, 0x14d6, 0x11df, 0x0f20 };
  static const uint16_t bias_bits[16]  = { 0x3800, 0x381c, 0x3877, 0x3906, 0x39a9, 0x3a41, 0x3ac0, 0x3b22,
                                           0x3b68, 0x3b9a, 0x3bbd, 0x3bd4, 0x3be3, 0x3bed, 0x3bf4, 0x3bf8 };
  static const uint16_t edge_bits[15]  = { 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4100, 0x4200, 0x4300, 0x4400,
                                           0x4480, 0x4500, 0x4580, 0x4600, 0x4680, 0x4700, 0x4780 };

  HVX_Vector y = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(ax, Q6_Vh_vsplat_R(slope_bits[0])), Q6_Vh_vsplat_R(0x3800));
  for (int i = 1; i < 16; ++i) {
    HVX_Vector candidate =
      Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(ax, Q6_Vh_vsplat_R(slope_bits[i])), Q6_Vh_vsplat_R(bias_bits[i]));
    HVX_VectorPred q_ge_edge = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(edge_bits[i - 1]));
    y                        = Q6_V_vmux_QVV(q_ge_edge, candidate, y);
  }

  HVX_VectorPred q_ge_8 = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(0x4800));
  y                     = Q6_V_vmux_QVV(q_ge_8, one_v, y);
  HVX_Vector y_neg      = Q6_Vhf_vsub_VhfVhf(one_v, y);
  return Q6_V_vmux_QVV(q_v_lt_0, y_neg, y);
}

static inline HVX_Vector htp_ops_unary_gelu_fp16_vec(HVX_Vector v, HVX_Vector zero_v, HVX_Vector one_v) {
  HVX_Vector c_2_sqrt_2_over_pi = Q6_Vh_vsplat_R(0x3e62);
  HVX_Vector c_0_044715         = Q6_Vh_vsplat_R(0x29b9);
  HVX_Vector x2                 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, v));
  HVX_Vector x3                 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x2, v));
  HVX_Vector cubic              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x3, c_0_044715));
  HVX_Vector inner              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, cubic));
  HVX_Vector sig_arg            = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(inner, c_2_sqrt_2_over_pi));
  HVX_Vector sig                = htp_ops_unary_sigmoid_fp16_vec(sig_arg, zero_v, one_v);
  return Q6_Vhf_vmpy_VhfVhf(v, sig);
}

static inline void htp_ops_unary_compute_fp16_chunk(__fp16 *dst, const __fp16 *src, int size, int opType) {
  int       i       = 0;
  const int vec_len = 128 / (int) sizeof(__fp16);
  const int vec_end = size & -vec_len;
  if (opType == HTP_OPS_UNARY_ABS) {
    const __fp16 *src_ptr  = src;
    __fp16       *dst_ptr  = dst;
    HVX_Vector    abs_mask = Q6_Vh_vsplat_R(0x7fff);
    for (; i < vec_end; i += vec_len) {
      const int pf = i + vec_len * HTP_OPS_UNARY_L2FETCH_VECS;
      if (pf < vec_end) {
        const int remain = (vec_end - pf) / vec_len;
        l2fetch(src + pf, 128, 128, remain < HTP_OPS_UNARY_L2FETCH_VECS ? remain : HTP_OPS_UNARY_L2FETCH_VECS, 0);
      }
      HVX_Vector v  = vmem(src_ptr);
      vmem(dst_ptr) = Q6_V_vand_VV(v, abs_mask);
      src_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_UNARY_NEG) {
    const __fp16 *src_ptr   = src;
    __fp16       *dst_ptr   = dst;
    HVX_Vector    sign_mask = Q6_Vh_vsplat_R(0x8000);
    for (; i < vec_end; i += vec_len) {
      const int pf = i + vec_len * HTP_OPS_UNARY_L2FETCH_VECS;
      if (pf < vec_end) {
        const int remain = (vec_end - pf) / vec_len;
        l2fetch(src + pf, 128, 128, remain < HTP_OPS_UNARY_L2FETCH_VECS ? remain : HTP_OPS_UNARY_L2FETCH_VECS, 0);
      }
      HVX_Vector v  = vmem(src_ptr);
      vmem(dst_ptr) = Q6_V_vxor_VV(v, sign_mask);
      src_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_UNARY_SQUARE) {
    const __fp16 *src_ptr = src;
    __fp16       *dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      const int pf = i + vec_len * HTP_OPS_UNARY_L2FETCH_VECS;
      if (pf < vec_end) {
        const int remain = (vec_end - pf) / vec_len;
        l2fetch(src + pf, 128, 128, remain < HTP_OPS_UNARY_L2FETCH_VECS ? remain : HTP_OPS_UNARY_L2FETCH_VECS, 0);
      }
      HVX_Vector v  = vmem(src_ptr);
      vmem(dst_ptr) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, v));
      src_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_UNARY_RSQRT) {
    const __fp16 *src_ptr = src;
    __fp16       *dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      const int pf = i + vec_len * HTP_OPS_UNARY_L2FETCH_VECS;
      if (pf < vec_end) {
        const int remain = (vec_end - pf) / vec_len;
        l2fetch(src + pf, 128, 128, remain < HTP_OPS_UNARY_L2FETCH_VECS ? remain : HTP_OPS_UNARY_L2FETCH_VECS, 0);
      }
      HVX_VectorPair sf = Q6_Wsf_vcvt_Vhf(vmem(src_ptr));
      HVX_Vector     r0 = htp_ops_unary_fast_rsqrt_vsf(Q6_V_lo_W(sf));
      HVX_Vector     r1 = htp_ops_unary_fast_rsqrt_vsf(Q6_V_hi_W(sf));
      vmem(dst_ptr)     = Q6_Vhf_vcvt_VsfVsf(r0, r1);
      src_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_UNARY_EXP) {
    const __fp16    *src_ptr = src;
    __fp16          *dst_ptr = dst;
    const HVX_Vector log2e_v = Q6_Vh_vsplat_R(0x3dc5);
    for (; i < vec_end; i += vec_len) {
      const int pf = i + vec_len * HTP_OPS_UNARY_L2FETCH_VECS;
      if (pf < vec_end) {
        const int remain = (vec_end - pf) / vec_len;
        l2fetch(src + pf, 128, 128, remain < HTP_OPS_UNARY_L2FETCH_VECS ? remain : HTP_OPS_UNARY_L2FETCH_VECS, 0);
      }
      HVX_Vector v      = vmem(src_ptr);
      HVX_Vector expArg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, log2e_v));
      vmem(dst_ptr)     = hvx_my_exp2_vhf(expArg);
      src_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_UNARY_EXPM1) {
    const __fp16    *src_ptr = src;
    __fp16          *dst_ptr = dst;
    const HVX_Vector log2e_v = Q6_Vh_vsplat_R(0x3dc5);
    const HVX_Vector one_v   = Q6_Vh_vsplat_R(0x3c00);
    for (; i < vec_end; i += vec_len) {
      const int pf = i + vec_len * HTP_OPS_UNARY_L2FETCH_VECS;
      if (pf < vec_end) {
        const int remain = (vec_end - pf) / vec_len;
        l2fetch(src + pf, 128, 128, remain < HTP_OPS_UNARY_L2FETCH_VECS ? remain : HTP_OPS_UNARY_L2FETCH_VECS, 0);
      }
      HVX_Vector v      = vmem(src_ptr);
      HVX_Vector expArg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, log2e_v));
      vmem(dst_ptr)     = Q6_Vhf_vsub_VhfVhf(hvx_my_exp2_vhf(expArg), one_v);
      src_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_UNARY_SIGMOID || opType == HTP_OPS_UNARY_GELU || opType == HTP_OPS_UNARY_SILU ||
             opType == HTP_OPS_UNARY_TANH) {
    HVX_Vector    zero_v   = Q6_V_vzero();
    HVX_Vector    one_v    = Q6_Vh_vsplat_R(0x3c00);
    HVX_Vector    two_v    = Q6_Vh_vsplat_R(0x4000);
    const __fp16 *src_ptr  = src;
    __fp16       *dst_ptr  = dst;
    const int     vec2_len = vec_len * 2;
    const int     vec2_end = vec_end & -vec2_len;
    for (; i < vec2_end; i += vec2_len) {
      const int pf = i + vec_len * HTP_OPS_UNARY_L2FETCH_VECS;
      if (pf < vec_end) {
        const int remain = (vec_end - pf) / vec_len;
        l2fetch(src + pf, 128, 128, remain < HTP_OPS_UNARY_L2FETCH_VECS ? remain : HTP_OPS_UNARY_L2FETCH_VECS, 0);
      }
      HVX_Vector v      = vmem(src_ptr);
      HVX_Vector v_next = vmem(src_ptr + vec_len);
      HVX_Vector vr;
      HVX_Vector vr_next;
      if (opType == HTP_OPS_UNARY_SIGMOID) {
        vr      = htp_ops_unary_sigmoid_fp16_vec(v, zero_v, one_v);
        vr_next = htp_ops_unary_sigmoid_fp16_vec(v_next, zero_v, one_v);
      } else if (opType == HTP_OPS_UNARY_GELU) {
        vr      = htp_ops_unary_gelu_fp16_vec(v, zero_v, one_v);
        vr_next = htp_ops_unary_gelu_fp16_vec(v_next, zero_v, one_v);
      } else if (opType == HTP_OPS_UNARY_SILU) {
        vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, htp_ops_unary_sigmoid_fp16_vec(v, zero_v, one_v)));
        vr_next =
          Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_next, htp_ops_unary_sigmoid_fp16_vec(v_next, zero_v, one_v)));
      } else {
        HVX_Vector sig =
          htp_ops_unary_sigmoid_fp16_fast_vec(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, two_v)), zero_v, one_v);
        HVX_Vector sig_next =
          htp_ops_unary_sigmoid_fp16_fast_vec(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_next, two_v)), zero_v, one_v);
        vr      = Q6_Vhf_vsub_VhfVhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sig, two_v)), one_v);
        vr_next = Q6_Vhf_vsub_VhfVhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sig_next, two_v)), one_v);
      }
      vmem(dst_ptr)           = vr;
      vmem(dst_ptr + vec_len) = vr_next;
      src_ptr += vec2_len;
      dst_ptr += vec2_len;
    }
    for (; i < vec_end; i += vec_len) {
      const int pf = i + vec_len * HTP_OPS_UNARY_L2FETCH_VECS;
      if (pf < vec_end) {
        const int remain = (vec_end - pf) / vec_len;
        l2fetch(src + pf, 128, 128, remain < HTP_OPS_UNARY_L2FETCH_VECS ? remain : HTP_OPS_UNARY_L2FETCH_VECS, 0);
      }
      HVX_Vector v = vmem(src_ptr);
      HVX_Vector vr;
      if (opType == HTP_OPS_UNARY_SIGMOID) {
        vr = htp_ops_unary_sigmoid_fp16_vec(v, zero_v, one_v);
      } else if (opType == HTP_OPS_UNARY_GELU) {
        vr = htp_ops_unary_gelu_fp16_vec(v, zero_v, one_v);
      } else if (opType == HTP_OPS_UNARY_SILU) {
        vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, htp_ops_unary_sigmoid_fp16_vec(v, zero_v, one_v)));
      } else {
        HVX_Vector sig =
          htp_ops_unary_sigmoid_fp16_fast_vec(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, two_v)), zero_v, one_v);
        vr = Q6_Vhf_vsub_VhfVhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sig, two_v)), one_v);
      }
      vmem(dst_ptr) = vr;
      src_ptr += vec_len;
      dst_ptr += vec_len;
    }
  }
  for (; i < size; ++i) {
    dst[i] = htp_ops_unary_apply_fp16(src[i], opType);
  }
}

static inline void htp_ops_unary_compute_int32_chunk(int32_t *dst, const int32_t *src, int size, int opType) {
  for (int i = 0; i < size; ++i) {
    dst[i] = htp_ops_unary_apply_int32(src[i], opType);
  }
}

typedef struct {
  HtpOpsUnaryTaskState *state;
  int                   start;
  int                   count;
} HtpOpsUnaryFixedTask;

static void htp_ops_unary_fixed_worker(void *data, int worker_index) {
  (void) worker_index;
  HtpOpsUnaryFixedTask *task  = (HtpOpsUnaryFixedTask *) data;
  HtpOpsUnaryTaskState *state = task->state;
  if (state->bytes == 2) {
    htp_ops_unary_compute_fp16_chunk(state->fp16_dst + task->start, state->fp16_src + task->start, task->count,
                                     state->opType);
  } else {
    htp_ops_unary_compute_int32_chunk(state->int32_dst + task->start, state->int32_src + task->start, task->count,
                                      state->opType);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline int htp_ops_unary_pick_task_count(int size, int bytes) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap <= 1) {
    return 1;
  }

  const int min_elems_per_task = (bytes == 2) ? HTP_OPS_UNARY_MT_MIN_FP16_ELEMS : HTP_OPS_UNARY_MT_MIN_INT32_ELEMS;
  int       task_count         = (size + min_elems_per_task - 1) / min_elems_per_task;
  if (task_count < 2) {
    return 1;
  }
  if (task_count > (int) worker_cap) {
    task_count = (int) worker_cap;
  }
  return task_count;
}

static inline void htp_ops_unary_run_task(HtpOpsUnaryTaskState *state, int size) {
  state->task_id = 0;
  state->size    = size;
  state->grain   = (state->bytes == 2) ? HTP_OPS_UNARY_MT_FP16_GRAIN_ELEMS : HTP_OPS_UNARY_MT_INT32_GRAIN_ELEMS;

  const int n_tasks = htp_ops_unary_pick_task_count(size, state->bytes);
  if (n_tasks <= 1) {
    if (state->bytes == 2) {
      htp_ops_unary_compute_fp16_chunk(state->fp16_dst, state->fp16_src, size, state->opType);
    } else {
      htp_ops_unary_compute_int32_chunk(state->int32_dst, state->int32_src, size, state->opType);
    }
    return;
  }

  state->n_tasks = n_tasks;
  worker_pool_job_t job;
  job.fptr = htp_ops_unary_fixed_worker;

  worker_pool_synctoken_init(&(state->sync_ctx), n_tasks);
  HtpOpsUnaryFixedTask *tasks           = WORKER_POOL_STACK_ALLOC(HtpOpsUnaryFixedTask, n_tasks);
  const int             total_blocks    = (size + state->grain - 1) / state->grain;
  const int             blocks_per_task = (total_blocks + n_tasks - 1) / n_tasks;
  for (int i = 0; i < n_tasks; ++i) {
    const int start_block = i * blocks_per_task;
    int       end_block   = start_block + blocks_per_task;
    if (end_block > total_blocks) {
      end_block = total_blocks;
    }
    const int start = start_block * state->grain;
    int       end   = end_block * state->grain;
    if (end > size) {
      end = size;
    }
    tasks[i].state = state;
    tasks[i].start = start;
    tasks[i].count = end - start;
    job.dptr       = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state->sync_ctx));
}

AEEResult htp_ops_unary(uint8_t *dst, uint8_t *src, int32_t size, int32_t opType, int32_t bytes) {
  if (bytes != 2 && bytes != 4) {
    return -1;
  }
  if (size <= 0) {
    return 0;
  }
  if (bytes == 4 && opType != HTP_OPS_UNARY_ABS && opType != HTP_OPS_UNARY_NEG) {
    return -1;
  }

  HtpOpsUnaryTaskState task_state = {};
  task_state.opType               = opType;
  task_state.bytes                = bytes;
  if (bytes == 2) {
    task_state.fp16_dst = (__fp16 *) dst;
    task_state.fp16_src = (const __fp16 *) src;
  } else {
    task_state.int32_dst = (int32_t *) dst;
    task_state.int32_src = (const int32_t *) src;
  }
  htp_ops_unary_run_task(&task_state, size);
  return 0;
}

}  // extern "C"
