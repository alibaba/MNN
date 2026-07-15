#include "dsp/mmap_mgr.h"
#include "dsp/hvx_utils.h"
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <qurt_memory.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

#include <remote.h>

extern "C" {


#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  int                batch;
  int                channels;
  int                pack;
  int                fullUnit;
  int                tail;
  int                loopUnit;
  float              epsilon;
  int32_t            RMSNorm;
  size_t             worker_vtcm_bytes;
  uint8_t*           workspace_base;
  __fp16*            dstBase;
  __fp16*            addOutBase;
  const __fp16*      src0Base;
  const __fp16*      src1Base;
  const float*       gammaBase;
  const float*       betaBase;
} HtpOpsAddFuseLayernormTaskState;

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                batch;
  int                channels;
  int                pack;
  int                fullUnit;
  int                tail;
  int                loopUnit;
  float              epsilon;
  int32_t            RMSNorm;
  uint8_t*           dstBase;
  const uint8_t*     srcBase;
  const uint8_t*     gammaBase;
  const uint8_t*     betaBase;
} HtpOpsLayerNormPackedTaskState;

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  int                grain;
  int                outterSize;
  int                innerSize;
  float              epsilon;
  int32_t            RMSNorm;
  uint8_t*           dstBase;
  const uint8_t*     srcBase;
  const uint8_t*     gammaBase;
  const uint8_t*     betaBase;
} HtpOpsLayerNormTaskState;

static inline float htp_ops_layernorm_reduce_sum2_f32(HVX_Vector acc0, HVX_Vector acc1) {
  HVX_Vector v = Q6_Vsf_vadd_VsfVsf(acc0, acc1);
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4));
  union { HVX_Vector v; float f[32]; } u = { .v = v };
  return u.f[0];
}

static inline float htp_ops_layernorm_fast_rsqrtf(float x) {
  union { float f; int32_t i; } u = { .f = x };
  float half_x = 0.5f * x;
  u.i = 0x5f3759df - (u.i >> 1);
  float y = u.f;
  y = y * (1.5f - half_x * y * y);
  y = y * (1.5f - half_x * y * y);
  return y;
}

static inline int htp_ops_add_fuse_layernorm_pick_task_count(int batch) {
  unsigned int worker_cap = g_max_num_workers;
  const int total_blocks = (batch + 1) / 2;
  if (worker_cap <= 1 || total_blocks <= 1) {
    return 1;
  }
  return total_blocks < (int)worker_cap ? total_blocks : (int)worker_cap;
}

static inline int htp_ops_layer_norm_packed_pick_task_count(int batch) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap <= 1 || batch <= 1) {
    return 1;
  }
  return batch < (int)worker_cap ? batch : (int)worker_cap;
}

static inline int htp_ops_layer_norm_pick_task_count(int outterSize, int innerSize) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap <= 1 || outterSize <= 1) {
    return 1;
  }
  const size_t bytes = (size_t)outterSize * innerSize * sizeof(__fp16);
  const size_t min_bytes_per_worker = 4096;
  int by_size = (int)((bytes + min_bytes_per_worker - 1) / min_bytes_per_worker);
  if (by_size < 1) {
    by_size = 1;
  }
  int n_tasks = outterSize < (int)worker_cap ? outterSize : (int)worker_cap;
  return n_tasks < by_size ? n_tasks : by_size;
}

static inline void htp_ops_add_fuse_rmsnorm_one_batch_no_beta(__fp16* dstBase, __fp16* addOutBase,
                                                              const __fp16* src0Base, const __fp16* src1Base,
                                                              const float* gammaBase, int32_t batch, int32_t n,
                                                              int32_t channels, int32_t pack, int32_t fullUnit,
                                                              int32_t tail, int32_t loopUnit, float epsilon,
                                                              __fp16* srcAdd) {
  (void)srcAdd;
  HVX_VectorPred q_tail = Q6_Q_vsetq_R(tail * 2);
  HVX_Vector vzero = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum0 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum1 = Q6_V_vsplat_R(0);

  for (int c = 0; c < loopUnit; ++c) {
    size_t offset = (size_t)(c * batch + n) * pack;
    HVX_Vector s0_hf = *((const HVX_UVector*)(src0Base + offset));
    HVX_Vector s1_hf = *((const HVX_UVector*)(src1Base + offset));
    HVX_Vector add_hf = Q6_Vhf_vadd_VhfVhf(s0_hf, s1_hf);
    if (c == fullUnit && tail != 0) {
      add_hf = Q6_V_vmux_QVV(q_tail, add_hf, vzero);
    }

    *((HVX_UVector*)(addOutBase + offset)) = add_hf;

    HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(add_hf);
    HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
    HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);
    vsqsum0 = Q6_Vsf_vadd_VsfVsf(vsqsum0, Q6_Vsf_vmpy_VsfVsf(s_sf0, s_sf0));
    vsqsum1 = Q6_Vsf_vadd_VsfVsf(vsqsum1, Q6_Vsf_vmpy_VsfVsf(s_sf1, s_sf1));
  }

  float sqsum = htp_ops_layernorm_reduce_sum2_f32(vsqsum0, vsqsum1);
  float inv_std = htp_ops_layernorm_fast_rsqrtf(sqsum / channels + epsilon);
  union { float f; int32_t i; } u_inv_std = { .f = inv_std };
  HVX_Vector vinv_std = Q6_V_vsplat_R(u_inv_std.i);

  for (int c = 0; c < loopUnit; ++c) {
    size_t offset = (size_t)(c * batch + n) * pack;
    HVX_Vector add_hf = *((const HVX_UVector*)(addOutBase + offset));

    const float* g_ptr = gammaBase + (size_t)c * pack;
    HVX_Vector vg0 = *((const HVX_UVector*)g_ptr);
    HVX_Vector vg1 = *((const HVX_UVector*)(g_ptr + 32));
    HVX_VectorPair g_deal = Q6_W_vdeal_VVR(vg1, vg0, -4);
    HVX_Vector scale = Q6_Vhf_vcvt_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(g_deal), vinv_std),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(g_deal), vinv_std));

    HVX_Vector out_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(add_hf, scale));
    if (c == fullUnit && tail != 0) {
      out_hf = Q6_V_vmux_QVV(q_tail, out_hf, vzero);
    }
    *((HVX_UVector*)(dstBase + offset)) = out_hf;
  }
}

static inline void htp_ops_add_fuse_layernorm_one_batch(__fp16* dstBase, __fp16* addOutBase,
                                                        const __fp16* src0Base, const __fp16* src1Base,
                                                        const float* gammaBase, const float* betaBase,
                                                        int32_t batch, int32_t n, int32_t channels, int32_t pack,
                                                        int32_t fullUnit, int32_t tail, int32_t loopUnit,
                                                        float epsilon, int32_t RMSNorm, __fp16* srcAdd) {
  if (RMSNorm && betaBase == NULL && gammaBase != NULL) {
    htp_ops_add_fuse_rmsnorm_one_batch_no_beta(dstBase, addOutBase, src0Base, src1Base, gammaBase,
                                               batch, n, channels, pack, fullUnit, tail, loopUnit, epsilon, srcAdd);
    return;
  }

  HVX_VectorPred q_tail = Q6_Q_vsetq_R(tail * 2);
  HVX_Vector vzero = Q6_V_vsplat_R(0);
  HVX_Vector vsum0 = Q6_V_vsplat_R(0);
  HVX_Vector vsum1 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum0 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum1 = Q6_V_vsplat_R(0);

  for (int c = 0; c < loopUnit; ++c) {
    size_t offset = (size_t)(c * batch + n) * pack;
    HVX_Vector s0_hf = *((const HVX_UVector*)(src0Base + offset));
    HVX_Vector s1_hf = *((const HVX_UVector*)(src1Base + offset));
    HVX_Vector add_hf = Q6_Vhf_vadd_VhfVhf(s0_hf, s1_hf);
    if (c == fullUnit && tail != 0) {
      add_hf = Q6_V_vmux_QVV(q_tail, add_hf, vzero);
    }

    *((HVX_UVector*)(srcAdd + (size_t)c * pack)) = add_hf;
    *((HVX_UVector*)(addOutBase + offset)) = add_hf;

    HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(add_hf);
    HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
    HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);
    if (!RMSNorm) {
      vsum0 = Q6_Vsf_vadd_VsfVsf(vsum0, s_sf0);
      vsum1 = Q6_Vsf_vadd_VsfVsf(vsum1, s_sf1);
    }
    vsqsum0 = Q6_Vsf_vadd_VsfVsf(vsqsum0, Q6_Vsf_vmpy_VsfVsf(s_sf0, s_sf0));
    vsqsum1 = Q6_Vsf_vadd_VsfVsf(vsqsum1, Q6_Vsf_vmpy_VsfVsf(s_sf1, s_sf1));
  }

  float sum = 0.0f;
  if (!RMSNorm) {
    sum = htp_ops_layernorm_reduce_sum2_f32(vsum0, vsum1);
  }
  float sqsum = htp_ops_layernorm_reduce_sum2_f32(vsqsum0, vsqsum1);

  float mean = 0.0f;
  if (!RMSNorm) {
    mean = sum / channels;
  }
  float var = sqsum / channels;
  if (!RMSNorm) {
    var -= mean * mean;
  }
  float inv_std = 1.0f / __builtin_sqrtf(var + epsilon);

  union { float f; int32_t i; } u_mean = { .f = mean };
  union { float f; int32_t i; } u_inv_std = { .f = inv_std };
  HVX_Vector vmean = Q6_V_vsplat_R(u_mean.i);
  HVX_Vector vinv_std = Q6_V_vsplat_R(u_inv_std.i);

  for (int c = 0; c < loopUnit; ++c) {
    const __fp16* add_ptr = srcAdd + (size_t)c * pack;
    HVX_Vector add_hf = *((const HVX_UVector*)add_ptr);
    HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(add_hf);
    HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
    HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);

    HVX_Vector out0;
    HVX_Vector out1;
    if (RMSNorm) {
      out0 = Q6_Vsf_vmpy_VsfVsf(s_sf0, vinv_std);
      out1 = Q6_Vsf_vmpy_VsfVsf(s_sf1, vinv_std);
    } else {
      out0 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s_sf0, vmean), vinv_std);
      out1 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s_sf1, vmean), vinv_std);
    }

    if (gammaBase) {
      const float* g_ptr = gammaBase + (size_t)c * pack;
      HVX_Vector vg0 = *((const HVX_UVector*)g_ptr);
      HVX_Vector vg1 = *((const HVX_UVector*)(g_ptr + 32));
      HVX_VectorPair g_deal = Q6_W_vdeal_VVR(vg1, vg0, -4);
      out0 = Q6_Vsf_vmpy_VsfVsf(out0, Q6_V_lo_W(g_deal));
      out1 = Q6_Vsf_vmpy_VsfVsf(out1, Q6_V_hi_W(g_deal));
    }
    if (betaBase) {
      const float* b_ptr = betaBase + (size_t)c * pack;
      HVX_Vector vb0 = *((const HVX_UVector*)b_ptr);
      HVX_Vector vb1 = *((const HVX_UVector*)(b_ptr + 32));
      HVX_VectorPair b_deal = Q6_W_vdeal_VVR(vb1, vb0, -4);
      out0 = Q6_Vsf_vadd_VsfVsf(out0, Q6_V_lo_W(b_deal));
      out1 = Q6_Vsf_vadd_VsfVsf(out1, Q6_V_hi_W(b_deal));
    }

    size_t offset = (size_t)(c * batch + n) * pack;
    HVX_Vector out_hf = Q6_Vhf_vcvt_VsfVsf(out0, out1);
    if (c == fullUnit && tail != 0) {
      out_hf = Q6_V_vmux_QVV(q_tail, out_hf, vzero);
    }
    *((HVX_UVector*)(dstBase + offset)) = out_hf;
  }
}

static inline void htp_ops_add_fuse_rmsnorm_two_batch_no_beta(__fp16* dstBase, __fp16* addOutBase,
                                                              const __fp16* src0Base, const __fp16* src1Base,
                                                              const float* gammaBase, int32_t batch, int32_t n0, int32_t n1,
                                                              int32_t channels, int32_t pack, int32_t fullUnit,
                                                              int32_t tail, int32_t loopUnit, float epsilon,
                                                              __fp16* srcAdd0, __fp16* srcAdd1) {
  HVX_VectorPred q_tail = Q6_Q_vsetq_R(tail * 2);
  HVX_Vector vzero = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum0_0 = Q6_V_vsplat_R(0), vsqsum0_1 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum1_0 = Q6_V_vsplat_R(0), vsqsum1_1 = Q6_V_vsplat_R(0);
  const uint32_t src_stride_bytes = (uint32_t)((size_t)batch * pack * sizeof(__fp16));
  l2fetch(src0Base + (size_t)n0 * pack, src_stride_bytes, 2 * VLEN, loopUnit, 0);
  l2fetch(src1Base + (size_t)n0 * pack, src_stride_bytes, 2 * VLEN, loopUnit, 0);

  for (int c = 0; c < loopUnit; ++c) {
    size_t offset0 = (size_t)(c * batch + n0) * pack;
    size_t offset1 = (size_t)(c * batch + n1) * pack;
    HVX_Vector s00_hf = *((const HVX_UVector*)(src0Base + offset0));
    HVX_Vector s01_hf = *((const HVX_UVector*)(src1Base + offset0));
    HVX_Vector s10_hf = *((const HVX_UVector*)(src0Base + offset1));
    HVX_Vector s11_hf = *((const HVX_UVector*)(src1Base + offset1));
    HVX_Vector add0_hf = Q6_Vhf_vadd_VhfVhf(s00_hf, s01_hf);
    HVX_Vector add1_hf = Q6_Vhf_vadd_VhfVhf(s10_hf, s11_hf);
    if (c == fullUnit && tail != 0) {
      add0_hf = Q6_V_vmux_QVV(q_tail, add0_hf, vzero);
      add1_hf = Q6_V_vmux_QVV(q_tail, add1_hf, vzero);
    }

    *((HVX_UVector*)(srcAdd0 + (size_t)c * pack)) = add0_hf;
    *((HVX_UVector*)(srcAdd1 + (size_t)c * pack)) = add1_hf;
    *((HVX_UVector*)(addOutBase + offset0)) = add0_hf;
    *((HVX_UVector*)(addOutBase + offset1)) = add1_hf;

    HVX_VectorPair s0_sf = Q6_Wsf_vcvt_Vhf(add0_hf);
    HVX_VectorPair s1_sf = Q6_Wsf_vcvt_Vhf(add1_hf);
    HVX_Vector s00 = Q6_V_lo_W(s0_sf);
    HVX_Vector s01 = Q6_V_hi_W(s0_sf);
    HVX_Vector s10 = Q6_V_lo_W(s1_sf);
    HVX_Vector s11 = Q6_V_hi_W(s1_sf);
    vsqsum0_0 = Q6_Vsf_vadd_VsfVsf(vsqsum0_0, Q6_Vsf_vmpy_VsfVsf(s00, s00));
    vsqsum0_1 = Q6_Vsf_vadd_VsfVsf(vsqsum0_1, Q6_Vsf_vmpy_VsfVsf(s01, s01));
    vsqsum1_0 = Q6_Vsf_vadd_VsfVsf(vsqsum1_0, Q6_Vsf_vmpy_VsfVsf(s10, s10));
    vsqsum1_1 = Q6_Vsf_vadd_VsfVsf(vsqsum1_1, Q6_Vsf_vmpy_VsfVsf(s11, s11));
  }

  float sqsum0 = htp_ops_layernorm_reduce_sum2_f32(vsqsum0_0, vsqsum0_1);
  float sqsum1 = htp_ops_layernorm_reduce_sum2_f32(vsqsum1_0, vsqsum1_1);
  const float denom = (float)channels;
  float inv_std0 = htp_ops_layernorm_fast_rsqrtf(sqsum0 / denom + epsilon);
  float inv_std1 = htp_ops_layernorm_fast_rsqrtf(sqsum1 / denom + epsilon);

  union { float f; int32_t i; } u_inv_std0 = { .f = inv_std0 };
  union { float f; int32_t i; } u_inv_std1 = { .f = inv_std1 };
  HVX_Vector vinv_std0 = Q6_V_vsplat_R(u_inv_std0.i);
  HVX_Vector vinv_std1 = Q6_V_vsplat_R(u_inv_std1.i);

  for (int c = 0; c < loopUnit; ++c) {
    HVX_Vector add0_hf = *((const HVX_UVector*)(srcAdd0 + (size_t)c * pack));
    HVX_Vector add1_hf = *((const HVX_UVector*)(srcAdd1 + (size_t)c * pack));

    const float* g_ptr = gammaBase + (size_t)c * pack;
    HVX_Vector vg0 = *((const HVX_UVector*)g_ptr);
    HVX_Vector vg1 = *((const HVX_UVector*)(g_ptr + 32));
    HVX_VectorPair g_deal = Q6_W_vdeal_VVR(vg1, vg0, -4);
    HVX_Vector g0 = Q6_V_lo_W(g_deal);
    HVX_Vector g1 = Q6_V_hi_W(g_deal);
    HVX_Vector scale0 = Q6_Vhf_vcvt_VsfVsf(Q6_Vsf_vmpy_VsfVsf(g0, vinv_std0),
                                            Q6_Vsf_vmpy_VsfVsf(g1, vinv_std0));
    HVX_Vector scale1 = Q6_Vhf_vcvt_VsfVsf(Q6_Vsf_vmpy_VsfVsf(g0, vinv_std1),
                                            Q6_Vsf_vmpy_VsfVsf(g1, vinv_std1));

    size_t offset0 = (size_t)(c * batch + n0) * pack;
    size_t offset1 = (size_t)(c * batch + n1) * pack;
    HVX_Vector out0_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(add0_hf, scale0));
    HVX_Vector out1_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(add1_hf, scale1));
    if (c == fullUnit && tail != 0) {
      out0_hf = Q6_V_vmux_QVV(q_tail, out0_hf, vzero);
      out1_hf = Q6_V_vmux_QVV(q_tail, out1_hf, vzero);
    }
    *((HVX_UVector*)(dstBase + offset0)) = out0_hf;
    *((HVX_UVector*)(dstBase + offset1)) = out1_hf;
  }
}

static inline void htp_ops_add_fuse_layernorm_two_batch(__fp16* dstBase, __fp16* addOutBase,
                                                        const __fp16* src0Base, const __fp16* src1Base,
                                                        const float* gammaBase, const float* betaBase,
                                                        int32_t batch, int32_t n0, int32_t n1,
                                                        int32_t channels, int32_t pack,
                                                        int32_t fullUnit, int32_t tail, int32_t loopUnit,
                                                        float epsilon, int32_t RMSNorm,
                                                        __fp16* srcAdd0, __fp16* srcAdd1) {
  if (RMSNorm && betaBase == NULL && gammaBase != NULL) {
    htp_ops_add_fuse_rmsnorm_two_batch_no_beta(dstBase, addOutBase, src0Base, src1Base, gammaBase,
                                               batch, n0, n1, channels, pack, fullUnit, tail, loopUnit,
                                               epsilon, srcAdd0, srcAdd1);
    return;
  }

  HVX_VectorPred q_tail = Q6_Q_vsetq_R(tail * 2);
  HVX_Vector vzero = Q6_V_vsplat_R(0);
  HVX_Vector vsum0_0 = Q6_V_vsplat_R(0), vsum0_1 = Q6_V_vsplat_R(0);
  HVX_Vector vsum1_0 = Q6_V_vsplat_R(0), vsum1_1 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum0_0 = Q6_V_vsplat_R(0), vsqsum0_1 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum1_0 = Q6_V_vsplat_R(0), vsqsum1_1 = Q6_V_vsplat_R(0);

  for (int c = 0; c < loopUnit; ++c) {
    size_t offset0 = (size_t)(c * batch + n0) * pack;
    size_t offset1 = (size_t)(c * batch + n1) * pack;
    HVX_Vector s00_hf = *((const HVX_UVector*)(src0Base + offset0));
    HVX_Vector s01_hf = *((const HVX_UVector*)(src1Base + offset0));
    HVX_Vector s10_hf = *((const HVX_UVector*)(src0Base + offset1));
    HVX_Vector s11_hf = *((const HVX_UVector*)(src1Base + offset1));
    HVX_Vector add0_hf = Q6_Vhf_vadd_VhfVhf(s00_hf, s01_hf);
    HVX_Vector add1_hf = Q6_Vhf_vadd_VhfVhf(s10_hf, s11_hf);
    if (c == fullUnit && tail != 0) {
      add0_hf = Q6_V_vmux_QVV(q_tail, add0_hf, vzero);
      add1_hf = Q6_V_vmux_QVV(q_tail, add1_hf, vzero);
    }

    *((HVX_UVector*)(srcAdd0 + (size_t)c * pack)) = add0_hf;
    *((HVX_UVector*)(srcAdd1 + (size_t)c * pack)) = add1_hf;
    *((HVX_UVector*)(addOutBase + offset0)) = add0_hf;
    *((HVX_UVector*)(addOutBase + offset1)) = add1_hf;

    HVX_VectorPair s0_sf = Q6_Wsf_vcvt_Vhf(add0_hf);
    HVX_VectorPair s1_sf = Q6_Wsf_vcvt_Vhf(add1_hf);
    HVX_Vector s00 = Q6_V_lo_W(s0_sf);
    HVX_Vector s01 = Q6_V_hi_W(s0_sf);
    HVX_Vector s10 = Q6_V_lo_W(s1_sf);
    HVX_Vector s11 = Q6_V_hi_W(s1_sf);

    if (!RMSNorm) {
      vsum0_0 = Q6_Vsf_vadd_VsfVsf(vsum0_0, s00);
      vsum0_1 = Q6_Vsf_vadd_VsfVsf(vsum0_1, s01);
      vsum1_0 = Q6_Vsf_vadd_VsfVsf(vsum1_0, s10);
      vsum1_1 = Q6_Vsf_vadd_VsfVsf(vsum1_1, s11);
    }
    vsqsum0_0 = Q6_Vsf_vadd_VsfVsf(vsqsum0_0, Q6_Vsf_vmpy_VsfVsf(s00, s00));
    vsqsum0_1 = Q6_Vsf_vadd_VsfVsf(vsqsum0_1, Q6_Vsf_vmpy_VsfVsf(s01, s01));
    vsqsum1_0 = Q6_Vsf_vadd_VsfVsf(vsqsum1_0, Q6_Vsf_vmpy_VsfVsf(s10, s10));
    vsqsum1_1 = Q6_Vsf_vadd_VsfVsf(vsqsum1_1, Q6_Vsf_vmpy_VsfVsf(s11, s11));
  }

  float sum0 = 0.0f, sum1 = 0.0f;
  if (!RMSNorm) {
    sum0 = htp_ops_layernorm_reduce_sum2_f32(vsum0_0, vsum0_1);
    sum1 = htp_ops_layernorm_reduce_sum2_f32(vsum1_0, vsum1_1);
  }
  float sqsum0 = htp_ops_layernorm_reduce_sum2_f32(vsqsum0_0, vsqsum0_1);
  float sqsum1 = htp_ops_layernorm_reduce_sum2_f32(vsqsum1_0, vsqsum1_1);

  float mean0 = 0.0f, mean1 = 0.0f;
  if (!RMSNorm) {
    mean0 = sum0 / channels;
    mean1 = sum1 / channels;
  }
  float var0 = sqsum0 / channels;
  float var1 = sqsum1 / channels;
  if (!RMSNorm) {
    var0 -= mean0 * mean0;
    var1 -= mean1 * mean1;
  }
  float inv_std0 = 1.0f / __builtin_sqrtf(var0 + epsilon);
  float inv_std1 = 1.0f / __builtin_sqrtf(var1 + epsilon);

  union { float f; int32_t i; } u_mean0 = { .f = mean0 };
  union { float f; int32_t i; } u_mean1 = { .f = mean1 };
  union { float f; int32_t i; } u_inv_std0 = { .f = inv_std0 };
  union { float f; int32_t i; } u_inv_std1 = { .f = inv_std1 };
  HVX_Vector vmean0 = Q6_V_vsplat_R(u_mean0.i);
  HVX_Vector vmean1 = Q6_V_vsplat_R(u_mean1.i);
  HVX_Vector vinv_std0 = Q6_V_vsplat_R(u_inv_std0.i);
  HVX_Vector vinv_std1 = Q6_V_vsplat_R(u_inv_std1.i);

  for (int c = 0; c < loopUnit; ++c) {
    HVX_Vector add0_hf = *((const HVX_UVector*)(srcAdd0 + (size_t)c * pack));
    HVX_Vector add1_hf = *((const HVX_UVector*)(srcAdd1 + (size_t)c * pack));
    HVX_VectorPair s0_sf = Q6_Wsf_vcvt_Vhf(add0_hf);
    HVX_VectorPair s1_sf = Q6_Wsf_vcvt_Vhf(add1_hf);
    HVX_Vector s00 = Q6_V_lo_W(s0_sf);
    HVX_Vector s01 = Q6_V_hi_W(s0_sf);
    HVX_Vector s10 = Q6_V_lo_W(s1_sf);
    HVX_Vector s11 = Q6_V_hi_W(s1_sf);

    HVX_Vector out00;
    HVX_Vector out01;
    HVX_Vector out10;
    HVX_Vector out11;
    if (RMSNorm) {
      out00 = Q6_Vsf_vmpy_VsfVsf(s00, vinv_std0);
      out01 = Q6_Vsf_vmpy_VsfVsf(s01, vinv_std0);
      out10 = Q6_Vsf_vmpy_VsfVsf(s10, vinv_std1);
      out11 = Q6_Vsf_vmpy_VsfVsf(s11, vinv_std1);
    } else {
      out00 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s00, vmean0), vinv_std0);
      out01 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s01, vmean0), vinv_std0);
      out10 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s10, vmean1), vinv_std1);
      out11 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s11, vmean1), vinv_std1);
    }

    if (gammaBase) {
      const float* g_ptr = gammaBase + (size_t)c * pack;
      HVX_Vector vg0 = *((const HVX_UVector*)g_ptr);
      HVX_Vector vg1 = *((const HVX_UVector*)(g_ptr + 32));
      HVX_VectorPair g_deal = Q6_W_vdeal_VVR(vg1, vg0, -4);
      HVX_Vector g0 = Q6_V_lo_W(g_deal);
      HVX_Vector g1 = Q6_V_hi_W(g_deal);
      out00 = Q6_Vsf_vmpy_VsfVsf(out00, g0);
      out01 = Q6_Vsf_vmpy_VsfVsf(out01, g1);
      out10 = Q6_Vsf_vmpy_VsfVsf(out10, g0);
      out11 = Q6_Vsf_vmpy_VsfVsf(out11, g1);
    }
    if (betaBase) {
      const float* b_ptr = betaBase + (size_t)c * pack;
      HVX_Vector vb0 = *((const HVX_UVector*)b_ptr);
      HVX_Vector vb1 = *((const HVX_UVector*)(b_ptr + 32));
      HVX_VectorPair b_deal = Q6_W_vdeal_VVR(vb1, vb0, -4);
      HVX_Vector b0 = Q6_V_lo_W(b_deal);
      HVX_Vector b1 = Q6_V_hi_W(b_deal);
      out00 = Q6_Vsf_vadd_VsfVsf(out00, b0);
      out01 = Q6_Vsf_vadd_VsfVsf(out01, b1);
      out10 = Q6_Vsf_vadd_VsfVsf(out10, b0);
      out11 = Q6_Vsf_vadd_VsfVsf(out11, b1);
    }

    size_t offset0 = (size_t)(c * batch + n0) * pack;
    size_t offset1 = (size_t)(c * batch + n1) * pack;
    HVX_Vector out0_hf = Q6_Vhf_vcvt_VsfVsf(out00, out01);
    HVX_Vector out1_hf = Q6_Vhf_vcvt_VsfVsf(out10, out11);
    if (c == fullUnit && tail != 0) {
      out0_hf = Q6_V_vmux_QVV(q_tail, out0_hf, vzero);
      out1_hf = Q6_V_vmux_QVV(q_tail, out1_hf, vzero);
    }
    *((HVX_UVector*)(dstBase + offset0)) = out0_hf;
    *((HVX_UVector*)(dstBase + offset1)) = out1_hf;
  }
}

static inline void htp_ops_add_fuse_layernorm_process_batch_block(HtpOpsAddFuseLayernormTaskState* state,
                                                                  int block_idx, __fp16* srcAdd0,
                                                                  __fp16* srcAdd1) {
  const int n = block_idx * 2;
  if (n + 1 < state->batch) {
    htp_ops_add_fuse_layernorm_two_batch(state->dstBase, state->addOutBase, state->src0Base, state->src1Base,
                                         state->gammaBase, state->betaBase, state->batch, n, n + 1,
                                         state->channels, state->pack, state->fullUnit, state->tail,
                                         state->loopUnit, state->epsilon, state->RMSNorm,
                                         srcAdd0, srcAdd1);
  } else if (n < state->batch) {
    htp_ops_add_fuse_layernorm_one_batch(state->dstBase, state->addOutBase, state->src0Base, state->src1Base,
                                         state->gammaBase, state->betaBase, state->batch, n,
                                         state->channels, state->pack, state->fullUnit, state->tail,
                                         state->loopUnit, state->epsilon, state->RMSNorm,
                                         srcAdd0);
  }
}

static void htp_ops_add_fuse_layernorm_worker(void* data, int worker_index) {
  HtpOpsAddFuseLayernormTaskState* state = (HtpOpsAddFuseLayernormTaskState*)data;
  uint8_t* worker_vtcm = state->workspace_base + (size_t)worker_index * state->worker_vtcm_bytes;
  uint8_t* worker_vtcm_ptr = worker_vtcm;
  __fp16* srcAdd0 = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)state->loopUnit * state->pack * sizeof(__fp16));
  __fp16* srcAdd1 = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)state->loopUnit * state->pack * sizeof(__fp16));

  const int total_blocks = (state->batch + 1) / 2;
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= total_blocks) {
      break;
    }
    htp_ops_add_fuse_layernorm_process_batch_block(state, (int)task_id, srcAdd0, srcAdd1);
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline void htp_ops_layer_norm_one_batch(const HtpOpsLayerNormTaskState* state, int o) {
  const int innerSize = state->innerSize;
  const int RMSNorm = state->RMSNorm;
  const uint8_t *srcInner = state->srcBase + (size_t)o * innerSize * sizeof(__fp16);
  uint8_t *dstInner = state->dstBase + (size_t)o * innerSize * sizeof(__fp16);

  float mean = 0.0f;
  float variance = 0.0f;
  float sum = 0.0f;
  float sqsum = 0.0f;

  const _Float16 *s = (const _Float16 *) srcInner;

  int fullUnit = innerSize / 64;

  HVX_Vector vsum0 = Q6_V_vsplat_R(0);
  HVX_Vector vsum1 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum0 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum1 = Q6_V_vsplat_R(0);

  const HVX_Vector *pv_in = (const HVX_Vector *) s;
  int n_vecs = fullUnit;

  for (int i = 0; i < fullUnit; ++i) {
      if (i % PREFETCH_N_VECS == 0) {
          int prefetch_idx = i + PREFETCH_N_VECS;
          if (prefetch_idx < n_vecs) {
              int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
              l2fetch(pv_in + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
          }
      }

      HVX_Vector s_hf = *((HVX_UVector *)(s + i * 64));
      HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(s_hf);
      HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
      HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);

      if (!RMSNorm) {
          vsum0 = Q6_Vsf_vadd_VsfVsf(vsum0, s_sf0);
          vsum1 = Q6_Vsf_vadd_VsfVsf(vsum1, s_sf1);
      }
      vsqsum0 = Q6_Vsf_vadd_VsfVsf(vsqsum0, Q6_Vsf_vmpy_VsfVsf(s_sf0, s_sf0));
      vsqsum1 = Q6_Vsf_vadd_VsfVsf(vsqsum1, Q6_Vsf_vmpy_VsfVsf(s_sf1, s_sf1));
  }

  if (fullUnit > 0) {
      HVX_Vector vsum = Q6_Vsf_vadd_VsfVsf(vsum0, vsum1);
      HVX_Vector vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum0, vsqsum1);

      vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 64));
      vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 32));
      vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 16));
      vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 8));
      vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 4));

      vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 64));
      vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 32));
      vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 16));
      vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 8));
      vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 4));

      union { HVX_Vector v; float f[32]; } usum = { .v = vsum };
      union { HVX_Vector v; float f[32]; } usqsum = { .v = vsqsum };

      if (!RMSNorm) sum += usum.f[0];
      sqsum += usqsum.f[0];
  }

  #pragma clang loop vectorize(disable)
  for (int i = fullUnit * 64; i < innerSize; ++i) {
    float val = (float) s[i];
    if (!RMSNorm) sum += val;
    sqsum += val * val;
  }
  if (!RMSNorm) mean = sum / innerSize;
  variance = sqsum / innerSize;
  if (!RMSNorm) variance = variance - mean * mean;
  float inv_std = 1.0f / __builtin_sqrtf(variance + state->epsilon);

  _Float16 *d = (_Float16 *) dstInner;
  const float *g = (const float *) state->gammaBase;
  const float *b = (const float *) state->betaBase;

  union { float f; int32_t i; } u_mean = { .f = mean };
  union { float f; int32_t i; } u_inv_std = { .f = inv_std };
  HVX_Vector vmean = Q6_V_vsplat_R(u_mean.i);
  HVX_Vector vinv_std = Q6_V_vsplat_R(u_inv_std.i);

  const HVX_Vector *pv_in2 = (const HVX_Vector *) s;
  const HVX_Vector *pv_g = (const HVX_Vector *) g;
  const HVX_Vector *pv_b = (const HVX_Vector *) b;

  for (int i = 0; i < fullUnit; ++i) {
      if (i % PREFETCH_N_VECS == 0) {
          int prefetch_idx = i + PREFETCH_N_VECS;
          if (prefetch_idx < n_vecs) {
              int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
              l2fetch(pv_in2 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
              if (pv_g) {
                  l2fetch(pv_g + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
              }
              if (pv_b) {
                  l2fetch(pv_b + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
              }
          }
      }

      HVX_Vector s_hf = *((HVX_UVector *)(s + i * 64));
      HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(s_hf);
      HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
      HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);

      HVX_Vector out0 = Q6_Vsf_vsub_VsfVsf(s_sf0, vmean);
      HVX_Vector out1 = Q6_Vsf_vsub_VsfVsf(s_sf1, vmean);
      out0 = Q6_Vsf_vmpy_VsfVsf(out0, vinv_std);
      out1 = Q6_Vsf_vmpy_VsfVsf(out1, vinv_std);

      if (g) {
          HVX_Vector vg0 = *((HVX_UVector *)(g + i * 64));
          HVX_Vector vg1 = *((HVX_UVector *)(g + i * 64 + 32));
          HVX_VectorPair g_deal = Q6_W_vdeal_VVR(vg1, vg0, -4);
          out0 = Q6_Vsf_vmpy_VsfVsf(out0, Q6_V_lo_W(g_deal));
          out1 = Q6_Vsf_vmpy_VsfVsf(out1, Q6_V_hi_W(g_deal));
      }
      if (b) {
          HVX_Vector vb0 = *((HVX_UVector *)(b + i * 64));
          HVX_Vector vb1 = *((HVX_UVector *)(b + i * 64 + 32));
          HVX_VectorPair b_deal = Q6_W_vdeal_VVR(vb1, vb0, -4);
          out0 = Q6_Vsf_vadd_VsfVsf(out0, Q6_V_lo_W(b_deal));
          out1 = Q6_Vsf_vadd_VsfVsf(out1, Q6_V_hi_W(b_deal));
      }

      HVX_Vector out_hf = Q6_Vhf_vcvt_VsfVsf(out0, out1);
      *((HVX_UVector *)(d + i * 64)) = out_hf;
  }

  #pragma clang loop vectorize(disable)
  for (int i = fullUnit * 64; i < innerSize; ++i) {
    float val = (float) s[i];
    float norm_val = (val - mean) * inv_std;
    if (g) norm_val *= g[i];
    if (b) norm_val += b[i];
    d[i] = (_Float16) norm_val;
  }
}

static inline void htp_ops_layer_norm_inner16_one_batch(const HtpOpsLayerNormTaskState* state, int o) {
  const _Float16* s = (const _Float16*)(state->srcBase + (size_t)o * 16 * sizeof(__fp16));
  _Float16* d = (_Float16*)(state->dstBase + (size_t)o * 16 * sizeof(__fp16));
  const float* g = (const float*)state->gammaBase;
  const float* b = (const float*)state->betaBase;

  float v0 = (float)s[0], v1 = (float)s[1], v2 = (float)s[2], v3 = (float)s[3];
  float v4 = (float)s[4], v5 = (float)s[5], v6 = (float)s[6], v7 = (float)s[7];
  float v8 = (float)s[8], v9 = (float)s[9], v10 = (float)s[10], v11 = (float)s[11];
  float v12 = (float)s[12], v13 = (float)s[13], v14 = (float)s[14], v15 = (float)s[15];

  float sum = 0.0f;
  if (!state->RMSNorm) {
    sum = (v0 + v1) + (v2 + v3) + (v4 + v5) + (v6 + v7) +
          (v8 + v9) + (v10 + v11) + (v12 + v13) + (v14 + v15);
  }
  const float sqsum = (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3) +
                      (v4 * v4 + v5 * v5) + (v6 * v6 + v7 * v7) +
                      (v8 * v8 + v9 * v9) + (v10 * v10 + v11 * v11) +
                      (v12 * v12 + v13 * v13) + (v14 * v14 + v15 * v15);
  const float mean = state->RMSNorm ? 0.0f : sum * 0.0625f;
  float variance = sqsum * 0.0625f;
  if (!state->RMSNorm) {
    variance -= mean * mean;
  }
  const float inv_std = htp_ops_layernorm_fast_rsqrtf(variance + state->epsilon);

#define HTP_LN16_STORE_BASE(i, value)              \
  do {                                             \
    float norm_val = ((value) - mean) * inv_std;   \
    d[(i)] = (_Float16)norm_val;                   \
  } while (0)
#define HTP_LN16_STORE_G(i, value)                 \
  do {                                             \
    float norm_val = ((value) - mean) * inv_std;   \
    norm_val *= g[(i)];                            \
    d[(i)] = (_Float16)norm_val;                   \
  } while (0)
#define HTP_LN16_STORE_B(i, value)                 \
  do {                                             \
    float norm_val = ((value) - mean) * inv_std;   \
    norm_val += b[(i)];                            \
    d[(i)] = (_Float16)norm_val;                   \
  } while (0)
#define HTP_LN16_STORE_GB(i, value)                \
  do {                                             \
    float norm_val = ((value) - mean) * inv_std;   \
    norm_val *= g[(i)];                            \
    norm_val += b[(i)];                            \
    d[(i)] = (_Float16)norm_val;                   \
  } while (0)
#define HTP_LN16_STORE_ALL(MACRO)     \
  do {                                \
    MACRO(0, v0);                     \
    MACRO(1, v1);                     \
    MACRO(2, v2);                     \
    MACRO(3, v3);                     \
    MACRO(4, v4);                     \
    MACRO(5, v5);                     \
    MACRO(6, v6);                     \
    MACRO(7, v7);                     \
    MACRO(8, v8);                     \
    MACRO(9, v9);                     \
    MACRO(10, v10);                   \
    MACRO(11, v11);                   \
    MACRO(12, v12);                   \
    MACRO(13, v13);                   \
    MACRO(14, v14);                   \
    MACRO(15, v15);                   \
  } while (0)
  if (g != NULL) {
    if (b != NULL) {
      HTP_LN16_STORE_ALL(HTP_LN16_STORE_GB);
    } else {
      HTP_LN16_STORE_ALL(HTP_LN16_STORE_G);
    }
  } else {
    if (b != NULL) {
      HTP_LN16_STORE_ALL(HTP_LN16_STORE_B);
    } else {
      HTP_LN16_STORE_ALL(HTP_LN16_STORE_BASE);
    }
  }
#undef HTP_LN16_STORE_ALL
#undef HTP_LN16_STORE_GB
#undef HTP_LN16_STORE_B
#undef HTP_LN16_STORE_G
#undef HTP_LN16_STORE_BASE
}

static inline void htp_ops_layer_norm_inner16_gb_range(const HtpOpsLayerNormTaskState* state,
                                                       int start, int end) {
  const _Float16* srcBase = (const _Float16*)state->srcBase;
  _Float16* dstBase = (_Float16*)state->dstBase;
  const float* g = (const float*)state->gammaBase;
  const float* b = (const float*)state->betaBase;

  const float g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3];
  const float g4 = g[4], g5 = g[5], g6 = g[6], g7 = g[7];
  const float g8 = g[8], g9 = g[9], g10 = g[10], g11 = g[11];
  const float g12 = g[12], g13 = g[13], g14 = g[14], g15 = g[15];
  const float b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const float b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const float b8 = b[8], b9 = b[9], b10 = b[10], b11 = b[11];
  const float b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];

#define HTP_LN16_LOAD_VALUES()                                      \
  float v0 = (float)s[0], v1 = (float)s[1], v2 = (float)s[2], v3 = (float)s[3]; \
  float v4 = (float)s[4], v5 = (float)s[5], v6 = (float)s[6], v7 = (float)s[7]; \
  float v8 = (float)s[8], v9 = (float)s[9], v10 = (float)s[10], v11 = (float)s[11]; \
  float v12 = (float)s[12], v13 = (float)s[13], v14 = (float)s[14], v15 = (float)s[15]

#define HTP_LN16_STORE_GB_CONST(i, value) \
  d[(i)] = (_Float16)(((value) - mean) * inv_std * g##i + b##i)

  int o = start;
  for (; o + 3 < end; o += 4) {
    _Float16* d = dstBase + (size_t)o * 16;
    const _Float16* s = srcBase + (size_t)o * 16;
    HVX_Vector in_hf = vmemu((const HVX_Vector*)s);
    HVX_VectorPair in_sf = Q6_Wsf_vcvt_Vhf(in_hf);
    union { HVX_Vector v; float f[32]; } inEven = { .v = Q6_V_lo_W(in_sf) };
    union { HVX_Vector v; float f[32]; } inOdd = { .v = Q6_V_hi_W(in_sf) };
    union { HVX_Vector v; float f[32]; } outEven;
    union { HVX_Vector v; float f[32]; } outOdd;
    for (int r = 0; r < 4; ++r) {
      const int base = r * 16;
      const int pairBase = base / 2;
      float row[16];
      row[0] = inEven.f[pairBase + 0];
      row[1] = inOdd.f[pairBase + 0];
      row[2] = inEven.f[pairBase + 1];
      row[3] = inOdd.f[pairBase + 1];
      row[4] = inEven.f[pairBase + 2];
      row[5] = inOdd.f[pairBase + 2];
      row[6] = inEven.f[pairBase + 3];
      row[7] = inOdd.f[pairBase + 3];
      row[8] = inEven.f[pairBase + 4];
      row[9] = inOdd.f[pairBase + 4];
      row[10] = inEven.f[pairBase + 5];
      row[11] = inOdd.f[pairBase + 5];
      row[12] = inEven.f[pairBase + 6];
      row[13] = inOdd.f[pairBase + 6];
      row[14] = inEven.f[pairBase + 7];
      row[15] = inOdd.f[pairBase + 7];
      const float sum = (row[0] + row[1]) + (row[2] + row[3]) + (row[4] + row[5]) + (row[6] + row[7]) +
                        (row[8] + row[9]) + (row[10] + row[11]) + (row[12] + row[13]) + (row[14] + row[15]);
      const float sqsum = (row[0] * row[0] + row[1] * row[1]) + (row[2] * row[2] + row[3] * row[3]) +
                          (row[4] * row[4] + row[5] * row[5]) + (row[6] * row[6] + row[7] * row[7]) +
                          (row[8] * row[8] + row[9] * row[9]) + (row[10] * row[10] + row[11] * row[11]) +
                          (row[12] * row[12] + row[13] * row[13]) + (row[14] * row[14] + row[15] * row[15]);
      const float mean = sum * 0.0625f;
      const float variance = sqsum * 0.0625f - mean * mean;
      const float inv_std = htp_ops_layernorm_fast_rsqrtf(variance + state->epsilon);
      outEven.f[pairBase + 0] = (row[0] - mean) * inv_std * g0 + b0;
      outOdd.f[pairBase + 0] = (row[1] - mean) * inv_std * g1 + b1;
      outEven.f[pairBase + 1] = (row[2] - mean) * inv_std * g2 + b2;
      outOdd.f[pairBase + 1] = (row[3] - mean) * inv_std * g3 + b3;
      outEven.f[pairBase + 2] = (row[4] - mean) * inv_std * g4 + b4;
      outOdd.f[pairBase + 2] = (row[5] - mean) * inv_std * g5 + b5;
      outEven.f[pairBase + 3] = (row[6] - mean) * inv_std * g6 + b6;
      outOdd.f[pairBase + 3] = (row[7] - mean) * inv_std * g7 + b7;
      outEven.f[pairBase + 4] = (row[8] - mean) * inv_std * g8 + b8;
      outOdd.f[pairBase + 4] = (row[9] - mean) * inv_std * g9 + b9;
      outEven.f[pairBase + 5] = (row[10] - mean) * inv_std * g10 + b10;
      outOdd.f[pairBase + 5] = (row[11] - mean) * inv_std * g11 + b11;
      outEven.f[pairBase + 6] = (row[12] - mean) * inv_std * g12 + b12;
      outOdd.f[pairBase + 6] = (row[13] - mean) * inv_std * g13 + b13;
      outEven.f[pairBase + 7] = (row[14] - mean) * inv_std * g14 + b14;
      outOdd.f[pairBase + 7] = (row[15] - mean) * inv_std * g15 + b15;
    }
    HVX_Vector out_hf = Q6_Vhf_vcvt_VsfVsf(outEven.v, outOdd.v);
    vmemu((HVX_Vector*)d) = out_hf;
  }

  for (; o < end; ++o) {
    const _Float16* s = srcBase + (size_t)o * 16;
    _Float16* d = dstBase + (size_t)o * 16;
    HTP_LN16_LOAD_VALUES();
    const float sum = (v0 + v1) + (v2 + v3) + (v4 + v5) + (v6 + v7) +
                      (v8 + v9) + (v10 + v11) + (v12 + v13) + (v14 + v15);
    const float sqsum = (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3) +
                        (v4 * v4 + v5 * v5) + (v6 * v6 + v7 * v7) +
                        (v8 * v8 + v9 * v9) + (v10 * v10 + v11 * v11) +
                        (v12 * v12 + v13 * v13) + (v14 * v14 + v15 * v15);
    const float mean = sum * 0.0625f;
    const float variance = sqsum * 0.0625f - mean * mean;
    const float inv_std = htp_ops_layernorm_fast_rsqrtf(variance + state->epsilon);
    HTP_LN16_STORE_GB_CONST(0, v0);
    HTP_LN16_STORE_GB_CONST(1, v1);
    HTP_LN16_STORE_GB_CONST(2, v2);
    HTP_LN16_STORE_GB_CONST(3, v3);
    HTP_LN16_STORE_GB_CONST(4, v4);
    HTP_LN16_STORE_GB_CONST(5, v5);
    HTP_LN16_STORE_GB_CONST(6, v6);
    HTP_LN16_STORE_GB_CONST(7, v7);
    HTP_LN16_STORE_GB_CONST(8, v8);
    HTP_LN16_STORE_GB_CONST(9, v9);
    HTP_LN16_STORE_GB_CONST(10, v10);
    HTP_LN16_STORE_GB_CONST(11, v11);
    HTP_LN16_STORE_GB_CONST(12, v12);
    HTP_LN16_STORE_GB_CONST(13, v13);
    HTP_LN16_STORE_GB_CONST(14, v14);
    HTP_LN16_STORE_GB_CONST(15, v15);
  }

#undef HTP_LN16_STORE_GB_CONST
#undef HTP_LN16_LOAD_VALUES
}

static inline void htp_ops_layer_norm_inner16_range(const HtpOpsLayerNormTaskState* state,
                                                    int start, int end) {
  if (!state->RMSNorm && state->gammaBase != NULL && state->betaBase != NULL) {
    htp_ops_layer_norm_inner16_gb_range(state, start, end);
    return;
  }
  for (int o = start; o < end; ++o) {
    htp_ops_layer_norm_inner16_one_batch(state, o);
  }
}

static void htp_ops_layer_norm_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsLayerNormTaskState* state = (HtpOpsLayerNormTaskState*)data;
  while (1) {
    const int task_id = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int start = task_id * state->grain;
    if (start >= state->outterSize) {
      break;
    }
    int end = start + state->grain;
    if (end > state->outterSize) {
      end = state->outterSize;
    }
    if (state->innerSize == 16) {
      htp_ops_layer_norm_inner16_range(state, start, end);
    } else {
      for (int o = start; o < end; ++o) {
        htp_ops_layer_norm_one_batch(state, o);
      }
    }
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

AEEResult htp_ops_layer_norm(uint8_t* dst, uint8_t* src,
                             uint8_t* gamma, uint8_t* beta,
                             int32_t outterSize, int32_t innerSize, float epsilon, int32_t RMSNorm) {
  if (outterSize <= 0 || innerSize <= 0) return 0;

  HtpOpsLayerNormTaskState state = {};
  state.outterSize = outterSize;
  state.innerSize = innerSize;
  state.epsilon = epsilon;
  state.RMSNorm = RMSNorm;
  state.dstBase = dst;
  state.srcBase = src;
  state.gammaBase = gamma;
  state.betaBase = beta;

  const int n_tasks = htp_ops_layer_norm_pick_task_count(outterSize, innerSize);
  if (n_tasks <= 1) {
    if (innerSize == 16) {
      htp_ops_layer_norm_inner16_range(&state, 0, outterSize);
    } else {
      for (int o = 0; o < outterSize; ++o) {
        htp_ops_layer_norm_one_batch(&state, o);
      }
    }
    return 0;
  }

  state.n_tasks = n_tasks;
  const int taskDivisor = innerSize == 16 ? n_tasks : n_tasks * 4;
  state.grain = (outterSize + taskDivisor - 1) / taskDivisor;
  if (state.grain < 1) {
    state.grain = 1;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_layer_norm_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  return 0;
}

static inline void htp_ops_layer_norm_packed_one_batch(const HtpOpsLayerNormPackedTaskState* state, int n) {
  const int batch = state->batch;
  const int channels = state->channels;
  const int pack = state->pack;
  const int fullUnit = state->fullUnit;
  const int tail = state->tail;
  const int loopUnit = state->loopUnit;
  const int RMSNorm = state->RMSNorm;
  uint8_t* dstBase = state->dstBase;
  const uint8_t* srcBase = state->srcBase;
  const uint8_t* gammaBase = state->gammaBase;
  const uint8_t* betaBase = state->betaBase;
  HVX_VectorPred q_tail = Q6_Q_vsetq_R(tail * 2);
  HVX_Vector vzero = Q6_V_vsplat_R(0);

  HVX_Vector vsum0 = Q6_V_vsplat_R(0);
  HVX_Vector vsum1 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum0 = Q6_V_vsplat_R(0);
  HVX_Vector vsqsum1 = Q6_V_vsplat_R(0);

  for (int c = 0; c < loopUnit; ++c) {
    HVX_Vector s_hf = *(HVX_Vector *)(srcBase + (size_t)(c * batch + n) * pack * 2);

    if (c == fullUnit && tail != 0) {
      s_hf = Q6_V_vmux_QVV(q_tail, s_hf, vzero);
    }

    HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(s_hf);
    HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
    HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);

    if (!RMSNorm) {
      vsum0 = Q6_Vsf_vadd_VsfVsf(vsum0, s_sf0);
      vsum1 = Q6_Vsf_vadd_VsfVsf(vsum1, s_sf1);
    }
    vsqsum0 = Q6_Vsf_vadd_VsfVsf(vsqsum0, Q6_Vsf_vmpy_VsfVsf(s_sf0, s_sf0));
    vsqsum1 = Q6_Vsf_vadd_VsfVsf(vsqsum1, Q6_Vsf_vmpy_VsfVsf(s_sf1, s_sf1));
  }

  float sum = 0.0f;
  if (!RMSNorm) {
    sum = htp_ops_layernorm_reduce_sum2_f32(vsum0, vsum1);
  }
  float sqsum = htp_ops_layernorm_reduce_sum2_f32(vsqsum0, vsqsum1);

  float mean = 0.0f;
  if (!RMSNorm) mean = sum / channels;
  float var = sqsum / channels;
  if (!RMSNorm) var = var - mean * mean;
  float inv_std = 1.0f / __builtin_sqrtf(var + state->epsilon);

  union { float f; int32_t i; } u_mean = { .f = mean };
  union { float f; int32_t i; } u_inv_std = { .f = inv_std };
  HVX_Vector vmean = Q6_V_vsplat_R(u_mean.i);
  HVX_Vector vinv_std = Q6_V_vsplat_R(u_inv_std.i);

  const HVX_Vector *pv_g = (const HVX_Vector *) gammaBase;
  const HVX_Vector *pv_b = (const HVX_Vector *) betaBase;

  for (int c = 0; c < loopUnit; ++c) {
    if (c % PREFETCH_N_VECS == 0) {
      int prefetch_idx = c + PREFETCH_N_VECS;
      if (prefetch_idx < loopUnit) {
        int prefetch_n_vecs = Q6_R_min_RR(loopUnit - prefetch_idx, PREFETCH_N_VECS);
        if (pv_g) {
          l2fetch(pv_g + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        }
        if (pv_b) {
          l2fetch(pv_b + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        }
      }
    }

    HVX_Vector s_hf = *(HVX_Vector *)(srcBase + (size_t)(c * batch + n) * pack * 2);
    HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(s_hf);
    HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
    HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);

    HVX_Vector out0 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s_sf0, vmean), vinv_std);
    HVX_Vector out1 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s_sf1, vmean), vinv_std);

    if (gammaBase) {
      HVX_Vector vg0 = *(HVX_Vector *)(gammaBase + c * pack * 4);
      HVX_Vector vg1 = *(HVX_Vector *)(gammaBase + c * pack * 4 + 128);
      HVX_VectorPair g_deal = Q6_W_vdeal_VVR(vg1, vg0, -4);
      out0 = Q6_Vsf_vmpy_VsfVsf(out0, Q6_V_lo_W(g_deal));
      out1 = Q6_Vsf_vmpy_VsfVsf(out1, Q6_V_hi_W(g_deal));
    }
    if (betaBase) {
      HVX_Vector vb0 = *(HVX_Vector *)(betaBase + c * pack * 4);
      HVX_Vector vb1 = *(HVX_Vector *)(betaBase + c * pack * 4 + 128);
      HVX_VectorPair b_deal = Q6_W_vdeal_VVR(vb1, vb0, -4);
      out0 = Q6_Vsf_vadd_VsfVsf(out0, Q6_V_lo_W(b_deal));
      out1 = Q6_Vsf_vadd_VsfVsf(out1, Q6_V_hi_W(b_deal));
    }

    HVX_Vector out_hf = Q6_Vhf_vcvt_VsfVsf(out0, out1);
    *(HVX_Vector *)(dstBase + (size_t)(c * batch + n) * pack * 2) = out_hf;
  }
}

static void htp_ops_layer_norm_packed_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsLayerNormPackedTaskState* state = (HtpOpsLayerNormPackedTaskState*)data;
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= state->batch) {
      break;
    }
    htp_ops_layer_norm_packed_one_batch(state, (int)task_id);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

AEEResult htp_ops_layer_norm_packed(uint8_t* dst, uint8_t* src,
                                    uint8_t* gamma, uint8_t* beta,
                                    int32_t batch, int32_t channels, float epsilon, int32_t RMSNorm) {
  if (batch <= 0 || channels <= 0) return 0;
  int pack = 4;
#ifdef __HVX_LENGTH__
  pack = __HVX_LENGTH__ / (int32_t)sizeof(int16_t);
#endif

  uint8_t *dstBase = dst;
  const uint8_t *srcBase = src;
  const uint8_t *gammaBase = gamma;
  const uint8_t *betaBase = beta;

  int fullUnit = channels / pack;
  int tail = channels % pack;
  int loopUnit = (channels + pack - 1) / pack;

  HtpOpsLayerNormPackedTaskState state = {};
  state.batch = batch;
  state.channels = channels;
  state.pack = pack;
  state.fullUnit = fullUnit;
  state.tail = tail;
  state.loopUnit = loopUnit;
  state.epsilon = epsilon;
  state.RMSNorm = RMSNorm;
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.gammaBase = gammaBase;
  state.betaBase = betaBase;

  const int n_tasks = htp_ops_layer_norm_packed_pick_task_count(batch);
  if (n_tasks <= 1) {
    for (int n = 0; n < batch; ++n) {
      htp_ops_layer_norm_packed_one_batch(&state, n);
    }
  } else {
    worker_pool_job_t job;
    job.fptr = htp_ops_layer_norm_packed_worker;
    job.dptr = &state;
    worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
    for (int i = 0; i < n_tasks; ++i) {
      worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&(state.sync_ctx));
  }
  return 0;
}

AEEResult htp_ops_add_fuse_layernorm(uint8_t* dst, uint8_t* add_out, uint8_t* src0, uint8_t* src1,
                                     uint8_t* gamma, uint8_t* beta,
                                     int32_t batch, int32_t channels, float epsilon, int32_t RMSNorm) {
  if (batch <= 0 || channels <= 0) return 0;
  int pack = __HVX_LENGTH__ / (int32_t)sizeof(int16_t);
  int fullUnit = channels / pack;
  int tail = channels % pack;
  int loopUnit = (channels + pack - 1) / pack;

  __fp16* dstBase = (__fp16*)dst;
  __fp16* addOutBase = (__fp16*)add_out;
  const __fp16* src0Base = (const __fp16*)src0;
  const __fp16* src1Base = (const __fp16*)src1;
  const float* gammaBase = (const float*)gamma;
  const float* betaBase = (const float*)beta;

  size_t dataBytes = (size_t)batch * loopUnit * pack * 2; (void)dataBytes;
  size_t paramBytes = (size_t)loopUnit * pack * 4; (void)paramBytes;

  uint8_t *vtcm_ptr = (uint8_t *)vtcm_manager_get_vtcm_base();
  float *gammaVtcm = NULL;
  float *betaVtcm = NULL;
  if (gammaBase) {
    gammaVtcm = (float*)vtcm_seq_alloc(&vtcm_ptr, paramBytes);
  }
  if (betaBase) {
    betaVtcm = (float*)vtcm_seq_alloc(&vtcm_ptr, paramBytes);
  }
  if ((gammaBase && gammaVtcm == NULL) || (betaBase && betaVtcm == NULL)) {
    return AEE_ENOMEMORY;
  }
  if (gammaVtcm) {
    memcpy(gammaVtcm, gammaBase, paramBytes);
    gammaBase = gammaVtcm;
  }
  if (betaVtcm) {
    memcpy(betaVtcm, betaBase, paramBytes);
    betaBase = betaVtcm;
  }

  if (batch == 1 && RMSNorm && betaBase == NULL && gammaBase != NULL) {
    htp_ops_add_fuse_rmsnorm_one_batch_no_beta(dstBase, addOutBase, src0Base, src1Base,
                                               gammaBase, batch, 0, channels, pack, fullUnit, tail,
                                               loopUnit, epsilon, NULL);
    return 0;
  }

  const int n_tasks = htp_ops_add_fuse_layernorm_pick_task_count(batch);
  const int worker_slots = n_tasks <= 1 ? 1 : (g_max_num_workers > 0 ? (int)g_max_num_workers : n_tasks);
  const size_t worker_vtcm_bytes = (size_t)2 * loopUnit * pack * sizeof(__fp16);
  uint8_t* workspace_base = vtcm_seq_alloc(&vtcm_ptr, (size_t)worker_slots * worker_vtcm_bytes);
  if (workspace_base == NULL) {
    return AEE_ENOMEMORY;
  }

  if (n_tasks <= 1) {
    uint8_t* worker_vtcm_ptr = workspace_base;
    __fp16* srcAdd0 = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)loopUnit * pack * sizeof(__fp16));
    __fp16* srcAdd1 = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)loopUnit * pack * sizeof(__fp16));
    const int total_blocks = (batch + 1) / 2;
    HtpOpsAddFuseLayernormTaskState state = {};
    state.batch = batch;
    state.channels = channels;
    state.pack = pack;
    state.fullUnit = fullUnit;
    state.tail = tail;
    state.loopUnit = loopUnit;
    state.epsilon = epsilon;
    state.RMSNorm = RMSNorm;
    state.dstBase = dstBase;
    state.addOutBase = addOutBase;
    state.src0Base = src0Base;
    state.src1Base = src1Base;
    state.gammaBase = gammaBase;
    state.betaBase = betaBase;
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
      htp_ops_add_fuse_layernorm_process_batch_block(&state, block_idx, srcAdd0, srcAdd1);
    }
  } else {
    HtpOpsAddFuseLayernormTaskState state = {};
    state.task_id = 0;
    state.n_tasks = n_tasks;
    state.batch = batch;
    state.channels = channels;
    state.pack = pack;
    state.fullUnit = fullUnit;
    state.tail = tail;
    state.loopUnit = loopUnit;
    state.epsilon = epsilon;
    state.RMSNorm = RMSNorm;
    state.worker_vtcm_bytes = worker_vtcm_bytes;
    state.workspace_base = workspace_base;
    state.dstBase = dstBase;
    state.addOutBase = addOutBase;
    state.src0Base = src0Base;
    state.src1Base = src1Base;
    state.gammaBase = gammaBase;
    state.betaBase = betaBase;

    worker_pool_job_t job;
    job.fptr = htp_ops_add_fuse_layernorm_worker;
    job.dptr = &state;

    worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
    for (int i = 0; i < n_tasks; ++i) {
      worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&(state.sync_ctx));
  }

  return 0;
}
}  // extern "C"
