#include "dsp/mmap_mgr.h"
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#include <remote.h>
#include "region_ops.h"
#include "dsp/dma_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_utils.h"
#include "dsp/hvx_math.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

extern "C" {


// ===== binary elementwise ops =====

typedef enum {
  HTP_OPS_BINARY_ADD = 1,
  HTP_OPS_BINARY_SUB = 2,
  HTP_OPS_BINARY_MUL = 3,
  HTP_OPS_BINARY_DIV = 4,
  HTP_OPS_BINARY_MAX = 5,
  HTP_OPS_BINARY_MIN = 6,
  HTP_OPS_BINARY_MUL_SILU = 7,
  HTP_OPS_BINARY_ADD_RELU = 8,
  HTP_OPS_BINARY_GREATER = 9,
  HTP_OPS_BINARY_LESS = 10,
  HTP_OPS_BINARY_SQUARED_DIFFERENCE = 11,
} HtpOpsBinaryOpType;

#define HTP_OPS_BINARY_DMA_CHUNK_BYTES (16 * 1024)
#define HTP_OPS_BINARY_DMA_NUM_BUFS 2
#define HTP_OPS_BINARY_MT_MIN_FP16_ELEMS 2048
#define HTP_OPS_BINARY_MT_MIN_INT32_ELEMS 1024
#define HTP_OPS_BINARY_MT_FP16_GRAIN_ELEMS (128 / (int)sizeof(__fp16))
#define HTP_OPS_BINARY_MT_INT32_GRAIN_ELEMS 128

typedef enum {
  HTP_OPS_BINARY_TASK_FP16 = 0,
  HTP_OPS_BINARY_TASK_FP16_RHS_SCALAR = 1,
  HTP_OPS_BINARY_TASK_FP16_LHS_SCALAR = 2,
  HTP_OPS_BINARY_TASK_INT32 = 3,
  HTP_OPS_BINARY_TASK_INT32_RHS_SCALAR = 4,
  HTP_OPS_BINARY_TASK_INT32_LHS_SCALAR = 5,
  HTP_OPS_BINARY_TASK_CMP_FP16_TO_INT32 = 6,
  HTP_OPS_BINARY_TASK_CMP_FP16_RHS_SCALAR_TO_INT32 = 7,
  HTP_OPS_BINARY_TASK_CMP_FP16_LHS_SCALAR_TO_INT32 = 8,
} HtpOpsBinaryTaskKind;

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  int                size;
  int                grain;
  int                opType;
  HtpOpsBinaryTaskKind kind;
  __fp16*            fp16_dst;
  const __fp16*      fp16_src0;
  const __fp16*      fp16_src1;
  __fp16             fp16_scalar;
  int32_t*           int32_dst;
  const int32_t*     int32_src0;
  const int32_t*     int32_src1;
  int32_t            int32_scalar;
} HtpOpsBinaryTaskState;

static inline void htp_ops_binary_compute_fp16_chunk(__fp16* __restrict dst,
                                                     const __fp16* __restrict src0,
                                                     const __fp16* __restrict src1,
                                                     int size, int opType);
static inline void htp_ops_binary_compute_fp16_rhs_scalar_chunk(__fp16* dst, const __fp16* src0,
                                                                _Float16 val1, int size, int opType,
                                                                HVX_Vector (*memLoadFunc)(const HVX_Vector*));
static inline void htp_ops_binary_compute_fp16_lhs_scalar_chunk(__fp16* dst, _Float16 val0,
                                                                const __fp16* src1, int size, int opType,
                                                                HVX_Vector (*memLoadFunc)(const HVX_Vector*));
static inline void htp_ops_binary_compute_int32_chunk(int32_t* dst, const int32_t* src0,
                                                      const int32_t* src1, int size, int opType);
static inline void htp_ops_binary_compute_int32_rhs_scalar_chunk(int32_t* dst, const int32_t* src0,
                                                                 int32_t val1, int size, int opType);
static inline void htp_ops_binary_compute_int32_lhs_scalar_chunk(int32_t* dst, int32_t val0,
                                                                 const int32_t* src1, int size, int opType);

static inline bool htp_ops_binary_is_aligned_128(const void* ptr) {
  return (((uintptr_t)ptr) & 127) == 0;
}

typedef HVX_Vector (*HtpOpsBinaryLoadFunc)(const HVX_Vector*);

static inline HVX_Vector htp_ops_binary_load_aligned(const HVX_Vector* ptr) {
  return vmem(ptr);
}

static inline HVX_Vector htp_ops_binary_load_unaligned(const HVX_Vector* ptr) {
  return vmemu(ptr);
}

static inline HtpOpsBinaryLoadFunc htp_ops_binary_pick_load_func(const void* ptr) {
  return htp_ops_binary_is_aligned_128(ptr) ? htp_ops_binary_load_aligned : htp_ops_binary_load_unaligned;
}

static inline _Float16 htp_ops_binary_relu_fp16_scalar(_Float16 value) {
  uint16_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x8000) != 0 && (bits & 0x7fff) != 0) {
    return (_Float16)0.0f;
  }
  return value;
}

static inline _Float16 htp_ops_binary_apply_fp16(_Float16 a, _Float16 b, int opType) {
  switch (opType) {
    case HTP_OPS_BINARY_ADD:
      return a + b;
    case HTP_OPS_BINARY_ADD_RELU:
      return htp_ops_binary_relu_fp16_scalar(a + b);
    case HTP_OPS_BINARY_SUB:
      return a - b;
    case HTP_OPS_BINARY_MUL:
      return a * b;
    case HTP_OPS_BINARY_SQUARED_DIFFERENCE: {
      const float v = (float)a - (float)b;
      return (_Float16)(v * v);
    }
    case HTP_OPS_BINARY_DIV:
      return a / b;
    case HTP_OPS_BINARY_MAX:
      return a > b ? a : b;
    case HTP_OPS_BINARY_MIN:
      return a < b ? a : b;
    case HTP_OPS_BINARY_MUL_SILU: {
      float a_f = (float)a;
      float b_f = (float)b;
      float sig_b = 1.0f / (1.0f + expf(-b_f));
      return (_Float16)(a_f * b_f * sig_b);
    }
    default:
      return a;
  }
}

static inline int32_t htp_ops_binary_apply_int32(int32_t a, int32_t b, int opType) {
  switch (opType) {
    case HTP_OPS_BINARY_ADD:
      return a + b;
    case HTP_OPS_BINARY_ADD_RELU: {
      int32_t result = a + b;
      return result < 0 ? 0 : result;
    }
    case HTP_OPS_BINARY_SUB:
      return a - b;
    case HTP_OPS_BINARY_MUL:
      return a * b;
    case HTP_OPS_BINARY_SQUARED_DIFFERENCE: {
      int32_t v = a - b;
      return v * v;
    }
    case HTP_OPS_BINARY_GREATER:
      return a > b ? 1 : 0;
    case HTP_OPS_BINARY_LESS:
      return a < b ? 1 : 0;
    default:
      return a;
  }
}

static inline int32_t htp_ops_binary_compare_fp16(float a, float b, int opType) {
  switch (opType) {
    case HTP_OPS_BINARY_GREATER:
      return a > b ? 1 : 0;
    case HTP_OPS_BINARY_LESS:
      return a < b ? 1 : 0;
    default:
      return 0;
  }
}

static inline bool htp_ops_binary_is_compare(int opType) {
  return opType == HTP_OPS_BINARY_GREATER || opType == HTP_OPS_BINARY_LESS;
}

static inline HVX_Vector htp_ops_binary_compare_fp16_vector(HVX_Vector v0, HVX_Vector v1, int opType) {
  HVX_VectorPred pred = opType == HTP_OPS_BINARY_GREATER ?
      Q6_Q_vcmp_gt_VhfVhf(v0, v1) : Q6_Q_vcmp_gt_VhfVhf(v1, v0);
  return Q6_V_vmux_QVV(pred, Q6_Vh_vsplat_R(0x3c00), Q6_V_vzero());
}

static inline HVX_Vector htp_ops_binary_compare_fp16_i32_vector(HVX_Vector v0, HVX_Vector v1, int opType) {
  HVX_VectorPred pred = opType == HTP_OPS_BINARY_GREATER ?
      Q6_Q_vcmp_gt_VhfVhf(v0, v1) : Q6_Q_vcmp_gt_VhfVhf(v1, v0);
  return Q6_V_vmux_QVV(pred, Q6_Vh_vsplat_R(1), Q6_V_vzero());
}

static inline void htp_ops_binary_compare_fp16_to_fp16_chunk(__fp16* out, const __fp16* src0,
                                                             const __fp16* src1, int size, int opType) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v0 = vmemu((const HVX_Vector*)(src0 + i));
    HVX_Vector v1 = vmemu((const HVX_Vector*)(src1 + i));
    vmemu((HVX_Vector*)(out + i)) = htp_ops_binary_compare_fp16_vector(v0, v1, opType);
  }
  if (i < size) {
    __fp16 tmp0[vec_len] __attribute__((aligned(128))) = {};
    __fp16 tmp1[vec_len] __attribute__((aligned(128))) = {};
    __fp16 tmpOut[vec_len] __attribute__((aligned(128)));
    const int remain = size - i;
    memcpy(tmp0, src0 + i, remain * sizeof(__fp16));
    memcpy(tmp1, src1 + i, remain * sizeof(__fp16));
    vmem(tmpOut) = htp_ops_binary_compare_fp16_vector(vmem(tmp0), vmem(tmp1), opType);
    memcpy(out + i, tmpOut, remain * sizeof(__fp16));
  }
}

static inline void htp_ops_binary_store_u16_to_i32(int32_t* out, HVX_Vector v) {
  HVX_VectorPair vp = Q6_Wuw_vunpack_Vuh(v);
  vmemu((HVX_Vector*)out) = Q6_V_lo_W(vp);
  vmemu((HVX_Vector*)(out + 32)) = Q6_V_hi_W(vp);
}

static inline void htp_ops_binary_compare_fp16_to_int32_chunk(int32_t* out, const __fp16* src0,
                                                              const __fp16* src1, int size, int opType) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v0 = vmemu((const HVX_Vector*)(src0 + i));
    HVX_Vector v1 = vmemu((const HVX_Vector*)(src1 + i));
    htp_ops_binary_store_u16_to_i32(out + i, htp_ops_binary_compare_fp16_i32_vector(v0, v1, opType));
  }
  if (i < size) {
    __fp16 tmp0[vec_len] __attribute__((aligned(128))) = {};
    __fp16 tmp1[vec_len] __attribute__((aligned(128))) = {};
    int32_t tmpOut[vec_len] __attribute__((aligned(128)));
    const int remain = size - i;
    memcpy(tmp0, src0 + i, remain * sizeof(__fp16));
    memcpy(tmp1, src1 + i, remain * sizeof(__fp16));
    htp_ops_binary_store_u16_to_i32(tmpOut, htp_ops_binary_compare_fp16_i32_vector(vmem(tmp0), vmem(tmp1), opType));
    memcpy(out + i, tmpOut, remain * sizeof(int32_t));
  }
}

static inline void htp_ops_binary_compare_fp16_rhs_scalar_to_int32_chunk(int32_t* out, const __fp16* src0,
                                                                         uint16_t val1_u16, int size, int opType) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  HVX_Vector v1 = Q6_Vh_vsplat_R(val1_u16);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v0 = vmemu((const HVX_Vector*)(src0 + i));
    htp_ops_binary_store_u16_to_i32(out + i, htp_ops_binary_compare_fp16_i32_vector(v0, v1, opType));
  }
  if (i < size) {
    __fp16 tmp0[vec_len] __attribute__((aligned(128))) = {};
    int32_t tmpOut[vec_len] __attribute__((aligned(128)));
    const int remain = size - i;
    memcpy(tmp0, src0 + i, remain * sizeof(__fp16));
    htp_ops_binary_store_u16_to_i32(tmpOut, htp_ops_binary_compare_fp16_i32_vector(vmem(tmp0), v1, opType));
    memcpy(out + i, tmpOut, remain * sizeof(int32_t));
  }
}

static inline void htp_ops_binary_compare_fp16_lhs_scalar_to_int32_chunk(int32_t* out, uint16_t val0_u16,
                                                                         const __fp16* src1, int size, int opType) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  HVX_Vector v0 = Q6_Vh_vsplat_R(val0_u16);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v1 = vmemu((const HVX_Vector*)(src1 + i));
    htp_ops_binary_store_u16_to_i32(out + i, htp_ops_binary_compare_fp16_i32_vector(v0, v1, opType));
  }
  if (i < size) {
    __fp16 tmp1[vec_len] __attribute__((aligned(128))) = {};
    int32_t tmpOut[vec_len] __attribute__((aligned(128)));
    const int remain = size - i;
    memcpy(tmp1, src1 + i, remain * sizeof(__fp16));
    htp_ops_binary_store_u16_to_i32(tmpOut, htp_ops_binary_compare_fp16_i32_vector(v0, vmem(tmp1), opType));
    memcpy(out + i, tmpOut, remain * sizeof(int32_t));
  }
}

static inline void htp_ops_binary_compare_fp16_rhs_scalar_to_fp16_chunk(__fp16* out, const __fp16* src0,
                                                                        uint16_t val1_u16, int size, int opType) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  HVX_Vector v1 = Q6_Vh_vsplat_R(val1_u16);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v0 = vmemu((const HVX_Vector*)(src0 + i));
    vmemu((HVX_Vector*)(out + i)) = htp_ops_binary_compare_fp16_vector(v0, v1, opType);
  }
  if (i < size) {
    __fp16 tmp0[vec_len] __attribute__((aligned(128))) = {};
    __fp16 tmpOut[vec_len] __attribute__((aligned(128)));
    const int remain = size - i;
    memcpy(tmp0, src0 + i, remain * sizeof(__fp16));
    vmem(tmpOut) = htp_ops_binary_compare_fp16_vector(vmem(tmp0), v1, opType);
    memcpy(out + i, tmpOut, remain * sizeof(__fp16));
  }
}

static inline void htp_ops_binary_compare_fp16_lhs_scalar_to_fp16_chunk(__fp16* out, uint16_t val0_u16,
                                                                        const __fp16* src1, int size, int opType) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  HVX_Vector v0 = Q6_Vh_vsplat_R(val0_u16);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v1 = vmemu((const HVX_Vector*)(src1 + i));
    vmemu((HVX_Vector*)(out + i)) = htp_ops_binary_compare_fp16_vector(v0, v1, opType);
  }
  if (i < size) {
    __fp16 tmp1[vec_len] __attribute__((aligned(128))) = {};
    __fp16 tmpOut[vec_len] __attribute__((aligned(128)));
    const int remain = size - i;
    memcpy(tmp1, src1 + i, remain * sizeof(__fp16));
    vmem(tmpOut) = htp_ops_binary_compare_fp16_vector(v0, vmem(tmp1), opType);
    memcpy(out + i, tmpOut, remain * sizeof(__fp16));
  }
}

static inline void htp_ops_binary_dispatch_task_range(HtpOpsBinaryTaskState* state, int offset, int count) {
  if (count <= 0) {
    return;
  }
  switch (state->kind) {
    case HTP_OPS_BINARY_TASK_FP16:
      htp_ops_binary_compute_fp16_chunk(state->fp16_dst + offset, state->fp16_src0 + offset,
                                        state->fp16_src1 + offset, count, state->opType);
      break;
    case HTP_OPS_BINARY_TASK_FP16_RHS_SCALAR:
      htp_ops_binary_compute_fp16_rhs_scalar_chunk(state->fp16_dst + offset, state->fp16_src0 + offset,
                                                   state->fp16_scalar, count, state->opType,
                                                   htp_ops_binary_pick_load_func(state->fp16_src0 + offset));
      break;
    case HTP_OPS_BINARY_TASK_FP16_LHS_SCALAR:
      htp_ops_binary_compute_fp16_lhs_scalar_chunk(state->fp16_dst + offset, state->fp16_scalar,
                                                   state->fp16_src1 + offset, count, state->opType,
                                                   htp_ops_binary_pick_load_func(state->fp16_src1 + offset));
      break;
    case HTP_OPS_BINARY_TASK_INT32:
      htp_ops_binary_compute_int32_chunk(state->int32_dst + offset, state->int32_src0 + offset,
                                         state->int32_src1 + offset, count, state->opType);
      break;
    case HTP_OPS_BINARY_TASK_INT32_RHS_SCALAR:
      htp_ops_binary_compute_int32_rhs_scalar_chunk(state->int32_dst + offset, state->int32_src0 + offset,
                                                    state->int32_scalar, count, state->opType);
      break;
    case HTP_OPS_BINARY_TASK_INT32_LHS_SCALAR:
      htp_ops_binary_compute_int32_lhs_scalar_chunk(state->int32_dst + offset, state->int32_scalar,
                                                    state->int32_src1 + offset, count, state->opType);
      break;
    case HTP_OPS_BINARY_TASK_CMP_FP16_TO_INT32:
      htp_ops_binary_compare_fp16_to_int32_chunk(state->int32_dst + offset, state->fp16_src0 + offset,
                                                 state->fp16_src1 + offset, count, state->opType);
      break;
    case HTP_OPS_BINARY_TASK_CMP_FP16_RHS_SCALAR_TO_INT32: {
      uint16_t value;
      memcpy(&value, &state->fp16_scalar, sizeof(value));
      htp_ops_binary_compare_fp16_rhs_scalar_to_int32_chunk(state->int32_dst + offset,
                                                            state->fp16_src0 + offset,
                                                            value, count, state->opType);
      break;
    }
    case HTP_OPS_BINARY_TASK_CMP_FP16_LHS_SCALAR_TO_INT32: {
      uint16_t value;
      memcpy(&value, &state->fp16_scalar, sizeof(value));
      htp_ops_binary_compare_fp16_lhs_scalar_to_int32_chunk(state->int32_dst + offset,
                                                            value,
                                                            state->fp16_src1 + offset,
                                                            count, state->opType);
      break;
    }
    default:
      break;
  }
}

typedef struct {
  HtpOpsBinaryTaskState* state;
  int start;
  int count;
} HtpOpsBinaryFixedTask;

static void htp_ops_binary_fixed_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsBinaryFixedTask* task = (HtpOpsBinaryFixedTask*)data;
  htp_ops_binary_dispatch_task_range(task->state, task->start, task->count);
  worker_pool_synctoken_jobdone(&(task->state->sync_ctx));
}

static inline int htp_ops_binary_pick_task_count(int size, int bytes) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap <= 1) {
    return 1;
  }

  const int min_elems_per_task = (bytes == 2) ? HTP_OPS_BINARY_MT_MIN_FP16_ELEMS : HTP_OPS_BINARY_MT_MIN_INT32_ELEMS;
  int task_count = (size + min_elems_per_task - 1) / min_elems_per_task;
  if (task_count < 2) {
    return 1;
  }
  if (task_count > (int)worker_cap) {
    task_count = (int)worker_cap;
  }
  return task_count;
}

static inline void htp_ops_binary_run_task(HtpOpsBinaryTaskState* state, int size, int bytes) {
  state->task_id = 0;
  state->size = size;
  state->grain = (bytes == 2) ? HTP_OPS_BINARY_MT_FP16_GRAIN_ELEMS : HTP_OPS_BINARY_MT_INT32_GRAIN_ELEMS;

  const int n_tasks = htp_ops_binary_pick_task_count(size, bytes);
  if (n_tasks <= 1) {
    htp_ops_binary_dispatch_task_range(state, 0, size);
    return;
  }

  state->n_tasks = n_tasks;
  worker_pool_job_t job;
  job.fptr = htp_ops_binary_fixed_worker;

  worker_pool_synctoken_init(&(state->sync_ctx), n_tasks);
  HtpOpsBinaryFixedTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsBinaryFixedTask, n_tasks);
  const int total_blocks = (size + state->grain - 1) / state->grain;
  const int blocks_per_task = (total_blocks + n_tasks - 1) / n_tasks;
  for (int i = 0; i < n_tasks; ++i) {
    const int start_block = i * blocks_per_task;
    int end_block = start_block + blocks_per_task;
    if (end_block > total_blocks) {
      end_block = total_blocks;
    }
    const int start = start_block * state->grain;
    int end = end_block * state->grain;
    if (end > size) {
      end = size;
    }
    tasks[i].state = state;
    tasks[i].start = start;
    tasks[i].count = end - start;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state->sync_ctx));
}

static inline HVX_Vector htp_ops_binary_mul_silu_sf_vec(HVX_Vector v0, HVX_Vector v1) {
  HVX_Vector zero_v = Q6_V_vzero();
  HVX_Vector one_v = Q6_V_vsplat_R(0x3f800000);
  HVX_Vector two_v = Q6_V_vsplat_R(0x40000000);
  HVX_Vector log2e_v = Q6_V_vsplat_R(0x3fb8aa3b);

  HVX_VectorPred q_v1_lt_0 = Q6_Q_vcmp_gt_VsfVsf(zero_v, v1);
  HVX_Vector neg_v1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(zero_v, v1));
  HVX_Vector z = Q6_V_vmux_QVV(q_v1_lt_0, v1, neg_v1);
  HVX_Vector exp_arg = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, log2e_v));
  HVX_Vector exp_val = hvx_my_exp2_vsf(exp_arg);

  HVX_Vector denom = Q6_Vqf32_vadd_VsfVsf(one_v, exp_val);
  HVX_Vector inv_denom = hvx_my_inv_vqf32_vsf(Q6_Vsf_equals_Vqf32(denom));
  HVX_Vector two_qf32 = Q6_Vqf32_vadd_VsfVsf(two_v, zero_v);
  HVX_Vector dy = Q6_Vqf32_vmpy_Vqf32Vqf32(denom, inv_denom);
  HVX_Vector two_minus_dy = Q6_Vqf32_vsub_Vqf32Vqf32(two_qf32, dy);
  inv_denom = Q6_Vqf32_vmpy_Vqf32Vqf32(inv_denom, two_minus_dy);

  HVX_Vector num = Q6_V_vmux_QVV(q_v1_lt_0, exp_val, one_v);
  HVX_Vector num_qf32 = Q6_Vqf32_vadd_VsfVsf(num, zero_v);
  HVX_Vector v1_qf32 = Q6_Vqf32_vadd_VsfVsf(v1, zero_v);
  HVX_Vector v0_qf32 = Q6_Vqf32_vadd_VsfVsf(v0, zero_v);
  HVX_Vector sig_v = Q6_Vqf32_vmpy_Vqf32Vqf32(num_qf32, inv_denom);
  HVX_Vector v1_sig_v = Q6_Vqf32_vmpy_Vqf32Vqf32(v1_qf32, sig_v);
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(v0_qf32, v1_sig_v));
}

static inline HVX_Vector htp_ops_binary_mul_silu_fp16_vec(HVX_Vector v0, HVX_Vector v1) {
  HVX_VectorPair v0_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(v0));
  HVX_VectorPair v1_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(v1));
  HVX_Vector vr0 = htp_ops_binary_mul_silu_sf_vec(Q6_V_lo_W(v0_sf), Q6_V_lo_W(v1_sf));
  HVX_Vector vr1 = htp_ops_binary_mul_silu_sf_vec(Q6_V_hi_W(v0_sf), Q6_V_hi_W(v1_sf));
  return Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(vr0, vr1));
}

static inline HVX_Vector htp_ops_binary_squared_difference_fp16_vec(HVX_Vector v0, HVX_Vector v1) {
  HVX_VectorPair v0_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(v0));
  HVX_VectorPair v1_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(v1));
  HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v0_sf), Q6_V_lo_W(v1_sf));
  HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v0_sf), Q6_V_hi_W(v1_sf));
  HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(d0, d0));
  HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(d1, d1));
  return Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(r0, r1));
}

static inline bool htp_ops_binary_supports_fp16_vector_tail(int opType) {
  return opType == HTP_OPS_BINARY_ADD || opType == HTP_OPS_BINARY_ADD_RELU ||
         opType == HTP_OPS_BINARY_SUB || opType == HTP_OPS_BINARY_MUL ||
         opType == HTP_OPS_BINARY_SQUARED_DIFFERENCE ||
         opType == HTP_OPS_BINARY_DIV || opType == HTP_OPS_BINARY_MAX ||
         opType == HTP_OPS_BINARY_MIN || opType == HTP_OPS_BINARY_MUL_SILU;
}

static inline HVX_Vector htp_ops_binary_compute_fp16_vector(HVX_Vector v0, HVX_Vector v1, int opType) {
  switch (opType) {
    case HTP_OPS_BINARY_ADD:
      return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0, v1));
    case HTP_OPS_BINARY_ADD_RELU: {
      HVX_Vector add = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0, v1));
      return Q6_Vhf_vmax_VhfVhf(add, Q6_V_vzero());
    }
    case HTP_OPS_BINARY_SUB:
      return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v0, v1));
    case HTP_OPS_BINARY_MUL:
      return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, v1));
    case HTP_OPS_BINARY_SQUARED_DIFFERENCE: {
      return htp_ops_binary_squared_difference_fp16_vec(v0, v1);
    }
    case HTP_OPS_BINARY_DIV: {
      HVX_Vector inv_v1 = hvx_my_inv_vhf(v1);
      return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, inv_v1));
    }
    case HTP_OPS_BINARY_MAX:
      return Q6_Vhf_vmax_VhfVhf(v0, v1);
    case HTP_OPS_BINARY_MIN:
      return Q6_Vhf_vmin_VhfVhf(v0, v1);
    case HTP_OPS_BINARY_MUL_SILU: {
      return htp_ops_binary_mul_silu_fp16_vec(v0, v1);
    }
    default:
      return v0;
  }
}

static inline bool htp_ops_binary_store_fp16_tail(__fp16* dst, HVX_Vector vr, int remain) {
  if (remain <= 0) {
    return true;
  }
  if (htp_ops_binary_is_aligned_128(dst)) {
    HVX_VectorPred q = Q6_Q_vsetq_R(remain * (int)sizeof(__fp16));
    Q6_vmem_QRIV(q, (HVX_Vector*)dst, vr);
    return true;
  }

  // Predicate VMEM stores require aligned addresses. For short unaligned rows, keep the math in
  // HVX and copy only the valid bytes from an aligned stack vector.
  __fp16 tmp[128 / (int)sizeof(__fp16)] __attribute__((aligned(128)));
  vmem(tmp) = vr;
  memcpy(dst, tmp, remain * sizeof(__fp16));
  return true;
}

static inline bool htp_ops_binary_compute_fp16_tail(__fp16* dst, const __fp16* src0,
                                                   const __fp16* src1, int remain, int opType) {
  if (!htp_ops_binary_supports_fp16_vector_tail(opType)) {
    return false;
  }
  const int vec_len = 128 / (int)sizeof(__fp16);
  __fp16 tmp0[vec_len] __attribute__((aligned(128))) = {};
  __fp16 tmp1[vec_len] __attribute__((aligned(128))) = {};
  for (int j = 0; j < remain; ++j) {
    tmp0[j] = src0[j];
    tmp1[j] = src1[j];
  }
  HVX_Vector v0 = vmem(tmp0);
  HVX_Vector v1 = vmem(tmp1);
  HVX_Vector vr = htp_ops_binary_compute_fp16_vector(v0, v1, opType);
  return htp_ops_binary_store_fp16_tail(dst, vr, remain);
}

static inline bool htp_ops_binary_compute_fp16_rhs_scalar_tail(__fp16* dst, const __fp16* src0,
                                                               _Float16 val1, int remain, int opType) {
  if (!htp_ops_binary_supports_fp16_vector_tail(opType)) {
    return false;
  }
  const int vec_len = 128 / (int)sizeof(__fp16);
  __fp16 tmp0[vec_len] __attribute__((aligned(128))) = {};
  for (int j = 0; j < remain; ++j) {
    tmp0[j] = src0[j];
  }
  uint16_t v1_u16 = *(uint16_t*)&val1;
  HVX_Vector v0 = vmem(tmp0);
  HVX_Vector v1 = Q6_Vh_vsplat_R(v1_u16);
  HVX_Vector vr = htp_ops_binary_compute_fp16_vector(v0, v1, opType);
  return htp_ops_binary_store_fp16_tail(dst, vr, remain);
}

static inline bool htp_ops_binary_compute_fp16_rhs_scalar_short(__fp16* dst, const __fp16* src0,
                                                                _Float16 val1, int size, int opType,
                                                                bool canOverread) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  if (size <= 0 || size >= vec_len || !htp_ops_binary_supports_fp16_vector_tail(opType)) {
    return false;
  }
  if (!canOverread) {
    return htp_ops_binary_compute_fp16_rhs_scalar_tail(dst, src0, val1, size, opType);
  }
  uint16_t v1_u16 = *(uint16_t*)&val1;
  HVX_Vector v0 = vmemu((const HVX_Vector*)src0);
  HVX_Vector v1 = Q6_Vh_vsplat_R(v1_u16);
  HVX_Vector vr = htp_ops_binary_compute_fp16_vector(v0, v1, opType);
  vstu_variable(dst, (uint32_t)(size * (int)sizeof(__fp16)), vr);
  return true;
}

static inline bool htp_ops_binary_compute_fp16_rhs_scalar_short_pair(__fp16* dst0, const __fp16* src0,
                                                                     _Float16 val0, _Float16 val1,
                                                                     int size, int opType,
                                                                     bool canOverreadTail) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  if (size <= vec_len / 2 || size >= vec_len || !htp_ops_binary_supports_fp16_vector_tail(opType)) {
    return false;
  }

  uint16_t v0_u16 = *(uint16_t*)&val0;
  uint16_t v1_u16 = *(uint16_t*)&val1;
  HVX_Vector s0 = Q6_Vh_vsplat_R(v0_u16);
  HVX_Vector s1 = Q6_Vh_vsplat_R(v1_u16);
  HVX_VectorPred q0 = Q6_Q_vsetq_R(size * (int)sizeof(__fp16));
  HVX_Vector smix = Q6_V_vmux_QVV(q0, s0, s1);
  HVX_Vector a0 = vmemu((const HVX_Vector*)src0);
  HVX_Vector r0 = htp_ops_binary_compute_fp16_vector(a0, smix, opType);
  vmemu((HVX_Vector*)dst0) = r0;

  const int overlap = vec_len - size;
  const int tail = size - overlap;
  if (tail <= 0) {
    return true;
  }
  __fp16* dstTail = dst0 + vec_len;
  const __fp16* srcTail = src0 + vec_len;
  if (canOverreadTail) {
    HVX_Vector a1 = vmemu((const HVX_Vector*)srcTail);
    HVX_Vector r1 = htp_ops_binary_compute_fp16_vector(a1, s1, opType);
    vstu_variable(dstTail, (uint32_t)(tail * (int)sizeof(__fp16)), r1);
    return true;
  }
  return htp_ops_binary_compute_fp16_rhs_scalar_tail(dstTail, srcTail, val1, tail, opType);
}

static inline bool htp_ops_binary_compute_fp16_lhs_scalar_tail(__fp16* dst, _Float16 val0,
                                                               const __fp16* src1, int remain, int opType) {
  if (!htp_ops_binary_supports_fp16_vector_tail(opType)) {
    return false;
  }
  const int vec_len = 128 / (int)sizeof(__fp16);
  __fp16 tmp1[vec_len] __attribute__((aligned(128))) = {};
  for (int j = 0; j < remain; ++j) {
    tmp1[j] = src1[j];
  }
  uint16_t v0_u16 = *(uint16_t*)&val0;
  HVX_Vector v0 = Q6_Vh_vsplat_R(v0_u16);
  HVX_Vector v1 = vmem(tmp1);
  HVX_Vector vr = htp_ops_binary_compute_fp16_vector(v0, v1, opType);
  return htp_ops_binary_store_fp16_tail(dst, vr, remain);
}

static inline void htp_ops_binary_compute_fp16_chunk(__fp16* __restrict dst,
                                                     const __fp16* __restrict src0,
                                                     const __fp16* __restrict src1,
                                                     int size, int opType) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  if (opType == HTP_OPS_BINARY_ADD) {
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0, v1));
      vmem(dst_ptr) = vr;
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_BINARY_ADD_RELU) {
    HVX_Vector zero_v = Q6_V_vzero();
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0, v1));
      vmem(dst_ptr) = Q6_Vhf_vmax_VhfVhf(vr, zero_v);
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_BINARY_SUB) {
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v0, v1));
      vmem(dst_ptr) = vr;
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_BINARY_MUL) {
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, v1));
      vmem(dst_ptr) = vr;
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_BINARY_SQUARED_DIFFERENCE) {
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      vmem(dst_ptr) = htp_ops_binary_squared_difference_fp16_vec(v0, v1);
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_BINARY_DIV) {
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      HVX_Vector inv_v1 = hvx_my_inv_vhf(v1);
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, inv_v1));
      vmem(dst_ptr) = vr;
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_BINARY_MAX) {
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      vmem(dst_ptr) = Q6_Vhf_vmax_VhfVhf(v0, v1);
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_BINARY_MIN) {
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      vmem(dst_ptr) = Q6_Vhf_vmin_VhfVhf(v0, v1);
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  } else if (opType == HTP_OPS_BINARY_MUL_SILU) {
    const __fp16* src0_ptr = src0;
    const __fp16* src1_ptr = src1;
    __fp16* dst_ptr = dst;
    const int vec2_len = vec_len * 2;
    const int vec2_end = vec_end & -vec2_len;
    for (; i < vec2_end; i += vec2_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      HVX_Vector v0_next = vmem(src0_ptr + vec_len);
      HVX_Vector v1_next = vmem(src1_ptr + vec_len);
      HVX_Vector vr = htp_ops_binary_mul_silu_fp16_vec(v0, v1);
      HVX_Vector vr_next = htp_ops_binary_mul_silu_fp16_vec(v0_next, v1_next);
      vmem(dst_ptr) = vr;
      vmem(dst_ptr + vec_len) = vr_next;
      src0_ptr += vec2_len;
      src1_ptr += vec2_len;
      dst_ptr += vec2_len;
    }
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v0 = vmem(src0_ptr);
      HVX_Vector v1 = vmem(src1_ptr);
      HVX_Vector vr = htp_ops_binary_mul_silu_fp16_vec(v0, v1);
      vmem(dst_ptr) = vr;
      src0_ptr += vec_len;
      src1_ptr += vec_len;
      dst_ptr += vec_len;
    }
  }
  if (i < size && htp_ops_binary_compute_fp16_tail(dst + i, src0 + i, src1 + i, size - i, opType)) {
    return;
  }
  for (; i < size; ++i) {
    dst[i] = htp_ops_binary_apply_fp16(src0[i], src1[i], opType);
  }
}

static inline void htp_ops_binary_compute_fp16_rhs_scalar_chunk(__fp16* dst, const __fp16* src0,
                                                                _Float16 val1, int size, int opType,
                                                                HtpOpsBinaryLoadFunc memLoadFunc) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  uint16_t v1_u16 = *(uint16_t*)&val1;
  HVX_Vector v1_vec = Q6_Vh_vsplat_R(v1_u16);
  if (opType == HTP_OPS_BINARY_ADD) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0, v1_vec));
      vmemu(dst + i) = vr;
    }
  } else if (opType == HTP_OPS_BINARY_ADD_RELU) {
    HVX_Vector zero_v = Q6_V_vzero();
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0, v1_vec));
      vmemu(dst + i) = Q6_Vhf_vmax_VhfVhf(vr, zero_v);
    }
  } else if (opType == HTP_OPS_BINARY_SUB) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v0, v1_vec));
      vmemu(dst + i) = vr;
    }
  } else if (opType == HTP_OPS_BINARY_MUL) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, v1_vec));
      vmemu(dst + i) = vr;
    }
  } else if (opType == HTP_OPS_BINARY_SQUARED_DIFFERENCE) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      vmemu(dst + i) = htp_ops_binary_squared_difference_fp16_vec(v0, v1_vec);
    }
  } else if (opType == HTP_OPS_BINARY_DIV) {
    HVX_Vector inv_v1 = hvx_my_inv_vhf(v1_vec);
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, inv_v1));
      vmemu(dst + i) = vr;
    }
  } else if (opType == HTP_OPS_BINARY_MAX) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      vmemu(dst + i) = Q6_Vhf_vmax_VhfVhf(v0, v1_vec);
    }
  } else if (opType == HTP_OPS_BINARY_MIN) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      vmemu(dst + i) = Q6_Vhf_vmin_VhfVhf(v0, v1_vec);
    }
  } else if (opType == HTP_OPS_BINARY_MUL_SILU) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v0 = memLoadFunc((const HVX_Vector*)(src0 + i));
      HVX_Vector vr = htp_ops_binary_mul_silu_fp16_vec(v0, v1_vec);
      vmemu(dst + i) = vr;
    }
  }
  if (i < size && htp_ops_binary_compute_fp16_rhs_scalar_tail(dst + i, src0 + i, val1, size - i, opType)) {
    return;
  }
  for (; i < size; ++i) {
    dst[i] = htp_ops_binary_apply_fp16(src0[i], val1, opType);
  }
}

static inline void htp_ops_binary_compute_fp16_lhs_scalar_chunk(__fp16* dst, _Float16 val0,
                                                                const __fp16* src1, int size, int opType,
                                                                HtpOpsBinaryLoadFunc memLoadFunc) {
  int i = 0;
  const int vec_len = 128 / (int)sizeof(__fp16);
  uint16_t v0_u16 = *(uint16_t*)&val0;
  HVX_Vector v0_vec = Q6_Vh_vsplat_R(v0_u16);
  if (opType == HTP_OPS_BINARY_ADD) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0_vec, v1));
      vmemu(dst + i) = vr;
    }
  } else if (opType == HTP_OPS_BINARY_ADD_RELU) {
    HVX_Vector zero_v = Q6_V_vzero();
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0_vec, v1));
      vmemu(dst + i) = Q6_Vhf_vmax_VhfVhf(vr, zero_v);
    }
  } else if (opType == HTP_OPS_BINARY_SUB) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v0_vec, v1));
      vmemu(dst + i) = vr;
    }
  } else if (opType == HTP_OPS_BINARY_MUL) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0_vec, v1));
      vmemu(dst + i) = vr;
    }
  } else if (opType == HTP_OPS_BINARY_SQUARED_DIFFERENCE) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      vmemu(dst + i) = htp_ops_binary_squared_difference_fp16_vec(v0_vec, v1);
    }
  } else if (opType == HTP_OPS_BINARY_DIV) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      HVX_Vector inv_v1 = hvx_my_inv_vhf(v1);
      HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0_vec, inv_v1));
      vmemu(dst + i) = vr;
    }
  } else if (opType == HTP_OPS_BINARY_MAX) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      vmemu(dst + i) = Q6_Vhf_vmax_VhfVhf(v0_vec, v1);
    }
  } else if (opType == HTP_OPS_BINARY_MIN) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      vmemu(dst + i) = Q6_Vhf_vmin_VhfVhf(v0_vec, v1);
    }
  } else if (opType == HTP_OPS_BINARY_MUL_SILU) {
    for (; i <= size - vec_len; i += vec_len) {
      HVX_Vector v1 = memLoadFunc((const HVX_Vector*)(src1 + i));
      HVX_Vector vr = htp_ops_binary_mul_silu_fp16_vec(v0_vec, v1);
      vmemu(dst + i) = vr;
    }
  }
  if (i < size && htp_ops_binary_compute_fp16_lhs_scalar_tail(dst + i, val0, src1 + i, size - i, opType)) {
    return;
  }
  for (; i < size; ++i) {
    dst[i] = htp_ops_binary_apply_fp16(val0, src1[i], opType);
  }
}

static inline void htp_ops_binary_compute_int32_chunk(int32_t* dst, const int32_t* src0,
                                                      const int32_t* src1, int size, int opType) {
  for (int i = 0; i < size; ++i) {
    dst[i] = htp_ops_binary_apply_int32(src0[i], src1[i], opType);
  }
}

static inline void htp_ops_binary_compute_int32_rhs_scalar_chunk(int32_t* dst, const int32_t* src0,
                                                                 int32_t val1, int size, int opType) {
  for (int i = 0; i < size; ++i) {
    dst[i] = htp_ops_binary_apply_int32(src0[i], val1, opType);
  }
}

static inline void htp_ops_binary_compute_int32_lhs_scalar_chunk(int32_t* dst, int32_t val0,
                                                                 const int32_t* src1, int size, int opType) {
  for (int i = 0; i < size; ++i) {
    dst[i] = htp_ops_binary_apply_int32(val0, src1[i], opType);
  }
}

static inline int htp_ops_binary_broadcast_offset(int index, int dims, const int32_t* outDims,
                                                  const int32_t* strides) {
  int offset = 0;
  for (int d = dims - 1; d >= 0; --d) {
    const int coord = index % outDims[d];
    index /= outDims[d];
    offset += coord * strides[d];
  }
  return offset;
}

static inline bool htp_ops_binary_try_broadcast(uint8_t* dst, const uint8_t* src0, const uint8_t* src1,
                                                int32_t outSize, int32_t opType, int32_t bytes,
                                                int32_t inputBytes, int32_t inputIsFloat,
                                                int32_t outputIsFloat,
                                                const int32_t* broadcastParams,
                                                int32_t broadcastParamCount) {
  if (broadcastParams == nullptr || broadcastParamCount < 25) {
    return false;
  }
  const int dims = broadcastParams[0];
  if (dims <= 0 || dims > 8) {
    return false;
  }
  const int32_t* outDims = broadcastParams + 1;
  const int32_t* in0Strides = broadcastParams + 9;
  const int32_t* in1Strides = broadcastParams + 17;

  int size = 1;
  for (int i = 0; i < dims; ++i) {
    if (outDims[i] <= 0) {
      return false;
    }
    size *= outDims[i];
  }
  if (size != outSize) {
    return false;
  }

  if (inputBytes == 2) {
    const __fp16* src0_fp16 = (const __fp16*)src0;
    const __fp16* src1_fp16 = (const __fp16*)src1;
    if (htp_ops_binary_is_compare(opType)) {
      if (bytes == 2) {
        __fp16* out = (__fp16*)dst;
        for (int i = 0; i < outSize; ++i) {
          const int off0 = htp_ops_binary_broadcast_offset(i, dims, outDims, in0Strides);
          const int off1 = htp_ops_binary_broadcast_offset(i, dims, outDims, in1Strides);
          out[i] = htp_ops_binary_compare_fp16(src0_fp16[off0], src1_fp16[off1], opType) ? (__fp16)1.0f : (__fp16)0.0f;
        }
        return true;
      }
      if (bytes == 4 && outputIsFloat) {
        float* out = (float*)dst;
        for (int i = 0; i < outSize; ++i) {
          const int off0 = htp_ops_binary_broadcast_offset(i, dims, outDims, in0Strides);
          const int off1 = htp_ops_binary_broadcast_offset(i, dims, outDims, in1Strides);
          out[i] = htp_ops_binary_compare_fp16(src0_fp16[off0], src1_fp16[off1], opType) ? 1.0f : 0.0f;
        }
        return true;
      }
      if (bytes == 4) {
        int32_t* out = (int32_t*)dst;
        for (int i = 0; i < outSize; ++i) {
          const int off0 = htp_ops_binary_broadcast_offset(i, dims, outDims, in0Strides);
          const int off1 = htp_ops_binary_broadcast_offset(i, dims, outDims, in1Strides);
          out[i] = htp_ops_binary_compare_fp16(src0_fp16[off0], src1_fp16[off1], opType);
        }
        return true;
      }
      return false;
    }
    if (bytes != 2) {
      return false;
    }
    __fp16* out = (__fp16*)dst;
    for (int i = 0; i < outSize; ++i) {
      const int off0 = htp_ops_binary_broadcast_offset(i, dims, outDims, in0Strides);
      const int off1 = htp_ops_binary_broadcast_offset(i, dims, outDims, in1Strides);
      out[i] = htp_ops_binary_apply_fp16(src0_fp16[off0], src1_fp16[off1], opType);
    }
    return true;
  }

  if (inputBytes == 4 && inputIsFloat && htp_ops_binary_is_compare(opType)) {
    const float* src0_float = (const float*)src0;
    const float* src1_float = (const float*)src1;
    if (bytes == 2) {
      __fp16* out = (__fp16*)dst;
      for (int i = 0; i < outSize; ++i) {
        const int off0 = htp_ops_binary_broadcast_offset(i, dims, outDims, in0Strides);
        const int off1 = htp_ops_binary_broadcast_offset(i, dims, outDims, in1Strides);
        out[i] = htp_ops_binary_compare_fp16(src0_float[off0], src1_float[off1], opType) ? (__fp16)1.0f : (__fp16)0.0f;
      }
      return true;
    }
    if (bytes == 4 && outputIsFloat) {
      float* out = (float*)dst;
      for (int i = 0; i < outSize; ++i) {
        const int off0 = htp_ops_binary_broadcast_offset(i, dims, outDims, in0Strides);
        const int off1 = htp_ops_binary_broadcast_offset(i, dims, outDims, in1Strides);
        out[i] = htp_ops_binary_compare_fp16(src0_float[off0], src1_float[off1], opType) ? 1.0f : 0.0f;
      }
      return true;
    }
    if (bytes == 4) {
      int32_t* out = (int32_t*)dst;
      for (int i = 0; i < outSize; ++i) {
        const int off0 = htp_ops_binary_broadcast_offset(i, dims, outDims, in0Strides);
        const int off1 = htp_ops_binary_broadcast_offset(i, dims, outDims, in1Strides);
        out[i] = htp_ops_binary_compare_fp16(src0_float[off0], src1_float[off1], opType);
      }
      return true;
    }
  }

  return false;
}

static inline bool htp_ops_binary_blit_can_use_fast_path(const HtpOpsBinaryRegion* r, int32_t bytes) {
  if (r->dstStride[2] != bytes) {
    return false;
  }
  if (r->size[2] == 1) {
    return true;
  }
  const bool src0_scalar = (r->src0Stride[2] == 0);
  const bool src1_scalar = (r->src1Stride[2] == 0);
  const bool src0_contig = (r->src0Stride[2] == bytes);
  const bool src1_contig = (r->src1Stride[2] == bytes);
  return (src0_scalar || src0_contig) && (src1_scalar || src1_contig) && !(src0_scalar && src1_scalar);
}

static inline void htp_ops_binary_blit_run_row(uint8_t* dstY, const uint8_t* src0Y, const uint8_t* src1Y,
                                               int size, int32_t src0StrideX, int32_t src1StrideX,
                                               int32_t bytes, int32_t opType) {
  if (bytes == 2) {
    __fp16* dst_fp16 = (__fp16*)dstY;
    const __fp16* src0_fp16 = (const __fp16*)src0Y;
    const __fp16* src1_fp16 = (const __fp16*)src1Y;
    if (src0StrideX == 0 && src1StrideX != 0) {
      htp_ops_binary_compute_fp16_lhs_scalar_chunk(dst_fp16, src0_fp16[0], src1_fp16, size, opType,
                                                   htp_ops_binary_pick_load_func(src1_fp16));
    } else if (src1StrideX == 0 && src0StrideX != 0) {
      htp_ops_binary_compute_fp16_rhs_scalar_chunk(dst_fp16, src0_fp16, src1_fp16[0], size, opType,
                                                   htp_ops_binary_pick_load_func(src0_fp16));
    } else if (src0StrideX == 0 && src1StrideX == 0) {
      const __fp16 value = htp_ops_binary_apply_fp16(src0_fp16[0], src1_fp16[0], opType);
      for (int i = 0; i < size; ++i) {
        dst_fp16[i] = value;
      }
    } else if (src0StrideX == bytes && src1StrideX == bytes &&
               (!htp_ops_binary_is_aligned_128(dst_fp16) ||
                !htp_ops_binary_is_aligned_128(src0_fp16) ||
                !htp_ops_binary_is_aligned_128(src1_fp16)) &&
               htp_ops_binary_supports_fp16_vector_tail(opType)) {
      int i = 0;
      const int vec_len = 128 / (int)sizeof(__fp16);
      for (; i <= size - vec_len; i += vec_len) {
        HVX_Vector v0 = vmemu((const HVX_Vector*)(src0_fp16 + i));
        HVX_Vector v1 = vmemu((const HVX_Vector*)(src1_fp16 + i));
        HVX_Vector vr = htp_ops_binary_compute_fp16_vector(v0, v1, opType);
        vmemu((HVX_Vector*)(dst_fp16 + i)) = vr;
      }
      for (; i < size; ++i) {
        dst_fp16[i] = htp_ops_binary_apply_fp16(src0_fp16[i], src1_fp16[i], opType);
      }
    } else {
      htp_ops_binary_compute_fp16_chunk(dst_fp16, src0_fp16, src1_fp16, size, opType);
    }
    return;
  }

  int32_t* dst_int32 = (int32_t*)dstY;
  const int32_t* src0_int32 = (const int32_t*)src0Y;
  const int32_t* src1_int32 = (const int32_t*)src1Y;
  if (src0StrideX == 0 && src1StrideX != 0) {
    htp_ops_binary_compute_int32_lhs_scalar_chunk(dst_int32, src0_int32[0], src1_int32, size, opType);
  } else if (src1StrideX == 0 && src0StrideX != 0) {
    htp_ops_binary_compute_int32_rhs_scalar_chunk(dst_int32, src0_int32, src1_int32[0], size, opType);
  } else if (src0StrideX == 0 && src1StrideX == 0) {
    const int32_t value = htp_ops_binary_apply_int32(src0_int32[0], src1_int32[0], opType);
    for (int i = 0; i < size; ++i) {
      dst_int32[i] = value;
    }
  } else {
    htp_ops_binary_compute_int32_chunk(dst_int32, src0_int32, src1_int32, size, opType);
  }
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* src0Base;
  const uint8_t* src1Base;
  const HtpOpsBinaryRegion* region;
  int32_t bytes;
  int32_t opType;
  int rowCount;
  worker_synctoken_t sync_ctx;
} HtpOpsBinaryBlitTaskState;

typedef struct {
  HtpOpsBinaryBlitTaskState* state;
  int begin;
  int end;
} HtpOpsBinaryBlitFixedTask;

static inline bool htp_ops_binary_blit_src_row_broadcast(const HtpOpsBinaryRegion* r, int srcIndex) {
  const int32_t* stride = srcIndex == 0 ? r->src0Stride : r->src1Stride;
  return r->size[2] > 0 && stride[2] == 2 &&
         (r->size[0] <= 1 || stride[0] == 0) &&
         (r->size[1] <= 1 || stride[1] == 0);
}

static inline void htp_ops_binary_blit_add_fp16_row_broadcast(HtpOpsBinaryBlitTaskState* state,
                                                              int begin, int end,
                                                              bool src1Broadcast) {
  const HtpOpsBinaryRegion* r = state->region;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = r->size[2] & -vec_len;
  const uint8_t* broadcastBase = src1Broadcast ? state->src1Base : state->src0Base;
  const uint8_t* varyingBase = src1Broadcast ? state->src0Base : state->src1Base;
  const int32_t* varyingStride = src1Broadcast ? r->src0Stride : r->src1Stride;

  for (int x = 0; x < vec_end; x += vec_len) {
    HVX_Vector vb = vmemu((const HVX_Vector*)((const __fp16*)broadcastBase + x));
    for (int index = begin; index < end; ++index) {
      const int z = index / r->size[1];
      const int y = index - z * r->size[1];
      const __fp16* src = (const __fp16*)(varyingBase + z * varyingStride[0] + y * varyingStride[1]) + x;
      __fp16* dst = (__fp16*)(state->dstBase + z * r->dstStride[0] + y * r->dstStride[1]) + x;
      HVX_Vector v = vmemu((const HVX_Vector*)src);
      vmemu((HVX_Vector*)dst) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, vb));
    }
  }

  if (vec_end < r->size[2]) {
    const int remain = r->size[2] - vec_end;
    __fp16 broadcastTail[vec_len] __attribute__((aligned(128))) = {};
    __fp16 srcTail[vec_len] __attribute__((aligned(128))) = {};
    memcpy(broadcastTail, (const __fp16*)broadcastBase + vec_end, (size_t)remain * sizeof(__fp16));
    HVX_Vector vb = vmem(broadcastTail);
    for (int index = begin; index < end; ++index) {
      const int z = index / r->size[1];
      const int y = index - z * r->size[1];
      const __fp16* src = (const __fp16*)(varyingBase + z * varyingStride[0] + y * varyingStride[1]) + vec_end;
      __fp16* dst = (__fp16*)(state->dstBase + z * r->dstStride[0] + y * r->dstStride[1]) + vec_end;
      HVX_Vector v;
      bool canOverread = false;
      if (index + 1 < state->rowCount) {
        const int nextIndex = index + 1;
        const int nextZ = nextIndex / r->size[1];
        const int nextY = nextIndex - nextZ * r->size[1];
        const __fp16* nextSrc = (const __fp16*)(varyingBase + nextZ * varyingStride[0] + nextY * varyingStride[1]);
        canOverread = nextSrc == src + remain;
      }
      if (canOverread) {
        v = vmemu((const HVX_Vector*)src);
      } else {
        memset(srcTail, 0, sizeof(srcTail));
        memcpy(srcTail, src, (size_t)remain * sizeof(__fp16));
        v = vmem(srcTail);
      }
      HVX_Vector sum = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, vb));
      vstu_variable(dst, (uint32_t)(remain * (int)sizeof(__fp16)), sum);
    }
  }
}

static inline bool htp_ops_binary_blit_try_add_fp16_small_y_broadcast(HtpOpsBinaryBlitTaskState* state,
                                                                      int begin, int end,
                                                                      bool src1Broadcast) {
  const HtpOpsBinaryRegion* r = state->region;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int rowBytes = r->size[2] * (int)sizeof(__fp16);
  if (r->size[0] > 1 || r->size[2] <= 0 || r->size[2] > vec_len * 2 ||
      r->dstStride[1] != rowBytes) {
    return false;
  }
  const int32_t* varyingStride = src1Broadcast ? r->src0Stride : r->src1Stride;
  const int32_t* broadcastStride = src1Broadcast ? r->src1Stride : r->src0Stride;
  if (varyingStride[1] != rowBytes || broadcastStride[1] != 0) {
    return false;
  }

  const uint8_t* broadcastBase = src1Broadcast ? state->src1Base : state->src0Base;
  const uint8_t* varyingBase = src1Broadcast ? state->src0Base : state->src1Base;
  HVX_Vector broadcast0 = vmemu((const HVX_Vector*)broadcastBase);
  HVX_Vector broadcast1 = Q6_V_vzero();
  if (r->size[2] > vec_len) {
    __fp16 tail[vec_len] __attribute__((aligned(128))) = {};
    memcpy(tail, (const __fp16*)broadcastBase + vec_len, (size_t)(r->size[2] - vec_len) * sizeof(__fp16));
    broadcast1 = vmem(tail);
  }

  for (int index = begin; index < end; ++index) {
    const __fp16* src = (const __fp16*)(varyingBase + (ptrdiff_t)index * varyingStride[1]);
    __fp16* dst = (__fp16*)(state->dstBase + (ptrdiff_t)index * r->dstStride[1]);
    HVX_Vector v0 = vmemu((const HVX_Vector*)src);
    vmemu((HVX_Vector*)dst) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v0, broadcast0));
    if (r->size[2] > vec_len) {
      const int remain = r->size[2] - vec_len;
      HVX_Vector v1;
      if (index + 1 < state->rowCount) {
        v1 = vmemu((const HVX_Vector*)(src + vec_len));
      } else {
        __fp16 srcTail[vec_len] __attribute__((aligned(128))) = {};
        memcpy(srcTail, src + vec_len, (size_t)remain * sizeof(__fp16));
        v1 = vmem(srcTail);
      }
      HVX_Vector sum = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v1, broadcast1));
      vstu_variable(dst + vec_len, (uint32_t)(remain * (int)sizeof(__fp16)), sum);
    }
  }
  return true;
}

static inline void htp_ops_binary_blit_run_rows(HtpOpsBinaryBlitTaskState* state, int begin, int end) {
  const HtpOpsBinaryRegion* r = state->region;
  const int vec_len_fp16 = 128 / (int)sizeof(__fp16);
  if (state->bytes == 2 && state->opType == HTP_OPS_BINARY_ADD &&
      r->dstStride[2] == 2 && r->src0Stride[2] == 2 && r->src1Stride[2] == 2 &&
      htp_ops_binary_supports_fp16_vector_tail(state->opType)) {
    const bool src0Broadcast = htp_ops_binary_blit_src_row_broadcast(r, 0);
    const bool src1Broadcast = htp_ops_binary_blit_src_row_broadcast(r, 1);
    if (src1Broadcast && !src0Broadcast) {
      if (htp_ops_binary_blit_try_add_fp16_small_y_broadcast(state, begin, end, true)) {
        return;
      }
      htp_ops_binary_blit_add_fp16_row_broadcast(state, begin, end, true);
      return;
    }
    if (src0Broadcast && !src1Broadcast) {
      if (htp_ops_binary_blit_try_add_fp16_small_y_broadcast(state, begin, end, false)) {
        return;
      }
      htp_ops_binary_blit_add_fp16_row_broadcast(state, begin, end, false);
      return;
    }
  }
  const bool short_rhs_scalar_fp16 =
      state->bytes == 2 && r->size[2] > 0 && r->size[2] < vec_len_fp16 &&
      r->dstStride[2] == 2 && r->src0Stride[2] == 2 && r->src1Stride[2] == 0 &&
      htp_ops_binary_supports_fp16_vector_tail(state->opType);
  for (int index = begin; index < end; ++index) {
    const int z = index / r->size[1];
    const int y = index - z * r->size[1];
    const uint8_t* src0Y = state->src0Base + z * r->src0Stride[0] + y * r->src0Stride[1];
    const uint8_t* src1Y = state->src1Base + z * r->src1Stride[0] + y * r->src1Stride[1];
    uint8_t* dstY = state->dstBase + z * r->dstStride[0] + y * r->dstStride[1];
    if (short_rhs_scalar_fp16) {
      if (index + 1 < end) {
        const int nextIndex = index + 1;
        const int nextZ = nextIndex / r->size[1];
        const int nextY = nextIndex - nextZ * r->size[1];
        const uint8_t* nextSrc0Y = state->src0Base + nextZ * r->src0Stride[0] + nextY * r->src0Stride[1];
        const uint8_t* nextSrc1Y = state->src1Base + nextZ * r->src1Stride[0] + nextY * r->src1Stride[1];
        uint8_t* nextDstY = state->dstBase + nextZ * r->dstStride[0] + nextY * r->dstStride[1];
        const bool pairContig = (nextSrc0Y == src0Y + r->size[2] * state->bytes) &&
                                (nextDstY == dstY + r->size[2] * state->bytes);
        bool canOverreadTail = false;
        if (index + 2 < state->rowCount) {
          const int tailIndex = index + 2;
          const int tailZ = tailIndex / r->size[1];
          const int tailY = tailIndex - tailZ * r->size[1];
          const uint8_t* tailSrc0Y = state->src0Base + tailZ * r->src0Stride[0] + tailY * r->src0Stride[1];
          canOverreadTail = (tailSrc0Y == nextSrc0Y + r->size[2] * state->bytes);
        }
        if (pairContig &&
            htp_ops_binary_compute_fp16_rhs_scalar_short_pair((__fp16*)dstY, (const __fp16*)src0Y,
                                                              *(__fp16*)src1Y, *(__fp16*)nextSrc1Y,
                                                              r->size[2], state->opType,
                                                              canOverreadTail)) {
          ++index;
          continue;
        }
      }
      bool canOverread = false;
      if (index + 1 < state->rowCount) {
        const int nextIndex = index + 1;
        const int nextZ = nextIndex / r->size[1];
        const int nextY = nextIndex - nextZ * r->size[1];
        const uint8_t* nextSrc0Y = state->src0Base + nextZ * r->src0Stride[0] + nextY * r->src0Stride[1];
        canOverread = (nextSrc0Y == src0Y + r->size[2] * state->bytes);
      }
      if (htp_ops_binary_compute_fp16_rhs_scalar_short((__fp16*)dstY, (const __fp16*)src0Y,
                                                       *(__fp16*)src1Y, r->size[2], state->opType,
                                                       canOverread)) {
        continue;
      }
    }
    htp_ops_binary_blit_run_row(dstY, src0Y, src1Y, r->size[2], r->src0Stride[2], r->src1Stride[2],
                                state->bytes, state->opType);
  }
}

static void htp_ops_binary_blit_fixed_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsBinaryBlitFixedTask* task = (HtpOpsBinaryBlitFixedTask*)data;
  htp_ops_binary_blit_run_rows(task->state, task->begin, task->end);
  worker_pool_synctoken_jobdone(&(task->state->sync_ctx));
}

static inline bool htp_ops_binary_blit_try_parallel(uint8_t* dstBase, const uint8_t* src0Base,
                                                    const uint8_t* src1Base, const HtpOpsBinaryRegion* r,
                                                    int32_t bytes, int32_t opType) {
  if (g_max_num_workers <= 1 || r->size[0] <= 0 || r->size[1] <= 0 || r->size[2] <= 0 ||
      r->dstStride[0] < 0 || r->dstStride[1] < 0 ||
      r->src0Stride[0] < 0 || r->src0Stride[1] < 0 ||
      r->src1Stride[0] < 0 || r->src1Stride[1] < 0) {
    return false;
  }
  const int rowBytes = r->size[2] * bytes;
  if ((r->size[1] > 1 && r->dstStride[1] < rowBytes) ||
      (r->size[0] > 1 && r->dstStride[0] < r->dstStride[1] * r->size[1])) {
    return false;
  }
  const int rowCount = r->size[0] * r->size[1];
  const int elems = rowCount * r->size[2];
  const int nTasks = htp_ops_binary_pick_task_count(elems, bytes);
  if (nTasks <= 1 || rowCount < nTasks) {
    return false;
  }

  HtpOpsBinaryBlitTaskState state = {};
  state.dstBase = dstBase;
  state.src0Base = src0Base;
  state.src1Base = src1Base;
  state.region = r;
  state.bytes = bytes;
  state.opType = opType;
  state.rowCount = rowCount;

  worker_pool_job_t job;
  job.fptr = htp_ops_binary_blit_fixed_worker;
  HtpOpsBinaryBlitFixedTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsBinaryBlitFixedTask, nTasks);
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  const int rowsPerTask = (rowCount + nTasks - 1) / nTasks;
  for (int i = 0; i < nTasks; ++i) {
    const int begin = i * rowsPerTask;
    int end = begin + rowsPerTask;
    if (end > rowCount) {
      end = rowCount;
    }
    tasks[i].state = &state;
    tasks[i].begin = begin;
    tasks[i].end = end;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

AEEResult htp_ops_binary_blit(uint8_t* dst, const uint8_t* src0, const uint8_t* src1,
                              uint8_t* region, int32_t regionCount, int32_t bytes, int32_t opType) {
  if (regionCount <= 0) {
    return 0;
  }
  if (bytes != 2 && bytes != 4) {
    return -1;
  }

  uint8_t* pDst = dst;
  const uint8_t* pSrc0 = src0;
  const uint8_t* pSrc1 = src1;
  const HtpOpsBinaryRegion* regions = (const HtpOpsBinaryRegion*)region;

  for (int i = 0; i < regionCount; ++i) {
    const HtpOpsBinaryRegion* r = regions + i;
    const uint8_t* src0Base = pSrc0 + r->src0Offset;
    const uint8_t* src1Base = pSrc1 + r->src1Offset;
    uint8_t* dstBase = pDst + r->dstOffset;

    if (htp_ops_binary_blit_can_use_fast_path(r, bytes)) {
      if (htp_ops_binary_blit_try_parallel(dstBase, src0Base, src1Base, r, bytes, opType)) {
        continue;
      }
      for (int z = 0; z < r->size[0]; ++z) {
        const uint8_t* src0Z = src0Base + z * r->src0Stride[0];
        const uint8_t* src1Z = src1Base + z * r->src1Stride[0];
        uint8_t* dstZ = dstBase + z * r->dstStride[0];
        for (int y = 0; y < r->size[1]; ++y) {
          const uint8_t* src0Y = src0Z + y * r->src0Stride[1];
          const uint8_t* src1Y = src1Z + y * r->src1Stride[1];
          uint8_t* dstY = dstZ + y * r->dstStride[1];
          htp_ops_binary_blit_run_row(dstY, src0Y, src1Y, r->size[2], r->src0Stride[2], r->src1Stride[2], bytes, opType);
        }
      }
      continue;
    }

    for (int z = 0; z < r->size[0]; ++z) {
      const uint8_t* src0Z = src0Base + z * r->src0Stride[0];
      const uint8_t* src1Z = src1Base + z * r->src1Stride[0];
      uint8_t* dstZ = dstBase + z * r->dstStride[0];
      for (int y = 0; y < r->size[1]; ++y) {
        const uint8_t* src0Y = src0Z + y * r->src0Stride[1];
        const uint8_t* src1Y = src1Z + y * r->src1Stride[1];
        uint8_t* dstY = dstZ + y * r->dstStride[1];
        for (int x = 0; x < r->size[2]; ++x) {
          if (bytes == 2) {
            const __fp16* a = (const __fp16*)(src0Y + x * r->src0Stride[2]);
            const __fp16* b = (const __fp16*)(src1Y + x * r->src1Stride[2]);
            __fp16* d = (__fp16*)(dstY + x * r->dstStride[2]);
            *d = htp_ops_binary_apply_fp16(*a, *b, opType);
          } else {
            const int32_t* a = (const int32_t*)(src0Y + x * r->src0Stride[2]);
            const int32_t* b = (const int32_t*)(src1Y + x * r->src1Stride[2]);
            int32_t* d = (int32_t*)(dstY + x * r->dstStride[2]);
            *d = htp_ops_binary_apply_int32(*a, *b, opType);
          }
        }
      }
    }
  }

  return 0;
}

AEEResult htp_ops_binary_elementwise(uint8_t* dst, uint8_t* src0_ptr, uint8_t* src1_ptr,
                                     int32_t outSize, int32_t in0Size, int32_t in1Size,
                                     int32_t opType, int32_t bytes, int32_t inputBytes,
                                     int32_t inputIsFloat, int32_t outputIsFloat,
                                     const int32_t* broadcastParams, int32_t broadcastParamCount) {
  if (bytes != 2 && bytes != 4) {
    return -1;
  }
  if (inputBytes != 2 && inputBytes != 4) {
    inputBytes = bytes;
  }
  if (outSize <= 0) {
    return 0;
  }

  uint8_t* dstBase = dst;
  const uint8_t* src0Base = src0_ptr;
  const uint8_t* src1Base = src1_ptr;

  if (broadcastParams != nullptr && broadcastParamCount > 0 &&
      htp_ops_binary_try_broadcast(dstBase, src0Base, src1Base, outSize, opType, bytes, inputBytes,
                                   inputIsFloat, outputIsFloat, broadcastParams, broadcastParamCount)) {
    return 0;
  }

  if (htp_ops_binary_is_compare(opType)) {
    if (bytes != 2 && bytes != 4) {
      return -1;
    }
    if (inputBytes == 2) {
      const __fp16* src0 = (const __fp16*)src0Base;
      const __fp16* src1 = (const __fp16*)src1Base;
      if (bytes == 2) {
        __fp16* out = (__fp16*)dstBase;
        if (outSize == in0Size && outSize == in1Size) {
          htp_ops_binary_compare_fp16_to_fp16_chunk(out, src0, src1, outSize, opType);
        } else if (outSize == in0Size && in1Size == 1) {
          const __fp16 val1 = src1[0];
          uint16_t val1_u16;
          memcpy(&val1_u16, &val1, sizeof(val1_u16));
          htp_ops_binary_compare_fp16_rhs_scalar_to_fp16_chunk(out, src0, val1_u16, outSize, opType);
        } else if (outSize == in1Size && in0Size == 1) {
          const __fp16 val0 = src0[0];
          uint16_t val0_u16;
          memcpy(&val0_u16, &val0, sizeof(val0_u16));
          htp_ops_binary_compare_fp16_lhs_scalar_to_fp16_chunk(out, val0_u16, src1, outSize, opType);
        }
      } else if (outputIsFloat) {
        float* out = (float*)dstBase;
        if (outSize == in0Size && outSize == in1Size) {
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(src0[i], src1[i], opType) ? 1.0f : 0.0f;
          }
        } else if (outSize == in0Size && in1Size == 1) {
          const __fp16 val1 = src1[0];
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(src0[i], val1, opType) ? 1.0f : 0.0f;
          }
        } else if (outSize == in1Size && in0Size == 1) {
          const __fp16 val0 = src0[0];
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(val0, src1[i], opType) ? 1.0f : 0.0f;
          }
        }
      } else {
        int32_t* out = (int32_t*)dstBase;
        if (outSize == in0Size && outSize == in1Size) {
          HtpOpsBinaryTaskState task_state = {};
          task_state.kind = HTP_OPS_BINARY_TASK_CMP_FP16_TO_INT32;
          task_state.opType = opType;
          task_state.int32_dst = out;
          task_state.fp16_src0 = src0;
          task_state.fp16_src1 = src1;
          htp_ops_binary_run_task(&task_state, outSize, 4);
        } else if (outSize == in0Size && in1Size == 1) {
          const __fp16 val1 = src1[0];
          HtpOpsBinaryTaskState task_state = {};
          task_state.kind = HTP_OPS_BINARY_TASK_CMP_FP16_RHS_SCALAR_TO_INT32;
          task_state.opType = opType;
          task_state.int32_dst = out;
          task_state.fp16_src0 = src0;
          task_state.fp16_scalar = val1;
          htp_ops_binary_run_task(&task_state, outSize, 4);
        } else if (outSize == in1Size && in0Size == 1) {
          const __fp16 val0 = src0[0];
          HtpOpsBinaryTaskState task_state = {};
          task_state.kind = HTP_OPS_BINARY_TASK_CMP_FP16_LHS_SCALAR_TO_INT32;
          task_state.opType = opType;
          task_state.int32_dst = out;
          task_state.fp16_src1 = src1;
          task_state.fp16_scalar = val0;
          htp_ops_binary_run_task(&task_state, outSize, 4);
        }
      }
      return 0;
    }
    if (inputIsFloat) {
      const float* src0 = (const float*)src0Base;
      const float* src1 = (const float*)src1Base;
      if (bytes == 2) {
        __fp16* out = (__fp16*)dstBase;
        if (outSize == in0Size && outSize == in1Size) {
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(src0[i], src1[i], opType) ? (__fp16)1.0f : (__fp16)0.0f;
          }
        } else if (outSize == in0Size && in1Size == 1) {
          const float val1 = src1[0];
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(src0[i], val1, opType) ? (__fp16)1.0f : (__fp16)0.0f;
          }
        } else if (outSize == in1Size && in0Size == 1) {
          const float val0 = src0[0];
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(val0, src1[i], opType) ? (__fp16)1.0f : (__fp16)0.0f;
          }
        }
      } else if (outputIsFloat) {
        float* out = (float*)dstBase;
        if (outSize == in0Size && outSize == in1Size) {
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(src0[i], src1[i], opType) ? 1.0f : 0.0f;
          }
        } else if (outSize == in0Size && in1Size == 1) {
          const float val1 = src1[0];
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(src0[i], val1, opType) ? 1.0f : 0.0f;
          }
        } else if (outSize == in1Size && in0Size == 1) {
          const float val0 = src0[0];
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(val0, src1[i], opType) ? 1.0f : 0.0f;
          }
        }
      } else {
        int32_t* out = (int32_t*)dstBase;
        if (outSize == in0Size && outSize == in1Size) {
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(src0[i], src1[i], opType);
          }
        } else if (outSize == in0Size && in1Size == 1) {
          const float val1 = src1[0];
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(src0[i], val1, opType);
          }
        } else if (outSize == in1Size && in0Size == 1) {
          const float val0 = src0[0];
          for (int i = 0; i < outSize; ++i) {
            out[i] = htp_ops_binary_compare_fp16(val0, src1[i], opType);
          }
        }
      }
      return 0;
    }
    const int32_t* src0 = (const int32_t*)src0Base;
    const int32_t* src1 = (const int32_t*)src1Base;
    if (bytes == 2) {
      __fp16* out = (__fp16*)dstBase;
      if (outSize == in0Size && outSize == in1Size) {
        for (int i = 0; i < outSize; ++i) {
          out[i] = htp_ops_binary_apply_int32(src0[i], src1[i], opType) ? (__fp16)1.0f : (__fp16)0.0f;
        }
      } else if (outSize == in0Size && in1Size == 1) {
        const int32_t val1 = src1[0];
        for (int i = 0; i < outSize; ++i) {
          out[i] = htp_ops_binary_apply_int32(src0[i], val1, opType) ? (__fp16)1.0f : (__fp16)0.0f;
        }
      } else if (outSize == in1Size && in0Size == 1) {
        const int32_t val0 = src0[0];
        for (int i = 0; i < outSize; ++i) {
          out[i] = htp_ops_binary_apply_int32(val0, src1[i], opType) ? (__fp16)1.0f : (__fp16)0.0f;
        }
      }
    } else {
      int32_t* out = (int32_t*)dstBase;
      if (outSize == in0Size && outSize == in1Size) {
        for (int i = 0; i < outSize; ++i) {
          out[i] = htp_ops_binary_apply_int32(src0[i], src1[i], opType);
        }
      } else if (outSize == in0Size && in1Size == 1) {
        const int32_t val1 = src1[0];
        for (int i = 0; i < outSize; ++i) {
          out[i] = htp_ops_binary_apply_int32(src0[i], val1, opType);
        }
      } else if (outSize == in1Size && in0Size == 1) {
        const int32_t val0 = src0[0];
        for (int i = 0; i < outSize; ++i) {
          out[i] = htp_ops_binary_apply_int32(val0, src1[i], opType);
        }
      }
    }
    return 0;
  }

  if (bytes == 2) {
    __fp16* dst_fp16 = (__fp16*)dstBase;
    const __fp16* src0 = (const __fp16*)src0Base;
    const __fp16* src1 = (const __fp16*)src1Base;

    if (outSize == in0Size && outSize == in1Size) {
      HtpOpsBinaryTaskState task_state = {};
      task_state.kind = HTP_OPS_BINARY_TASK_FP16;
      task_state.opType = opType;
      task_state.fp16_dst = dst_fp16;
      task_state.fp16_src0 = src0;
      task_state.fp16_src1 = src1;
      htp_ops_binary_run_task(&task_state, outSize, 2);
    } else if (outSize == in0Size && in1Size == 1) {
      __fp16 val1 = src1[0];
      HtpOpsBinaryTaskState task_state = {};
      task_state.kind = HTP_OPS_BINARY_TASK_FP16_RHS_SCALAR;
      task_state.opType = opType;
      task_state.fp16_dst = dst_fp16;
      task_state.fp16_src0 = src0;
      task_state.fp16_scalar = val1;
      htp_ops_binary_run_task(&task_state, outSize, 2);
    } else if (outSize == in1Size && in0Size == 1) {
      __fp16 val0 = src0[0];
      HtpOpsBinaryTaskState task_state = {};
      task_state.kind = HTP_OPS_BINARY_TASK_FP16_LHS_SCALAR;
      task_state.opType = opType;
      task_state.fp16_dst = dst_fp16;
      task_state.fp16_src1 = src1;
      task_state.fp16_scalar = val0;
      htp_ops_binary_run_task(&task_state, outSize, 2);
    }
  } else if (bytes == 4) {
    int32_t* dst_int32 = (int32_t*)dstBase;
    const int32_t* src0 = (const int32_t*)src0Base;
    const int32_t* src1 = (const int32_t*)src1Base;

    if (outSize == in0Size && outSize == in1Size) {
      HtpOpsBinaryTaskState task_state = {};
      task_state.kind = HTP_OPS_BINARY_TASK_INT32;
      task_state.opType = opType;
      task_state.int32_dst = dst_int32;
      task_state.int32_src0 = src0;
      task_state.int32_src1 = src1;
      htp_ops_binary_run_task(&task_state, outSize, 4);
    } else if (outSize == in0Size && in1Size == 1) {
      int32_t val1 = src1[0];
      HtpOpsBinaryTaskState task_state = {};
      task_state.kind = HTP_OPS_BINARY_TASK_INT32_RHS_SCALAR;
      task_state.opType = opType;
      task_state.int32_dst = dst_int32;
      task_state.int32_src0 = src0;
      task_state.int32_scalar = val1;
      htp_ops_binary_run_task(&task_state, outSize, 4);
    } else if (outSize == in1Size && in0Size == 1) {
      int32_t val0 = src0[0];
      HtpOpsBinaryTaskState task_state = {};
      task_state.kind = HTP_OPS_BINARY_TASK_INT32_LHS_SCALAR;
      task_state.opType = opType;
      task_state.int32_dst = dst_int32;
      task_state.int32_src1 = src1;
      task_state.int32_scalar = val0;
      htp_ops_binary_run_task(&task_state, outSize, 4);
    }
  }
  return 0;
}

static inline int htp_ops_select_cond_at(const uint8_t* cond, int index, int condBytes) {
  if (condBytes == 1) {
    return ((const uint8_t*)cond)[index] != 0;
  }
  if (condBytes == 2) {
    return ((const uint16_t*)cond)[index] != 0;
  }
  return ((const uint32_t*)cond)[index] != 0;
}

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  int n_tasks;
  int outSize;
  int condStep;
  int in1Step;
  int in2Step;
  int in1Mode;
  int in2Mode;
  int channelSize;
  int innerSize;
  int bytes;
  int condBytes;
  uint8_t* dst;
  const uint8_t* cond;
  const uint8_t* src1;
  const uint8_t* src2;
} HtpOpsSelectTaskState;

static inline int htp_ops_select_input_index(int index, int step, int mode, int channelSize, int innerSize) {
  if (mode == 0) {
    return 0;
  }
  if (mode == 2) {
    return (index / innerSize) % channelSize;
  }
  return index * step;
}

static inline void htp_ops_select_compute_range(HtpOpsSelectTaskState* state, int start, int end) {
  if (state->bytes == 1) {
    uint8_t* out = state->dst;
    const uint8_t* in1 = state->src1;
    const uint8_t* in2 = state->src2;
    for (int32_t i = start; i < end; ++i) {
      int in1Index = htp_ops_select_input_index(i, state->in1Step, state->in1Mode, state->channelSize, state->innerSize);
      int in2Index = htp_ops_select_input_index(i, state->in2Step, state->in2Mode, state->channelSize, state->innerSize);
      out[i] = htp_ops_select_cond_at(state->cond, i * state->condStep, state->condBytes)
                   ? in1[in1Index]
                   : in2[in2Index];
    }
    return;
  }
  if (state->bytes == 2) {
    uint16_t* out = (uint16_t*)state->dst;
    const uint16_t* in1 = (const uint16_t*)state->src1;
    const uint16_t* in2 = (const uint16_t*)state->src2;
    for (int32_t i = start; i < end; ++i) {
      int in1Index = htp_ops_select_input_index(i, state->in1Step, state->in1Mode, state->channelSize, state->innerSize);
      int in2Index = htp_ops_select_input_index(i, state->in2Step, state->in2Mode, state->channelSize, state->innerSize);
      out[i] = htp_ops_select_cond_at(state->cond, i * state->condStep, state->condBytes)
                   ? in1[in1Index]
                   : in2[in2Index];
    }
    return;
  }
  uint32_t* out = (uint32_t*)state->dst;
  const uint32_t* in1 = (const uint32_t*)state->src1;
  const uint32_t* in2 = (const uint32_t*)state->src2;
  for (int32_t i = start; i < end; ++i) {
    int in1Index = htp_ops_select_input_index(i, state->in1Step, state->in1Mode, state->channelSize, state->innerSize);
    int in2Index = htp_ops_select_input_index(i, state->in2Step, state->in2Mode, state->channelSize, state->innerSize);
    out[i] = htp_ops_select_cond_at(state->cond, i * state->condStep, state->condBytes)
                 ? in1[in1Index]
                 : in2[in2Index];
  }
}

static void htp_ops_select_worker_loop(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsSelectTaskState* state = (HtpOpsSelectTaskState*)data;
  const int grain = 2048;
  const int total_blocks = (state->outSize + grain - 1) / grain;
  const int blocks_per_task = (total_blocks + state->n_tasks - 1) / state->n_tasks;
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= state->n_tasks) {
      break;
    }
    int start = (int)task_id * blocks_per_task * grain;
    if (start >= state->outSize) {
      break;
    }
    int end = start + blocks_per_task * grain;
    if (end > state->outSize) {
      end = state->outSize;
    }
    htp_ops_select_compute_range(state, start, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline int htp_ops_select_pick_task_count(int outSize) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap <= 1) {
    return 1;
  }
  int task_count = (outSize + 2047) / 2048;
  if (task_count < 2) {
    return 1;
  }
  if (task_count > (int)worker_cap) {
    task_count = (int)worker_cap;
  }
  return task_count;
}

static inline HVX_Vector htp_ops_select_i32_cond_to_fp16(HVX_Vector cond0, HVX_Vector cond1,
                                                         uint16_t trueBits, uint16_t falseBits) {
  HVX_Vector zero = Q6_V_vzero();
  HVX_Vector trueValue = Q6_V_vsplat_R((int32_t)trueBits);
  HVX_Vector falseValue = Q6_V_vsplat_R((int32_t)falseBits);
  HVX_VectorPred pred0 = Q6_Q_vcmp_gt_VwVw(cond0, zero);
  HVX_VectorPred pred1 = Q6_Q_vcmp_gt_VwVw(cond1, zero);
  HVX_Vector out0 = Q6_V_vmux_QVV(pred0, trueValue, falseValue);
  HVX_Vector out1 = Q6_V_vmux_QVV(pred1, trueValue, falseValue);
  return Q6_Vh_vpacke_VwVw(out1, out0);
}

static inline void htp_ops_select_prelu_channel_fp16_range(HtpOpsSelectTaskState* state, int start, int end) {
  uint16_t falseBits = ((const uint16_t*)state->src2)[0];
  const uint16_t* slope = (const uint16_t*)state->src1;
  uint16_t* out = (uint16_t*)state->dst;
  const uint32_t* cond = (const uint32_t*)state->cond;
  const int vec_len = 128 / (int)sizeof(uint16_t);

  int i = start;
  while (i < end) {
    const int innerOffset = i % state->innerSize;
    int run = state->innerSize - innerOffset;
    if (run > end - i) {
      run = end - i;
    }
    const int channel = (i / state->innerSize) % state->channelSize;
    const uint16_t trueBits = slope[channel];
    const int vec_end = i + (run & -vec_len);
    for (; i < vec_end; i += vec_len) {
      HVX_Vector cond0 = vmemu((const HVX_Vector*)(cond + i));
      HVX_Vector cond1 = vmemu((const HVX_Vector*)(cond + i + 32));
      vmemu((HVX_Vector*)(out + i)) = htp_ops_select_i32_cond_to_fp16(cond0, cond1, trueBits, falseBits);
    }
    const int tailEnd = i + (run - (run & -vec_len));
    for (; i < tailEnd; ++i) {
      out[i] = cond[i] != 0 ? trueBits : falseBits;
    }
  }
}

static void htp_ops_select_prelu_channel_fp16_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsSelectTaskState* state = (HtpOpsSelectTaskState*)data;
  const int grain = 4096;
  while (1) {
    const int task_id = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int start = task_id * grain;
    if (start >= state->outSize) {
      break;
    }
    int end = start + grain;
    if (end > state->outSize) {
      end = state->outSize;
    }
    htp_ops_select_prelu_channel_fp16_range(state, start, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_select_try_prelu_channel_fp16(HtpOpsSelectTaskState* state) {
  if (state->bytes != 2 || state->condBytes != 4 || state->condStep != 1 ||
      state->in1Mode != 2 || state->in2Mode != 0 || state->channelSize <= 0 ||
      state->innerSize <= 0) {
    return false;
  }

  const int n_tasks = htp_ops_select_pick_task_count(state->outSize);
  if (n_tasks <= 1) {
    htp_ops_select_prelu_channel_fp16_range(state, 0, state->outSize);
    return true;
  }

  state->n_tasks = n_tasks;
  state->task_id = 0;
  worker_pool_job_t job;
  job.fptr = htp_ops_select_prelu_channel_fp16_worker;
  job.dptr = state;
  worker_pool_synctoken_init(&(state->sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state->sync_ctx));
  return true;
}

static inline void htp_ops_select_same_fp16_i32_cond_range(HtpOpsSelectTaskState* state, int start, int end) {
  uint16_t* out = (uint16_t*)state->dst;
  const uint16_t* in1 = (const uint16_t*)state->src1;
  const uint16_t* in2 = (const uint16_t*)state->src2;
  const uint32_t* cond = (const uint32_t*)state->cond;
  const int vec_len = 128 / (int)sizeof(uint16_t);
  const int vec_start = (start + vec_len - 1) & -vec_len;
  const int vec_end = end & -vec_len;
  int i = start;
  for (; i < vec_start && i < end; ++i) {
    out[i] = cond[i] != 0 ? in1[i] : in2[i];
  }
  HVX_Vector zero = Q6_V_vzero();
  for (; i < vec_end; i += vec_len) {
    HVX_Vector cond0 = vmemu((const HVX_Vector*)(cond + i));
    HVX_Vector cond1 = vmemu((const HVX_Vector*)(cond + i + 32));
    HVX_Vector mask = htp_ops_select_i32_cond_to_fp16(cond0, cond1, 1, 0);
    HVX_VectorPred pred = Q6_Q_vcmp_gt_VhVh(mask, zero);
    HVX_Vector v1 = vmemu((const HVX_Vector*)(in1 + i));
    HVX_Vector v2 = vmemu((const HVX_Vector*)(in2 + i));
    vmemu((HVX_Vector*)(out + i)) = Q6_V_vmux_QVV(pred, v1, v2);
  }
  for (; i < end; ++i) {
    out[i] = cond[i] != 0 ? in1[i] : in2[i];
  }
}

static void htp_ops_select_same_fp16_i32_cond_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsSelectTaskState* state = (HtpOpsSelectTaskState*)data;
  const int grain = 4096;
  while (1) {
    const int task_id = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int start = task_id * grain;
    if (start >= state->outSize) {
      break;
    }
    int end = start + grain;
    if (end > state->outSize) {
      end = state->outSize;
    }
    htp_ops_select_same_fp16_i32_cond_range(state, start, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_select_try_same_fp16_i32_cond_parallel(HtpOpsSelectTaskState* state) {
  if (state->bytes != 2 || state->condBytes != 4 || state->condStep != 1 ||
      state->in1Mode != 1 || state->in2Mode != 1) {
    return false;
  }
  const int n_tasks = htp_ops_select_pick_task_count(state->outSize);
  if (n_tasks <= 1) {
    htp_ops_select_same_fp16_i32_cond_range(state, 0, state->outSize);
    return true;
  }
  state->n_tasks = n_tasks;
  state->task_id = 0;
  worker_pool_job_t job;
  job.fptr = htp_ops_select_same_fp16_i32_cond_worker;
  job.dptr = state;
  worker_pool_synctoken_init(&(state->sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state->sync_ctx));
  return true;
}

AEEResult htp_ops_select(uint8_t* dst, uint8_t* cond_ptr, uint8_t* src1_ptr, uint8_t* src2_ptr,
                         int32_t outSize, int32_t condSize, int32_t in1Size, int32_t in2Size,
                         int32_t bytes, int32_t condBytes, int32_t channelSize, int32_t innerSize) {
  if (bytes != 1 && bytes != 2 && bytes != 4) {
    return -1;
  }
  if (condBytes != 1 && condBytes != 2 && condBytes != 4) {
    return -1;
  }
  if (outSize <= 0) {
    return 0;
  }
  const int in1Mode = (in1Size == 1) ? 0 : ((in1Size == outSize) ? 1 : 2);
  const int in2Mode = (in2Size == 1) ? 0 : ((in2Size == outSize) ? 1 : 2);
  if (!((condSize == outSize || condSize == 1) &&
        (in1Mode != 2 || (channelSize > 0 && innerSize > 0 && in1Size == channelSize)) &&
        (in2Mode != 2 || (channelSize > 0 && innerSize > 0 && in2Size == channelSize)))) {
    return -2;
  }

  const int condStep = (condSize == 1) ? 0 : 1;
  const int in1Step = (in1Size == 1) ? 0 : 1;
  const int in2Step = (in2Size == 1) ? 0 : 1;

  HtpOpsSelectTaskState state = {};
  state.outSize = outSize;
  state.condStep = condStep;
  state.in1Step = in1Step;
  state.in2Step = in2Step;
  state.in1Mode = in1Mode;
  state.in2Mode = in2Mode;
  state.channelSize = channelSize;
  state.innerSize = innerSize;
  state.bytes = bytes;
  state.condBytes = condBytes;
  state.dst = dst;
  state.cond = cond_ptr;
  state.src1 = src1_ptr;
  state.src2 = src2_ptr;

  if (htp_ops_select_try_same_fp16_i32_cond_parallel(&state)) {
    return 0;
  }
  if (htp_ops_select_try_prelu_channel_fp16(&state)) {
    return 0;
  }

  const int n_tasks = htp_ops_select_pick_task_count(outSize);
  if (n_tasks <= 1) {
    htp_ops_select_compute_range(&state, 0, outSize);
    return 0;
  }

  state.n_tasks = n_tasks;
  worker_pool_job_t job;
  job.fptr = htp_ops_select_worker_loop;
  job.dptr = &state;

  worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return 0;
}

typedef enum {
  HTP_OPS_REDUCTION_SUM = 1,
  HTP_OPS_REDUCTION_MAXIMUM = 2,
  HTP_OPS_REDUCTION_MEAN = 3,
} HtpOpsReductionOpType;

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  int n_tasks;
  int outside;
  int reduce;
  int inside;
  int opType;
  __fp16* dst;
  const __fp16* src;
} HtpOpsReductionTaskState;

static inline void htp_ops_reduction_copy_fp16(uint8_t* dst, const uint8_t* src, size_t bytes) {
  const size_t vecBytes = __HVX_LENGTH__;
  size_t offset = 0;
  for (; offset + vecBytes <= bytes; offset += vecBytes) {
    HVX_Vector v = vmemu((const HVX_Vector*)(src + offset));
    vmemu((HVX_Vector*)(dst + offset)) = v;
  }
  if (offset < bytes) {
    memcpy(dst + offset, src + offset, bytes - offset);
  }
}

static inline void htp_ops_reduce_fp16_scalar_range(HtpOpsReductionTaskState* state, int start, int end) {
  const int reduce = state->reduce;
  const int inside = state->inside;
  const __fp16* src = state->src;
  __fp16* dst = state->dst;
  for (int index = start; index < end; ++index) {
    const int o = index / inside;
    const int i = index - o * inside;
    const __fp16* src_base = src + (o * reduce * inside + i);
    if (state->opType == HTP_OPS_REDUCTION_MAXIMUM) {
      __fp16 best = src_base[0];
      for (int r = 1; r < reduce; ++r) {
        const __fp16 value = src_base[r * inside];
        best = value > best ? value : best;
      }
      dst[index] = best;
    } else {
      float sum = 0.0f;
      for (int r = 0; r < reduce; ++r) {
        sum += (float)src_base[r * inside];
      }
      if (state->opType == HTP_OPS_REDUCTION_MEAN) {
        sum /= (float)reduce;
      }
      dst[index] = (__fp16)sum;
    }
  }
}

static inline float htp_ops_reduction_reduce_sum2_f32(HVX_Vector acc0, HVX_Vector acc1) {
  HVX_Vector v = Q6_Vsf_vadd_VsfVsf(acc0, acc1);
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64)));
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32)));
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16)));
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8)));
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4)));
  float tmp[32] __attribute__((aligned(128)));
  *(HVX_Vector*)tmp = v;
  return tmp[0];
}

static inline uint16_t htp_ops_reduce_sum_fp16_inside1_hvx(const __fp16* src, int reduce) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = reduce & -vec_len;
  HVX_Vector acc0 = Q6_V_vzero();
  HVX_Vector acc1 = Q6_V_vzero();
  int r = 0;
  for (; r < vec_end; r += vec_len) {
    HVX_Vector v = vmemu((const HVX_Vector*)(src + r));
    HVX_VectorPair sf = hvx_my_vhf_to_wsf(v);
    acc0 = Q6_Vsf_vadd_VsfVsf(acc0, Q6_V_lo_W(sf));
    acc1 = Q6_Vsf_vadd_VsfVsf(acc1, Q6_V_hi_W(sf));
  }
  if (r < reduce && reduce - r <= vec_len) {
    __fp16 tail[vec_len] __attribute__((aligned(128))) = {};
    memcpy(tail, src + r, (size_t)(reduce - r) * sizeof(__fp16));
    HVX_VectorPair sf = hvx_my_vhf_to_wsf(vmem(tail));
    acc0 = Q6_Vsf_vadd_VsfVsf(acc0, Q6_V_lo_W(sf));
    acc1 = Q6_Vsf_vadd_VsfVsf(acc1, Q6_V_hi_W(sf));
    r = reduce;
  }
  float sum = htp_ops_reduction_reduce_sum2_f32(acc0, acc1);
  for (; r < reduce; ++r) {
    sum += (float)src[r];
  }
  __fp16 result = (__fp16)sum;
  return *(uint16_t*)&result;
}

static inline uint16_t htp_ops_reduce_max_fp16_inside1_hvx(const __fp16* src, int reduce) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = reduce & -vec_len;
  __fp16 bestScalar = src[0];
  HVX_Vector best = Q6_Vh_vsplat_R(((const uint16_t*)src)[0]);
  int r = 0;
  for (; r < vec_end; r += vec_len) {
    HVX_Vector v = vmemu((const HVX_Vector*)(src + r));
    best = Q6_Vhf_vmax_VhfVhf(best, v);
  }
  best = Q6_Vhf_vmax_VhfVhf(best, Q6_V_vror_VR(best, 64));
  best = Q6_Vhf_vmax_VhfVhf(best, Q6_V_vror_VR(best, 32));
  best = Q6_Vhf_vmax_VhfVhf(best, Q6_V_vror_VR(best, 16));
  best = Q6_Vhf_vmax_VhfVhf(best, Q6_V_vror_VR(best, 8));
  best = Q6_Vhf_vmax_VhfVhf(best, Q6_V_vror_VR(best, 4));
  best = Q6_Vhf_vmax_VhfVhf(best, Q6_V_vror_VR(best, 2));
  uint16_t tmp[vec_len] __attribute__((aligned(128)));
  vmemu((HVX_Vector*)tmp) = best;
  bestScalar = *(__fp16*)&tmp[0];
  for (; r < reduce; ++r) {
    const __fp16 value = src[r];
    bestScalar = value > bestScalar ? value : bestScalar;
  }
  return *(uint16_t*)&bestScalar;
}

static inline void htp_ops_reduce_fp16_inside1_range(HtpOpsReductionTaskState* state,
                                                     int outsideStart, int outsideEnd) {
  const int reduce = state->reduce;
  const __fp16* src = state->src;
  __fp16* dst = state->dst;
  for (int o = outsideStart; o < outsideEnd; ++o) {
    const __fp16* src_outer = src + (size_t)o * reduce;
    if (state->opType == HTP_OPS_REDUCTION_MAXIMUM) {
      const uint16_t bits = htp_ops_reduce_max_fp16_inside1_hvx(src_outer, reduce);
      dst[o] = *(__fp16*)&bits;
    } else {
      const uint16_t bits = htp_ops_reduce_sum_fp16_inside1_hvx(src_outer, reduce);
      float value = (float)(*(__fp16*)&bits);
      if (state->opType == HTP_OPS_REDUCTION_MEAN) {
        value /= (float)reduce;
      }
      dst[o] = (__fp16)value;
    }
  }
}

static inline void htp_ops_reduce_fp16_inside_vector_range(HtpOpsReductionTaskState* state,
                                                           int outsideStart, int outsideEnd) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = state->inside & -vec_len;
  const int reduce = state->reduce;
  const int inside = state->inside;
  const __fp16* src = state->src;
  __fp16* dst = state->dst;
  for (int o = outsideStart; o < outsideEnd; ++o) {
    const __fp16* src_outer = src + o * reduce * inside;
    __fp16* dst_outer = dst + o * inside;
    int i = 0;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector acc = vmemu((const HVX_Vector*)(src_outer + i));
      for (int r = 1; r < reduce; ++r) {
        HVX_Vector v = vmemu((const HVX_Vector*)(src_outer + r * inside + i));
        if (state->opType == HTP_OPS_REDUCTION_MAXIMUM) {
          acc = Q6_Vhf_vmax_VhfVhf(acc, v);
        } else {
          acc = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(acc, v));
        }
      }
      if (state->opType == HTP_OPS_REDUCTION_MEAN) {
        const __fp16 scaleValue = (__fp16)(1.0f / (float)reduce);
        uint16_t scaleBits = *(uint16_t*)&scaleValue;
        HVX_Vector scale = Q6_Vh_vsplat_R(scaleBits);
        acc = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(acc, scale));
      }
      vmemu((HVX_Vector*)(dst_outer + i)) = acc;
    }
    if (i < inside) {
      HtpOpsReductionTaskState tail = *state;
      tail.outside = 1;
      tail.src = src_outer;
      tail.dst = dst_outer;
      htp_ops_reduce_fp16_scalar_range(&tail, i, inside);
    }
  }
}

static inline void htp_ops_reduce_fp16_inside_small_vector_range(HtpOpsReductionTaskState* state,
                                                                 int outsideStart, int outsideEnd) {
  const int reduce = state->reduce;
  const int inside = state->inside;
  const int rowBytes = inside * (int)sizeof(__fp16);
  const HVX_VectorPred qTail = Q6_Q_vsetq_R(rowBytes);
  const HVX_Vector zero = Q6_V_vzero();
  const __fp16* src = state->src;
  __fp16* dst = state->dst;
  for (int o = outsideStart; o < outsideEnd; ++o) {
    __fp16 tmp[64] __attribute__((aligned(128)));
    const __fp16* src_outer = src + o * reduce * inside;
    __fp16* dst_outer = dst + o * inside;
    if (state->opType == HTP_OPS_REDUCTION_MAXIMUM) {
      HVX_Vector acc = zero;
      for (int r = 0; r < reduce; ++r) {
        memcpy(tmp, src_outer + r * inside, rowBytes);
        HVX_Vector v = Q6_V_vmux_QVV(qTail, vmem((const HVX_Vector*)tmp), zero);
        acc = r == 0 ? v : Q6_Vhf_vmax_VhfVhf(acc, v);
      }
      vstu_variable(dst_outer, (uint32_t)rowBytes, acc);
    } else {
      HVX_Vector acc0 = zero;
      HVX_Vector acc1 = zero;
      for (int r = 0; r < reduce; ++r) {
        memcpy(tmp, src_outer + r * inside, rowBytes);
        HVX_Vector v = Q6_V_vmux_QVV(qTail, vmem((const HVX_Vector*)tmp), zero);
        HVX_VectorPair sf = hvx_my_vhf_to_wsf(v);
        acc0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc0, Q6_V_lo_W(sf)));
        acc1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc1, Q6_V_hi_W(sf)));
      }
      if (state->opType == HTP_OPS_REDUCTION_MEAN) {
        const float scaleValue = 1.0f / (float)reduce;
        int32_t scaleBits = 0;
        memcpy(&scaleBits, &scaleValue, sizeof(scaleBits));
        HVX_Vector scale = Q6_V_vsplat_R(scaleBits);
        acc0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(acc0, scale));
        acc1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(acc1, scale));
      }
      HVX_Vector out = Q6_Vhf_vcvt_VsfVsf(acc0, acc1);
      vstu_variable(dst_outer, (uint32_t)rowBytes, out);
    }
  }
}

static void htp_ops_reduction_worker_loop(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsReductionTaskState* state = (HtpOpsReductionTaskState*)data;
  if (state->inside == 1) {
    const int blocks_per_task = (state->outside + state->n_tasks - 1) / state->n_tasks;
    while (1) {
      unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
      if ((int)task_id >= state->n_tasks) {
        break;
      }
      int start = (int)task_id * blocks_per_task;
      int end = start + blocks_per_task;
      if (end > state->outside) {
        end = state->outside;
      }
      if (start < end) {
        htp_ops_reduce_fp16_inside1_range(state, start, end);
      }
    }
  } else if (state->inside >= 32) {
    const int blocks_per_task = (state->outside + state->n_tasks - 1) / state->n_tasks;
    while (1) {
      unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
      if ((int)task_id >= state->n_tasks) {
        break;
      }
      int start = (int)task_id * blocks_per_task;
      int end = start + blocks_per_task;
      if (end > state->outside) {
        end = state->outside;
      }
      if (start < end) {
        if (state->inside >= 64) {
          htp_ops_reduce_fp16_inside_vector_range(state, start, end);
        } else {
          htp_ops_reduce_fp16_inside_small_vector_range(state, start, end);
        }
      }
    }
  } else {
    const int output_size = state->outside * state->inside;
    const int grain = 2048;
    const int total_blocks = (output_size + grain - 1) / grain;
    const int blocks_per_task = (total_blocks + state->n_tasks - 1) / state->n_tasks;
    while (1) {
      unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
      if ((int)task_id >= state->n_tasks) {
        break;
      }
      int start = (int)task_id * blocks_per_task * grain;
      int end = start + blocks_per_task * grain;
      if (start >= output_size) {
        break;
      }
      if (end > output_size) {
        end = output_size;
      }
      htp_ops_reduce_fp16_scalar_range(state, start, end);
    }
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline int htp_ops_reduction_pick_task_count(int outside, int inside) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap <= 1) {
    return 1;
  }
  int work_units = inside >= 32 ? outside : outside * inside;
  int task_count = (work_units + 2047) / 2048;
  if (inside >= 32) {
    task_count = outside;
  }
  if (task_count < 2) {
    return 1;
  }
  if (task_count > (int)worker_cap) {
    task_count = (int)worker_cap;
  }
  return task_count;
}

AEEResult htp_ops_reduction(uint8_t* dst, const uint8_t* src, int32_t outside, int32_t reduce,
                            int32_t inside, int32_t opType, int32_t bytes) {
  if (dst == nullptr || src == nullptr || outside <= 0 || reduce <= 0 || inside <= 0) {
    return AEE_EBADPARM;
  }
  if (bytes != 2 || (opType != HTP_OPS_REDUCTION_SUM && opType != HTP_OPS_REDUCTION_MAXIMUM &&
                     opType != HTP_OPS_REDUCTION_MEAN)) {
    return AEE_EBADPARM;
  }
  if (reduce == 1) {
    htp_ops_reduction_copy_fp16(dst, src, (size_t)outside * inside * sizeof(__fp16));
    return AEE_SUCCESS;
  }
  HtpOpsReductionTaskState state = {};
  state.outside = outside;
  state.reduce = reduce;
  state.inside = inside;
  state.opType = opType;
  state.dst = (__fp16*)dst;
  state.src = (const __fp16*)src;

  const int n_tasks = htp_ops_reduction_pick_task_count(outside, inside);
  if (n_tasks <= 1) {
    if (inside == 1) {
      htp_ops_reduce_fp16_inside1_range(&state, 0, outside);
    } else if (inside >= 64) {
      htp_ops_reduce_fp16_inside_vector_range(&state, 0, outside);
    } else if (inside >= 32) {
      htp_ops_reduce_fp16_inside_small_vector_range(&state, 0, outside);
    } else {
      htp_ops_reduce_fp16_scalar_range(&state, 0, outside * inside);
    }
    return AEE_SUCCESS;
  }

  state.n_tasks = n_tasks;
  worker_pool_job_t job;
  job.fptr = htp_ops_reduction_worker_loop;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return AEE_SUCCESS;
}

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  int n_tasks;
  int outside;
  int reduce;
  int inside;
  __fp16* dst;
  const __fp16* src;
  const __fp16* mask;
} HtpOpsMaskedReductionTaskState;

static inline void htp_ops_masked_reduce_sum_inside_vector_range(HtpOpsMaskedReductionTaskState* state,
                                                                 int outsideStart, int outsideEnd) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = state->inside & -vec_len;
  const int reduce = state->reduce;
  const int inside = state->inside;
  const __fp16* src = state->src;
  const __fp16* mask = state->mask;
  __fp16* dst = state->dst;
  for (int o = outsideStart; o < outsideEnd; ++o) {
    const __fp16* src_outer = src + (int64_t)o * reduce * inside;
    const __fp16* mask_outer = mask + (int64_t)o * reduce;
    __fp16* dst_outer = dst + (int64_t)o * inside;
    int i = 0;
    for (; i < vec_end; i += vec_len) {
      HVX_Vector acc = Q6_V_vzero();
      for (int r = 0; r < reduce; ++r) {
        if (mask_outer[r] == (__fp16)0.0f) {
          continue;
        }
        HVX_Vector v = vmemu((const HVX_Vector*)(src_outer + (int64_t)r * inside + i));
        acc = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(acc, v));
      }
      vmemu((HVX_Vector*)(dst_outer + i)) = acc;
    }
    for (; i < inside; ++i) {
      float sum = 0.0f;
      for (int r = 0; r < reduce; ++r) {
        if (mask_outer[r] != (__fp16)0.0f) {
          sum += (float)src_outer[(int64_t)r * inside + i];
        }
      }
      dst_outer[i] = (__fp16)sum;
    }
  }
}

static void htp_ops_masked_reduction_worker_loop(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsMaskedReductionTaskState* state = (HtpOpsMaskedReductionTaskState*)data;
  const int blocks_per_task = (state->outside + state->n_tasks - 1) / state->n_tasks;
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= state->n_tasks) {
      break;
    }
    int start = (int)task_id * blocks_per_task;
    int end = start + blocks_per_task;
    if (end > state->outside) {
      end = state->outside;
    }
    if (start < end) {
      htp_ops_masked_reduce_sum_inside_vector_range(state, start, end);
    }
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

AEEResult htp_ops_masked_reduction(uint8_t* dst, const uint8_t* src, const uint8_t* mask,
                                   int32_t outside, int32_t reduce, int32_t inside,
                                   int32_t opType, int32_t bytes) {
  if (dst == nullptr || src == nullptr || mask == nullptr || outside <= 0 || reduce <= 0 || inside <= 0) {
    return AEE_EBADPARM;
  }
  if (bytes != 2 || opType != HTP_OPS_REDUCTION_SUM) {
    return AEE_EBADPARM;
  }
  HtpOpsMaskedReductionTaskState state = {};
  state.outside = outside;
  state.reduce = reduce;
  state.inside = inside;
  state.dst = (__fp16*)dst;
  state.src = (const __fp16*)src;
  state.mask = (const __fp16*)mask;

  int n_tasks = g_max_num_workers;
  if (n_tasks > outside) {
    n_tasks = outside;
  }
  if (n_tasks <= 1) {
    htp_ops_masked_reduce_sum_inside_vector_range(&state, 0, outside);
    return AEE_SUCCESS;
  }

  state.n_tasks = n_tasks;
  worker_pool_job_t job;
  job.fptr = htp_ops_masked_reduction_worker_loop;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return AEE_SUCCESS;
}

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  int n_tasks;
  int size;
  int grain;
  __fp16* dst;
  const __fp16* src;
  __fp16 minValue;
  __fp16 maxValue;
} HtpOpsRelu6TaskState;

static inline uint16_t htp_ops_relu6_fp16_bits(const void* value) {
  uint16_t bits;
  memcpy(&bits, value, sizeof(bits));
  return bits;
}

static inline bool htp_ops_relu6_is_aligned_128(const void* ptr) {
  return (((uintptr_t)ptr) & 127) == 0;
}

static inline HVX_Vector htp_ops_relu6_load_vec_aligned(const __fp16* src) {
  return vmem(src);
}

static inline HVX_Vector htp_ops_relu6_load_vec_unaligned(const __fp16* src) {
  return vmemu((const HVX_Vector*)src);
}

static inline void htp_ops_relu6_store_vec_aligned(__fp16* dst, HVX_Vector v) {
  vmem(dst) = v;
}

static inline void htp_ops_relu6_store_vec_unaligned(__fp16* dst, HVX_Vector v) {
  vmemu((HVX_Vector*)dst) = v;
}

static inline void htp_ops_relu6_store_tail(__fp16* dst, const __fp16* src, int remain,
                                            _Float16 minValue, _Float16 maxValue) {
  for (int i = 0; i < remain; ++i) {
    __fp16 v = src[i];
    v = v < minValue ? minValue : v;
    v = v > maxValue ? maxValue : v;
    dst[i] = v;
  }
}

static inline void htp_ops_relu6_fp16_range(HtpOpsRelu6TaskState* state, int start, int end) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  int i = start;
  const int vec_end = start + ((end - start) / vec_len) * vec_len;
  const __fp16 minValue = state->minValue;
  const __fp16 maxValue = state->maxValue;
  const uint16_t minBits = htp_ops_relu6_fp16_bits(&minValue);
  const uint16_t maxBits = htp_ops_relu6_fp16_bits(&maxValue);
  const HVX_Vector zero_v = Q6_V_vzero();
  const HVX_Vector max_v = Q6_Vh_vsplat_R(maxBits);
  const bool aligned = htp_ops_relu6_is_aligned_128(state->src + i) &&
                       htp_ops_relu6_is_aligned_128(state->dst + i);

#define HTP_OPS_RELU6_RUN_VECTOR_LOOP(load_vec, store_vec)                                      \
  do {                                                                                          \
    if (minBits == 0 && maxBits >= 0x7c00) {                                                    \
      for (; i + 4 * vec_len <= vec_end; i += 4 * vec_len) {                                    \
        HVX_Vector v0 = load_vec(state->src + i);                                               \
        HVX_Vector v1 = load_vec(state->src + i + vec_len);                                     \
        HVX_Vector v2 = load_vec(state->src + i + 2 * vec_len);                                 \
        HVX_Vector v3 = load_vec(state->src + i + 3 * vec_len);                                 \
        store_vec(state->dst + i, Q6_Vhf_vmax_VhfVhf(v0, zero_v));                              \
        store_vec(state->dst + i + vec_len, Q6_Vhf_vmax_VhfVhf(v1, zero_v));                    \
        store_vec(state->dst + i + 2 * vec_len, Q6_Vhf_vmax_VhfVhf(v2, zero_v));                \
        store_vec(state->dst + i + 3 * vec_len, Q6_Vhf_vmax_VhfVhf(v3, zero_v));                \
      }                                                                                         \
      for (; i < vec_end; i += vec_len) {                                                       \
        HVX_Vector v = load_vec(state->src + i);                                                \
        store_vec(state->dst + i, Q6_Vhf_vmax_VhfVhf(v, zero_v));                               \
      }                                                                                         \
    } else if (minBits == 0) {                                                                  \
      for (; i + 4 * vec_len <= vec_end; i += 4 * vec_len) {                                    \
        HVX_Vector v0 = load_vec(state->src + i);                                               \
        HVX_Vector v1 = load_vec(state->src + i + vec_len);                                     \
        HVX_Vector v2 = load_vec(state->src + i + 2 * vec_len);                                 \
        HVX_Vector v3 = load_vec(state->src + i + 3 * vec_len);                                 \
        v0 = Q6_Vhf_vmax_VhfVhf(v0, zero_v);                                                    \
        v1 = Q6_Vhf_vmax_VhfVhf(v1, zero_v);                                                    \
        v2 = Q6_Vhf_vmax_VhfVhf(v2, zero_v);                                                    \
        v3 = Q6_Vhf_vmax_VhfVhf(v3, zero_v);                                                    \
        store_vec(state->dst + i, Q6_Vhf_vmin_VhfVhf(v0, max_v));                               \
        store_vec(state->dst + i + vec_len, Q6_Vhf_vmin_VhfVhf(v1, max_v));                     \
        store_vec(state->dst + i + 2 * vec_len, Q6_Vhf_vmin_VhfVhf(v2, max_v));                 \
        store_vec(state->dst + i + 3 * vec_len, Q6_Vhf_vmin_VhfVhf(v3, max_v));                 \
      }                                                                                         \
      for (; i < vec_end; i += vec_len) {                                                       \
        HVX_Vector v = load_vec(state->src + i);                                                \
        v = Q6_Vhf_vmax_VhfVhf(v, zero_v);                                                      \
        store_vec(state->dst + i, Q6_Vhf_vmin_VhfVhf(v, max_v));                                \
      }                                                                                         \
    } else {                                                                                    \
      const HVX_Vector min_v = Q6_Vh_vsplat_R(minBits);                                        \
      for (; i + 4 * vec_len <= vec_end; i += 4 * vec_len) {                                    \
        HVX_Vector v0 = load_vec(state->src + i);                                               \
        HVX_Vector v1 = load_vec(state->src + i + vec_len);                                     \
        HVX_Vector v2 = load_vec(state->src + i + 2 * vec_len);                                 \
        HVX_Vector v3 = load_vec(state->src + i + 3 * vec_len);                                 \
        v0 = Q6_Vhf_vmax_VhfVhf(v0, min_v);                                                     \
        v1 = Q6_Vhf_vmax_VhfVhf(v1, min_v);                                                     \
        v2 = Q6_Vhf_vmax_VhfVhf(v2, min_v);                                                     \
        v3 = Q6_Vhf_vmax_VhfVhf(v3, min_v);                                                     \
        store_vec(state->dst + i, Q6_Vhf_vmin_VhfVhf(v0, max_v));                               \
        store_vec(state->dst + i + vec_len, Q6_Vhf_vmin_VhfVhf(v1, max_v));                     \
        store_vec(state->dst + i + 2 * vec_len, Q6_Vhf_vmin_VhfVhf(v2, max_v));                 \
        store_vec(state->dst + i + 3 * vec_len, Q6_Vhf_vmin_VhfVhf(v3, max_v));                 \
      }                                                                                         \
      for (; i < vec_end; i += vec_len) {                                                       \
        HVX_Vector v = load_vec(state->src + i);                                                \
        v = Q6_Vhf_vmax_VhfVhf(v, min_v);                                                       \
        store_vec(state->dst + i, Q6_Vhf_vmin_VhfVhf(v, max_v));                                \
      }                                                                                         \
    }                                                                                           \
  } while (0)

  if (aligned) {
    HTP_OPS_RELU6_RUN_VECTOR_LOOP(htp_ops_relu6_load_vec_aligned, htp_ops_relu6_store_vec_aligned);
  } else {
    HTP_OPS_RELU6_RUN_VECTOR_LOOP(htp_ops_relu6_load_vec_unaligned, htp_ops_relu6_store_vec_unaligned);
  }

#undef HTP_OPS_RELU6_RUN_VECTOR_LOOP

  if (i < end) {
    htp_ops_relu6_store_tail(state->dst + i, state->src + i, end - i, minValue, maxValue);
  }
}

static void htp_ops_relu6_worker_loop(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsRelu6TaskState* state = (HtpOpsRelu6TaskState*)data;
  const int grain = state->grain;
  const int total_blocks = (state->size + grain - 1) / grain;
  const int blocks_per_task = (total_blocks + state->n_tasks - 1) / state->n_tasks;
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= state->n_tasks) {
      break;
    }
    int start = (int)task_id * blocks_per_task * grain;
    int end = start + blocks_per_task * grain;
    if (start >= state->size) {
      break;
    }
    if (end > state->size) {
      end = state->size;
    }
    htp_ops_relu6_fp16_range(state, start, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

AEEResult htp_ops_relu6(uint8_t* dst, const uint8_t* src, int32_t size, int32_t bytes,
                        float minValue, float maxValue) {
  if (dst == nullptr || src == nullptr || size < 0 || bytes != 2) {
    return AEE_EBADPARM;
  }
  HtpOpsRelu6TaskState state = {};
  state.size = size;
  state.dst = (__fp16*)dst;
  state.src = (const __fp16*)src;
  state.minValue = (__fp16)minValue;
  state.maxValue = (__fp16)maxValue;
  state.grain = 32768;
  int n_tasks = (size + state.grain - 1) / state.grain;
  if (n_tasks > (int)g_max_num_workers) {
    n_tasks = (int)g_max_num_workers;
  }
  if (n_tasks <= 1) {
    htp_ops_relu6_fp16_range(&state, 0, size);
    return AEE_SUCCESS;
  }
  state.n_tasks = n_tasks;
  worker_pool_job_t job;
  job.fptr = htp_ops_relu6_worker_loop;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return AEE_SUCCESS;
}

AEEResult htp_ops_relu(uint8_t* dst, const uint8_t* src, int32_t size, int32_t bytes, float slope) {
  if (dst == nullptr || src == nullptr || size < 0 || bytes != 2) {
    return AEE_EBADPARM;
  }
  HtpOpsRelu6TaskState state = {};
  state.size = size;
  state.dst = (__fp16*)dst;
  state.src = (const __fp16*)src;
  state.minValue = (__fp16)0.0f;
  state.maxValue = (__fp16)INFINITY;
  state.grain = 32768;
  if (slope == 0.0f) {
    int n_tasks = (size + state.grain - 1) / state.grain;
    if (n_tasks > (int)g_max_num_workers) {
      n_tasks = (int)g_max_num_workers;
    }
    if (n_tasks <= 1) {
      htp_ops_relu6_fp16_range(&state, 0, size);
      return AEE_SUCCESS;
    }
    state.n_tasks = n_tasks;
    worker_pool_job_t job;
    job.fptr = htp_ops_relu6_worker_loop;
    job.dptr = &state;
    worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
    for (int i = 0; i < n_tasks; ++i) {
      worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&(state.sync_ctx));
    return AEE_SUCCESS;
  }

  __fp16* dstFp16 = (__fp16*)dst;
  const __fp16* srcFp16 = (const __fp16*)src;
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  const __fp16 slopeFp16 = (__fp16)slope;
  uint16_t slopeBits;
  memcpy(&slopeBits, &slopeFp16, sizeof(slopeBits));
  const HVX_Vector slopeVec = Q6_Vh_vsplat_R(slopeBits);
  const HVX_Vector zeroVec = Q6_V_vzero();
  const bool aligned = htp_ops_relu6_is_aligned_128(srcFp16) && htp_ops_relu6_is_aligned_128(dstFp16);
  int i = 0;
  if (aligned) {
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v = vmem((const HVX_Vector*)(srcFp16 + i));
      HVX_Vector scaled = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, slopeVec));
      HVX_VectorPred negative = Q6_Q_vcmp_gt_VhfVhf(zeroVec, v);
      vmem((HVX_Vector*)(dstFp16 + i)) = Q6_V_vmux_QVV(negative, scaled, v);
    }
  } else {
    for (; i < vec_end; i += vec_len) {
      HVX_Vector v = vmemu((const HVX_Vector*)(srcFp16 + i));
      HVX_Vector scaled = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, slopeVec));
      HVX_VectorPred negative = Q6_Q_vcmp_gt_VhfVhf(zeroVec, v);
      vmemu((HVX_Vector*)(dstFp16 + i)) = Q6_V_vmux_QVV(negative, scaled, v);
    }
  }
  for (; i < size; ++i) {
    const __fp16 v = srcFp16[i];
    dstFp16[i] = v < (__fp16)0.0f ? v * slopeFp16 : v;
  }
  return AEE_SUCCESS;
}

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  int                plane;
  int                cPack;
  int                pack;
  __fp16*            dst;
  const __fp16*      src;
  const __fp16*      slope;
} HtpOpsPReluTaskState;

static inline void htp_ops_prelu_channel_fp16_blocks(HtpOpsPReluTaskState* state, int start, int end) {
  const HVX_Vector zeroVec = Q6_V_vzero();
  for (int block = start; block < end; ++block) {
    const int cp = block / state->plane;
    const int offset = block * state->pack;
    HVX_Vector v = vmemu((const HVX_Vector*)(state->src + offset));
    HVX_Vector slopeVec = vmemu((const HVX_Vector*)(state->slope + cp * state->pack));
    HVX_Vector scaled = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, slopeVec));
    HVX_VectorPred negative = Q6_Q_vcmp_gt_VhfVhf(zeroVec, v);
    vmemu((HVX_Vector*)(state->dst + offset)) = Q6_V_vmux_QVV(negative, scaled, v);
  }
}

static void htp_ops_prelu_channel_worker_loop(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsPReluTaskState* state = (HtpOpsPReluTaskState*)data;
  const int grain = 256;
  const int totalBlocks = state->plane * state->cPack;
  const int totalGrains = (totalBlocks + grain - 1) / grain;
  const int grainsPerTask = (totalGrains + state->n_tasks - 1) / state->n_tasks;
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= state->n_tasks) {
      break;
    }
    int start = (int)task_id * grainsPerTask * grain;
    if (start >= totalBlocks) {
      break;
    }
    int end = start + grainsPerTask * grain;
    if (end > totalBlocks) {
      end = totalBlocks;
    }
    htp_ops_prelu_channel_fp16_blocks(state, start, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

AEEResult htp_ops_prelu(uint8_t* dst, const uint8_t* src, const uint8_t* slope, int32_t size, int32_t bytes,
                        int32_t plane, int32_t channel, int32_t slopeCount, int32_t pack) {
  if (dst == nullptr || src == nullptr || slope == nullptr || size < 0 || bytes != 2 ||
      plane <= 0 || channel <= 0 || slopeCount <= 0 || pack <= 0) {
    return AEE_EBADPARM;
  }
  const __fp16* slopeFp16 = (const __fp16*)slope;
  if (slopeCount == 1) {
    return htp_ops_relu(dst, src, size, bytes, (float)slopeFp16[0]);
  }

  const int vec_len = 128 / (int)sizeof(__fp16);
  if (pack != vec_len || slopeCount != channel) {
    return AEE_EUNSUPPORTED;
  }
  const int cPack = (channel + pack - 1) / pack;
  if (size < cPack * plane * pack) {
    return AEE_EBADPARM;
  }

  HtpOpsPReluTaskState state = {};
  state.plane = plane;
  state.cPack = cPack;
  state.pack = pack;
  state.dst = (__fp16*)dst;
  state.src = (const __fp16*)src;
  state.slope = slopeFp16;

  const int totalBlocks = plane * cPack;
  int n_tasks = (totalBlocks + 255) / 256;
  if (n_tasks > (int)g_max_num_workers) {
    n_tasks = (int)g_max_num_workers;
  }
  if (n_tasks <= 1) {
    htp_ops_prelu_channel_fp16_blocks(&state, 0, totalBlocks);
    return AEE_SUCCESS;
  }
  state.n_tasks = n_tasks;
  worker_pool_job_t job;
  job.fptr = htp_ops_prelu_channel_worker_loop;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return AEE_SUCCESS;
}

}  // extern "C"
