#include "dsp/mmap_mgr.h"
#include "dsp/hvx_utils.h"
#include <HAP_mem.h>
#include "AEEStdDef.h"
#include "remote.h"
#include "hexagon_protos.h"
#include "hexagon_types.h"
#include "htp_ops.h"
#include "AEEStdErr.h"
#include "qurt_memory.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "HAP_compute_res.h"
#include "dsp/worker_pool.h"

extern "C" {


#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

typedef struct {
  uint8_t* dst;
  const uint8_t* src;
  const uint8_t* scale;
  const uint8_t* bias;
  int plane;
  int pack;
  int c_begin;
  int c_end;
  int has_bias;
  worker_synctoken_t* sync;
} ScaleTask;

static inline void htp_ops_scale_c_range(const ScaleTask* task) {
  const int plane = task->plane;
  const int pack = task->pack;
  const size_t c_stride = (size_t)plane * pack * sizeof(int16_t);
  const size_t param_stride = (size_t)pack * sizeof(int16_t);
  for (int c = task->c_begin; c < task->c_end; ++c) {
    HVX_Vector v_scale = *(HVX_Vector*)(task->scale + (size_t)c * param_stride);
    HVX_Vector* __restrict src_v = (HVX_Vector*)(task->src + (size_t)c * c_stride);
    HVX_Vector* __restrict dst_v = (HVX_Vector*)(task->dst + (size_t)c * c_stride);

    if (plane == 128) {
      if (c == task->c_begin) {
        l2fetch(src_v, VLEN, VLEN, 128, 0);
      }
      if (c + 1 < task->c_end) {
        l2fetch(src_v + 128, VLEN, VLEN, 128, 0);
      }
    } else if (plane > PREFETCH_N_VECS) {
      int prefetch_n_vecs = Q6_R_min_RR(plane - PREFETCH_N_VECS, PREFETCH_N_VECS);
      l2fetch(src_v + PREFETCH_N_VECS, VLEN, VLEN, prefetch_n_vecs, 0);
    }

    int i = 0;
    if (task->has_bias) {
      HVX_Vector v_bias = *(HVX_Vector*)(task->bias + (size_t)c * param_stride);
      for (; i <= plane - 4; i += 4) {
        HVX_Vector v0 = Q6_Vhf_vmpy_VhfVhf(src_v[i], v_scale);
        HVX_Vector v1 = Q6_Vhf_vmpy_VhfVhf(src_v[i + 1], v_scale);
        HVX_Vector v2 = Q6_Vhf_vmpy_VhfVhf(src_v[i + 2], v_scale);
        HVX_Vector v3 = Q6_Vhf_vmpy_VhfVhf(src_v[i + 3], v_scale);
        dst_v[i] = Q6_Vhf_vadd_VhfVhf(v0, v_bias);
        dst_v[i + 1] = Q6_Vhf_vadd_VhfVhf(v1, v_bias);
        dst_v[i + 2] = Q6_Vhf_vadd_VhfVhf(v2, v_bias);
        dst_v[i + 3] = Q6_Vhf_vadd_VhfVhf(v3, v_bias);
      }
      for (; i < plane; ++i) {
        HVX_Vector v_out = Q6_Vhf_vmpy_VhfVhf(src_v[i], v_scale);
        dst_v[i] = Q6_Vhf_vadd_VhfVhf(v_out, v_bias);
      }
    } else {
      for (; i <= plane - 4; i += 4) {
        HVX_Vector v0 = src_v[i];
        HVX_Vector v1 = src_v[i + 1];
        HVX_Vector v2 = src_v[i + 2];
        HVX_Vector v3 = src_v[i + 3];
        dst_v[i] = Q6_Vhf_vmpy_VhfVhf(v0, v_scale);
        dst_v[i + 1] = Q6_Vhf_vmpy_VhfVhf(v1, v_scale);
        dst_v[i + 2] = Q6_Vhf_vmpy_VhfVhf(v2, v_scale);
        dst_v[i + 3] = Q6_Vhf_vmpy_VhfVhf(v3, v_scale);
      }
      for (; i < plane; ++i) {
        HVX_Vector v = src_v[i];
        dst_v[i] = Q6_Vhf_vmpy_VhfVhf(v, v_scale);
      }
    }
  }
}

static void htp_ops_scale_worker(void* data, int worker_index) {
  (void)worker_index;
  ScaleTask* task = (ScaleTask*)data;
  htp_ops_scale_c_range(task);
  worker_pool_synctoken_jobdone(task->sync);
}

AEEResult htp_ops_scale(uint8_t* dst, uint8_t* src,
                        uint8_t* scaleBias,
                        int32_t plane, int32_t cPack, int32_t hasBias) {
  if (plane <= 0 || cPack <= 0 || dst == NULL || src == NULL || scaleBias == NULL) return 0;
  int pack = 4;
#ifdef __HVX_LENGTH__
  pack = __HVX_LENGTH__ / (int32_t)sizeof(int16_t);
#endif

  const uint8_t* scaleBase = scaleBias;
  const uint8_t* biasBase = scaleBias + (size_t)cPack * pack * sizeof(int16_t);
  int task_count = 1;
  if (cPack >= 4 && plane * cPack >= 512 && g_max_num_workers > 1) {
    task_count = (int)g_max_num_workers;
    if (task_count > cPack) {
      task_count = cPack;
    }
    if (task_count > 4) {
      task_count = 4;
    }
  }

  if (task_count <= 1) {
    worker_synctoken_t unused_sync;
    ScaleTask task = {dst, src, scaleBase, biasBase, plane, pack, 0, cPack, hasBias != 0, &unused_sync};
    htp_ops_scale_c_range(&task);
    return 0;
  }

  ScaleTask* tasks = WORKER_POOL_STACK_ALLOC(ScaleTask, task_count);
  worker_synctoken_t sync;
  worker_pool_job_t job;
  job.fptr = htp_ops_scale_worker;
  worker_pool_synctoken_init(&sync, task_count);
  const int c_per_task = (cPack + task_count - 1) / task_count;
  for (int i = 0; i < task_count; ++i) {
    int c_begin = i * c_per_task;
    int c_end = c_begin + c_per_task;
    if (c_end > cPack) {
      c_end = cPack;
    }
    tasks[i].dst = dst;
    tasks[i].src = src;
    tasks[i].scale = scaleBase;
    tasks[i].bias = biasBase;
    tasks[i].plane = plane;
    tasks[i].pack = pack;
    tasks[i].c_begin = c_begin;
    tasks[i].c_end = c_end;
    tasks[i].has_bias = hasBias != 0;
    tasks[i].sync = &sync;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&sync);

  return 0;
}

}  // extern "C"
