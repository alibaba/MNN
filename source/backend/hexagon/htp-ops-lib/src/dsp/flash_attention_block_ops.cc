#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "attention_private.hpp"
#include "dsp/worker_pool.h"

extern "C" {

typedef struct {
  __fp16* m_new;
  __fp16* l_new;
  __fp16* out_new;
  const __fp16* q;
  const __fp16* k;
  const __fp16* v;
  const __fp16* running_m;
  const __fp16* running_l;
  const __fp16* running_out;
  const __fp16* packed_k;
  const __fp16* packed_v;
  uint8_t* workspace;
  int32_t batch;
  int32_t heads;
  int32_t tokens;
  int32_t chunk;
  int32_t chunk_padded;
  int32_t head_dim;
  int32_t k_icP;
  int32_t v_ocP;
  size_t packed_k_batch_elems;
  size_t packed_v_batch_elems;
  size_t workspace_head_bytes;
  float scale;
  worker_synctoken_t sync_ctx;
} FlashAttentionBlockState;

typedef struct {
  FlashAttentionBlockState* state;
  int32_t begin;
  int32_t end;
} FlashAttentionBlockTask;

static inline int32_t flash_attention_block_align_up(int32_t value, int32_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

static inline size_t flash_attention_block_align_up_size(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

static inline size_t flash_attention_block_hmx_workspace_head_bytes() {
  size_t offset = 0;
  offset = flash_attention_block_align_up_size(offset + 256 * 32 * sizeof(float), 128);
  offset = flash_attention_block_align_up_size(offset + 256 * 32 * sizeof(__fp16), 128);
  offset = flash_attention_block_align_up_size(offset + 256 * 64 * sizeof(float), 128);
  offset = flash_attention_block_align_up_size(offset + 256 * sizeof(float), 128);
  offset = flash_attention_block_align_up_size(offset + 256 * sizeof(float), 128);
  return offset;
}

static inline size_t flash_attention_block_packed_v_batch_elems(int32_t heads, int32_t chunk_padded,
                                                                  int32_t v_ocP) {
  const int32_t seq_blocks = (chunk_padded + ATTN_HMX_KV_BLOCK - 1) / ATTN_HMX_KV_BLOCK;
  return (size_t)seq_blocks * heads * v_ocP * ATTN_HMX_KV_BLOCK_TILES * 1024;
}

static inline size_t flash_attention_block_packed_k_batch_elems(int32_t heads, int32_t chunk_padded,
                                                                  int32_t k_icP) {
  const int32_t seq_blocks = (chunk_padded + ATTN_HMX_KV_BLOCK - 1) / ATTN_HMX_KV_BLOCK;
  return (size_t)seq_blocks * heads * ATTN_HMX_KV_BLOCK_TILES * k_icP * 1024;
}

static void flash_attention_block_pack_k(const FlashAttentionBlockState* s, __fp16* packed_k) {
  for (int32_t b = 0; b < s->batch; ++b) {
    __fp16* batch_dst = packed_k + (size_t)b * s->packed_k_batch_elems;
    const __fp16* batch_src = s->k + (size_t)b * s->chunk * s->heads * s->head_dim;
    for (int32_t t = 0; t < s->chunk; ++t) {
      const int32_t seq_tile = t / 32;
      const int32_t seq_inner = t % 32;
      for (int32_t h = 0; h < s->heads; ++h) {
        const __fp16* k_src = batch_src + ((size_t)t * s->heads + h) * s->head_dim;
        __fp16* k_dst_row = batch_dst + attn_hmx_k_tile_index(seq_tile, h, 0, s->heads, s->k_icP) * 1024;
        for (int32_t dim_tile = 0; dim_tile < s->k_icP; ++dim_tile) {
          __fp16* k_dst = k_dst_row + (size_t)dim_tile * 1024 + seq_inner * 2;
          const int32_t dim_base = dim_tile * 32;
          int32_t valid = s->head_dim - dim_base;
          if (valid > 32) {
            valid = 32;
          }
          for (int32_t dim = 0; dim + 1 < valid; dim += 2) {
            memcpy(k_dst + (dim / 2) * 64, k_src + dim_base + dim, sizeof(uint32_t));
          }
          if ((valid & 1) != 0) {
            k_dst[((valid - 1) / 2) * 64] = k_src[dim_base + valid - 1];
          }
        }
      }
    }
  }
}

static void flash_attention_block_pack_v(const FlashAttentionBlockState* s, __fp16* packed_v) {
  for (int32_t b = 0; b < s->batch; ++b) {
    __fp16* batch_dst = packed_v + (size_t)b * s->packed_v_batch_elems;
    const __fp16* batch_src = s->v + (size_t)b * s->chunk * s->heads * s->head_dim;
    for (int32_t t = 0; t < s->chunk; ++t) {
      const int32_t seq_tile = t / 32;
      const int32_t seq_inner = t % 32;
      const int32_t seq_pair = seq_inner / 2;
      const int32_t seq_lane = seq_inner % 2;
      for (int32_t h = 0; h < s->heads; ++h) {
        const __fp16* v_src = batch_src + ((size_t)t * s->heads + h) * s->head_dim;
        for (int32_t dim_tile = 0; dim_tile < s->v_ocP; ++dim_tile) {
          __fp16* v_dst = batch_dst + attn_hmx_v_tile_index(dim_tile, seq_tile, h, s->heads, s->v_ocP) * 1024 +
                          seq_pair * 64 + seq_lane;
          const int32_t dim_base = dim_tile * 32;
          int32_t valid = s->head_dim - dim_base;
          if (valid > 32) {
            valid = 32;
          }
          for (int32_t dim = 0; dim < valid; ++dim) {
            v_dst[dim * 2] = v_src[dim_base + dim];
          }
        }
      }
    }
  }
}

static inline void flash_attention_block_head(const FlashAttentionBlockState* s, int32_t bh) {
  const int32_t b = bh / s->heads;
  const int32_t h = bh - b * s->heads;
  const int32_t tokens = s->tokens;
  const int32_t chunk = s->chunk;
  const int32_t head_dim = s->head_dim;

  const __fp16* q_head = s->q + ((int64_t)b * s->heads + h) * tokens * head_dim;
  const __fp16* k_base = s->k + ((int64_t)b * chunk * s->heads + h) * head_dim;
  const __fp16* v_base = s->v + ((int64_t)b * chunk * s->heads + h) * head_dim;
  const __fp16* old_m = s->running_m + ((int64_t)b * s->heads + h) * tokens;
  const __fp16* old_l = s->running_l + ((int64_t)b * s->heads + h) * tokens;
  const __fp16* old_out = s->running_out + ((int64_t)b * s->heads + h) * tokens * head_dim;
  __fp16* dst_m = s->m_new + ((int64_t)b * s->heads + h) * tokens;
  __fp16* dst_l = s->l_new + ((int64_t)b * s->heads + h) * tokens;
  __fp16* dst_out = s->out_new + ((int64_t)b * s->heads + h) * tokens * head_dim;

  for (int32_t token = 0; token < tokens; ++token) {
    const __fp16* q_row = q_head + (int64_t)token * head_dim;
    float max_score = -INFINITY;
    for (int32_t t = 0; t < chunk; ++t) {
      const __fp16* k_row = k_base + (int64_t)t * s->heads * head_dim;
      float score = 0.0f;
      for (int32_t d = 0; d < head_dim; ++d) {
        score += (float)q_row[d] * (float)k_row[d];
      }
      score *= s->scale;
      if (score > max_score) {
        max_score = score;
      }
    }

    const float old_m_value = (float)old_m[token];
    const float old_l_value = (float)old_l[token];
    const float new_m_value = old_m_value > max_score ? old_m_value : max_score;
    const float old_scale = old_l_value > 0.0f ? expf(old_m_value - new_m_value) : 0.0f;
    float new_l_value = old_l_value * old_scale;

    for (int32_t d = 0; d < head_dim; ++d) {
      dst_out[(int64_t)token * head_dim + d] = (__fp16)((float)old_out[(int64_t)token * head_dim + d] * old_scale);
    }

    for (int32_t t = 0; t < chunk; ++t) {
      const __fp16* k_row = k_base + (int64_t)t * s->heads * head_dim;
      const __fp16* v_row = v_base + (int64_t)t * s->heads * head_dim;
      float score = 0.0f;
      for (int32_t d = 0; d < head_dim; ++d) {
        score += (float)q_row[d] * (float)k_row[d];
      }
      const float weight = expf(score * s->scale - new_m_value);
      new_l_value += weight;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int64_t offset = (int64_t)token * head_dim + d;
        dst_out[offset] = (__fp16)((float)dst_out[offset] + weight * (float)v_row[d]);
      }
    }

    dst_m[token] = (__fp16)new_m_value;
    dst_l[token] = (__fp16)new_l_value;
  }
}

static inline void flash_attention_block_head_cached_qk(const FlashAttentionBlockState* s, int32_t bh) {
  const int32_t b = bh / s->heads;
  const int32_t h = bh - b * s->heads;
  const int32_t tokens = s->tokens;
  const int32_t chunk = s->chunk;
  const int32_t head_dim = s->head_dim;

  const __fp16* q_head = s->q + ((int64_t)b * s->heads + h) * tokens * head_dim;
  const __fp16* k_base = s->k + ((int64_t)b * chunk * s->heads + h) * head_dim;
  const __fp16* v_base = s->v + ((int64_t)b * chunk * s->heads + h) * head_dim;
  const __fp16* old_m = s->running_m + ((int64_t)b * s->heads + h) * tokens;
  const __fp16* old_l = s->running_l + ((int64_t)b * s->heads + h) * tokens;
  const __fp16* old_out = s->running_out + ((int64_t)b * s->heads + h) * tokens * head_dim;
  __fp16* dst_m = s->m_new + ((int64_t)b * s->heads + h) * tokens;
  __fp16* dst_l = s->l_new + ((int64_t)b * s->heads + h) * tokens;
  __fp16* dst_out = s->out_new + ((int64_t)b * s->heads + h) * tokens * head_dim;
  float row_scores[256];

  for (int32_t token = 0; token < tokens; ++token) {
    const __fp16* q_row = q_head + (int64_t)token * head_dim;
    float max_score = -INFINITY;
    for (int32_t t = 0; t < chunk; ++t) {
      const __fp16* k_row = k_base + (int64_t)t * s->heads * head_dim;
      float score = 0.0f;
      for (int32_t d = 0; d < head_dim; ++d) {
        score += (float)q_row[d] * (float)k_row[d];
      }
      score *= s->scale;
      row_scores[t] = score;
      if (score > max_score) {
        max_score = score;
      }
    }

    const float old_m_value = (float)old_m[token];
    const float old_l_value = (float)old_l[token];
    const float new_m_value = old_m_value > max_score ? old_m_value : max_score;
    const float old_scale = old_l_value > 0.0f ? expf(old_m_value - new_m_value) : 0.0f;
    float new_l_value = old_l_value * old_scale;

    for (int32_t d = 0; d < head_dim; ++d) {
      dst_out[(int64_t)token * head_dim + d] = (__fp16)((float)old_out[(int64_t)token * head_dim + d] * old_scale);
    }

    for (int32_t t = 0; t < chunk; ++t) {
      const __fp16* v_row = v_base + (int64_t)t * s->heads * head_dim;
      const float weight = expf(row_scores[t] - new_m_value);
      new_l_value += weight;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int64_t offset = (int64_t)token * head_dim + d;
        dst_out[offset] = (__fp16)((float)dst_out[offset] + weight * (float)v_row[d]);
      }
    }

    dst_m[token] = (__fp16)new_m_value;
    dst_l[token] = (__fp16)new_l_value;
  }
}

static inline void flash_attention_block_combine_out64(__fp16* dst, const __fp16* old_out, const float* temp_out,
                                                         float old_scale) {
  HVX_Vector v_old = vmemu(old_out);
  HVX_VectorPair old_fp32 = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(v_old));
  HVX_Vector v_scale = Q6_V_vsplat_R(*(int*)&old_scale);
  HVX_Vector v_temp0 = vmemu(temp_out);
  HVX_Vector v_temp1 = vmemu(temp_out + 32);
  HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(old_fp32), v_scale)), v_temp0));
  HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(old_fp32), v_scale)), v_temp1));
  vmemu(dst) = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(v_out0, v_out1));
}

static inline void flash_attention_block_hmx_matmul_ex(uint8_t* c, const uint8_t* a, const uint8_t* b,
                                                         int M, int K, int N, int max_K, int a_stride,
                                                         int output_layout_type, float output_scale,
                                                         int weight_layout_type, int kv_head, int n_kv_heads,
                                                         int output_stride, int output_row_offset) {
  sync_attention_matmul_lock_acquire();
  hmx_manager_enable_execution();
  attn_hmx_matmul(c, a, b, M, K, N, max_K, a_stride, output_layout_type, output_scale,
                  weight_layout_type, kv_head, n_kv_heads, output_stride, output_row_offset);
  hmx_manager_disable_execution();
  sync_attention_matmul_lock_release();
}

static inline void flash_attention_block_old_scales(float* old_scales, const __fp16* old_m, const __fp16* old_l,
                                                      const float* new_ms, int32_t row_begin, int32_t rows) {
  const float log2e = 1.4426950408889634f;
  HVX_Vector v_log2e = Q6_V_vsplat_R(*(const int*)&log2e);
  _Alignas(128) float delta[32];
  for (int32_t r = 0; r < rows; r += 32) {
    int32_t count = rows - r;
    if (count > 32) {
      count = 32;
    }
    for (int32_t i = 0; i < count; ++i) {
      const int32_t token = row_begin + r + i;
      delta[i] = ((float)old_l[token] > 0.0f) ? ((float)old_m[token] - new_ms[r + i]) : -INFINITY;
    }
    HVX_Vector v_delta = vmem(delta);
    v_delta = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_delta, v_log2e));
    HVX_Vector v_scale = hvx_my_exp2_vsf(v_delta);
    if (count < 32) {
      HVX_VectorPred q_tail = Q6_Q_vsetq_R(count * (int)sizeof(float));
      v_scale = Q6_V_vmux_QVV(q_tail, v_scale, Q6_V_vzero());
    }
    vmemu(old_scales + r) = v_scale;
  }
}

static inline void flash_attention_block_head_hmx_qk_sv(const FlashAttentionBlockState* s, int32_t bh) {
  const int32_t b = bh / s->heads;
  const int32_t h = bh - b * s->heads;
  const int32_t tokens = s->tokens;
  const int32_t chunk = s->chunk;
  const int32_t head_dim = s->head_dim;
  const int32_t block_rows = 256;

  const __fp16* q_head = s->q + ((int64_t)b * s->heads + h) * tokens * head_dim;
  const __fp16* old_m = s->running_m + ((int64_t)b * s->heads + h) * tokens;
  const __fp16* old_l = s->running_l + ((int64_t)b * s->heads + h) * tokens;
  const __fp16* old_out = s->running_out + ((int64_t)b * s->heads + h) * tokens * head_dim;
  __fp16* dst_m = s->m_new + ((int64_t)b * s->heads + h) * tokens;
  __fp16* dst_l = s->l_new + ((int64_t)b * s->heads + h) * tokens;
  __fp16* dst_out = s->out_new + ((int64_t)b * s->heads + h) * tokens * head_dim;
  const __fp16* packed_k = s->packed_k + (size_t)b * s->packed_k_batch_elems;
  const __fp16* packed_v = s->packed_v + (size_t)b * s->packed_v_batch_elems;

  uint8_t* workspace = s->workspace + (size_t)bh * s->workspace_head_bytes;
  size_t workspace_offset = 0;
  float* scores = (float*)(workspace + workspace_offset);
  workspace_offset = flash_attention_block_align_up_size(workspace_offset + 256 * 32 * sizeof(float), 128);
  __fp16* linear_s = (__fp16*)(workspace + workspace_offset);
  workspace_offset = flash_attention_block_align_up_size(workspace_offset + 256 * 32 * sizeof(__fp16), 128);
  float* temp_out = (float*)(workspace + workspace_offset);
  workspace_offset = flash_attention_block_align_up_size(workspace_offset + 256 * 64 * sizeof(float), 128);
  float* old_scales = (float*)(workspace + workspace_offset);
  workspace_offset = flash_attention_block_align_up_size(workspace_offset + 256 * sizeof(float), 128);
  float* new_ms = (float*)(workspace + workspace_offset);

  for (int32_t row_begin = 0; row_begin < tokens; row_begin += block_rows) {
    int32_t rows = tokens - row_begin;
    if (rows > block_rows) {
      rows = block_rows;
    }
    memset(linear_s, 0, (size_t)rows * 32 * sizeof(__fp16));

    flash_attention_block_hmx_matmul_ex((uint8_t*)scores, (const uint8_t*)(q_head + (size_t)row_begin * head_dim),
                                          (const uint8_t*)packed_k, rows, head_dim, s->chunk_padded, head_dim,
                                          head_dim, ATTN_HMX_OUT_LINEAR_FP32_SCALED, s->scale,
                                          ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, h, s->heads, 32, 0);

    for (int32_t r = 0; r < rows; ++r) {
      const int32_t token = row_begin + r;
      float* row_scores = scores + (size_t)r * 32;
      const float max_score = sync_attention_max_f32(row_scores, chunk);

      const float old_m_value = (float)old_m[token];
      const float new_m_value = old_m_value > max_score ? old_m_value : max_score;
      new_ms[r] = new_m_value;
      dst_m[token] = (__fp16)new_m_value;
    }

    flash_attention_block_old_scales(old_scales, old_m, old_l, new_ms, row_begin, rows);

    for (int32_t r = 0; r < rows; ++r) {
      const int32_t token = row_begin + r;
      float* row_scores = scores + (size_t)r * 32;
      const float old_l_value = (float)old_l[token];
      __fp16* row_s = linear_s + (size_t)r * 32;
      const float exp_sum = sync_attention_exp_and_sum(row_scores, chunk, new_ms[r]);
      sync_attention_normalize_to_fp16(row_s, row_scores, chunk, 1.0f);

      dst_l[token] = (__fp16)(old_l_value * old_scales[r] + exp_sum);
    }

    flash_attention_block_hmx_matmul_ex((uint8_t*)temp_out, (const uint8_t*)linear_s, (const uint8_t*)packed_v,
                                          rows, chunk, head_dim, s->chunk_padded, 32,
                                          ATTN_HMX_OUT_LINEAR_FP32_SCALED, 1.0f,
                                          ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, h, s->heads, head_dim, 0);

    for (int32_t r = 0; r < rows; ++r) {
      const int32_t token = row_begin + r;
      const int64_t offset = (int64_t)token * head_dim;
      flash_attention_block_combine_out64(dst_out + offset, old_out + offset,
                                            temp_out + (size_t)r * head_dim, old_scales[r]);
    }
  }
}

static void flash_attention_block_worker(void* data, int worker_index) {
  (void)worker_index;
  FlashAttentionBlockTask* task = (FlashAttentionBlockTask*)data;
  for (int32_t i = task->begin; i < task->end; ++i) {
    if (task->state->packed_k != NULL && task->state->packed_v != NULL && task->state->workspace != NULL &&
        task->state->chunk_padded <= 32 && task->state->head_dim == 64) {
      flash_attention_block_head_hmx_qk_sv(task->state, i);
    } else if (task->state->chunk <= 256) {
      flash_attention_block_head_cached_qk(task->state, i);
    } else {
      flash_attention_block_head(task->state, i);
    }
  }
  worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

static int flash_attention_block_pick_tasks(int total_heads) {
  int tasks = (int)g_max_num_workers;
  if (tasks <= 1 || total_heads <= 1) {
    return 1;
  }
  if (tasks > total_heads) {
    tasks = total_heads;
  }
  return tasks;
}

AEEResult htp_ops_flash_attention_block(uint8_t* m_new, uint8_t* l_new, uint8_t* out_new,
                                          uint8_t* q, uint8_t* k, uint8_t* v, uint8_t* mask,
                                          uint8_t* running_m, uint8_t* running_l, uint8_t* running_out,
                                          uint8_t* external_workspace,
                                          int32_t batch, int32_t heads, int32_t tokens, int32_t chunk,
                                          int32_t head_dim, float scale) {
  (void)mask;
  if (m_new == NULL || l_new == NULL || out_new == NULL || q == NULL || k == NULL || v == NULL ||
      running_m == NULL || running_l == NULL || running_out == NULL ||
      batch <= 0 || heads <= 0 || tokens <= 0 || chunk <= 0 || head_dim <= 0) {
    return AEE_EBADPARM;
  }

  FlashAttentionBlockState state = {};
  state.m_new = (__fp16*)m_new;
  state.l_new = (__fp16*)l_new;
  state.out_new = (__fp16*)out_new;
  state.q = (const __fp16*)q;
  state.k = (const __fp16*)k;
  state.v = (const __fp16*)v;
  state.running_m = (const __fp16*)running_m;
  state.running_l = (const __fp16*)running_l;
  state.running_out = (const __fp16*)running_out;
  state.batch = batch;
  state.heads = heads;
  state.tokens = tokens;
  state.chunk = chunk;
  state.chunk_padded = flash_attention_block_align_up(chunk, 32);
  state.head_dim = head_dim;
  state.k_icP = flash_attention_block_align_up(head_dim, 32) / 32;
  state.v_ocP = flash_attention_block_align_up(head_dim, 32) / 32;
  state.packed_k_batch_elems =
      flash_attention_block_packed_k_batch_elems(heads, state.chunk_padded, state.k_icP);
  state.packed_v_batch_elems =
      flash_attention_block_packed_v_batch_elems(heads, state.chunk_padded, state.v_ocP);
  state.workspace_head_bytes = flash_attention_block_hmx_workspace_head_bytes();
  state.scale = scale;

  if (state.chunk_padded <= 32 && head_dim == 64 && external_workspace != NULL) {
    uint8_t* cursor = external_workspace;
    state.packed_k = (__fp16*)cursor;
    cursor += flash_attention_block_align_up_size((size_t)batch * state.packed_k_batch_elems * sizeof(__fp16), 128);
    state.packed_v = (__fp16*)cursor;
    cursor += flash_attention_block_align_up_size((size_t)batch * state.packed_v_batch_elems * sizeof(__fp16), 128);
    state.workspace = cursor;
    flash_attention_block_pack_k(&state, (__fp16*)state.packed_k);
    flash_attention_block_pack_v(&state, (__fp16*)state.packed_v);
  }

  const int32_t total_heads = batch * heads;
  int tasks = flash_attention_block_pick_tasks(total_heads);
  if (tasks <= 1) {
    for (int32_t i = 0; i < total_heads; ++i) {
      if (state.packed_k != NULL && state.packed_v != NULL && state.workspace != NULL &&
          state.chunk_padded <= 32 && state.head_dim == 64) {
        flash_attention_block_head_hmx_qk_sv(&state, i);
      } else if (state.chunk <= 256) {
        flash_attention_block_head_cached_qk(&state, i);
      } else {
        flash_attention_block_head(&state, i);
      }
    }
    return AEE_SUCCESS;
  }

  FlashAttentionBlockTask task_storage[16];
  worker_pool_job_t jobs[16];
  int task_count = tasks;
  if (task_count > 16) {
    task_count = 16;
  }
  worker_pool_synctoken_init(&state.sync_ctx, task_count);
  for (int i = 0; i < task_count; ++i) {
    const int32_t begin = (int64_t)total_heads * i / task_count;
    const int32_t end = (int64_t)total_heads * (i + 1) / task_count;
    task_storage[i].state = &state;
    task_storage[i].begin = begin;
    task_storage[i].end = end;
    jobs[i].fptr = flash_attention_block_worker;
    jobs[i].dptr = &task_storage[i];
    worker_pool_submit(NULL, jobs[i]);
  }
  worker_pool_synctoken_wait(&state.sync_ctx);
  return AEE_SUCCESS;
}

}  // extern "C"
