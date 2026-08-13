#include "attention_private.hpp"

// Worker-side phase profiling for the grouped-causal prefill path, mirroring the queue-thread timers in
// attention_hmx.cc. Per-worker slots => no atomics needed. Surfaced into profile[251..255] by
// execute_command.cc. [0]=Q gather, [1]=QK submit+wait, [2]=softmax, [3]=SV submit+wait, [4]=O scatter.
#ifndef HTP_WATTN_PHASE_PROFILE
#  define HTP_WATTN_PHASE_PROFILE 0
#endif
#if HTP_WATTN_PHASE_PROFILE
#  define HTP_WATTN_MAX_WORKERS 32
unsigned long long g_wattn_phase_us[HTP_WATTN_MAX_WORKERS][5] = {};
#  define WATTN_T0() (HAP_perf_get_time_us())
#  define WATTN_ADD(w, i, t0) \
    (g_wattn_phase_us[(w) & (HTP_WATTN_MAX_WORKERS - 1)][(i)] += HAP_perf_get_time_us() - (t0))
#else
#  define WATTN_T0()          (0ULL)
#  define WATTN_ADD(w, i, t0) ((void) (t0))
#endif

static inline void sync_attention_add_mask_fp16(float* row_scores, const __fp16* mask_ptr, int len) {
  for (int i = 0; i < len; ++i) {
    row_scores[i] += (float)mask_ptr[i];
  }
}

static inline int sync_attention_clamp_valid_end(const SyncAttentionTaskState* state, int valid_end) {
  if (valid_end > state->N) {
    valid_end = state->N;
  }
  if (valid_end < 0) {
    valid_end = 0;
  }
  return valid_end;
}

static inline int sync_attention_causal_valid_end(const SyncAttentionTaskState* state, int q_end) {
  return sync_attention_clamp_valid_end(state, state->seq_current + q_end);
}

static inline int sync_attention_mask_start_pos(const SyncAttentionTaskState* state) {
  int maskStartPos = state->N - state->mask_stride;
  return maskStartPos < 0 ? 0 : maskStartPos;
}

static inline void sync_attention_softmax_row(__fp16* row_s, float* row_scores, int validN) {
  float max_value = sync_attention_max_f32(row_scores, validN);
  float sum_exp = sync_attention_exp_and_sum(row_scores, validN, max_value);
  float inv_sum = 1.0f / sum_exp;
  sync_attention_normalize_to_fp16(row_s, row_scores, validN, inv_sum);
}

// ATTN_SCORES_FP16: fused softmax over an fp16 scores row. Reads fp16, promotes to fp32 in registers for
// max/exp/sum (no fp32 temp), writes the exp result as fp16 into row_s, then normalizes row_s in place.
// Half the read traffic of the fp32 path; the math stays fp32 for numerical stability.
//
// The row is staged into an L1-resident scratch on the max pass, so the DDR traffic is one read of the
// score row plus one write of linear_S. Before: scores were read twice (max, exp) and linear_S was
// written, read back and rewritten (normalize) -- 10 bytes of DDR per element, now 4. The arithmetic and
// its ordering are unchanged, so results stay bit-identical.
static inline void sync_attention_softmax_row_fp16(__fp16 *row_s, const __fp16 *row_scores16, int validN) {
  __fp16 stage[ATTN_FIXED_WORKSPACE_KV] __attribute__((aligned(128)));
  if (validN > ATTN_FIXED_WORKSPACE_KV) {  // guaranteed by sync_attention_can_group_causal; be defensive
    FARF(ERROR, "softmax_row_fp16 validN overflow: %d", validN);
    return;
  }
  float      neg_inf = -INFINITY;
  HVX_Vector v_max   = Q6_V_vsplat_R(*(int *) &neg_inf);
  int        i       = 0;
  for (; i < validN; i += 64) {
    HVX_Vector raw              = vmemu(row_scores16 + i);
    *(HVX_Vector *) (stage + i) = raw;  // i is a multiple of 64 fp16 => 128B aligned
    HVX_VectorPair sf           = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(raw));
    HVX_Vector     lo  = Q6_V_lo_W(sf);
    HVX_Vector     hi  = Q6_V_hi_W(sf);
    int            rem = validN - i;
    if (rem < 64) {
      lo = Q6_V_vmux_QVV(Q6_Q_vsetq_R((rem < 32 ? rem : 32) * (int) sizeof(float)), lo, v_max);
      hi = (rem > 32) ? Q6_V_vmux_QVV(Q6_Q_vsetq_R((rem - 32) * (int) sizeof(float)), hi, v_max) : v_max;
    }
    v_max = Q6_Vsf_vmax_VsfVsf(v_max, lo);
    v_max = Q6_Vsf_vmax_VsfVsf(v_max, hi);
  }
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 64));
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 32));
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 16));
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 8));
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 4));
  float max_arr[32] __attribute__((aligned(128)));
  *(HVX_Vector *) max_arr = v_max;
  float max_value         = max_arr[0];

  const float log2e   = 1.4426950408889634f;
  HVX_Vector  v_log2e = Q6_V_vsplat_R(*(const int *) &log2e);
  HVX_Vector  v_maxs  = Q6_V_vsplat_R(*(const int *) &max_value);
  HVX_Vector  v_sum   = Q6_V_vzero();
  for (i = 0; i < validN; i += 64) {
    HVX_VectorPair sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(*(const HVX_Vector *) (stage + i)));
    HVX_Vector     s0 = Q6_Vsf_equals_Vqf32(
      Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(sf), v_maxs)), v_log2e));
    HVX_Vector s1 = Q6_Vsf_equals_Vqf32(
      Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(sf), v_maxs)), v_log2e));
    HVX_Vector e0  = hvx_my_exp2_vsf(s0);
    HVX_Vector e1  = hvx_my_exp2_vsf(s1);
    int        rem = validN - i;
    if (rem < 64) {
      e0 = Q6_V_vmux_QVV(Q6_Q_vsetq_R((rem < 32 ? rem : 32) * (int) sizeof(float)), e0, Q6_V_vzero());
      e1 = (rem > 32) ? Q6_V_vmux_QVV(Q6_Q_vsetq_R((rem - 32) * (int) sizeof(float)), e1, Q6_V_vzero()) : Q6_V_vzero();
    }
    v_sum              = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, e0));
    v_sum              = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, e1));
    // Straight into the stage: lanes past validN were already zeroed above and the stage is scratch,
    // so a full aligned vector store is fine (no tail handling needed here any more).
    *(HVX_Vector *) (stage + i) = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(e0, e1));
  }
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 64)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 32)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 16)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 8)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 4)));
  float sum_arr[32] __attribute__((aligned(128)));
  *(HVX_Vector *) sum_arr = v_sum;
  float inv_sum           = sum_arr[0] > 0.0f ? 1.0f / sum_arr[0] : 0.0f;

  uint16_t   inv_bits = hmx_fp16_bits(inv_sum);
  HVX_Vector v_inv    = Q6_V_vsplat_R(((unsigned) inv_bits) | (((unsigned) inv_bits) << 16));
  for (int j = 0; j < validN; j += 64) {
    HVX_Vector r   = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(*(const HVX_Vector *) (stage + j), v_inv));
    int        rem = validN - j;
    if (rem >= 64) {
      vmemu(row_s + j) = r;
    } else {
      sync_attention_store_fp16_tail(row_s + j, r, rem);
    }
  }
}

static inline int sync_attention_prepare_online_row(const SyncAttentionTaskState* state, float* row_scores,
                                                    __fp16* row_s, int row, int page_start, int page_valid,
                                                    int global_q, float* row_max, float* row_sum,
                                                    float* row_scale) {
  int valid = page_valid;
  if (state->mask_stride < 0) {
    int validN = sync_attention_causal_valid_end(state, global_q + 1);
    valid = validN - page_start;
    if (valid > page_valid) valid = page_valid;
  } else if (state->mask != NULL && state->mask_stride > 0) {
    int maskStartPos = sync_attention_mask_start_pos(state);
    int overlap_start = maskStartPos - page_start;
    if (overlap_start < 0) overlap_start = 0;
    if (overlap_start < valid) {
      int mask_offset = page_start + overlap_start - maskStartPos;
      int overlap_len = valid - overlap_start;
      const __fp16* mask_h = (const __fp16*)state->mask + (size_t)global_q * state->mask_stride + mask_offset;
      sync_attention_add_mask_fp16(row_scores + overlap_start, mask_h, overlap_len);
    }
  }
  if (valid <= 0) {
    row_scale[row] = 1.0f;
    sync_attention_zero_fp16(row_s, page_valid);
    return 0;
  }

  float block_max = row_scores[0];
  for (int i = 1; i < valid; ++i) {
    if (row_scores[i] > block_max) block_max = row_scores[i];
  }
  float old_max = row_max[row];
  float old_sum = row_sum[row];
  float new_max = old_sum > 0.0f ? (old_max > block_max ? old_max : block_max) : block_max;
  float old_scale = old_sum > 0.0f ? expf(old_max - new_max) : 0.0f;
  row_scale[row] = old_scale;
  for (int i = 0; i < valid; ++i) {
    row_scores[i] -= new_max;
  }
  float block_sum = sync_attention_exp_and_sum(row_scores, valid, 0.0f);
  sync_attention_normalize_to_fp16(row_s, row_scores, valid, 1.0f);
  if (valid < page_valid) {
    sync_attention_zero_fp16(row_s + valid, page_valid - valid);
  }
  row_max[row] = new_max;
  row_sum[row] = old_sum * old_scale + block_sum;
  return 1;
}

static inline void sync_attention_scale_packed_rows(__fp16* dst, int rows, int head_dim, const float* row_scale) {
  int packs = head_dim / 64;
  for (int p = 0; p < packs; ++p) {
    __fp16* base = dst + (size_t)p * rows * 64;
    for (int r = 0; r < rows; ++r) {
      float scale = row_scale[r];
      __fp16* d = base + (size_t)r * 64;
      for (int i = 0; i < 64; ++i) {
        d[i] = (__fp16)((float)d[i] * scale);
      }
    }
  }
}

static inline void sync_attention_copy_scaled_packed_output(__fp16* dst, const __fp16* src, int rows,
                                                            int output_stride, int row_offset, int head_dim,
                                                            const float* row_sum) {
  int packs = head_dim / 64;
  for (int p = 0; p < packs; ++p) {
    const __fp16* s_base = src + (size_t)p * rows * 64;
    __fp16* d_base = dst + (size_t)p * output_stride * 64 + row_offset * 64;
    for (int r = 0; r < rows; ++r) {
      float inv = row_sum[r] > 0.0f ? 1.0f / row_sum[r] : 0.0f;
      const __fp16* s = s_base + (size_t)r * 64;
      __fp16* d = d_base + (size_t)r * 64;
      for (int i = 0; i < 64; ++i) {
        d[i] = (__fp16)((float)s[i] * inv);
      }
    }
  }
}

static inline void sync_attention_copy_decode_group_output(__fp16* O, const __fp16* packed_O, int head_base,
                                                           int group_heads, int head_dim, const float* row_sum) {
  for (int h = 0; h < group_heads; ++h) {
    __fp16* dst = O + (size_t)(head_base + h) * head_dim;
    for (int pack = 0; pack < head_dim / 64; ++pack) {
      const __fp16* src = packed_O + (size_t)(pack * group_heads + h) * 64;
      if (row_sum == NULL) {
        memcpy(dst + pack * 64, src, 64 * sizeof(__fp16));
      } else {
        float scale = row_sum[h] > 0.0f ? 1.0f / row_sum[h] : 0.0f;
        for (int i = 0; i < 64; ++i) {
          dst[pack * 64 + i] = (__fp16)((float)src[i] * scale);
        }
      }
    }
  }
}

static void sync_attention_process_online_pages(const SyncAttentionTaskState* state, int task_id, int worker_index,
                                                int decode_grouped) {
  const int rows = decode_grouped ? state->gqa_factor : state->qo_len;
  const int kv_head = decode_grouped ? task_id : task_id / state->gqa_factor;
  const int head_base = decode_grouped ? task_id * state->gqa_factor : task_id;
  const int q_stride = decode_grouped ? state->head_dim : state->qo_stride;
  const __fp16* q_ptr = state->Q + (size_t)head_base * state->head_dim;
  uint8_t* worker_workspace = sync_attention_worker_workspace(state, worker_index);
  float* scores = NULL;
  __fp16* linear_S = NULL;
  __fp16* accum_O = NULL;
  __fp16* temp_O = NULL;
  __fp16* page_temp_O = NULL;
  float* row_max = NULL;
  float* row_sum = NULL;
  float* row_scale = NULL;
  sync_attention_page_block_offsets(state, worker_workspace, &scores, &linear_S, &accum_O, &temp_O, &row_max,
                                    &row_sum, &row_scale, rows);
  page_temp_O = temp_O;

  sync_attention_reset_online_rows(row_max, row_sum, row_scale, rows);
  sync_attention_zero_packed_output(accum_O, rows, rows, 0, state->K_dim_padded);

  for (int page = 0; page < state->page_count; page += state->online_block_pages) {
    int block_start = page * state->page_size;
    if (block_start >= state->N) break;
    int block_valid = state->N - block_start;
    if (block_valid > state->online_block_size) block_valid = state->online_block_size;
    if (block_valid <= 0) continue;
    int page_end = page + state->online_block_pages;
    if (page_end > state->page_count) page_end = state->page_count;

    sync_attention_run_page_qk_block(state, scores, q_ptr, rows, q_stride, kv_head, page, page_end,
                                     block_start, block_valid);

    int any_valid = 0;
    for (int r = 0; r < rows; ++r) {
      any_valid |= sync_attention_prepare_online_row(state,
                                                     scores + (size_t)r * state->online_block_size,
                                                     linear_S + (size_t)r * state->online_block_size,
                                                     r, block_start, block_valid, decode_grouped ? 0 : r,
                                                     row_max, row_sum, row_scale);
    }
    if (!any_valid) {
      continue;
    }

    sync_attention_run_page_sv_block(state, temp_O, page_temp_O, linear_S, rows, kv_head, page, page_end,
                                     block_start, block_valid);
    sync_attention_scale_packed_rows(accum_O, rows, state->K_dim_padded, row_scale);
    sync_attention_add_packed_rows(accum_O, temp_O, rows, state->K_dim_padded);
  }

  if (!decode_grouped) {
    __fp16* head_O = state->O + (size_t)task_id * (state->head_dim / 64) * state->qo_total_len * 64;
    sync_attention_copy_scaled_packed_output(head_O, accum_O, rows, state->qo_total_len, state->q_offset,
                                             state->K_dim_padded, row_sum);
    return;
  }
  sync_attention_copy_decode_group_output(state->O, accum_O, head_base, rows, state->head_dim, row_sum);
}

static inline int sync_attention_use_prefill64_blocks(const SyncAttentionTaskState* state) {
  return state->seq_current == 0 && state->qo_len == state->N && state->N <= 128;
}

static inline void sync_attention_clear_linear_block(__fp16* linear_S, const SyncAttentionTaskState* state,
                                                     int q_begin, int q_rows, int block_valid_end) {
  __fp16* block_s = linear_S + (size_t)q_begin * state->N_padded;
  if (block_valid_end == 64 && state->N_padded > 64) {
    HVX_Vector v_zero = Q6_V_vzero();
    for (int r = 0; r < q_rows; ++r) {
      vmemu(block_s + (size_t)r * state->N_padded) = v_zero;
    }
    return;
  }
  memset(block_s, 0, (size_t)q_rows * state->N_padded * sizeof(__fp16));
}

static inline void sync_attention_run_causal_qk_block(const SyncAttentionTaskState* state, float* scores,
                                                      int head_id, int h_kv, int q_begin, int q_rows,
                                                      int block_valid_end, int block_valid_end_padded) {
  const __fp16* q_ptr = state->Q + (size_t)q_begin * state->qo_stride + head_id * state->head_dim;
  if (state->page_count > 0) {
    sync_attention_run_page_qk(state, scores, q_ptr, q_rows, state->qo_stride, q_begin, h_kv, block_valid_end);
  } else {
    run_locked_attn_hmx_matmul_ex((uint8_t*)scores, (uint8_t*)q_ptr, (uint8_t*)state->pastK,
                                  q_rows, state->head_dim, block_valid_end_padded, state->K_dim_padded,
                                  state->qo_stride, ATTN_HMX_OUT_LINEAR_FP32_SCALED, state->scale,
                                  ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, h_kv, state->n_kv_heads,
                                  state->N_padded, q_begin);
  }
}

static inline void sync_attention_normalize_causal_block(const SyncAttentionTaskState* state, float* scores,
                                                         __fp16* linear_S, int q_begin, int q_rows,
                                                         int block_valid_end, int linear_precleared) {
  for (int q_i = q_begin; q_i < q_begin + q_rows; ++q_i) {
    float* row_scores = &scores[(size_t)q_i * state->N_padded];
    __fp16* row_s = linear_S + (size_t)q_i * state->N_padded;
    int validN = sync_attention_causal_valid_end(state, q_i + 1);
    sync_attention_softmax_row(row_s, row_scores, validN);
    if (!linear_precleared && validN < block_valid_end) {
      sync_attention_zero_fp16(row_s + validN, block_valid_end - validN);
    }
  }
}

static inline void sync_attention_run_causal_sv_block(const SyncAttentionTaskState* state, __fp16* head_O,
                                                      __fp16* temp_O, __fp16* linear_S, int h_kv, int q_begin,
                                                      int q_rows, int block_valid_end) {
  const __fp16* block_s = linear_S + (size_t)q_begin * state->N_padded;
  const int output_stride = state->qo_total_len;
  const int output_row_offset = state->q_offset + q_begin;
  if (state->page_count > 0) {
    sync_attention_run_page_sv(state, head_O, temp_O, block_s, q_rows, output_stride, output_row_offset, h_kv,
                               block_valid_end);
  } else {
    run_locked_attn_hmx_matmul_ex((uint8_t*)head_O, (uint8_t*)block_s, (uint8_t*)state->pastV,
                                  q_rows, block_valid_end, state->K_dim_padded, state->N_padded,
                                  state->N_padded, ATTN_HMX_OUT_PACKED_FP16, 1.0f,
                                  ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, h_kv, state->n_kv_heads,
                                  output_stride, output_row_offset);
  }
}

static void sync_attention_process_head(const SyncAttentionTaskState* state, int head_id, int worker_index) {
  const int h_kv = head_id / state->gqa_factor;
  if (sync_attention_try_page_causal_len2(state, head_id, worker_index)) {
    return;
  }
  if (state->online_pages) {
    sync_attention_process_online_pages(state, head_id, worker_index, 0);
    return;
  }

  uint8_t* worker_workspace = sync_attention_worker_workspace(state, worker_index);
  float* scores = NULL;
  __fp16* linear_S = NULL;
  __fp16* temp_O = NULL;
  sync_attention_workspace_offsets(state, worker_workspace, state->qo_len, &scores, &linear_S, &temp_O);
  __fp16* head_O = state->O + (size_t)head_id * (state->head_dim / 64) * state->qo_total_len * 64;

  if (state->mask_stride < 0) {
    int q_block_rows = sync_attention_use_prefill64_blocks(state) ? 64 : 32;
    int prezero_linear_s = (q_block_rows == 64);
    for (int q_begin = 0; q_begin < state->qo_len; q_begin += q_block_rows) {
      int q_rows = state->qo_len - q_begin;
      if (q_rows > q_block_rows) {
        q_rows = q_block_rows;
      }
      int block_valid_end = sync_attention_causal_valid_end(state, q_begin + q_rows);
      int block_valid_end_padded = (block_valid_end + 31) & ~31;
      if (block_valid_end_padded <= 0) {
        continue;
      }
      if (prezero_linear_s) {
        sync_attention_clear_linear_block(linear_S, state, q_begin, q_rows, block_valid_end);
      }

      sync_attention_run_causal_qk_block(state, scores, head_id, h_kv, q_begin, q_rows,
                                         block_valid_end, block_valid_end_padded);
      sync_attention_normalize_causal_block(state, scores, linear_S, q_begin, q_rows,
                                            block_valid_end, prezero_linear_s);
      sync_attention_run_causal_sv_block(state, head_O, temp_O, linear_S, h_kv, q_begin,
                                         q_rows, block_valid_end);
    }
    return;
  }

  if (state->page_count > 0) {
    sync_attention_run_page_qk(state, scores, state->Q + head_id * state->head_dim, state->qo_len, state->qo_stride,
                               0, h_kv, state->N);
  } else {
    run_locked_attn_hmx_matmul((uint8_t*)scores, (uint8_t*)(state->Q + head_id * state->head_dim), (uint8_t*)state->pastK,
                               state->qo_len, state->head_dim, state->N_padded, state->K_dim_padded, state->qo_stride,
                               ATTN_HMX_OUT_LINEAR_FP32_SCALED, state->scale, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, h_kv,
                               state->n_kv_heads);
  }

  int maskStartPos = sync_attention_mask_start_pos(state);

  for (int q_i = 0; q_i < state->qo_len; ++q_i) {
    float* row_scores = &scores[q_i * state->N_padded];
    __fp16* row_s = linear_S + (size_t)q_i * state->N_padded;
    int validN = state->N;
    if (state->mask != NULL && state->mask_stride > 0) {
      const float* m_ptr = state->mask + (size_t)q_i * state->mask_stride;
      sync_attention_add_mask(row_scores, m_ptr, state->N, maskStartPos);
    }
    sync_attention_softmax_row(row_s, row_scores, validN);
    if (validN < state->N) {
      sync_attention_zero_fp16(row_s + validN, state->N - validN);
    }
  }

  if (state->page_count > 0) {
    sync_attention_run_page_sv(state, head_O, temp_O, linear_S, state->qo_len, state->qo_total_len, state->q_offset,
                               h_kv, state->N);
  } else {
    run_locked_attn_hmx_matmul_ex((uint8_t*)head_O, (uint8_t*)linear_S, (uint8_t*)state->pastV, state->qo_len,
                                  state->N, state->K_dim_padded, state->N_padded, state->N_padded,
                                  ATTN_HMX_OUT_PACKED_FP16, 1.0f, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, h_kv,
                                  state->n_kv_heads, state->qo_total_len, state->q_offset);
  }
}

static inline int sync_attention_decode_group_q_rows(const SyncAttentionTaskState* state) {
  if (sync_attention_can_group_causal(state->qo_len, state->seq_current, state->N - state->seq_current,
                                      state->total_heads * state->gqa_factor, state->total_heads,
                                      state->head_dim, state->N, state->mask_stride)) {
    int q_rows = sync_attention_causal_group_q_rows(state->qo_len);
    if (MNN_ATTENTION_HMX_COMBINE_DECODE && state->page_count > 0 && state->seq_current >= state->page_size) {
      int page_fast_q_rows = 32 / state->gqa_factor;
      if (page_fast_q_rows < 1) {
        page_fast_q_rows = 1;
      }
      if (q_rows > page_fast_q_rows) {
        q_rows = page_fast_q_rows;
      }
    }
    return q_rows;
  }
  int group_q_rows = 32 / state->gqa_factor;
  if (group_q_rows < 1) group_q_rows = 1;
  if (group_q_rows > state->qo_len) group_q_rows = state->qo_len;
  return group_q_rows;
}

// q_block < 0 processes every q block of this kv_head (decode, and the un-split prefill path);
// q_block >= 0 processes only that one block, which is how the grouped-causal prefill path gets
// n_kv_heads * causal_q_blocks tasks instead of n_kv_heads. Blocks are independent: each derives its
// own causal extent from q_base and writes its own rows of O.
static void sync_attention_process_decode_group(const SyncAttentionTaskState *state, int kv_head, int worker_index,
                                                int q_block) {
  if (state->online_pages) {
    sync_attention_process_online_pages(state, kv_head, worker_index, 1);
    return;
  }
  const int group_heads = state->gqa_factor;
  const int head_base = kv_head * group_heads;
  int group_q_rows = sync_attention_decode_group_q_rows(state);
  const int group_rows = group_heads * group_q_rows;

  uint8_t* worker_workspace = sync_attention_worker_workspace(state, worker_index);
  float* scores = NULL;
  __fp16* linear_S = NULL;
  __fp16* temp_O = NULL;
  sync_attention_workspace_offsets(state, worker_workspace, group_rows, &scores, &linear_S, &temp_O);

  if (state->qo_len > 1) {
    __fp16* packed_Q = linear_S;
    int     q_first  = 0;
    int     q_limit  = state->qo_len;
    if (q_block >= 0) {
      q_first = q_block * group_q_rows;
      if (q_first >= state->qo_len) {
        return;
      }
      q_limit = q_first + group_q_rows;
      if (q_limit > state->qo_len) {
        q_limit = state->qo_len;
      }
    }
    for (int q_base = q_first; q_base < q_limit; q_base += group_q_rows) {
      int q_count = state->qo_len - q_base;
      if (q_count > group_q_rows) {
        q_count = group_q_rows;
      }
      int rows = q_count * group_heads;
      int block_valid_end = sync_attention_causal_valid_end(state, q_base + q_count);

      unsigned long long _tw = WATTN_T0();
      for (int q = 0; q < q_count; ++q) {
        for (int h = 0; h < group_heads; ++h) {
          const __fp16* src = state->Q + (size_t)(q_base + q) * state->qo_stride + (head_base + h) * state->head_dim;
          __fp16* dst = packed_Q + (size_t)(q * group_heads + h) * state->head_dim;
          for (int d = 0; d < state->head_dim; d += 64) {
            vmemu(dst + d) = vmemu(src + d);
          }
        }
      }

      WATTN_ADD(worker_index, 0, _tw);
      _tw = WATTN_T0();
      if (state->page_count > 0) {
        sync_attention_run_page_qk(state, scores, packed_Q, rows, state->head_dim, 0, kv_head, block_valid_end);
      } else {
        int block_valid_end_padded = (block_valid_end + 31) & ~31;
        run_locked_attn_hmx_matmul((uint8_t*)scores, (uint8_t*)packed_Q, (uint8_t*)state->pastK,
                                   rows, state->head_dim, block_valid_end_padded, state->K_dim_padded, state->head_dim,
                                   ATTN_HMX_OUT_LINEAR_FP32_SCALED, state->scale, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256,
                                   kv_head, state->n_kv_heads);
      }

      WATTN_ADD(worker_index, 1, _tw);
      _tw              = WATTN_T0();
      __fp16 *scores16 = (__fp16 *) scores;  // ATTN_SCORES_FP16: scores buffer reused as fp16 (lower half)
      for (int q = 0; q < q_count; ++q) {
        int validN = sync_attention_causal_valid_end(state, q_base + q + 1);
        for (int h = 0; h < group_heads; ++h) {
          int     row   = q * group_heads + h;
          __fp16* row_s = linear_S + (size_t)row * state->N_padded;
          if (state->scores_fp16) {
            sync_attention_softmax_row_fp16(row_s, scores16 + (size_t) row * state->N_padded, validN);
          } else {
            sync_attention_softmax_row(row_s, scores + (size_t) row * state->N_padded, validN);
          }
          if (validN < block_valid_end) {
            sync_attention_zero_fp16(row_s + validN, block_valid_end - validN);
          }
        }
      }

      WATTN_ADD(worker_index, 2, _tw);
      _tw              = WATTN_T0();
      __fp16* packed_O = (__fp16*)scores;
      if (state->page_count > 0) {
        sync_attention_run_page_sv(state, packed_O, temp_O, linear_S, rows, rows, 0, kv_head, block_valid_end);
      } else {
        run_locked_attn_hmx_matmul((uint8_t*)packed_O, (uint8_t*)linear_S, (uint8_t*)state->pastV,
                                   rows, block_valid_end, state->K_dim_padded, state->N_padded, state->N_padded,
                                   ATTN_HMX_OUT_PACKED_FP16, 1.0f, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256,
                                   kv_head, state->n_kv_heads);
      }
      WATTN_ADD(worker_index, 3, _tw);
      _tw = WATTN_T0();
      for (int q = 0; q < q_count; ++q) {
        for (int h = 0; h < group_heads; ++h) {
          int row = q * group_heads + h;
          __fp16* dst = state->O + (size_t)(head_base + h) * (state->head_dim / 64) * state->qo_total_len * 64;
          for (int pack_idx = 0; pack_idx < state->head_dim / 64; ++pack_idx) {
            const __fp16* src = packed_O + (size_t)(pack_idx * rows + row) * 64;
            vmemu(dst + (size_t)pack_idx * state->qo_total_len * 64 +
                  (state->q_offset + q_base + q) * 64) = vmemu(src);
          }
        }
      }
      WATTN_ADD(worker_index, 4, _tw);
    }
    return;
  }

  int attention_valid_end = state->N;
  if (state->mask_stride < 0) {
    attention_valid_end = sync_attention_causal_valid_end(state, 1);
  }

  if (state->page_count > 0) {
    sync_attention_run_page_qk(state, scores, state->Q + (size_t)head_base * state->head_dim,
                               group_heads, state->head_dim, 0, kv_head, attention_valid_end);
  } else {
    run_locked_attn_hmx_matmul((uint8_t*)scores, (uint8_t*)(state->Q + (size_t)head_base * state->head_dim),
                               (uint8_t*)state->pastK, group_heads, state->head_dim,
                               (attention_valid_end + 31) & ~31, state->K_dim_padded, state->head_dim,
                               ATTN_HMX_OUT_LINEAR_FP32_SCALED, state->scale, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256,
                               kv_head, state->n_kv_heads);
  }

  int maskStartPos = sync_attention_mask_start_pos(state);

  for (int h = 0; h < group_heads; ++h) {
    float* row_scores = &scores[(size_t)h * state->N_padded];
    __fp16* row_s = linear_S + (size_t)h * state->N_padded;
    int validN = state->N;
    if (state->mask_stride < 0) {
      validN = attention_valid_end;
    } else if (state->mask != NULL && state->mask_stride > 0) {
      sync_attention_add_mask(row_scores, state->mask, state->N, maskStartPos);
    }
    sync_attention_softmax_row(row_s, row_scores, validN);
    if (validN < attention_valid_end) {
      sync_attention_zero_fp16(row_s + validN, attention_valid_end - validN);
    }
  }

  __fp16* packed_O = (__fp16*)scores;
  if (state->page_count > 0) {
    sync_attention_run_page_sv(state, packed_O, temp_O, linear_S, group_heads, group_heads, 0, kv_head,
                               attention_valid_end);
  } else {
    run_locked_attn_hmx_matmul((uint8_t*)packed_O, (uint8_t*)linear_S, (uint8_t*)state->pastV, group_heads,
                               attention_valid_end, state->K_dim_padded, state->N_padded, state->N_padded,
                               ATTN_HMX_OUT_PACKED_FP16, 1.0f, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head,
                               state->n_kv_heads);
  }
  sync_attention_copy_decode_group_output(state->O, packed_O, head_base, group_heads, state->head_dim, NULL);
}

static inline void sync_attention_process_task(const SyncAttentionTaskState* state, int task_id, int worker_index) {
  if (state->prefill_segment_q > 0) {
    const int segment_index = task_id / state->prefill_n_heads;
    const int head_id = task_id - segment_index * state->prefill_n_heads;
    const int q_offset = segment_index * state->prefill_segment_q;
    int segment_q = state->qo_len - q_offset;
    if (segment_q > state->prefill_segment_q) {
      segment_q = state->prefill_segment_q;
    }

    SyncAttentionTaskState segment = *state;
    segment.qo_len = segment_q;
    segment.q_offset = q_offset;
    segment.qo_total_len = state->qo_len;
    segment.seq_current = state->seq_current + q_offset;
    segment.N = segment.seq_current + segment_q;
    segment.N_padded = (segment.N + 31) / 32 * 32;
    segment.Q = state->Q + (size_t)q_offset * state->qo_stride;
    segment.prefill_segment_q = 0;
    segment.prefill_n_heads = 0;
    sync_attention_process_head(&segment, head_id, worker_index);
    return;
  }
  if (state->decode_grouped) {
    if (state->causal_q_blocks > 1) {
      // task_id -> (kv_head, q_block). kv_head is the fast axis so that the n_kv_heads tasks of one
      // q block are handed out together. Blocks run in DESCENDING q order because a causal block's
      // cost grows with q_base (block 0 attends to ~group_q_rows keys, the last one to all of them);
      // starting with the most expensive leaves only cheap tasks for the tail of the schedule.
      const int kv_head = task_id % state->n_kv_heads;
      const int q_block = state->causal_q_blocks - 1 - task_id / state->n_kv_heads;
      sync_attention_process_decode_group(state, kv_head, worker_index, q_block);
    } else {
      sync_attention_process_decode_group(state, task_id, worker_index, -1);
    }
  } else {
    sync_attention_process_head(state, task_id, worker_index);
  }
}

// See the declaration in attention_private.hpp. Splits the grouped-causal prefill into one task per
// (kv_head, q block) so the task count stops being tied to n_kv_heads.
int sync_attention_finalize_causal_tasks(SyncAttentionTaskState *state) {
  state->causal_q_blocks     = 1;
  // Only split when the coarse decomposition (one task per kv_head) leaves a partial scheduling wave,
  // because then its tail runs with most threads idle. If n_kv_heads is a multiple of the worker count
  // the coarse tasks already fill whole waves and the finer split only adds per-task overhead and costs
  // K/V locality: measured on v81 (8 workers, 8 kv heads) splitting anyway made FLASH_ATTN 145.9 ->
  // 151.1 ms. On 6 workers, where 8 tasks means two waves, it is 214.9 -> 183.2 ms.
  const unsigned int workers = g_max_num_workers;
  if (state->decode_grouped && state->qo_len > 1 && !state->online_pages && workers > 1 && state->n_kv_heads > 0 &&
      ((unsigned int) state->n_kv_heads % workers) != 0) {
    const int group_q_rows = sync_attention_decode_group_q_rows(state);
    if (group_q_rows > 0) {
      const int blocks = (state->qo_len + group_q_rows - 1) / group_q_rows;
      if (blocks > 1) {
        state->causal_q_blocks = blocks;
        state->total_heads     = state->n_kv_heads * blocks;
      }
    }
  }
  return sync_attention_pick_task_count(state->total_heads);
}

static void sync_attention_worker(void* data, int worker_index) {
  SyncAttentionTaskState* state = (SyncAttentionTaskState*)data;
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= state->total_heads) {
      break;
    }
    sync_attention_process_task(state, (int)task_id, worker_index);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

void sync_attention_run_tasks(SyncAttentionTaskState* state, int n_tasks) {
  hmx_queue_begin();
  if (n_tasks <= 1) {
    for (int task = 0; task < state->total_heads; ++task) {
      sync_attention_process_task(state, task, 0);
    }
    hmx_queue_end();
    return;
  }

  worker_pool_job_t job;
  job.fptr = sync_attention_worker;
  job.dptr = state;

  worker_pool_synctoken_init(&(state->sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state->sync_ctx));
  hmx_queue_end();
}
