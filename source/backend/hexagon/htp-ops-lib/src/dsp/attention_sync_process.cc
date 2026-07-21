#include "attention_private.hpp"

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

    const int combine_hmx = decode_grouped && MNN_ATTENTION_HMX_COMBINE_DECODE;
    if (combine_hmx) {
      sync_attention_hmx_section_begin();
      attn_hmx_matmul_page_qk_block(state, scores, q_ptr, rows, q_stride, kv_head, page, page_end,
                                    block_start, block_valid);
    } else {
      sync_attention_run_page_qk_block(state, scores, q_ptr, rows, q_stride, kv_head, page, page_end,
                                       block_start, block_valid);
    }

    int any_valid = 0;
    for (int r = 0; r < rows; ++r) {
      any_valid |= sync_attention_prepare_online_row(state,
                                                     scores + (size_t)r * state->online_block_size,
                                                     linear_S + (size_t)r * state->online_block_size,
                                                     r, block_start, block_valid, decode_grouped ? 0 : r,
                                                     row_max, row_sum, row_scale);
    }
    if (!any_valid) {
      if (combine_hmx) {
        sync_attention_hmx_section_end();
      }
      continue;
    }

    if (combine_hmx) {
      attn_hmx_matmul_page_sv_block(state, temp_O, page_temp_O, linear_S, rows, kv_head, page, page_end,
                                    block_start, block_valid);
      sync_attention_hmx_section_end();
    } else {
      sync_attention_run_page_sv_block(state, temp_O, page_temp_O, linear_S, rows, kv_head, page, page_end,
                                       block_start, block_valid);
    }
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

static void sync_attention_process_decode_group(const SyncAttentionTaskState* state, int kv_head, int worker_index) {
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
    for (int q_base = 0; q_base < state->qo_len; q_base += group_q_rows) {
      int q_count = state->qo_len - q_base;
      if (q_count > group_q_rows) {
        q_count = group_q_rows;
      }
      int rows = q_count * group_heads;
      int block_valid_end = sync_attention_causal_valid_end(state, q_base + q_count);

      for (int q = 0; q < q_count; ++q) {
        for (int h = 0; h < group_heads; ++h) {
          const __fp16* src = state->Q + (size_t)(q_base + q) * state->qo_stride + (head_base + h) * state->head_dim;
          __fp16* dst = packed_Q + (size_t)(q * group_heads + h) * state->head_dim;
          memcpy(dst, src, (size_t)state->head_dim * sizeof(__fp16));
        }
      }

      if (state->page_count > 0) {
        sync_attention_run_page_qk(state, scores, packed_Q, rows, state->head_dim, 0, kv_head, block_valid_end);
      } else {
        int block_valid_end_padded = (block_valid_end + 31) & ~31;
        run_locked_attn_hmx_matmul((uint8_t*)scores, (uint8_t*)packed_Q, (uint8_t*)state->pastK,
                                   rows, state->head_dim, block_valid_end_padded, state->K_dim_padded, state->head_dim,
                                   ATTN_HMX_OUT_LINEAR_FP32_SCALED, state->scale, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256,
                                   kv_head, state->n_kv_heads);
      }

      for (int q = 0; q < q_count; ++q) {
        int validN = sync_attention_causal_valid_end(state, q_base + q + 1);
        for (int h = 0; h < group_heads; ++h) {
          int row = q * group_heads + h;
          float* row_scores = scores + (size_t)row * state->N_padded;
          __fp16* row_s = linear_S + (size_t)row * state->N_padded;
          sync_attention_softmax_row(row_s, row_scores, validN);
          if (validN < block_valid_end) {
            sync_attention_zero_fp16(row_s + validN, block_valid_end - validN);
          }
        }
      }

      __fp16* packed_O = (__fp16*)scores;
      if (state->page_count > 0) {
        sync_attention_run_page_sv(state, packed_O, temp_O, linear_S, rows, rows, 0, kv_head, block_valid_end);
      } else {
        run_locked_attn_hmx_matmul((uint8_t*)packed_O, (uint8_t*)linear_S, (uint8_t*)state->pastV,
                                   rows, block_valid_end, state->K_dim_padded, state->N_padded, state->N_padded,
                                   ATTN_HMX_OUT_PACKED_FP16, 1.0f, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256,
                                   kv_head, state->n_kv_heads);
      }
      for (int q = 0; q < q_count; ++q) {
        for (int h = 0; h < group_heads; ++h) {
          int row = q * group_heads + h;
          __fp16* dst = state->O + (size_t)(head_base + h) * (state->head_dim / 64) * state->qo_total_len * 64;
          for (int pack_idx = 0; pack_idx < state->head_dim / 64; ++pack_idx) {
            const __fp16* src = packed_O + (size_t)(pack_idx * rows + row) * 64;
            memcpy(dst + (size_t)pack_idx * state->qo_total_len * 64 +
                   (state->q_offset + q_base + q) * 64, src, 64 * sizeof(__fp16));
          }
        }
      }
    }
    return;
  }

  int attention_valid_end = state->N;
  if (state->mask_stride < 0) {
    attention_valid_end = sync_attention_causal_valid_end(state, 1);
  }

  const int combine_hmx = MNN_ATTENTION_HMX_COMBINE_DECODE;
  if (combine_hmx) {
    sync_attention_hmx_section_begin();
  }
  if (state->page_count > 0) {
    if (combine_hmx) {
      attn_hmx_matmul_pages_qk(state, scores, state->Q + (size_t)head_base * state->head_dim,
                               group_heads, state->head_dim, 0, kv_head, attention_valid_end);
    } else {
      sync_attention_run_page_qk(state, scores, state->Q + (size_t)head_base * state->head_dim,
                                 group_heads, state->head_dim, 0, kv_head, attention_valid_end);
    }
  } else {
    if (combine_hmx) {
      attn_hmx_matmul((uint8_t*)scores, (uint8_t*)(state->Q + (size_t)head_base * state->head_dim), (uint8_t*)state->pastK,
                      group_heads, state->head_dim, (attention_valid_end + 31) & ~31, state->K_dim_padded,
                      state->head_dim, ATTN_HMX_OUT_LINEAR_FP32_SCALED, state->scale,
                      ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head, state->n_kv_heads, 0, 0);
    } else {
      run_locked_attn_hmx_matmul((uint8_t*)scores, (uint8_t*)(state->Q + (size_t)head_base * state->head_dim),
                                 (uint8_t*)state->pastK, group_heads, state->head_dim,
                                 (attention_valid_end + 31) & ~31, state->K_dim_padded, state->head_dim,
                                 ATTN_HMX_OUT_LINEAR_FP32_SCALED, state->scale, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256,
                                 kv_head, state->n_kv_heads);
    }
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
    if (combine_hmx) {
      attn_hmx_matmul_pages_sv(state, packed_O, temp_O, linear_S, group_heads, group_heads, 0, kv_head,
                               attention_valid_end);
    } else {
      sync_attention_run_page_sv(state, packed_O, temp_O, linear_S, group_heads, group_heads, 0, kv_head,
                                 attention_valid_end);
    }
  } else {
    if (combine_hmx) {
      attn_hmx_matmul((uint8_t*)packed_O, (uint8_t*)linear_S, (uint8_t*)state->pastV, group_heads, attention_valid_end,
                      state->K_dim_padded, state->N_padded, state->N_padded, ATTN_HMX_OUT_PACKED_FP16, 1.0f,
                      ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head, state->n_kv_heads, 0, 0);
    } else {
      run_locked_attn_hmx_matmul((uint8_t*)packed_O, (uint8_t*)linear_S, (uint8_t*)state->pastV, group_heads,
                                 attention_valid_end, state->K_dim_padded, state->N_padded, state->N_padded,
                                 ATTN_HMX_OUT_PACKED_FP16, 1.0f, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head,
                                 state->n_kv_heads);
    }
  }
  if (combine_hmx) {
    sync_attention_hmx_section_end();
  }
  sync_attention_copy_decode_group_output(state->O, packed_O, head_base, group_heads, state->head_dim, NULL);
}

static void sync_attention_worker(void* data, int worker_index) {
  SyncAttentionTaskState* state = (SyncAttentionTaskState*)data;
#if !MNN_ATTENTION_HMX_ENABLE_PER_MATMUL
  hmx_manager_enable_execution();
#endif
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= state->total_heads) {
      break;
    }
    if (state->decode_grouped) {
      sync_attention_process_decode_group(state, (int)task_id, worker_index);
    } else {
      sync_attention_process_head(state, (int)task_id, worker_index);
    }
  }
#if !MNN_ATTENTION_HMX_ENABLE_PER_MATMUL
  hmx_manager_disable_execution();
#endif
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

void sync_attention_run_tasks(SyncAttentionTaskState* state, int n_tasks) {
  if (n_tasks <= 1) {
#if !MNN_ATTENTION_HMX_ENABLE_PER_MATMUL
    hmx_manager_enable_execution();
#endif
    for (int task = 0; task < state->total_heads; ++task) {
      if (state->decode_grouped) {
        sync_attention_process_decode_group(state, task, 0);
      } else {
        sync_attention_process_head(state, task, 0);
      }
    }
#if !MNN_ATTENTION_HMX_ENABLE_PER_MATMUL
    hmx_manager_disable_execution();
#endif
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
}
