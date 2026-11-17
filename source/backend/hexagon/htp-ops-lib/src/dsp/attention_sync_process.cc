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

// ATTN_SCORES_FP16: softmax over an fp16 scores row, done entirely in fp16 vectors.
//
// The arithmetic is fp16 rather than fp32 because that is what makes the pass fast: an HVX vector then
// holds 64 lanes instead of 32, which halves the number of exp2 evaluations, and exp2 dominates. The
// fp16 path measured the worker-side softmax phase at pp2048 going 2728 -> 1372 ms (accumulated over
// workers) -- and that is the phase which sets attention's wall clock, since the single queue thread
// only has ~200 ms of work against the workers' ~2700.
//
// The row is staged into an L1-resident scratch on the max pass, so the memory traffic is one read of
// the score row plus one write of linear_S. Keeping the scratch matters: a variant that dropped it and
// re-read the scores instead measured 20% slower on this phase, because L1 beats both VTCM and DDR for
// passes 2 and 3.
//
// Precision: the scores arrive as fp16 and P is consumed as fp16 by the SV matmul, so the only new error
// is in exp2 itself (~1e-3 relative, versus ~1e-7 for the fp32 polynomial) -- the same order as the fp16
// storage already in the path. The denominator is still accumulated in fp32, where a 2048-term sum would
// otherwise lose more than the exp2 approximation does. llama.cpp's HTP flash attention makes the same
// fp16 choice for the same reason.

static inline HVX_Vector sync_attention_fp16_splat(float value) {
  uint16_t bits = hmx_fp16_bits(value);
  return Q6_V_vsplat_R(((unsigned) bits) | (((unsigned) bits) << 16));
}

// fp16's most negative finite value; lanes past validN are forced to it so they lose every max.
#define ATTN_FP16_NINF_SPLAT Q6_Vh_vsplat_R(0xfbff)

// Reduce a per-lane fp16 max down to the row max, leaving it in EVERY lane. Unlike the fp32 version
// this needs no scalar extract and splat before the subtract.
static inline HVX_Vector sync_attention_fp16_rowmax(HVX_Vector v_max) {
  v_max = Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vror_VR(v_max, 64));
  v_max = Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vror_VR(v_max, 32));
  v_max = Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vror_VR(v_max, 16));
  v_max = Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vror_VR(v_max, 8));
  v_max = Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vror_VR(v_max, 4));
  return Q6_Vhf_vmax_VhfVhf(v_max, Q6_V_vror_VR(v_max, 2));
}

// Passes 2 and 3 of the fp16 softmax over a row already staged in L1, with its row max already reduced
// into every lane of `v_max`. Split out so each caller can fuse its own load with pass 1 rather than
// walking the row an extra time: the row-major reader loads contiguously, the tile reader un-interleaves
// a row pair. `stage` is overwritten with the exp values.
// Pass 2: replace the staged scores with exp2((s - max) * log2e) and return 1/sum. Shared by the
// row-major and tile writers, which differ only in how they get the result out of the scratch.
static inline float sync_attention_softmax_fp16_exp(__fp16 *stage, int validN, HVX_Vector v_max) {
  const HVX_Vector v_log2e = Q6_Vh_vsplat_R(0x3dc5);  // fp16 1.4427
  HVX_Vector       v_sum   = Q6_V_vzero();
  for (int i = 0; i < validN; i += 64) {
    HVX_Vector d   = Q6_Vqf16_vsub_VhfVhf(*(const HVX_Vector *) (stage + i), v_max);
    HVX_Vector e   = hvx_my_exp2_vhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(d, v_log2e)));
    int        rem = validN - i;
    if (rem < 64) {
      e = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem * (int) sizeof(__fp16)), e, Q6_V_vzero());
    }
    *(HVX_Vector *) (stage + i) = e;
    // fp32 denominator. The lane order the widening produces does not matter for a sum.
    HVX_VectorPair ef           = Q6_Wsf_vcvt_Vhf(e);
    v_sum                       = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_lo_W(ef)));
    v_sum                       = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_hi_W(ef)));
  }
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 64)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 32)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 16)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 8)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 4)));
  float sum_arr[32] __attribute__((aligned(128)));
  *(HVX_Vector *) sum_arr = v_sum;
  return sum_arr[0] > 0.0f ? 1.0f / sum_arr[0] : 0.0f;
}

// Pass 3, row-major: scale the staged exp by 1/sum into linear_S.
static inline void sync_attention_softmax_fp16_store_row(__fp16 *row_s, const __fp16 *stage, int validN,
                                                         float inv_sum) {
  HVX_Vector v_inv = sync_attention_fp16_splat(inv_sum);
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

// Pass 3, tile layout: scale both staged rows and interleave them back into the HMX tile layout the SV
// matmul wants, so that matmul needs no activation transform at all. One vshuff(-2) per 64 columns
// produces the two tiles covering those columns -- the exact inverse of attn_tile_read_row_pair.
//
// Writes the full `out_cols` rather than just validN, zeroing the causal tail: in tile layout a row's
// tail is not contiguous, so the caller cannot clear it with a memset afterwards.
static inline void sync_attention_softmax_fp16_store_tile_pair(__fp16 *p_tiles, const __fp16 *stage0,
                                                               const __fp16 *stage1, int vi, int out_cols, int validN0,
                                                               int validN1, float inv0, float inv1) {
  HVX_Vector v_inv0 = sync_attention_fp16_splat(inv0);
  HVX_Vector v_inv1 = sync_attention_fp16_splat(inv1);
  HVX_Vector v_zero = Q6_V_vzero();
  for (int c = 0; c < out_cols; c += 64) {
    HVX_Vector r0 =
      c < validN0 ? Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(*(const HVX_Vector *) (stage0 + c), v_inv0)) : v_zero;
    HVX_Vector r1 =
      c < validN1 ? Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(*(const HVX_Vector *) (stage1 + c), v_inv1)) : v_zero;
    HVX_VectorPair vp = Q6_W_vshuff_VVR(r1, r0, -2);
    HVX_Vector    *lo = (HVX_Vector *) (p_tiles + (size_t) (c / 32) * 1024) + vi;
    *lo               = Q6_V_lo_W(vp);
    if (c + 64 <= out_cols) {
      *(lo + 16) = Q6_V_hi_W(vp);
    }
  }
}

// Row-major entry: stage the score row in L1 while reducing its max, then share the tail.
static inline void sync_attention_softmax_row_fp16(__fp16 *row_s, const __fp16 *row_scores16, int validN) {
  __fp16 stage[ATTN_FIXED_WORKSPACE_KV] __attribute__((aligned(128)));
  if (validN > ATTN_FIXED_WORKSPACE_KV) {  // guaranteed by sync_attention_can_group_causal; be defensive
    FARF(ERROR, "softmax_row_fp16 validN overflow: %d", validN);
    return;
  }
  const HVX_Vector v_ninf = ATTN_FP16_NINF_SPLAT;
  HVX_Vector       v_max  = v_ninf;
  for (int i = 0; i < validN; i += 64) {
    HVX_Vector raw              = vmemu(row_scores16 + i);
    *(HVX_Vector *) (stage + i) = raw;  // i is a multiple of 64 fp16 => 128B aligned
    const int rem               = validN - i;
    if (rem < 64) {
      raw = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem * (int) sizeof(__fp16)), raw, v_ninf);
    }
    v_max = Q6_Vhf_vmax_VhfVhf(v_max, raw);
  }
  const float inv = sync_attention_softmax_fp16_exp(stage, validN, sync_attention_fp16_rowmax(v_max));
  sync_attention_softmax_fp16_store_row(row_s, stage, validN, inv);
}

// Read 64 columns of rows (r, r+1) out of the HMX tile layout.
//
// Tile (ox, oy) sits at `tiles[oy * 1024]` and holds 32 rows x 32 columns with rows r and r+1
// interleaved halfword-wise in vector r/2. One vdeal(-2) across the matching vectors of two adjacent
// tiles therefore un-interleaves 64 columns of row r into the low half and of row r+1 into the high
// half -- the same 64-column chunk the row-major softmax loads, except nobody had to write it out in
// row order first. A trailing odd tile has only 32 columns, which a single-vector vdeal gives.
static inline void attn_tile_read_row_pair(const __fp16 *tiles, int vi, int c, int cols, HVX_Vector *r0,
                                           HVX_Vector *r1) {
  const HVX_Vector *p = (const HVX_Vector *) (tiles + (size_t) (c / 32) * 1024) + vi;
  if (c + 64 <= cols) {
    HVX_VectorPair vp = Q6_W_vdeal_VVR(p[16], p[0], -2);
    *r0               = Q6_V_lo_W(vp);
    *r1               = Q6_V_hi_W(vp);
    return;
  }
  HVX_Vector d = Q6_Vh_vdeal_Vh(p[0]);
  *r0          = d;
  *r1          = Q6_V_vror_VR(d, 64);
}

// Softmax over one row pair of an M-tile, reading the QK scores straight out of their HMX tile layout
// (ATTN_SCORES_TILE). Un-interleaving the pair and staging both rows in L1 costs one vdeal per 64
// columns and buys the whole row-major score store on the queue thread.
//
// Both rows go through sync_attention_softmax_fp16_tail, so the arithmetic is identical to the
// row-major path; only the load differs. validN may differ between the two rows (causal), so the
// gather spans the wider one and each row's own extent bounds its max and its exp.
static inline void sync_attention_softmax_tile_pair(const __fp16 *s_tiles, __fp16 *p_tiles, int vi, int cols,
                                                    int out_cols, int validN0, int validN1) {
  __fp16 stage0[ATTN_FIXED_WORKSPACE_KV] __attribute__((aligned(128)));
  __fp16 stage1[ATTN_FIXED_WORKSPACE_KV] __attribute__((aligned(128)));
  const int validN = validN0 > validN1 ? validN0 : validN1;
  if (validN > ATTN_FIXED_WORKSPACE_KV) {  // guaranteed by sync_attention_can_group_causal; be defensive
    FARF(ERROR, "softmax_tile_pair validN overflow: %d", validN);
    return;
  }
  const HVX_Vector v_ninf = ATTN_FP16_NINF_SPLAT;
  HVX_Vector       v_max0 = v_ninf;
  HVX_Vector       v_max1 = v_ninf;
  for (int c = 0; c < validN; c += 64) {
    HVX_Vector r0, r1;
    attn_tile_read_row_pair(s_tiles, vi, c, cols, &r0, &r1);
    *(HVX_Vector *) (stage0 + c) = r0;
    *(HVX_Vector *) (stage1 + c) = r1;
    const int n0                 = validN0 - c;
    const int n1                 = validN1 - c;
    if (n0 > 0) {
      v_max0 =
        Q6_Vhf_vmax_VhfVhf(v_max0, n0 < 64 ? Q6_V_vmux_QVV(Q6_Q_vsetq_R(n0 * (int) sizeof(__fp16)), r0, v_ninf) : r0);
    }
    if (n1 > 0) {
      v_max1 =
        Q6_Vhf_vmax_VhfVhf(v_max1, n1 < 64 ? Q6_V_vmux_QVV(Q6_Q_vsetq_R(n1 * (int) sizeof(__fp16)), r1, v_ninf) : r1);
    }
  }
  const float inv0 = sync_attention_softmax_fp16_exp(stage0, validN0, sync_attention_fp16_rowmax(v_max0));
  const float inv1 =
    validN1 > 0 ? sync_attention_softmax_fp16_exp(stage1, validN1, sync_attention_fp16_rowmax(v_max1)) : 0.0f;
  sync_attention_softmax_fp16_store_tile_pair(p_tiles, stage0, stage1, vi, out_cols, validN0, validN1, inv0, inv1);
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

  float block_max = sync_attention_max_f32(row_scores, valid);
  float old_max = row_max[row];
  float old_sum = row_sum[row];
  float new_max = old_sum > 0.0f ? (old_max > block_max ? old_max : block_max) : block_max;
  float old_scale = old_sum > 0.0f ? expf(old_max - new_max) : 0.0f;
  row_scale[row] = old_scale;
  // exp_and_sum already subtracts the max it is handed, so pass new_max instead of pre-subtracting it in
  // a separate pass over the row.
  float block_sum = sync_attention_exp_and_sum(row_scores, valid, new_max);
  sync_attention_normalize_to_fp16(row_s, row_scores, valid, 1.0f);
  if (valid < page_valid) {
    sync_attention_zero_fp16(row_s + valid, page_valid - valid);
  }
  row_max[row] = new_max;
  row_sum[row] = old_sum * old_scale + block_sum;
  return 1;
}

// accum = accum * row_scale[r] + src, one packed row (64 fp16 = exactly one HVX vector) at a time. This
// is the online-softmax rescale fused with the block accumulation: done as two passes it reads and writes
// accum twice.
static inline void sync_attention_rescale_add_packed_rows(__fp16 *accum, const __fp16 *src, int rows, int head_dim,
                                                          const float *row_scale) {
  int packs = head_dim / 64;
  for (int p = 0; p < packs; ++p) {
    __fp16       *a_base = accum + (size_t) p * rows * 64;
    const __fp16 *s_base = src + (size_t) p * rows * 64;
    for (int r = 0; r < rows; ++r) {
      HVX_Vector v_scale = sync_attention_fp16_splat(row_scale[r]);
      __fp16    *a       = a_base + (size_t) r * 64;
      HVX_Vector scaled  = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vmemu(a), v_scale));
      vmemu(a)           = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(scaled, vmemu(s_base + (size_t) r * 64)));
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
      float         inv   = row_sum[r] > 0.0f ? 1.0f / row_sum[r] : 0.0f;
      HVX_Vector    v_inv = sync_attention_fp16_splat(inv);
      const __fp16 *s     = s_base + (size_t) r * 64;
      __fp16       *d     = d_base + (size_t) r * 64;
      vmemu(d)            = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vmemu(s), v_inv));
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
        float      scale       = row_sum[h] > 0.0f ? 1.0f / row_sum[h] : 0.0f;
        HVX_Vector v_scale     = sync_attention_fp16_splat(scale);
        vmemu(dst + pack * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vmemu(src), v_scale));
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
    sync_attention_rescale_add_packed_rows(accum_O, temp_O, rows, state->K_dim_padded, row_scale);
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
                                      state->total_heads * state->gqa_factor, state->total_heads, state->head_dim,
                                      state->N, state->mask_stride, state->workspace_kv, state->kv_chunk)) {
    int q_rows =
      sync_attention_causal_group_q_rows(state->qo_len, state->N, state->head_dim, state->gqa_factor, state->kv_chunk);
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
  // Redirect to whichever VTCM regions sync_attention_try_vtcm_workspace managed to place. The two are
  // independent: a long sequence gets scores resident while linear_S / temp_O stay in DDR.
  if (state->scores_vtcm != NULL) {
    scores = (float *) (state->scores_vtcm + (size_t) worker_index * state->scores_vtcm_stride);
  }
  if (state->act_vtcm != NULL) {
    uint8_t *act = state->act_vtcm + (size_t) worker_index * state->act_vtcm_stride;
    if (state->scores_tile) {
      // Tile layout writes P over S in place, so the activation region only holds temp_O.
      temp_O = (__fp16 *) act;
    } else {
      linear_S = (__fp16 *) act;
      temp_O   = (__fp16 *) (act + attn_align_128((size_t) group_rows * state->N_padded * sizeof(__fp16)));
    }
  }
  if (state->scores_tile) {
    // The softmax stages both rows of a pair in L1 before writing, so the tile it writes back is one it
    // has finished reading -- P can land on top of S. That halves the resident footprint, which is what
    // lets it stay in VTCM at seq 2048 and removes the SV activation DMA entirely (2.0 GB/run).
    linear_S = (__fp16 *) scores;
  }

  if (state->qo_len > 1) {
    // packed_Q cannot live in linear_S on the tile path -- that aliases the score buffer, which QK is
    // about to write. temp_O is free here: it only receives the SV output, which happens after QK has
    // consumed packed_Q.
    __fp16 *packed_Q = state->scores_tile ? temp_O : linear_S;
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

      for (int q = 0; q < q_count; ++q) {
        for (int h = 0; h < group_heads; ++h) {
          const __fp16* src = state->Q + (size_t)(q_base + q) * state->qo_stride + (head_base + h) * state->head_dim;
          __fp16* dst = packed_Q + (size_t)(q * group_heads + h) * state->head_dim;
          for (int d = 0; d < state->head_dim; d += 64) {
            vmemu(dst + d) = vmemu(src + d);
          }
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

      __fp16 *scores16 = (__fp16 *) scores;  // ATTN_SCORES_FP16: scores buffer reused as fp16 (lower half)
      if (state->scores_tile) {
        // ATTN_SCORES_TILE: the scores are still in the HMX tile layout, so walk row pairs -- a tile
        // vector holds two rows -- instead of single rows.
        const int cols = state->N_padded;
        for (int row = 0; row < rows; row += 2) {
          const int ox      = row / 32;
          const int vi      = (row % 32) >> 1;
          const int validN0 = sync_attention_causal_valid_end(state, q_base + row / group_heads + 1);
          const int validN1 =
            (row + 1 < rows) ? sync_attention_causal_valid_end(state, q_base + (row + 1) / group_heads + 1) : 0;
          sync_attention_softmax_tile_pair(scores16 + (size_t) ox * cols * 32, linear_S + (size_t) ox * cols * 32, vi,
                                           cols, (block_valid_end + 31) & ~31, validN0, validN1);
        }
      } else {
        for (int q = 0; q < q_count; ++q) {
          int validN = sync_attention_causal_valid_end(state, q_base + q + 1);
          for (int h = 0; h < group_heads; ++h) {
            int     row   = q * group_heads + h;
            __fp16 *row_s = linear_S + (size_t) row * state->N_padded;
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
      }

      // Cannot alias `scores` on the tile path: that is P now, and the SV matmul reads it throughout.
      // temp_O is exactly rows * head_dim_padded, and the all-page fusion the tile layout forces does
      // not use it for anything else.
      __fp16 *packed_O = state->scores_tile ? temp_O : (__fp16 *) scores;
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
            vmemu(dst + (size_t)pack_idx * state->qo_total_len * 64 +
                  (state->q_offset + q_base + q) * 64) = vmemu(src);
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
