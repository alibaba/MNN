#include "attention_private.hpp"
#include "HAP_perf.h"

// STAGE 0 (prefill-attn decouple plan): race-free per-phase timers on the SINGLE
// hmx_queue thread (all page QK/SV run here). Splits queue-thread busy into
// [0]=DMA-wait (activation+weight load), [1]=HMX-compute, [2]=score-store.
// Surfaced by execute_command.cc into profile[244..246]. Cumulative. Default off.
#ifndef HTP_QATTN_PHASE_PROFILE
#  define HTP_QATTN_PHASE_PROFILE 0
#endif
#if HTP_QATTN_PHASE_PROFILE
unsigned long long g_qattn_phase_us[3] = { 0, 0, 0 };
unsigned long long g_qattn_dma_bytes   = 0;  // weight(K/V)-DMA bytes, for bandwidth = bytes / [244]
// dmstart counts: [0]=weight(K/V) chunks, [1]=activation(Q/S) tiles. [244]/(sum) = per-transfer wait,
// which tells whether the DMA wait is byte-bound or transfer-count(latency)-bound.
unsigned long long g_qattn_dma_calls[2] = { 0, 0 };
unsigned long long g_qattn_act_bytes    = 0;  // activation(Q/S)-DMA bytes
#  define QATTN_T0()         (HAP_perf_get_time_us())
#  define QATTN_ADD(i, t0)   (g_qattn_phase_us[(i)] += HAP_perf_get_time_us() - (t0))
#  define QATTN_BYTES(n)     (g_qattn_dma_bytes += (unsigned long long) (n))
#  define QATTN_CALL(i)      (g_qattn_dma_calls[(i)]++)
#  define QATTN_ACT_BYTES(n) (g_qattn_act_bytes += (unsigned long long) (n))
#else
#  define QATTN_T0()       (0ULL)
#  define QATTN_ADD(i, t0) ((void) (t0))
#  define QATTN_BYTES(n)   ((void) 0)
#  define QATTN_CALL(i)      ((void) 0)
#  define QATTN_ACT_BYTES(n) ((void) 0)
#endif

static inline void attn_hmx_load_k_tiles(const __fp16* activation, const __fp16* weight, int kp) {
  for (int k = 0; k < kp; k += 32) {
    int tiles = kp - k;
    if (tiles > 32) tiles = 32;
    hmx_load_tiles_fp16(activation + (size_t)k * 1024, weight + (size_t)k * 1024, tiles);
  }
}

static inline HVX_Vector attn_hmx_scale_splat(float scale) {
  return hmx_fp16_scale_splat(scale);
}

static inline void attn_hmx_load_activation_dma(dma_desc_2d_t* desc, const uint8_t* src, __fp16* raw, int roi_width,
                                                int rows, int src_stride, int pair_packs,
                                                int clear_valid_rows_only) {
  const int raw_row_stride = pair_packs * 64;
  const int raw_width = raw_row_stride * (int)sizeof(int16_t);
  if (roi_width < raw_width) {
    if (clear_valid_rows_only) {
      for (int r = 0; r < rows; ++r) {
        memset(raw + (size_t)r * raw_row_stride, 0, (size_t)raw_row_stride * sizeof(int16_t));
      }
    } else {
      memset(raw, 0, (size_t)pair_packs * 32 * 64 * sizeof(int16_t));
    }
  }
  attn_prepare_dma_desc_2d(desc, src, raw, (uint32_t)roi_width, (uint32_t)rows,
                           (uint32_t)(src_stride * sizeof(int16_t)), (uint32_t)raw_width, 0);
  QATTN_CALL(1);
  QATTN_ACT_BYTES((size_t) rows * (size_t) roi_width);
  unsigned long long _tq = QATTN_T0();
  dma_wait_for_idle();
  dmstart((dma_desc_1d_t*)desc);
  dma_wait_for_idle();
  QATTN_ADD(0, _tq);
}

static inline void attn_hmx_compute_output_tiles(__fp16* vtcm_output, __fp16* vtcm_activation,
                                                 __fp16* vtcm_weight, int oy_start, int oy_end, int kp) {
  unsigned long long _tq = QATTN_T0();
  for (int oy = oy_start; oy < oy_end; ++oy) {
    __fp16* vtcm_dst = vtcm_output + (oy - oy_start) * 1024;
    int oy_offset = (oy - oy_start) * 16 * kp;
    attn_hmx_load_k_tiles(vtcm_activation, vtcm_weight + oy_offset * 64, kp);
    hmx_consume_accumulator_fp16(vtcm_dst);
  }
  QATTN_ADD(1, _tq);
}

static inline void attn_hmx_start_weight_dma(dma_desc_2d_t* descs, const uint8_t* b, __fp16* vtcm_weight,
                                             int oy_start, int oy_end, int kp, int layout, int kv_head,
                                             int n_kv_heads, int k_icP, int v_ocP, int wait_after) {
  attn_prepare_weight_dma_descs(nullptr, descs, b, vtcm_weight, oy_start, oy_end, kp, layout, kv_head, n_kv_heads,
                                k_icP, v_ocP);
  QATTN_BYTES((size_t) (oy_end - oy_start) * (size_t) kp * 1024 * sizeof(int16_t));  // ~K/V tile bytes
  QATTN_CALL(0);
  unsigned long long _tq = QATTN_T0();
  dma_wait_for_idle();
  dmstart((dma_desc_1d_t*)&descs[0]);
  if (wait_after) {
    dma_wait_for_idle();
  }
  QATTN_ADD(0, _tq);
}

void attn_hmx_matmul(uint8_t * c, const uint8_t * a, const uint8_t * b, int M, int K, int N, int max_K,
                            int a_stride, int outputLayoutType, float outputScale,
                            int weightLayoutType, int kv_head, int n_kv_heads, int output_stride, int output_row_offset) {
  int np_chunk = ATTN_HMX_KV_BLOCK_TILES;
  int pack = 64;
  int act_src_stride = a_stride > 0 ? a_stride : pack;
  int mp = (M + 31) / 32;
  int np = N / 32;
  int kp = (K + 31) / 32;
  int pair_packs = (K + 63) / 64;
  int k_icP = (max_K + 31) / 32;
  int v_ocP = np;
  if (pair_packs > ATTN_HMX_MAX_PAIR_PACKS) {
    FARF(ERROR, "attn_hmx_matmul pair_packs overflow: %d", pair_packs);
    return;
  }
  if (kp > ATTN_HMX_MAX_KP) {
    FARF(ERROR, "attn_hmx_matmul kp overflow: %d", kp);
    return;
  }

  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 2 * (size_t)np_chunk * 32 * kp * 32 * sizeof(int16_t));
  __fp16  *vtcm_activation_raw = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t)pair_packs * 32 * 64 * sizeof(int16_t));
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, kp * 32 * 32 * sizeof(int16_t));
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 32 * 32 * np_chunk * sizeof(int16_t));
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  if (outputLayoutType == ATTN_HMX_OUT_LINEAR_FP32_SCALED) {
    hmx_init_column_scales(vtcm_scales, attn_hmx_scale_splat(outputScale));
  } else {
    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));
  }
  hmx_set_output_scales(vtcm_scales);

  _Alignas(64) dma_desc_2d_t act_desc;
  _Alignas(64) dma_desc_2d_t weight_descs[2][ATTN_HMX_MAX_WEIGHT_DESCS];
  size_t weight_buf_elems = (size_t)np_chunk * 32 * kp * 32;
  __fp16* vtcm_weight_bufs[2] = {vtcm_weight, vtcm_weight + weight_buf_elems};

  if (M > 32 && N <= ATTN_HMX_KV_BLOCK && np <= np_chunk) {
    if (np > 0) {
      attn_hmx_start_weight_dma(weight_descs[0], b, vtcm_weight_bufs[0], 0, np, kp, weightLayoutType,
                                kv_head, n_kv_heads, k_icP, v_ocP, 1);
    }

    for (int ox = 0; ox < mp; ++ox) {
      int valid_xi = M - ox * 32;
      if (valid_xi > 32) valid_xi = 32;
      if (valid_xi <= 0) {
        break;
      }

      int roi_width = a_stride > 0 ? K * (int)sizeof(int16_t) : pair_packs * 64 * (int)sizeof(int16_t);
      size_t src_offset = a_stride > 0 ? ((size_t)ox * 32 * act_src_stride) : ((size_t)ox * 32 * 64);
      attn_hmx_load_activation_dma(&act_desc, a + src_offset * sizeof(int16_t), vtcm_activation_raw,
                                   roi_width, valid_xi, act_src_stride, pair_packs, 0);

      attn_transform_activation_hmx(vtcm_activation, vtcm_activation_raw, valid_xi, pair_packs, kp);

      attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation, vtcm_weight_bufs[0], 0, np, kp);
      if (outputLayoutType == ATTN_HMX_OUT_PACKED_FP16) {
        attn_store_output_packed_fp16(c, vtcm_output, 0, np, ox, M, valid_xi, output_stride, output_row_offset);
      } else {
        attn_store_output_linear_fp32(c, vtcm_output, 0, np, ox, N, valid_xi, output_stride, output_row_offset);
      }
    }
    return;
  }

  for (int ox = 0; ox < mp; ++ox) {
    int valid_xi = M - ox * 32;
    if (valid_xi > 32) valid_xi = 32;
    if (valid_xi <= 0) {
      break;
    }

    int roi_width = a_stride > 0 ? K * (int)sizeof(int16_t) : pair_packs * 64 * (int)sizeof(int16_t);
    size_t src_offset = a_stride > 0 ? ((size_t)ox * 32 * act_src_stride) : ((size_t)ox * 32 * 64);
    attn_hmx_load_activation_dma(&act_desc, a + src_offset * sizeof(int16_t), vtcm_activation_raw,
                                 roi_width, valid_xi, act_src_stride, pair_packs, 0);

    attn_transform_activation_hmx(vtcm_activation, vtcm_activation_raw, valid_xi, pair_packs, kp);

    int current_weight_buf_idx = 0;
    if (np > 0) {
      int first_oy_end = np_chunk;
      if (first_oy_end > np) first_oy_end = np;
      attn_hmx_start_weight_dma(weight_descs[current_weight_buf_idx], b, vtcm_weight_bufs[current_weight_buf_idx],
                                0, first_oy_end, kp, weightLayoutType, kv_head, n_kv_heads, k_icP, v_ocP, 1);
    }

    for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
      int oy_end = oy_start + np_chunk;
      if (oy_end > np) oy_end = np;
      __fp16* vtcm_weight_current = vtcm_weight_bufs[current_weight_buf_idx];
      int next_oy_start = oy_start + np_chunk;
      int next_weight_buf_idx = 1 - current_weight_buf_idx;
      bool has_next_chunk = next_oy_start < np;
      if (has_next_chunk) {
        int next_oy_end = next_oy_start + np_chunk;
        if (next_oy_end > np) next_oy_end = np;
        attn_hmx_start_weight_dma(weight_descs[next_weight_buf_idx], b, vtcm_weight_bufs[next_weight_buf_idx],
                                  next_oy_start, next_oy_end, kp, weightLayoutType, kv_head, n_kv_heads,
                                  k_icP, v_ocP, 0);
      }

      attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation, vtcm_weight_current, oy_start, oy_end, kp);
      if (outputLayoutType == ATTN_HMX_OUT_PACKED_FP16) {
        attn_store_output_packed_fp16(c, vtcm_output, oy_start, oy_end, ox, M, valid_xi, output_stride, output_row_offset);
      } else {
        attn_store_output_linear_fp32(c, vtcm_output, oy_start, oy_end, ox, N, valid_xi, output_stride, output_row_offset);
      }
      if (has_next_chunk) {
        dma_wait_for_idle();
        current_weight_buf_idx = next_weight_buf_idx;
      }
    }
  }

}

// One page's contribution to a causal block: the KV rows [start, start+valid) and their padded tile count.
typedef struct {
  int page;
  int start;
  int valid;
  int n;   // valid padded up to a 32-row tile boundary
  int np;  // n / 32 tiles
} AttnPageSpan;

// Walk to the next page at or after `from` that still has KV rows below valid_end. Returns 0 when done.
static inline int attn_next_page_span(const SyncAttentionTaskState *state, int valid_end, int from, AttnPageSpan *out) {
  for (int p = from; p < state->page_count; ++p) {
    int ps = p * state->page_size;
    if (ps >= valid_end) {
      break;
    }
    int pv = valid_end - ps;
    if (pv > state->page_size) {
      pv = state->page_size;
    }
    if (pv <= 0) {
      continue;
    }
    int n = (pv + 31) & ~31;
    if (n < 32) {
      continue;
    }
    out->page  = p;
    out->start = ps;
    out->valid = pv;
    out->n     = n;
    out->np    = n / 32;
    return 1;
  }
  return 0;
}

static inline void attn_wait_weight_dma(void) {
  unsigned long long _tq = QATTN_T0();
  dma_wait_for_idle();
  QATTN_ADD(0, _tq);
}

// ATTN_SCORES_FP16: dispatch the QK score store to fp16 or fp32. `scores` is the fp32 buffer base; for
// fp16 we reinterpret it as __fp16* and offset by `col_off` elements (scores buffer is fp32-sized, so
// the fp16 data uses its lower half).
static inline void attn_store_qk_scores(const SyncAttentionTaskState *state, float *scores, size_t col_off,
                                        __fp16 *vtcm_output, int oy_start, int oy_end, int ox, int N, int valid_xi,
                                        int q_row_offset) {
  unsigned long long _tq = QATTN_T0();
  if (state->scores_fp16) {
    attn_store_output_linear_fp16((uint8_t *) ((__fp16 *) scores + col_off), vtcm_output, oy_start, oy_end, ox, N,
                                  valid_xi, state->N_padded, q_row_offset);
  } else {
    attn_store_output_linear_fp32((uint8_t *) (scores + col_off), vtcm_output, oy_start, oy_end, ox, N, valid_xi,
                                  state->N_padded, q_row_offset);
  }
  QATTN_ADD(2, _tq);
}

void attn_hmx_matmul_pages_qk(const SyncAttentionTaskState* state, float* scores, const __fp16* q_ptr, int rows,
                                     int q_stride, int q_row_offset, int kv_head, int valid_end) {
  const int np_chunk = ATTN_HMX_KV_BLOCK_TILES;
  const int M = rows;
  const int K = state->head_dim;
  const int max_K = state->K_dim_padded;
  const int act_src_stride = q_stride;
  const int mp = (M + 31) / 32;
  const int kp = (K + 31) / 32;
  const int pair_packs = (K + 63) / 64;
  const int k_icP = (max_K + 31) / 32;

  if (pair_packs > ATTN_HMX_MAX_PAIR_PACKS || kp > ATTN_HMX_MAX_KP) {
    FARF(ERROR, "attn_hmx_matmul_pages_qk overflow: pair_packs=%d kp=%d", pair_packs, kp);
    return;
  }

  uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
  __fp16* vtcm_weight = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 2 * (size_t)np_chunk * 32 * kp * 32 * sizeof(int16_t));
  // Raw staging holds all mp M-tiles so they can be fetched with ONE DMA transfer (the queue-thread
  // DMA wait is ~0.75us of fixed per-transfer latency, bytes-independent -- see raw_tile_elems users).
  const size_t raw_tile_elems = (size_t) pair_packs * 32 * 64;
  __fp16 *vtcm_activation_raw = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) mp * raw_tile_elems * sizeof(int16_t));
  // Plan(a) K-reuse: for M>32 (large q-block prefill) keep all mp transformed Q
  // M-tiles resident so a page's K is read ONCE and reused across every M-tile,
  // instead of re-reading the whole K per M-tile. mp==1 for the M<=32 paths.
  const size_t act_tile_elems      = (size_t) kp * 32 * 32;
  __fp16      *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) mp * act_tile_elems * sizeof(int16_t));
  __fp16* vtcm_output = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 32 * 32 * np_chunk * sizeof(int16_t));
  __fp16* vtcm_scales = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, attn_hmx_scale_splat(state->scale));
  hmx_set_output_scales(vtcm_scales);

  _Alignas(64) dma_desc_2d_t act_desc;
  _Alignas(64) dma_desc_2d_t weight_descs[2][ATTN_HMX_MAX_WEIGHT_DESCS];
  size_t weight_buf_elems = (size_t)np_chunk * 32 * kp * 32;
  __fp16* vtcm_weight_bufs[2] = {vtcm_weight, vtcm_weight + weight_buf_elems};

  if (M > 32 && valid_end <= ATTN_HMX_KV_BLOCK) {
    for (int page = 0; page < state->page_count; ++page) {
      int page_start = page * state->page_size;
      if (page_start >= valid_end) break;
      int page_valid = valid_end - page_start;
      if (page_valid > state->page_size) page_valid = state->page_size;
      if (page_valid <= 0) continue;
      if (state->async_push != NULL && page_start + page_valid > state->async_push_seq_begin) {
        sync_attention_wait_async_push(state);
      }

      int N = (page_valid + 31) & ~31;
      int np = N / 32;
      const uint8_t* b = state->pastKPages[page];
      attn_hmx_start_weight_dma(weight_descs[0], b, vtcm_weight_bufs[0], 0, np, kp,
                                ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head, state->n_kv_heads, k_icP, np, 1);

      // All mp Q M-tiles in ONE DMA (per-transfer latency, not bytes, dominates the DMA wait).
      attn_hmx_load_activation_dma(&act_desc, (const uint8_t *) q_ptr, vtcm_activation_raw, K * (int) sizeof(int16_t),
                                   M, act_src_stride, pair_packs, 1);

      for (int ox = 0; ox < mp; ++ox) {
        int valid_xi = M - ox * 32;
        if (valid_xi > 32) valid_xi = 32;
        if (valid_xi <= 0) break;

        attn_transform_activation_hmx(vtcm_activation, vtcm_activation_raw + (size_t) ox * 32 * pair_packs * 64,
                                      valid_xi, pair_packs, kp);

        attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation, vtcm_weight_bufs[0], 0, np, kp);
        attn_store_qk_scores(state, scores, (size_t) page_start, vtcm_output, 0, np, ox, N, valid_xi, q_row_offset);
      }
    }
    return;
  }

  // Plan(a) K-reuse general path (M>32, valid_end>ATTN_HMX_KV_BLOCK): read each K
  // chunk ONCE and reuse it across all mp Q M-tiles, instead of re-reading the
  // whole K for every M-tile (the old per-ox page loop below re-read K mp times).
  // All mp Q M-tiles are preloaded + transformed up front into vtcm_activation[ox].
  // Compute is unchanged; K/V DMA bytes drop ~mp x.
  if (M > 32) {
    // All mp Q M-tiles in ONE DMA transfer, then transform each tile out of the shared raw staging.
    attn_hmx_load_activation_dma(&act_desc, (const uint8_t *) q_ptr, vtcm_activation_raw, K * (int) sizeof(int16_t), M,
                                 act_src_stride, pair_packs, 1);
    for (int ox = 0; ox < mp; ++ox) {
      int valid_xi = M - ox * 32;
      if (valid_xi > 32) {
        valid_xi = 32;
      }
      if (valid_xi <= 0) {
        break;
      }
      attn_transform_activation_hmx(vtcm_activation + (size_t) ox * act_tile_elems,
                                    vtcm_activation_raw + (size_t) ox * 32 * pair_packs * 64, valid_xi, pair_packs, kp);
    }

    // Cross-page double buffering: with page_size <= ATTN_HMX_KV_BLOCK a page's K is exactly one weight
    // chunk, so the old per-page loop below fetched it synchronously (wait_after=1) with no overlap at
    // all. Fetch page p+1 while page p computes/stores instead; the weight DMA then hides behind compute.
    if (state->page_size <= ATTN_HMX_KV_BLOCK) {
      AttnPageSpan cur;
      AttnPageSpan nxt;
      int          cur_buf = 0;
      int          have    = attn_next_page_span(state, valid_end, 0, &cur);
      if (have) {
        if (state->async_push != NULL && cur.start + cur.valid > state->async_push_seq_begin) {
          sync_attention_wait_async_push(state);
        }
        attn_hmx_start_weight_dma(weight_descs[cur_buf], state->pastKPages[cur.page], vtcm_weight_bufs[cur_buf], 0,
                                  cur.np, kp, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head, state->n_kv_heads, k_icP,
                                  cur.np, 0);
      }
      while (have) {
        attn_wait_weight_dma();  // cur page's K has landed
        const int nxt_buf   = 1 - cur_buf;
        const int have_next = attn_next_page_span(state, valid_end, cur.page + 1, &nxt);
        if (have_next) {
          if (state->async_push != NULL && nxt.start + nxt.valid > state->async_push_seq_begin) {
            sync_attention_wait_async_push(state);
          }
          attn_hmx_start_weight_dma(weight_descs[nxt_buf], state->pastKPages[nxt.page], vtcm_weight_bufs[nxt_buf], 0,
                                    nxt.np, kp, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head, state->n_kv_heads, k_icP,
                                    nxt.np, 0);
        }
        for (int ox = 0; ox < mp; ++ox) {
          int valid_xi = M - ox * 32;
          if (valid_xi > 32) {
            valid_xi = 32;
          }
          if (valid_xi <= 0) {
            break;
          }
          attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation + (size_t) ox * act_tile_elems,
                                        vtcm_weight_bufs[cur_buf], 0, cur.np, kp);
          attn_store_qk_scores(state, scores, (size_t) cur.start, vtcm_output, 0, cur.np, ox, cur.n, valid_xi,
                               q_row_offset);
        }
        cur     = nxt;
        cur_buf = nxt_buf;
        have    = have_next;
      }
      return;
    }

    for (int page = 0; page < state->page_count; ++page) {
      int page_start = page * state->page_size;
      if (page_start >= valid_end) {
        break;
      }
      int page_valid = valid_end - page_start;
      if (page_valid > state->page_size) {
        page_valid = state->page_size;
      }
      if (page_valid <= 0) {
        continue;
      }
      if (state->async_push != NULL && page_start + page_valid > state->async_push_seq_begin) {
        sync_attention_wait_async_push(state);
      }

      int N  = (page_valid + 31) & ~31;
      int np = N / 32;
      if (np <= 0) {
        continue;
      }
      const uint8_t *b                      = state->pastKPages[page];
      int            current_weight_buf_idx = 0;
      int            first_oy_end           = np_chunk;
      if (first_oy_end > np) {
        first_oy_end = np;
      }
      attn_hmx_start_weight_dma(weight_descs[current_weight_buf_idx], b, vtcm_weight_bufs[current_weight_buf_idx], 0,
                                first_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head, state->n_kv_heads, k_icP,
                                np, 1);

      for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
        int oy_end = oy_start + np_chunk;
        if (oy_end > np) {
          oy_end = np;
        }
        __fp16 *vtcm_weight_current = vtcm_weight_bufs[current_weight_buf_idx];
        int     next_oy_start       = oy_start + np_chunk;
        int     next_weight_buf_idx = 1 - current_weight_buf_idx;
        bool    has_next_chunk      = next_oy_start < np;
        if (has_next_chunk) {
          int next_oy_end = next_oy_start + np_chunk;
          if (next_oy_end > np) {
            next_oy_end = np;
          }
          attn_hmx_start_weight_dma(weight_descs[next_weight_buf_idx], b, vtcm_weight_bufs[next_weight_buf_idx],
                                    next_oy_start, next_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head,
                                    state->n_kv_heads, k_icP, np, 0);
        }

        for (int ox = 0; ox < mp; ++ox) {
          int valid_xi = M - ox * 32;
          if (valid_xi > 32) {
            valid_xi = 32;
          }
          if (valid_xi <= 0) {
            break;
          }
          attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation + (size_t) ox * act_tile_elems,
                                        vtcm_weight_current, oy_start, oy_end, kp);
          attn_store_qk_scores(state, scores, (size_t) page_start, vtcm_output, oy_start, oy_end, ox, N, valid_xi,
                               q_row_offset);
        }

        if (has_next_chunk) {
          dma_wait_for_idle();
          current_weight_buf_idx = next_weight_buf_idx;
        }
      }
    }
    return;
  }

  for (int ox = 0; ox < mp; ++ox) {
    int valid_xi = M - ox * 32;
    if (valid_xi > 32) valid_xi = 32;
    if (valid_xi <= 0) break;

    int roi_width = K * (int)sizeof(int16_t);
    size_t src_offset = (size_t)ox * 32 * act_src_stride;
    attn_hmx_load_activation_dma(&act_desc, (const uint8_t*)q_ptr + src_offset * sizeof(int16_t),
                                 vtcm_activation_raw, roi_width, valid_xi, act_src_stride, pair_packs, 0);

    int qk_prefetched = 0;
    int current_page = 0;
    int current_buf = 0;
    int current_page_start = 0;
    int current_page_valid = 0;
    int current_N = 0;
    int current_np = 0;
    if (M <= 32 && np_chunk >= (state->page_size + 31) / 32 && valid_end > 0 &&
        current_page < state->page_count && current_page * state->page_size < valid_end) {
      current_page_start = current_page * state->page_size;
      current_page_valid = valid_end - current_page_start;
      if (current_page_valid > state->page_size) current_page_valid = state->page_size;
      if (state->async_push != NULL && current_page_start + current_page_valid > state->async_push_seq_begin) {
        sync_attention_wait_async_push(state);
      }
      current_N = (current_page_valid + 31) & ~31;
      current_np = current_N / 32;
      attn_hmx_start_weight_dma(weight_descs[current_buf], state->pastKPages[current_page],
                                vtcm_weight_bufs[current_buf], 0, current_np, kp,
                                ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head, state->n_kv_heads, k_icP, current_np, 0);
      qk_prefetched = current_np > 0;
    }

    attn_transform_activation_hmx(vtcm_activation, vtcm_activation_raw, valid_xi, pair_packs, kp);

    if (qk_prefetched) {
      while (current_np > 0) {
        dma_wait_for_idle();

        int next_page = current_page + 1;
        int next_buf = 1 - current_buf;
        int has_next_page = 0;
        int next_page_valid = 0;
        int next_np = 0;
        if (next_page < state->page_count && next_page * state->page_size < valid_end) {
          int next_page_start = next_page * state->page_size;
          next_page_valid = valid_end - next_page_start;
          if (next_page_valid > state->page_size) next_page_valid = state->page_size;
          if (next_page_valid > 0) {
            if (state->async_push != NULL && next_page_start + next_page_valid > state->async_push_seq_begin) {
              sync_attention_wait_async_push(state);
            }
            int next_N = (next_page_valid + 31) & ~31;
            next_np = next_N / 32;
            attn_prepare_weight_dma_descs(nullptr, weight_descs[next_buf], state->pastKPages[next_page],
                                          vtcm_weight_bufs[next_buf], 0, next_np, kp,
                                          ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head, state->n_kv_heads, k_icP,
                                          next_np);
            dmstart((dma_desc_1d_t*)&weight_descs[next_buf][0]);
            has_next_page = 1;
          }
        }

        __fp16* vtcm_weight_current = vtcm_weight_bufs[current_buf];
        attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation, vtcm_weight_current, 0, current_np, kp);
        // Must honour state->scores_fp16 like the other score stores: a grouped-causal prefill block
        // reaches this M<=32 path when rows <= 32 (a chunk whose page-fast clamp shrinks group_q_rows,
        // or a very short qo_len), and its softmax then reads those scores as fp16 -- an fp32 store here
        // would be read back at the wrong stride. Latent today: no tested prompt reaches it (checked with
        // a one-shot FARF on zh700 / prompt1024), so this is consistency by inspection.
        attn_store_qk_scores(state, scores, (size_t) current_page_start, vtcm_output, 0, current_np, ox, current_N,
                             valid_xi, q_row_offset);

        if (!has_next_page) {
          break;
        }
        current_page = next_page;
        current_page_start = current_page * state->page_size;
        current_page_valid = next_page_valid;
        current_N = (current_page_valid + 31) & ~31;
        current_np = next_np;
        current_buf = next_buf;
      }
      continue;
    }
    if (M <= 32 && np_chunk >= (state->page_size + 31) / 32) {
      continue;
    }

    for (int page = 0; page < state->page_count; ++page) {
      int page_start = page * state->page_size;
      if (page_start >= valid_end) break;
      int page_valid = valid_end - page_start;
      if (page_valid > state->page_size) page_valid = state->page_size;
      if (page_valid <= 0) continue;
      if (state->async_push != NULL && page_start + page_valid > state->async_push_seq_begin) {
        sync_attention_wait_async_push(state);
      }

      int N = (page_valid + 31) & ~31;
      int np = N / 32;
      const uint8_t* b = state->pastKPages[page];
      int current_weight_buf_idx = 0;
      if (np > 0) {
        int first_oy_end = np_chunk;
        if (first_oy_end > np) first_oy_end = np;
        attn_hmx_start_weight_dma(weight_descs[current_weight_buf_idx], b, vtcm_weight_bufs[current_weight_buf_idx],
                                  0, first_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head,
                                  state->n_kv_heads, k_icP, np, 1);
      }

      for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
        int oy_end = oy_start + np_chunk;
        if (oy_end > np) oy_end = np;
        __fp16* vtcm_weight_current = vtcm_weight_bufs[current_weight_buf_idx];
        int next_oy_start = oy_start + np_chunk;
        int next_weight_buf_idx = 1 - current_weight_buf_idx;
        bool has_next_chunk = next_oy_start < np;
        if (has_next_chunk) {
          int next_oy_end = next_oy_start + np_chunk;
          if (next_oy_end > np) next_oy_end = np;
          attn_hmx_start_weight_dma(weight_descs[next_weight_buf_idx], b, vtcm_weight_bufs[next_weight_buf_idx],
                                    next_oy_start, next_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head,
                                    state->n_kv_heads, k_icP, np, 0);
        }

        attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation, vtcm_weight_current, oy_start, oy_end, kp);
        attn_store_qk_scores(state, scores, (size_t) page_start, vtcm_output, oy_start, oy_end, ox, N, valid_xi,
                             q_row_offset);
        if (has_next_chunk) {
          dma_wait_for_idle();
          current_weight_buf_idx = next_weight_buf_idx;
        }
      }
    }
  }

}

void attn_hmx_matmul_pages_sv(const SyncAttentionTaskState* state, __fp16* dst, __fp16* temp_O,
                                     const __fp16* linear_S, int rows, int output_stride, int row_offset, int kv_head,
                                     int valid_end) {
  const int np_chunk = ATTN_HMX_KV_BLOCK_TILES;
  const int M = rows;
  const int N = state->K_dim_padded;
  const int max_kp = (state->page_size + 31) / 32;
  const int max_pair_packs = (state->page_size + 63) / 64;
  const int np = N / 32;
  const int mp             = (M + 31) / 32;

  if (max_pair_packs > ATTN_HMX_MAX_PAIR_PACKS || max_kp > ATTN_HMX_MAX_KP) {
    FARF(ERROR, "attn_hmx_matmul_pages_sv overflow: pair_packs=%d kp=%d", max_pair_packs, max_kp);
    sync_attention_zero_packed_output(dst, rows, output_stride, row_offset, N);
    return;
  }

  if (M <= 32 && valid_end > 0 && valid_end <= ATTN_FIXED_WORKSPACE_KV && np <= ATTN_HMX_KV_BLOCK_TILES) {
    int fused_pages = (valid_end + state->page_size - 1) / state->page_size;
    if (fused_pages > state->page_count) fused_pages = state->page_count;
    if (fused_pages > 0 && fused_pages <= 8) {
      uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
      __fp16* vtcm_weights = (__fp16*)vtcm_seq_alloc(&vtcm_ptr,
                                                     (size_t)fused_pages * np * 32 * max_kp * 32 * sizeof(int16_t));
      __fp16* vtcm_activation_raw = (__fp16*)vtcm_seq_alloc(&vtcm_ptr,
                                                            (size_t)max_pair_packs * 32 * 64 * sizeof(int16_t));
      __fp16* vtcm_activations = (__fp16*)vtcm_seq_alloc(&vtcm_ptr,
                                                         (size_t)fused_pages * max_kp * 32 * 32 * sizeof(int16_t));
      __fp16* vtcm_output = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 32 * 32 * np_chunk * sizeof(int16_t));
      __fp16* vtcm_scales = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 256);

      hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));
      hmx_set_output_scales(vtcm_scales);

      _Alignas(64) dma_desc_2d_t act_desc;
      _Alignas(64) dma_desc_2d_t weight_descs[ATTN_HMX_MAX_WEIGHT_DESCS];
      int page_kp[8];

      for (int page = 0; page < fused_pages; ++page) {
        int page_start = page * state->page_size;
        int page_valid = valid_end - page_start;
        if (page_valid > state->page_size) page_valid = state->page_size;
        if (page_valid <= 0) {
          page_kp[page] = 0;
          continue;
        }
        int K = page_valid;
        int kp = (K + 31) / 32;
        int pair_packs = (K + 63) / 64;
        page_kp[page] = kp;

        int roi_width = K * (int)sizeof(int16_t);
        attn_hmx_load_activation_dma(&act_desc, (const uint8_t*)(linear_S + page_start),
                                     vtcm_activation_raw, roi_width, M, state->N_padded, pair_packs, 1);

        __fp16* page_activation = vtcm_activations + (size_t)page * max_kp * 1024;

        const uint8_t* b = state->pastVPages[page];
        __fp16* page_weight = vtcm_weights + (size_t)page * np * 32 * max_kp * 32;
        attn_hmx_start_weight_dma(weight_descs, b, page_weight, 0, np, kp, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256,
                                  kv_head, state->n_kv_heads, max_kp, np, 0);
        attn_transform_activation_hmx(page_activation, vtcm_activation_raw, M, pair_packs, kp);
        dma_wait_for_idle();
      }

      for (int oy = 0; oy < np; ++oy) {
        __fp16* vtcm_dst = vtcm_output + oy * 1024;
        for (int page = 0; page < fused_pages; ++page) {
          int kp = page_kp[page];
          if (kp <= 0) continue;
          __fp16* page_activation = vtcm_activations + (size_t)page * max_kp * 1024;
          __fp16* page_weight = vtcm_weights + (size_t)page * np * 32 * max_kp * 32;
          int oy_offset = oy * 16 * kp;
          attn_hmx_load_k_tiles(page_activation, page_weight + oy_offset * 64, kp);
        }
        hmx_consume_accumulator_fp16(vtcm_dst);
      }

      attn_store_output_packed_fp16((uint8_t*)dst, vtcm_output, 0, np, 0, M, M, output_stride, row_offset);
      return;
    }
  }

  uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
  __fp16* vtcm_weight = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 2 * (size_t)np_chunk * 32 * max_kp * 32 * sizeof(int16_t));
  // Raw staging sized for all mp M-tiles so a page's S rows load with ONE DMA transfer.
  const size_t sv_raw_tile_elems = (size_t) max_pair_packs * 32 * 64;
  __fp16 *vtcm_activation_raw = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) mp * sv_raw_tile_elems * sizeof(int16_t));
  // Plan(a) V-reuse: for M>32 keep all mp transformed S (softmax) M-tiles resident
  // so each page's V is read ONCE and reused across every M-tile (vs re-reading V
  // per M-tile). mp==1 for the M<=32 paths.
  const size_t sv_act_tile_elems   = (size_t) max_kp * 32 * 32;
  __fp16 *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) mp * sv_act_tile_elems * sizeof(int16_t));
  __fp16* vtcm_output = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 32 * 32 * np_chunk * sizeof(int16_t));
  __fp16* vtcm_scales = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(vtcm_scales);

  _Alignas(64) dma_desc_2d_t act_desc;
  _Alignas(64) dma_desc_2d_t weight_descs[2][ATTN_HMX_MAX_WEIGHT_DESCS];
  size_t weight_buf_elems = (size_t)np_chunk * 32 * max_kp * 32;
  __fp16* vtcm_weight_bufs[2] = {vtcm_weight, vtcm_weight + weight_buf_elems};
  int wrote_output = 0;

  if (M > 32 && valid_end <= ATTN_HMX_KV_BLOCK && N <= ATTN_HMX_KV_BLOCK) {
    for (int page = 0; page < state->page_count; ++page) {
      int page_start = page * state->page_size;
      if (page_start >= valid_end) break;
      int page_valid = valid_end - page_start;
      if (page_valid > state->page_size) page_valid = state->page_size;
      if (page_valid <= 0) continue;

      int K = page_valid;
      int kp = (K + 31) / 32;
      int pair_packs = (K + 63) / 64;
      const uint8_t* b = state->pastVPages[page];
      int accum_page = wrote_output;
      uint8_t* out_ptr = accum_page ? (uint8_t*)temp_O : (uint8_t*)dst;
      int out_stride = accum_page ? 0 : output_stride;
      int out_row_offset = accum_page ? 0 : row_offset;

      attn_hmx_start_weight_dma(weight_descs[0], b, vtcm_weight_bufs[0], 0, np, kp,
                                ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head, state->n_kv_heads, max_kp, np, 1);

      // All mp S M-tiles in ONE DMA (per-transfer latency, not bytes, dominates the DMA wait).
      attn_hmx_load_activation_dma(&act_desc, (const uint8_t *) (linear_S + page_start), vtcm_activation_raw,
                                   K * (int) sizeof(int16_t), M, state->N_padded, pair_packs, 1);

      for (int ox = 0; ox < (M + 31) / 32; ++ox) {
        int valid_xi = M - ox * 32;
        if (valid_xi > 32) valid_xi = 32;
        if (valid_xi <= 0) break;

        attn_transform_activation_hmx(vtcm_activation, vtcm_activation_raw + (size_t) ox * 32 * pair_packs * 64,
                                      valid_xi, pair_packs, kp);

        attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation, vtcm_weight_bufs[0], 0, np, kp);
        attn_store_output_packed_fp16(out_ptr, vtcm_output, 0, np, ox, M, valid_xi, out_stride, out_row_offset);
      }

      if (accum_page) {
        sync_attention_accumulate_packed_output(dst, temp_O, rows, output_stride, row_offset, N);
      } else {
        wrote_output = 1;
      }
    }

    if (!wrote_output) {
      sync_attention_zero_packed_output(dst, rows, output_stride, row_offset, N);
    }
    return;
  }

  // Cross-page double buffering (mirrors the QK path): a page's V is fetched while the previous page's
  // S-transform / HMX / store run, instead of the old synchronous per-page fetch.
  if (state->page_size <= ATTN_HMX_KV_BLOCK && np > 0 && np <= np_chunk) {
    AttnPageSpan cur;
    AttnPageSpan nxt;
    int          cur_buf = 0;
    int          have    = attn_next_page_span(state, valid_end, 0, &cur);
    if (have) {
      attn_hmx_start_weight_dma(weight_descs[cur_buf], state->pastVPages[cur.page], vtcm_weight_bufs[cur_buf], 0, np,
                                (cur.valid + 31) / 32, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head, state->n_kv_heads,
                                max_kp, np, 0);
    }
    while (have) {
      const int K              = cur.valid;
      const int kp             = (K + 31) / 32;
      const int pair_packs     = (K + 63) / 64;
      const int accum_page     = wrote_output;
      uint8_t  *out_ptr        = accum_page ? (uint8_t *) temp_O : (uint8_t *) dst;
      const int out_stride     = accum_page ? 0 : output_stride;
      const int out_row_offset = accum_page ? 0 : row_offset;

      // Loading S also waits out this page's V DMA (attn_hmx_load_activation_dma drains the engine).
      attn_hmx_load_activation_dma(&act_desc, (const uint8_t *) (linear_S + cur.start), vtcm_activation_raw,
                                   K * (int) sizeof(int16_t), M, state->N_padded, pair_packs, 1);

      const int nxt_buf   = 1 - cur_buf;
      const int have_next = attn_next_page_span(state, valid_end, cur.page + 1, &nxt);
      if (have_next) {
        attn_hmx_start_weight_dma(weight_descs[nxt_buf], state->pastVPages[nxt.page], vtcm_weight_bufs[nxt_buf], 0, np,
                                  (nxt.valid + 31) / 32, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head, state->n_kv_heads,
                                  max_kp, np, 0);
      }

      for (int ox = 0; ox < mp; ++ox) {
        int valid_xi = M - ox * 32;
        if (valid_xi > 32) {
          valid_xi = 32;
        }
        if (valid_xi <= 0) {
          break;
        }
        attn_transform_activation_hmx(vtcm_activation + (size_t) ox * sv_act_tile_elems,
                                      vtcm_activation_raw + (size_t) ox * 32 * pair_packs * 64, valid_xi, pair_packs,
                                      kp);
        attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation + (size_t) ox * sv_act_tile_elems,
                                      vtcm_weight_bufs[cur_buf], 0, np, kp);
        attn_store_output_packed_fp16(out_ptr, vtcm_output, 0, np, ox, M, valid_xi, out_stride, out_row_offset);
      }

      if (accum_page) {
        sync_attention_accumulate_packed_output(dst, temp_O, rows, output_stride, row_offset, N);
      } else {
        wrote_output = 1;
      }

      cur     = nxt;
      cur_buf = nxt_buf;
      have    = have_next;
    }

    if (!wrote_output) {
      sync_attention_zero_packed_output(dst, rows, output_stride, row_offset, N);
    }
    return;
  }

  for (int page = 0; page < state->page_count; ++page) {
    int page_start = page * state->page_size;
    if (page_start >= valid_end) break;
    int page_valid = valid_end - page_start;
    if (page_valid > state->page_size) page_valid = state->page_size;
    if (page_valid <= 0) continue;

    int K = page_valid;
    int kp = (K + 31) / 32;
    int pair_packs = (K + 63) / 64;
    const uint8_t* b = state->pastVPages[page];
    int accum_page = wrote_output;
    uint8_t       *out_ptr        = accum_page ? (uint8_t *) temp_O : (uint8_t *) dst;
    int            out_stride     = accum_page ? 0 : output_stride;
    int            out_row_offset = accum_page ? 0 : row_offset;

    // Plan(a) V-reuse: preload + transform all mp S M-tiles for this page so each
    // V chunk is read ONCE and reused across every M-tile (ox inner), instead of
    // re-reading the whole page's V per M-tile. The mp tiles come in ONE DMA transfer.
    attn_hmx_load_activation_dma(&act_desc, (const uint8_t *) (linear_S + page_start), vtcm_activation_raw,
                                 K * (int) sizeof(int16_t), M, state->N_padded, pair_packs, 1);
    for (int ox = 0; ox < mp; ++ox) {
      int valid_xi = M - ox * 32;
      if (valid_xi > 32) valid_xi = 32;
      if (valid_xi <= 0) {
        break;
      }
      attn_transform_activation_hmx(vtcm_activation + (size_t) ox * sv_act_tile_elems,
                                    vtcm_activation_raw + (size_t) ox * 32 * pair_packs * 64, valid_xi, pair_packs, kp);
    }

    int current_weight_buf_idx = 0;
    if (np > 0) {
      int first_oy_end = np_chunk;
      if (first_oy_end > np) {
        first_oy_end = np;
      }
      attn_hmx_start_weight_dma(weight_descs[current_weight_buf_idx], b, vtcm_weight_bufs[current_weight_buf_idx], 0,
                                first_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head, state->n_kv_heads, max_kp,
                                np, 1);
    }

    for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
      int oy_end = oy_start + np_chunk;
      if (oy_end > np) {
        oy_end = np;
      }
      __fp16 *vtcm_weight_current = vtcm_weight_bufs[current_weight_buf_idx];
      int     next_oy_start       = oy_start + np_chunk;
      int     next_weight_buf_idx = 1 - current_weight_buf_idx;
      bool    has_next_chunk      = next_oy_start < np;
      if (has_next_chunk) {
        int next_oy_end = next_oy_start + np_chunk;
        if (next_oy_end > np) {
          next_oy_end = np;
        }
        attn_hmx_start_weight_dma(weight_descs[next_weight_buf_idx], b, vtcm_weight_bufs[next_weight_buf_idx],
                                  next_oy_start, next_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head,
                                  state->n_kv_heads, max_kp, np, 0);
      }

      for (int ox = 0; ox < mp; ++ox) {
        int valid_xi = M - ox * 32;
        if (valid_xi > 32) {
          valid_xi = 32;
        }
        if (valid_xi <= 0) {
          break;
        }
        attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation + (size_t) ox * sv_act_tile_elems,
                                      vtcm_weight_current, oy_start, oy_end, kp);
        attn_store_output_packed_fp16(out_ptr, vtcm_output, oy_start, oy_end, ox, M, valid_xi, out_stride,
                                      out_row_offset);
      }

      if (has_next_chunk) {
        dma_wait_for_idle();
        current_weight_buf_idx = next_weight_buf_idx;
      }
    }

    if (accum_page) {
      sync_attention_accumulate_packed_output(dst, temp_O, rows, output_stride, row_offset, N);
    } else {
      wrote_output = 1;
    }
  }

  if (!wrote_output) {
    sync_attention_zero_packed_output(dst, rows, output_stride, row_offset, N);
  }
}

void attn_hmx_matmul_page_qk_block(const SyncAttentionTaskState* state, float* scores, const __fp16* q_ptr,
                                          int rows, int q_stride, int kv_head, int page_begin, int page_end,
                                          int block_start, int block_valid) {
  const int np_chunk = ATTN_HMX_KV_BLOCK_TILES;
  const int M = rows;
  const int K = state->head_dim;
  const int max_K = state->K_dim_padded;
  const int act_src_stride = q_stride;
  const int mp = (M + 31) / 32;
  const int kp = (K + 31) / 32;
  const int pair_packs = (K + 63) / 64;
  const int k_icP = (max_K + 31) / 32;

  if (page_begin < 0 || page_begin >= state->page_count || page_begin >= page_end || block_valid <= 0) {
    return;
  }
  if (page_end > state->page_count) {
    page_end = state->page_count;
  }
  if (pair_packs > ATTN_HMX_MAX_PAIR_PACKS || kp > ATTN_HMX_MAX_KP) {
    FARF(ERROR, "attn_hmx_matmul_page_qk_block overflow: pair_packs=%d kp=%d", pair_packs, kp);
    return;
  }

  uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
  __fp16* vtcm_weight = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 2 * (size_t)np_chunk * 32 * kp * 32 * sizeof(int16_t));
  __fp16* vtcm_activation_raw = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, (size_t)pair_packs * 32 * 64 * sizeof(int16_t));
  __fp16* vtcm_activation = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, kp * 32 * 32 * sizeof(int16_t));
  __fp16* vtcm_output = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 32 * 32 * np_chunk * sizeof(int16_t));
  __fp16* vtcm_scales = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, attn_hmx_scale_splat(state->scale));
  hmx_set_output_scales(vtcm_scales);

  _Alignas(64) dma_desc_2d_t act_desc;
  _Alignas(64) dma_desc_2d_t weight_descs[2][ATTN_HMX_MAX_WEIGHT_DESCS];
  size_t weight_buf_elems = (size_t)np_chunk * 32 * kp * 32;
  __fp16* vtcm_weight_bufs[2] = {vtcm_weight, vtcm_weight + weight_buf_elems};

  for (int ox = 0; ox < mp; ++ox) {
    int valid_xi = M - ox * 32;
    if (valid_xi > 32) valid_xi = 32;
    if (valid_xi <= 0) break;

    int roi_width = K * (int)sizeof(int16_t);
    size_t src_offset = (size_t)ox * 32 * act_src_stride;
    attn_hmx_load_activation_dma(&act_desc, (const uint8_t*)q_ptr + src_offset * sizeof(int16_t),
                                 vtcm_activation_raw, roi_width, valid_xi, act_src_stride, pair_packs, 0);

    attn_transform_activation_hmx(vtcm_activation, vtcm_activation_raw, valid_xi, pair_packs, kp);

    for (int page = page_begin; page < page_end; ++page) {
      int page_start = page * state->page_size;
      int block_offset = page_start - block_start;
      if (block_offset < 0 || block_offset >= block_valid) continue;
      int page_valid = block_valid - block_offset;
      if (page_valid > state->page_size) page_valid = state->page_size;
      if (page_valid <= 0) continue;
      if (state->async_push != NULL && page_start + page_valid > state->async_push_seq_begin) {
        sync_attention_wait_async_push(state);
      }

      int N = (page_valid + 31) & ~31;
      int np = N / 32;
      const uint8_t* b = state->pastKPages[page];
      int current_weight_buf_idx = 0;
      if (np > 0) {
        int first_oy_end = np_chunk;
        if (first_oy_end > np) first_oy_end = np;
        attn_hmx_start_weight_dma(weight_descs[current_weight_buf_idx], b, vtcm_weight_bufs[current_weight_buf_idx],
                                  0, first_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head,
                                  state->n_kv_heads, k_icP, np, 1);
      }

      for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
        int oy_end = oy_start + np_chunk;
        if (oy_end > np) oy_end = np;
        __fp16* vtcm_weight_current = vtcm_weight_bufs[current_weight_buf_idx];
        int next_oy_start = oy_start + np_chunk;
        int next_weight_buf_idx = 1 - current_weight_buf_idx;
        bool has_next_chunk = next_oy_start < np;
        if (has_next_chunk) {
          int next_oy_end = next_oy_start + np_chunk;
          if (next_oy_end > np) next_oy_end = np;
          attn_hmx_start_weight_dma(weight_descs[next_weight_buf_idx], b, vtcm_weight_bufs[next_weight_buf_idx],
                                    next_oy_start, next_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256, kv_head,
                                    state->n_kv_heads, k_icP, np, 0);
        }

        attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation, vtcm_weight_current, oy_start, oy_end, kp);
        attn_store_output_linear_fp32((uint8_t*)(scores + block_offset), vtcm_output, oy_start, oy_end, ox, N,
                                      valid_xi, state->online_block_size, 0);
        if (has_next_chunk) {
          dma_wait_for_idle();
          current_weight_buf_idx = next_weight_buf_idx;
        }
      }
    }
  }

}

void attn_hmx_matmul_page_sv_block(const SyncAttentionTaskState* state, __fp16* dst, __fp16* page_temp_O,
                                          const __fp16* linear_S, int rows, int kv_head, int page_begin,
                                          int page_end, int block_start, int block_valid) {
  const int np_chunk = ATTN_HMX_KV_BLOCK_TILES;
  const int M = rows;
  const int N = state->K_dim_padded;
  const int max_kp = (state->page_size + 31) / 32;
  const int max_pair_packs = (state->page_size + 63) / 64;
  const int np = N / 32;

  if (page_begin < 0 || page_begin >= state->page_count || page_begin >= page_end || block_valid <= 0) {
    sync_attention_zero_packed_output(dst, rows, rows, 0, N);
    return;
  }
  if (page_end > state->page_count) {
    page_end = state->page_count;
  }
  if (max_pair_packs > ATTN_HMX_MAX_PAIR_PACKS || max_kp > ATTN_HMX_MAX_KP) {
    FARF(ERROR, "attn_hmx_matmul_page_sv_block overflow: pair_packs=%d kp=%d", max_pair_packs, max_kp);
    sync_attention_zero_packed_output(dst, rows, rows, 0, N);
    return;
  }

  uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
  __fp16* vtcm_weight = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 2 * (size_t)np_chunk * 32 * max_kp * 32 * sizeof(int16_t));
  __fp16* vtcm_activation_raw = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, (size_t)max_pair_packs * 32 * 64 * sizeof(int16_t));
  __fp16* vtcm_activation = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, max_kp * 32 * 32 * sizeof(int16_t));
  __fp16* vtcm_output = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 32 * 32 * np_chunk * sizeof(int16_t));
  __fp16* vtcm_scales = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(vtcm_scales);

  _Alignas(64) dma_desc_2d_t act_desc;
  _Alignas(64) dma_desc_2d_t weight_descs[2][ATTN_HMX_MAX_WEIGHT_DESCS];
  size_t weight_buf_elems = (size_t)np_chunk * 32 * max_kp * 32;
  __fp16* vtcm_weight_bufs[2] = {vtcm_weight, vtcm_weight + weight_buf_elems};
  int wrote_output = 0;

  for (int page = page_begin; page < page_end; ++page) {
    int page_start = page * state->page_size;
    int block_offset = page_start - block_start;
    if (block_offset < 0 || block_offset >= block_valid) continue;
    int page_valid = block_valid - block_offset;
    if (page_valid > state->page_size) page_valid = state->page_size;
    if (page_valid <= 0) continue;

    int K = page_valid;
    int kp = (K + 31) / 32;
    int pair_packs = (K + 63) / 64;
    const uint8_t* b = state->pastVPages[page];
    int accum_page = wrote_output;

    for (int ox = 0; ox < (M + 31) / 32; ++ox) {
      int valid_xi = M - ox * 32;
      if (valid_xi > 32) valid_xi = 32;
      if (valid_xi <= 0) break;

      int roi_width = K * (int)sizeof(int16_t);
      size_t src_offset = (size_t)ox * 32 * state->online_block_size + block_offset;
      attn_hmx_load_activation_dma(&act_desc, (const uint8_t*)linear_S + src_offset * sizeof(int16_t),
                                   vtcm_activation_raw, roi_width, valid_xi, state->online_block_size,
                                   pair_packs, 0);

      attn_transform_activation_hmx(vtcm_activation, vtcm_activation_raw, valid_xi, pair_packs, kp);

      int current_weight_buf_idx = 0;
      if (np > 0) {
        int first_oy_end = np_chunk;
        if (first_oy_end > np) first_oy_end = np;
        attn_hmx_start_weight_dma(weight_descs[current_weight_buf_idx], b, vtcm_weight_bufs[current_weight_buf_idx],
                                  0, first_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head,
                                  state->n_kv_heads, max_kp, np, 1);
      }

      uint8_t* out_ptr = accum_page ? (uint8_t*)page_temp_O : (uint8_t*)dst;
      for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
        int oy_end = oy_start + np_chunk;
        if (oy_end > np) oy_end = np;
        __fp16* vtcm_weight_current = vtcm_weight_bufs[current_weight_buf_idx];
        int next_oy_start = oy_start + np_chunk;
        int next_weight_buf_idx = 1 - current_weight_buf_idx;
        bool has_next_chunk = next_oy_start < np;
        if (has_next_chunk) {
          int next_oy_end = next_oy_start + np_chunk;
          if (next_oy_end > np) next_oy_end = np;
          attn_hmx_start_weight_dma(weight_descs[next_weight_buf_idx], b, vtcm_weight_bufs[next_weight_buf_idx],
                                    next_oy_start, next_oy_end, kp, ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256, kv_head,
                                    state->n_kv_heads, max_kp, np, 0);
        }

        attn_hmx_compute_output_tiles(vtcm_output, vtcm_activation, vtcm_weight_current, oy_start, oy_end, kp);
        attn_store_output_packed_fp16(out_ptr, vtcm_output, oy_start, oy_end, ox, M, valid_xi, rows, 0);
        if (has_next_chunk) {
          dma_wait_for_idle();
          current_weight_buf_idx = next_weight_buf_idx;
        }
      }
    }

    if (accum_page) {
      sync_attention_add_packed_rows(dst, page_temp_O, rows, N);
    } else {
      wrote_output = 1;
    }
  }

  if (!wrote_output) {
    sync_attention_zero_packed_output(dst, rows, rows, 0, N);
  }
}
