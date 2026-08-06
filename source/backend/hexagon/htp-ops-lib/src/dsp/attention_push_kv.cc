#include "attention_private.hpp"

static inline void push_k_scalar_rows(const PushKVTaskState* state, int h_kv, int seq_begin, int seq_end);
static inline void push_v_scalar_rows(const PushKVTaskState* state, int h_kv, int seq_begin, int seq_end);
static inline void push_k_hvx_rows(const PushKVTaskState* state, int h_kv, int seq_begin, int seq_end, int worker_index);
static inline void push_v_hvx_rows(const PushKVTaskState* state, int h_kv, int seq_begin, int seq_end);
static inline void push_kv_process_chunk(const PushKVTaskState* state, int chunk_id, int worker_index);
static inline void push_v_clear_seq_tail(const PushKVTaskState* state);
static void push_kv_worker(void* data, int worker_index);

static inline const __fp16* push_v_src_ptr(const PushKVTaskState* state, int h_kv, int local_seq, int dim_base) {
  if (!state->value_c4) {
    return state->V + (size_t)local_seq * state->kv_stride + h_kv * state->head_dim + dim_base;
  }
  const int pack = 64;
  const int channel = h_kv * state->head_dim + dim_base;
  const int token = state->value_token_offset + local_seq;
  return state->V + (size_t)(channel / pack) * state->value_seq_len * pack + (size_t)token * pack + (channel % pack);
}

AEEResult htp_ops_push_kv(uint8_t* pPastK,
                           uint8_t* pPastV,
                           uint8_t* pK,
                           uint8_t* pV,
                           int32_t seq_current,
                           int32_t seq_add,
                           int32_t n_kv_heads,
                           int32_t head_dim,
                           int32_t max_kv_len,
                           int32_t value_c4,
                           int32_t value_token_offset,
                           int32_t value_seq_len) {
  __fp16 *pastKBase = (__fp16 *)(pPastK);
  __fp16 *pastVBase = (__fp16 *)(pPastV);
  const __fp16 *kBase = (const __fp16 *)(pK);
  const __fp16 *vBase = (const __fp16 *)(pV);

  int K_dim = head_dim;
  int past_kv_len = seq_current;
  int insert_len = seq_add;
  int k_icP = (K_dim + 31) / 32;
  int v_ocP = (K_dim + 31) / 32;
  size_t kv_stride = n_kv_heads * head_dim;
  int seq_chunks = (insert_len + ATTN_HMX_KV_BLOCK - 1) / ATTN_HMX_KV_BLOCK;
  HVX_Vector* k_hvx_scratch = NULL;
  (void)max_kv_len;

  int total_tasks = seq_chunks * n_kv_heads;
  int worker_tasks = push_kv_pick_task_count(total_tasks);
  if (past_kv_len > 0 && insert_len <= 64) {
    worker_tasks = 1;
  }
  PushKVTaskState state = {};
  state.pastK = pastKBase;
  state.pastV = pastVBase;
  state.K = kBase;
  state.V = vBase;
  state.past_kv_len = past_kv_len;
  state.insert_len = insert_len;
  state.n_kv_heads = n_kv_heads;
  state.head_dim = head_dim;
  state.k_icP = k_icP;
  state.v_ocP = v_ocP;
  state.seq_chunks = seq_chunks;
  state.kv_stride = kv_stride;
  state.value_c4 = value_c4;
  state.value_token_offset = value_token_offset;
  state.value_seq_len = value_seq_len > 0 ? value_seq_len : insert_len;
  state.k_hvx_scratch_base = k_hvx_scratch;

  if (seq_add == 1) {
    for (int h_kv = 0; h_kv < n_kv_heads; ++h_kv) {
      push_k_scalar_rows(&state, h_kv, past_kv_len, past_kv_len + 1);
      push_v_scalar_rows(&state, h_kv, past_kv_len, past_kv_len + 1);
    }
    return 0;
  }

  unsigned int worker_cap = g_max_num_workers > 0 ? g_max_num_workers : 1;
  k_hvx_scratch = (HVX_Vector*)vtcm_manager_reserve_area("attn_push_k_hvx_scratch",
                                                         (size_t)worker_cap * ATTN_HMX_PUSH_K_HVX_SCRATCH_BYTES, 128);
  state.k_hvx_scratch_base = k_hvx_scratch;

  if (worker_tasks <= 1) {
    for (int chunk_id = 0; chunk_id < total_tasks; ++chunk_id) {
      push_kv_process_chunk(&state, chunk_id, 0);
    }
    push_v_clear_seq_tail(&state);
    return 0;
  }

  worker_pool_job_t job;
  job.fptr = push_kv_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), worker_tasks);
  for (int i = 0; i < worker_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  push_v_clear_seq_tail(&state);

  return 0;
}

static inline void push_k_scalar_rows(const PushKVTaskState* state, int h_kv, int seq_begin, int seq_end) {
  for (int global_seq = seq_begin; global_seq < seq_end; ++global_seq) {
    int local_seq = global_seq - state->past_kv_len;
    int seq_tile = global_seq / 32;
    int seq_inner = global_seq % 32;
    const __fp16* k_src = state->K + (size_t)local_seq * state->kv_stride + h_kv * state->head_dim;
    __fp16* k_dst_row = state->pastK + attn_hmx_k_tile_index(seq_tile, h_kv, 0, state->n_kv_heads, state->k_icP) * 1024;
    for (int dim_tile = 0; dim_tile < state->k_icP; ++dim_tile) {
      __fp16* k_dst = k_dst_row + dim_tile * 1024 + seq_inner * 2;
      int dim_base = dim_tile * 32;
      int remain = state->head_dim - dim_base;
      int valid = remain > 32 ? 32 : remain;
      int dim = 0;
      for (; dim + 1 < valid; dim += 2) {
        memcpy(k_dst + (dim / 2) * 64, k_src + dim_base + dim, sizeof(uint32_t));
      }
      if (dim < valid) {
        k_dst[(dim / 2) * 64] = k_src[dim_base + dim];
      }
    }
  }
}

static inline void push_v_scalar_rows(const PushKVTaskState* state, int h_kv, int seq_begin, int seq_end) {
  for (int global_seq = seq_begin; global_seq < seq_end; ++global_seq) {
    int local_seq = global_seq - state->past_kv_len;
    int seq_tile = global_seq / 32;
    int seq_inner = global_seq % 32;
    int seq_pair = seq_inner / 2;
    int seq_lane = seq_inner % 2;
    for (int dim_tile = 0; dim_tile < state->v_ocP; ++dim_tile) {
      __fp16* v_dst = state->pastV + attn_hmx_v_tile_index(dim_tile, seq_tile, h_kv, state->n_kv_heads, state->v_ocP) * 1024 + seq_pair * 64 + seq_lane;
      int dim_base = dim_tile * 32;
      int remain = state->head_dim - dim_base;
      int valid = remain > 32 ? 32 : remain;
      const __fp16* v_src = push_v_src_ptr(state, h_kv, local_seq, dim_base);
      for (int dim = 0; dim < valid; ++dim) {
        v_dst[dim * 2] = v_src[dim];
      }
    }
  }
}

static inline void push_k_hvx_rows(const PushKVTaskState* state, int h_kv, int seq_begin, int seq_end, int worker_index) {
  int scalar_begin = seq_begin;
  int hvx_begin = (seq_begin + 63) & ~63;
  int hvx_end = seq_end & ~63;
  if (hvx_begin > hvx_end) {
    hvx_begin = hvx_end;
  }
  if (scalar_begin < hvx_begin) {
    push_k_scalar_rows(state, h_kv, scalar_begin, hvx_begin);
  }

  for (int window_seq = hvx_begin; window_seq < hvx_end; window_seq += 64) {
    HVX_Vector* vtcm_tile = state->k_hvx_scratch_base ? (state->k_hvx_scratch_base + worker_index * 128) : nullptr;
    if (!vtcm_tile) {
      push_k_scalar_rows(state, h_kv, window_seq, window_seq + 64);
      continue;
    }

    int seq_tile0 = window_seq / 32;
    int seq_tile1 = seq_tile0 + 1;
    const __fp16* src_base = state->K + (size_t)(window_seq - state->past_kv_len) * state->kv_stride + h_kv * state->head_dim;

    int dim_pair = 0;
    for (; dim_pair + 1 < state->k_icP; dim_pair += 2) {
      int dim_base = dim_pair * 32;
      if (dim_base + 64 > state->head_dim) {
        break;
      }
      for (int yi = 0; yi < 64; ++yi) {
        vmemu(&vtcm_tile[yi]) = vmemu(src_base + (size_t)yi * state->kv_stride + dim_base);
      }
      hvx_transpose_64x64(vtcm_tile);

      __fp16* dst00 = state->pastK + attn_hmx_k_tile_index(seq_tile0, h_kv, dim_pair, state->n_kv_heads, state->k_icP) * 1024;
      __fp16* dst01 = state->pastK + attn_hmx_k_tile_index(seq_tile0, h_kv, dim_pair + 1, state->n_kv_heads, state->k_icP) * 1024;
      __fp16* dst10 = state->pastK + attn_hmx_k_tile_index(seq_tile1, h_kv, dim_pair, state->n_kv_heads, state->k_icP) * 1024;
      __fp16* dst11 = state->pastK + attn_hmx_k_tile_index(seq_tile1, h_kv, dim_pair + 1, state->n_kv_heads, state->k_icP) * 1024;

      for (int xi_pair = 0; xi_pair < 16; ++xi_pair) {
        HVX_Vector v0 = vtcm_tile[xi_pair * 2];
        HVX_Vector v1 = vtcm_tile[xi_pair * 2 + 1];
        HVX_VectorPair pair_left = Q6_W_vdeal_VVR(v1, v0, 64);
        vmemu(dst00 + xi_pair * 64) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(pair_left));
        vmemu(dst10 + xi_pair * 64) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(pair_left));

        HVX_Vector v2 = vtcm_tile[32 + xi_pair * 2];
        HVX_Vector v3 = vtcm_tile[32 + xi_pair * 2 + 1];
        HVX_VectorPair pair_right = Q6_W_vdeal_VVR(v3, v2, 64);
        vmemu(dst01 + xi_pair * 64) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(pair_right));
        vmemu(dst11 + xi_pair * 64) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(pair_right));
      }
    }

    if (dim_pair < state->k_icP) {
      push_k_scalar_rows(state, h_kv, window_seq, window_seq + 64);
    }
    scalar_begin = window_seq + 64;
  }

  if (scalar_begin < seq_end) {
    push_k_scalar_rows(state, h_kv, scalar_begin, seq_end);
  }
}

static inline void push_v_hvx_rows(const PushKVTaskState* state, int h_kv, int seq_begin, int seq_end) {
  int pair_begin = (seq_begin + 1) & ~1;
  int pair_end = seq_end & ~1;
  if (seq_begin < pair_begin) {
    push_v_scalar_rows(state, h_kv, seq_begin, pair_begin);
  }

  for (int global_seq = pair_begin; global_seq < pair_end; global_seq += 2) {
    int local_seq = global_seq - state->past_kv_len;
    int seq_tile = global_seq / 32;
    int seq_pair = (global_seq % 32) / 2;

    int dim_pair = 0;
    for (; dim_pair + 1 < state->v_ocP; dim_pair += 2) {
      int dim_base = dim_pair * 32;
      if (dim_base + 64 > state->head_dim) {
        break;
      }

      __fp16* v_dst0 = state->pastV + attn_hmx_v_tile_index(dim_pair, seq_tile, h_kv, state->n_kv_heads, state->v_ocP) * 1024 + seq_pair * 64;
      __fp16* v_dst1 = state->pastV + attn_hmx_v_tile_index(dim_pair + 1, seq_tile, h_kv, state->n_kv_heads, state->v_ocP) * 1024 + seq_pair * 64;
      const __fp16* v_src0 = push_v_src_ptr(state, h_kv, local_seq, dim_base);
      const __fp16* v_src1 = push_v_src_ptr(state, h_kv, local_seq + 1, dim_base);
      HVX_Vector v0 = vmemu(v_src0);
      HVX_Vector v1 = vmemu(v_src1);
      HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
      vmemu(v_dst0) = Q6_V_lo_W(vp);
      vmemu(v_dst1) = Q6_V_hi_W(vp);
    }

    for (int dim_tile = dim_pair; dim_tile < state->v_ocP; ++dim_tile) {
      __fp16* v_dst = state->pastV + attn_hmx_v_tile_index(dim_tile, seq_tile, h_kv, state->n_kv_heads, state->v_ocP) * 1024 + seq_pair * 64;
      int dim_base = dim_tile * 32;
      int remain = state->head_dim - dim_base;
      int valid = remain > 32 ? 32 : remain;
      const __fp16* v_src0 = push_v_src_ptr(state, h_kv, local_seq, dim_base);
      const __fp16* v_src1 = push_v_src_ptr(state, h_kv, local_seq + 1, dim_base);
      for (int dim = 0; dim < valid; ++dim) {
        v_dst[dim * 2] = v_src0[dim];
        v_dst[dim * 2 + 1] = v_src1[dim];
      }
    }
  }

  if (pair_end < seq_end) {
    push_v_scalar_rows(state, h_kv, pair_end, seq_end);
  }
}

static inline void push_kv_process_chunk(const PushKVTaskState* state, int chunk_id, int worker_index) {
  int seq_chunk = chunk_id / state->n_kv_heads;
  int h_kv = chunk_id % state->n_kv_heads;
  int seq_begin = state->past_kv_len + seq_chunk * ATTN_HMX_KV_BLOCK;
  int seq_end = seq_begin + ATTN_HMX_KV_BLOCK;
  int global_end = state->past_kv_len + state->insert_len;
  if (seq_begin >= global_end) {
    return;
  }
  if (seq_end > global_end) {
    seq_end = global_end;
  }
  push_k_hvx_rows(state, h_kv, seq_begin, seq_end, worker_index);
  push_v_hvx_rows(state, h_kv, seq_begin, seq_end);
}

static inline void push_v_clear_seq_tail(const PushKVTaskState* state) {
  int seq_end = state->past_kv_len + state->insert_len;
  int old_seq_tail_end = (state->past_kv_len + 31) & ~31;
  int seq_tail_end = (seq_end + 31) & ~31;
  if (seq_end == seq_tail_end || old_seq_tail_end == seq_tail_end) {
    return;
  }

  for (int h_kv = 0; h_kv < state->n_kv_heads; ++h_kv) {
    for (int global_seq = seq_end; global_seq < seq_tail_end; ++global_seq) {
      int seq_tile = global_seq / 32;
      int seq_inner = global_seq % 32;
      int seq_pair = seq_inner / 2;
      int seq_lane = seq_inner % 2;
      for (int dim_tile = 0; dim_tile < state->v_ocP; ++dim_tile) {
        __fp16* v_dst = state->pastV + attn_hmx_v_tile_index(dim_tile, seq_tile, h_kv, state->n_kv_heads, state->v_ocP) * 1024 +
                        seq_pair * 64 + seq_lane;
        for (int dim = 0; dim < 32; ++dim) {
          v_dst[dim * 2] = (__fp16)0.0f;
        }
      }
    }
  }
}

static void push_kv_worker(void* data, int worker_index) {
  PushKVTaskState* state = (PushKVTaskState*)data;
  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    int total_tasks = state->seq_chunks * state->n_kv_heads;
    if ((int)task_id >= total_tasks) {
      break;
    }
    push_kv_process_chunk(state, (int)task_id, worker_index);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

int flash_attn_try_single_token_output(uint8_t* pOut, const uint8_t* pV, int32_t qo_len, int32_t seq_current,
                                                     int32_t seq_add, int32_t n_heads, int32_t n_kv_heads,
                                                     int32_t head_dim, int32_t value_c4) {
  (void)value_c4;
  if (pOut == NULL || pV == NULL || qo_len != 1 || seq_current != 0 || seq_add != 1 ||
      n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || (head_dim % 64) != 0 ||
      (n_heads % n_kv_heads) != 0) {
    return 0;
  }

  const __fp16* v = (const __fp16*)pV;
  __fp16* out = (__fp16*)pOut;
  if (n_heads == n_kv_heads) {
    memcpy(out, v, (size_t)n_heads * head_dim * sizeof(__fp16));
    return 1;
  }

  const int gqa_factor = n_heads / n_kv_heads;
  const int packs = head_dim / 64;
  for (int h = 0; h < n_heads; ++h) {
    const int kv_h = h / gqa_factor;
    const __fp16* src = v + (size_t)kv_h * head_dim;
    __fp16* dst = out + (size_t)h * head_dim;
    for (int p = 0; p < packs; ++p) {
      vmemu(dst + p * 64) = vmemu(src + p * 64);
    }
  }
  return 1;
}

AEEResult htp_ops_push_kv_pages(uint8_t** pPastKPages,
                                       uint8_t** pPastVPages,
                                       uint8_t* pK,
                                       uint8_t* pV,
                                       int32_t seq_current,
                                       int32_t seq_add,
                                       int32_t n_kv_heads,
                                       int32_t head_dim,
                                       int32_t page_count,
                                       int32_t page_size,
                                       int32_t value_c4,
                                       int32_t value_token_offset,
                                       int32_t value_seq_len) {
  if (seq_add <= 0) {
    return 0;
  }
  if (pPastKPages == NULL || pPastVPages == NULL || pK == NULL || pV == NULL || page_count <= 0 || page_size <= 0) {
    return AEE_EBADPARM;
  }
  int global_seq = seq_current;
  int remain = seq_add;
  int input_offset = 0;
  size_t kv_stride_bytes = (size_t)n_kv_heads * head_dim * sizeof(__fp16);
  while (remain > 0) {
    int page_index = global_seq / page_size;
    int page_offset = global_seq - page_index * page_size;
    if (page_index < 0 || page_index >= page_count) {
      return AEE_EBADPARM;
    }
    int chunk = page_size - page_offset;
    if (chunk > remain) {
      chunk = remain;
    }
    AEEResult ret = htp_ops_push_kv(pPastKPages[page_index],
                                    pPastVPages[page_index],
                                    pK + (size_t)input_offset * kv_stride_bytes,
                                    value_c4 ? pV : pV + (size_t)input_offset * kv_stride_bytes,
                                    page_offset, chunk, n_kv_heads, head_dim, page_size,
                                    value_c4, value_token_offset + input_offset,
                                    value_seq_len > 0 ? value_seq_len : seq_add);
    if (ret != 0) {
      return ret;
    }
    global_seq += chunk;
    input_offset += chunk;
    remain -= chunk;
  }
  return 0;
}

void push_kv_pages_async_worker(void* data, int worker_index) {
  (void)worker_index;
  AsyncPushKVPagesState* state = (AsyncPushKVPagesState*)data;
  state->status = htp_ops_push_kv_pages(state->pastKPages, state->pastVPages, state->K, state->V, state->seq_current,
                                        state->seq_add, state->n_kv_heads, state->head_dim, state->page_count,
                                        state->page_size, state->value_c4, state->value_token_offset,
                                        state->value_seq_len);
  state->done = 1;
}
