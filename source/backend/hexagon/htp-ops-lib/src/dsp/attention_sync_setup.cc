#include "attention_private.hpp"

static int sync_attention_should_segment_prefill(const SyncAttentionTaskState* state) {
  return state->mask_stride < 0 && !state->decode_grouped && state->qo_len > ATTN_PREFILL_SEGMENT_Q;
}

static int sync_attention_group_q_rows(int qo_len, int gqa_factor) {
  int group_q_rows = 32 / gqa_factor;
  if (group_q_rows < 1) {
    group_q_rows = 1;
  }
  return qo_len < group_q_rows ? qo_len : group_q_rows;
}

static int sync_attention_rows_per_task(int qo_len, int gqa_factor, int decode_grouped) {
  return decode_grouped ? gqa_factor * sync_attention_group_q_rows(qo_len, gqa_factor) : qo_len;
}

int sync_attention_causal_group_q_rows(int qo_len, int seq_len, int head_dim, int gqa_factor, int kv_chunk) {
  const size_t vtcm = (size_t) vtcm_manager_get_vtcm_size();
  const int    g =
    htp_attn_causal_group_q_rows(qo_len, seq_len, head_dim, gqa_factor, (int) g_max_num_workers, vtcm, kv_chunk);
  if (seq_len <= ATTN_GROUPED_SCORES_VTCM_KV) {
    // The picker already verified scores + kernel for the groups it tried; when nothing fit it returned the
    // full-size group, so still check that group's kernel scratch before committing to it.
    return htp_attn_vtcm_kernel_bytes(gqa_factor * g, head_dim, kv_chunk) <= vtcm ? g : 0;
  }
  // Past 2048 the score region may not fit VTCM at all. That only costs the QK store and the softmax read
  // a DDR round trip, so grouping is still worth taking -- but the HMX kernel scratch has to fit, because
  // it grows with the rows in the group and with the KV chunk, and vtcm_seq_alloc does not bounds-check.
  if (g < 16) {
    return htp_attn_vtcm_kernel_bytes(gqa_factor * g, head_dim, kv_chunk) <= vtcm ? g : 0;
  }
  const int q_max = qo_len < HTP_ATTN_CAUSAL_GROUP_Q_ROWS_MAX ? qo_len : HTP_ATTN_CAUSAL_GROUP_Q_ROWS_MAX;
  int       g_ddr = q_max;
  while (g_ddr >= 16 && htp_attn_vtcm_kernel_bytes(gqa_factor * g_ddr, head_dim, kv_chunk) > vtcm) {
    g_ddr >>= 1;
  }
  if (g_ddr < 16) {
    return 0;  // no group fits the kernel; the caller keeps the page-by-page path
  }
  // The largest group wins for K/V traffic (it is read once per block, so traffic scales as 1/group), and
  // keeping the scores in VTCM saves a DDR round trip whose cost does not depend on the group. Prefer the
  // VTCM-score group only when it is at most one halving below the best DDR one -- one more halving of the
  // group doubles the K/V re-reads and measured slower (2B pp4096: g=16/VTCM 2294 t/s vs g=64/DDR 2374),
  // while for 0.6B the one-halving trade is a large win (g=32/VTCM 2169 vs g=16/VTCM 1506).
  const int workers = g_max_num_workers > 0 ? (int) g_max_num_workers : 1;
  if (g >= 16 && (long long) g * 2 >= (long long) g_ddr &&
      htp_attn_vtcm_scores_fit(gqa_factor * g, seq_len, head_dim, kv_chunk, workers, vtcm)) {
    return g;
  }
  return g_ddr;
}

static int sync_attention_can_group_causal_shape(int qo_len, int n_heads, int n_kv_heads, int head_dim,
                                                 int mask_stride) {
  if (!MNN_ATTENTION_GROUP_CAUSAL_PREFILL || mask_stride >= 0 || qo_len <= 8 || head_dim % 64 != 0 ||
      n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
    return 0;
  }
  const int gqa_factor = n_heads / n_kv_heads;
  // The grouped packing packs rows = q_count * gqa_factor and HMX pads the partial 32-row tile,
  // so nothing here needs gqa_factor to divide a tile. The old <= 4 bound was a tuning boundary
  // and excluded 5..8 (Qwen2.5-7B 28/4=7, Qwen2.5-14B/32B 40/8=5, LLaMA3-70B 64/8=8) from the
  // grouped-causal prefill path entirely: on v81 MNN-Pocket-0.3B (14/2=7) that fallback cost 2.8x
  // the KV DMA traffic and 42.5 -> 14.2 ms of attention per pp512 prefill (pp512 6031 -> 9250 t/s),
  // with bit-identical greedy output. Decode grouping is unaffected (it has no such bound).
  return gqa_factor > 1 && gqa_factor <= 8;
}

static int sync_attention_worker_count(int total_tasks, int decode_grouped, int qo_len) {
  int n_tasks = sync_attention_pick_task_count(total_tasks);
  if (decode_grouped && qo_len == 1 && n_tasks > 2) {
    n_tasks = 2;
  }
  return n_tasks;
}

static void sync_attention_configure_tasks(SyncAttentionTaskState* state, int n_heads, int* task_rows,
                                           int* n_tasks) {
  const int seq_add = state->N - state->seq_current;
  const int causal_grouped = sync_attention_can_group_causal(state->qo_len, state->seq_current, seq_add, n_heads,
                                                             state->n_kv_heads, state->head_dim, state->N,
                                                             state->mask_stride, state->workspace_kv, state->kv_chunk);
  const int decode_grouped = causal_grouped ||
      sync_attention_can_group_decode(state->qo_len, n_heads, state->n_kv_heads, state->head_dim, state->N,
                                      state->mask_stride);
  const int total_tasks = decode_grouped ? state->n_kv_heads : n_heads;

  *task_rows            = causal_grouped ?
                            state->gqa_factor * sync_attention_causal_group_q_rows(state->qo_len, state->N, state->head_dim,
                                                                                   state->gqa_factor, state->kv_chunk) :
                            sync_attention_rows_per_task(state->qo_len, state->gqa_factor, decode_grouped);
  *n_tasks = sync_attention_worker_count(total_tasks, decode_grouped, state->qo_len);
  state->total_heads = total_tasks;
  state->decode_grouped = decode_grouped;
}

static int sync_attention_init_common_state(SyncAttentionTaskState *state, __fp16 *O, const __fp16 *Q,
                                            const float *mask, uint8_t *workspace, int qo_len, int seq_current,
                                            int seq_add, int n_heads, int n_kv_heads, int head_dim, float scale,
                                            int mask_stride, int *task_rows, int *n_tasks, int workspace_kv,
                                            int kv_chunk) {
  if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
    return -1;
  }
  const int gqa_factor = n_heads / n_kv_heads;
  const int N = seq_current + seq_add;
  const int K_dim_padded = (head_dim + 31) / 32 * 32;

  memset(state, 0, sizeof(*state));
  state->task_id = 0;
  state->total_heads = n_heads;
  state->gqa_factor = gqa_factor;
  state->qo_len = qo_len;
  state->qo_total_len = qo_len;
  state->q_offset = 0;
  state->n_kv_heads = n_kv_heads;
  state->head_dim = head_dim;
  state->mask_stride = mask_stride;
  state->seq_current = seq_current;
  state->N = N;
  state->N_padded = (N + 31) / 32 * 32;
  state->K_dim_padded = K_dim_padded;
  state->k_icP = (head_dim + 31) / 32;
  state->v_ocP = (head_dim + 31) / 32;
  state->qo_stride = (size_t)n_heads * head_dim;
  state->workspace_base = workspace;
  state->workspace_kv       = workspace_kv;
  state->kv_chunk           = kv_chunk;
  state->kernel_vtcm_base   = (uint8_t *) vtcm_manager_get_vtcm_base();
  state->scores_vtcm        = NULL;
  state->scores_vtcm_stride = 0;
  state->act_vtcm           = NULL;
  state->act_vtcm_stride    = 0;
  state->activation_in_vtcm = 0;
  state->O = O;
  state->Q = Q;
  state->mask = mask;
  state->scale = scale;
  sync_attention_configure_tasks(state, n_heads, task_rows, n_tasks);
  return 0;
}

static void sync_attention_run_segmented_prefill(SyncAttentionTaskState* state) {
  const int total_q = state->qo_len;
  const int n_heads = state->n_kv_heads * state->gqa_factor;
  const int segment_count = (total_q + ATTN_PREFILL_SEGMENT_Q - 1) / ATTN_PREFILL_SEGMENT_Q;
  const int task_rows = ATTN_PREFILL_SEGMENT_Q;
  state->task_id = 0;
  state->total_heads = n_heads * segment_count;
  state->decode_grouped = 0;
  state->prefill_segment_q = ATTN_PREFILL_SEGMENT_Q;
  state->prefill_n_heads = n_heads;
  state->worker_workspace_bytes = state->page_count > 0
      ? (state->online_pages
             ? sync_attention_page_block_workspace_bytes(task_rows, state->online_block_size, state->K_dim_padded)
             : sync_attention_page_head_workspace_bytes(task_rows, state->N, state->K_dim_padded))
      : sync_attention_head_workspace_bytes(task_rows, state->N);
  sync_attention_run_tasks(state, sync_attention_worker_count(n_heads, 0, ATTN_PREFILL_SEGMENT_Q));
  state->prefill_segment_q = 0;
  state->prefill_n_heads = 0;
}

int sync_attention(__fp16 *restrict O, const __fp16 *restrict Q, const float *restrict mask, uint8_t *workspace,
                   __fp16 *pastK, __fp16 *pastV, int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads,
                   int head_dim, float scale, int mask_stride, int workspace_kv, int kv_chunk) {
  SyncAttentionTaskState state;
  int task_rows = 0;
  int n_tasks = 0;
  if (sync_attention_init_common_state(&state, O, Q, mask, workspace, qo_len, seq_current, seq_add, n_heads, n_kv_heads,
                                       head_dim, scale, mask_stride, &task_rows, &n_tasks, workspace_kv,
                                       kv_chunk) != 0) {
    return -1;
  }
  state.pastK = pastK;
  state.pastV = pastV;
  state.worker_workspace_bytes = sync_attention_head_workspace_bytes(task_rows, state.N);
  state.online_pages = 0;

  if (sync_attention_should_segment_prefill(&state)) {
    sync_attention_run_segmented_prefill(&state);
  } else {
    n_tasks = sync_attention_finalize_causal_tasks(&state);
    sync_attention_run_tasks(&state, n_tasks);
  }

  return 0;
}

int sync_attention_pages(__fp16 *restrict O, const __fp16 *restrict Q, const float *restrict mask, uint8_t *workspace,
                         uint8_t **pastKPages, uint8_t **pastVPages, int qo_len, int seq_current, int seq_add,
                         int n_heads, int n_kv_heads, int head_dim, float scale, int mask_stride, int page_count,
                         int page_size, AsyncPushKVPagesState *async_push, int allow_online_pages, int workspace_kv,
                         int kv_chunk) {
  SyncAttentionTaskState state;
  int task_rows = 0;
  int n_tasks = 0;
  if (sync_attention_init_common_state(&state, O, Q, mask, workspace, qo_len, seq_current, seq_add, n_heads, n_kv_heads,
                                       head_dim, scale, mask_stride, &task_rows, &n_tasks, workspace_kv,
                                       kv_chunk) != 0) {
    return -1;
  }
  int online_block_pages = 1;
  int online_block_size = page_size;
  int online_pages =
    sync_attention_use_online_pages(allow_online_pages, head_dim, page_size, state.N, mask_stride, workspace_kv);
  state.worker_workspace_bytes = online_pages
      ? sync_attention_page_block_workspace_bytes(task_rows, online_block_size, state.K_dim_padded)
      : sync_attention_page_head_workspace_bytes(task_rows, state.N, state.K_dim_padded);
  state.pastKPages = pastKPages;
  state.pastVPages = pastVPages;
  state.page_count = page_count;
  state.page_size = page_size;
  state.online_block_pages = online_block_pages;
  state.online_block_size = online_block_size;
  state.async_push = async_push;
  state.async_push_seq_begin = async_push != NULL ? async_push->seq_current : seq_current;
  state.online_pages = online_pages;
  // The online path's grouped entry (sync_attention_process_online_pages with decode_grouped=1) lays out
  // exactly gqa_factor rows and passes global_q=0, i.e. it assumes a single query position -- it is a
  // decode-only layout. Grouped-causal prefill must therefore fall back to the per-head online entry,
  // which sizes rows=qo_len and passes the real global_q.
  //
  // Today the two gates make this unreachable by construction: can_group_causal requires
  // seq_len <= sync_attention_fixed_kv(workspace_kv) while use_online_pages requires the opposite. That is
  // an implicit coupling between two unrelated predicates, and widening either one silently computes only
  // the first token of the prompt (measured: forcing it on at pp512 produced garbage output at a pp512
  // reading indistinguishable from skipping attention altogether). Degrade explicitly instead.
  if (online_pages && state.decode_grouped && state.qo_len > 1) {
    state.decode_grouped = 0;
    state.total_heads    = n_heads;
  }
  // fp16 QK scores only in the grouped-causal prefill path (decode_grouped && qo_len>1).
  state.scores_fp16            = ATTN_SCORES_FP16 && state.decode_grouped && state.qo_len > 1;
  // Candidate for tile-layout scores; sync_attention_try_vtcm_workspace clears it again if the score
  // buffer could not be placed in VTCM, because the HMX accumulator store only reaches VTCM.
  state.scores_tile            = ATTN_SCORES_TILE && state.scores_fp16;

  if (sync_attention_should_segment_prefill(&state)) {
    sync_attention_run_segmented_prefill(&state);
  } else {
    // Must come after the page fields above: the q-block size depends on page_count / page_size.
    n_tasks = sync_attention_finalize_causal_tasks(&state);
    sync_attention_try_vtcm_workspace(&state, task_rows);
    sync_attention_run_tasks(&state, n_tasks);
  }

  return 0;
}

size_t sync_attention_head_workspace_bytes(int qo_len, int seq_len) {
  int N_padded = (seq_len + 31) / 32 * 32;

  size_t offset = 0;
  offset += (size_t)qo_len * N_padded * sizeof(float);
  offset = attn_align_128(offset);
  offset += (size_t)qo_len * N_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  return offset;
}

size_t sync_attention_page_head_workspace_bytes(int qo_len, int seq_len, int head_dim) {
  int N_padded = (seq_len + 31) / 32 * 32;
  int head_dim_padded = (head_dim + 31) / 32 * 32;

  size_t offset = 0;
  offset += (size_t)qo_len * N_padded * sizeof(float);
  offset = attn_align_128(offset);
  offset += (size_t)qo_len * N_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  offset += (size_t)qo_len * head_dim_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  return offset;
}

size_t sync_attention_page_block_workspace_bytes(int qo_len, int block_len, int head_dim) {
  int block_padded = (block_len + 31) / 32 * 32;
  int head_dim_padded = (head_dim + 31) / 32 * 32;

  size_t offset = 0;
  offset += (size_t)qo_len * block_padded * sizeof(float);
  offset = attn_align_128(offset);
  offset += (size_t)qo_len * block_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  offset += (size_t)qo_len * head_dim_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  offset += (size_t)qo_len * head_dim_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  offset += 3 * (size_t)qo_len * sizeof(float);
  offset = attn_align_128(offset);
  return offset;
}

int sync_attention_can_group_decode(int qo_len, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                                  int mask_stride) {
  int N_padded = (seq_len + 31) / 32 * 32;
  if (head_dim % 64 != 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0 ||
      n_heads / n_kv_heads <= 1 || N_padded * (int)sizeof(float) < head_dim * (int)sizeof(__fp16)) {
    return 0;
  }
  if (qo_len == 1) {
    return 1;
  }
  return qo_len <= 8 && mask_stride < 0 && qo_len * (n_heads / n_kv_heads) <= 32;
}

int sync_attention_can_group_causal(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads, int head_dim,
                                    int seq_len, int mask_stride, int workspace_kv, int kv_chunk) {
  // The grouped path materialises a full N-column score row, so it can only run within the columns the
  // host actually allocated for this command -- not the compile-time ATTN_FIXED_WORKSPACE_KV, which a
  // reused command may long have outgrown.
  if (!sync_attention_can_group_causal_shape(qo_len, n_heads, n_kv_heads, head_dim, mask_stride) || seq_add != qo_len ||
      seq_current < 0 || seq_len != seq_current + seq_add || seq_len > sync_attention_fixed_kv(workspace_kv)) {
    return 0;
  }
  const int N_padded = (seq_len + 31) / 32 * 32;
  if (N_padded < head_dim) {
    return 0;
  }
  // The group picker also decides whether the HMX kernel scratch fits VTCM at this shape; 0 means no
  // group does, and then the page-by-page path has to handle the op.
  return sync_attention_causal_group_q_rows(qo_len, seq_len, head_dim, n_heads / n_kv_heads, kv_chunk) > 0;
}

int sync_attention_task_rows(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads, int head_dim,
                             int seq_len, int mask_stride, int workspace_kv, int kv_chunk) {
  if (sync_attention_can_group_causal(qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, seq_len, mask_stride,
                                      workspace_kv, kv_chunk)) {
    int gqa_factor = n_heads / n_kv_heads;
    return gqa_factor * sync_attention_causal_group_q_rows(qo_len, seq_len, head_dim, gqa_factor, kv_chunk);
  }
  if (sync_attention_can_group_decode(qo_len, n_heads, n_kv_heads, head_dim, seq_len, mask_stride)) {
    return sync_attention_rows_per_task(qo_len, n_heads / n_kv_heads, 1);
  }
  if (mask_stride < 0 && qo_len > ATTN_PREFILL_SEGMENT_Q) {
    return ATTN_PREFILL_SEGMENT_Q;
  }
  return qo_len;
}

int sync_attention_total_tasks(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads, int head_dim,
                               int seq_len, int mask_stride, int workspace_kv, int kv_chunk) {
  if (sync_attention_can_group_causal(qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, seq_len, mask_stride,
                                      workspace_kv, kv_chunk)) {
    return n_kv_heads;
  }
  if (sync_attention_can_group_decode(qo_len, n_heads, n_kv_heads, head_dim, seq_len, mask_stride)) {
    return n_kv_heads;
  }
  return n_heads;
}

// Place the per-worker QK score buffers in VTCM when they fit, and push the HMX kernels' bump base above
// them. QK writes `scores` and the worker softmax reads them back, so this removes two DDR passes over
// rows*N_padded per q block.
//
// Only `scores` moves. `linear_S` (and `packed_Q`, which aliases it) stay in DDR because the QK and SV
// activation loads fetch them with the DMA engine, which is for DDR<->VTCM transfers -- moving the whole
// workspace into VTCM makes those DMAs read garbage and the model emits '!!!!' at full speed.
//
// No reservation is needed because graph ops execute in sequence, so attention owns VTCM for its whole
// duration; the existing code already relies on that (attention and the prefill matmul both bump from
// vtcm_base). Within the op the HMX jobs are serialised by the HMX unit and keep sharing one kernel
// scratch region; only the score buffers need to be per worker, because the softmax that reads them runs
// on the worker threads, outside the HMX job.
//
// The headroom left for the kernels comes from htp_attn_vtcm_kernel_bytes(), computed per shape rather
// than guessed -- see that function for why a flat reserve was wrong.
void sync_attention_try_vtcm_workspace(SyncAttentionTaskState *state, int task_rows) {
  uint8_t *vtcm_base         = (uint8_t *) vtcm_manager_get_vtcm_base();
  state->scores_vtcm         = NULL;
  state->scores_vtcm_stride  = 0;
  state->act_vtcm            = NULL;
  state->act_vtcm_stride     = 0;
  state->activation_in_vtcm  = 0;
  // Tile-layout scores need the HMX accumulator store to land in the score buffer, which only works if
  // that buffer ends up in VTCM. Withdraw the candidate until it does.
  // Tile layout covers whole 32-row M-tiles, and the activation buffers are sized for task_rows rows
  // (by the host for the DDR one), so a row count that is not a multiple of 32 would need padding the
  // caller has not allocated. gqa_factor * group_q_rows is a multiple of 32 for every shape measured;
  // anything else falls back to row-major.
  //
  // Tile layout is only readable by the all-page SV fusion, which holds at most ATTN_HMX_MAX_FUSED_PAGES
  // pages. A q block whose causal end passes that many pages falls to the row-major SV, which refuses
  // tile-layout P -- so the layout has to stay off whenever any block can reach that far. This used to be
  // implied by the 2048 workspace cap; it is explicit now that the grouped path covers 4096.
  const long long fused_kv         = (long long) ATTN_HMX_MAX_FUSED_PAGES * (long long) state->page_size;
  // The fusion also requires np = K_dim_padded / 32 <= ATTN_HMX_KV_BLOCK_TILES (attention_hmx.cc:745),
  // i.e. head_dim <= 256; beyond that the fusion is skipped and the row-major SV rejects a tile-layout P.
  const int       want_scores_tile = state->scores_tile && (task_rows % 32) == 0 && (long long) state->N <= fused_kv &&
                               (state->K_dim_padded / 32) <= ATTN_HMX_KV_BLOCK_TILES;
  state->scores_tile         = 0;
  // Restricted to the grouped-causal prefill path: those are the QK/SV branches whose activation staging
  // has been taught to transform out of VTCM directly (attn_stage_activation). Any other path would hand
  // a VTCM source to the DMA engine.
  if (vtcm_base == NULL || task_rows <= 0 || state->page_count <= 0 || state->online_pages || !state->decode_grouped ||
      state->qo_len <= 1) {
    return;
  }
  unsigned int workers = g_max_num_workers;
  if (workers < 1) {
    workers = 1;
  }
  const size_t vtcm_total   = (size_t) vtcm_manager_get_vtcm_size();
  // Leave the kernels exactly what their bump allocations need for this shape, from the same estimate the
  // q-group picker uses, so the two cannot disagree about what fits.
  const size_t kernel_bytes = htp_attn_vtcm_kernel_bytes(task_rows, state->head_dim, state->kv_chunk);
  if (vtcm_total <= kernel_bytes) {
    return;
  }
  const size_t budget = vtcm_total - kernel_bytes;

  // Scores go in first: they carry most of the win (an HMX store plus an HVX softmax read per q block),
  // and are the smaller of the two regions when scores_fp16 is set, since that path only uses the lower
  // half of the fp32-sized buffer. Strides are indexed by worker_index, which can reach
  // g_max_num_workers - 1 regardless of how many tasks were submitted.
  const size_t score_elem   = state->scores_fp16 ? sizeof(__fp16) : sizeof(float);
  // Tile-layout scores cover whole 32-row M-tiles: the HMX store always writes 32 rows, even for the
  // last partial one, so the buffer has to hold them or the store runs past its end.
  const size_t score_rows   = want_scores_tile ? (size_t) ((task_rows + 31) / 32 * 32) : (size_t) task_rows;
  const size_t score_stride = attn_align_128(score_rows * state->N_padded * score_elem);
  if ((size_t) workers * score_stride > budget) {
    return;  // keep everything in the DDR workspace the host allocated
  }
  state->scores_vtcm        = vtcm_base;
  state->scores_vtcm_stride = score_stride;
  state->scores_tile        = want_scores_tile;
  size_t used               = (size_t) workers * score_stride;

  // Then linear_S + temp_O, laid out exactly like sync_attention_page_offsets so the call site splits them
  // the same way. On the tile path linear_S aliases the score region (P is written over S in place), so
  // only temp_O is left -- which is what lets this fit at seq 2048, where the pair needed 8.7 MB.
  const size_t linear_bytes =
    state->scores_tile ? 0 : attn_align_128((size_t) task_rows * state->N_padded * sizeof(__fp16));
  const size_t act_stride =
    linear_bytes + attn_align_128((size_t) task_rows * ((state->head_dim + 31) / 32 * 32) * sizeof(__fp16));
  if (used + (size_t) workers * act_stride <= budget) {
    state->act_vtcm           = vtcm_base + used;
    state->act_vtcm_stride    = act_stride;
    state->activation_in_vtcm = 1;
    used += (size_t) workers * act_stride;
  }
  state->kernel_vtcm_base = vtcm_base + used;
}

int sync_attention_use_online_pages(int allow_online_pages, int head_dim, int page_size, int seq_len, int mask_stride,
                                    int workspace_kv) {
  // Same bound as can_group_causal: a row that does not fit the columns the host allocated for this
  // command has to be walked page by page, no matter what the compile-time limit says.
  const int fixed_kv = sync_attention_fixed_kv(workspace_kv);
  return allow_online_pages && (head_dim % 64) == 0 && (page_size % 32) == 0 &&
         (seq_len > fixed_kv || mask_stride > fixed_kv);
}

int sync_attention_pick_task_count(int total_heads) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap < 1) {
    worker_cap = 1;
  }
  if (total_heads < 2 || worker_cap < 2) {
    return 1;
  }
  return total_heads < (int)worker_cap ? total_heads : (int)worker_cap;
}

int push_kv_pick_task_count(int total_tasks) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap < 1) {
    worker_cap = 1;
  }
  if (total_tasks < 2 || worker_cap < 2) {
    return 1;
  }
  return total_tasks < (int)worker_cap ? total_tasks : (int)worker_cap;
}
