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

int sync_attention_causal_group_q_rows(int qo_len) {
  int group_q_rows = MNN_ATTENTION_HMX_COMBINE_DECODE ? 64 : ATTN_PREFILL_SEGMENT_Q;
  return qo_len < group_q_rows ? qo_len : group_q_rows;
}

static int sync_attention_can_group_causal_shape(int qo_len, int n_heads, int n_kv_heads, int head_dim,
                                                 int mask_stride) {
  if (mask_stride >= 0 || qo_len <= 8 || head_dim % 64 != 0 ||
      n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
    return 0;
  }
  if (qo_len > ATTN_PREFILL_SEGMENT_Q && !MNN_ATTENTION_HMX_COMBINE_DECODE) {
    return 0;
  }
  const int gqa_factor = n_heads / n_kv_heads;
  return gqa_factor > 1 && gqa_factor <= 4;
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
                                                             state->mask_stride);
  const int decode_grouped = causal_grouped ||
      sync_attention_can_group_decode(state->qo_len, n_heads, state->n_kv_heads, state->head_dim, state->N,
                                      state->mask_stride);
  const int total_tasks = decode_grouped ? state->n_kv_heads : n_heads;

  *task_rows = causal_grouped ? state->gqa_factor * sync_attention_causal_group_q_rows(state->qo_len) :
      sync_attention_rows_per_task(state->qo_len, state->gqa_factor, decode_grouped);
  *n_tasks = sync_attention_worker_count(total_tasks, decode_grouped, state->qo_len);
  state->total_heads = total_tasks;
  state->decode_grouped = decode_grouped;
}

static int sync_attention_init_common_state(SyncAttentionTaskState* state, __fp16* O, const __fp16* Q,
                                            const float* mask, uint8_t* workspace, int qo_len, int seq_current,
                                            int seq_add, int n_heads, int n_kv_heads, int head_dim, float scale,
                                            int mask_stride, int* task_rows, int* n_tasks) {
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
  state->O = O;
  state->Q = Q;
  state->mask = mask;
  state->scale = scale;
  sync_attention_configure_tasks(state, n_heads, task_rows, n_tasks);
  return 0;
}

static void sync_attention_run_segmented_prefill(SyncAttentionTaskState* state) {
  const __fp16* q_base = state->Q;
  const int total_q = state->qo_len;
  const int base_seq_current = state->seq_current;
  const int n_heads = state->n_kv_heads * state->gqa_factor;
  for (int q_offset = 0; q_offset < total_q; q_offset += ATTN_PREFILL_SEGMENT_Q) {
    int segment_q = total_q - q_offset;
    if (segment_q > ATTN_PREFILL_SEGMENT_Q) {
      segment_q = ATTN_PREFILL_SEGMENT_Q;
    }
    SyncAttentionTaskState segment = *state;
    segment.task_id = 0;
    segment.qo_len = segment_q;
    segment.q_offset = q_offset;
    segment.qo_total_len = total_q;
    segment.seq_current = base_seq_current + q_offset;
    segment.N = base_seq_current + q_offset + segment_q;
    segment.N_padded = (segment.N + 31) / 32 * 32;
    segment.Q = q_base + (size_t)q_offset * state->qo_stride;
    int task_rows = segment_q;
    int n_tasks = sync_attention_worker_count(n_heads, 0, segment_q);
    segment.total_heads = n_heads;
    segment.decode_grouped = 0;
    segment.worker_workspace_bytes = state->page_count > 0
        ? (segment.online_pages ? sync_attention_page_block_workspace_bytes(task_rows, segment.online_block_size, segment.K_dim_padded)
                                : sync_attention_page_head_workspace_bytes(task_rows, segment.N, segment.K_dim_padded))
        : sync_attention_head_workspace_bytes(task_rows, segment.N);
    sync_attention_run_tasks(&segment, n_tasks);
  }
}

int sync_attention(__fp16 *restrict O, const __fp16 *restrict Q, const float *restrict mask, uint8_t *workspace, __fp16 *pastK,
                   __fp16 *pastV, int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads, int head_dim,
                   float scale, int mask_stride) {
  SyncAttentionTaskState state;
  int task_rows = 0;
  int n_tasks = 0;
  if (sync_attention_init_common_state(&state, O, Q, mask, workspace, qo_len, seq_current, seq_add,
                                       n_heads, n_kv_heads, head_dim, scale, mask_stride, &task_rows,
                                       &n_tasks) != 0) {
    return -1;
  }
  state.pastK = pastK;
  state.pastV = pastV;
  state.worker_workspace_bytes = sync_attention_head_workspace_bytes(task_rows, state.N);
  state.online_pages = 0;

  if (sync_attention_should_segment_prefill(&state)) {
    sync_attention_run_segmented_prefill(&state);
  } else {
    sync_attention_run_tasks(&state, n_tasks);
  }

  return 0;
}

int sync_attention_pages(__fp16 *restrict O, const __fp16 *restrict Q, const float *restrict mask, uint8_t *workspace,
                         uint8_t **pastKPages, uint8_t **pastVPages, int qo_len, int seq_current, int seq_add,
                         int n_heads, int n_kv_heads, int head_dim, float scale, int mask_stride,
                         int page_count, int page_size, AsyncPushKVPagesState* async_push, int allow_online_pages) {
  SyncAttentionTaskState state;
  int task_rows = 0;
  int n_tasks = 0;
  if (sync_attention_init_common_state(&state, O, Q, mask, workspace, qo_len, seq_current, seq_add,
                                       n_heads, n_kv_heads, head_dim, scale, mask_stride, &task_rows,
                                       &n_tasks) != 0) {
    return -1;
  }
  int online_block_pages = 1;
  int online_block_size = page_size;
  int online_pages = sync_attention_use_online_pages(allow_online_pages, head_dim, page_size, state.N, mask_stride);
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

  if (sync_attention_should_segment_prefill(&state)) {
    sync_attention_run_segmented_prefill(&state);
  } else {
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

int sync_attention_can_group_causal(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads,
                                                   int head_dim, int seq_len, int mask_stride) {
  if (!sync_attention_can_group_causal_shape(qo_len, n_heads, n_kv_heads, head_dim, mask_stride) ||
      seq_add != qo_len || seq_current < 0 || seq_len != seq_current + seq_add ||
      seq_len > ATTN_FIXED_WORKSPACE_KV) {
    return 0;
  }
  const int N_padded = (seq_len + 31) / 32 * 32;
  return N_padded >= head_dim;
}

int sync_attention_task_rows(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                           int mask_stride) {
  if (sync_attention_can_group_causal(qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, seq_len, mask_stride)) {
    int gqa_factor = n_heads / n_kv_heads;
    return gqa_factor * sync_attention_causal_group_q_rows(qo_len);
  }
  if (sync_attention_can_group_decode(qo_len, n_heads, n_kv_heads, head_dim, seq_len, mask_stride)) {
    return sync_attention_rows_per_task(qo_len, n_heads / n_kv_heads, 1);
  }
  if (mask_stride < 0 && qo_len > ATTN_PREFILL_SEGMENT_Q) {
    return ATTN_PREFILL_SEGMENT_Q;
  }
  return qo_len;
}

int sync_attention_total_tasks(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                             int mask_stride) {
  if (sync_attention_can_group_causal(qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, seq_len, mask_stride)) {
    return n_kv_heads;
  }
  if (sync_attention_can_group_decode(qo_len, n_heads, n_kv_heads, head_dim, seq_len, mask_stride)) {
    return n_kv_heads;
  }
  return n_heads;
}

int sync_attention_use_online_pages(int allow_online_pages, int head_dim, int page_size, int seq_len,
                                                  int mask_stride) {
  return allow_online_pages && (head_dim % 64) == 0 && (page_size % 32) == 0 &&
         (seq_len > ATTN_FIXED_WORKSPACE_KV || mask_stride > ATTN_FIXED_WORKSPACE_KV);
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
