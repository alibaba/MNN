#include "attention_private.hpp"

#include "dsp/hmx_queue.h"

#define MNN_ATTENTION_HMX_SHORT_JOB_ROWS 8
#define MNN_ATTENTION_HMX_SHORT_JOB_SPIN_COUNT 2000

static inline void attention_hmx_queue_execute(hmx_queue_callback_t callback, void* data, int rows) {
  if (rows <= MNN_ATTENTION_HMX_SHORT_JOB_ROWS) {
    hmx_queue_execute_with_spin(callback, data, MNN_ATTENTION_HMX_SHORT_JOB_SPIN_COUNT);
    return;
  }
  hmx_queue_execute(callback, data);
}

typedef struct {
  uint8_t* c;
  const uint8_t* a;
  const uint8_t* b;
  int M;
  int K;
  int N;
  int max_K;
  int a_stride;
  int output_layout_type;
  float output_scale;
  int weight_layout_type;
  int kv_head;
  int n_kv_heads;
  int output_stride;
  int output_row_offset;
} AttentionHmxMatmulJob;

static void attention_hmx_matmul_job(void* opaque) {
  AttentionHmxMatmulJob* job = (AttentionHmxMatmulJob*)opaque;
  attn_hmx_matmul(job->c, job->a, job->b, job->M, job->K, job->N, job->max_K, job->a_stride,
                  job->output_layout_type, job->output_scale, job->weight_layout_type, job->kv_head,
                  job->n_kv_heads, job->output_stride, job->output_row_offset);
}

void queued_attn_hmx_matmul(uint8_t* c, const uint8_t* a, const uint8_t* b, int M, int K, int N, int max_K,
                            int a_stride, int output_layout_type, float output_scale, int weight_layout_type,
                            int kv_head, int n_kv_heads, int output_stride, int output_row_offset) {
  AttentionHmxMatmulJob job = {c, a, b, M, K, N, max_K, a_stride, output_layout_type, output_scale,
                              weight_layout_type, kv_head, n_kv_heads, output_stride, output_row_offset};
  attention_hmx_queue_execute(attention_hmx_matmul_job, &job, M);
}

typedef struct {
  const SyncAttentionTaskState* state;
  float* scores;
  const __fp16* q_ptr;
  int rows;
  int q_stride;
  int q_row_offset;
  int kv_head;
  int valid_end;
} AttentionHmxPagesQKJob;

static void attention_hmx_pages_qk_job(void* opaque) {
  AttentionHmxPagesQKJob* job = (AttentionHmxPagesQKJob*)opaque;
  attn_hmx_matmul_pages_qk(job->state, job->scores, job->q_ptr, job->rows, job->q_stride, job->q_row_offset,
                           job->kv_head, job->valid_end);
}

void queued_attn_hmx_matmul_pages_qk(const SyncAttentionTaskState* state, float* scores, const __fp16* q_ptr,
                                     int rows, int q_stride, int q_row_offset, int kv_head, int valid_end) {
  AttentionHmxPagesQKJob job = {state, scores, q_ptr, rows, q_stride, q_row_offset, kv_head, valid_end};
  attention_hmx_queue_execute(attention_hmx_pages_qk_job, &job, rows);
}

typedef struct {
  const SyncAttentionTaskState* state;
  __fp16* dst;
  __fp16* temp_O;
  const __fp16* linear_S;
  int rows;
  int output_stride;
  int row_offset;
  int kv_head;
  int valid_end;
} AttentionHmxPagesSVJob;

static void attention_hmx_pages_sv_job(void* opaque) {
  AttentionHmxPagesSVJob* job = (AttentionHmxPagesSVJob*)opaque;
  attn_hmx_matmul_pages_sv(job->state, job->dst, job->temp_O, job->linear_S, job->rows, job->output_stride,
                           job->row_offset, job->kv_head, job->valid_end);
}

void queued_attn_hmx_matmul_pages_sv(const SyncAttentionTaskState* state, __fp16* dst, __fp16* temp_O,
                                     const __fp16* linear_S, int rows, int output_stride, int row_offset,
                                     int kv_head, int valid_end) {
  AttentionHmxPagesSVJob job = {state, dst, temp_O, linear_S, rows, output_stride, row_offset, kv_head, valid_end};
  attention_hmx_queue_execute(attention_hmx_pages_sv_job, &job, rows);
}

typedef struct {
  const SyncAttentionTaskState* state;
  float* scores;
  const __fp16* q_ptr;
  int rows;
  int q_stride;
  int kv_head;
  int page_begin;
  int page_end;
  int block_start;
  int block_valid;
} AttentionHmxPageQKBlockJob;

static void attention_hmx_page_qk_block_job(void* opaque) {
  AttentionHmxPageQKBlockJob* job = (AttentionHmxPageQKBlockJob*)opaque;
  attn_hmx_matmul_page_qk_block(job->state, job->scores, job->q_ptr, job->rows, job->q_stride, job->kv_head,
                                job->page_begin, job->page_end, job->block_start, job->block_valid);
}

void queued_attn_hmx_matmul_page_qk_block(const SyncAttentionTaskState* state, float* scores, const __fp16* q_ptr,
                                          int rows, int q_stride, int kv_head, int page_begin, int page_end,
                                          int block_start, int block_valid) {
  AttentionHmxPageQKBlockJob job = {state, scores, q_ptr, rows, q_stride, kv_head, page_begin, page_end,
                                    block_start, block_valid};
  attention_hmx_queue_execute(attention_hmx_page_qk_block_job, &job, rows);
}

typedef struct {
  const SyncAttentionTaskState* state;
  __fp16* dst;
  __fp16* page_temp_O;
  const __fp16* linear_S;
  int rows;
  int kv_head;
  int page_begin;
  int page_end;
  int block_start;
  int block_valid;
} AttentionHmxPageSVBlockJob;

static void attention_hmx_page_sv_block_job(void* opaque) {
  AttentionHmxPageSVBlockJob* job = (AttentionHmxPageSVBlockJob*)opaque;
  attn_hmx_matmul_page_sv_block(job->state, job->dst, job->page_temp_O, job->linear_S, job->rows, job->kv_head,
                                job->page_begin, job->page_end, job->block_start, job->block_valid);
}

void queued_attn_hmx_matmul_page_sv_block(const SyncAttentionTaskState* state, __fp16* dst, __fp16* page_temp_O,
                                          const __fp16* linear_S, int rows, int kv_head, int page_begin,
                                          int page_end, int block_start, int block_valid) {
  AttentionHmxPageSVBlockJob job = {state, dst, page_temp_O, linear_S, rows, kv_head, page_begin, page_end,
                                    block_start, block_valid};
  attention_hmx_queue_execute(attention_hmx_page_sv_block_job, &job, rows);
}
