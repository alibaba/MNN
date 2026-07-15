#pragma once

#include "dsp/mmap_mgr.h"
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
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dsp/hvx_math.h"
#include "dsp/hvx_convert.h"
#include "dsp/dma_utils.h"
#include "dsp/ops.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "transpose_hvx.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"

#if defined(__HEXAGON_ARCH__) && (__HEXAGON_ARCH__ >= 81)
#define MNN_ATTENTION_HMX_ENABLE_PER_MATMUL 1
#define MNN_ATTENTION_HMX_COMBINE_DECODE 1
#else
#define MNN_ATTENTION_HMX_ENABLE_PER_MATMUL 0
#define MNN_ATTENTION_HMX_COMBINE_DECODE 0
#endif

enum AttnHmxOutputLayoutType {
  ATTN_HMX_OUT_PACKED_FP16 = 0,
  ATTN_HMX_OUT_LINEAR_FP32_SCALED = 1,
};

#define ATTN_HMX_MAX_PAIR_PACKS 64
#define ATTN_HMX_KV_BLOCK 256
#define ATTN_HMX_KV_BLOCK_TILES (ATTN_HMX_KV_BLOCK / 32)
#define ATTN_HMX_MAX_KP 256
#define ATTN_HMX_MAX_WEIGHT_DESCS 8
#define ATTN_HMX_PUSH_K_HVX_SCRATCH_BYTES (128 * 128)
#define ATTN_FIXED_WORKSPACE_KV 2048
#define ATTN_PREFILL_SEGMENT_Q 64

enum AttnHmxWeightLayoutType {
  ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256 = 0,
  ATTN_HMX_WEIGHT_LAYOUT_V_BLOCK256 = 1,
};

static inline size_t attn_hmx_k_tile_index(int seq_tile, int kv_head, int k_tile, int n_kv_heads, int k_icP) {
  return ((((size_t)(seq_tile / ATTN_HMX_KV_BLOCK_TILES) * n_kv_heads + kv_head) * ATTN_HMX_KV_BLOCK_TILES +
           (seq_tile % ATTN_HMX_KV_BLOCK_TILES)) *
          k_icP +
          k_tile);
}

static inline size_t attn_hmx_v_tile_index(int dim_tile, int seq_tile, int kv_head, int n_kv_heads, int v_ocP) {
  return ((((size_t)(seq_tile / ATTN_HMX_KV_BLOCK_TILES) * n_kv_heads + kv_head) * v_ocP + dim_tile) *
          ATTN_HMX_KV_BLOCK_TILES +
          (seq_tile % ATTN_HMX_KV_BLOCK_TILES));
}

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  __fp16* pastK;
  __fp16* pastV;
  const __fp16* K;
  const __fp16* V;
  int past_kv_len;
  int insert_len;
  int n_kv_heads;
  int head_dim;
  int k_icP;
  int v_ocP;
  int seq_chunks;
  size_t kv_stride;
  int value_c4;
  int value_token_offset;
  int value_seq_len;
  HVX_Vector* k_hvx_scratch_base;
} PushKVTaskState;

typedef struct {
  volatile int done;
  AEEResult status;
  uint8_t** pastKPages;
  uint8_t** pastVPages;
  uint8_t* K;
  uint8_t* V;
  int32_t seq_current;
  int32_t seq_add;
  int32_t n_kv_heads;
  int32_t head_dim;
  int32_t page_count;
  int32_t page_size;
  int32_t value_c4;
  int32_t value_token_offset;
  int32_t value_seq_len;
} AsyncPushKVPagesState;

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  int total_heads;
  int gqa_factor;
  int qo_len;
  int qo_total_len;
  int q_offset;
  int n_kv_heads;
  int head_dim;
  int mask_stride;
  int seq_current;
  int N;
  int N_padded;
  int K_dim_padded;
  int k_icP;
  int v_ocP;
  size_t qo_stride;
  size_t worker_workspace_bytes;
  uint8_t* workspace_base;
  __fp16* O;
  const __fp16* Q;
  const float* mask;
  __fp16* pastK;
  __fp16* pastV;
  uint8_t** pastKPages;
  uint8_t** pastVPages;
  int page_count;
  int page_size;
  int online_block_pages;
  int online_block_size;
  AsyncPushKVPagesState* async_push;
  int async_push_seq_begin;
  float scale;
  int decode_grouped;
  int online_pages;
} SyncAttentionTaskState;

size_t sync_attention_head_workspace_bytes(int qo_len, int seq_len);
size_t sync_attention_page_head_workspace_bytes(int qo_len, int seq_len, int head_dim);
size_t sync_attention_page_block_workspace_bytes(int qo_len, int block_len, int head_dim);
int sync_attention_can_group_decode(int qo_len, int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                    int mask_stride);
int sync_attention_can_group_causal(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads,
                                    int head_dim, int seq_len, int mask_stride);
int sync_attention_causal_group_q_rows(int qo_len);
int sync_attention_task_rows(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads,
                             int head_dim, int seq_len, int mask_stride);
int sync_attention_total_tasks(int qo_len, int seq_current, int seq_add, int n_heads, int n_kv_heads,
                               int head_dim, int seq_len, int mask_stride);
int sync_attention_use_online_pages(int allow_online_pages, int head_dim, int page_size, int seq_len, int mask_stride);
int sync_attention_pick_task_count(int total_heads);
int push_kv_pick_task_count(int total_tasks);

void attn_hmx_matmul(uint8_t* c, const uint8_t* a, const uint8_t* b, int M, int K, int N, int max_K, int a_stride,
                     int outputLayoutType, float outputScale, int weightLayoutType, int kv_head, int n_kv_heads,
                     int output_stride, int output_row_offset);
void attn_hmx_matmul_pages_qk(const SyncAttentionTaskState* state, float* scores, const __fp16* q_ptr, int rows,
                              int q_stride, int q_row_offset, int kv_head, int valid_end);
void attn_hmx_matmul_pages_sv(const SyncAttentionTaskState* state, __fp16* dst, __fp16* temp_O,
                              const __fp16* linear_S, int rows, int output_stride, int row_offset, int kv_head,
                              int valid_end);
void attn_hmx_matmul_page_qk_block(const SyncAttentionTaskState* state, float* scores, const __fp16* q_ptr,
                                   int rows, int q_stride, int kv_head, int page_begin, int page_end,
                                   int block_start, int block_valid);
void attn_hmx_matmul_page_sv_block(const SyncAttentionTaskState* state, __fp16* dst, __fp16* page_temp_O,
                                   const __fp16* linear_S, int rows, int kv_head, int page_begin,
                                   int page_end, int block_start, int block_valid);

AEEResult htp_ops_push_kv(uint8_t* pPastK, uint8_t* pPastV, uint8_t* pK, uint8_t* pV, int32_t seq_current,
                          int32_t seq_add, int32_t n_kv_heads, int32_t head_dim, int32_t max_kv_len,
                          int32_t value_c4, int32_t value_token_offset, int32_t value_seq_len);
AEEResult htp_ops_push_kv_pages(uint8_t** pPastKPages, uint8_t** pPastVPages, uint8_t* pK, uint8_t* pV,
                                int32_t seq_current, int32_t seq_add, int32_t n_kv_heads, int32_t head_dim,
                                int32_t page_count, int32_t page_size, int32_t value_c4,
                                int32_t value_token_offset, int32_t value_seq_len);
void push_kv_pages_async_worker(void* data, int worker_index);
int flash_attn_try_single_token_output(uint8_t* pOut, const uint8_t* pV, int32_t qo_len, int32_t seq_current,
                                       int32_t seq_add, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
                                       int32_t value_c4);

int sync_attention(__fp16 *restrict O, const __fp16 *restrict Q, const float *restrict mask, uint8_t *workspace,
                   __fp16 *pastK, __fp16 *pastV, int qo_len, int seq_current, int seq_add, int n_heads,
                   int n_kv_heads, int head_dim, float scale, int mask_stride);
int sync_attention_pages(__fp16 *restrict O, const __fp16 *restrict Q, const float *restrict mask, uint8_t *workspace,
                         uint8_t **pastKPages, uint8_t **pastVPages, int qo_len, int seq_current, int seq_add,
                         int n_heads, int n_kv_heads, int head_dim, float scale, int mask_stride,
                         int page_count, int page_size, AsyncPushKVPagesState* async_push, int allow_online_pages);
void sync_attention_run_tasks(SyncAttentionTaskState* state, int n_tasks);

extern "C" AEEResult htp_ops_flash_attn(uint8_t* pOut, uint8_t* pQ, uint8_t* pK, uint8_t* pV, uint8_t* pMask,
                                        uint8_t* pWorkspace, uint8_t* pPastK, uint8_t* pPastV, int32_t qo_len,
                                        int32_t seq_current, int32_t seq_add, int32_t n_heads, int32_t n_kv_heads,
                                        int32_t head_dim, float scale, int32_t mask_stride, int32_t max_kv_len,
                                        int32_t value_c4);
extern "C" AEEResult htp_ops_flash_attn_pages(uint8_t* pOut, uint8_t* pQ, uint8_t* pK, uint8_t* pV, uint8_t* pMask,
                                              uint8_t* pWorkspace, uint8_t** pPastKPages, uint8_t** pPastVPages,
                                              int32_t qo_len, int32_t seq_current, int32_t seq_add, int32_t n_heads,
                                              int32_t n_kv_heads, int32_t head_dim, float scale, int32_t mask_stride,
                                              int32_t max_kv_len, int32_t page_count, int32_t page_size,
                                              int32_t value_c4);

static inline dma_desc_1d_t* attn_prepare_chained_dma_desc_2d(dma_desc_1d_t* current_descs, dma_desc_2d_t* desc,
                                                              const void* src, void* dst, uint32_t width,
                                                              uint32_t height, uint32_t src_stride,
                                                              uint32_t dst_stride);
static inline void attn_store_output_packed_fp16(uint8_t* c, __fp16* vtcm_output, int oy_start, int oy_end, int ox, int M,
                                                 int valid_xi, int output_stride, int output_row_offset);
static inline void attn_store_output_linear_fp32(uint8_t* c, __fp16* vtcm_output, int oy_start, int oy_end, int ox, int N,
                                                 int valid_xi, int output_stride, int output_row_offset);
static inline void run_locked_attn_hmx_matmul(uint8_t* c, const uint8_t* a, const uint8_t* b, int M, int K, int N,
                                              int max_K, int a_stride, int outputLayoutType, float outputScale,
                                              int weightLayoutType, int kv_head, int n_kv_heads);
static inline void run_locked_attn_hmx_matmul_ex(uint8_t* c, const uint8_t* a, const uint8_t* b, int M, int K, int N,
                                                 int max_K, int a_stride, int outputLayoutType, float outputScale,
                                                 int weightLayoutType, int kv_head, int n_kv_heads, int output_stride,
                                                 int output_row_offset);
static inline void run_locked_attn_hmx_matmul_pages_qk(const SyncAttentionTaskState* state, float* scores,
                                                       const __fp16* q_ptr, int rows, int q_stride, int q_row_offset,
                                                       int kv_head, int valid_end);
static inline void run_locked_attn_hmx_matmul_pages_sv(const SyncAttentionTaskState* state, __fp16* dst, __fp16* temp_O,
                                                       const __fp16* linear_S, int rows, int output_stride,
                                                       int row_offset, int kv_head, int valid_end);
static inline void sync_attention_wait_async_push(const SyncAttentionTaskState* state);
static inline void sync_attention_add_mask(float* row_scores, const float* mask_ptr, int N, int mask_start_pos);
static inline float sync_attention_max_f32(const float* row_scores, int N);
static inline float sync_attention_exp_and_sum(float* row_scores, int N, float max_value);
static inline void sync_attention_normalize_to_fp16(__fp16* row_s, float* row_scores, int N, float inv_sum);
static inline void sync_attention_zero_fp16(__fp16* row_s, int N);
static inline void sync_attention_zero_packed_output(__fp16* dst, int rows, int output_stride, int row_offset,
                                                     int head_dim);
static inline void sync_attention_accumulate_packed_output(__fp16* dst, const __fp16* src, int rows, int output_stride,
                                                           int row_offset, int head_dim);
static inline void sync_attention_add_packed_rows(__fp16* dst, const __fp16* src, int rows, int head_dim);
static inline void sync_attention_page_offsets(const SyncAttentionTaskState* state, uint8_t* worker_workspace,
                                               float** scores, __fp16** linear_S, __fp16** temp_O, int rows);
static inline void sync_attention_page_block_offsets(const SyncAttentionTaskState* state, uint8_t* worker_workspace,
                                                     float** scores, __fp16** linear_S, __fp16** accum_O,
                                                     __fp16** temp_O, float** row_max, float** row_sum,
                                                     float** row_scale, int rows);
static inline uint8_t* sync_attention_worker_workspace(const SyncAttentionTaskState* state, int worker_index);
static inline void sync_attention_workspace_offsets(const SyncAttentionTaskState* state, uint8_t* worker_workspace,
                                                    int rows, float** scores, __fp16** linear_S, __fp16** temp_O);
static inline void sync_attention_reset_online_rows(float* row_max, float* row_sum, float* row_scale, int rows);
static inline void sync_attention_run_page_qk(const SyncAttentionTaskState* state, float* scores, const __fp16* q_ptr,
                                              int rows, int q_stride, int q_row_offset, int kv_head, int valid_end);
static inline void sync_attention_run_page_sv(const SyncAttentionTaskState* state, __fp16* dst, __fp16* temp_O,
                                              const __fp16* linear_S, int rows, int output_stride, int row_offset,
                                              int kv_head, int valid_end);
static inline void sync_attention_run_page_qk_block(const SyncAttentionTaskState* state, float* scores,
                                                    const __fp16* q_ptr, int rows, int q_stride, int kv_head,
                                                    int page_begin, int page_end, int block_start, int block_valid);
static inline void sync_attention_run_page_sv_block(const SyncAttentionTaskState* state, __fp16* dst,
                                                    __fp16* page_temp_O, const __fp16* linear_S, int rows,
                                                    int kv_head, int page_begin, int page_end, int block_start,
                                                    int block_valid);
static inline int sync_attention_try_page_causal_len2(const SyncAttentionTaskState* state, int head_id,
                                                      int worker_index);

#include "attention_common.hpp"
